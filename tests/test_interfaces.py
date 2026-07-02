"""
接口测试 —— 验证数据在各层接口之间传递的完整性与一致性。

覆盖：
- State 字段在节点间传递不丢失、不畸变
- Pydantic 模型序列化往返
- Checkpoint 持久化 fidelity
- 条件路由正确性
- Prompt 模板注入
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import MemorySaver

from core.agents.opponent import opponent_compute_node, opponent_interact_node
from core.agents.presenter import presenter_compute_node, presenter_interact_node
from core.agents.referee import referee_deliberate_node
from core.prompts.templates import (
    FINAL_SUMMARY_PROMPT,
    OPPONENT_SYSTEM_PROMPT,
    PRESENTER_SYSTEM_PROMPT,
    REFEREE_SYSTEM_PROMPT,
    final_summary_prompt,
    opponent_prompt,
    presenter_prompt,
    referee_prompt,
)
from core.schemas import RefereeJudgment, RoundRecord
from core.state import AgentState
from core.graph import _route_after_referee, build_graph
from tests.helpers import make_mock_model, make_state

# =============================================================================
# Prompt 模板测试
# =============================================================================


class TestPromptStrings:
    """验证所有 Prompt 字符串存在且为 str。"""

    def test_opponent_prompt_is_plain_str(self):
        assert isinstance(OPPONENT_SYSTEM_PROMPT, str)
        result = opponent_prompt("测试论题")
        assert isinstance(result, str)
        assert "测试论题" in result

    def test_presenter_prompt_is_plain_str(self):
        assert isinstance(PRESENTER_SYSTEM_PROMPT, str)
        result = presenter_prompt("原始论题", "批判内容", "用户回应")
        assert isinstance(result, str)
        assert "原始论题" in result
        assert "批判内容" in result
        assert "用户回应" in result

    def test_referee_prompt_is_plain_str(self):
        assert isinstance(REFEREE_SYSTEM_PROMPT, str)
        result = referee_prompt("当前论题", "草稿", "确认版", 1)
        assert isinstance(result, str)
        assert "当前论题" in result
        assert "草稿" in result
        assert "确认版" in result
        assert "第 1 轮" in result

    def test_final_summary_prompt_is_plain_str(self):
        assert isinstance(FINAL_SUMMARY_PROMPT, str)
        result = final_summary_prompt("初始论题", "最终论题", "[]")
        assert isinstance(result, str)
        assert "初始论题" in result
        assert "最终论题" in result


# =============================================================================
# Agent Node 输出接口测试
# =============================================================================


class TestNodeOutputInterface:
    """验证节点返回的 dict key 与 AgentState 兼容。"""

    def test_opponent_compute_output_keys_match_state(self):
        state = make_state()
        model = make_mock_model("批判")
        result = opponent_compute_node(state, model=model)
        for key in result:
            assert key in AgentState.__annotations__, f"Unknown key: {key}"

    def test_opponent_interact_output_keys_match_state(self):
        state = make_state(
            _critique="c",
            status="awaiting_critique_response",
        )
        # Mock interrupt to avoid GraphInterrupt
        with patch("core.agents.opponent.interrupt") as mock_int:
            mock_int.return_value = "回应"
            result = opponent_interact_node(state)
        for key in result:
            assert key in AgentState.__annotations__, f"Unknown key: {key}"

    def test_presenter_compute_output_keys_match_state(self):
        state = make_state(_critique="c", _user_response="u")
        model = make_mock_model("草稿")
        result = presenter_compute_node(state, model=model)
        for key in result:
            assert key in AgentState.__annotations__, f"Unknown key: {key}"

    def test_presenter_interact_output_keys_match_state(self):
        state = make_state(
            _draft_thesis="d",
            status="awaiting_thesis_confirmation",
        )
        with patch("core.agents.presenter.interrupt") as mock_int:
            mock_int.return_value = "确认"
            result = presenter_interact_node(state)
        for key in result:
            assert key in AgentState.__annotations__, f"Unknown key: {key}"

    def test_referee_output_keys_match_state(self):
        state = make_state(
            _critique="c",
            _user_response="u",
            _draft_thesis="d",
            _confirmed_thesis="cf",
        )
        # Mock the full referee flow
        mock_model = MagicMock()
        structured_mock = MagicMock()
        structured_mock.invoke.return_value = RefereeJudgment(
            continue_debate=True,
            new_thesis="新论题",
            reasoning="理由",
        )
        mock_model.with_structured_output.return_value = structured_mock
        mock_model.invoke.return_value = MagicMock(content="总结")

        result = referee_deliberate_node(state, model=mock_model)
        for key in result:
            assert key in AgentState.__annotations__, f"Unknown key: {key}"

    def test_state_merge_is_additive(self):
        """多个节点的部分更新合并后 state 完整。"""
        # 模拟一轮完整流程的返回值合并
        state = make_state(messages=[], history=[])
        model = make_mock_model("批判")
        r1 = opponent_compute_node(state, model=model)  # 不修改 state
        assert state["messages"] == []  # 原始未变

        # 手动合并
        merged = {**state, **r1}
        assert merged["_critique"] == "批判"
        assert merged["status"] == "awaiting_critique_response"
        assert len(merged["messages"]) == 1

    def test_all_nodes_produce_same_message_structure(self):
        """所有节点产生的消息都包含 role/content/round。"""
        state = make_state(
            _critique="c",
            _user_response="u",
            _draft_thesis="d",
            _confirmed_thesis="cf",
        )
        model = make_mock_model("测试内容")

        # 模拟 referee 的 structured_output
        mock_ref_model = MagicMock()
        structured_mock = MagicMock()
        structured_mock.invoke.return_value = RefereeJudgment(
            continue_debate=True,
            new_thesis="新论题",
            reasoning="理由",
        )
        mock_ref_model.with_structured_output.return_value = structured_mock
        mock_ref_model.invoke.return_value = MagicMock(content="总结")

        nodes = [
            opponent_compute_node(state, model=model),
            presenter_compute_node(state, model=model),
            referee_deliberate_node(state, model=mock_ref_model),
        ]

        for node_result in nodes:
            msgs = node_result.get("messages", [])
            for msg in msgs:
                assert "role" in msg
                assert "content" in msg
                assert "round" in msg


# =============================================================================
# Pydantic 序列化往返测试
# =============================================================================


class TestSerializationFidelity:
    """Pydantic 模型 dict 序列化往返保真度。"""

    def test_referee_judgment_roundtrip(self):
        original = RefereeJudgment(
            continue_debate=False,
            new_thesis="最终论题：AI 应在高风险领域受监管。",
            reasoning="论题已清晰明确，边界条件完整。",
            improvement_hint="建议在实际政策制定中参考此论题。",
        )
        restored = RefereeJudgment(**original.model_dump())
        assert restored.continue_debate == original.continue_debate
        assert restored.new_thesis == original.new_thesis
        assert restored.reasoning == original.reasoning
        assert restored.improvement_hint == original.improvement_hint

    def test_round_record_roundtrip(self):
        original = RoundRecord(
            round_number=1,
            thesis_before="原始论题",
            critique="批判",
            user_response="回应",
            draft_thesis="草稿",
            confirmed_thesis="确认版",
            thesis_after="演化论题",
            continue_debate=True,
            referee_reasoning="理由",
        )
        restored = RoundRecord(**original.model_dump())
        assert restored.round_number == original.round_number
        assert restored.thesis_before == original.thesis_before
        assert restored.thesis_after == original.thesis_after

    def test_extra_fields_forbidden_in_judgment(self):
        """RefereeJudgment 拒绝未定义的字段。"""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RefereeJudgment(
                continue_debate=True,
                new_thesis="论题",
                reasoning="理由",
                unknown_field="不应存在",
            )


# =============================================================================
# Checkpoint 持久化测试
# =============================================================================


class TestCheckpointInterface:
    """验证 state 在 checkpoint 中完整保存和恢复。"""

    def test_state_survives_interrupt(self):
        checkpointer = MemorySaver()
        from langgraph.types import Command

        def _mock_oc(state):
            return {
                "_critique": "批判",
                "messages": state["messages"] + [{"role": "opponent", "content": "批判", "round": 1}],
                "status": "awaiting_critique_response",
            }

        def _mock_oi(state):
            from langgraph.types import interrupt

            resp = interrupt(state["_critique"])
            return {
                "_user_response": str(resp),
                "messages": state["messages"] + [{"role": "user", "content": str(resp), "round": 1}],
                "status": "presenter_computing",
            }

        def _mock_pc(state):
            return {
                "_draft_thesis": "草稿",
                "messages": state["messages"] + [{"role": "presenter", "content": "草稿", "round": 1}],
                "status": "awaiting_thesis_confirmation",
            }

        def _mock_pi(state):
            from langgraph.types import interrupt

            cf = interrupt(state["_draft_thesis"])
            return {
                "_confirmed_thesis": str(cf),
                "messages": state["messages"] + [{"role": "user", "content": str(cf), "round": 1}],
                "status": "referee_deliberating",
            }

        def _mock_rd(state):
            record = RoundRecord(
                round_number=1,
                thesis_before=state["current_thesis"],
                critique=state["_critique"],
                user_response=state["_user_response"],
                draft_thesis=state["_draft_thesis"],
                confirmed_thesis=state["_confirmed_thesis"],
                thesis_after="最终论题",
                continue_debate=False,
                referee_reasoning="完成",
            )
            return {
                "messages": state["messages"] + [{"role": "referee", "content": "结束", "round": 1}],
                "history": state["history"] + [record],
                "status": "done",
                "final_result": "报告",
            }

        graph = build_graph(
            _mock_oc,
            _mock_oi,
            _mock_pc,
            _mock_pi,
            _mock_rd,
            checkpointer=checkpointer,
        )

        thread_id = str(uuid4())
        config: dict = {"configurable": {"thread_id": thread_id}}
        graph.invoke(make_state(), config)

        # 读取 checkpoint 快照
        snapshot = graph.get_state(config)
        saved = snapshot.values
        assert saved["current_thesis"] == "人工智能应该被严格监管以确保安全性。"
        assert saved["round"] == 1
        assert saved["_critique"] == "批判"
        assert len(saved["messages"]) == 1

        # Resume
        graph.invoke(Command(resume="回应"), config)
        graph.invoke(Command(resume="确认"), config)
        final_snapshot = graph.get_state(config)
        final = final_snapshot.values
        assert final["status"] == "done"
        assert len(final["history"]) == 1
        assert final["final_result"] == "报告"

    def test_current_thesis_persists_across_interrupts(self):
        """跨多个中断点 current_thesis 保持正确。"""
        checkpointer = MemorySaver()
        from langgraph.types import Command

        def _oc(state):
            return {
                "_critique": "批判",
                "messages": state["messages"] + [{"role": "opponent", "content": "批判", "round": state["round"]}],
                "status": "awaiting_critique_response",
            }

        def _oi(state):
            from langgraph.types import interrupt

            r = interrupt(state["_critique"])
            return {
                "_user_response": str(r),
                "messages": state["messages"] + [{"role": "user", "content": str(r), "round": state["round"]}],
                "status": "presenter_computing",
            }

        def _pc(state):
            return {
                "_draft_thesis": "草稿",
                "messages": state["messages"] + [{"role": "presenter", "content": "草稿", "round": state["round"]}],
                "status": "awaiting_thesis_confirmation",
            }

        def _pi(state):
            from langgraph.types import interrupt

            cf = interrupt(state["_draft_thesis"])
            return {
                "_confirmed_thesis": str(cf),
                "messages": state["messages"] + [{"role": "user", "content": str(cf), "round": state["round"]}],
                "status": "referee_deliberating",
            }

        def _rd(state):
            record = RoundRecord(
                round_number=state["round"],
                thesis_before=state["current_thesis"],
                critique=state["_critique"],
                user_response=state["_user_response"],
                draft_thesis=state["_draft_thesis"],
                confirmed_thesis=state["_confirmed_thesis"],
                thesis_after="演化后-V1",
                continue_debate=True,
                referee_reasoning="继续",
            )
            return {
                "current_thesis": "演化后-V1",
                "messages": state["messages"] + [{"role": "referee", "content": "继续", "round": state["round"]}],
                "history": state["history"] + [record],
                "status": "opponent_computing",
            }

        graph = build_graph(
            _oc,
            _oi,
            _pc,
            _pi,
            _rd,
            checkpointer=checkpointer,
        )

        config: dict = {"configurable": {"thread_id": str(uuid4())}}
        state = make_state(current_thesis="持久性测试论题")
        graph.invoke(state, config)

        snapshot = graph.get_state(config)
        assert snapshot.values["current_thesis"] == "持久性测试论题"

        graph.invoke(Command(resume="R1回应"), config)
        graph.invoke(Command(resume="R1确认"), config)

        final = graph.get_state(config)
        assert final.values["current_thesis"] == "演化后-V1"
        assert final.values["round"] == 2


# =============================================================================
# 路由测试
# =============================================================================


class TestRoutingInterface:
    """条件路由正确性。"""

    def test_route_done_to_end(self):
        from langgraph.graph import END

        state = make_state(status="done")
        assert _route_after_referee(state) == END

    def test_route_other_to_next_round(self):
        non_done_statuses = [
            "opponent_computing",
            "awaiting_critique_response",
            "presenter_computing",
            "awaiting_thesis_confirmation",
            "referee_deliberating",
            "idle",
        ]
        for s in non_done_statuses:
            state = make_state(status=s)  # type: ignore[arg-type]
            assert _route_after_referee(state) == "next_round", f"Failed for {s}"

    def test_route_never_returns_ambiguous(self):
        all_statuses = [
            "idle",
            "opponent_computing",
            "awaiting_critique_response",
            "presenter_computing",
            "awaiting_thesis_confirmation",
            "referee_deliberating",
            "done",
        ]
        from langgraph.graph import END

        for s in all_statuses:
            state = make_state(status=s)  # type: ignore[arg-type]
            result = _route_after_referee(state)
            assert result in (END, "next_round"), f"{s} → {result}"


# =============================================================================
# Prompt 模板扩展测试
# =============================================================================


class TestPromptTemplatesExtended:
    """Prompt 模板函数的边界条件。"""

    def test_referee_prompt_with_history_summary(self):
        """referee_prompt 包含 history_summary 时格式正确。"""
        result = referee_prompt(
            current_thesis="论题",
            draft_thesis="草稿",
            confirmed_thesis="确认",
            round_num=3,
            history_summary="第1轮: A→B\n第2轮: B→C",
        )
        assert "第 3 轮" in result
        assert "前轮摘要" in result
        assert "第1轮: A→B" in result
        assert "高度重复" in result  # 注示文本

    def test_referee_prompt_without_history_summary(self):
        """referee_prompt 不含 history_summary 时不应有前轮摘要。"""
        result = referee_prompt(
            current_thesis="论题",
            draft_thesis="草稿",
            confirmed_thesis="确认",
            round_num=1,
        )
        assert "前轮摘要" not in result
        assert "第 1 轮" in result

    def test_opponent_prompt_includes_thesis(self):
        """opponent_prompt 包含完整论题文本。"""
        result = opponent_prompt("AI 应受严格监管。")
        assert "AI 应受严格监管。" in result
        assert len(result) > 20

    def test_presenter_prompt_includes_all_context(self):
        """presenter_prompt 包含原始论题、批判和用户回应。"""
        result = presenter_prompt("论题A", "批判B", "回应C")
        assert "论题A" in result
        assert "批判B" in result
        assert "回应C" in result


# =============================================================================
# State 边界测试
# =============================================================================


class TestStateEdgeCases:
    """AgentState 和路由的边界值测试。"""

    def test_missing_status_key_raises(self):
        """state 缺少 status 键时 _route_after_referee 抛出 KeyError。"""
        state: dict = {"current_thesis": "x", "round": 1, "messages": [], "history": []}
        with pytest.raises(KeyError):
            _route_after_referee(state)  # type: ignore[arg-type]

    def test_invalid_status_defaults_to_next_round(self):
        """未知 status 值默认路由到 'next_round'。"""
        state = make_state(status="totally_invalid_status")  # type: ignore[arg-type]
        assert _route_after_referee(state) == "next_round"


# =============================================================================
# Pydantic 扩展边界测试
# =============================================================================


class TestSerializationEdgeCases:
    """Pydantic 模型的边界值与扩展行为。"""

    def test_referee_judgment_min_length_fields(self):
        """RefereeJudgment 单字符 new_thesis/reasoning 通过验证。"""
        j = RefereeJudgment(
            continue_debate=True,
            new_thesis="x",
            reasoning="y",
        )
        assert j.new_thesis == "x"
        assert j.reasoning == "y"

    def test_round_record_constructed_from_dict(self):
        """RoundRecord 可从普通字典构造（模拟 checkpoint 恢复）。"""
        d = {
            "round_number": 2,
            "thesis_before": "前",
            "critique": "批判",
            "user_response": "回应",
            "draft_thesis": "草稿",
            "confirmed_thesis": "确认",
            "thesis_after": "后",
            "continue_debate": True,
            "referee_reasoning": "理由",
        }
        record = RoundRecord(**d)
        assert record.round_number == 2
        assert record.thesis_before == "前"
        assert record.thesis_after == "后"
        # model_dump 往返
        restored = RoundRecord(**record.model_dump())
        assert restored.thesis_before == "前"

    def test_round_record_model_dump_roundtrip(self):
        """RoundRecord model_dump → 重建验证（完整往返）。"""
        record = RoundRecord(
            round_number=1,
            thesis_before="A",
            critique="B",
            user_response="C",
            draft_thesis="D",
            confirmed_thesis="E",
            thesis_after="F",
            continue_debate=False,
            referee_reasoning="完成",
        )
        restored = RoundRecord(**record.model_dump())
        assert restored.round_number == record.round_number
        assert restored.thesis_before == "A"
        assert restored.thesis_after == "F"
