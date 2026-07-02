"""Agent 节点单元测试 —— 所有 LLM 调用使用 Mock。

测试 6 个节点函数：
- opponent_compute_node: LLM 批判生成
- opponent_interact_node: interrupt() 交互
- presenter_compute_node: LLM 精确化
- presenter_interact_node: interrupt() 交互
- referee_deliberate_node: 结构化判定（继续/结束）
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_openai import ChatOpenAI

from core.agents.opponent import opponent_compute_node, opponent_interact_node
from core.agents.presenter import presenter_compute_node, presenter_interact_node
from core.agents.referee import referee_deliberate_node
from core.schemas import RefereeJudgment, RoundRecord
from tests.helpers import make_mock_model, make_state


def _make_mock_referee_model(judgment: RefereeJudgment) -> MagicMock:
    """构造 Mock LLM，.with_structured_output().invoke() 返回给定 judgment。"""
    mock = MagicMock(spec=ChatOpenAI)
    structured_mock = MagicMock()
    structured_mock.invoke.return_value = judgment
    mock.with_structured_output.return_value = structured_mock

    # 同时 mock 普通 invoke（用于 final_result 摘要生成）
    summary_response = MagicMock()
    summary_response.content = "最终总结报告。"
    mock.invoke.return_value = summary_response
    return mock


# =============================================================================
# Opponent Compute 测试
# =============================================================================


class TestOpponentComputeNode:
    """opponent_compute_node 测试。"""

    def test_returns_correct_keys(self):
        state = make_state()
        model = make_mock_model("论题过于宽泛，缺乏具体边界条件。")
        result = opponent_compute_node(state, model=model)

        assert "_critique" in result
        assert "messages" in result
        assert "status" in result

    def test_critique_is_non_empty_str(self):
        state = make_state()
        critique_text = "该论题存在三个主要漏洞：1."
        model = make_mock_model(critique_text)
        result = opponent_compute_node(state, model=model)

        assert isinstance(result["_critique"], str)
        assert len(result["_critique"]) > 0
        assert result["_critique"] == critique_text

    def test_status_transitions_to_awaiting_response(self):
        state = make_state(status="opponent_computing")
        model = make_mock_model("批判内容")
        result = opponent_compute_node(state, model=model)

        assert result["status"] == "awaiting_critique_response"

    def test_appends_message_with_correct_role(self):
        state = make_state(messages=[], round=2)
        model = make_mock_model("批判")
        result = opponent_compute_node(state, model=model)

        msgs = result["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "opponent"
        assert msgs[0]["content"] == "批判"
        assert msgs[0]["round"] == 2

    def test_does_not_mutate_original_state(self):
        original = make_state(current_thesis="原始论题", messages=[], status="opponent_computing")
        model = make_mock_model("批判")
        opponent_compute_node(original, model=model)

        assert original["current_thesis"] == "原始论题"
        assert original["messages"] == []
        assert original["status"] == "opponent_computing"

    def test_passes_current_thesis_to_llm(self):
        state = make_state(current_thesis="测试论题：AI 必须被监管")
        model = make_mock_model("批判")

        opponent_compute_node(state, model=model)

        call_args = model.invoke.call_args[0][0]
        human_msg = call_args[1]
        assert "测试论题：AI 必须被监管" in human_msg.content


# =============================================================================
# Opponent Interact 测试
# =============================================================================


class TestOpponentInteractNode:
    """opponent_interact_node 测试。"""

    def test_interrupt_called_with_critique(self):
        state = make_state(
            _critique="你的论题存在逻辑漏洞…",
            status="awaiting_critique_response",
        )

        with patch("core.agents.opponent.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "我承认论题需要更精确的边界条件。"
            result = opponent_interact_node(state)

        mock_interrupt.assert_called_once_with("你的论题存在逻辑漏洞…")
        assert result["_user_response"] == "我承认论题需要更精确的边界条件。"

    def test_status_transitions_to_presenter_computing(self):
        state = make_state(
            _critique="批判内容",
            status="awaiting_critique_response",
        )

        with patch("core.agents.opponent.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "用户回应"
            result = opponent_interact_node(state)

        assert result["status"] == "presenter_computing"

    def test_appends_user_message(self):
        state = make_state(
            messages=[{"role": "opponent", "content": "批判", "round": 1}],
            _critique="批判",
            round=1,
            status="awaiting_critique_response",
        )

        with patch("core.agents.opponent.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "我的回应"
            result = opponent_interact_node(state)

        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "我的回应"
        assert msgs[-1]["round"] == 1


# =============================================================================
# Presenter Compute 测试
# =============================================================================


class TestPresenterComputeNode:
    """presenter_compute_node 测试。"""

    def test_returns_correct_keys(self):
        state = make_state(
            _critique="批判",
            _user_response="用户回应",
        )
        model = make_mock_model("精确化后的论题：AI 应在特定领域受到监管。")
        result = presenter_compute_node(state, model=model)

        assert "_draft_thesis" in result
        assert "messages" in result
        assert "status" in result

    def test_draft_is_non_empty_str(self):
        state = make_state(
            _critique="批判",
            _user_response="用户回应",
        )
        draft_text = "精确化：AI 监管应区分高风险与低风险应用场景。"
        model = make_mock_model(draft_text)
        result = presenter_compute_node(state, model=model)

        assert isinstance(result["_draft_thesis"], str)
        assert len(result["_draft_thesis"]) > 0
        assert result["_draft_thesis"] == draft_text

    def test_status_transitions_to_awaiting_confirmation(self):
        state = make_state(_critique="c", _user_response="u")
        model = make_mock_model("精确化论题")
        result = presenter_compute_node(state, model=model)

        assert result["status"] == "awaiting_thesis_confirmation"

    def test_appends_message_with_correct_role(self):
        state = make_state(
            messages=[],
            _critique="批判",
            _user_response="用户回应",
            round=1,
        )
        model = make_mock_model("精确化论题")
        result = presenter_compute_node(state, model=model)

        msgs = result["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "presenter"
        assert msgs[0]["content"] == "精确化论题"
        assert msgs[0]["round"] == 1

    def test_does_not_mutate_original_state(self):
        original = make_state(_critique="批判", _user_response="用户回应", messages=[])
        model = make_mock_model("精确化论题")
        presenter_compute_node(original, model=model)

        assert original["_draft_thesis"] == ""
        assert original["messages"] == []

    def test_passes_full_context_to_llm(self):
        state = make_state(
            current_thesis="原始论题",
            _critique="论题过于宽泛",
            _user_response="应限定在高风险 AI 领域",
        )
        model = make_mock_model("精确化论题")

        presenter_compute_node(state, model=model)

        call_args = model.invoke.call_args[0][0]
        human_msg = call_args[1]
        assert "原始论题" in human_msg.content
        assert "论题过于宽泛" in human_msg.content
        assert "应限定在高风险 AI 领域" in human_msg.content


# =============================================================================
# Presenter Interact 测试
# =============================================================================


class TestPresenterInteractNode:
    """presenter_interact_node 测试。"""

    def test_interrupt_called_with_draft(self):
        state = make_state(
            _draft_thesis="精确化论题：AI 应受监管。",
            status="awaiting_thesis_confirmation",
        )

        with patch("core.agents.presenter.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "精确化论题：AI 应受监管。"
            presenter_interact_node(state)

        mock_interrupt.assert_called_once_with("精确化论题：AI 应受监管。")

    def test_status_transitions_to_referee_deliberating(self):
        state = make_state(
            _draft_thesis="草稿",
            status="awaiting_thesis_confirmation",
        )

        with patch("core.agents.presenter.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "确认版论题"
            result = presenter_interact_node(state)

        assert result["status"] == "referee_deliberating"

    def test_user_can_edit_thesis(self):
        """用户可以编辑草稿后再确认。"""
        state = make_state(
            _draft_thesis="原始草稿",
            status="awaiting_thesis_confirmation",
        )

        with patch("core.agents.presenter.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "经过编辑的新论题"
            result = presenter_interact_node(state)

        assert result["_confirmed_thesis"] == "经过编辑的新论题"

    def test_appends_user_message(self):
        state = make_state(
            messages=[{"role": "presenter", "content": "草稿", "round": 1}],
            _draft_thesis="草稿",
            round=1,
            status="awaiting_thesis_confirmation",
        )

        with patch("core.agents.presenter.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "确认版"
            result = presenter_interact_node(state)

        msgs = result["messages"]
        assert len(msgs) == 2
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "确认版"


# =============================================================================
# Referee Deliberate 测试
# =============================================================================


class TestRefereeDeliberateNode:
    """referee_deliberate_node 测试。"""

    def _make_judgment(self, **overrides) -> RefereeJudgment:
        defaults = {
            "continue_debate": True,
            "new_thesis": "拼合后的新论题。",
            "reasoning": "论题仍有歧义，需要进一步精确化。",
            "improvement_hint": "建议明确监管范围。",
        }
        return RefereeJudgment(**{**defaults, **overrides})

    def test_returns_correct_keys(self):
        state = make_state(
            _critique="批判",
            _user_response="用户回应",
            _draft_thesis="草稿论题",
            _confirmed_thesis="确认论题",
        )
        judgment = self._make_judgment()
        model = _make_mock_referee_model(judgment)
        result = referee_deliberate_node(state, model=model)

        assert "messages" in result
        assert "history" in result
        assert "status" in result

    def test_status_continue_when_not_done(self):
        state = make_state(
            _critique="批判",
            _user_response="回应",
            _draft_thesis="草稿",
            _confirmed_thesis="确认",
        )
        judgment = self._make_judgment(continue_debate=True)
        model = _make_mock_referee_model(judgment)
        result = referee_deliberate_node(state, model=model)

        assert result["status"] == "opponent_computing"
        assert "current_thesis" in result
        assert result["current_thesis"] == "拼合后的新论题。"
        # 正常轮次不追加裁判消息
        assert len(result["messages"]) == len(state["messages"])

    def test_status_done_when_referee_ends(self):
        state = make_state(
            _critique="批判",
            _user_response="回应",
            _draft_thesis="草稿",
            _confirmed_thesis="确认",
        )
        judgment = self._make_judgment(continue_debate=False)
        model = _make_mock_referee_model(judgment)
        result = referee_deliberate_node(state, model=model)

        assert result["status"] == "done"
        assert "final_result" in result
        assert len(result["final_result"]) > 0
        # 终止时追加裁判消息
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "referee"

    def test_no_referee_message_when_continue(self):
        """正常轮次（continue_debate=True）：裁判不产生对用户可见的消息。"""
        state = make_state(
            messages=[],
            _critique="c",
            _user_response="u",
            _draft_thesis="d",
            _confirmed_thesis="cf",
        )
        judgment = self._make_judgment(continue_debate=True)
        model = _make_mock_referee_model(judgment)
        result = referee_deliberate_node(state, model=model)

        msgs = result["messages"]
        assert len(msgs) == 0  # 正常轮次不追加裁判消息

    def test_appends_referee_message_when_done(self):
        """辩论终止（continue_debate=False）：裁判输出最终总结作为消息。"""
        state = make_state(
            messages=[],
            _critique="c",
            _user_response="u",
            _draft_thesis="d",
            _confirmed_thesis="cf",
        )
        judgment = self._make_judgment(continue_debate=False)
        model = _make_mock_referee_model(judgment)
        result = referee_deliberate_node(state, model=model)

        msgs = result["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "referee"
        assert len(msgs[0]["content"]) > 0  # 最终总结作为消息内容

    def test_appends_round_record_to_history(self):
        state = make_state(
            history=[],
            _critique="c",
            _user_response="u",
            _draft_thesis="d",
            _confirmed_thesis="cf",
            current_thesis="进入轮次的论题",
        )
        judgment = self._make_judgment(new_thesis="拼合后论题")
        model = _make_mock_referee_model(judgment)
        result = referee_deliberate_node(state, model=model)

        history = result["history"]
        assert len(history) == 1
        record = history[0]
        assert isinstance(record, RoundRecord)
        assert record.round_number == 1
        assert record.thesis_before == "进入轮次的论题"
        assert record.critique == "c"
        assert record.user_response == "u"
        assert record.draft_thesis == "d"
        assert record.confirmed_thesis == "cf"
        assert record.thesis_after == "拼合后论题"
        assert record.continue_debate is True
        assert len(record.referee_reasoning) > 0

    def test_preserves_existing_history(self):
        existing = RoundRecord(
            round_number=1,
            thesis_before="旧论题",
            critique="旧批判",
            user_response="旧回应",
            draft_thesis="旧草稿",
            confirmed_thesis="旧确认",
            thesis_after="旧拼合",
            continue_debate=True,
            referee_reasoning="旧理由",
        )
        state = make_state(
            history=[existing],
            round=2,
            _critique="c2",
            _user_response="u2",
            _draft_thesis="d2",
            _confirmed_thesis="cf2",
            current_thesis="旧拼合",
        )
        judgment = self._make_judgment(new_thesis="新拼合")
        model = _make_mock_referee_model(judgment)
        result = referee_deliberate_node(state, model=model)

        history = result["history"]
        assert len(history) == 2
        assert history[0].round_number == 1
        assert history[1].round_number == 2

    def test_does_not_mutate_original_state(self):
        original = make_state(
            history=[],
            _critique="c",
            _user_response="u",
            _draft_thesis="d",
            _confirmed_thesis="cf",
        )
        judgment = self._make_judgment()
        model = _make_mock_referee_model(judgment)
        referee_deliberate_node(original, model=model)

        assert original["history"] == []
        assert original["current_thesis"] == "人工智能应该被严格监管以确保安全性。"

    def test_judgment_is_referee_judgment_instance(self):
        state = make_state(
            _critique="c",
            _user_response="u",
            _draft_thesis="d",
            _confirmed_thesis="cf",
        )
        judgment = self._make_judgment()
        model = _make_mock_referee_model(judgment)
        result = referee_deliberate_node(state, model=model)

        record = result["history"][0]
        assert record.thesis_after == "拼合后的新论题。"
        assert record.referee_reasoning == "论题仍有歧义，需要进一步精确化。"


# =============================================================================
# Interrupt 幂等性测试
# =============================================================================


class TestInterruptIdempotency:
    """验证 interact 节点 resume 时的幂等行为。"""

    def test_opponent_interact_resume_does_not_duplicate_messages(self):
        state = make_state(
            messages=[{"role": "opponent", "content": "批判", "round": 1}],
            _critique="批判",
            round=1,
            status="awaiting_critique_response",
        )

        with patch("core.agents.opponent.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "用户回应"
            result = opponent_interact_node(state)

        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "opponent"
        assert result["messages"][1]["role"] == "user"

    def test_presenter_interact_resume_does_not_duplicate_messages(self):
        state = make_state(
            messages=[{"role": "presenter", "content": "草稿", "round": 1}],
            _draft_thesis="草稿",
            round=1,
            status="awaiting_thesis_confirmation",
        )

        with patch("core.agents.presenter.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "确认版"
            result = presenter_interact_node(state)

        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "presenter"
        assert result["messages"][1]["role"] == "user"


# =============================================================================
# Opponent 边界/错误路径测试
# =============================================================================


class TestOpponentEdgeCases:
    """opponent_compute_node 边界值与错误路径。"""

    def test_model_none_uses_default(self):
        """model=None 时应调用 get_chat_model() 获取默认 LLM。"""
        state = make_state()
        with patch("core.agents._base.get_chat_model") as mock_get:
            mock_model = make_mock_model("批判")
            mock_get.return_value = mock_model
            result = opponent_compute_node(state, model=None)
        assert "_critique" in result
        mock_get.assert_called_once_with(temperature=0.7)

    def test_empty_llm_response_handled(self):
        """LLM 返回空字符串时不崩溃。"""
        state = make_state()
        model = make_mock_model("")
        result = opponent_compute_node(state, model=model)
        assert isinstance(result["_critique"], str)
        assert result["_critique"] == ""

    def test_whitespace_only_llm_response(self):
        """LLM 返回仅空格时，strip 后为空字符串。"""
        state = make_state()
        model = make_mock_model("   \n  ")
        result = opponent_compute_node(state, model=model)
        assert result["_critique"] == ""

    def test_non_string_llm_content(self):
        """LLM 返回非字符串 content（如 list）时 str() 降级。"""
        state = make_state()
        model = MagicMock(spec=ChatOpenAI)
        response = MagicMock()
        response.content = ["意外", "的", "列表"]
        model.invoke.return_value = response
        result = opponent_compute_node(state, model=model)
        assert isinstance(result["_critique"], str)
        assert len(result["_critique"]) > 0  # str() 降级成功，返回了非空字符串


class TestOpponentInteractEdgeCases:
    """opponent_interact_node 边界值测试。"""

    def test_missing_critique_raises_key_error(self):
        """state 缺少 _critique 时应抛出 KeyError。"""
        state = make_state()
        # 删除 _critique 键来模拟 state 损坏
        broken: dict = {k: v for k, v in state.items() if k != "_critique"}
        with pytest.raises(KeyError):
            opponent_interact_node(broken)  # type: ignore[arg-type]


# =============================================================================
# Presenter 边界/错误路径测试
# =============================================================================


class TestPresenterEdgeCases:
    """presenter_compute_node 边界值与错误路径。"""

    def test_model_none_uses_default(self):
        """model=None 时应调用 get_chat_model() 获取默认 LLM。"""
        state = make_state(_critique="c", _user_response="u")
        with patch("core.agents._base.get_chat_model") as mock_get:
            mock_model = make_mock_model("草稿")
            mock_get.return_value = mock_model
            result = presenter_compute_node(state, model=None)
        assert "_draft_thesis" in result
        mock_get.assert_called_once_with(temperature=0.7)

    def test_empty_llm_response_handled(self):
        """LLM 返回空字符串时不崩溃。"""
        state = make_state(_critique="c", _user_response="u")
        model = make_mock_model("")
        result = presenter_compute_node(state, model=model)
        assert isinstance(result["_draft_thesis"], str)
        assert result["_draft_thesis"] == ""

    def test_non_string_llm_content(self):
        """LLM 返回非字符串 content 时 str() 降级。"""
        state = make_state(_critique="c", _user_response="u")
        model = MagicMock(spec=ChatOpenAI)
        response = MagicMock()
        response.content = ["精", "确", "化"]
        model.invoke.return_value = response
        result = presenter_compute_node(state, model=model)
        assert isinstance(result["_draft_thesis"], str)
        assert len(result["_draft_thesis"]) > 0  # str() 降级成功，返回了非空字符串


# =============================================================================
# Referee 边界/错误路径测试
# =============================================================================


class TestRefereeEdgeCases:
    """referee_deliberate_node 边界值与错误路径。"""

    def _make_judgment(self, **overrides) -> RefereeJudgment:
        defaults = {
            "continue_debate": True,
            "new_thesis": "拼合后的新论题。",
            "reasoning": "需要进一步深化。",
            "improvement_hint": "建议明确边界。",
        }
        return RefereeJudgment(**{**defaults, **overrides})

    def test_model_none_uses_default(self):
        """model=None 时应调用 get_chat_model() 获取默认 LLM。"""
        state = make_state(
            _critique="c",
            _user_response="u",
            _draft_thesis="d",
            _confirmed_thesis="cf",
        )
        judgment = self._make_judgment()
        with patch("core.agents.referee.get_chat_model") as mock_get:
            mock_model = MagicMock()
            structured_mock = MagicMock()
            structured_mock.invoke.return_value = judgment
            mock_model.with_structured_output.return_value = structured_mock
            # 也 mock 普通 invoke（final_result 备用）
            summary_response = MagicMock()
            summary_response.content = "总结"
            mock_model.invoke.return_value = summary_response
            mock_get.return_value = mock_model
            result = referee_deliberate_node(state, model=None)
        assert "status" in result
        mock_get.assert_called_once_with(temperature=0.0)

    def test_dict_format_history_from_checkpoint(self):
        """checkpoint 恢复后 history 元素为 dict（非 Pydantic）时兼容。"""
        state = make_state(
            history=[
                {
                    "round_number": 1,
                    "thesis_before": "旧论题",
                    "critique": "旧批判",
                    "user_response": "旧回应",
                    "draft_thesis": "旧草稿",
                    "confirmed_thesis": "旧确认",
                    "thesis_after": "旧拼合",
                    "continue_debate": True,
                    "referee_reasoning": "理由",
                }
            ],
            round=2,
            _critique="c",
            _user_response="u",
            _draft_thesis="d",
            _confirmed_thesis="cf",
            current_thesis="旧拼合",
        )
        judgment = self._make_judgment(new_thesis="新拼合")
        model = _make_mock_referee_model(judgment)
        result = referee_deliberate_node(state, model=model)

        # 验证 dict 格式的 history 被正确追加
        assert len(result["history"]) == 2
        assert result["history"][1].round_number == 2

    def test_done_with_dict_history_generates_final_summary(self):
        """终局总结应兼容 checkpoint 恢复后的 dict 格式 history。"""
        state = make_state(
            history=[
                {
                    "round_number": 1,
                    "thesis_before": "初始论题",
                    "critique": "旧批判",
                    "user_response": "旧回应",
                    "draft_thesis": "旧草稿",
                    "confirmed_thesis": "旧确认",
                    "thesis_after": "旧拼合",
                    "continue_debate": True,
                    "referee_reasoning": "继续",
                }
            ],
            round=2,
            current_thesis="旧拼合",
            _critique="c2",
            _user_response="u2",
            _draft_thesis="d2",
            _confirmed_thesis="cf2",
        )
        judgment = self._make_judgment(
            continue_debate=False,
            new_thesis="最终论题",
            reasoning="可以结束。",
        )
        model = _make_mock_referee_model(judgment)

        result = referee_deliberate_node(state, model=model)

        assert result["status"] == "done"
        assert result["final_result"] == "最终总结报告。"
        assert len(result["history"]) == 2

    def test_extract_json_from_plain_json(self):
        """_extract_json 支持直接解析 JSON 对象。"""
        from core.agents.referee import _extract_json

        result = _extract_json('{"continue_debate": true, "new_thesis": "论题"}')
        assert result == {"continue_debate": True, "new_thesis": "论题"}

    def test_extract_json_from_markdown_block(self):
        """_extract_json 支持 Markdown JSON 代码块。"""
        from core.agents.referee import _extract_json

        result = _extract_json('说明\n```json\n{"reasoning": "理由"}\n```')
        assert result == {"reasoning": "理由"}

    def test_extract_json_from_outer_braces(self):
        """_extract_json 支持从普通文本中提取最外层 JSON 对象。"""
        from core.agents.referee import _extract_json

        result = _extract_json('裁判输出如下：{"improvement_hint": "继续收窄边界"}。')
        assert result == {"improvement_hint": "继续收窄边界"}

    def test_extract_json_returns_none_for_invalid_content(self):
        """_extract_json 无法解析时返回 None。"""
        from core.agents.referee import _extract_json

        assert _extract_json("这不是 JSON") is None

    def test_build_history_summary_accepts_round_record_and_dict(self):
        """_build_history_summary 兼容 RoundRecord 与 checkpoint dict。"""
        from core.agents.referee import _build_history_summary

        first = RoundRecord(
            round_number=1,
            thesis_before="初始论题",
            critique="批判",
            user_response="回应",
            draft_thesis="草稿",
            confirmed_thesis="确认",
            thesis_after="一轮后论题",
            continue_debate=True,
            referee_reasoning="继续",
        )
        second = {
            "round_number": 2,
            "thesis_before": "一轮后论题",
            "thesis_after": "二轮后论题",
            "continue_debate": False,
        }
        state = make_state(history=[first, second])

        result = _build_history_summary(state)

        assert "Round 1: 初始论题 -> 一轮后论题 (continue: True)" in result
        assert "Round 2: 一轮后论题 -> 二轮后论题 (continue: False)" in result

    def test_get_initial_thesis_from_dict_history(self):
        """_get_initial_thesis 在 history[0] 为 dict 时正确回退。"""
        from core.agents.referee import _get_initial_thesis

        state = make_state(
            history=[
                {
                    "round_number": 1,
                    "thesis_before": "初始论题（dict格式）",
                    "critique": "c",
                    "user_response": "u",
                    "draft_thesis": "d",
                    "confirmed_thesis": "cf",
                    "thesis_after": "后",
                    "continue_debate": True,
                    "referee_reasoning": "r",
                }
            ],
            current_thesis="当前论题",
        )
        result = _get_initial_thesis(state)
        assert result == "初始论题（dict格式）"

    def test_get_initial_thesis_empty_history(self):
        """_get_initial_thesis 在 history 为空时返回 current_thesis。"""
        from core.agents.referee import _get_initial_thesis

        state = make_state(history=[], current_thesis="唯一论题")
        result = _get_initial_thesis(state)
        assert result == "唯一论题"

    def test_large_round_number(self):
        """大轮次编号（9999）不导致崩溃。"""
        state = make_state(
            round=9999,
            _critique="c",
            _user_response="u",
            _draft_thesis="d",
            _confirmed_thesis="cf",
        )
        judgment = self._make_judgment()
        model = _make_mock_referee_model(judgment)
        result = referee_deliberate_node(state, model=model)

        assert result["status"] == "opponent_computing"
        record = result["history"][0]
        assert record.round_number == 9999