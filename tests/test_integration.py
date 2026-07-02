"""
端到端集成测试 —— 多轮论题演化全生命周期。

使用 Mock LLM 节点 + 真实 LangGraph 图 + 真实 interrupt()/Command(resume=...)。
验证中断暂停/恢复、多轮循环、状态累积、thesis 演化。
"""

from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from core.graph import build_graph
from tests.helpers import make_initial_state
from tests.mock_nodes import (
    make_mock_referee,
    mock_opponent_compute,
    mock_opponent_interact,
    mock_presenter_compute,
    mock_presenter_interact,
)

# =============================================================================


class TestSingleRoundLifecycle:
    """验证单轮（裁判判定结束）的完整中断-恢复生命周期。"""

    def test_full_single_round(self):
        """从 idle → done，经过两次 interrupt。"""
        checkpointer = MemorySaver()
        graph = build_graph(
            opponent_compute_node=mock_opponent_compute,
            opponent_interact_node=mock_opponent_interact,
            presenter_compute_node=mock_presenter_compute,
            presenter_interact_node=mock_presenter_interact,
            referee_deliberate_node=make_mock_referee(continue_debate=False),
            checkpointer=checkpointer,
        )

        thread_id = str(uuid4())
        config: dict = {"configurable": {"thread_id": thread_id}}
        initial_state = make_initial_state("AI 应受监管。")

        # ---- Step 1: invoke → 停在第一个 interrupt (critique) ----
        result = graph.invoke(initial_state, config)
        assert result["status"] == "awaiting_critique_response"
        assert result["round"] == 1
        assert len(result["messages"]) == 1  # 只有 opponent msg
        assert result["messages"][0]["role"] == "opponent"

        # ---- Step 2: resume with user response → 停在第二个 interrupt (confirmation) ----
        result = graph.invoke(Command(resume="我同意，但需限定范围。"), config)
        assert result["status"] == "awaiting_thesis_confirmation"
        assert len(result["messages"]) == 3  # opponent + user + presenter
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["opponent", "user", "presenter"]

        # ---- Step 3: resume with confirmation → 裁判结束，debate done ----
        result = graph.invoke(Command(resume="确认：AI应受监管，重点在高风险领域。"), config)
        assert result["status"] == "done"
        assert len(result["messages"]) == 5  # o+u+p+u+referee
        assert len(result["history"]) == 1
        assert result["final_result"] == "终局总结。"
        assert result["history"][0].thesis_before == "AI 应受监管。"

    def test_round_cache_cleared_after_single_round(self):
        """验证 done 后轮次缓存字段状态（单轮结束无 next_round）。"""
        checkpointer = MemorySaver()
        graph = build_graph(
            mock_opponent_compute,
            mock_opponent_interact,
            mock_presenter_compute,
            mock_presenter_interact,
            make_mock_referee(continue_debate=False),
            checkpointer=checkpointer,
        )

        config: dict = {"configurable": {"thread_id": str(uuid4())}}
        graph.invoke(make_initial_state(), config)
        graph.invoke(Command(resume="回应"), config)
        result = graph.invoke(Command(resume="确认"), config)

        # single round, no next_round node runs, so caches remain as-is
        assert result["status"] == "done"


# =============================================================================
# 多轮生命周期测试
# =============================================================================


class TestMultiRoundLifecycle:
    """验证多轮论题演化的完整生命周期。"""

    def test_two_round_debate(self):
        """两轮辩论：第一轮 continue，第二轮 done。"""
        checkpointer = MemorySaver()

        # 第一轮裁判：继续
        r1_referee = make_mock_referee(
            continue_debate=True,
            new_thesis="AI 应在高风险领域受监管（第1轮拼合）",
            reasoning="论题已有改进，但仍有细化空间。",
        )

        graph = build_graph(
            mock_opponent_compute,
            mock_opponent_interact,
            mock_presenter_compute,
            mock_presenter_interact,
            r1_referee,
            checkpointer=checkpointer,
        )

        config: dict = {"configurable": {"thread_id": str(uuid4())}}

        # Round 1
        graph.invoke(make_initial_state("AI 应受监管。"), config)
        graph.invoke(Command(resume="R1: 用户回应"), config)
        result = graph.invoke(Command(resume="R1: 确认论题"), config)

        # 第一轮结束，应继续
        assert result["status"] == "awaiting_critique_response"  # 已进入 R2 的 opponent_interact
        assert result["round"] == 2
        assert result["current_thesis"] == "AI 应在高风险领域受监管（第1轮拼合）"
        assert len(result["history"]) == 1
        assert result["history"][0].round_number == 1

        # 替换裁判为结束版并重建图（模拟 R2 裁判判定结束）
        graph2 = build_graph(
            mock_opponent_compute,
            mock_opponent_interact,
            mock_presenter_compute,
            mock_presenter_interact,
            make_mock_referee(
                continue_debate=False,
                new_thesis="AI 应在高风险、高影响领域受严格监管（最终版）",
                reasoning="论题已足够精确和完善。",
                final_result="经过两轮演化，论题从宽泛走向精确。",
            ),
            checkpointer=checkpointer,
        )

        # Round 2: resume from R2's critique
        graph2.invoke(Command(resume="R2: 用户回应"), config)
        graph2.invoke(Command(resume="R2: 确认论题"), config)
        result = graph2.invoke(None, config)

        assert result["status"] == "done"
        assert len(result["history"]) == 2
        assert result["history"][0].round_number == 1
        assert result["history"][1].round_number == 2
        assert result["final_result"] == "经过两轮演化，论题从宽泛走向精确。"

        # 验证 thesis 演化链
        assert result["history"][0].thesis_before == "AI 应受监管。"
        assert result["history"][0].thesis_after == "AI 应在高风险领域受监管（第1轮拼合）"
        assert result["history"][1].thesis_before == "AI 应在高风险领域受监管（第1轮拼合）"
        assert result["history"][1].thesis_after == "AI 应在高风险、高影响领域受严格监管（最终版）"

    def test_thesis_evolution_across_rounds(self):
        """验证 current_thesis 随轮次演化。"""
        checkpointer = MemorySaver()

        graph = build_graph(
            mock_opponent_compute,
            mock_opponent_interact,
            mock_presenter_compute,
            mock_presenter_interact,
            make_mock_referee(
                continue_debate=True,
                new_thesis="演化后的论题-V1",
            ),
            checkpointer=checkpointer,
        )

        config: dict = {"configurable": {"thread_id": str(uuid4())}}

        r1 = graph.invoke(make_initial_state("初始论题"), config)
        assert r1["current_thesis"] == "初始论题"

        graph.invoke(Command(resume="R1回应"), config)
        r1c = graph.invoke(Command(resume="R1确认"), config)

        # 进入 R2，current_thesis 已更新
        assert r1c["current_thesis"] == "演化后的论题-V1"
        assert r1c["round"] == 2


# =============================================================================
# 中断状态验证测试
# =============================================================================


class TestInterruptState:
    """验证中断点前后的状态一致性。"""

    def test_state_survives_interrupt(self):
        """验证中断时 state 被正确保存，resume 后可恢复。"""
        checkpointer = MemorySaver()
        graph = build_graph(
            mock_opponent_compute,
            mock_opponent_interact,
            mock_presenter_compute,
            mock_presenter_interact,
            make_mock_referee(continue_debate=False),
            checkpointer=checkpointer,
        )

        thread_id = str(uuid4())
        config: dict = {"configurable": {"thread_id": thread_id}}

        # Invoke → interrupt at critique
        graph.invoke(make_initial_state("持久性测试论题"), config)

        # 通过 get_state 读取 checkpoint
        snapshot = graph.get_state(config)
        saved = snapshot.values
        assert saved["current_thesis"] == "持久性测试论题"
        assert saved["round"] == 1
        assert saved["status"] == "awaiting_critique_response"
        assert len(saved["messages"]) == 1

    def test_messages_not_duplicated_on_resume(self):
        """resume 后消息数量正确，无重复。"""
        checkpointer = MemorySaver()
        graph = build_graph(
            mock_opponent_compute,
            mock_opponent_interact,
            mock_presenter_compute,
            mock_presenter_interact,
            make_mock_referee(continue_debate=False),
            checkpointer=checkpointer,
        )

        config: dict = {"configurable": {"thread_id": str(uuid4())}}

        graph.invoke(make_initial_state(), config)
        # 1 msg (opponent) at interrupt

        result = graph.invoke(Command(resume="回应"), config)
        # 3 msgs (opponent + user + presenter) at 2nd interrupt
        assert len(result["messages"]) == 3

        result = graph.invoke(Command(resume="确认"), config)
        # 5 msgs (opponent + user + presenter + user + referee) after done
        assert len(result["messages"]) == 5

        # 验证角色序列
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["opponent", "user", "presenter", "user", "referee"]
