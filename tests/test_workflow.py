"""
Workflow 图编排的单元测试。

测试目标：验证图的节点注册、边连接、条件路由和中断配置。
所有节点用纯函数 Mock，不涉及 LLM 调用。
"""

from unittest.mock import MagicMock, patch

import pytest

from core.schemas import RoundRecord
from core.state import AgentState
from core.graph import (
    _next_round_node,
    _route_after_referee,
    _start_node,
    build_graph,
)
from tests.helpers import make_initial_state


def _mock_opponent_compute(state: AgentState) -> dict:
    return {
        "_critique": f"第{state['round']}轮批判",
        "messages": state["messages"]
        + [{"role": "opponent", "content": f"批判-{state['round']}", "round": state["round"]}],
        "status": "awaiting_critique_response",
    }


def _mock_opponent_interact(state: AgentState) -> dict:
    return {
        "_user_response": f"用户回应-{state['round']}",
        "messages": state["messages"]
        + [{"role": "user", "content": f"回应-{state['round']}", "round": state["round"]}],
        "status": "presenter_computing",
    }


def _mock_presenter_compute(state: AgentState) -> dict:
    return {
        "_draft_thesis": f"第{state['round']}轮草稿论题",
        "messages": state["messages"]
        + [{"role": "presenter", "content": f"草稿-{state['round']}", "round": state["round"]}],
        "status": "awaiting_thesis_confirmation",
    }


def _mock_presenter_interact(state: AgentState) -> dict:
    return {
        "_confirmed_thesis": f"确认论题-{state['round']}",
        "messages": state["messages"]
        + [{"role": "user", "content": f"确认-{state['round']}", "round": state["round"]}],
        "status": "referee_deliberating",
    }


def _mock_referee_continue(state: AgentState) -> dict:
    """Mock 裁判：继续下一轮。"""
    new_record = RoundRecord(
        round_number=state["round"],
        thesis_before=state["current_thesis"],
        critique=state["_critique"],
        user_response=state["_user_response"],
        draft_thesis=state["_draft_thesis"],
        confirmed_thesis=state["_confirmed_thesis"],
        thesis_after=f"新论题-轮{state['round']}",
        continue_debate=True,
        referee_reasoning="继续理由",
    )
    return {
        "current_thesis": f"新论题-轮{state['round']}",
        "messages": state["messages"] + [{"role": "referee", "content": "继续", "round": state["round"]}],
        "history": state["history"] + [new_record],
        "status": "opponent_computing",
    }


def _mock_referee_done(state: AgentState) -> dict:
    """Mock 裁判：结束辩论。"""
    new_record = RoundRecord(
        round_number=state["round"],
        thesis_before=state["current_thesis"],
        critique=state["_critique"],
        user_response=state["_user_response"],
        draft_thesis=state["_draft_thesis"],
        confirmed_thesis=state["_confirmed_thesis"],
        thesis_after=f"最终论题-轮{state['round']}",
        continue_debate=False,
        referee_reasoning="论题已足够完善",
    )
    return {
        "messages": state["messages"] + [{"role": "referee", "content": "结束", "round": state["round"]}],
        "history": state["history"] + [new_record],
        "status": "done",
        "final_result": "终局总结报告。",
    }


# =============================================================================
# Start Node 测试
# =============================================================================


class TestStartNode:
    """_start_node 测试。"""

    def test_sets_status_to_opponent_computing(self):
        state = make_initial_state(status="idle", round=0)
        result = _start_node(state)

        assert result["status"] == "opponent_computing"

    def test_sets_round_to_one(self):
        state = make_initial_state(status="idle", round=0)
        result = _start_node(state)

        assert result["round"] == 1

    def test_only_touches_round_and_status(self):
        state = make_initial_state(status="idle", round=0)
        result = _start_node(state)

        assert set(result.keys()) == {"round", "status"}

    def test_missing_state_key_fails_before_agent_nodes(self):
        """入口 state 不完整时，start_node 应在调度层早失败。"""
        state = dict(make_initial_state())
        state.pop("_improvement_hint")

        with pytest.raises(KeyError, match="_improvement_hint"):
            _start_node(state)  # type: ignore[arg-type]


# =============================================================================
# Next Round Node 测试
# =============================================================================


class TestNextRoundNode:
    """_next_round_node 测试。"""

    def test_increments_round(self):
        state = make_initial_state(round=2)
        result = _next_round_node(state)

        assert result["round"] == 3

    def test_clears_round_cache(self):
        state = make_initial_state(
            _critique="批判",
            _user_response="回应",
            _draft_thesis="草稿",
            _confirmed_thesis="确认",
        )
        result = _next_round_node(state)

        assert result["_critique"] == ""
        assert result["_user_response"] == ""
        assert result["_draft_thesis"] == ""
        assert result["_confirmed_thesis"] == ""

    def test_returns_exactly_six_keys(self):
        state = make_initial_state(round=1)
        result = _next_round_node(state)

        assert set(result.keys()) == {
            "round",
            "_critique",
            "_user_response",
            "_draft_thesis",
            "_confirmed_thesis",
            "_improvement_hint",
        }


# =============================================================================
# 条件路由测试
# =============================================================================


class TestRouteAfterReferee:
    """_route_after_referee 测试。"""

    def test_routes_to_end_when_done(self):
        state = make_initial_state(status="done")
        from langgraph.graph import END

        assert _route_after_referee(state) == END

    def test_routes_to_next_round_when_opponent_computing(self):
        state = make_initial_state(status="opponent_computing")
        assert _route_after_referee(state) == "next_round"

    def test_route_never_returns_none(self):
        """所有合法 status 都应路由到有效目标。"""
        statuses = [
            "idle",
            "opponent_computing",
            "awaiting_critique_response",
            "presenter_computing",
            "awaiting_thesis_confirmation",
            "referee_deliberating",
            "done",
        ]
        for s in statuses:
            state = make_initial_state(status=s)  # type: ignore[arg-type]
            result = _route_after_referee(state)
            assert result is not None, f"status={s} 返回了 None"
            assert result in ("next_round", "__end__"), f"status={s} 返回了 {result}"


# =============================================================================
# 图构建测试
# =============================================================================


class TestBuildGraph:
    """build_graph() 测试。"""

    def test_returns_compiled_graph(self):
        graph = build_graph(
            _mock_opponent_compute,
            _mock_opponent_interact,
            _mock_presenter_compute,
            _mock_presenter_interact,
            _mock_referee_continue,
        )

        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")

    def test_graph_has_all_nodes(self):
        graph = build_graph(
            _mock_opponent_compute,
            _mock_opponent_interact,
            _mock_presenter_compute,
            _mock_presenter_interact,
            _mock_referee_continue,
        )

        node_names = set(graph.nodes.keys())  # type: ignore[union-attr]
        assert "start" in node_names
        assert "opponent_compute" in node_names
        assert "opponent_interact" in node_names
        assert "presenter_compute" in node_names
        assert "presenter_interact" in node_names
        assert "referee_deliberate" in node_names
        assert "next_round" in node_names

    def test_all_mock_nodes_are_invoked(self):
        """验证图运行时所有 mock 节点都被调用（使用 spy）。"""
        from langgraph.checkpoint.memory import MemorySaver

        oc_spy = MagicMock(side_effect=_mock_opponent_compute)
        oi_spy = MagicMock(side_effect=_mock_opponent_interact)
        pc_spy = MagicMock(side_effect=_mock_presenter_compute)
        pi_spy = MagicMock(side_effect=_mock_presenter_interact)
        rd_spy = MagicMock(side_effect=_mock_referee_done)

        graph = build_graph(
            oc_spy,
            oi_spy,
            pc_spy,
            pi_spy,
            rd_spy,
            checkpointer=MemorySaver(),
        )

        initial_state = make_initial_state()
        config = {"configurable": {"thread_id": "test-invoke"}}

        # 注意：无 checkpointer 时 interrupt() 会失败。
        # 但我们的 mock opponent_interact 不调用 interrupt()，
        # 所以可以直接 invoke 走通全图。
        graph.invoke(initial_state, config)

        oc_spy.assert_called_once()
        oi_spy.assert_called_once()
        pc_spy.assert_called_once()
        pi_spy.assert_called_once()
        rd_spy.assert_called_once()

    def test_no_interrupt_before_configured(self):
        """新图不使用 interrupt_before —— 人工介入由动态 interrupt() 负责。"""
        graph = build_graph(
            _mock_opponent_compute,
            _mock_opponent_interact,
            _mock_presenter_compute,
            _mock_presenter_interact,
            _mock_referee_continue,
        )

        # interrupt_before_nodes 应为空或不存在
        ib_nodes = getattr(graph, "interrupt_before_nodes", None)
        assert ib_nodes is None or ib_nodes == []

    def test_graph_runs_full_cycle_to_done(self):
        """使用 mock，验证图可以从 idle 走到 done。"""
        from langgraph.checkpoint.memory import MemorySaver

        graph = build_graph(
            _mock_opponent_compute,
            _mock_opponent_interact,
            _mock_presenter_compute,
            _mock_presenter_interact,
            _mock_referee_done,  # 裁判判定结束
            checkpointer=MemorySaver(),
        )

        initial_state = make_initial_state()
        config = {"configurable": {"thread_id": "test-full"}}
        result = graph.invoke(initial_state, config)

        assert result["status"] == "done"
        assert len(result["history"]) == 1
        assert result["final_result"] == "终局总结报告。"


# =============================================================================
# 路由边界测试
# =============================================================================


class TestRouteEdgeCases:
    """_route_after_referee 边界值测试。"""

    def test_missing_status_key_raises(self):
        """state 缺少 status 时抛出 KeyError。"""
        state: dict = {"current_thesis": "x", "round": 1}
        with pytest.raises(KeyError):
            _route_after_referee(state)  # type: ignore[arg-type]

    def test_unknown_status_defaults_to_next_round(self):
        """未知 status 值默认路由到 next_round。"""
        state = make_initial_state(status="bogus_value")  # type: ignore[arg-type]
        assert _route_after_referee(state) == "next_round"


# =============================================================================
# 调度节点边界测试
# =============================================================================


class TestSchedulingEdgeCases:
    """start_node 和 next_round_node 的边界行为。"""

    def test_start_node_overwrites_non_idle_status(self):
        """start_node 即使 status 不是 idle 也会覆盖。"""
        state = make_initial_state(status="done", round=5)
        result = _start_node(state)
        assert result["status"] == "opponent_computing"
        assert result["round"] == 1

    def test_next_round_preserves_unrelated_fields(self):
        """next_round_node 不修改 current_thesis 等非缓存字段。"""
        state = make_initial_state(
            round=2,
            current_thesis="保持不变的论题",
            history=[MagicMock()],
            final_result="不改变",
            _critique="c",
            _user_response="u",
            _draft_thesis="d",
            _confirmed_thesis="cf",
        )
        result = _next_round_node(state)
        assert result["round"] == 3
        # 缓存字段被清除
        assert result["_critique"] == ""
        assert result["_user_response"] == ""
        assert result["_draft_thesis"] == ""
        assert result["_confirmed_thesis"] == ""
        # 非缓存字段不在返回 dict 中（节点不应修改它们）
        assert "current_thesis" not in result
        assert "history" not in result
        assert "final_result" not in result


# =============================================================================
# 图构建边界测试
# =============================================================================


class TestBuildGraphEdgeCases:
    """build_graph() 的边界行为。"""

    def test_build_without_checkpointer_compiles(self):
        """checkpointer=None 时图应编译成功（虽然 interrupt 会在运行时失败）。"""

        def _oc(s):
            return {"_critique": "c", "messages": [], "status": "awaiting_critique_response"}

        def _oi(s):
            return {"_user_response": "u", "messages": [], "status": "presenter_computing"}

        def _pc(s):
            return {"_draft_thesis": "d", "messages": [], "status": "awaiting_thesis_confirmation"}

        def _pi(s):
            return {"_confirmed_thesis": "cf", "messages": [], "status": "referee_deliberating"}

        def _rd(s):
            return {"messages": [], "history": [], "status": "done", "final_result": "done"}

        graph = build_graph(_oc, _oi, _pc, _pi, _rd, checkpointer=None)
        assert graph is not None
        assert hasattr(graph, "invoke")
