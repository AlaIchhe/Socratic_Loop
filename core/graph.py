"""
LangGraph 编排层 —— 纯粹的流转调度。

原则：
1. 图中不包含任何 LLM 调用逻辑，只做状态路由和调度。
2. 所有 LLM 逻辑封装在 agents/ 的节点函数中。
3. 使用动态 interrupt() 替代静态 interrupt_before，实现精准的人工介入。
4. 条件边（Conditional Edge）根据 state["status"] 决定下一步流向。

图结构：
    START
      │
      └──→ start (round=1, status=opponent_computing)
             │
             └──→ opponent_compute    (LLM: 批判 current_thesis)
                    │
                    └──→ opponent_interact   (interrupt: 展示 critique)
                           │
                           └──→ presenter_compute   (LLM: 精确化用户回应)
                                  │
                                  └──→ presenter_interact  (interrupt: 展示 draft)
                                         │
                                         └──→ referee_deliberate  (LLM: 拼合, 判定)
                                                │
                                         ┌──────┴──────┐
                                         │             │
                                  [continue]       [done]
                                         │             │
                                   next_round        END
                                         │
                                         └──→ opponent_compute (循环)
"""

from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from core.agents.opponent import opponent_compute_node, opponent_interact_node
from core.agents.presenter import presenter_compute_node, presenter_interact_node
from core.agents.referee import referee_deliberate_node
from core.state import AgentState, validate_state_shape

# =============================================================================
# 纯调度节点（无 LLM 逻辑）
# =============================================================================


def _start_node(state: AgentState) -> dict:
    """初始化节点：将 idle 状态转为 opponent_computing，触发首轮批判。

    该节点仅在入口调用一次，用于状态机启动。
    """
    validate_state_shape(state)
    return {"round": 1, "status": "opponent_computing"}


def _next_round_node(state: AgentState) -> dict:
    """轮次推进节点：round+1，清空本轮缓存，准备下一轮。

    这是纯调度逻辑，不涉及任何 LLM。
    """
    return {
        "round": state["round"] + 1,
        "_critique": "",
        "_user_response": "",
        "_draft_thesis": "",
        "_confirmed_thesis": "",
        "_improvement_hint": "",
    }


# =============================================================================
# 条件路由
# =============================================================================


def _route_after_referee(state: AgentState) -> str:
    """Referee 审议后的条件路由。

    依据 state["status"] 判定：
    - "done"                  → 辩论结束，流向 END
    - 达到 max_rounds         → 强制终止，流向 END（安全阀）
    - "opponent_computing"    → 继续下一轮
    """
    if state["status"] == "done":
        return END

    # 安全阀：达到最大轮次强制终止，防止异常状态下无限循环
    max_rounds = state.get("max_rounds", 10)
    if state["round"] >= max_rounds:
        return END

    return "next_round"


# =============================================================================
# 图构建
# =============================================================================


def build_graph(
    opponent_compute_node: Callable[[AgentState], dict[str, Any]],
    opponent_interact_node: Callable[[AgentState], dict[str, Any]],
    presenter_compute_node: Callable[[AgentState], dict[str, Any]],
    presenter_interact_node: Callable[[AgentState], dict[str, Any]],
    referee_deliberate_node: Callable[[AgentState], dict[str, Any]],
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph[Any]:
    """组装 LangGraph 状态图。

    所有 LLM 节点以依赖注入方式传入，图本身只负责编排。
    使用动态 interrupt() 进行人工介入，无需 interrupt_before 配置。

    Args:
        opponent_compute_node:   批判者计算节点
        opponent_interact_node:  批判者交互节点（含 interrupt()）
        presenter_compute_node:  精确化者计算节点
        presenter_interact_node: 精确化者交互节点（含 interrupt()）
        referee_deliberate_node: 裁判审议节点
        checkpointer:            LangGraph checkpointer 实例（必须传入才能支持
                                 interrupt() 暂停/恢复和 get_state()）。

    Returns:
        编译后的 CompiledStateGraph。
    """
    workflow = StateGraph(AgentState)

    # 注册所有节点
    workflow.add_node("start", _start_node)
    workflow.add_node("opponent_compute", opponent_compute_node)  # type: ignore[arg-type]
    workflow.add_node("opponent_interact", opponent_interact_node)  # type: ignore[arg-type]
    workflow.add_node("presenter_compute", presenter_compute_node)  # type: ignore[arg-type]
    workflow.add_node("presenter_interact", presenter_interact_node)  # type: ignore[arg-type]
    workflow.add_node("referee_deliberate", referee_deliberate_node)  # type: ignore[arg-type]
    workflow.add_node("next_round", _next_round_node)

    # 固定边
    workflow.set_entry_point("start")
    workflow.add_edge("start", "opponent_compute")
    workflow.add_edge("opponent_compute", "opponent_interact")
    workflow.add_edge("opponent_interact", "presenter_compute")
    workflow.add_edge("presenter_compute", "presenter_interact")
    workflow.add_edge("presenter_interact", "referee_deliberate")

    # Referee 后的条件边
    workflow.add_conditional_edges(
        "referee_deliberate",
        _route_after_referee,
        {
            END: END,
            "next_round": "next_round",
        },
    )

    # 下一轮 → 批判者计算（形成循环）
    workflow.add_edge("next_round", "opponent_compute")

    # 编译图：不配置 interrupt_before，人工介入由动态 interrupt() 负责
    return workflow.compile(checkpointer=checkpointer)


# =============================================================================
# 默认图构建器 —— 供 UI 层使用，避免直接依赖 agents/
# =============================================================================


def build_default_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph[Any]:
    """使用默认 agent 节点组装 LangGraph 状态图。

    这是 UI 层和快捷入口的推荐入口：调用者无需了解 agent 节点的
    具体实现，只需传入 checkpointer 即可获得可编译的图。

    与 build_graph() 的区别：build_graph() 接受节点函数作为参数
    （适合测试注入 mock），build_default_graph() 使用真实的 LLM 节点。

    Args:
        checkpointer: LangGraph checkpointer 实例（必须传入才能支持
                     interrupt() 暂停/恢复和 get_state()）。

    Returns:
        编译后的 CompiledStateGraph。
    """
    return build_graph(
        opponent_compute_node=opponent_compute_node,
        opponent_interact_node=opponent_interact_node,
        presenter_compute_node=presenter_compute_node,
        presenter_interact_node=presenter_interact_node,
        referee_deliberate_node=referee_deliberate_node,
        checkpointer=checkpointer,
    )
