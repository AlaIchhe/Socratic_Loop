"""
共享 Mock Agent 节点 —— 消除跨测试文件的 mock 节点重复。

提供完整的 mock 节点（含 interrupt()），用于集成测试和接口测试。
"""

from langgraph.types import interrupt

from core.schemas import RoundRecord
from core.state import AgentState

# =============================================================================
# Mock Opponent
# =============================================================================


def mock_opponent_compute(state: AgentState) -> dict:
    """Mock 批判者计算节点：返回固定批判文本。"""
    return {
        "_critique": f"[Critique R{state['round']}] 论题存在模糊之处",
        "messages": state["messages"]
        + [
            {
                "role": "opponent",
                "content": f"[Critique R{state['round']}] 论题存在模糊之处",
                "round": state["round"],
            }
        ],
        "status": "awaiting_critique_response",
    }


def mock_opponent_interact(state: AgentState) -> dict:
    """Mock 批判者交互节点：含真实 interrupt()，暂停等待用户回应。"""
    critique = state["_critique"]
    user_response = interrupt(critique)
    return {
        "_user_response": str(user_response),
        "messages": state["messages"]
        + [
            {
                "role": "user",
                "content": str(user_response),
                "round": state["round"],
            }
        ],
        "status": "presenter_computing",
    }


# =============================================================================
# Mock Presenter
# =============================================================================


def mock_presenter_compute(state: AgentState) -> dict:
    """Mock 精确化者计算节点：返回固定草稿文本。"""
    return {
        "_draft_thesis": f"[Draft R{state['round']}] 精确化后的论题",
        "messages": state["messages"]
        + [
            {
                "role": "presenter",
                "content": f"[Draft R{state['round']}] 精确化后的论题",
                "round": state["round"],
            }
        ],
        "status": "awaiting_thesis_confirmation",
    }


def mock_presenter_interact(state: AgentState) -> dict:
    """Mock 精确化者交互节点：含真实 interrupt()，暂停等待用户确认。"""
    draft = state["_draft_thesis"]
    confirmed = interrupt(draft)
    return {
        "_confirmed_thesis": str(confirmed),
        "messages": state["messages"]
        + [
            {
                "role": "user",
                "content": str(confirmed),
                "round": state["round"],
            }
        ],
        "status": "referee_deliberating",
    }


# =============================================================================
# Mock Referee 工厂
# =============================================================================


def make_mock_referee(
    continue_debate: bool = True,
    new_thesis: str = "拼合后的新论题",
    reasoning: str = "裁判理由",
    final_result: str = "终局总结。",
):
    """构造 mock 裁判节点函数（闭包工厂）。

    Args:
        continue_debate: 裁判是否继续下一轮。
        new_thesis: 裁判拼合后的新论题。
        reasoning: 裁判判定理由。
        final_result: 终局总结（仅 continue_debate=False 时使用）。
    """

    def _referee(state: AgentState) -> dict:
        record = RoundRecord(
            round_number=state["round"],
            thesis_before=state["current_thesis"],
            critique=state["_critique"],
            user_response=state["_user_response"],
            draft_thesis=state["_draft_thesis"],
            confirmed_thesis=state["_confirmed_thesis"],
            thesis_after=new_thesis,
            continue_debate=continue_debate,
            referee_reasoning=reasoning,
        )
        result: dict = {
            "messages": state["messages"]
            + [
                {
                    "role": "referee",
                    "content": f"[Judgment R{state['round']}] {reasoning}",
                    "round": state["round"],
                }
            ],
            "history": state["history"] + [record],
        }
        if continue_debate:
            result["current_thesis"] = new_thesis
            result["status"] = "opponent_computing"
            result["_improvement_hint"] = ""
        else:
            result["status"] = "done"
            result["final_result"] = final_result
        return result

    return _referee
