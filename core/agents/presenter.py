"""
Presenter 节点 —— 精确化者（分为 compute + interact 两个节点）。

职责：
1. presenter_compute_node: 读取用户的非正式回应，调用 LLM 精确化为学术论题。
2. presenter_interact_node: 将精确化草稿通过 interrupt() 展示给用户确认。

拆分原因：LangGraph 的 interrupt() 在 resume 时会整节点重新执行。
将 LLM 调用放在 compute 节点中可避免 resume 时重复调用 LLM。

契约：
- 输入：完整的 AgentState（TypedDict）
- 输出：dict，仅包含需要更新的字段
- 不修改 state 本身，不产生副作用
"""

from langchain_core.language_models import BaseChatModel
from langgraph.types import interrupt

from core.agents._base import invoke_llm, make_message
from core.prompts.templates import PRESENTER_SYSTEM_PROMPT, presenter_prompt
from core.state import AgentState


def presenter_compute_node(
    state: AgentState, model: BaseChatModel | None = None
) -> dict:
    """精确化者计算节点：将用户回应转化为精确论题表述。

    Args:
        state: 全局 AgentState，至少需包含 current_thesis / _critique /
               _user_response / round / messages。
        model: 可注入的 LLM 实例。

    Returns:
        dict，包含以下键：
        - _draft_thesis: str          LLM 精确化后的论题草稿
        - messages: list[dict]        追加了 presenter 消息的列表
        - status: "awaiting_thesis_confirmation"
    """
    draft = invoke_llm(
        model=model,
        temperature=float(state.get("agent_temperature", 0.7)),
        system_prompt=PRESENTER_SYSTEM_PROMPT,
        user_prompt=presenter_prompt(
            current_thesis=state["current_thesis"],
            critique=state["_critique"],
            user_response=state["_user_response"],
        ),
        model_config=state.get("model_config"),
    )

    return {
        "_draft_thesis": draft,
        "messages": state["messages"]
        + [make_message("presenter", draft, state["round"])],
        "status": "awaiting_thesis_confirmation",
    }


def presenter_interact_node(state: AgentState) -> dict:
    """精确化者交互节点：展示草稿并收取用户确认。

    此节点不含 LLM 调用，仅做两件事：
    1. 从 state 读取 _draft_thesis，通过 interrupt() 展示给用户
    2. resume 后将用户确认（可能编辑过的）论题写入 _confirmed_thesis

    Args:
        state: 全局 AgentState，至少需包含 _draft_thesis / round / messages。

    Returns:
        dict，包含以下键：
        - _confirmed_thesis: str      用户确认后的论题
        - messages: list[dict]        追加了 user 消息的列表
        - status: "referee_deliberating"
    """
    draft = state["_draft_thesis"]

    confirmed = interrupt(draft)

    return {
        "_confirmed_thesis": str(confirmed),
        "messages": state["messages"]
        + [make_message("user", str(confirmed), state["round"])],
        "status": "referee_deliberating",
    }
