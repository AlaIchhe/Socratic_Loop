"""
Referee 节点 —— 论题拼合者与终局判定者。

职责：
1. 读取本轮的所有输入（current_thesis / _draft_thesis / _confirmed_thesis）。
2. 调用 LLM 获取结构化裁定（支持两种策略：with_structured_output / JSON-mode）。
3. 将本轮论题演化归档为 RoundRecord 追加到 history。
4. 判定是否继续下一轮或结束辩论。
5. 结束时生成 final_result。

策略说明：
- 默认使用 with_structured_output（OpenAI 原生支持）。
- 当 settings.llm_force_json_mode=True 时切换到 JSON-mode 手动解析（DeepSeek 等
  不支持 with_structured_output 的提供商，通过 LLM_FORCE_JSON_MODE 环境变量控制）。
- 两种策略共享相同的归档、路由和总结生成逻辑。

契约：
- 输入：完整的 AgentState（TypedDict）
- 输出：dict，仅包含需要更新的字段
- 不修改 state 本身，不产生副作用
"""

import json
import re
from collections.abc import Mapping

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from config.model import get_chat_model
from config.settings import settings
from core.agents._base import extract_content, invoke_with_retry, make_message
from core.prompts.templates import (
    FINAL_SUMMARY_PROMPT,
    REFEREE_SYSTEM_PROMPT,
    final_summary_prompt,
    referee_prompt,
)
from core.schemas import RefereeJudgment, RoundRecord
from core.state import AgentState

# =============================================================================
# 公开 API
# =============================================================================


def referee_deliberate_node(
    state: AgentState,
    model: BaseChatModel | None = None,
    *,
    json_mode: bool = False,
) -> dict:
    """裁判审议节点：拼合论题并判定是否继续。

    支持两种 LLM 输出策略：
    - json_mode=False（默认）：使用 with_structured_output（OpenAI 等提供商）。
    - json_mode=True：使用 JSON-mode 提示 + 手动解析（DeepSeek 等提供商）。

    Args:
        state: 全局 AgentState。
        model: 可注入的 LLM 实例。默认通过 get_chat_model() 读取配置。
        json_mode: 是否使用 JSON-mode 手动解析（默认 False）。
                   显式传参优先级高于 settings.llm_force_json_mode。

    Returns:
        dict，包含以下键：
        - current_thesis: str          裁判拼合后的新论题（仅 continue 时）
        - messages: list[dict]         追加了 referee 消息的列表
        - history: list[RoundRecord]   追加了本轮归档的列表
        - status: "opponent_computing" | "done"
        - final_result: str            终局总结报告（仅 done 时）
        - _improvement_hint: str       下一轮批判方向指引（仅 continue 时）
    """
    # 决定结构化输出策略：
    # 优先级：显式 json_mode 参数 > state["model_config"]["json_mode"] > settings.llm_force_json_mode（全局回退）
    effective_json_mode = json_mode
    if not effective_json_mode:
        model_cfg = state.get("model_config")
        if model_cfg and model_cfg.get("json_mode"):
            effective_json_mode = True
        else:
            effective_json_mode = settings.llm_force_json_mode

    if model is None:
        # 优先用 state["model_config"] 动态创建；回退到 env-only 路径
        from config.model_manager import build_chat_model, config_from_session

        cfg = config_from_session(state.get("model_config"))
        if cfg is not None:
            cfg.temperature = 0.0  # Referee 始终用 0.0
            model = build_chat_model(cfg)
        else:
            model = get_chat_model(temperature=0.0)

    # --- Step 1: 获取 RefereeJudgment ---
    history_summary = _build_history_summary(state)

    if effective_json_mode:
        judgment = _judge_via_json_mode(
            model,
            state,
            history_summary,
        )
    else:
        judgment = _judge_via_structured_output(
            model,
            state,
            history_summary,
        )

    # --- Step 2: 本轮归档 ---
    result: dict = {
        "history": state["history"] + [_build_round_record(state, judgment)],
    }

    # --- Step 3: 判定路由 ---
    if judgment.continue_debate:
        result["current_thesis"] = judgment.new_thesis
        result["status"] = "opponent_computing"
        result["messages"] = state["messages"]
        result["_improvement_hint"] = judgment.improvement_hint
    else:
        result["status"] = "done"
        final_summary_text = _generate_final_summary(model, state, result, judgment)
        result["messages"] = state["messages"] + [
            make_message("referee", final_summary_text, state["round"])
        ]
        result["final_result"] = final_summary_text

    return result


# =============================================================================
# 策略实现：with_structured_output（默认）
# =============================================================================


def _judge_via_structured_output(
    model: BaseChatModel,
    state: AgentState,
    history_summary: str,
) -> RefereeJudgment:
    """使用 with_structured_output 获取裁判裁定。"""
    structured_model = model.with_structured_output(RefereeJudgment)

    system_msg = SystemMessage(content=REFEREE_SYSTEM_PROMPT)
    user_msg = HumanMessage(
        content=referee_prompt(
            current_thesis=state["current_thesis"],
            draft_thesis=state["_draft_thesis"],
            confirmed_thesis=state["_confirmed_thesis"],
            round_num=state["round"],
            history_summary=history_summary,
        )
    )

    raw = invoke_with_retry(
        structured_model, [system_msg, user_msg], label="RefereeJudgment"
    )
    return raw if isinstance(raw, RefereeJudgment) else RefereeJudgment(**raw)  # type: ignore[arg-type]


# =============================================================================
# 策略实现：JSON-mode 手动解析（DeepSeek 等兼容方案）
# =============================================================================


def _judge_via_json_mode(
    model: BaseChatModel,
    state: AgentState,
    history_summary: str,
) -> RefereeJudgment:
    """使用 JSON-mode 提示 + 手动解析获取裁判裁定。

    DeepSeek 等不支持 with_structured_output 的提供商使用此路径。
    流程：常规 LLM 调用 → regex 提取 JSON → Pydantic 验证。
    """
    system_msg = SystemMessage(content=REFEREE_SYSTEM_PROMPT)
    user_msg = HumanMessage(
        content=referee_prompt(
            current_thesis=state["current_thesis"],
            draft_thesis=state["_draft_thesis"],
            confirmed_thesis=state["_confirmed_thesis"],
            round_num=state["round"],
            history_summary=history_summary,
        )
    )

    response = invoke_with_retry(
        model, [system_msg, user_msg], label="RefereeJudgment(JSON-mode)"
    )
    content = extract_content(response).strip()

    parsed = _extract_json(content)
    if parsed is None:
        raise ValueError(f"无法从裁判响应中解析 JSON:\n{content[:500]}")

    return RefereeJudgment(**parsed)  # type: ignore[arg-type]  # runtime-validated by _extract_json + pydantic


def _extract_json(text: str) -> dict[str, object] | None:
    """从 LLM 响应中提取 JSON 对象。兼容含 Markdown 代码块的输出。

    按优先级尝试：
    1. 直接 json.loads 全文
    2. 提取 ```json ... ``` 代码块
    3. 提取最外层 { ... } 块
    """
    # 直接解析
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        return parsed if isinstance(parsed, dict) else None

    # 提取 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
        else:
            return parsed if isinstance(parsed, dict) else None

    # 提取最外层 { ... }
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            parsed = json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass
        else:
            return parsed if isinstance(parsed, dict) else None

    return None


# =============================================================================
# 共享辅助
# =============================================================================


def _build_round_record(state: AgentState, judgment: RefereeJudgment) -> RoundRecord:
    """根据当前 state 和裁判裁定构造本轮归档记录。"""
    return RoundRecord(
        round_number=state["round"],
        thesis_before=state["current_thesis"],
        critique=state["_critique"],
        user_response=state["_user_response"],
        draft_thesis=state["_draft_thesis"],
        confirmed_thesis=state["_confirmed_thesis"],
        thesis_after=judgment.new_thesis,
        continue_debate=judgment.continue_debate,
        referee_reasoning=judgment.reasoning,
    )


def _build_history_summary(state: AgentState) -> str:
    """构建前轮历史摘要（用于终局判定上下文）。

    注：checkpointer 恢复时 Pydantic 模型会序列化为 dict，故需兼容两种访问方式。
    """
    if not state["history"]:
        return ""
    summaries = []
    for r in state["history"]:
        record = _dump_round_record(r)
        rn = record.get("round_number", "?")
        tb = record.get("thesis_before", "?")
        ta = record.get("thesis_after", "?")
        cb = record.get("continue_debate", "?")
        summaries.append(f"Round {rn}: {tb} -> {ta} (continue: {cb})")
    return "\n".join(summaries)


def _dump_round_record(record: RoundRecord | Mapping[str, object]) -> dict[str, object]:
    """将 RoundRecord 或 checkpoint dict 统一转为可 JSON 序列化的 dict。"""
    if isinstance(record, RoundRecord):
        return record.model_dump()
    return dict(record)


def _generate_final_summary(
    model: BaseChatModel,
    state: AgentState,
    result: dict,
    judgment: RefereeJudgment,
) -> str:
    """生成辩论终止时的最终总结报告。"""
    history_json = json.dumps(
        [_dump_round_record(r) for r in result["history"]],
        ensure_ascii=False,
    )
    summary_system = SystemMessage(content=FINAL_SUMMARY_PROMPT)
    summary_user = HumanMessage(
        content=final_summary_prompt(
            initial_thesis=_get_initial_thesis(state),
            final_thesis=judgment.new_thesis,
            history_json=history_json,
        )
    )
    summary_response = invoke_with_retry(
        model, [summary_system, summary_user], label="FinalSummary"
    )
    return extract_content(summary_response).strip()


def _get_initial_thesis(state: AgentState) -> str:
    """从历史记录中推断初始论题。

    第一轮的 thesis_before 即为用户最初输入的论题。
    注：checkpointer 恢复时 Pydantic 模型会序列化为 dict。
    """
    if state["history"]:
        first = state["history"][0]
        if hasattr(first, "thesis_before"):
            return str(first.thesis_before)
        if isinstance(first, dict):
            return str(first.get("thesis_before", state["current_thesis"]))
        return state["current_thesis"]
    return state["current_thesis"]
