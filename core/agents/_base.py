"""
Agent 节点共享基础工具 —— 消除 Compute/Interact 节点间的模板重复。

原则：
1. 只提取纯机械性重复（内容提取、消息构造、节点骨架），不引入不必要抽象。
2. 不对裁判节点强制套用模板 —— 其 with_structured_output + 双路径逻辑足够独特。
3. 所有函数无副作用，不访问全局状态。
"""

import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from config.model import get_chat_model
from config.settings import settings
from core.state import AgentState

logger = logging.getLogger(__name__)

# =============================================================================
# LLM 调用重试配置（运行时从 config/settings.py:settings 读取）
# =============================================================================

_RETRYABLE_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
)
"""可重试的瞬时错误类型。"""


def _is_retryable(error: Exception) -> bool:
    """判断异常是否可重试（网络/超时问题，非逻辑/鉴权错误）。"""
    if isinstance(error, _RETRYABLE_ERRORS):
        return True
    # langchain / openai 的 RateLimitError, APITimeoutError 等
    error_name = type(error).__name__
    return any(
        keyword in error_name.lower()
        for keyword in ("timeout", "ratelimit", "connection", "apiconnection")
    )


# =============================================================================
# 内容提取 & 消息构造（消除 3×/6× 重复）
# =============================================================================


def extract_content(response: BaseMessage) -> str:
    """从 LLM 响应中提取纯文本内容。

    兼容纯文本（response.content 为 str）和多模态（list[dict]）两种响应格式。
    """
    content = response.content
    return content if isinstance(content, str) else str(content)


def make_message(role: str, content: str, round_num: int) -> dict[str, object]:
    """构造符合 AgentState.messages 格式的消息字典。

    Args:
        role: 消息角色（opponent / presenter / referee / user）。
        content: 消息正文。
        round_num: 所属轮次编号。

    返回的 dict 包含 timestamp 字段（float, time.time()），
    用于 UI 层在消息气泡底部显示时间。
    """
    return {
        "role": role,
        "content": content,
        "round": round_num,
        "timestamp": time.time(),
    }


# =============================================================================
# Compute 节点工厂（消除 Opponent/Presenter 间的 3× 模板重复）
# =============================================================================


class _Invocable(Protocol):
    """任何具有 .invoke(messages) -> BaseMessage 方法的对象。

    兼容 BaseChatModel 和 with_structured_output() 返回的 Runnable。
    使用 Protocol 而非 BaseChatModel 以接受结构化输出包装器。
    """

    def invoke(self, input: Any, **kwargs: Any) -> Any: ...


def invoke_with_retry(
    invocable: _Invocable,
    messages: list[BaseMessage],
    *,
    label: str = "LLM",
    on_retry: Callable[[int, int, float, Exception], None] | None = None,
) -> BaseMessage:
    """调用 invocable.invoke(messages)，失败时自动重试。

    对瞬时网络/超时/速率限制错误自动重试（最多 3 次，指数退避 1s/2s/4s）。

    Args:
        invocable: 可调用 .invoke(messages) 的对象（BaseChatModel 或 structured model）。
        messages: 消息列表。
        label: 日志标签（用于区分不同调用点）。
        on_retry: 可选回调，签名 (attempt, max_retries, wait_seconds, exception) -> None。
                  在每次重试前调用，用于向 UI 层报告重试进度。

    Returns:
        invocable.invoke() 的原始返回值。

    Raises:
        最后一次尝试的异常（重试耗尽后）。
    """
    # 每次调用时从 settings 读取，便于测试覆盖和运行时调整
    max_retries = settings.llm_max_retries
    backoff_base = settings.llm_retry_backoff_base

    retry_count = 0
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            result = cast("BaseMessage", invocable.invoke(messages))
            return result
        except Exception as exc:
            last_error = exc
            retry_count += 1
            if not _is_retryable(exc) or attempt == max_retries:
                raise
            wait = backoff_base * (2 ** (attempt - 1))
            logger.warning(
                "%s 调用失败（第 %d/%d 次），%s 秒后重试: %s",
                label,
                attempt,
                max_retries,
                wait,
                exc,
            )
            if on_retry:
                with contextlib.suppress(Exception):
                    on_retry(attempt, max_retries, wait, exc)
            time.sleep(wait)

    assert last_error is not None
    raise last_error


def invoke_llm(
    model: BaseChatModel | None,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    *,
    model_config: dict[str, object] | None = None,
    on_retry: Callable[[int, int, float, Exception], None] | None = None,
) -> str:
    """调用 LLM 并返回提取后的文本内容（含自动重试）。

    封装 model 懒初始化 + SystemMessage/HumanMessage 构造 + invoke + 内容提取。

    Args:
        model: 可选的预注入模型（None 则按 model_config 或 env 创建）。
        temperature: LLM 温度参数（0.0 = 确定性，0.7 = 创造性）。
        system_prompt: 系统提示。
        user_prompt: 用户提示。
        model_config: 前端驱动的模型配置 dict（None 则回退到 env）。
            结构对齐 config.model_manager.ModelConfig。
        on_retry: 可选的重试进度回调。

    Returns:
        LLM 响应文本（已 strip）。
    """
    if model is None and model_config is not None:
        from config.model_manager import build_chat_model, config_from_session

        cfg = config_from_session(model_config)
        if cfg is not None:
            # model_config 中的 temperature 优先级低于显式传入的 temperature 参数
            # （显式 temperature 是 node 根据角色指定的：Opponent/Presenter=0.7, Referee=0.0）
            cfg.temperature = temperature
            model = build_chat_model(cfg)

    if model is None:
        model = get_chat_model(temperature=temperature)

    response = invoke_with_retry(
        model,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        label="LLM",
        on_retry=on_retry,
    )
    return extract_content(response).strip()


# =============================================================================
# 类型别名
# =============================================================================

#: Agent 节点函数的类型签名。
NodeFunc = Callable[[AgentState], dict]
