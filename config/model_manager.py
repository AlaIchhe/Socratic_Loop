"""
模型管理器 —— 从 ModelConfig 动态创建 ChatOpenAI 实例。

这是 config/model.py 的"运行时多实例"版本。
node 函数统一走本模块，model.py 保留给测试和 env-only 场景。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI


@dataclass
class ModelConfig:
    """前端传入的模型配置（对应 cl.user_session["model_config"]）。

    字段对齐 AgentState["model_config"] dict 表示。
    """

    provider: str
    """Provider 标识（"openai" / "deepseek" / ...）。"""

    base_url: str
    """API 端点 URL。"""

    api_key: str
    """API Key。"""

    model: str
    """模型名称。"""

    temperature: float = 0.7
    """生成温度。"""

    max_tokens: int | None = None
    """最大生成 token 数。None 表示不限制。"""

    json_mode: bool = False
    """是否强制 JSON mode（response_format=json_object）。"""


def build_chat_model(config: ModelConfig) -> ChatOpenAI:
    """根据 ModelConfig 创建 ChatOpenAI 实例。

    针对 json_mode 等 provider 相关能力自动应用 response_format；
    由调用方决定是否启用（可结合 providers.REGISTRY 的 supports_json_mode）。
    """
    kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "base_url": config.base_url,
        "api_key": config.api_key,
        "streaming": True,
    }
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    if config.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return ChatOpenAI(**kwargs)


def config_from_session(session_config: dict[str, Any] | None) -> ModelConfig | None:
    """从 session dict 构造 ModelConfig，None 表示"未配置"。

    缺失必要字段（base_url / api_key / model）时返回 None，
    交由调用方决定回退策略（回退到 env 或报错）。
    """
    if not session_config:
        return None

    base_url = session_config.get("base_url")
    api_key = session_config.get("api_key")
    model = session_config.get("model")

    if not base_url or not api_key or not model:
        return None

    return ModelConfig(
        provider=str(session_config.get("provider", "custom")),
        base_url=str(base_url),
        api_key=str(api_key),
        model=str(model),
        temperature=float(session_config.get("temperature", 0.7)),
        max_tokens=session_config.get("max_tokens")
        if session_config.get("max_tokens") is not None
        else None,
        json_mode=bool(session_config.get("json_mode", False)),
    )
