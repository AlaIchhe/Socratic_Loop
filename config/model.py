"""
模型工厂 —— 统一管理 LLM 实例的创建。

运行时多实例路径请用 `config.model_manager`（前端驱动配置），
本模块保留给测试和 env-only 场景（向后兼容）。

配置读取委托 config/settings.py:settings（pydantic-settings 单一读取器），
本模块保留 load_model_config() / has_configured_api_key() 作为薄封装，
保持向后兼容（含测试中通过 env mapping 注入的路径）。

使用方式：
    get_chat_model(temperature) —— 从 os.environ 读取配置（运行时路径）。
    get_chat_model(temperature, model_name=..., base_url=..., api_key=...) —— 显式参数覆盖。
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from config.settings import AppSettings

#: 未配置 API Key 时使用的占位符值。
_PLACEHOLDER_API_KEY = "sk-not-configured"


@dataclass(frozen=True)
class ModelConfig:
    """从环境变量解析出的模型配置。"""

    model_name: str
    base_url: str | None
    api_key: str | None


def _get_env_str(source: Mapping[str, object], key: str) -> str:
    """从环境变量映射读取字符串值；非字符串按缺失处理。"""
    value = source.get(key, "")
    return value if isinstance(value, str) else ""


def _parse_from_mapping(source: Mapping[str, object]) -> ModelConfig:
    """从显式映射解析配置（测试注入路径）。"""
    api_key = (
        _get_env_str(source, "LLM_API_KEY")
        or _get_env_str(source, "OPENAI_API_KEY")
        or None
    )
    if api_key == _PLACEHOLDER_API_KEY:
        api_key = None

    return ModelConfig(
        model_name=_get_env_str(source, "LLM_MODEL") or "gpt-4o",
        base_url=_get_env_str(source, "LLM_BASE_URL") or None,
        api_key=api_key,
    )


def load_model_config(env: Mapping[str, object] | None = None) -> ModelConfig:
    """从环境变量映射读取模型配置。

    - env 为 None 时：构造全新 AppSettings() 读取当前 os.environ
      （运行时路径；每次重新构造以响应运行时的 os.environ 变更，如测试 patch）。
    - env 为 Mapping 时：从映射解析（测试注入路径，保持向后兼容）。
    """
    if env is not None:
        return _parse_from_mapping(env)

    # 运行时路径：重新构造以读取最新 os.environ
    s = AppSettings()
    return ModelConfig(
        # 默认模型名：前端驱动时由 Settings panel 覆盖；env-only 场景回退到 gpt-4o
        model_name=_get_env_str(os.environ, "LLM_MODEL") or "gpt-4o",
        # 空串视为未配置（与旧行为一致："" or None → None）
        base_url=s.llm_base_url or None,
        api_key=s.effective_api_key(),
    )


def has_configured_api_key(env: Mapping[str, object] | None = None) -> bool:
    """判断环境中是否配置了真实 API Key（占位符不算已配置）。"""
    return load_model_config(env).api_key is not None


def get_chat_model(
    temperature: float = 0.7,
    *,
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ChatOpenAI:
    """创建 ChatOpenAI 实例。

    通过环境变量切换供应商：

    OpenAI（默认）:
        LLM_MODEL=gpt-4o
        OPENAI_API_KEY=sk-...

    DeepSeek:
        LLM_MODEL=deepseek-chat
        LLM_BASE_URL=https://api.deepseek.com/v1
        LLM_API_KEY=sk-...

    其他 OpenAI 兼容供应商（如 Ollama、vLLM 等）同理。

    若未配置任何 API Key，会在标准错误流输出诊断信息，
    并将占位符传入 ChatOpenAI——真正调用 LLM 时才会因鉴权失败而报错。

    Args:
        temperature: 0.0 用于裁判（确定性评分），0.7 用于陈述者和反驳者。
        model_name: 可选的模型名覆盖（优先级高于环境变量）。
        base_url: 可选的端点覆盖（优先级高于环境变量）。
        api_key: 可选的 API Key 覆盖（优先级高于环境变量）。
            空串 "" 视同未传入，会回退到环境变量。

    Returns:
        配置好的 ChatOpenAI 实例。
    """
    config = load_model_config()

    # 参数优先级：显式传入 api_key > 环境变量 config.api_key > placeholder
    # 注意：空串视为未传入（与 str | None 语义一致）
    effective_key = api_key if api_key else config.api_key

    if not effective_key:
        warnings.warn(
            "未检测到 LLM_API_KEY 或 OPENAI_API_KEY 环境变量。"
            "请在项目根目录的 .env 文件中配置 API Key，"
            "或通过环境变量设置。示例：LLM_API_KEY=sk-your-key",
            RuntimeWarning,
            stacklevel=2,
        )
        effective_key = _PLACEHOLDER_API_KEY

    return ChatOpenAI(
        model=model_name or config.model_name,
        temperature=temperature,
        base_url=base_url if base_url is not None else config.base_url,
        api_key=effective_key,  # type: ignore[arg-type]  # langchain 类型桩使用 SecretStr
        streaming=True,  # 启用 token 级回调，供 graph.astream() 使用
    )
