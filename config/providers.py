"""
Provider 注册表 —— 借鉴 Dify 的 model_providers/ 目录模式。

每个 Provider 是一个 dataclass，声明：
- display_name: 前端显示名
- default_base_url: 默认端点（用户可覆盖）
- default_models: 静态 fallback 模型列表（当 /v1/models 不可用时）
- supports_json_mode: 是否原生支持 response_format=json_object
- max_temperature: 参数上限（前端 slider 用）

加新 Provider = 在 REGISTRY 里加一行，零侵入。

凭证校验与模型发现使用 httpx（chainlit 的 transitive dep，无需新增依赖）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import httpx


# =============================================================================
# 校验结果
# =============================================================================

ValidationStatus = Literal[
    "valid",
    "invalid_key",
    "connection_error",
    "unknown_error",
]


@dataclass(frozen=True)
class ValidationResult:
    """凭证校验结果。"""

    status: ValidationStatus
    """校验状态：valid / invalid_key / connection_error / unknown_error。"""

    message: str
    """人类可读的结果信息（成功时提示可用模型数，失败时提示错误细节）。"""

    models: list[str] = field(default_factory=list)
    """校验成功时顺带拉取的可用模型列表（可选）。"""

    @property
    def valid(self) -> bool:
        return self.status == "valid"


# =============================================================================
# Provider 定义
# =============================================================================


@dataclass(frozen=True)
class Provider:
    """一个 LLM provider 的完整描述。"""

    key: str
    """唯一标识（"openai" / "deepseek" / ...），也是前端 dropdown 的 value。"""

    display_name: str
    """前端显示名。"""

    default_base_url: str | None
    """默认端点。None 表示该 provider 必须由用户显式提供。"""

    default_models: list[str]
    """静态 fallback 模型列表（当 /v1/models 不可用时）。"""

    supports_json_mode: bool = True
    """是否原生支持 response_format=json_object。"""

    max_temperature: float = 2.0
    """前端 temperature slider 的上限。"""

    # ------------------------------------------------------------------
    # 凭证校验 & 模型发现
    # ------------------------------------------------------------------

    def validate(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 10.0,
    ) -> ValidationResult:
        """发一次极小真实调用验证凭证。

        策略：POST {base_url}/chat/completions，max_tokens=1，单条短消息。
        通过 → valid；401/403 → invalid_key；超时/网络错误 → connection_error。
        """
        if not base_url:
            return ValidationResult(
                status="connection_error",
                message="Base URL 不能为空",
            )
        if not api_key:
            return ValidationResult(
                status="invalid_key",
                message="API Key 不能为空",
            )

        url = base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.default_models[0] if self.default_models else "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }

        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        except httpx.TimeoutException:
            return ValidationResult(
                status="connection_error",
                message=f"连接超时（{timeout}s）。请检查 Base URL 是否正确。",
            )
        except httpx.ConnectError as e:
            return ValidationResult(
                status="connection_error",
                message=f"无法连接到端点：{e}",
            )
        except httpx.HTTPError as e:
            return ValidationResult(
                status="unknown_error",
                message=f"HTTP 错误：{e}",
            )

        if resp.status_code in (401, 403):
            return ValidationResult(
                status="invalid_key",
                message=f"鉴权失败（HTTP {resp.status_code}）。请检查 API Key 是否正确。",
            )

        if resp.status_code == 404:
            return ValidationResult(
                status="connection_error",
                message="端点不存在（HTTP 404）。请检查 Base URL 是否正确。",
            )

        if resp.status_code >= 400:
            return ValidationResult(
                status="unknown_error",
                message=f"服务端返回错误（HTTP {resp.status_code}）：{resp.text[:200]}",
            )

        # 校验成功 —— 顺带尝试拉取可用模型列表
        models = self.fetch_models(base_url, api_key, timeout=timeout)
        return ValidationResult(
            status="valid",
            message=f"连接成功！发现 {len(models)} 个可用模型。"
            if models
            else "连接成功（但无法拉取模型列表，使用默认列表）。",
            models=models,
        )

    def fetch_models(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 10.0,
    ) -> list[str]:
        """调 {base_url}/v1/models 拉取可用模型。失败返回空列表。"""
        if not base_url or not api_key:
            return []

        url = base_url.rstrip("/") + "/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = httpx.get(url, headers=headers, timeout=timeout)
        except httpx.HTTPError:
            return []

        if resp.status_code != 200:
            return []

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            return []

        raw_models = data.get("data", [])
        if not isinstance(raw_models, list):
            return []

        ids = []
        for m in raw_models:
            if isinstance(m, dict) and "id" in m:
                ids.append(str(m["id"]))
        ids.sort()
        return ids


# =============================================================================
# 注册表
# =============================================================================

REGISTRY: dict[str, Provider] = {
    "openai": Provider(
        key="openai",
        display_name="OpenAI",
        default_base_url="https://api.openai.com/v1",
        default_models=["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"],
        supports_json_mode=True,
    ),
    "deepseek": Provider(
        key="deepseek",
        display_name="DeepSeek",
        default_base_url="https://api.deepseek.com/v1",
        default_models=["deepseek-chat", "deepseek-reasoner"],
        supports_json_mode=False,
    ),
    "kimi": Provider(
        key="kimi",
        display_name="Kimi (月之暗面)",
        default_base_url="https://api.moonshot.cn/v1",
        default_models=["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        supports_json_mode=True,
    ),
    "tongyi": Provider(
        key="tongyi",
        display_name="通义千问 (DashScope)",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_models=["qwen-max", "qwen-plus", "qwen-turbo"],
        supports_json_mode=True,
    ),
    "zhipu": Provider(
        key="zhipu",
        display_name="智谱 GLM",
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        default_models=["glm-4-plus", "glm-4-flash", "glm-4v-plus"],
        supports_json_mode=True,
    ),
    "custom": Provider(
        key="custom",
        display_name="自定义 (OpenAI 兼容)",
        default_base_url=None,
        default_models=[],
        supports_json_mode=True,
    ),
}


def get_provider(key: str) -> Provider | None:
    """按 key 查找 provider。"""
    return REGISTRY.get(key)


def list_providers() -> list[Provider]:
    """返回所有已注册的 provider（按注册表顺序）。"""
    return list(REGISTRY.values())
