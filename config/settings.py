"""应用配置单一读取器 —— 所有环境变量的权威来源。

使用 pydantic_settings.BaseSettings 集中声明、验证、提供默认值。
本模块仅读取 os.environ，不自动加载 .env（由调用方通过 python-dotenv 填充）。

使用方式:
    from config.settings import settings

    model_name = settings.llm_model
    port = settings.port
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: 未配置 API Key 时使用的占位符值（与 config/model.py 保持一致）。
_PLACEHOLDER_API_KEY = "sk-not-configured"


class AppSettings(BaseSettings):
    """应用配置 —— 从 os.environ 读取，全部带有安全默认值。"""

    model_config = SettingsConfigDict(
        env_file=None,  # 由调用方通过 python-dotenv 填充 os.environ
        populate_by_name=True,  # 允许同时用字段名和环境变量名赋值
        env_prefix="",  # 无前缀，直接使用 LLM_MODEL 等变量名
    )

    # ── 服务器 ──
    port: int = Field(
        default=8000,
        description="Chainlit 服务端口",
    )

    # ── LLM ──
    llm_model: str = Field(
        default="gpt-4o",
        description="LLM 模型名称",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="LLM API 端点（None = OpenAI 官方）",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="LLM API Key（优先级高于 OPENAI_API_KEY）",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API Key（LLM_API_KEY 未设置时回退）",
    )

    # ── LangSmith 追踪 ──
    langchain_tracing_v2: bool = Field(
        default=False,
        description="是否启用 LangSmith V2 追踪",
    )
    langchain_api_key: str = Field(
        default="",
        description="LangSmith API Key",
    )
    langchain_project: str = Field(
        default="socratic-loop",
        description="LangSmith 项目名称",
    )

    # ── 网络/重试 ──
    llm_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="LLM 调用最大重试次数（含首次调用）",
    )
    llm_retry_backoff_base: float = Field(
        default=1.0,
        ge=0.0,
        description="指数退避基数（秒）：第 n 次重试等待 base * 2^(n-1) 秒",
    )

    # ── Referee 双策略 ──
    llm_force_json_mode: bool = Field(
        default=False,
        description="强制 Referee 使用 JSON-mode（DeepSeek 等不原生支持 with_structured_output 的提供商设为 true）",
    )

    def effective_api_key(self) -> str | None:
        """返回有效的 API Key（LLM_API_KEY 优先，OPENAI_API_KEY 回退）。

        空串和占位符 "sk-not-configured" 视为未配置，返回 None。
        """
        key = self.llm_api_key or self.openai_api_key
        if not key or key == _PLACEHOLDER_API_KEY:
            return None
        return key


#: 模块级配置单例 —— 整个应用共享。
settings = AppSettings()
