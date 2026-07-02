"""config/model.py 单元测试 —— 覆盖 get_chat_model() 所有分支。

通过 Mock ChatOpenAI 和 os.environ 验证工厂函数的行为，
不依赖真实 API Key。
"""

import os
import warnings
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# 辅助函数
# =============================================================================


def _call_with_env(env_vars: dict[str, str], temperature: float = 0.7) -> dict:
    """Mock ChatOpenAI 并调用 get_chat_model()，返回传给 ChatOpenAI 的关键字参数。

    通过拦截 ChatOpenAI 构造函数的调用，验证工厂函数的参数传递行为。
    """
    with patch.dict(os.environ, env_vars, clear=True), patch("config.model.ChatOpenAI") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        from config.model import get_chat_model

        get_chat_model(temperature=temperature)
        # 返回第一次调用 ChatOpenAI 时的关键字参数
        return mock_cls.call_args.kwargs


# =============================================================================
# 配置解析测试
# =============================================================================


class TestLoadModelConfig:
    """load_model_config() 只解析配置，不构造模型。"""

    def test_loads_defaults_from_mapping(self):
        from config.model import load_model_config

        config = load_model_config({"OPENAI_API_KEY": "sk-test"})

        assert config.model_name == "gpt-4o"
        assert config.base_url is None
        assert config.api_key == "sk-test"

    def test_llm_api_key_has_priority(self):
        from config.model import load_model_config

        config = load_model_config(
            {
                "LLM_API_KEY": "sk-primary",
                "OPENAI_API_KEY": "sk-fallback",
            }
        )

        assert config.api_key == "sk-primary"

    def test_empty_strings_become_absent_values(self):
        from config.model import load_model_config

        config = load_model_config(
            {
                "LLM_MODEL": "deepseek-chat",
                "LLM_BASE_URL": "",
                "LLM_API_KEY": "",
            }
        )

        assert config.model_name == "deepseek-chat"
        assert config.base_url is None
        assert config.api_key is None

    def test_placeholder_key_is_treated_as_missing(self):
        from config.model import load_model_config

        config = load_model_config({"LLM_API_KEY": "sk-not-configured"})

        assert config.api_key is None

    def test_has_configured_api_key_rejects_placeholder(self):
        from config.model import has_configured_api_key

        assert has_configured_api_key({"LLM_API_KEY": "sk-not-configured"}) is False

    def test_has_configured_api_key_accepts_real_key(self):
        from config.model import has_configured_api_key

        assert has_configured_api_key({"LLM_API_KEY": "sk-real"}) is True


# =============================================================================
# 默认值 / 基本行为测试
# =============================================================================


class TestGetChatModelDefaults:
    """get_chat_model() 无环境变量时的默认行为。"""

    def test_returns_chat_openai_instance(self):
        """应返回 ChatOpenAI 实例（未 crash）。"""
        kwargs = _call_with_env({"OPENAI_API_KEY": "sk-test-key"})
        assert len(kwargs) > 0  # ChatOpenAI 被调用

    def test_default_model_name(self):
        """无 LLM_MODEL 时默认使用 gpt-4o。"""
        kwargs = _call_with_env({"OPENAI_API_KEY": "sk-test-key"})
        assert kwargs["model"] == "gpt-4o"

    def test_default_base_url_is_none(self):
        """无 LLM_BASE_URL 时 base_url 应为 None（OpenAI 原生）。"""
        kwargs = _call_with_env({"OPENAI_API_KEY": "sk-test-key"})
        assert kwargs["base_url"] is None


# =============================================================================
# 环境变量读取测试
# =============================================================================


class TestGetChatModelEnvVars:
    """get_chat_model() 尊重环境变量覆盖。"""

    def test_uses_llm_model_env_var(self):
        """LLM_MODEL 环境变量应被正确读取。"""
        kwargs = _call_with_env(
            {
                "LLM_MODEL": "deepseek-chat",
                "LLM_API_KEY": "sk-test",
            }
        )
        assert kwargs["model"] == "deepseek-chat"

    def test_uses_llm_base_url_env_var(self):
        """LLM_BASE_URL 环境变量应被正确读取。"""
        kwargs = _call_with_env(
            {
                "LLM_BASE_URL": "https://api.deepseek.com/v1",
                "LLM_API_KEY": "sk-test",
            }
        )
        assert kwargs["base_url"] == "https://api.deepseek.com/v1"

    def test_uses_llm_api_key_directly(self):
        """LLM_API_KEY 应被直接使用。"""
        kwargs = _call_with_env({"LLM_API_KEY": "sk-custom-key"})
        assert kwargs["api_key"] == "sk-custom-key"

    def test_falls_back_to_openai_api_key(self):
        """LLM_API_KEY 未设置时回退到 OPENAI_API_KEY。"""
        kwargs = _call_with_env({"OPENAI_API_KEY": "sk-openai-key"})
        assert kwargs["api_key"] == "sk-openai-key"

    def test_llm_api_key_priority_over_openai(self):
        """LLM_API_KEY 优先级高于 OPENAI_API_KEY。"""
        kwargs = _call_with_env(
            {
                "LLM_API_KEY": "sk-primary",
                "OPENAI_API_KEY": "sk-fallback",
            }
        )
        assert kwargs["api_key"] == "sk-primary"


# =============================================================================
# API Key 缺失处理测试
# =============================================================================


class TestGetChatModelMissingKey:
    """get_chat_model() 缺少 API Key 时的行为。"""

    def test_emits_warning_when_no_api_key(self):
        """两个 API Key 都缺失时应发出 RuntimeWarning。"""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("config.model.ChatOpenAI"),
            pytest.warns(RuntimeWarning, match="未检测到"),
        ):
            from config.model import get_chat_model

            get_chat_model()

    def test_returns_instance_even_without_key(self):
        """即使没有 API Key，也应返回 ChatOpenAI 实例（惰性失败）。"""
        with patch.dict(os.environ, {}, clear=True), patch("config.model.ChatOpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from config.model import get_chat_model

                result = get_chat_model()
            assert result is mock_instance

    def test_uses_placeholder_when_no_key(self):
        """缺失 API Key 时使用占位符 'sk-not-configured'。"""
        kwargs = _call_with_env({})
        assert kwargs["api_key"] == "sk-not-configured"


# =============================================================================
# 空字符串处理测试
# =============================================================================


class TestGetChatModelEmptyStrings:
    """get_chat_model() 空字符串 → None 的 or 逻辑。"""

    def test_empty_base_url_becomes_none(self):
        """LLM_BASE_URL='' 时 or None 应转为 None。"""
        kwargs = _call_with_env(
            {
                "LLM_BASE_URL": "",
                "LLM_API_KEY": "sk-test",
            }
        )
        assert kwargs["base_url"] is None

    def test_empty_llm_api_key_falls_back(self):
        """LLM_API_KEY='' 时应回退到 OPENAI_API_KEY。"""
        kwargs = _call_with_env(
            {
                "LLM_API_KEY": "",
                "OPENAI_API_KEY": "sk-openai-fallback",
            }
        )
        assert kwargs["api_key"] == "sk-openai-fallback"


# =============================================================================
# Temperature 参数测试
# =============================================================================


class TestGetChatModelTemperature:
    """get_chat_model() 的 temperature 参数。"""

    def test_temperature_passed_to_chat_openai(self):
        """temperature=0.7 应被传入 ChatOpenAI。"""
        kwargs = _call_with_env({"OPENAI_API_KEY": "sk-test"}, temperature=0.7)
        assert kwargs["temperature"] == 0.7

    def test_temperature_zero_accepted(self):
        """temperature=0.0（裁判场景）应正常工作。"""
        kwargs = _call_with_env({"OPENAI_API_KEY": "sk-test"}, temperature=0.0)
        assert kwargs["temperature"] == 0.0

    def test_temperature_default_is_point_seven(self):
        """默认 temperature 应为 0.7。"""
        kwargs = _call_with_env({"OPENAI_API_KEY": "sk-test"})
        assert kwargs["temperature"] == 0.7
