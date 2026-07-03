"""config/model_manager.py 单元测试。"""

from unittest.mock import patch

from config.model_manager import ModelConfig, build_chat_model, config_from_session


# =============================================================================
# build_chat_model
# =============================================================================


class TestBuildChatModel:
    """build_chat_model() 按 ModelConfig 创建 ChatOpenAI 实例。"""

    def _build(self, **overrides) -> dict:
        """构造 ModelConfig 并调用 build_chat_model，返回传给 ChatOpenAI 的 kwargs。"""
        defaults = dict(
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=4096,
            json_mode=False,
        )
        defaults.update(overrides)
        cfg = ModelConfig(**defaults)

        with patch("config.model_manager.ChatOpenAI") as mock_cls:
            build_chat_model(cfg)
            return mock_cls.call_args.kwargs

    def test_basic_fields(self):
        kwargs = self._build()
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["temperature"] == 0.7
        assert kwargs["base_url"] == "https://api.openai.com/v1"
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["streaming"] is True

    def test_max_tokens_included_when_set(self):
        kwargs = self._build(max_tokens=2048)
        assert kwargs["max_tokens"] == 2048

    def test_max_tokens_omitted_when_none(self):
        kwargs = self._build(max_tokens=None)
        assert "max_tokens" not in kwargs

    def test_json_mode_adds_response_format(self):
        kwargs = self._build(json_mode=True)
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_no_response_format_when_json_mode_false(self):
        kwargs = self._build(json_mode=False)
        assert "response_format" not in kwargs

    def test_returns_chat_openai_instance(self):
        cfg = ModelConfig(
            provider="deepseek",
            base_url="https://api.deepseek.com/v1",
            api_key="sk-ds",
            model="deepseek-chat",
        )
        with patch("config.model_manager.ChatOpenAI") as mock_cls:
            build_chat_model(cfg)
            mock_cls.assert_called_once()


# =============================================================================
# config_from_session
# =============================================================================


class TestConfigFromSession:
    """config_from_session() 从 session dict 构造 ModelConfig。"""

    def test_complete_dict(self):
        cfg = config_from_session(
            {
                "provider": "deepseek",
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "sk-ds",
                "model": "deepseek-chat",
                "temperature": 0.5,
                "max_tokens": 2048,
                "json_mode": True,
            }
        )
        assert cfg is not None
        assert cfg.provider == "deepseek"
        assert cfg.base_url == "https://api.deepseek.com/v1"
        assert cfg.api_key == "sk-ds"
        assert cfg.model == "deepseek-chat"
        assert cfg.temperature == 0.5
        assert cfg.max_tokens == 2048
        assert cfg.json_mode is True

    def test_none_returns_none(self):
        assert config_from_session(None) is None

    def test_empty_dict_returns_none(self):
        assert config_from_session({}) is None

    def test_missing_base_url_returns_none(self):
        cfg = config_from_session(
            {
                "api_key": "sk-test",
                "model": "gpt-4o",
            }
        )
        assert cfg is None

    def test_missing_api_key_returns_none(self):
        cfg = config_from_session(
            {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o",
            }
        )
        assert cfg is None

    def test_missing_model_returns_none(self):
        cfg = config_from_session(
            {
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-test",
            }
        )
        assert cfg is None

    def test_defaults_for_optional_fields(self):
        cfg = config_from_session(
            {
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-test",
                "model": "gpt-4o",
            }
        )
        assert cfg is not None
        assert cfg.provider == "custom"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens is None
        assert cfg.json_mode is False

    def test_provider_defaults_to_custom(self):
        cfg = config_from_session(
            {
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-test",
                "model": "gpt-4o",
                "provider": "openai",
            }
        )
        assert cfg is not None
        assert cfg.provider == "openai"
