"""config.model.get_chat_model 显式参数透传测试。"""

import os
import warnings
from unittest.mock import patch

from config.model import get_chat_model


class TestGetChatModelApiKeyParam:
    """显式 api_key 参数的优先级。"""

    def test_explicit_api_key_passed_to_chatai(self):
        """显式 api_key 应透传到 ChatOpenAI 构造器。"""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("config.model.ChatOpenAI") as mock_cls,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            get_chat_model(temperature=0.7, api_key="sk-explicit")
            _, kwargs = mock_cls.call_args
            assert kwargs["api_key"] == "sk-explicit"

    def test_explicit_api_key_overrides_env(self):
        with (
            patch.dict(os.environ, {"LLM_API_KEY": "sk-env"}, clear=True),
            patch("config.model.ChatOpenAI") as mock_cls,
        ):
            get_chat_model(api_key="sk-explicit")
            _, kwargs = mock_cls.call_args
            assert kwargs["api_key"] == "sk-explicit"

    def test_empty_string_api_key_falls_back_to_env(self):
        """空串等价于未传入，回退到 env。"""
        with (
            patch.dict(os.environ, {"LLM_API_KEY": "sk-env"}, clear=True),
            patch("config.model.ChatOpenAI") as mock_cls,
        ):
            get_chat_model(api_key="")
            _, kwargs = mock_cls.call_args
            assert kwargs["api_key"] == "sk-env"

    def test_none_api_key_falls_back_to_env(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fallback"}, clear=True),
            patch("config.model.ChatOpenAI") as mock_cls,
        ):
            get_chat_model(api_key=None)
            _, kwargs = mock_cls.call_args
            assert kwargs["api_key"] == "sk-fallback"

    def test_no_key_warns_and_uses_placeholder(self):
        """既无显式 key 也无 env key 时发出 RuntimeWarning 并使用占位符。"""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("config.model.ChatOpenAI") as mock_cls,
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            get_chat_model()
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            _, kwargs = mock_cls.call_args
            assert kwargs["api_key"] == "sk-not-configured"

    def test_explicit_params_passthrough(self):
        """model_name/base_url/api_key 全部显式传入时应全部透传。"""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("config.model.ChatOpenAI") as mock_cls,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            get_chat_model(
                temperature=0.3,
                model_name="my-model",
                base_url="https://example.com/v1",
                api_key="sk-xyz",
            )
            _, kwargs = mock_cls.call_args
            assert kwargs["model"] == "my-model"
            assert kwargs["temperature"] == 0.3
            assert kwargs["base_url"] == "https://example.com/v1"
            assert kwargs["api_key"] == "sk-xyz"
            assert kwargs["streaming"] is True

    def test_default_call_signature_still_works(self):
        """仅传 temperature 单参的旧代码路径应仍能工作（向后兼容）。"""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}, clear=True),
            patch("config.model.ChatOpenAI") as mock_cls,
        ):
            get_chat_model(0.5)
            _, kwargs = mock_cls.call_args
            assert kwargs["temperature"] == 0.5
            assert kwargs["model"] == "gpt-4o"  # default
            assert kwargs["api_key"] == "sk-env"
