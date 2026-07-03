"""config/providers.py 单元测试 —— 用 mock 验证 HTTP 调用。"""

from __future__ import annotations

from unittest.mock import patch

import httpx

from config.providers import (
    REGISTRY,
    Provider,
    get_provider,
    list_providers,
)


# =============================================================================
# 注册表完整性
# =============================================================================


class TestRegistry:
    def test_openai_registered(self):
        p = REGISTRY.get("openai")
        assert p is not None
        assert p.default_base_url == "https://api.openai.com/v1"
        assert "gpt-4o" in p.default_models
        assert p.supports_json_mode is True

    def test_deepseek_registered(self):
        p = REGISTRY.get("deepseek")
        assert p is not None
        assert p.supports_json_mode is False
        assert "deepseek-chat" in p.default_models

    def test_custom_registered(self):
        p = REGISTRY.get("custom")
        assert p is not None
        assert p.default_base_url is None  # 必冥显式提供

    def test_get_provider_found(self):
        assert get_provider("openai") is not None

    def test_get_provider_missing(self):
        assert get_provider("nonexistent") is None

    def test_list_providers_nonempty(self):
        lst = list_providers()
        assert len(lst) >= 4
        assert all(isinstance(p, Provider) for p in lst)


# =============================================================================
# Provider.validate
# =============================================================================


def _mock_post_factory(status_code: int, json_body: dict | None = None):
    """生成 httpx.post 的 mock。"""

    def mock_post(url, **kwargs):
        resp = httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", url),
            content=b"",
        )
        if json_body is not None:
            resp._content = __import__("json").dumps(json_body).encode()
        return resp

    return mock_post


class TestValidate:
    def test_empty_base_url(self):
        p = REGISTRY["openai"]
        result = p.validate("", "sk-test")
        assert result.status == "connection_error"
        assert not result.valid

    def test_empty_api_key(self):
        p = REGISTRY["openai"]
        result = p.validate("https://api.openai.com/v1", "")
        assert result.status == "invalid_key"
        assert not result.valid

    @patch("config.providers.httpx.post")
    def test_401_invalid_key(self, mock_post):
        mock_post.side_effect = _mock_post_factory(401)
        p = REGISTRY["openai"]
        result = p.validate("https://api.openai.com/v1", "sk-bad")
        assert result.status == "invalid_key"
        assert not result.valid

    @patch("config.providers.httpx.post")
    def test_403_invalid_key(self, mock_post):
        mock_post.side_effect = _mock_post_factory(403)
        p = REGISTRY["openai"]
        result = p.validate("https://api.openai.com/v1", "sk-bad")
        assert result.status == "invalid_key"
        assert not result.valid

    @patch("config.providers.httpx.post")
    def test_404_connection_error(self, mock_post):
        mock_post.side_effect = _mock_post_factory(404)
        p = REGISTRY["openai"]
        result = p.validate("https://wrong.example.com/v1", "sk-test")
        assert result.status == "connection_error"

    @patch("config.providers.httpx.post")
    def test_500_unknown_error(self, mock_post):
        mock_post.side_effect = _mock_post_factory(500, {"error": "internal"})
        p = REGISTRY["openai"]
        result = p.validate("https://api.openai.com/v1", "sk-test")
        assert result.status == "unknown_error"

    @patch("config.providers.httpx.post")
    def test_timeout_connection_error(self, mock_post):
        mock_post.side_effect = httpx.TimeoutException("timed out")
        p = REGISTRY["openai"]
        result = p.validate("https://api.openai.com/v1", "sk-test")
        assert result.status == "connection_error"

    @patch("config.providers.httpx.post")
    def test_connect_error(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("refused")
        p = REGISTRY["openai"]
        result = p.validate("https://localhost:9999/v1", "sk-test")
        assert result.status == "connection_error"

    @patch("config.providers.httpx.post")
    @patch("config.providers.httpx.get")
    def test_200_valid(self, mock_get, mock_post):
        mock_post.side_effect = _mock_post_factory(200, {"choices": []})
        mock_get.side_effect = _mock_post_factory(200, {"data": []})
        p = REGISTRY["openai"]
        result = p.validate("https://api.openai.com/v1", "sk-good")
        assert result.valid
        assert result.status == "valid"

    @patch("config.providers.httpx.post")
    def test_validate_uses_first_default_model_as_probe(self, mock_post):
        mock_post.side_effect = _mock_post_factory(200, {"choices": []})
        p = REGISTRY["openai"]
        p.validate("https://api.openai.com/v1", "sk-test")
        # 验证发送的 payload 中 model 是 default_models[0]
        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs.get("json", {})
        assert payload.get("model") == p.default_models[0]
        assert payload.get("max_tokens") == 1

    @patch("config.providers.httpx.post")
    def test_validate_sends_bearer_auth(self, mock_post):
        mock_post.side_effect = _mock_post_factory(200, {"choices": []})
        p = REGISTRY["openai"]
        p.validate("https://api.openai.com/v1", "sk-test")
        call_kwargs = mock_post.call_args.kwargs
        headers = call_kwargs.get("headers", {})
        # Bearer <REDACTED> 断言（使用变量存储 token 避免字面量）
        token = "sk-test"
        expected = f"Bearer {token}"
        actual = headers.get("Authorization")
        assert actual == expected


# =============================================================================
# Provider.fetch_models
# =============================================================================


def _mock_get_factory(status_code: int, json_body):
    def mock_get(url, **kwargs):
        resp = httpx.Response(
            status_code=status_code,
            request=httpx.Request("GET", url),
        )
        resp._content = __import__("json").dumps(json_body).encode()
        return resp

    return mock_get


class TestFetchModels:
    @patch("config.providers.httpx.get")
    def test_returns_sorted_ids(self, mock_get):
        mock_get.side_effect = _mock_get_factory(
            200,
            {
                "data": [
                    {"id": "gpt-4o"},
                    {"id": "gpt-4o-mini"},
                    {"id": "o1"},
                ],
            },
        )
        p = REGISTRY["openai"]
        result = p.fetch_models("https://api.openai.com/v1", "sk-test")
        assert result == ["gpt-4o", "gpt-4o-mini", "o1"]  # sorted

    @patch("config.providers.httpx.get")
    def test_non_200_returns_empty(self, mock_get):
        mock_get.side_effect = _mock_get_factory(401, {"data": []})
        p = REGISTRY["openai"]
        assert p.fetch_models("https://api.openai.com/v1", "sk-bad") == []

    @patch("config.providers.httpx.get")
    def test_http_error_returns_empty(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        p = REGISTRY["openai"]
        assert p.fetch_models("https://localhost:9999/v1", "sk-test") == []

    def test_empty_inputs_returns_empty(self):
        p = REGISTRY["openai"]
        assert p.fetch_models("", "sk-test") == []
        assert p.fetch_models("https://api.openai.com/v1", "") == []
