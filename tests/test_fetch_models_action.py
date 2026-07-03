"""app.py 中 _fetch_models_action 的单元测试 —— 用 mock 验证各路径。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app as app_module


# =============================================================================
# 辅助
# =============================================================================


def _make_session_config(**overrides) -> dict:
    """生成一个合法的 session config dict，可覆盖字段。"""
    base = {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test",
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 4096,
        "json_mode": False,
    }
    base.update(overrides)
    return base


# =============================================================================
# 路径 1: session 无 config → 提示先配置
# =============================================================================


class TestNoSessionConfig:
    @pytest.mark.asyncio
    async def test_warns_when_no_model_config(self):
        with patch.object(app_module.cl, "user_session") as mock_session:
            mock_session.get.return_value = None  # 无 model_config

            with patch.object(app_module.cl, "Message") as mock_msg_cls:
                mock_msg = AsyncMock()
                mock_msg_cls.return_value = mock_msg

                await app_module._fetch_models_action()

                mock_msg_cls.assert_called_once()
                call_kwargs = mock_msg_cls.call_args.kwargs
                content = call_kwargs.get("content", "")
                assert "请先配置" in content


# =============================================================================
# 路径 2: base_url / api_key 缺失 → 提示填写
# =============================================================================


class TestMissingCredentials:
    @pytest.mark.asyncio
    async def test_warns_when_base_url_empty(self):
        with (
            patch.object(app_module.cl, "user_session") as mock_session,
            patch.object(app_module.providers, "get_provider") as mock_get_provider,
        ):
            mock_session.get.return_value = _make_session_config(base_url="")
            mock_get_provider.return_value = MagicMock()

            with patch.object(app_module.cl, "Message") as mock_msg_cls:
                mock_msg = AsyncMock()
                mock_msg_cls.return_value = mock_msg

                await app_module._fetch_models_action()

                content = mock_msg_cls.call_args.kwargs.get("content", "")
                assert "Base URL" in content

    @pytest.mark.asyncio
    async def test_warns_when_api_key_empty(self):
        with (
            patch.object(app_module.cl, "user_session") as mock_session,
            patch.object(app_module.providers, "get_provider") as mock_get_provider,
        ):
            mock_session.get.return_value = _make_session_config(api_key="")
            mock_get_provider.return_value = MagicMock()

            with patch.object(app_module.cl, "Message") as mock_msg_cls:
                mock_msg = AsyncMock()
                mock_msg_cls.return_value = mock_msg

                await app_module._fetch_models_action()

                content = mock_msg_cls.call_args.kwargs.get("content", "")
                assert "API Key" in content


# =============================================================================
# 路径 3: fetch_models 返回空 → 提示失败
# =============================================================================


class TestFetchEmpty:
    @pytest.mark.asyncio
    async def test_fetch_returns_empty_list(self):
        mock_provider = MagicMock()
        mock_provider.fetch_models.return_value = []

        with (
            patch.object(app_module.cl, "user_session") as mock_session,
            patch.object(app_module.providers, "get_provider", return_value=mock_provider),
            patch.object(app_module.asyncio, "get_event_loop") as mock_loop,
            patch.object(app_module.cl, "Message") as mock_msg_cls,
        ):
            mock_session.get.return_value = _make_session_config()
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=[])
            mock_msg = AsyncMock()
            mock_msg_cls.return_value = mock_msg

            await app_module._fetch_models_action()

            # 应该发送失败提示
            content = mock_msg_cls.call_args.kwargs.get("content", "")
            assert "无法拉取" in content


# =============================================================================
# 路径 4: fetch_models 返回列表 → 刷新 widget
# =============================================================================


class TestFetchSuccess:
    @pytest.mark.asyncio
    async def test_refreshes_widget_with_fetched_models(self):
        fetched = ["gpt-4o-latest", "gpt-4o", "gpt-4o-mini", "o1"]
        mock_provider = MagicMock()
        mock_provider.fetch_models.return_value = fetched

        with (
            patch.object(app_module.cl, "user_session") as mock_session,
            patch.object(app_module.providers, "get_provider", return_value=mock_provider),
            patch.object(app_module.asyncio, "get_event_loop") as mock_loop,
            patch.object(app_module, "_build_settings_widget") as mock_build,
            patch.object(app_module.cl, "Message") as mock_msg_cls,
        ):
            mock_session.get.return_value = _make_session_config(model="gpt-4o")
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=fetched)
            mock_widget = AsyncMock()
            mock_build.return_value = mock_widget
            mock_msg = AsyncMock()
            mock_msg_cls.return_value = mock_msg

            await app_module._fetch_models_action()

            # 验证 _build_settings_widget 以刷新后的 model_options 调用
            mock_build.assert_called_once()
            call_kwargs = mock_build.call_args.kwargs
            assert call_kwargs["model_options"] == fetched
            # 当前 model 保留在列表中
            assert call_kwargs["model"] == "gpt-4o"
            # 验证 widget.refresh 被调用
            mock_widget.refresh.assert_awaited_once()
            # 验证成功消息
            content = mock_msg_cls.call_args.kwargs.get("content", "")
            assert "已刷新模型列表" in content
            assert "4" in content  # 共 4 个模型

    @pytest.mark.asyncio
    async def test_current_model_not_in_fetched_list_picks_first(self):
        fetched = ["gpt-4o-latest", "gpt-4o-mini", "o1"]
        mock_provider = MagicMock()
        mock_provider.fetch_models.return_value = fetched

        with (
            patch.object(app_module.cl, "user_session") as mock_session,
            patch.object(app_module.cl, "Message") as mock_msg_cls,
            patch.object(app_module.providers, "get_provider", return_value=mock_provider),
            patch.object(app_module.asyncio, "get_event_loop") as mock_loop,
            patch.object(app_module, "_build_settings_widget") as mock_build,
        ):
            # 当前 model 不在拉取结果中
            mock_session.get.return_value = _make_session_config(model="old-model-xyz")
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=fetched)
            mock_build.return_value = AsyncMock()
            mock_msg_cls.return_value = AsyncMock()

            await app_module._fetch_models_action()

            # 应该选第一个 fetched model 作为 initial
            call_kwargs = mock_build.call_args.kwargs
            assert call_kwargs["model"] == fetched[0]


# =============================================================================
# 路径 5: action callback 正确分发
# =============================================================================


class TestActionCallback:
    @pytest.mark.asyncio
    async def test_action_callback_dispatches(self):
        mock_action = MagicMock()

        with patch.object(app_module, "_fetch_models_action", new_callable=AsyncMock) as mock_handler:
            await app_module.on_fetch_models(mock_action)
            mock_handler.assert_awaited_once()
