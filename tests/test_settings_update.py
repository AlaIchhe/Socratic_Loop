"""app.py 中 on_settings_update 的单元测试 —— 验证「刷新模型列表」的手动触发语义。"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app as app_module


# =============================================================================
# 辅助
# =============================================================================


def _make_payload(**overrides) -> dict:
    """生成一个合法的 Settings payload，可覆盖字段。"""
    base = {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test",
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 4096,
        "json_mode": False,
        "refresh_models": False,
    }
    base.update(overrides)
    return base


@contextmanager
def _fake_context():
    """用一个 mock 替换 chainlit 模块上的 context 属性（绕过 lazy proxy）。

    app.py 内写的是 cl.context.emitter.send_toast，
    通过 patch.object(app_module.cl, "context", ctx_obj) 让 cl.context 直接返回 mock。
    因为 hasattr/cl.context access 走的是模块的 __dict__，直接覆盖即可。
    """
    ctx_obj = MagicMock()
    ctx_obj.emitter = MagicMock()
    ctx_obj.emitter.send_toast = AsyncMock()

    import chainlit

    original = chainlit.__dict__.get("context")
    chainlit.context = ctx_obj
    try:
        yield ctx_obj
    finally:
        if original is not None:
            chainlit.context = original


# =============================================================================
# 路径 1: 保存但未请求刷新 → 不拉取模型，不发 toast
# =============================================================================


class TestNoRefreshOnSave:
    @pytest.mark.asyncio
    async def test_save_without_refresh_flag_does_not_fetch(self):
        with (
            _fake_context(),  # noqa: F841
            patch.object(app_module.cl, "user_session") as mock_session,
            patch.object(app_module.providers, "get_provider") as mock_get_provider,
        ):
            mock_session.get.return_value = None
            mock_provider = MagicMock()
            mock_provider.default_models = ["gpt-4o", "gpt-4o-mini"]
            mock_provider.fetch_models.return_value = []
            mock_get_provider.return_value = mock_provider

            await app_module.on_settings_update(_make_payload(refresh_models=False))

            # 不应发起 fetch_models 调用
            mock_get_provider.return_value.fetch_models.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_save_persists_config(self):
        with (
            _fake_context(),  # noqa: F841  (ctx not asserted here)
            patch.object(app_module.cl, "user_session") as mock_session,
            patch.object(app_module.providers, "get_provider") as mock_get_provider,
        ):
            mock_provider = MagicMock()
            mock_provider.default_models = ["gpt-4o", "gpt-4o-mini"]
            mock_provider.fetch_models.return_value = []
            mock_get_provider.return_value = mock_provider

            await app_module.on_settings_update(
                _make_payload(
                    provider="openai",
                    base_url="https://api.openai.com/v1",
                    api_key="sk-new",
                    model="gpt-4o",
                    temperature=0.3,
                    refresh_models=False,
                )
            )

            saved_config = mock_session.set.call_args_list[0][0][1]
            assert saved_config["provider"] == "openai"
            assert saved_config["api_key"] == "sk-new"
            assert saved_config["temperature"] == 0.3

            # 刷新未请求 → fetch_models 从未被调用
            assert mock_get_provider.return_value.fetch_models.call_count == 0


# =============================================================================
# 路径 2: 请求刷新但凭证缺失 → 警告 toast，不拉取
# =============================================================================


class TestRefreshWithMissingCredentials:
    @pytest.mark.asyncio
    async def test_warns_when_api_key_missing(self):
        with (
            _fake_context() as ctx_obj,
            patch.object(app_module.cl, "user_session"),
            patch.object(app_module.providers, "get_provider") as mock_get_provider,
            patch.object(app_module, "_build_settings_widget") as mock_build,
        ):
            mock_provider = MagicMock()
            mock_provider.default_models = ["gpt-4o", "gpt-4o-mini"]
            mock_provider.fetch_models.return_value = []
            mock_get_provider.return_value = mock_provider
            mock_build.return_value = AsyncMock()

            await app_module.on_settings_update(
                _make_payload(api_key="", refresh_models=True)
            )

            ctx_obj.emitter.send_toast.assert_awaited_once()
            call_positional = ctx_obj.emitter.send_toast.call_args.args[0]
            call_kwargs = ctx_obj.emitter.send_toast.call_args.kwargs
            assert call_kwargs["type"] == "warning"
            assert "Base URL 和 API Key" in call_positional
            # 开关应被复位
            assert mock_build.call_args.kwargs["refresh_models"] is False

    @pytest.mark.asyncio
    async def test_warns_when_base_url_missing(self):
        with (
            _fake_context() as ctx_obj,
            patch.object(app_module.cl, "user_session"),
            patch.object(app_module.providers, "get_provider") as mock_get_provider,
            patch.object(app_module, "_build_settings_widget") as mock_build,
        ):
            mock_provider = MagicMock()
            mock_provider.default_base_url = None  # 不自动填充
            mock_get_provider.return_value = mock_provider
            mock_build.return_value = AsyncMock()

            await app_module.on_settings_update(
                _make_payload(base_url="", refresh_models=True)
            )

            call_kwargs = ctx_obj.emitter.send_toast.call_args.kwargs
            assert call_kwargs["type"] == "warning"
            assert mock_build.call_args.kwargs["refresh_models"] is False


# =============================================================================
# 路径 3: 请求刷新且凭证齐全，拉取成功 → toast + 刷新下拉 + 复位开关
# =============================================================================


class TestRefreshSuccess:
    @pytest.mark.asyncio
    async def test_fetches_and_refreshes_on_request(self):
        fetched = ["gpt-4o-latest", "gpt-4o", "gpt-4o-mini"]
        mock_provider = MagicMock()
        mock_provider.fetch_models.return_value = fetched

        with (
            _fake_context() as ctx_obj,
            patch.object(app_module.cl, "user_session"),
            patch.object(
                app_module.providers, "get_provider", return_value=mock_provider
            ),
            patch.object(app_module.asyncio, "get_event_loop") as mock_loop,
            patch.object(app_module, "_build_settings_widget") as mock_build,
        ):
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=fetched)
            mock_build.return_value = AsyncMock()

            await app_module.on_settings_update(
                _make_payload(model="gpt-4o", refresh_models=True)
            )

            # fetch_models 被调用
            mock_loop.return_value.run_in_executor.assert_awaited_once()
            # 成功 toast
            call_positional = ctx_obj.emitter.send_toast.call_args.args[0]
            call_kwargs = ctx_obj.emitter.send_toast.call_args.kwargs
            assert call_kwargs["type"] == "success"
            assert "3" in call_positional
            assert "已刷新模型列表" in call_positional
            # widget 刷新使用拉取的列表，且开关复位
            assert mock_build.call_args.kwargs["model_options"] == fetched
            assert mock_build.call_args.kwargs["refresh_models"] is False
            mock_build.return_value.refresh.assert_awaited_once()
            # 当前 model 还在列表中，保留
            assert mock_build.call_args.kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_current_model_not_in_fetched_list_picks_first(self):
        fetched = ["gpt-4o-latest", "gpt-4o-mini", "o1"]
        mock_provider = MagicMock()
        mock_provider.fetch_models.return_value = fetched

        with (
            _fake_context(),  # ctx unused but needed for active cl.context
            patch.object(app_module.cl, "user_session"),
            patch.object(
                app_module.providers, "get_provider", return_value=mock_provider
            ),
            patch.object(app_module.asyncio, "get_event_loop") as mock_loop,
            patch.object(app_module, "_build_settings_widget") as mock_build,
        ):
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=fetched)
            mock_build.return_value = AsyncMock()

            await app_module.on_settings_update(
                _make_payload(model="old-model-xyz", refresh_models=True)
            )

            assert mock_build.call_args.kwargs["model"] == fetched[0]


# =============================================================================
# 路径 4: 请求刷新但拉取失败 → 警告 toast，不更换模型列表，开关复位
# =============================================================================


class TestRefreshFailure:
    @pytest.mark.asyncio
    async def test_empty_fetch_warns_and_keeps_current_options(self):
        mock_provider = MagicMock()
        mock_provider.fetch_models.return_value = []

        with (
            _fake_context() as ctx_obj,
            patch.object(app_module.cl, "user_session"),
            patch.object(
                app_module.providers, "get_provider", return_value=mock_provider
            ),
            patch.object(app_module.asyncio, "get_event_loop") as mock_loop,
            patch.object(app_module, "_build_settings_widget") as mock_build,
        ):
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=[])
            mock_build.return_value = AsyncMock()

            await app_module.on_settings_update(
                _make_payload(model="gpt-4o", refresh_models=True)
            )

            call_positional = ctx_obj.emitter.send_toast.call_args.args[0]
            call_kwargs = ctx_obj.emitter.send_toast.call_args.kwargs
            assert call_kwargs["type"] == "warning"
            assert "无法拉取" in call_positional
            # 模型下拉列表保持原样（model_options 未传入 → 使用 provider 默认）
            assert mock_build.call_args.kwargs["model_options"] is None
            assert mock_build.call_args.kwargs["refresh_models"] is False
