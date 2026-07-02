"""Agent 共享基础工具测试。"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from core.agents._base import _is_retryable, invoke_with_retry


class TestRetryClassification:
    """瞬时错误识别。"""

    def test_builtin_timeout_is_retryable(self):
        assert _is_retryable(TimeoutError("timeout")) is True

    def test_rate_limit_named_error_is_retryable(self):
        class RateLimitError(Exception):
            pass

        assert _is_retryable(RateLimitError("too many requests")) is True

    def test_value_error_is_not_retryable(self):
        assert _is_retryable(ValueError("bad input")) is False


class TestInvokeWithRetry:
    """invoke_with_retry 重试行为。"""

    def test_retries_retryable_errors_then_returns_response(self):
        invocable = MagicMock()
        expected = AIMessage(content="ok")
        invocable.invoke.side_effect = [TimeoutError("temporary"), expected]

        with patch("core.agents._base.time.sleep") as sleep:
            result = invoke_with_retry(invocable, ["message"], label="test")

        assert result is expected
        assert invocable.invoke.call_count == 2
        sleep.assert_called_once_with(1.0)

    def test_does_not_retry_non_retryable_error(self):
        invocable = MagicMock()
        invocable.invoke.side_effect = ValueError("bad request")

        with patch("core.agents._base.time.sleep") as sleep, pytest.raises(ValueError):
            invoke_with_retry(invocable, ["message"], label="test")

        assert invocable.invoke.call_count == 1
        sleep.assert_not_called()

    def test_raises_after_retry_budget_exhausted(self):
        invocable = MagicMock()
        invocable.invoke.side_effect = TimeoutError("still down")

        with patch("core.agents._base.time.sleep") as sleep, pytest.raises(TimeoutError):
            invoke_with_retry(invocable, ["message"], label="test")

        assert invocable.invoke.call_count == 3
        assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0]
