"""
共享测试辅助函数 —— 消除跨测试文件的 _make_state / _make_mock_model 重复。

所有测试文件从此模块导入，而非各自定义工厂函数。
"""

from typing import cast
from unittest.mock import MagicMock

from langchain_openai import ChatOpenAI

from core.state import AgentState
from core.state import make_initial_state as make_core_initial_state


def make_state(**overrides: object) -> AgentState:  # pyright: ignore[reportArgumentType]
    """构造测试用 AgentState。

    默认状态：round=1, status="opponent_computing", 空缓存。
    所有字段均可通过 overrides 覆盖。

    Args:
        **overrides: 任意 AgentState 键值对，覆盖默认值。
    """
    defaults: AgentState = {
        "current_thesis": "人工智能应该被严格监管以确保安全性。",
        "round": 1,
        "agent_temperature": 0.7,
        "status": "opponent_computing",
        "messages": [],
        "history": [],
        "final_result": "",
        "_critique": "",
        "_user_response": "",
        "_draft_thesis": "",
        "_confirmed_thesis": "",
        "_improvement_hint": "",
        "max_rounds": 10,
        "model_config": None,
    }
    return cast(AgentState, {**defaults, **overrides})


def make_initial_state(
    thesis: str = "AI 应该被严格监管。",
    **overrides: object,  # pyright: ignore[reportArgumentType]
) -> AgentState:
    """构造测试用初始 AgentState（status="idle"）。

    用于集成测试中作为 graph.invoke() 的入口状态。

    Args:
        thesis: 初始论题文本（快捷参数，等同于 current_thesis）。
        **overrides: 任意 AgentState 键值对，覆盖默认值。
    """
    defaults = make_core_initial_state(thesis)
    return cast(AgentState, {**defaults, **overrides})


def make_mock_model(response_text: str) -> MagicMock:
    """构造 Mock ChatOpenAI，.invoke() 返回给定的 response_text。

    Args:
        response_text: mock_response.content 的值。
    """
    mock = MagicMock(spec=ChatOpenAI)
    mock_response = MagicMock()
    mock_response.content = response_text
    mock.invoke.return_value = mock_response
    return mock
