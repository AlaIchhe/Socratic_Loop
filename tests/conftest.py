"""Pytest 配置 —— 全局 fixtures。"""
import asyncio

import pytest


@pytest.fixture(autouse=True)
def _event_loop():
    """确保每个测试用例都有可用的事件循环（Windows 兼容性）。"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
