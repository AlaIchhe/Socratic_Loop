"""
Socratic Loop —— Chainlit 入口。

将 LangGraph 状态机挂载到 Chainlit 对话循环，通过 interrupt() 实现人在循环。
"""
import asyncio
from uuid import uuid4

import chainlit as cl
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from core.graph import build_default_graph
from core.state import make_initial_state

# 全局图编译（共享单图，多用户用 thread_id 隔离）
_graph = build_default_graph(checkpointer=MemorySaver())


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@cl.on_chat_start
async def start() -> None:
    """初始化新对话：分配 thread_id，发送欢迎消息。"""
    thread_id = str(uuid4())
    cl.user_session.set("thread_id", thread_id)

    welcome = (
        "👋 欢迎来到苏格拉底式学习循环。\n\n"
        "请发送你想深入探讨的**论题**（一句话观点），我将通过连续追问帮你深化理解。"
    )
    await cl.Message(content=welcome).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """处理用户消息：驱动 LangGraph 图执行。"""
    thread_id = cl.user_session.get("thread_id", str(uuid4()))
    cfg = _config(thread_id)
    content = message.content

    # pending 状态 → resume；否则作为首轮 thesis 启动
    pending = cl.user_session.get("pending")
    graph_input: object = Command(resume=content) if pending else make_initial_state(thesis=content)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _graph.invoke(graph_input, cfg)
        )
        if pending:
            cl.user_session.set("pending", None)
        await _handle_result(result)

    except GraphInterrupt:
        snapshot = _graph.get_state(cfg)
        state = snapshot.values
        status = state.get("status")

        if status == "awaiting_critique_response":
            critique = state.get("_critique", "")
            await cl.Message(content=f"🎯 **批判**\n\n{critique}", author="Opponent").send()
            await cl.Message(content="请回应上述批判（输入你的思考）：").send()
            cl.user_session.set("pending", "critique")

        elif status == "awaiting_thesis_confirmation":
            draft = state.get("_draft_thesis", "")
            await cl.Message(content=f"✍️ **精确化草稿**\n\n{draft}", author="Presenter").send()
            await cl.Message(
                content="请确认或编辑你的论题：",
                actions=[
                    cl.Action(name="confirm", payload={"value": draft}, label="✅ 确认"),
                    cl.Action(name="edit", payload={"value": "__edit__"}, label="✏️ 编辑"),
                ],
            ).send()
            cl.user_session.set("pending", "draft")
    except Exception as e:
        await cl.Message(content=f"❌ 错误: {str(e)}").send()


async def _handle_result(result: dict) -> None:
    """处理图运行完成的结果。"""
    status = result.get("status")
    if status == "done":
        final = result.get("final_result", "")
        await cl.Message(content=f"📜 **辩论结束**\n\n{final}", author="Referee").send()
    elif status == "awaiting_critique_response":
        critique = result.get("_critique", "")
        await cl.Message(content=f"🎯 **批判（新一轮）**\n\n{critique}", author="Opponent").send()
        await cl.Message(content="请回应上述批判：").send()
        cl.user_session.set("pending", "critique")


@cl.action_callback("confirm")
async def on_confirm(action: cl.Action) -> None:
    """用户点击"确认"按钮：用草稿值 resume。"""
    draft = action.payload.get("value", "")
    thread_id = cl.user_session.get("thread_id", str(uuid4()))
    cfg = _config(thread_id)
    cl.user_session.set("pending", None)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: _graph.invoke(Command(resume=draft), cfg))
    await _handle_result(result)


@cl.action_callback("edit")
async def on_edit(action: cl.Action) -> None:
    """用户点击"编辑"按钮：提示用户直接输入新版本。"""
    await cl.Message(content="请直接输入你编辑后的论题：").send()
    cl.user_session.set("pending", "draft")
