"""
Socratic Loop —— Chainlit 入口。

将 LangGraph 状态机挂载到 Chainlit 对话循环，通过 interrupt() 实现人在循环。
LLM 配置由前端 Settings 面板驱动，.env 仅作初始默认值。
"""

import asyncio
from uuid import uuid4

import chainlit as cl
from chainlit.input_widget import NumberInput, Select, Slider, Switch, TextInput
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from config import providers
from config.model_manager import ModelConfig, config_from_session
from config.settings import settings
from core.graph import build_default_graph
from core.state import make_initial_state

# 全局图编译（共享单图，多用户用 thread_id 隔离）
_graph = build_default_graph(checkpointer=MemorySaver())


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _build_settings_widget(
    *,
    provider: str = "openai",
    base_url: str = "",
    api_key: str = "",
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int | None = 4096,
    json_mode: bool = False,
    model_options: list[str] | None = None,
) -> cl.ChatSettings:
    """构造 Settings widget。model_options 为 None 时使用 provider 的默认列表。"""
    provider_obj = providers.get_provider(provider)
    if model_options is None:
        model_options = provider_obj.default_models if provider_obj else ["gpt-4o"]

    # 保证当前 model 在 options 中
    if model not in model_options:
        model_options = [model] + model_options
    initial_model = model

    return cl.ChatSettings(
        [
            Select(
                id="provider",
                label="Provider",
                values=list(providers.REGISTRY.keys()),
                initial_value=provider,
            ),
            TextInput(
                id="base_url",
                label="Base URL",
                initial=base_url,
                placeholder="https://api.openai.com/v1",
            ),
            TextInput(
                id="api_key",
                label="API Key",
                initial=api_key,
                placeholder="sk-...",
                tooltip="仅保存在浏览器 session 中，不会上传到服务器",
            ),
            Select(
                id="model",
                label="Model",
                values=model_options,
                initial_value=initial_model,
            ),
            Slider(
                id="temperature",
                label="Temperature",
                min=0.0,
                max=float(provider_obj.max_temperature) if provider_obj else 2.0,
                step=0.1,
                initial=temperature,
            ),
            NumberInput(
                id="max_tokens",
                label="Max Tokens",
                initial=max_tokens if max_tokens is not None else 4096,
            ),
            Switch(
                id="json_mode",
                label="JSON Mode",
                initial=json_mode,
                tooltip="DeepSeek 等不支持 with_structured_output 的提供商请开启",
            ),
        ]
    )


# =============================================================================
# 首次启动：渲染 Settings 面板
# =============================================================================


@cl.on_chat_start
async def start() -> None:
    """初始化新对话：分配 thread_id，渲染 Settings 面板，发送欢迎消息。"""
    thread_id = str(uuid4())
    cl.user_session.set("thread_id", thread_id)

    # 从 .env 读默认值（首次启动时填充 Settings widget）
    default_base_url = settings.llm_base_url or ""
    default_api_key = settings.effective_api_key() or ""

    widget = _build_settings_widget(
        provider="openai",
        base_url=default_base_url,
        api_key=default_api_key,
        model="gpt-4o",
        temperature=0.7,
        max_tokens=4096,
        json_mode=settings.llm_force_json_mode,
    )
    await widget.send()

    welcome = (
        "👋 欢迎来到苏格拉底式学习循环。\n\n"
        "请发送你想深入探讨的**论题**（一句话观点），我将通过连续追问帮你深化理解。\n\n"
        "💡 提示：点击右上角 ⚙️ Settings 可随时切换模型、调整参数；"
        "配置后发 `/test` 可测试连接。"
    )
    await cl.Message(content=welcome).send()


# =============================================================================
# Settings 变更回调
# =============================================================================


@cl.on_settings_update
async def on_settings_update(payload: dict) -> None:
    """用户改 Settings panel 时触发：持久化到 session。"""
    provider_obj = providers.get_provider(payload.get("provider", "openai"))

    # 当 provider 切换时，自动填充其默认 base_url（仅当用户未自定义时）
    base_url = payload.get("base_url", "")
    if not base_url and provider_obj and provider_obj.default_base_url:
        base_url = provider_obj.default_base_url

    config = ModelConfig(
        provider=payload.get("provider", "openai"),
        base_url=base_url or payload.get("base_url", ""),
        api_key=payload.get("api_key", ""),
        model=payload.get("model", "gpt-4o"),
        temperature=float(payload.get("temperature", 0.7)),
        max_tokens=payload.get("max_tokens"),
        json_mode=bool(payload.get("json_mode", False)),
    )
    cl.user_session.set("model_config", config.__dict__)


# =============================================================================
# 测试连接（通过 /test 命令触发，避免依赖 Button widget）
# =============================================================================


async def _run_connection_test() -> None:
    """校验凭证 + 拉取可用模型，并更新 Settings widget。"""
    session_config = cl.user_session.get("model_config")
    if not session_config:
        await cl.Message(
            content="⚠️ 请先配置 Provider / Base URL / API Key（改 Settings 面板后自动保存）。"
        ).send()
        return

    provider_obj = providers.get_provider(session_config.get("provider", "openai"))
    if not provider_obj:
        await cl.Message(content="❌ 未知 provider。").send()
        return

    base_url = session_config.get("base_url", "")
    api_key = session_config.get("api_key", "")

    # 异步执行校验（避免阻塞事件循环）
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: provider_obj.validate(base_url, api_key)
    )

    if not result.valid:
        await cl.Message(content=f"❌ 连接失败：{result.message}").send()
        return

    # 校验成功 —— 更新 Model dropdown
    if result.models:
        cl.user_session.set("validated_models", result.models)
        # 重新发送 Settings widget 以更新 Model 下拉列表
        current_model = session_config.get("model", "")
        new_initial = (
            current_model
            if current_model in result.models
            else (result.models[0] if result.models else "")
        )

        widget = _build_settings_widget(
            provider=session_config.get("provider", "openai"),
            base_url=base_url,
            api_key=api_key,
            model=new_initial,
            temperature=float(session_config.get("temperature", 0.7)),
            max_tokens=session_config.get("max_tokens"),
            json_mode=bool(session_config.get("json_mode", False)),
            model_options=result.models,
        )
        await widget.refresh()

    await cl.Message(content=f"✅ {result.message}").send()


# =============================================================================
# 消息处理
# =============================================================================


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """处理用户消息：驱动 LangGraph 图执行。"""
    content = message.content.strip()

    # /test 命令触发连接测试
    if content == "/test":
        await _run_connection_test()
        return

    thread_id = cl.user_session.get("thread_id", str(uuid4()))
    cfg = _config(thread_id)

    # 从 session 读前端配置的 model_config
    model_config = cl.user_session.get("model_config")
    if isinstance(model_config, ModelConfig):
        model_config = model_config.__dict__

    # pending 状态 → resume；否则作为首轮 thesis 启动
    pending = cl.user_session.get("pending")
    if pending:
        graph_input: object = Command(resume=content)
    else:
        initial = make_initial_state(thesis=content)
        if model_config:
            validated = config_from_session(model_config)
            if validated is not None:
                initial["model_config"] = validated.__dict__
        graph_input = initial

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
            await cl.Message(
                content=f"🎯 **批判**\n\n{critique}", author="Opponent"
            ).send()
            await cl.Message(content="请回应上述批判（输入你的思考）：").send()
            cl.user_session.set("pending", "critique")

        elif status == "awaiting_thesis_confirmation":
            draft = state.get("_draft_thesis", "")
            await cl.Message(
                content=f"✍️ **精确化草稿**\n\n{draft}", author="Presenter"
            ).send()
            await cl.Message(
                content="请确认或编辑你的论题：",
                actions=[
                    cl.Action(
                        name="confirm", payload={"value": draft}, label="✅ 确认"
                    ),
                    cl.Action(
                        name="edit", payload={"value": "__edit__"}, label="✏️ 编辑"
                    ),
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
        await cl.Message(
            content=f"🎯 **批判（新一轮）**\n\n{critique}", author="Opponent"
        ).send()
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
    result = await loop.run_in_executor(
        None, lambda: _graph.invoke(Command(resume=draft), cfg)
    )
    await _handle_result(result)


@cl.action_callback("edit")
async def on_edit(action: cl.Action) -> None:
    """用户点击"编辑"按钮：提示用户直接输入新版本。"""
    await cl.Message(content="请直接输入你编辑后的论题：").send()
    cl.user_session.set("pending", "draft")
