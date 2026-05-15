"""Run lifecycle service layer.

作用说明：
  集中化管理 runs 的业务逻辑，包括创建 runs、格式化 SSE 帧、
  消费 StreamBridge 事件。Router 模块（thread_runs, runs）只是
  薄薄的 HTTP 处理器，委托给这里的服务函数。

核心组件：
  - format_sse()：格式化 SSE 帧
  - normalize_stream_modes() / normalize_input()：输入规范化
  - build_run_config()：构建 RunnableConfig
  - start_run()：创建 RunRecord 并启动后台 agent 任务
  - sse_consumer()：异步生成器，从 bridge 消费事件并 yield SSE 帧
"""

# 类型注解前瞻引用，兼容 Python 3.9+
from __future__ import annotations

# 导入 asyncio，提供异步编程原语
# - asyncio.create_task：创建后台任务
# - asyncio.CancelledError：任务取消异常
import asyncio

# 导入 dataclasses，用于创建和替换 RunContext 等数据类
import dataclasses

# 导入 json，序列化 SSE data 字段
import json

# 导入 logging，记录日志
import logging

# 导入 re，正则表达式验证 assistant_id
import re

# 导入 Any，用于注解接受任意类型
from typing import Any

# FastAPI 相关导入
# HTTPException：抛出 HTTP 错误
# Request：FastAPI 请求对象，用于获取请求头等
from fastapi import HTTPException, Request

# LangChain 消息类型
# HumanMessage：用户消息类型，用于 normalize_input
from langchain_core.messages import HumanMessage

# 从 gateway.deps 导入依赖获取函数
# get_run_context / get_run_manager / get_run_store / get_stream_bridge
# 都是从 app.state 获取单例的依赖注入函数
from app.gateway.deps import get_run_context, get_run_manager, get_run_store, get_stream_bridge

# 工具函数：清理日志参数（防止日志注入）
from app.gateway.utils import sanitize_log_param

# 从 deerflow.runtime 导入运行时相关组件
# END_SENTINEL / HEARTBEAT_SENTINEL：流结束/心跳哨兵
# ConflictError / UnsupportedStrategyError：运行冲突异常
# DisconnectMode：断开连接模式枚举（cancel/continue）
# RunManager / RunRecord / RunStatus：运行管理相关
# StreamBridge：流桥接器
# run_agent：后台 agent 执行函数
from deerflow.runtime import (
    END_SENTINEL,
    HEARTBEAT_SENTINEL,
    ConflictError,
    DisconnectMode,
    RunManager,
    RunRecord,
    RunStatus,
    StreamBridge,
    UnsupportedStrategyError,
    run_agent,
)

# 获取本模块的 logger 实例
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE formatting
# ---------------------------------------------------------------------------


# format_sse：格式化单个 SSE 帧
#
# 参数：
#   event: str，SSE 事件名（如 "metadata", "values", "messages-tuple", "end", "error"）
#   data: Any，要发送的数据（JSON 可序列化）
#   event_id: str | None，可选的事件 ID（用于 Last-Event-ID 断线重连）
#
# 返回值：
#   str，格式化的 SSE 帧字符串
#
# SSE 帧格式：
#   event: <event_name>\n
#   data: <json_payload>\n
#   id: <event_id>\n
#   \n
#   （空行结束帧）
#
# 设计说明：
#   - 字段顺序：event: -> data: -> id:（可选）-> 空行
#   - 对齐 LangGraph Platform 的 wire 格式
#   - 被 useStream React hook 和 Python langgraph-sdk SSE 解码器消费
#
# 示例输出：
#   event: metadata
#   data: {"run_id": "abc123", "thread_id": "thread-456"}
#   id: 1747296000000-1
#
#
def format_sse(event: str, data: Any, *, event_id: str | None = None) -> str:
    """Format a single SSE frame.

    Field order: ``event:`` -> ``data:`` -> ``id:`` (optional) -> blank line.
    This matches the LangGraph Platform wire format consumed by the
    ``useStream`` React hook and the Python ``langgraph-sdk`` SSE decoder.
    """
    # 将数据 JSON 序列化，default=str 处理不可序列化的对象
    # ensure_ascii=False 保留 Unicode 字符（如中文）
    payload = json.dumps(data, default=str, ensure_ascii=False)

    # 构建 SSE 帧各部分
    # parts 列表最终用 "\n".join() 连接
    parts = [f"event: {event}", f"data: {payload}"]

    # 如果有 event_id，添加 id 字段
    # id 字段用于 SSE 的 Last-Event-ID 机制，客户端可据此断线重连
    if event_id:
        parts.append(f"id: {event_id}")

    # 两个空行结束 SSE 帧（标准要求）
    # SSE 协议规定帧以一个空行结束
    parts.append("")
    parts.append("")

    # 返回格式化后的 SSE 帧字符串
    # 格式：event: <name>\ndata: <json>\n\n
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Input / config helpers
# ---------------------------------------------------------------------------


# normalize_stream_modes：规范化 stream_mode 参数为列表
#
# 参数：
#   raw: list[str] | str | None，原始的 stream_mode 参数
#     可能来自请求体，可能是单个字符串或列表
#
# 返回值：
#   list[str]，规范化的 stream_mode 列表
#
# 默认值：
#   ["values"]，对应 useStream 期望的默认值
#
# 说明：
#   前端可能传单个字符串 "values" 或列表 ["values", "messages-tuple"]
#   此函数统一转换为列表格式
def normalize_stream_modes(raw: list[str] | str | None) -> list[str]:
    """Normalize the stream_mode parameter to a list.

    Default matches what ``useStream`` expects: values + messages-tuple.
    """
    # None -> 默认 ["values"]
    # 如果请求中没有指定 stream_mode，使用默认值
    if raw is None:
        return ["values"]

    # 字符串 -> 包装成列表
    # 例如："values" -> ["values"]
    if isinstance(raw, str):
        return [raw]

    # 列表 -> 直接返回（空列表也返回 ["values"]）
    # 如果传了空列表，使用默认值而不是空列表
    return raw if raw else ["values"]


# normalize_input：转换 LangGraph Platform 输入格式为 LangChain state dict
#
# 参数：
#   raw_input: dict[str, Any] | None，原始输入
#     通常是 {"messages": [{"role": "user", "content": "..."}]} 格式
#
# 返回值：
#   dict[str, Any]，转换后的输入
#     其中 messages 被转换为 LangChain HumanMessage 对象列表
#
# 说明：
#   将 {messages: [{role: "user", content: "..."}]} 格式
#   转换为 LangChain 的 HumanMessage 对象列表
#   这样 agent.astream() 才能正确处理输入
def normalize_input(raw_input: dict[str, Any] | None) -> dict[str, Any]:
    """Convert LangGraph Platform input format to LangChain state dict."""
    # None -> 空字典
    if raw_input is None:
        return {}

    # 提取 messages
    messages = raw_input.get("messages")

    # 如果有 messages 且是列表，转换每个消息
    if messages and isinstance(messages, list):
        converted = []

        # 遍历每个消息字典
        for msg in messages:
            if isinstance(msg, dict):
                # 提取 role（支持 "role" 或 "type"，LangGraph 兼容）
                # "user" 和 "human" 角色都转换为 HumanMessage
                role = msg.get("role", msg.get("type", "user"))
                content = msg.get("content", "")

                # user/human 角色转换为 HumanMessage
                if role in ("user", "human"):
                    converted.append(HumanMessage(content=content))
                else:
                    # 其他角色也当作 HumanMessage（TODO: 处理其他消息类型）
                    # 例如 "assistant" 角色也当作 HumanMessage
                    converted.append(HumanMessage(content=content))
            else:
                # 非字典，直接保留（可能是已转换的 LangChain 消息对象）
                converted.append(msg)

        # 合并转换后的 messages 到原始输入
        return {**raw_input, "messages": converted}

    # 如果没有 messages，直接返回原始输入
    return raw_input


# 默认的 assistant ID
# 当请求中没有指定 assistant_id 时使用此默认值
_DEFAULT_ASSISTANT_ID = "lead_agent"


# resolve_agent_factory：从配置解析 agent 工厂函数
#
# 参数：
#   assistant_id: str | None，assistant ID
#     用于指定使用哪个 agent 实现
#
# 返回值：
#   callable，agent 工厂函数（make_lead_agent）
#
# 设计说明：
#   自定义 agent 实现为 lead_agent + agent_name 注入到 configurable
#   所有 assistant_id 值都映射到同一工厂函数
#   路由发生在 make_lead_agent 读取 cfg["agent_name"] 时
def resolve_agent_factory(assistant_id: str | None):
    """Resolve the agent factory callable from config.

    Custom agents are implemented as ``lead_agent`` + an ``agent_name``
    injected into ``configurable`` — see :func:`build_run_config`.  All
    ``assistant_id`` values therefore map to the same factory; the routing
    happens inside ``make_lead_agent`` when it reads ``cfg["agent_name"]``.
    """
    # 延迟导入避免循环依赖
    from deerflow.agents.lead_agent.agent import make_lead_agent

    # 返回工厂函数，调用者用 config 参数创建 agent
    return make_lead_agent


# build_run_config：构建 RunnableConfig 字典
#
# 参数：
#   thread_id: str，线程 ID
#   request_config: dict[str, Any] | None，请求中的配置
#     来自 body.config，可能是 {"configurable": {...}, "recursion_limit": 100}
#   metadata: dict[str, Any] | None，元数据
#   assistant_id: str | None，assistant ID
#
# 返回值：
#   dict[str, Any]，RunnableConfig 参数字典
#
# 设计说明：
#   - 当 assistant_id 不是 "lead_agent"/None 时，注入 agent_name 到 configurable
#   - make_lead_agent 读取这个 key 加载对应的 agent 配置
#   - 与 IM channel 的 _resolve_run_params 逻辑一致
#
# LangGraph >= 0.6.0 引入了 context 作为传递线程级数据的首选方式
# 此函数优先使用 context，次选 configurable
def build_run_config(
    thread_id: str,
    request_config: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    *,
    assistant_id: str | None = None,
) -> dict[str, Any]:
    """Build a RunnableConfig dict for the agent.

    When *assistant_id* refers to a custom agent (anything other than
    ``"lead_agent"`` / ``None``), the name is forwarded as
    ``configurable["agent_name"]``.  ``make_lead_agent`` reads this key to
    load the matching ``agents/<name>/SOUL.md`` and per-agent config —
    without it the agent silently runs as the default lead agent.

    This mirrors the channel manager's ``_resolve_run_params`` logic so that
    the LangGraph Platform-compatible HTTP API and the IM channel path behave
    identically.
    """
    # 初始化基本配置
    # recursion_limit 防止无限递归，默认 100
    config: dict[str, Any] = {"recursion_limit": 100}

    if request_config:
        # LangGraph >= 0.6.0 引入 context 作为传递线程级数据的首选方式
        # 如果同时发送了 context 和 configurable，优先使用 context
        if "context" in request_config:
            if "configurable" in request_config:
                # 记录警告，同时发送了 context 和 configurable
                logger.warning(
                    "build_run_config: client sent both 'context' and 'configurable'; preferring 'context' (LangGraph >= 0.6.0). thread_id=%s, caller_configurable keys=%s",
                    thread_id,
                    list(request_config.get("configurable", {}).keys()),
                )
            # 优先使用 context
            config["context"] = request_config["context"]
        else:
            # 使用 configurable 传递 thread_id 和其他配置
            # configurable 是 LangGraph 的标准配置传递方式
            configurable = {"thread_id": thread_id}
            configurable.update(request_config.get("configurable", {}))
            config["configurable"] = configurable

        # 复制其他配置项（排除 configurable 和 context）
        # 例如 recursion_limit, callbacks 等
        for k, v in request_config.items():
            if k not in ("configurable", "context"):
                config[k] = v
    else:
        # 无 request_config，只设置 thread_id
        config["configurable"] = {"thread_id": thread_id}

    # 如果指定了非默认的 assistant_id，注入 agent_name
    # 如果 request_config 中已经有 configurable["agent_name"]，保留它
    if assistant_id and assistant_id != _DEFAULT_ASSISTANT_ID and "configurable" in config:
        if "agent_name" not in config["configurable"]:
            # 规范化 assistant_id：小写 + 替换下划线为连字符
            # 例如："My_Agent" -> "my-agent"
            normalized = assistant_id.strip().lower().replace("_", "-")

            # 验证格式：只允许字母、数字、连字符
            if not normalized or not re.fullmatch(r"[a-z0-9-]+", normalized):
                raise ValueError(f"Invalid assistant_id {assistant_id!r}: must contain only letters, digits, and hyphens after normalization.")

            # 注入 agent_name 到 configurable
            config["configurable"]["agent_name"] = normalized

    # 合并元数据
    # metadata 用于存储 run 的额外信息（如创建时间、用户信息）
    if metadata:
        config.setdefault("metadata", {}).update(metadata)

    return config


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


# start_run：创建 RunRecord 并启动后台 agent 任务
#
# 参数：
#   body: RunCreateRequest，请求体（类型 Any 避免循环导入）
#     包含 assistant_id, input, config, stream_mode 等
#   thread_id: str，目标线程 ID
#   request: Request，FastAPI 请求（用于从 app.state 获取单例）
#
# 返回值：
#   RunRecord，创建的运行记录
#     包含 run_id, thread_id, status, task 等信息
#
# 实现流程：
#   1. 获取 bridge / run_mgr / run_ctx
#   2. 解析断开连接模式
#   3. 解析 follow_up_to_run_id
#   4. 创建或拒绝运行记录
#   5. 确保线程元数据存在
#   6. 解析 agent factory 和配置
#   7. 注入 DeerFlow 特定的 context 配置
#   8. 创建 asyncio.Task 后台运行 agent
#
# 调用链：
#   router.post("/stream") -> start_run() -> asyncio.create_task(run_agent())
async def start_run(
    body: Any,
    thread_id: str,
    request: Request,
) -> RunRecord:
    """Create a RunRecord and launch the background agent task.

    Parameters
    ----------
    body : RunCreateRequest
        The validated request body (typed as Any to avoid circular import
        with the router module that defines the Pydantic model).
    thread_id : str
        Target thread.
    request : Request
        FastAPI request — used to retrieve singletons from ``app.state``.
    """
    # 从 app.state 获取单例
    # 这些是通过 FastAPI 依赖注入设置的
    bridge = get_stream_bridge(request)        # StreamBridge 实例
    run_mgr = get_run_manager(request)          # RunManager 实例
    run_ctx = get_run_context(request)          # RunContext 实例

    # 解析断开连接模式：cancel 或 continue_
    # cancel：客户端断开时 abort 后台任务
    # continue：让任务继续运行，事件被丢弃
    disconnect = DisconnectMode.cancel if body.on_disconnect == "cancel" else DisconnectMode.continue_

    # 解析 follow_up_to_run_id：优先使用请求中的，否则自动检测
    # follow_up_to_run_id 用于关联上下文中断的对话
    follow_up_to_run_id = getattr(body, "follow_up_to_run_id", None)

    # 如果请求中没有指定 follow_up_to_run_id，自动检测
    if follow_up_to_run_id is None:
        run_store = get_run_store(request)
        try:
            # 获取该线程最近一次成功运行的 run_id
            recent_runs = await run_store.list_by_thread(thread_id, limit=1)
            if recent_runs and recent_runs[0].get("status") == "success":
                follow_up_to_run_id = recent_runs[0]["run_id"]
        except Exception:
            pass  # 不阻塞运行创建

    # 如果有 follow_up_to_run_id，更新 run_ctx
    # dataclasses.replace 创建一个新的 RunContext，替换 follow_up_to_run_id
    if follow_up_to_run_id:
        run_ctx = dataclasses.replace(run_ctx, follow_up_to_run_id=follow_up_to_run_id)

    # 创建或拒绝运行（检查并发冲突）
    # 如果线程正在运行且 multitask_strategy="reject"，抛出 ConflictError
    try:
        record = await run_mgr.create_or_reject(
            thread_id,
            body.assistant_id,
            on_disconnect=disconnect,
            metadata=body.metadata or {},
            kwargs={"input": body.input, "config": body.config},
            multitask_strategy=body.multitask_strategy,
            follow_up_to_run_id=follow_up_to_run_id,
        )
    except ConflictError as exc:
        # 409 Conflict：线程正在运行且不允许并发
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except UnsupportedStrategyError as exc:
        # 501 Not Implemented：不支持的 multitask_strategy
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    # 确保线程元数据存在（即使是从未显式创建的线程也能出现在 /threads/search 中）
    # 这创建了一个最小化的 thread_meta 记录
    try:
        existing = await run_ctx.thread_meta_repo.get(thread_id)
        if existing is None:
            # 线程不存在，创建新记录
            await run_ctx.thread_meta_repo.create(
                thread_id,
                assistant_id=body.assistant_id,
                metadata=body.metadata,
            )
        else:
            # 线程已存在，更新状态为 running
            await run_ctx.thread_meta_repo.update_status(thread_id, "running")
    except Exception:
        # 非致命错误，记录警告但不阻塞
        logger.warning("Failed to upsert thread_meta for %s (non-fatal)", sanitize_log_param(thread_id))

    # 解析 agent factory
    # 根据 assistant_id 决定使用哪个 agent 工厂
    agent_factory = resolve_agent_factory(body.assistant_id)

    # 规范化输入
    # 将 {messages: [...]} 转换为 LangChain 格式
    graph_input = normalize_input(body.input)

    # 构建配置
    # 包含 thread_id, configurable, metadata 等
    config = build_run_config(thread_id, body.config, body.metadata, assistant_id=body.assistant_id)

    # 注入 DeerFlow 特定的 context 配置
    # context 字段是 langgraph-compat 层的自定义扩展，携带 agent 配置
    # 只转发 agent 相关的 key；未知 key（如 thread_id）被忽略
    context = getattr(body, "context", None)
    if context:
        # DeerFlow 支持的 context 配置键
        # 这些键会被注入到 agent 的 configurable 中
        _CONTEXT_CONFIGURABLE_KEYS = {
            "model_name",          # 指定 LLM 模型
            "mode",                # 运行模式
            "thinking_enabled",    # 是否启用思考
            "reasoning_effort",    # 推理力度
            "is_plan_mode",        # 是否启用计划模式
            "subagent_enabled",    # 是否启用子代理
            "max_concurrent_subagents",  # 最大并发子代理数
        }
        configurable = config.setdefault("configurable", {})
        for key in _CONTEXT_CONFIGURABLE_KEYS:
            if key in context:
                configurable.setdefault(key, context[key])

    # 规范化流模式
    # 确保是列表格式，默认 ["values"]
    stream_modes = normalize_stream_modes(body.stream_mode)

    # 创建后台任务运行 agent
    # asyncio.create_task 将 coroutine 调度为 Task，在后台异步执行
    # 这样 HTTP 请求可以立即返回，agent 在后台运行
    task = asyncio.create_task(
        run_agent(
            bridge,              # StreamBridge，用于发布事件
            run_mgr,             # RunManager，用于更新状态
            record,              # RunRecord，包含 run_id 等
            ctx=run_ctx,         # RunContext，包含 checkpointer 等
            agent_factory=agent_factory,    # agent 工厂函数
            graph_input=graph_input,         # 输入数据
            config=config,                   # 配置字典
            stream_modes=stream_modes,      # 流模式列表
            stream_subgraphs=body.stream_subgraphs,  # 是否流式子图
            interrupt_before=body.interrupt_before,  # 中断前的节点
            interrupt_after=body.interrupt_after,    # 中断后的节点
        )
    )

    # 将 task 关联到 RunRecord
    # 这样 cancel_run 可以取消这个 task
    record.task = task

    # Title 同步由 worker.py 的 finally 块处理
    # 它从 checkpoint 读取 title 并在运行完成后调用 thread_meta_repo.update_display_name

    return record


# sse_consumer：异步生成器，从 bridge 消费事件并 yield SSE 帧
#
# 参数：
#   bridge: StreamBridge，流桥接器
#     用于订阅事件流
#   record: RunRecord，运行记录
#     包含 run_id, on_disconnect 等信息
#   request: Request，FastAPI 请求（用于检测断开连接）
#   run_mgr: RunManager，运行管理器（用于取消操作）
#
# 返回值：
#   AsyncIterator[str]，yield SSE 帧字符串
#
# 实现逻辑：
#   1. 获取 Last-Event-ID 请求头（用于断线重连）
#   2. 订阅 bridge 的事件流
#   3. 对每种事件类型进行适当处理：
#      - HEARTBEAT_SENTINEL -> yield ":" heartbeat 帧
#      - END_SENTINEL -> yield "end" 事件并 return
#      - 普通事件 -> format_sse 并 yield
#   4. 检查客户端断开连接，超时退出
#   5. finally 块实现 on_disconnect 语义
#
# on_disconnect 语义：
#   - cancel：客户端断开时 abort 后台任务
#   - continue：让任务继续运行，事件被丢弃
#
# 调用链：
#   StreamingResponse(sse_consumer(...)) -> 迭代 yield 的 SSE 帧 -> 发送到客户端
async def sse_consumer(
    bridge: StreamBridge,
    record: RunRecord,
    request: Request,
    run_mgr: RunManager,
):
    """Async generator that yields SSE frames from the bridge.

    The ``finally`` block implements ``on_disconnect`` semantics:
    - ``cancel``: abort the background task on client disconnect.
    - ``continue``: let the task run; events are discarded.
    """
    # 从请求头获取 Last-Event-ID（断线重连用）
    # SSE 客户端断开后重连时会发送此头，包含上次收到的最大事件 ID
    last_event_id = request.headers.get("Last-Event-ID")

    try:
        # 订阅 bridge 的事件流
        # 这是一个异步迭代器，遍历所有事件直到 END_SENTINEL
        async for entry in bridge.subscribe(record.run_id, last_event_id=last_event_id):
            # 检查客户端是否已断开
            # is_disconnected() 在客户端断开时返回 True
            if await request.is_disconnected():
                # 客户端断开，退出循环
                # finally 块会根据 on_disconnect 决定是否取消任务
                break

            # 处理心跳哨兵：发送空评论保持连接
            # SSE 要求定期发送内容防止连接超时
            # 格式 ": heartbeat\n\n" 是 SSE 标准的注释格式
            if entry is HEARTBEAT_SENTINEL:
                yield ": heartbeat\n\n"
                continue

            # 处理结束哨兵：发送 end 事件并退出
            # 表示 agent 运行完成
            if entry is END_SENTINEL:
                # format_sse 返回 "event: end\ndata: null\nid: <id>\n\n"
                yield format_sse("end", None, event_id=entry.id or None)
                return

            # 处理普通事件：格式化为 SSE 帧并发送
            # entry 是 StreamEvent，包含 id, event, data
            # format_sse 将其转换为 SSE 帧格式
            yield format_sse(entry.event, entry.data, event_id=entry.id or None)

    finally:
        # 实现 on_disconnect 语义
        # 如果运行还在 pending/running 状态，且断开模式是 cancel
        if record.status in (RunStatus.pending, RunStatus.running):
            if record.on_disconnect == DisconnectMode.cancel:
                # 取消后台任务
                # 这会导致 run_agent 收到 asyncio.CancelledError
                await run_mgr.cancel(record.run_id)