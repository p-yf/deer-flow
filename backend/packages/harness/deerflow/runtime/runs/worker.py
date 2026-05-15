"""Background agent execution.

作用说明：
  在 asyncio.Task 中运行 agent graph，将事件发布到 StreamBridge
  使用 graph.astream(stream_mode=[...]) 订阅 LangGraph 流式输出

核心设计：
  - run_agent() 是异步后台任务，执行 agent 并将事件发布到 bridge
  - 支持多种 stream_mode：values（状态快照）、messages（token 增量）、custom（自定义）
  - 事件通过 serialize() 序列化为 SSE 格式后发布

注意：
  - "events" 模式不支持，因为它需要 astream_events() + checkpoint callbacks
  - Python 公开 API 没有暴露这些内部机制
"""

# 类型注解前瞻引用，兼容 Python 3.9+
from __future__ import annotations

# 导入 asyncio，提供异步编程原语
# - asyncio.create_task：创建后台任务
# - asyncio.CancelledError：任务取消异常
import asyncio

# 导入 copy，用于深拷贝 checkpoint 快照（用于回滚）
import copy

# 导入 inspect，用于检查方法是否异步
import inspect

# 导入 logging，记录日志
import logging

# 导入 dataclass 相关
from dataclasses import dataclass, field

# 导入类型相关
from typing import TYPE_CHECKING, Any, Literal

# 仅类型检查时导入（避免循环导入）
if TYPE_CHECKING:
    from langchain_core.messages import HumanMessage

# 从 serialization 模块导入 serialize 函数
# 将 LangGraph 流式事件序列化为 SSE 兼容格式
from deerflow.runtime.serialization import serialize

# 从 stream_bridge 模块导入 StreamBridge
from deerflow.runtime.stream_bridge import StreamBridge

# 从当前包导入 RunManager 和 RunRecord
from .manager import RunManager, RunRecord

# 从 schemas 导入 RunStatus 枚举
from .schemas import RunStatus

# 获取本模块的 logger 实例
logger = logging.getLogger(__name__)

# LangGraph astream() 支持的有效 stream_mode 集合
# - values：完整状态快照
# - updates：每个节点的 writes
# - checkpoints：checkpoint 元数据
# - tasks：子任务状态
# - debug：调试信息
# - messages：LLM token 增量
# - custom：StreamWriter 自定义事件
_VALID_LG_MODES = {"values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"}


# RunContext：单次 agent run 的基础设施依赖
#
# 作用：
#   将 checkpointer、store、event_store 等依赖打包成一个对象
#   避免 run_agent() 等函数参数列表不断增长
#
# 属性：
#   checkpointer：LangGraph checkpointer 实例，用于持久化线程状态
#   store：状态存储（可选），用于跨线程共享状态
#   event_store：事件存储（可选），用于记录 human_message 等事件
#   run_events_config：token usage 跟踪配置（可选）
#   thread_meta_repo：线程元数据仓库（可选），用于更新 title 等
#   follow_up_to_run_id：跟进的前一个 run_id（可选），用于关联上下文中断的对话
@dataclass(frozen=True)
class RunContext:
    """Infrastructure dependencies for a single agent run.

    Groups checkpointer, store, and persistence-related singletons so that
    ``run_agent`` (and any future callers) receive one object instead of a
    growing list of keyword arguments.
    """

    # LangGraph checkpointer，用于持久化线程状态
    # 允许恢复中断的对话，存储 checkpoint 到持久化存储
    checkpointer: Any

    # 状态存储（可选）
    # 用于跨线程共享状态，如键值存储
    store: Any | None = field(default=None)

    # 事件存储（可选），用于记录 human_message 等事件
    # 支持事件回放和审计
    event_store: Any | None = field(default=None)

    # token usage 跟踪配置（可选）
    # 控制是否跟踪 token 使用量
    run_events_config: Any | None = field(default=None)

    # 线程元数据仓库（可选）
    # 用于更新线程的 display_name（如 title）、状态等
    thread_meta_repo: Any | None = field(default=None)

    # 跟进的前一个 run_id（可选）
    # 用于关联上下文中断的对话
    # 当用户中断后继续对话时，新的 run 会关联到上一个 run
    follow_up_to_run_id: str | None = field(default=None)


# run_agent：后台执行 agent 的异步任务
#
# 参数：
#   bridge: StreamBridge，流桥接器，用于发布事件
#   run_manager: RunManager，运行管理器，用于更新状态
#   record: RunRecord，运行记录，包含 run_id、thread_id 等
#   ctx: RunContext，上下文依赖，包含 checkpointer 等
#   agent_factory: Any，agent 工厂函数，如 make_lead_agent
#   graph_input: dict，输入图的数据，通常是 {"messages": [...]}
#   config: dict，RunnableConfig 参数字典
#   stream_modes: list[str] | None，请求的流模式列表
#   stream_subgraphs: bool，是否流式子图
#   interrupt_before: 中断前的节点列表
#   interrupt_after: 中断后的节点列表
#
# 实现流程：
#   1. 标记状态为 running
#   2. 捕获预运行 checkpoint 快照（用于回滚）
#   3. 发布 metadata 事件（run_id, thread_id）
#   4. 构建 agent（注入 runtime、callbacks）
#   5. 调用 astream() 订阅流式输出
#   6. 发布事件到 bridge
#   7. 结束时标记状态并清理资源
#
# 调用链：
#   asyncio.create_task(run_agent(...)) -> 后台运行
#     -> agent.astream() -> serialize() -> bridge.publish()
#     -> bridge.publish_end()
async def run_agent(
    bridge: StreamBridge,
    run_manager: RunManager,
    record: RunRecord,
    *,
    ctx: RunContext,
    agent_factory: Any,
    graph_input: dict,
    config: dict,
    stream_modes: list[str] | None = None,
    stream_subgraphs: bool = False,
    interrupt_before: list[str] | Literal["*"] | None = None,
    interrupt_after: list[str] | Literal["*"] | None = None,
) -> None:
    """Execute an agent in the background, publishing events to *bridge*."""

    # 从 RunContext 解包基础设施依赖
    checkpointer = ctx.checkpointer
    store = ctx.store
    event_store = ctx.event_store
    run_events_config = ctx.run_events_config
    thread_meta_repo = ctx.thread_meta_repo
    follow_up_to_run_id = ctx.follow_up_to_run_id

    # 提取 run_id 和 thread_id
    # 用于日志和事件发布
    run_id = record.run_id
    thread_id = record.thread_id

    # 将请求的流模式转换为集合
    # 用于快速检查和去重
    requested_modes: set[str] = set(stream_modes or ["values"])

    # 预运行 checkpoint ID（用于回滚）
    # 如果用户请求回滚，需要恢复到运行前的状态
    pre_run_checkpoint_id: str | None = None

    # 预运行快照（用于回滚）
    # 存储运行前的完整 checkpoint 数据
    pre_run_snapshot: dict[str, Any] | None = None

    # 快照捕获是否失败
    # 如果失败，跳过回滚操作
    snapshot_capture_failed = False

    # 初始化 RunJournal 用于事件捕获
    journal = None
    if event_store is not None:
        # 延迟导入避免循环依赖
        from deerflow.runtime.journal import RunJournal

        # 创建 RunJournal 实例
        # RunJournal 是一个 LangChain callback handler
        # 捕获 on_llm_end（token usage）和 on_chain_start/end（生命周期）
        journal = RunJournal(
            run_id=run_id,
            thread_id=thread_id,
            event_store=event_store,
            # 是否跟踪 token usage
            track_token_usage=getattr(run_events_config, "track_token_usage", True),
        )

        # 写入 human_message 事件（使用 model_dump 格式，与 checkpoint 对齐）
        # 记录用户发送的消息，用于事件回放
        human_msg = _extract_human_message(graph_input)
        if human_msg is not None:
            msg_metadata = {}
            # 如果有关联的之前 run，添加 follow_up_to_run_id
            if follow_up_to_run_id:
                msg_metadata["follow_up_to_run_id"] = follow_up_to_run_id

            # 写入事件存储
            await event_store.put(
                thread_id=thread_id,
                run_id=run_id,
                event_type="human_message",
                category="message",
                content=human_msg.model_dump(),
                metadata=msg_metadata or None,
            )
            # 设置第一条 human message 内容
            # 用于后续的 title 生成等
            content = human_msg.content
            journal.set_first_human_message(content if isinstance(content, str) else str(content))

    # 检查是否请求了 "events" 模式（不支持）
    # "events" 模式需要 astream_events() + checkpoint callbacks
    # Python 公开 API 没有暴露这些内部机制
    if "events" in requested_modes:
        logger.info(
            "Run %s: 'events' stream_mode not supported in gateway (requires astream_events + checkpoint callbacks). Skipping.",
            run_id,
        )

    try:
        # 1. 标记状态为 running
        # 这样客户端可以通过 API 查询运行状态
        await run_manager.set_status(run_id, RunStatus.running)

        # 2. 捕获预运行 checkpoint 快照（用于回滚）
        # 如果用户请求回滚，恢复到运行前的状态
        if checkpointer is not None:
            try:
                # 构建检查点查询配置
                # checkpoint_ns="" 表示查询所有命名空间
                config_for_check = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

                # 获取最新的 checkpoint 元组
                ckpt_tuple = await checkpointer.aget_tuple(config_for_check)
                if ckpt_tuple is not None:
                    # 提取 checkpoint_id
                    # 用于验证恢复后的 checkpoint
                    ckpt_config = getattr(ckpt_tuple, "config", {}).get("configurable", {})
                    pre_run_checkpoint_id = ckpt_config.get("checkpoint_id")

                    # 深拷贝快照数据（用于回滚）
                    # 避免后续修改影响原始数据
                    pre_run_snapshot = {
                        "checkpoint_ns": ckpt_config.get("checkpoint_ns", ""),
                        "checkpoint": copy.deepcopy(getattr(ckpt_tuple, "checkpoint", {})),
                        "metadata": copy.deepcopy(getattr(ckpt_tuple, "metadata", {})),
                        "pending_writes": copy.deepcopy(getattr(ckpt_tuple, "pending_writes", []) or []),
                    }
            except Exception:
                # 快照捕获失败，记录警告但继续运行
                # 后续回滚将被跳过
                snapshot_capture_failed = True
                logger.warning("Could not capture pre-run checkpoint snapshot for run %s", run_id, exc_info=True)

        # 3. 发布 metadata 事件 — useStream 需要 run_id 和 thread_id
        # 这是 SSE 流的第一条事件
        await bridge.publish(
            run_id,
            "metadata",
            {
                "run_id": run_id,
                "thread_id": thread_id,
            },
        )

        # 4. 构建 agent
        # 注入 runtime context，使 middlewares 能访问 thread_id
        from langchain_core.runnables import RunnableConfig
        from langgraph.runtime import Runtime

        # 注入 runtime context，使 middlewares 能访问 thread_id
        #（langgraph-cli 自动完成，我们需手动处理）
        runtime = Runtime(context={"thread_id": thread_id}, store=store)

        # 如果调用者已设置 context key（LangGraph >= 0.6.0 优先使用）
        # 确保 thread_id 也可用
        if "context" in config and isinstance(config["context"], dict):
            config["context"].setdefault("thread_id", thread_id)

        # 设置 __pregel_runtime（LangGraph 内部使用）
        # 这是一个内部 API，用于在节点间传递 runtime
        config.setdefault("configurable", {})["__pregel_runtime"] = runtime

        # 注入 RunJournal 作为 LangChain callback handler
        # on_llm_end 捕获 token usage；on_chain_start/end 捕获生命周期
        if journal is not None:
            config.setdefault("callbacks", []).append(journal)

        # 创建 RunnableConfig
        runnable_config = RunnableConfig(**config)

        # 通过工厂函数创建 agent
        # agent_factory 是 make_lead_agent
        # 会读取 configurable 中的配置创建对应的 agent
        agent = agent_factory(config=runnable_config)

        # 5. 附加 checkpointer 和 store
        # 这些在 agent.astream() 中自动使用
        if checkpointer is not None:
            agent.checkpointer = checkpointer
        if store is not None:
            agent.store = store

        # 6. 设置中断节点
        # 用于实现断点续传功能
        if interrupt_before:
            agent.interrupt_before_nodes = interrupt_before
        if interrupt_after:
            agent.interrupt_after_nodes = interrupt_after

        # 7. 构建 LangGraph stream_mode 列表
        #    - "events" 不是有效的 astream 模式，跳过
        #    - "messages-tuple" 映射到 LangGraph 的 "messages" 模式
        lg_modes: list[str] = []
        for m in requested_modes:
            if m == "messages-tuple":
                # 前端请求 messages-tuple，映射到 LangGraph 的 messages
                lg_modes.append("messages")
            elif m == "events":
                # 跳过 — 见上面的日志
                continue
            elif m in _VALID_LG_MODES:
                lg_modes.append(m)

        # 如果为空，默认使用 "values"
        if not lg_modes:
            lg_modes = ["values"]

        # 去重（保持顺序）
        # 防止前端传重复的模式
        seen: set[str] = set()
        deduped: list[str] = []
        for m in lg_modes:
            if m not in seen:
                seen.add(m)
                deduped.append(m)
        lg_modes = deduped

        logger.info("Run %s: streaming with modes %s (requested: %s)", run_id, lg_modes, requested_modes)

        # 8. 使用 graph.astream 流式输出
        # 根据模式数量决定使用哪种迭代方式
        if len(lg_modes) == 1 and not stream_subgraphs:
            # 单模式且无子图：astream 直接返回 chunks
            # 这种情况下 astream(stream_mode="values") 返回 chunk 而不是元组
            single_mode = lg_modes[0]

            # 迭代流式输出
            async for chunk in agent.astream(graph_input, config=runnable_config, stream_mode=single_mode):
                # 检查是否请求中止
                # 用户可以通过 cancel_run 中断运行
                if record.abort_event.is_set():
                    logger.info("Run %s abort requested — stopping", run_id)
                    break

                # 将 LangGraph 模式转换为 SSE 事件名
                # 通常相同，如 "values" -> "values", "messages" -> "messages"
                sse_event = _lg_mode_to_sse_event(single_mode)

                # 序列化 chunk 并发布到 bridge
                # serialize() 根据 mode 选择不同的序列化策略
                # bridge.publish() 是异步的，不会阻塞
                await bridge.publish(run_id, sse_event, serialize(chunk, mode=single_mode))
        else:
            # 多模式或子图：astream 返回 (mode, chunk) 元组
            # 例如同时订阅 ["values", "messages"]
            async for item in agent.astream(
                graph_input,
                config=runnable_config,
                stream_mode=lg_modes,
                subgraphs=stream_subgraphs,
            ):
                # 检查是否请求中止
                if record.abort_event.is_set():
                    logger.info("Run %s abort requested — stopping", run_id)
                    break

                # 解包流式项
                # item 是 (mode, chunk) 元组或 (ns, mode, chunk) 三元组（子图）
                mode, chunk = _unpack_stream_item(item, lg_modes, stream_subgraphs)
                if mode is None:
                    # 无法解析，跳过
                    continue

                # 转换并发布事件
                sse_event = _lg_mode_to_sse_event(mode)
                await bridge.publish(run_id, sse_event, serialize(chunk, mode=mode))

        # 9. 最终状态处理
        if record.abort_event.is_set():
            # 用户请求中止
            action = record.abort_action
            if action == "rollback":
                # 用户请求回滚，恢复到运行前的状态
                await run_manager.set_status(run_id, RunStatus.error, error="Rolled back by user")
                try:
                    await _rollback_to_pre_run_checkpoint(
                        checkpointer=checkpointer,
                        thread_id=thread_id,
                        run_id=run_id,
                        pre_run_checkpoint_id=pre_run_checkpoint_id,
                        pre_run_snapshot=pre_run_snapshot,
                        snapshot_capture_failed=snapshot_capture_failed,
                    )
                    logger.info("Run %s rolled back to pre-run checkpoint %s", run_id, pre_run_checkpoint_id)
                except Exception:
                    logger.warning("Failed to rollback checkpoint for run %s", run_id, exc_info=True)
            else:
                # 中断，不回滚
                await run_manager.set_status(run_id, RunStatus.interrupted)
        else:
            # 正常结束
            await run_manager.set_status(run_id, RunStatus.success)

    # 处理任务取消
    # 通常来自 cancel_run 或客户端断开
    except asyncio.CancelledError:
        action = record.abort_action
        if action == "rollback":
            await run_manager.set_status(run_id, RunStatus.error, error="Rolled back by user")
            try:
                await _rollback_to_pre_run_checkpoint(
                    checkpointer=checkpointer,
                    thread_id=thread_id,
                    run_id=run_id,
                    pre_run_checkpoint_id=pre_run_checkpoint_id,
                    pre_run_snapshot=pre_run_snapshot,
                    snapshot_capture_failed=snapshot_capture_failed,
                )
                logger.info("Run %s was cancelled and rolled back", run_id)
            except Exception:
                logger.warning("Run %s cancellation rollback failed", run_id, exc_info=True)
        else:
            await run_manager.set_status(run_id, RunStatus.interrupted)
            logger.info("Run %s was cancelled", run_id)

    # 处理其他异常
    except Exception as exc:
        error_msg = f"{exc}"
        logger.exception("Run %s failed: %s", run_id, error_msg)
        await run_manager.set_status(run_id, RunStatus.error, error=error_msg)

        # 发布错误事件
        # 前端可以通过监听 "error" 事件显示错误信息
        await bridge.publish(
            run_id,
            "error",
            {
                "message": error_msg,
                "name": type(exc).__name__,
            },
        )

    finally:
        # finally 块：无论成功、失败还是取消，都会执行

        # 刷新缓冲的 journal 事件并持久化完成数据
        if journal is not None:
            try:
                await journal.flush()
            except Exception:
                logger.warning("Failed to flush journal for run %s", run_id, exc_info=True)

            # 从 RunStore 获取 token usage 等完成数据并更新
            completion = journal.get_completion_data()
            await run_manager.update_run_completion(run_id, status=record.status.value, **completion)

        # 从 checkpoint 同步 title 到 threads_meta.display_name
        # TitleMiddleware 在 agent 运行过程中生成 title
        # 这里读取并同步到持久化存储
        if checkpointer is not None:
            try:
                ckpt_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
                ckpt_tuple = await checkpointer.aget_tuple(ckpt_config)
                if ckpt_tuple is not None:
                    ckpt = getattr(ckpt_tuple, "checkpoint", {}) or {}
                    title = ckpt.get("channel_values", {}).get("title")
                    if title:
                        await thread_meta_repo.update_display_name(thread_id, title)
            except Exception:
                logger.debug("Failed to sync title for thread %s (non-fatal)", thread_id)

        # 根据运行结果更新 threads_meta 状态
        try:
            final_status = "idle" if record.status == RunStatus.success else record.status.value
            await thread_meta_repo.update_status(thread_id, final_status)
        except Exception:
            logger.debug("Failed to update thread_meta status for %s (non-fatal)", thread_id)

        # 发布流结束事件
        # 通知消费者没有更多事件
        await bridge.publish_end(run_id)

        # 延迟 60 秒后清理 bridge 资源
        # 给订阅者时间消费完剩余事件
        # 使用 create_task 不阻塞，因为是后台任务
        asyncio.create_task(bridge.cleanup(run_id, delay=60))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# _call_checkpointer_method：调用 checkpointer 方法（支持异步/同步）
#
# 参数：
#   checkpointer：checkpointer 实例
#   async_name：异步方法名（如 "aget_tuple"）
#   sync_name：同步方法名（如 "get_tuple"）
#   *args, **kwargs：传递给方法的参数
#
# 设计说明：
#   某些 checkpointer 只有异步版本（如 MemoryCheckpointer）
#   某些只有同步版本（如 SqliteCheckpointer）
#   此函数自动适配，优先使用异步版本
async def _call_checkpointer_method(checkpointer: Any, async_name: str, sync_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call a checkpointer method, supporting async and sync variants."""
    # 优先尝试异步方法，再尝试同步方法
    # getattr 第三个参数是默认值，如果都不存在返回 None
    method = getattr(checkpointer, async_name, None) or getattr(checkpointer, sync_name, None)
    if method is None:
        raise AttributeError(f"Missing checkpointer method: {async_name}/{sync_name}")

    # 调用方法
    result = method(*args, **kwargs)

    # 如果结果是可等待的，await 它
    # 同步 checkpointer 返回的是普通值，不需要 await
    if inspect.isawaitable(result):
        return await result

    return result


# _rollback_to_pre_run_checkpoint：回滚到预运行 checkpoint
#
# 参数：
#   checkpointer：checkpointer 实例
#   thread_id：线程 ID
#   run_id：运行 ID
#   pre_run_checkpoint_id：预运行 checkpoint ID
#   pre_run_snapshot：预运行快照
#   snapshot_capture_failed：快照捕获是否失败
#
# 实现逻辑：
#   1. 如果没有 checkpointer，直接返回
#   2. 如果快照捕获失败，记录警告并跳过
#   3. 如果没有快照，删除线程到空状态
#   4. 否则，恢复 checkpoint 和 pending_writes
async def _rollback_to_pre_run_checkpoint(
    *,
    checkpointer: Any,
    thread_id: str,
    run_id: str,
    pre_run_checkpoint_id: str | None,
    pre_run_snapshot: dict[str, Any] | None,
    snapshot_capture_failed: bool,
) -> None:
    """Restore thread state to the checkpoint snapshot captured before run start."""
    # 无 checkpointer，无法回滚
    if checkpointer is None:
        logger.info("Run %s rollback requested but no checkpointer is configured", run_id)
        return

    # 快照捕获失败，无法回滚
    if snapshot_capture_failed:
        logger.warning("Run %s rollback skipped: pre-run checkpoint snapshot capture failed", run_id)
        return

    # 无快照，回滚到空状态（删除线程）
    if pre_run_snapshot is None:
        await _call_checkpointer_method(checkpointer, "adelete_thread", "delete_thread", thread_id)
        logger.info("Run %s rollback reset thread %s to empty state", run_id, thread_id)
        return

    # 验证快照有效性
    checkpoint_to_restore = None
    metadata_to_restore: dict[str, Any] = {}
    checkpoint_ns = ""

    # 提取 checkpoint
    checkpoint = pre_run_snapshot.get("checkpoint")
    if not isinstance(checkpoint, dict):
        logger.warning("Run %s rollback skipped: invalid pre-run checkpoint snapshot", run_id)
        return

    checkpoint_to_restore = checkpoint

    # 设置 checkpoint_id（如果快照中没有）
    if checkpoint_to_restore.get("id") is None and pre_run_checkpoint_id is not None:
        checkpoint_to_restore = {**checkpoint_to_restore, "id": pre_run_checkpoint_id}

    if checkpoint_to_restore.get("id") is None:
        logger.warning("Run %s rollback skipped: pre-run checkpoint has no checkpoint id", run_id)
        return

    # 提取 metadata 和 checkpoint_ns
    metadata = pre_run_snapshot.get("metadata", {})
    metadata_to_restore = metadata if isinstance(metadata, dict) else {}
    raw_checkpoint_ns = pre_run_snapshot.get("checkpoint_ns")
    checkpoint_ns = raw_checkpoint_ns if isinstance(raw_checkpoint_ns, str) else ""

    # 提取 channel_versions
    # channel_versions 记录每个 channel 的版本，用于增量更新
    channel_versions = checkpoint_to_restore.get("channel_versions")
    new_versions = dict(channel_versions) if isinstance(channel_versions, dict) else {}

    # 构建恢复配置
    restore_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}

    # 恢复 checkpoint
    # aput/put 将 checkpoint 写入持久化存储
    restored_config = await _call_checkpointer_method(
        checkpointer,
        "aput",
        "put",
        restore_config,
        checkpoint_to_restore,
        metadata_to_restore if isinstance(metadata_to_restore, dict) else {},
        new_versions,
    )

    # 验证恢复的配置
    if not isinstance(restored_config, dict):
        raise RuntimeError(f"Run {run_id} rollback restore returned invalid config: expected dict")

    restored_configurable = restored_config.get("configurable", {})
    if not isinstance(restored_configurable, dict):
        raise RuntimeError(f"Run {run_id} rollback restore returned invalid config payload")

    restored_checkpoint_id = restored_configurable.get("checkpoint_id")
    if not restored_checkpoint_id:
        raise RuntimeError(f"Run {run_id} rollback restore did not return checkpoint_id")

    # 恢复 pending_writes（如果有）
    # pending_writes 是还未写入 checkpoint 的临时数据
    # 如正在执行的子任务的状态
    pending_writes = pre_run_snapshot.get("pending_writes", [])
    if not pending_writes:
        return

    # 按 task_id 分组 pending writes
    # 每个 task_id 对应一个子任务的待写入数据
    writes_by_task: dict[str, list[tuple[str, Any]]] = {}
    for item in pending_writes:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise RuntimeError(f"Run {run_id} rollback failed: pending_write is not a 3-tuple: {item!r}")

        task_id, channel, value = item
        if not isinstance(channel, str):
            raise RuntimeError(f"Run {run_id} rollback failed: pending_write has non-string channel: task_id={task_id!r}, channel={channel!r}")

        writes_by_task.setdefault(str(task_id), []).append((channel, value))

    # 恢复每个 task 的 pending writes
    for task_id, writes in writes_by_task.items():
        await _call_checkpointer_method(
            checkpointer,
            "aput_writes",
            "put_writes",
            restored_config,
            writes,
            task_id=task_id,
        )


# _lg_mode_to_sse_event：将 LangGraph 内部 stream_mode 映射到 SSE 事件名
#
# 参数：
#   mode: str，LangGraph 内部模式名
#
# 返回值：
#   str，SSE 事件名
#
# 说明：
#   LangGraph 的 astream(stream_mode="messages") 产生 message tuples
#   SSE 协议中客户端显式请求时称为 "messages-tuple"
#   但 LangGraph Platform 默认的 SSE 事件名就是 "messages"
#   所以这里直接返回原始 mode
def _lg_mode_to_sse_event(mode: str) -> str:
    """Map LangGraph internal stream_mode name to SSE event name.

    LangGraph's ``astream(stream_mode="messages")`` produces message
    tuples.  The SSE protocol calls this ``messages-tuple`` when the
    client explicitly requests it, but the default SSE event name used
    by LangGraph Platform is simply ``"messages"``.
    """
    # 所有 LG 模式 1:1 映射到 SSE 事件名 — "messages" 保持 "messages"
    return mode


# _extract_human_message：从 graph_input 提取或构造 HumanMessage
#
# 参数：
#   graph_input: dict，包含 messages 的字典
#
# 返回值：
#   HumanMessage | None
#
# 设计说明：
#   用于事件记录，返回 LangChain HumanMessage 以使用 .model_dump()
#   获取与 checkpoint 对齐的序列化格式
def _extract_human_message(graph_input: dict) -> HumanMessage | None:
    """Extract or construct a HumanMessage from graph_input for event recording.

    Returns a LangChain HumanMessage so callers can use .model_dump() to get
    the checkpoint-aligned serialization format.
    """
    from langchain_core.messages import HumanMessage

    # 提取 messages
    messages = graph_input.get("messages")
    if not messages:
        return None

    # 获取最后一条消息
    # 通常用户消息在最后
    last = messages[-1] if isinstance(messages, list) else messages

    # 判断类型并构造 HumanMessage
    if isinstance(last, HumanMessage):
        return last

    if isinstance(last, str):
        return HumanMessage(content=last) if last else None

    if hasattr(last, "content"):
        content = last.content
        return HumanMessage(content=content)

    if isinstance(last, dict):
        content = last.get("content", "")
        return HumanMessage(content=content) if content else None

    return None


# _unpack_stream_item：解包多模式或子图的流式项为 (mode, chunk)
#
# 参数：
#   item: Any，流式项
#   lg_modes: list[str]，LangGraph 模式列表
#   stream_subgraphs: bool，是否流式子图
#
# 返回值：
#   tuple[str | None, Any]，(mode, chunk)
#
# 实现逻辑：
#   - 如果流式子图：item 可能是 (ns, mode, chunk) 或 (mode, chunk)
#   - 如果多模式：item 是 (mode, chunk) 元组
#   - 如果单模式：item 直接是 chunk
def _unpack_stream_item(
    item: Any,
    lg_modes: list[str],
    stream_subgraphs: bool,
) -> tuple[str | None, Any]:
    """Unpack a multi-mode or subgraph stream item into (mode, chunk).

    Returns ``(None, None)`` if the item cannot be parsed.
    """
    # 子图流式输出
    if stream_subgraphs:
        # 三元组：(namespace, mode, chunk)
        if isinstance(item, tuple) and len(item) == 3:
            _ns, mode, chunk = item
            return str(mode), chunk

        # 二元组：(mode, chunk)
        if isinstance(item, tuple) and len(item) == 2:
            mode, chunk = item
            return str(mode), chunk

        return None, None

    # 多模式流式输出：二元组 (mode, chunk)
    if isinstance(item, tuple) and len(item) == 2:
        mode, chunk = item
        return str(mode), chunk

    # 回退：单元素输出，假设是第一个模式
    return lg_modes[0] if lg_modes else None, item