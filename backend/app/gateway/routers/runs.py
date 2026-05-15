"""Stateless runs endpoints -- stream and wait without a pre-existing thread.

这些端点auto-create一个临时 thread 当请求体中没有提供 thread_id 时。
当提供了 thread_id 时，会复用该线程以保留对话历史。

SSE 格式与 LangGraph Platform 协议对齐，使 @langchain/langgraph-sdk/react 的
useStream React hook 无需修改即可工作。
"""

# 类型注解前瞻引用，兼容 Python 3.9+
from __future__ import annotations

# 导入 asyncio，用于处理任务取消
import asyncio

# 导入 logging，记录日志
import logging

# 导入 uuid，用于生成线程 ID
import uuid

# FastAPI 相关导入
# APIRouter：创建路由
# Request：FastAPI 请求对象
from fastapi import APIRouter, Request

# StreamingResponse：流式响应，用于 SSE
from fastapi.responses import StreamingResponse

# 从 deps 导入依赖获取函数
# get_checkpointer：获取检查点存储
# get_run_manager：获取运行管理器
# get_stream_bridge：获取流桥接器
from app.gateway.deps import get_checkpointer, get_run_manager, get_stream_bridge

# 从 thread_runs 导入请求模型
from app.gateway.routers.thread_runs import RunCreateRequest

# 从 services 导入核心服务函数
# sse_consumer：从 bridge 消费事件并 yield SSE 帧
# start_run：创建 RunRecord 并启动后台 agent 任务
from app.gateway.services import sse_consumer, start_run

# 从 deerflow.runtime 导入序列化函数
# serialize_channel_values：将 channel values 转换为普通 dict
from deerflow.runtime import serialize_channel_values

# 获取本模块的 logger 实例
logger = logging.getLogger(__name__)

# 创建 APIRouter，前缀为 /api/runs，标签为 ["runs"]
router = APIRouter(prefix="/api/runs", tags=["runs"])


# _resolve_thread_id：从请求体中解析或生成 thread_id
#
# 参数：
#   body: RunCreateRequest，请求体
#
# 返回值：
#   str，线程 ID
#
# 说明：
#   - 如果请求的 config.configurable.thread_id 存在，使用它
#   - 否则生成一个新的 UUID 作为 thread_id
#   - 这样可以复用已有线程或创建新线程
def _resolve_thread_id(body: RunCreateRequest) -> str:
    """Return the thread_id from the request body, or generate a new one."""
    # 尝试从 request_config 中获取 thread_id
    thread_id = (body.config or {}).get("configurable", {}).get("thread_id")
    if thread_id:
        # 如果存在，转换为字符串
        return str(thread_id)
    # 不存在，生成新的 UUID
    return str(uuid.uuid4())


# stateless_stream：创建运行并通过 SSE 流式传输事件
#
# HTTP 方法：POST
# 路径：/api/runs/stream
#
# 参数：
#   body: RunCreateRequest，请求体
#   request: Request，FastAPI 请求对象
#
# 返回值：
#   StreamingResponse，SSE 流式响应
#
# 说明：
#   如果提供了 config.configurable.thread_id，运行会在该线程上创建以保留对话历史。
#   否则创建一个新的临时线程。
#
# SSE 响应头：
#   - Content-Type: text/event-stream
#   - Cache-Control: no-cache
#   - Connection: keep-alive
#   - X-Accel-Buffering: no（禁用 nginx 缓冲）
#   - Content-Location: /api/threads/{thread_id}/runs/{run_id}
#
# 调用链：
#   POST /api/runs/stream -> stateless_stream() -> start_run() -> run_agent()
#                                                        ->
#                                                  StreamingResponse(sse_consumer())
@router.post("/stream")
async def stateless_stream(body: RunCreateRequest, request: Request) -> StreamingResponse:
    """Create a run and stream events via SSE.

    If ``config.configurable.thread_id`` is provided, the run is created
    on the given thread so that conversation history is preserved.
    Otherwise a new temporary thread is created.
    """
    # 解析或生成 thread_id
    thread_id = _resolve_thread_id(body)

    # 从 app.state 获取 StreamBridge 实例
    # StreamBridge 用于在生产者和消费者之间传递事件
    bridge = get_stream_bridge(request)

    # 从 app.state 获取 RunManager 实例
    # RunManager 管理运行的创建、状态更新、取消等
    run_mgr = get_run_manager(request)

    # 创建 RunRecord 并启动后台 agent 任务
    # start_run 会创建 asyncio.create_task(run_agent(...))
    # 返回的 record 包含 run_id, thread_id, status, task 等
    record = await start_run(body, thread_id, request)

    # 返回 StreamingResponse
    # media_type="text/event-stream" 表示这是 SSE 响应
    # sse_consumer 是一个异步生成器，yield SSE 帧字符串
    # FastAPI 会迭代生成器并将内容流式发送给客户端
    return StreamingResponse(
        sse_consumer(bridge, record, request, run_mgr),
        media_type="text/event-stream",
        headers={
            # 禁用缓存，确保客户端始终获取最新事件
            "Cache-Control": "no-cache",
            # 保持连接，允许长时间流式传输
            "Connection": "keep-alive",
            # 禁用 nginx 缓冲，确保事件实时到达客户端
            "X-Accel-Buffering": "no",
            # Content-Location 头包含运行资源 URL
            # useStream React hook 使用此头来提取 run 元数据
            "Content-Location": f"/api/threads/{thread_id}/runs/{record.run_id}",
        },
    )


# stateless_wait：创建运行并阻塞直到完成
#
# HTTP 方法：POST
# 路径：/api/runs/wait
#
# 参数：
#   body: RunCreateRequest，请求体
#   request: Request，FastAPI 请求对象
#
# 返回值：
#   dict，包含最终状态或错误信息
#
# 说明：
#   如果提供了 config.configurable.thread_id，运行会在该线程上创建以保留对话历史。
#   否则创建一个新的临时线程。
#
# 与 stateless_stream 的区别：
#   - stream：立即返回 StreamingResponse，流式传输事件
#   - wait：阻塞直到 agent 完成，返回最终状态
#
# 返回值格式：
#   - 成功：serialize_channel_values(checkpoint.channel_values)
#   - 失败：{"status": "error", "error": "错误信息"}
@router.post("/wait", response_model=dict)
async def stateless_wait(body: RunCreateRequest, request: Request) -> dict:
    """Create a run and block until completion.

    If ``config.configurable.thread_id`` is provided, the run is created
    on the given thread so that conversation history is preserved.
    Otherwise a new temporary thread is created.
    """
    # 解析或生成 thread_id
    thread_id = _resolve_thread_id(body)

    # 创建 RunRecord 并启动后台 agent 任务
    # 不会立即返回，等待任务完成
    record = await start_run(body, thread_id, request)

    # 等待任务完成
    if record.task is not None:
        try:
            # await 阻塞直到任务完成
            # 如果任务被取消，抛出 CancelledError
            await record.task
        except asyncio.CancelledError:
            # 任务被取消，忽略
            pass

    # 获取 checkpointer 以读取最终状态
    checkpointer = get_checkpointer(request)

    # 构建配置查询最新 checkpoint
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # 获取最新的 checkpoint 元组
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple is not None:
            # 提取 checkpoint
            checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) or {}
            # 提取 channel_values（包含 messages, title 等）
            channel_values = checkpoint.get("channel_values", {})
            # 序列化并返回（剥离 __pregel_* 等内部 key）
            return serialize_channel_values(channel_values)
    except Exception:
        # 获取失败，记录异常
        logger.exception("Failed to fetch final state for run %s", record.run_id)

    # 如果获取失败，返回状态和错误信息
    return {"status": record.status.value, "error": record.error}