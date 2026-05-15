# DeerFlow 流式输出设计 - StreamBridge 抽象基类
#
# 作用说明：
#   StreamBridge 解耦 agent workers（生产者）与 SSE endpoints（消费者）
#   生产者调用 publish() 发布事件，消费者通过 subscribe() 消费事件
#
# 核心设计：
#   - 生产者侧：publish() 将事件放入队列
#   - 消费者侧：subscribe() 从队列异步读取事件，支持心跳和断线重连
#   - 每个 run_id 有独立的订阅队列，实现多路复用

# 从 __future__ 导入注解支持，允许在类型注解中使用字符串形式的类型引用
# 使代码兼容 Python 3.9+ 的类型提示写法
from __future__ import annotations

# 导入 abc 模块，提供抽象基类机制
# StreamBridge 继承 abc.ABC 成为抽象基类，其方法必须被子类实现
import abc

# 导入 AsyncIterator 类型，用于定义异步生成器返回类型
# subscribe() 方法返回一个异步迭代器，消费者从中获取事件
from collections.abc import AsyncIterator

# 导入 dataclass 装饰器，用于创建数据结构
# StreamEvent 是数据类，表示单个流事件
from dataclasses import dataclass

# 导入 Any 类型，用于注解可以接受任意类型的数据
from typing import Any


# @dataclass(frozen=True) 定义不可变数据类
# frozen=True 使实例不可修改，类似 namedtuple 的不可变特性
@dataclass(frozen=True)
class StreamEvent:
    """Single stream event.

    Attributes:
        id: Monotonically increasing event ID (used as SSE ``id:`` field,
            supports ``Last-Event-ID`` reconnection).
        event: SSE event name, e.g. ``"metadata"``, ``"updates"``,
            ``"events"``, ``"error"``, ``"end"``.
        data: JSON-serialisable payload.
    """

    # 事件 ID：单调递增的整数字符串，用于 SSE 的 id: 字段
    # 客户端可使用 Last-Event-ID 进行断线重连，从指定 ID 之后恢复接收
    # 格式为 "时间戳-序号"，如 "1747296000000-1"
    id: str

    # 事件类型：SSE 事件名称，用于区分不同类型的事件
    # 常见类型：
    #   - "metadata"：元数据事件，包含 run_id 和 thread_id
    #   - "values"：完整状态快照，包含 title、messages、artifacts 等
    #   - "messages"：LLM token 增量（messages-tuple 模式）
    #   - "updates"：每个节点的 writes
    #   - "custom"：StreamWriter 自定义事件
    #   - "error"：错误事件
    #   - "end"：流结束事件
    event: str

    # 事件负载：JSON 可序列化的任意数据
    # 序列化方式见 serialization.py
    data: Any


# 心跳哨兵：用于保持连接活跃的虚假事件
# 当消费者在 heartbeat_interval 秒内未收到任何事件时，
# StreamBridge 会发送此哨兵事件防止 HTTP 连接超时
# 前端 useStream hook 会忽略此类事件
HEARTBEAT_SENTINEL = StreamEvent(id="", event="__heartbeat__", data=None)

# 结束哨兵：标记流结束的特殊事件
# 生产者调用 publish_end() 后，消费者会收到此哨兵事件
# 收到此事件后，消费者应停止迭代
END_SENTINEL = StreamEvent(id="", event="__end__", data=None)


# StreamBridge：流桥接器的抽象基类
#
# 作用说明：
#   解耦 agent workers（生产者）与 SSE endpoints（消费者）
#   生产者调用 publish() 发布事件，消费者通过 subscribe() 消费事件
#
# 核心设计：
#   - 生产者侧：publish() 将事件放入队列
#   - 消费者侧：subscribe() 从队列异步读取事件，支持心跳和断线重连
#   - 每个 run_id 有独立的订阅队列，实现多路复用
class StreamBridge(abc.ABC):
    """Abstract base for stream bridges."""

    # publish：发布单个事件到指定 run_id 的流中
    #
    # 参数：
    #   run_id: str，运行的唯一标识符，用于多路复用多个并发的流
    #   event: str，事件类型名称（如 "metadata", "updates", "messages"）
    #   data: Any，JSON 可序列化的事件负载
    #
    # 返回值：None（异步方法）
    #
    # 生产者（agent worker）调用此方法将事件推入队列
    # 消费者通过 subscribe() 接收这些事件
    @abc.abstractmethod
    async def publish(self, run_id: str, event: str, data: Any) -> None:
        """Enqueue a single event for *run_id* (producer side)."""

    # publish_end：标记指定 run_id 的流结束
    #
    # 参数：
    #   run_id: str，运行的唯一标识符
    #
    # 调用后，消费者订阅会收到 END_SENTINEL 哨兵事件
    # 用于告知消费者该 run_id 不再会有新事件产生
    # 通常在 agent 执行完成、出错或被取消时调用
    @abc.abstractmethod
    async def publish_end(self, run_id: str) -> None:
        """Signal that no more events will be produced for *run_id*."""

    # subscribe：订阅指定 run_id 的事件流
    #
    # 参数：
    #   run_id: str，运行的唯一标识符
    #   last_event_id: str | None，可选的断线重连 ID
    #     消费者提供上次接收到的最大事件 ID，Bridge 会从该 ID 之后开始推送
    #     实现 Last-Event-ID 语义，支持 HTTP SSE 的断线重连
    #   heartbeat_interval: float，心跳间隔秒数（默认 15.0）
    #     如果超过此时间没有事件到达，Bridge 会发送 HEARTBEAT_SENTINEL
    #     防止 HTTP 连接因空闲被代理或负载均衡器关闭
    #
    # 返回值：
    #   AsyncIterator[StreamEvent]，异步迭代器，yield StreamEvent
    #
    # 使用示例：
    #   async for event in bridge.subscribe(run_id, last_event_id="1747296000000-5"):
    #       if event is END_SENTINEL:
    #           break
    #       print(event.data)
    #
    # 消费者（Gateway SSE endpoint）使用此方法获取事件
    @abc.abstractmethod
    def subscribe(
        self,
        run_id: str,
        *,
        last_event_id: str | None = None,
        heartbeat_interval: float = 15.0,
    ) -> AsyncIterator[StreamEvent]:
        """Async iterator that yields events for *run_id* (consumer side).

        Yields :data:`HEARTBEAT_SENTINEL` when no event arrives within
        *heartbeat_interval* seconds.  Yields :data:`END_SENTINEL` once
        the producer calls :meth:`publish_end`.
        """

    # cleanup：释放与 run_id 关联的资源
    #
    # 参数：
    #   run_id: str，运行的唯一标识符
    #   delay: float，延迟释放的秒数（默认 0）
    #     如果 delay > 0，实现可以等待一段时间后再释放
    #     让晚到的订阅者有机会消费完剩余事件
    #     在 worker.py 中，延迟设为 60 秒
    #
    # 注意：
    #   当没有订阅者时应该调用此方法清理资源
    #   通常在 publish_end() 后由后台任务异步调用
    @abc.abstractmethod
    async def cleanup(self, run_id: str, *, delay: float = 0) -> None:
        """Release resources associated with *run_id*.

        If *delay* > 0 the implementation should wait before releasing,
        giving late subscribers a chance to drain remaining events.
        """

    # close：关闭流桥接器，释放后端资源
    #
    # 子类可以实现此方法以优雅关闭连接池、线程等资源
    # 默认实现为空操作（no-op）
    #
    # 调用时机：
    #   - FastAPI 应用关闭时（lifespan 管理）
    #   - 用于清理所有 run_id 的资源
    async def close(self) -> None:
        """Release backend resources.  Default is a no-op."""
