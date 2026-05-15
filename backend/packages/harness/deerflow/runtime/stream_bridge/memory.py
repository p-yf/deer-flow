"""In-memory stream bridge backed by an in-process event log.

设计说明：
  - 每个 run_id 有独立的 _RunStream，包含事件列表和 asyncio.Condition
  - 使用 asyncio.Condition 实现消费者等待/通知机制
  - 事件列表有界，溢出时删除最旧的事件并更新 start_offset
  - 支持 Last-Event-ID 断线重连：从指定 ID 之后开始消费
  - 心跳机制：在超时未收到事件时发送 HEARTBEAT_SENTINEL

使用场景：
  - Gateway 模式的进程内流式传输
  - 开发/测试环境（生产环境可能需要 Redis 跨进程）
"""

# 类型注解前瞻引用，使代码兼容 Python 3.9+
from __future__ import annotations

# 导入 asyncio，提供异步编程原语
# - asyncio.Condition：条件变量，用于消费者等待新事件
# - asyncio.wait_for：支持超时的等待
# - asyncio.sleep：延迟执行
import asyncio

# 导入 logging 模块，记录警告和错误
import logging

# 导入 time 模块，用于生成事件 ID 中的时间戳
import time

# 导入类型：AsyncIterator 用于注解异步生成器返回类型
from collections.abc import AsyncIterator

# 导入 dataclass 和 field：创建数据结构
# @dataclass 自动生成 __init__/__repr__ 等方法
# field(default_factory=...) 创建需要延迟初始化的字段
from dataclasses import dataclass, field

# 导入 Any 用于注解接受任意类型
from typing import Any

# 从当前包的 base 模块导入：
# - StreamBridge：抽象基类
# - StreamEvent：单个事件的数据类
# - HEARTBEAT_SENTINEL / END_SENTINEL：心跳和结束哨兵
from .base import END_SENTINEL, HEARTBEAT_SENTINEL, StreamBridge, StreamEvent

# 获取本模块的 logger 实例
logger = logging.getLogger(__name__)


# _RunStream：每个 run_id 的独立事件流数据结构
#
# 属性：
#   events: list[StreamEvent] - 事件列表，有界（queue_maxsize）
#   condition: asyncio.Condition - 异步条件变量，通知消费者新事件到达
#   ended: bool - 标记流是否已结束（生产者调用 publish_end）
#   start_offset: int - 事件列表的逻辑起始偏移量
#
# 设计说明：
#   - 使用偏移量而非直接删除旧事件，因为多个消费者可能处于不同位置
#   - 溢出时删除列表头部元素并递增 start_offset
#   - condition 用于生产者通知消费者新事件已到达
@dataclass
class _RunStream:
    # 事件列表，存储 StreamEvent 对象
    # 初始为空列表，由 default_factory 创建
    # 有界队列：超出容量时删除最旧的事件
    events: list[StreamEvent] = field(default_factory=list)

    # 异步条件变量，用于消费者等待和通知机制
    # 消费者调用 condition.wait() 阻塞等待新事件
    # 生产者调用 condition.notify_all() 通知消费者
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    # 标记流是否已结束，默认为 False
    # 当生产者调用 publish_end() 时设为 True
    # 消费者订阅时会检查此标志决定是否发送 END_SENTINEL
    ended: bool = False

    # 事件列表的逻辑起始偏移量
    # 当 events 溢出删除头部元素时，start_offset 随之增加
    # 用于支持多个消费者从不同位置消费同一流
    # 例如：消费者 A 处理到 offset 5，消费者 B 处理到 offset 10
    # 当 events[0:2] 被删除时，start_offset 从 0 增加到 2
    # 此时消费者 A 的逻辑位置 5 对应实际索引 5 - start_offset = 3
    start_offset: int = 0


# MemoryStreamBridge：基于内存的流桥接器实现
#
# 作用说明：
#   将事件存储在内存中的事件日志，支持多消费者和断线重连
#   每个 run_id 有独立的 _RunStream 实例
#
# 特性：
#   - 有界事件缓冲：超出 queue_maxsize 时丢弃最旧的事件
#   - 多消费者支持：通过偏移量跟踪每个消费者的位置
#   - Last-Event-ID 恢复：客户端可从指定事件 ID 之后恢复
#   - 心跳保活：在超时未收到事件时发送心跳
#
# 使用场景：
#   - Gateway 模式的进程内流式传输
#   - 开发/测试环境（生产环境可能需要 Redis 跨进程）
class MemoryStreamBridge(StreamBridge):
    """Per-run in-memory event log implementation.

    Events are retained for a bounded time window per run so late subscribers
    and reconnecting clients can replay buffered events from ``Last-Event-ID``.
    """

    # __init__：构造函数
    #
    # 参数：
    #   queue_maxsize: int，每个 run_id 的事件缓冲最大容量（默认 256）
    #
    # 说明：
    #   - _streams：存储 run_id -> _RunStream 的映射
    #   - _counters：存储 run_id -> 事件序号的映射（用于生成递增 ID）
    def __init__(self, *, queue_maxsize: int = 256) -> None:
        # 事件缓冲最大容量
        # 每个 run_id 的 events 列表最多存储此数量的事件
        # 超出时删除最旧的事件
        self._maxsize = queue_maxsize

        # run_id -> _RunStream 的字典
        # 每个 run_id 独立一个 _RunStream，实现多路复用
        self._streams: dict[str, _RunStream] = {}

        # run_id -> 事件计数器的字典（用于生成唯一递增 ID）
        # 计数器确保同一 run_id 内的事件 ID 严格递增
        # 格式为 "时间戳-序号"
        self._counters: dict[str, int] = {}

    # -- helpers ---------------------------------------------------------------

    # _get_or_create_stream：获取或创建指定 run_id 的 _RunStream
    #
    # 参数：
    #   run_id: str，运行的唯一标识符
    #
    # 返回值：
    #   _RunStream，对应 run_id 的事件流
    #
    # 说明：
    #   首次访问时创建新的 _RunStream 和计数器
    #   线程安全：由调用者保证在 async with stream.condition 中使用
    def _get_or_create_stream(self, run_id: str) -> _RunStream:
        # 检查是否已存在该 run_id 的流
        if run_id not in self._streams:
            # 创建新的 _RunStream（events 初始为空列表）
            self._streams[run_id] = _RunStream()
            # 初始化该 run_id 的事件计数器为 0
            self._counters[run_id] = 0
        return self._streams[run_id]

    # _next_id：生成唯一的递增事件 ID
    #
    # 参数：
    #   run_id: str，运行的唯一标识符
    #
    # 返回值：
    #   str，格式为 "时间戳-序号" 的唯一 ID
    #
    # 说明：
    #   - 时间戳（毫秒级）+ 序号确保全局唯一且递增
    #   - 时间戳部分支持按时间排序
    #   - 序号部分确保同一毫秒内的多个事件也有唯一 ID
    #   - ID 用于 SSE 的 id: 字段，支持 Last-Event-ID 断线重连
    def _next_id(self, run_id: str) -> str:
        # 递增计数器并获取当前值
        # 初始为 0，每次调用后加 1
        # 使用 get(run_id, 0) 处理首次访问
        self._counters[run_id] = self._counters.get(run_id, 0) + 1

        # 获取当前时间戳（毫秒级）
        # 乘以 1000 得到毫秒精度
        ts = int(time.time() * 1000)

        # 序号 = 计数器值 - 1（因为刚递增过）
        # 例如：第一次调用 counter 变成 1，序号为 0
        seq = self._counters[run_id] - 1

        # 返回 "时间戳-序号" 格式的唯一 ID
        # 例如："1747296000000-0", "1747296000000-1"
        return f"{ts}-{seq}"

    # _resolve_start_offset：根据 last_event_id 解析起始偏移量
    #
    # 参数：
    #   stream: _RunStream，事件流对象
    #   last_event_id: str | None，上次接收到的最大事件 ID
    #
    # 返回值：
    #   int，消费者应该开始消费的事件偏移量
    #
    # 说明：
    #   - 如果 last_event_id 为 None，从 stream.start_offset 开始
    #   - 如果 last_event_id 在 events 中找到，从该事件之后开始
    #   - 如果 last_event_id 找不到（已过期），从最早保留的事件开始
    #
    # 设计原因：
    #   当 events 溢出丢弃旧事件时，某些 last_event_id 可能已不在
    #   此时需要从最早保留的事件开始，并记录警告日志
    def _resolve_start_offset(self, stream: _RunStream, last_event_id: str | None) -> int:
        # 如果没有 last_event_id，从流的起始偏移开始
        # 表示从头开始消费
        if last_event_id is None:
            return stream.start_offset

        # 遍历事件列表，查找 last_event_id 对应的位置
        # 注意：这里的 enumerate 返回的是 events 中的索引
        for index, entry in enumerate(stream.events):
            if entry.id == last_event_id:
                # 找到匹配事件，返回该事件之后的位置
                # start_offset + index + 1 = 下一个要消费的事件的偏移量
                # 例如：start_offset=0, index=2 -> next_offset=3
                return stream.start_offset + index + 1

        # last_event_id 不在保留的事件中（可能已因溢出被删除）
        # 这通常发生在客户端断开连接时间过长，事件已被清理
        if stream.events:
            # 记录警告，说明客户端指定的 ID 已过期
            # 客户端需要从最早保留的事件重新开始
            logger.warning(
                "last_event_id=%s not found in retained buffer; replaying from earliest retained event",
                last_event_id,
            )
        # 从最早保留的事件开始
        # 即 start_offset 指向的位置
        return stream.start_offset

    # -- StreamBridge API ------------------------------------------------------

    # publish：发布事件到指定 run_id 的流中
    #
    # 参数：
    #   run_id: str，运行的唯一标识符
    #   event: str，事件类型名称（如 "metadata", "values", "messages"）
    #   data: Any，事件负载（JSON 可序列化）
    #
    # 实现细节：
    #   1. 获取或创建该 run_id 的 _RunStream
    #   2. 生成唯一的 StreamEvent（包含自增 ID）
    #   3. 在锁保护下将事件添加到列表
    #   4. 如果超出容量，删除最旧的事件并更新 start_offset
    #   5. 通知所有等待的消费者有新事件到达
    #
    # 生产者（worker.py 中的 run_agent）调用此方法
    async def publish(self, run_id: str, event: str, data: Any) -> None:
        # 获取或创建该 run_id 的事件流
        stream = self._get_or_create_stream(run_id)

        # 生成带唯一 ID 的事件对象
        # _next_id 生成 "时间戳-序号" 格式的 ID
        entry = StreamEvent(id=self._next_id(run_id), event=event, data=data)

        # 在条件变量锁保护下操作事件列表
        # 这样多个并发 publish 调用不会破坏 events 列表
        async with stream.condition:
            # 将新事件追加到列表尾部
            stream.events.append(entry)

            # 检查是否超出容量限制
            if len(stream.events) > self._maxsize:
                # 计算溢出数量 = 当前长度 - 最大容量
                # 例如：maxsize=256, len=258 -> overflow=2
                overflow = len(stream.events) - self._maxsize

                # 删除列表头部的 overflow 个最旧事件
                # del list[:n] 删除前 n 个元素
                del stream.events[:overflow]

                # 更新起始偏移量（逻辑删除）
                # 表示实际数据已删除，但逻辑位置前移
                # 消费者通过比较 next_offset 和 start_offset 检测落后
                stream.start_offset += overflow

            # 通知所有等待的消费者有新事件
            # 所有等待 stream.condition.wait() 的消费者都会被唤醒
            # 唤醒后它们会检查是否有新事件可消费
            stream.condition.notify_all()

    # publish_end：标记指定 run_id 的流结束
    #
    # 参数：
    #   run_id: str，运行的唯一标识符
    #
    # 说明：
    #   设置 ended=True 并通知消费者
    #   之后 subscribe() 会在消费完所有事件后返回 END_SENTINEL
    #
    # 调用时机：
    #   - worker.py 中 agent 执行完成或出错
    #   - 总是与 bridge.publish_end(run_id) 配对调用
    async def publish_end(self, run_id: str) -> None:
        # 获取或创建该 run_id 的事件流
        stream = self._get_or_create_stream(run_id)

        # 在条件变量锁保护下设置结束标记
        async with stream.condition:
            stream.ended = True

            # 通知所有等待的消费者
            # 它们会被唤醒并检查 ended 标志
            stream.condition.notify_all()

    # subscribe：订阅指定 run_id 的事件流
    #
    # 参数：
    #   run_id: str，运行的唯一标识符
    #   last_event_id: str | None，可选的断线重连 ID
    #     消费者提供上次接收到的最大事件 ID
    #     Bridge 从该 ID 之后开始推送，实现断线重连
    #   heartbeat_interval: float，心跳间隔秒数（默认 15.0）
    #     如果超过此时间没有事件到达，发送心跳保活
    #     防止 HTTP 连接因空闲被代理或负载均衡器关闭
    #
    # 返回值：
    #   AsyncIterator[StreamEvent]，异步迭代器
    #
    # 实现逻辑：
    #   1. 初始化订阅位置（根据 last_event_id 解析）
    #   2. 循环等待并消费事件：
    #      - 如果当前位置有事件，消费它并前移
    #      - 如果流已结束，yield END_SENTINEL 并返回
    #      - 如果等待超时，yield HEARTBEAT_SENTINEL 保活
    #
    # 消费者（services.py 中的 sse_consumer）调用此方法
    async def subscribe(
        self,
        run_id: str,
        *,
        last_event_id: str | None = None,
        heartbeat_interval: float = 15.0,
    ) -> AsyncIterator[StreamEvent]:
        # 获取或创建该 run_id 的事件流
        stream = self._get_or_create_stream(run_id)

        # 在条件变量锁保护下解析起始偏移量
        # 这确保在解析期间没有其他 publish 调用修改状态
        async with stream.condition:
            # 根据 last_event_id 计算初始偏移量
            # 如果 last_event_id 有效，next_offset 指向下一个要消费的事件
            # 如果 last_event_id 无效，next_offset 指向最早保留的事件
            next_offset = self._resolve_start_offset(stream, last_event_id)

        # 主循环：持续消费事件直到收到 END_SENTINEL
        while True:
            # 进入临界区，检查和消费事件
            async with stream.condition:
                # 检查是否落后于保留窗口
                # 如果 next_offset < start_offset，说明客户端错过了已删除的事件
                if next_offset < stream.start_offset:
                    # 记录警告，消费者错过了已删除的事件
                    # 这通常发生在客户端断开时间过长
                    logger.warning(
                        "subscriber for run %s fell behind retained buffer; resuming from offset %s",
                        run_id,
                        stream.start_offset,
                    )
                    # 从最早保留的事件开始
                    next_offset = stream.start_offset

                # 计算本地索引：偏移量 - 起始偏移
                # 例如：next_offset=5, start_offset=2 -> local_index=3
                # 表示要消费的事件在 events[3]
                local_index = next_offset - stream.start_offset

                # 判断是否有事件可消费
                if 0 <= local_index < len(stream.events):
                    # 当前位置有事件，取出并前移偏移量
                    entry = stream.events[local_index]
                    next_offset += 1
                elif stream.ended:
                    # 流已结束，没有更多事件
                    # 发送结束哨兵，通知消费者停止迭代
                    entry = END_SENTINEL
                else:
                    # 流未结束但当前位置无事件
                    # 需要等待新事件或超时
                    try:
                        # 等待条件变量通知或超时
                        # wait() 释放锁并阻塞，直到被 notify_all() 或超时
                        # 这允许其他 publish 调用获得锁
                        await asyncio.wait_for(stream.condition.wait(), timeout=heartbeat_interval)
                    except TimeoutError:
                        # 超时，说明 heartbeat_interval 时间内没有新事件
                        # 发送心跳保活，防止 HTTP 连接断开
                        entry = HEARTBEAT_SENTINEL
                    else:
                        # 收到通知，说明有新事件到达
                        # 继续循环检查新事件
                        continue

            # 检查是否收到结束哨兵
            if entry is END_SENTINEL:
                # yield 结束哨兵并退出生成器
                # 调用者看到 END_SENTINEL 后应停止迭代
                yield END_SENTINEL
                return

            # yield 普通事件（可能是心跳或实际数据）
            yield entry

    # cleanup：释放与 run_id 关联的资源
    #
    # 参数：
    #   run_id: str，运行的唯一标识符
    #   delay: float，延迟释放的秒数（默认 0）
    #
    # 说明：
    #   - 如果 delay > 0，先等待一段时间让订阅者消费完
    #   - 从字典中移除该 run_id 的流和计数器
    #
    # 调用时机：
    #   - worker.py 中 run_agent 结束后异步调用
    #   - 延迟 60 秒，给订阅者时间消费完剩余事件
    async def cleanup(self, run_id: str, *, delay: float = 0) -> None:
        # 如果指定了延迟，先等待
        # 这样可以让晚到的订阅者消费完剩余事件
        if delay > 0:
            await asyncio.sleep(delay)

        # 从字典中移除流和计数器（如果存在）
        # pop(key, None) 如果 key 不存在返回 None，不会抛出异常
        self._streams.pop(run_id, None)
        self._counters.pop(run_id, None)

    # close：关闭流桥接器，清理所有资源
    #
    # 说明：
    #   清理所有 run_id 的流和计数器，用于优雅关闭
    #   通常在 FastAPI 应用关闭时调用
    async def close(self) -> None:
        # 清空所有事件流
        self._streams.clear()

        # 清空所有计数器
        self._counters.clear()