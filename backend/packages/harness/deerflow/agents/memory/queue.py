"""Memory update queue with debounce mechanism."""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from deerflow.config.memory_config import get_memory_config

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Context for a conversation to be processed for memory update."""

    thread_id: str
    messages: list[Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    agent_name: str | None = None
    correction_detected: bool = False
    reinforcement_detected: bool = False


class MemoryUpdateQueue:
    """Queue for memory updates with debounce mechanism.

    This queue collects conversation contexts and processes them after
    a configurable debounce period. Multiple conversations received within
    the debounce window are batched together.
    """

    def __init__(self):
        """Initialize the memory update queue."""
        self._queue: list[ConversationContext] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._processing = False

    def add(
        self,
        thread_id: str,
        # 线程ID，标识这个对话属于哪个会话
        messages: list[Any],
        # messages：对话消息列表，包含用户输入和AI回复
        agent_name: str | None = None,
        # agent_name：可选，如果提供则记忆按agent分别存储
        correction_detected: bool = False,
        # correction_detected：用户是否纠正了AI的输出（True表示检测到修正信号）
        reinforcement_detected: bool = False,
        # reinforcement_detected：用户是否肯定了AI的输出（True表示检测到正向强化信号）
    ) -> None:
        """Add a conversation to the update queue.

        Args:
            thread_id: The thread ID.
            messages: The conversation messages.
            agent_name: If provided, memory is stored per-agent. If None, uses global memory.
            correction_detected: Whether recent turns include an explicit correction signal.
            reinforcement_detected: Whether recent turns include a positive reinforcement signal.
        """
        config = get_memory_config()
        # 从配置文件获取memory配置，检查功能是否启用

        if not config.enabled:
            # 如果memory功能被禁用（config.enabled = False）
            return
            # 直接返回，不做任何处理

        with self._lock:
            # 获取线程锁，确保多线程环境下安全操作
            # 所有后续的队列操作都在锁保护下进行

            # 查找队列中是否已存在该thread_id的待处理上下文
            # 使用next()遍历队列，找到第一个thread_id匹配的context
            existing_context = next(
                (context for context in self._queue if context.thread_id == thread_id),
                # 生成器表达式：遍历_queue中所有context，匹配thread_id
                None,
                # 如果找不到匹配的，返回None作为默认值
            )

            # 合并correction_detected信号
            # 逻辑：如果当前调用检测到True，或者之前已存在且也检测到True，则为True
            # 这是为了不丢失之前检测到的修正信号
            merged_correction_detected = correction_detected or (
                existing_context.correction_detected
                if existing_context is not None
                else False
            )

            # 合并reinforcement_detected信号，与correction同理
            merged_reinforcement_detected = reinforcement_detected or (
                existing_context.reinforcement_detected
                if existing_context is not None
                else False
            )

            # 创建新的ConversationContext对象
            context = ConversationContext(
                thread_id=thread_id,
                # 线程ID

                messages=messages,
                # 消息列表（更新为最新的对话内容）

                agent_name=agent_name,
                # Agent名称

                correction_detected=merged_correction_detected,
                # 合并后的修正标志（可能包含之前的历史信号）

                reinforcement_detected=merged_reinforcement_detected,
                # 合并后的强化标志（可能包含之前的历史信号）
            )

            # 从队列中移除该thread_id的旧上下文（如果存在）
            # 过滤掉所有thread_id等于当前thread_id的元素（即删除旧的）
            # 为什么要这样做：同一thread_id只保留最新的一次add，避免重复处理
            self._queue = [c for c in self._queue if c.thread_id != thread_id]

            # 将新创建的context添加到队列末尾
            self._queue.append(context)

            # 重置防抖定时器
            # 如果之前有定时器存在，会被取消
            # 然后启动一个新的定时器，在debounce_seconds后触发_process_queue
            # 关键效果：如果短时间内多次调用add()，只有最后一次会触发实际处理
            self._reset_timer()

        logger.info(
            "Memory update queued for thread %s, queue size: %d",
            thread_id,
            len(self._queue)
        )
        # 记录日志：线程ID和当前队列中的待处理项数量

    def _reset_timer(self) -> None:
        """Reset the debounce timer."""
        config = get_memory_config()

        # Cancel existing timer if any
        if self._timer is not None:
            self._timer.cancel()

        # Start new timer
        self._timer = threading.Timer(
            config.debounce_seconds,
            self._process_queue,
        )
        self._timer.daemon = True
        self._timer.start()

        logger.debug("Memory update timer set for %ss", config.debounce_seconds)

    def _process_queue(self) -> None:
        """Process all queued conversation contexts."""
        # Import here to avoid circular dependency
        from deerflow.agents.memory.updater import MemoryUpdater

        with self._lock:
            if self._processing:
                # Already processing, reschedule
                self._reset_timer()
                return

            if not self._queue:
                return

            self._processing = True
            contexts_to_process = self._queue.copy()
            self._queue.clear()
            self._timer = None

        logger.info("Processing %d queued memory updates", len(contexts_to_process))

        try:
            updater = MemoryUpdater()

            for context in contexts_to_process:
                try:
                    logger.info("Updating memory for thread %s", context.thread_id)
                    success = updater.update_memory(
                        messages=context.messages,
                        thread_id=context.thread_id,
                        agent_name=context.agent_name,
                        correction_detected=context.correction_detected,
                        reinforcement_detected=context.reinforcement_detected,
                    )
                    if success:
                        logger.info("Memory updated successfully for thread %s", context.thread_id)
                    else:
                        logger.warning("Memory update skipped/failed for thread %s", context.thread_id)
                except Exception as e:
                    logger.error("Error updating memory for thread %s: %s", context.thread_id, e)

                # Small delay between updates to avoid rate limiting
                if len(contexts_to_process) > 1:
                    time.sleep(0.5)

        finally:
            with self._lock:
                self._processing = False

    def flush(self) -> None:
        """Force immediate processing of the queue.

        This is useful for testing or graceful shutdown.
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

        self._process_queue()

    def clear(self) -> None:
        """Clear the queue without processing.

        This is useful for testing.
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._queue.clear()
            self._processing = False

    @property
    def pending_count(self) -> int:
        """Get the number of pending updates."""
        with self._lock:
            return len(self._queue)

    @property
    def is_processing(self) -> bool:
        """Check if the queue is currently being processed."""
        with self._lock:
            return self._processing


# Global singleton instance
_memory_queue: MemoryUpdateQueue | None = None
_queue_lock = threading.Lock()


def get_memory_queue() -> MemoryUpdateQueue:
    """Get the global memory update queue singleton.

    Returns:
        The memory update queue instance.
    """
    global _memory_queue
    with _queue_lock:
        if _memory_queue is None:
            _memory_queue = MemoryUpdateQueue()
        return _memory_queue


def reset_memory_queue() -> None:
    """Reset the global memory queue.

    This is useful for testing.
    """
    global _memory_queue
    with _queue_lock:
        if _memory_queue is not None:
            _memory_queue.clear()
        _memory_queue = None
