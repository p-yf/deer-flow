"""Memory update queue with debounce mechanism."""

# ============================================================
# 导入标准库
# ============================================================

# logging：标准库日志模块，用于记录运行日志
import logging

# threading：标准库线程模块，用于实现防抖定时器和线程同步
import threading

# time：标准库时间模块，用于处理更新间隔
import time

# dataclasses：标准库数据类模块，用于定义 ConversationContext
from dataclasses import dataclass, field

# datetime：标准库日期时间模块
# - UTC：协调世界时
# - datetime：日期时间类
from datetime import UTC, datetime

# typing：类型注解模块
from typing import Any

# ============================================================
# 导入 DeerFlow 项目内部模块
# ============================================================

# deerflow.config.memory_config.get_memory_config：
#   获取记忆配置的函数
#   来自：本项目 packages/harness/deerflow/config/memory_config.py
from deerflow.config.memory_config import get_memory_config

# ============================================================
# 模块级变量初始化
# ============================================================

# 创建模块级 logger，用于记录队列运行日志
logger = logging.getLogger(__name__)

# MAX_RESULTS：搜索结果最大数量（此文件未使用，但保留以保持一致性）
MAX_RESULTS = 5


# ============================================================
# 数据结构定义
# ============================================================

# ConversationContext：对话上下文数据类
#
# 作用说明：
#   封装一次需要处理记忆更新的对话上下文。
#   用于在队列中存储待处理的对话。
#
# 字段说明：
#   - thread_id：线程唯一标识符
#   - messages：对话消息列表
#   - timestamp：时间戳，记录添加时间
#   - agent_name：可选的 agent 名称
#   - correction_detected：是否检测到用户纠正信号
#   - reinforcement_detected：是否检测到用户强化信号
@dataclass
class ConversationContext:
    """Context for a conversation to be processed for memory update."""

    # 线程唯一标识符
    thread_id: str
    # 对话消息列表
    messages: list[Any]
    # 时间戳，记录添加时间，默认为当前时间
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    # 可选的 agent 名称，如果提供则记忆按 agent 分别存储
    agent_name: str | None = None
    # 用户是否纠正了 AI 的输出
    correction_detected: bool = False
    # 用户是否肯定了 AI 的输出
    reinforcement_detected: bool = False


# ============================================================
# MemoryUpdateQueue 主类
# ============================================================

# MemoryUpdateQueue：记忆更新队列类
#
# 作用说明：
#   带有防抖机制的对话队列，用于批量处理记忆更新。
#   多次快速添加的更新请求会合并为一次处理。
#
# 设计考虑：
#   - 使用线程锁确保多线程安全
#   - 使用 Timer 实现防抖延迟
#   - 同一 thread_id 的多次添加会合并为最新的一次
class MemoryUpdateQueue:
    """Queue for memory updates with debounce mechanism.

    This queue collects conversation contexts and processes them after
    a configurable debounce period. Multiple conversations received within
    the debounce window are batched together.
    """

    # 构造函数
    #
    # 初始化一个空的队列和相关的线程同步原语
    def __init__(self):
        """Initialize the memory update queue."""
        # _queue：待处理的对话上下文列表
        self._queue: list[ConversationContext] = []
        # _lock：线程锁，保护队列的并发访问
        self._lock = threading.Lock()
        # _timer：防抖定时器
        self._timer: threading.Timer | None = None
        # _processing：是否正在处理队列
        self._processing = False

    # add：添加对话到更新队列
    #
    # 参数：
    #   thread_id: str，线程 ID，标识这个对话属于哪个会话
    #   messages: list[Any]，对话消息列表，包含用户输入和 AI 回复
    #   agent_name: str | None，可选，如果提供则记忆按 agent 分别存储
    #   correction_detected: bool，用户是否纠正了 AI 的输出（True 表示检测到修正信号）
    #   reinforcement_detected: bool，用户是否肯定了 AI 的输出（True 表示检测到正向强化信号）
    #
    # 作用说明：
    #   将对话上下文添加到队列，并重置防抖定时器。
    #   如果同一 thread_id 已存在，会先移除旧的条目。
    def add(
        self,
        thread_id: str,
        messages: list[Any],
        agent_name: str | None = None,
        correction_detected: bool = False,
        reinforcement_detected: bool = False,
    ) -> None:
        """Add a conversation to the update queue.

        Args:
            thread_id: The thread ID.
            messages: The conversation messages.
            agent_name: If provided, memory is stored per-agent. If None, uses global memory.
            correction_detected: Whether recent turns include an explicit correction signal.
            reinforcement_detected: Whether recent turns include a positive reinforcement signal.
        """
        # 获取记忆配置
        config = get_memory_config()

        # 如果 memory 功能被禁用，直接返回
        if not config.enabled:
            return

        with self._lock:
            # 查找队列中是否已存在该 thread_id 的待处理上下文
            # 使用 next() 遍历队列，找到第一个 thread_id 匹配的 context
            existing_context = next(
                (context for context in self._queue if context.thread_id == thread_id),
                None,
            )

            # 合并 correction_detected 信号
            # 逻辑：如果当前调用检测到 True，或者之前已存在且也检测到 True，则为 True
            merged_correction_detected = correction_detected or (
                existing_context.correction_detected
                if existing_context is not None
                else False
            )

            # 合并 reinforcement_detected 信号
            merged_reinforcement_detected = reinforcement_detected or (
                existing_context.reinforcement_detected
                if existing_context is not None
                else False
            )

            # 创建新的 ConversationContext 对象
            context = ConversationContext(
                thread_id=thread_id,
                messages=messages,
                agent_name=agent_name,
                correction_detected=merged_correction_detected,
                reinforcement_detected=merged_reinforcement_detected,
            )

            # 从队列中移除该 thread_id 的旧上下文（如果存在）
            # 同一 thread_id 只保留最新的一次 add，避免重复处理
            self._queue = [c for c in self._queue if c.thread_id != thread_id]

            # 将新创建的 context 添加到队列末尾
            self._queue.append(context)

            # 重置防抖定时器
            self._reset_timer()

        # 记录日志
        logger.info(
            "Memory update queued for thread %s, queue size: %d",
            thread_id,
            len(self._queue)
        )

    # _reset_timer：重置防抖定时器
    #
    # 作用说明：
    #   取消现有的定时器（如果有），然后启动一个新的定时器。
    #   定时器在 debounce_seconds 后触发 _process_queue。
    #
    # 设计考虑：
    #   关键效果：如果短时间内多次调用 add()，只有最后一次会触发实际处理
    def _reset_timer(self) -> None:
        """Reset the debounce timer."""
        # 获取记忆配置
        config = get_memory_config()

        # 取消现有的定时器（如果有）
        if self._timer is not None:
            self._timer.cancel()

        # 启动新的定时器
        self._timer = threading.Timer(
            config.debounce_seconds,  # 防抖等待秒数
            self._process_queue,        # 超时后调用的处理函数
        )
        # 设置为守护线程，这样主程序退出时定时器也会被清除
        self._timer.daemon = True
        # 启动定时器
        self._timer.start()

        # 记录调试日志
        logger.debug("Memory update timer set for %ss", config.debounce_seconds)

    # _process_queue：处理队列中的所有对话上下文
    #
    # 作用说明：
    #   在防抖延迟后被调用，处理队列中所有待处理的对话。
    #   为每个对话调用 MemoryUpdater 进行实际的记忆更新。
    def _process_queue(self) -> None:
        """Process all queued conversation contexts."""
        # 延迟导入，避免循环依赖
        # MemoryUpdater 在 deerflow.agents.memory.updater 中定义
        from deerflow.agents.memory.updater import MemoryUpdater

        with self._lock:
            # 如果正在处理中，重新调度
            if self._processing:
                self._reset_timer()
                return

            # 如果队列为空，直接返回
            if not self._queue:
                return

            # 标记为正在处理
            self._processing = True
            # 复制队列并清空原队列
            contexts_to_process = self._queue.copy()
            self._queue.clear()
            self._timer = None

        # 记录日志
        logger.info("Processing %d queued memory updates", len(contexts_to_process))

        try:
            # 创建 MemoryUpdater 实例
            updater = MemoryUpdater()

            # 遍历所有待处理的上下文
            for context in contexts_to_process:
                try:
                    # 记录日志
                    logger.info("Updating memory for thread %s", context.thread_id)

                    # 调用 updater 进行实际的记忆更新
                    success = updater.update_memory(
                        messages=context.messages,
                        thread_id=context.thread_id,
                        agent_name=context.agent_name,
                        correction_detected=context.correction_detected,
                        reinforcement_detected=context.reinforcement_detected,
                    )

                    # 根据更新结果记录日志
                    if success:
                        logger.info("Memory updated successfully for thread %s", context.thread_id)
                    else:
                        logger.warning("Memory update skipped/failed for thread %s", context.thread_id)

                except Exception as e:
                    # 记录更新过程中的错误
                    logger.error("Error updating memory for thread %s: %s", context.thread_id, e)

                # 如果有多个上下文需要处理，添加小延迟避免限流
                if len(contexts_to_process) > 1:
                    time.sleep(0.5)

        finally:
            # 处理完成后，重置处理状态
            with self._lock:
                self._processing = False

    # flush：强制立即处理队列
    #
    # 作用说明：
    #   取消定时器并立即处理队列中的所有上下文。
    #   用于测试或优雅关闭场景。
    def flush(self) -> None:
        """Force immediate processing of the queue.

        This is useful for testing or graceful shutdown.
        """
        with self._lock:
            # 取消定时器
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

        # 立即处理队列
        self._process_queue()

    # clear：清空队列
    #
    # 作用说明：
    #   取消定时器并清空队列，不进行任何处理。
    #   用于测试场景。
    def clear(self) -> None:
        """Clear the queue without processing.

        This is useful for testing.
        """
        with self._lock:
            # 取消定时器
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            # 清空队列
            self._queue.clear()
            # 重置处理状态
            self._processing = False

    # pending_count：获取待处理更新的数量
    #
    # 返回值：
    #   int：队列中待处理的上下文数量
    @property
    def pending_count(self) -> int:
        """Get the number of pending updates."""
        with self._lock:
            return len(self._queue)

    # is_processing：检查是否正在处理
    #
    # 返回值：
    #   bool：是否正在处理队列
    @property
    def is_processing(self) -> bool:
        """Check if the queue is currently being processed."""
        with self._lock:
            return self._processing


# ============================================================
# 模块级变量和单例模式
# ============================================================

# _memory_queue：全局单例队列实例
_memory_queue: MemoryUpdateQueue | None = None

# _queue_lock：保护队列创建的线程锁
_queue_lock = threading.Lock()


# get_memory_queue：获取全局记忆更新队列单例
#
# 作用说明：
#   返回全局的 MemoryUpdateQueue 单例实例。
#   使用双检查锁定模式确保线程安全初始化。
#
# 调用位置：
#   MemoryMiddleware.after_agent() 调用此函数
#   来源文件：deerflow/agents/middlewares/memory_middleware.py
#
# 返回值：
#   MemoryUpdateQueue：记忆更新队列实例
def get_memory_queue() -> MemoryUpdateQueue:
    """Get the global memory update queue singleton.

    Returns:
        The memory update queue instance.
    """
    global _memory_queue
    with _queue_lock:
        # 双检查锁定：第一次检查避免已创建后的锁定开销
        if _memory_queue is None:
            # 创建新的 MemoryUpdateQueue 实例
            _memory_queue = MemoryUpdateQueue()
        return _memory_queue


# reset_memory_queue：重置全局记忆队列
#
# 作用说明：
#   清空并重置全局的 MemoryUpdateQueue 单例。
#   用于测试场景。
def reset_memory_queue() -> None:
    """Reset the global memory queue.

    This is useful for testing.
    """
    global _memory_queue
    with _queue_lock:
        if _memory_queue is not None:
            _memory_queue.clear()
        _memory_queue = None
