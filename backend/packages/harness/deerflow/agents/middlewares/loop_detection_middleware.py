"""LoopDetectionMiddleware - 循环检测与打破中间件。

功能概述：
  P0 安全机制：防止 agent 无限循环调用相同的工具和参数。

问题背景：
  如果不加以限制，重复的工具调用循环会一直持续直到递归限制杀死运行。

检测策略：
  1. after_model 钩子中，对工具调用（名称 + 参数）进行 MD5 哈希
  2. 在滑动窗口（默认 20 条）中跟踪最近的哈希
  3. 如果相同哈希出现 >= warn_threshold 次（默认 3），注入警告消息（每个哈希只注入一次）
  4. 如果出现 >= hard_limit 次（默认 5），剥离所有 tool_calls，强制 agent 产生最终文本回答

哈希算法特点：
  - 顺序无关：相同的多集合工具调用产生相同的哈希
  - read_file：按行号范围分桶（每 200 行为一个桶），避免微小差异导致不同哈希
  - write_file/str_replace：使用完整参数 JSON（内容敏感）
  - 其他工具：提取显著字段（path、url、query 等）

滑动窗口：
  - 每个线程独立跟踪（通过 thread_id）
  - 使用 LRU 驱逐防止内存溢出（最大 100 个线程）

执行位置：在 SubagentLimitMiddleware 之后执行。
"""
"""Middleware to detect and break repetitive tool call loops.

P0 safety: prevents the agent from calling the same tool with the same
arguments indefinitely until the recursion limit kills the run.

Detection strategy:
  1. After each model response, hash the tool calls (name + args).
  2. Track recent hashes in a sliding window.
  3. If the same hash appears >= warn_threshold times, inject a
     "you are repeating yourself — wrap up" system message (once per hash).
  4. If it appears >= hard_limit times, strip all tool_calls from the
     response so the agent is forced to produce a final text answer.
"""

# 导入标准库
# hashlib: 用于计算工具调用的哈希值
import hashlib
# json: 用于序列化和反序列化 JSON 数据
import json
# logging: 用于记录日志
import logging
# threading: 用于线程安全的锁
import threading
# collections: 导入 OrderedDict（有序字典）和 defaultdict（默认值字典）
from collections import OrderedDict, defaultdict
# typing: 导入 override（方法重写标记）
from typing import override

# 从 langchain.agents 导入 AgentState（agent 基础状态类）
from langchain.agents import AgentState
# 从 langchain.agents.middleware 导入 AgentMiddleware（中间件基类）
from langchain.agents.middleware import AgentMiddleware
# 从 langchain_core.messages 导入 HumanMessage（人类消息）
# 用于注入警告消息（注意：不是 SystemMessage）
from langchain_core.messages import HumanMessage
# 从 langgraph.runtime 导入 Runtime（LangGraph 运行时上下文）
from langgraph.runtime import Runtime

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)

# 模块级常量：默认阈值
# 可以通过构造函数覆盖
_DEFAULT_WARN_THRESHOLD = 3  # 注入警告前的相同调用次数
_DEFAULT_HARD_LIMIT = 5     # 强制停止前的相同调用次数
_DEFAULT_WINDOW_SIZE = 20     # 跟踪最近 N 个工具调用
_DEFAULT_MAX_TRACKED_THREADS = 100  # LRU 驱逐限制：最大跟踪线程数


# 模块级函数：标准化工具调用参数
#
# 兼容处理：有些 provider 将 args 序列化为 JSON 字符串而不是字典
# 这个函数防御性地解析这些情况，同时为非字典负载保留稳定的回退键
#
# 参数：
#   raw_args: 原始参数（可能是 dict、str 或 None）
#
# 返回值：
#   (dict, str | None) 元组
#   - dict: 标准化的参数字典
#   - str | None: 如果无法标准化，返回原始值的 JSON 字符串作为回退键
def _normalize_tool_call_args(raw_args: object) -> tuple[dict, str | None]:
    """Normalize tool call args to a dict plus an optional fallback key.

    Some providers serialize ``args`` as a JSON string instead of a dict.
    We defensively parse those cases so loop detection does not crash while
    still preserving a stable fallback key for non-dict payloads.
    """
    # 已经是字典，直接返回
    if isinstance(raw_args, dict):
        return raw_args, None

    # 是字符串，尝试解析为 JSON
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except (TypeError, ValueError, json.JSONDecodeError):
            # 解析失败，使用原始字符串作为回退键
            return {}, raw_args

        # 解析成功，检查是否是字典
        if isinstance(parsed, dict):
            return parsed, None
        # 解析成功但不是字典，使用 JSON 序列化作为回退键
        return {}, json.dumps(parsed, sort_keys=True, default=str)

    # 是 None
    if raw_args is None:
        return {}, None

    # 其他类型，使用 JSON 序列化作为回退键
    return {}, json.dumps(raw_args, sort_keys=True, default=str)


# 模块级函数：从显著的参数生成稳定的键
#
# 参数：
#   name: 工具名称
#   args: 参数字典
#   fallback_key: 回退键（如果参数无法标准化时使用）
#
# 返回值：
#   字符串，稳定的工具调用键
#
# 设计考虑：
#   - read_file：按行号范围分桶（每 200 行为一个桶），避免微小差异导致不同哈希
#   - write_file/str_replace：使用完整参数的 JSON（内容敏感，相同路径可能有不同内容）
#   - 其他工具：提取显著字段（path、url、query 等）
def _stable_tool_key(name: str, args: dict, fallback_key: str | None) -> str:
    """Derive a stable key from salient args without overfitting to noise."""
    # read_file 特殊处理：按行号范围分桶
    if name == "read_file" and fallback_key is None:
        path = args.get("path") or ""
        start_line = args.get("start_line")
        end_line = args.get("end_line")

        bucket_size = 200  # 每 200 行为一个桶

        # 安全转换 start_line 为整数
        try:
            start_line = int(start_line) if start_line is not None else 1
        except (TypeError, ValueError):
            start_line = 1

        # 安全转换 end_line 为整数
        try:
            end_line = int(end_line) if end_line is not None else start_line
        except (TypeError, ValueError):
            end_line = start_line

        # 确保 start_line <= end_line
        start_line, end_line = sorted((start_line, end_line))
        # 计算桶范围（从 0 开始）
        bucket_start = max(start_line, 1)
        bucket_end = max(end_line, 1)
        bucket_start = (bucket_start - 1) // bucket_size
        bucket_end = (bucket_end - 1) // bucket_size

        # 返回 "path:start_bucket-end_bucket" 格式
        return f"{path}:{bucket_start}-{bucket_end}"

    # write_file / str_replace 特殊处理：内容敏感
    # 这些工具相同路径可能更新不同内容，只用显著字段会错误合并
    # 所以使用完整参数的 JSON
    if name in {"write_file", "str_replace"}:
        if fallback_key is not None:
            return fallback_key
        return json.dumps(args, sort_keys=True, default=str)

    # 其他工具：提取显著字段
    # 这些字段通常表示工具操作的主要对象
    salient_fields = ("path", "url", "query", "command", "pattern", "glob", "cmd")
    stable_args = {field: args[field] for field in salient_fields if args.get(field) is not None}

    if stable_args:
        # 有显著字段，使用显著字段的 JSON
        return json.dumps(stable_args, sort_keys=True, default=str)

    # 没有显著字段，使用回退键或完整参数的 JSON
    if fallback_key is not None:
        return fallback_key

    return json.dumps(args, sort_keys=True, default=str)


# 模块级函数：对工具调用列表进行确定性哈希
#
# 参数：
#   tool_calls: 工具调用列表
#
# 返回值：
#   字符串，12 字符的 MD5 哈希
#
# 设计考虑：
#   - 顺序无关：相同的多集合工具调用产生相同的哈希
#   - 每个工具调用被规范化为 "name:key" 格式
#   - 规范化后的列表排序后进行哈希
def _hash_tool_calls(tool_calls: list[dict]) -> str:
    """Deterministic hash of a set of tool calls (name + stable key).

    This is intended to be order-independent: the same multiset of tool calls
    should always produce the same hash, regardless of their input order.
    """
    # 规范化每个工具调用为 (name, key) 结构
    normalized: list[str] = []
    for tc in tool_calls:
        name = tc.get("name", "")  # 工具名称
        args, fallback_key = _normalize_tool_call_args(tc.get("args", {}))  # 参数
        key = _stable_tool_key(name, args, fallback_key)  # 稳定键

        # 格式："工具名:键"
        normalized.append(f"{name}:{key}")

    # 排序，使不同顺序的相同调用集合产生相同结果
    normalized.sort()
    # JSON 序列化后计算 MD5 哈希，只取前 12 字符
    blob = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


# 模块级字符串常量：警告消息和强制停止消息
#
# 警告消息：出现在达到 warn_threshold 次时
_WARNING_MSG = "[LOOP DETECTED] You are repeating the same tool calls. Stop calling tools and produce your final answer now. If you cannot complete the task, summarize what you accomplished so far."

# 强制停止消息：出现在达到 hard_limit 次时
_HARD_STOP_MSG = "[FORCED STOP] Repeated tool calls exceeded the safety limit. Producing final answer with results collected so far."


# LoopDetectionMiddleware 类：检测和打破重复工具调用循环
#
# 工作流程：
#   1. 在 after_model 钩子中检查最后一条 AI 消息是否有 tool_calls
#   2. 对 tool_calls 计算哈希
#   3. 在滑动窗口中跟踪哈希出现次数
#   4. 达到警告阈值：注入警告消息
#   5. 达到硬限制：剥离 tool_calls，强制产生文本回答
#
# 关键设计：
#   - 每个线程独立跟踪（通过 thread_id）
#   - 使用 LRU 驱逐防止内存溢出
#   - 每个哈希只警告一次
class LoopDetectionMiddleware(AgentMiddleware[AgentState]):
    """Detects and breaks repetitive tool call loops.

    Args:
        warn_threshold: Number of identical tool call sets before injecting
            a warning message. Default: 3.
        hard_limit: Number of identical tool call sets before stripping
            tool_calls entirely. Default: 5.
        window_size: Size of the sliding window for tracking calls.
            Default: 20.
        max_tracked_threads: Maximum number of threads to track before
            evicting the least recently used. Default: 100.
    """

    # 构造函数
    #
    # 参数：
    #   warn_threshold: 相同调用次数达到此值时注入警告（默认 3）
    #   hard_limit: 相同调用次数达到此值时强制停止（默认 5）
    #   window_size: 滑动窗口大小（默认 20）
    #   max_tracked_threads: 最大跟踪线程数，超过则 LRU 驱逐（默认 100）
    def __init__(
        self,
        warn_threshold: int = _DEFAULT_WARN_THRESHOLD,
        hard_limit: int = _DEFAULT_HARD_LIMIT,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        max_tracked_threads: int = _DEFAULT_MAX_TRACKED_THREADS,
    ):
        # 调用父类构造函数
        super().__init__()
        # 保存配置参数
        self.warn_threshold = warn_threshold
        self.hard_limit = hard_limit
        self.window_size = window_size
        self.max_tracked_threads = max_tracked_threads

        # 线程锁，保护共享数据结构
        self._lock = threading.Lock()

        # Per-thread 跟踪历史：OrderedDict 实现 LRU
        # key: thread_id, value: 哈希列表（最近的 tool_calls 哈希）
        self._history: OrderedDict[str, list[str]] = OrderedDict()

        # 每个线程已警告的哈希集合：避免重复警告
        # key: thread_id, value: 哈希集合
        self._warned: dict[str, set[str]] = defaultdict(set)

    # 内部方法：从 runtime 提取 thread_id
    #
    # 参数：
    #   runtime: LangGraph 运行时上下文
    #
    # 返回值：
    #   thread_id 字符串，如果找不到则返回 "default"
    def _get_thread_id(self, runtime: Runtime) -> str:
        """Extract thread_id from runtime context for per-thread tracking."""
        thread_id = runtime.context.get("thread_id") if runtime.context else None
        if thread_id:
            return thread_id
        return "default"

    # 内部方法：如果跟踪的线程数超过限制，执行 LRU 驱逐
    #
    # 注意：必须在持有 self._lock 的情况下调用
    def _evict_if_needed(self) -> None:
        """Evict least recently used threads if over the limit.

        Must be called while holding self._lock.
        """
        # 当历史记录超过最大线程数时
        while len(self._history) > self.max_tracked_threads:
            # 驱逐最旧的（OrderedDict 的第一项）
            evicted_id, _ = self._history.popitem(last=False)
            # 同时清除警告记录
            self._warned.pop(evicted_id, None)
            logger.debug("Evicted loop tracking for thread %s (LRU)", evicted_id)

    # 内部方法：跟踪工具调用并检查是否有循环
    #
    # 参数：
    #   state: AgentState，当前 agent 状态
    #   runtime: LangGraph 运行时上下文
    #
    # 返回值：
    #   (warning_message_or_none, should_hard_stop) 元组
    #   - 第一个元素：警告消息字符串，如果不需要警告则为 None
    #   - 第二个元素：是否应该强制停止
    def _track_and_check(self, state: AgentState, runtime: Runtime) -> tuple[str | None, bool]:
        """Track tool calls and check for loops.

        Returns:
            (warning_message_or_none, should_hard_stop)
        """
        # 获取消息列表
        messages = state.get("messages", [])
        if not messages:
            return None, False

        # 获取最后一条消息
        last_msg = messages[-1]
        # 确保是 AI 消息
        if getattr(last_msg, "type", None) != "ai":
            return None, False

        # 获取工具调用
        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return None, False

        # 获取 thread_id
        thread_id = self._get_thread_id(runtime)

        # 计算工具调用的哈希
        call_hash = _hash_tool_calls(tool_calls)

        # 使用锁保护共享数据结构
        with self._lock:
            # 更新 LRU：将当前 thread_id 移到末尾（最近使用）
            if thread_id in self._history:
                self._history.move_to_end(thread_id)
            else:
                # 新 thread，创建空列表
                self._history[thread_id] = []
                # 检查是否需要 LRU 驱逐
                self._evict_if_needed()

            # 获取该线程的跟踪历史
            history = self._history[thread_id]
            # 添加当前哈希
            history.append(call_hash)

            # 如果历史超过窗口大小，裁剪到窗口大小
            if len(history) > self.window_size:
                history[:] = history[-self.window_size:]

            # 计算当前哈希在历史中的出现次数
            count = history.count(call_hash)

            # 获取工具名称列表（用于日志）
            tool_names = [tc.get("name", "?") for tc in tool_calls]

            # 检查是否达到硬限制
            if count >= self.hard_limit:
                # 记录错误日志
                logger.error(
                    "Loop hard limit reached — forcing stop",
                    extra={
                        "thread_id": thread_id,
                        "call_hash": call_hash,
                        "count": count,
                        "tools": tool_names,
                    },
                )
                # 返回强制停止消息和 True
                return _HARD_STOP_MSG, True

            # 检查是否达到警告阈值
            if count >= self.warn_threshold:
                warned = self._warned[thread_id]
                # 如果这个哈希还没警告过
                if call_hash not in warned:
                    # 添加到已警告集合
                    warned.add(call_hash)
                    # 记录警告日志
                    logger.warning(
                        "Repetitive tool calls detected — injecting warning",
                        extra={
                            "thread_id": thread_id,
                            "call_hash": call_hash,
                            "count": count,
                            "tools": tool_names,
                        },
                    )
                    # 返回警告消息和 False（不强制停止）
                    return _WARNING_MSG, False
                # 已经警告过了，不再返回警告
                return None, False

        # 没有达到任何阈值，正常继续
        return None, False

    # 静态方法：向 AIMessage 内容追加文本
    #
    # 参数：
    #   content: AIMessage 的 content 字段（可能是 str、list 或 None）
    #   text: 要追加的文本
    #
    # 返回值：
    #   更新后的 content（类型与输入相同）
    #
    # 兼容处理：
    #   - 如果是 list（如 Anthropic thinking 模式），追加新的 text block
    #   - 如果是 str，直接拼接
    #   - 如果是 None，返回文本
    @staticmethod
    def _append_text(content: str | list | None, text: str) -> str | list:
        """Append *text* to AIMessage content, handling str, list, and None.

        When content is a list of content blocks (e.g. Anthropic thinking mode),
        we append a new ``{"type": "text", ...}`` block instead of concatenating
        a string to a list, which would raise ``TypeError``.
        """
        if content is None:
            return text
        if isinstance(content, list):
            # 追加新的 text block
            return [*content, {"type": "text", "text": f"\n\n{text}"}]
        if isinstance(content, str):
            return content + f"\n\n{text}"
        # 其他类型，转换为字符串后拼接
        return str(content) + f"\n\n{text}"

    # 内部方法：应用循环检测逻辑
    #
    # 参数：
    #   state: AgentState，当前 agent 状态
    #   runtime: LangGraph 运行时上下文
    #
    # 返回值：
    #   状态更新字典，如果不需要更新则返回 None
    def _apply(self, state: AgentState, runtime: Runtime) -> dict | None:
        # 调用跟踪和检查方法
        warning, hard_stop = self._track_and_check(state, runtime)

        if hard_stop:
            # 达到硬限制：剥离 tool_calls，强制产生文本回答
            messages = state.get("messages", [])
            last_msg = messages[-1]
            # 复制消息并更新：清空 tool_calls，追加强制停止消息
            stripped_msg = last_msg.model_copy(
                update={
                    "tool_calls": [],  # 清空工具调用
                    "content": self._append_text(last_msg.content, _HARD_STOP_MSG),  # 追加消息
                }
            )
            # 返回只包含被修改消息的状态更新
            return {"messages": [stripped_msg]}

        if warning:
            # 达到警告阈值：注入警告消息
            # 注意：使用 HumanMessage 而不是 SystemMessage
            # 原因：Anthropic 模型要求系统消息只在对话开始时出现
            # 中间注入系统消息会导致 _format_messages() 崩溃
            # HumanMessage 与所有 provider 兼容
            return {"messages": [HumanMessage(content=warning)]}

        return None

    # after_model 钩子方法：同步版本
    #
    # 在模型执行后被调用，检查循环
    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    # aafter_model 钩子方法：异步版本
    #
    # 在模型执行后被调用，检查循环
    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    # reset 方法：清除跟踪状态
    #
    # 参数：
    #   thread_id: 如果提供，只清除该线程的跟踪；否则清除所有
    def reset(self, thread_id: str | None = None) -> None:
        """Clear tracking state. If thread_id given, clear only that thread."""
        with self._lock:
            if thread_id:
                # 清除指定线程的跟踪
                self._history.pop(thread_id, None)
                self._warned.pop(thread_id, None)
            else:
                # 清除所有跟踪
                self._history.clear()
                self._warned.clear()