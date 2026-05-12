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

# ============================================================
# 导入标准库
# ============================================================

# hashlib：标准库哈希模块，用于计算工具调用的 MD5 哈希值
import hashlib

# json：标准库 JSON 模块，用于序列化和反序列化 JSON 数据
import json

# logging：标准库日志模块，用于记录中间件运行日志
import logging

# threading：标准库线程模块，用于线程安全的锁
import threading

# collections 导入：
#   - OrderedDict：有序字典，用于实现 LRU 缓存
#   - defaultdict：默认值字典，用于存储每个线程的已警告哈希集合
from collections import OrderedDict, defaultdict

# typing 导入：
#   - override：方法重写标记，用于明确表示重写父类方法
from typing import override

# ============================================================
# 导入 LangChain / LangGraph 相关模块
# ============================================================

# langchain.agents.AgentState：
#   LangChain agent 的基础状态类，所有自定义状态类继承自此
from langchain.agents import AgentState

# langchain.agents.middleware.AgentMiddleware：
#   LangChain 的中间件基类，所有自定义中间件必须继承此类
from langchain.agents.middleware import AgentMiddleware

# langchain_core.messages.HumanMessage：
#   人类消息类型，用于注入警告消息
#   注意：使用 HumanMessage 而不是 SystemMessage
#   原因：某些模型（如 Anthropic）要求系统消息只在对话开始时出现
from langchain_core.messages import HumanMessage

# langgraph.runtime.Runtime：
#   LangGraph 运行时上下文，在钩子方法中作为参数传入
from langgraph.runtime import Runtime

# ============================================================
# 模块级变量初始化
# ============================================================

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)

# ============================================================
# 模块级常量定义
# ============================================================

# 默认阈值常量：可以通过构造函数覆盖
_DEFAULT_WARN_THRESHOLD = 3   # 注入警告前的相同调用次数
_DEFAULT_HARD_LIMIT = 5       # 强制停止前的相同调用次数
_DEFAULT_WINDOW_SIZE = 20     # 跟踪最近 N 个工具调用哈希
_DEFAULT_MAX_TRACKED_THREADS = 100  # LRU 驱逐限制：最大跟踪线程数


# ============================================================
# 模块级辅助函数
# ============================================================

# _normalize_tool_call_args：标准化工具调用参数
#
# 方法作用：
#   兼容处理不同 provider 将 args 序列化为不同格式的情况，
#   同时为非字典负载保留稳定的回退键。
#
# 参数：
#   raw_args: object，原始参数（可能是 dict、str 或 None）
#
# 返回值：
#   tuple[dict, str | None]：元组
#     - dict: 标准化的参数字典
#     - str | None: 如果无法标准化，返回原始值的 JSON 字符串作为回退键
#
# 兼容处理：
#   有些 provider 将 args 序列化为 JSON 字符串而不是字典
#   这个函数防御性地解析这些情况，同时为非字典负载保留稳定的回退键
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


# _stable_tool_key：从显著参数生成稳定的键
#
# 方法作用：
#   从工具调用的显著参数生成一个稳定的键，用于哈希计算。
#   不同类型的工具使用不同的键生成策略。
#
# 参数：
#   name: str，工具名称
#   args: dict，参数字典
#   fallback_key: str | None，如果参数无法标准化时使用的回退键
#
# 返回值：
#   str：稳定的工具调用键
#
# 设计考虑：
#   - read_file：按行号范围分桶（每 200 行为一个桶），避免微小差异导致不同哈希
#   - write_file/str_replace：使用完整参数的 JSON（内容敏感，相同路径可能有不同内容）
#   - 其他工具：提取显著字段（path、url、query 等）
def _stable_tool_key(name: str, args: dict, fallback_key: str | None) -> str:
    """Derive a stable key from salient args without overfitting to noise."""
    # read_file 特殊处理：按行号范围分桶
    if name == "read_file" and fallback_key is None:
        # 获取路径和行号范围
        path = args.get("path") or ""
        start_line = args.get("start_line")
        end_line = args.get("end_line")

        # 每 200 行为一个桶
        bucket_size = 200

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

        # 计算桶范围（行号从 1 开始，所以需要减 1）
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


# _hash_tool_calls：对工具调用列表进行确定性哈希
#
# 方法作用：
#   将工具调用列表转换为一个确定性的哈希值，用于检测循环。
#
# 参数：
#   tool_calls: list[dict]，工具调用列表
#
# 返回值：
#   str：12 字符的 MD5 哈希
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
# 提示 agent 停止调用工具，产生最终回答
_WARNING_MSG = "[LOOP DETECTED] You are repeating the same tool calls. Stop calling tools and produce your final answer now. If you cannot complete the task, summarize what you accomplished so far."

# 强制停止消息：出现在达到 hard_limit 次时
# 强制剥离所有 tool_calls，强制产生文本回答
_HARD_STOP_MSG = "[FORCED STOP] Repeated tool calls exceeded the safety limit. Producing final answer with results collected so far."


# ============================================================
# LoopDetectionMiddleware 主类
# ============================================================

# LoopDetectionMiddleware 类：检测和打破重复工具调用循环
#
# 核心作用：
#   P0 安全机制，防止 agent 无限循环调用相同的工具和参数。
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

    # state_schema：类变量，指定该中间件使用的状态类型
    state_schema = AgentState

    # ============================================================
    # 构造函数
    # ============================================================

    # __init__：构造函数
    #
    # 参数：
    #   warn_threshold: int，相同调用次数达到此值时注入警告（默认 3）
    #   hard_limit: int，相同调用次数达到此值时强制停止（默认 5）
    #   window_size: int，滑动窗口大小（默认 20）
    #   max_tracked_threads: int，最大跟踪线程数，超过则 LRU 驱逐（默认 100）
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

        # 线程锁，保护共享数据结构（_history 和 _warned）
        self._lock = threading.Lock()

        # Per-thread 跟踪历史：OrderedDict 实现 LRU
        # key: thread_id, value: 哈希列表（最近的 tool_calls 哈希）
        self._history: OrderedDict[str, list[str]] = OrderedDict()

        # 每个线程已警告的哈希集合：避免重复警告
        # key: thread_id, value: 哈希集合
        self._warned: dict[str, set[str]] = defaultdict(set)

    # ============================================================
    # 内部辅助方法
    # ============================================================

    # _get_thread_id：从 runtime 提取 thread_id
    #
    # 方法作用：
    #   从 LangGraph 运行时上下文获取 thread_id，
    #   用于每个线程独立跟踪。
    #
    # 参数：
    #   runtime: Runtime，LangGraph 运行时上下文
    #
    # 返回值：
    #   str：thread_id 字符串，如果找不到则返回 "default"
    def _get_thread_id(self, runtime: Runtime) -> str:
        """Extract thread_id from runtime context for per-thread tracking."""
        # runtime.context 是 LangGraph 传递的上下文信息字典
        thread_id = runtime.context.get("thread_id") if runtime.context else None
        if thread_id:
            return thread_id
        # 如果没有找到 thread_id，使用 "default" 作为回退
        return "default"

    # _evict_if_needed：如果跟踪的线程数超过限制，执行 LRU 驱逐
    #
    # 方法作用：
    #   当跟踪的线程数超过 max_tracked_threads 时，
    #   驱逐最旧的（least recently used）线程的跟踪数据。
    #
    # 注意：
    #   必须在持有 self._lock 的情况下调用
    def _evict_if_needed(self) -> None:
        """Evict least recently used threads if over the limit.

        Must be called while holding self._lock.
        """
        # 当历史记录超过最大线程数时
        while len(self._history) > self.max_tracked_threads:
            # 驱逐最旧的（OrderedDict 的第一项）
            # popitem(last=False) 返回并移除第一项 (key, value)
            evicted_id, _ = self._history.popitem(last=False)
            # 同时清除该线程的警告记录
            self._warned.pop(evicted_id, None)
            logger.debug("Evicted loop tracking for thread %s (LRU)", evicted_id)

    # _track_and_check：跟踪工具调用并检查是否有循环
    #
    # 方法作用：
    #   在滑动窗口中跟踪工具调用的哈希，并检查是否达到警告或停止阈值。
    #
    # 参数：
    #   state: AgentState，当前 agent 状态
    #   runtime: Runtime，LangGraph 运行时上下文
    #
    # 返回值：
    #   tuple[str | None, bool]：元组
    #     - 第一个元素：警告消息字符串，如果不需要警告则为 None
    #     - 第二个元素：是否应该强制停止
    def _track_and_check(self, state: AgentState, runtime: Runtime) -> tuple[str | None, bool]:
        """Track tool calls and check for loops.

        Returns:
            (warning_message_or_none, should_hard_stop)
        """
        # 从状态中获取消息列表
        messages = state.get("messages", [])
        if not messages:
            return None, False

        # 获取最后一条消息
        last_msg = messages[-1]
        # 确保是 AI 消息
        if getattr(last_msg, "type", None) != "ai":
            return None, False

        # 获取工具调用列表
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
            # 添加当前哈希到历史
            history.append(call_hash)

            # 如果历史超过窗口大小，裁剪到窗口大小
            # 只保留最近的 window_size 个哈希
            if len(history) > self.window_size:
                history[:] = history[-self.window_size:]

            # 计算当前哈希在历史中的出现次数
            count = history.count(call_hash)

            # 获取工具名称列表（用于日志）
            tool_names = [tc.get("name", "?") for tc in tool_calls]

            # ---- 检查是否达到硬限制 ----
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

            # ---- 检查是否达到警告阈值 ----
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

    # _append_text：向 AIMessage 内容追加文本
    #
    # 方法作用：
    #   向 AIMessage 的 content 字段追加文本，
    #   处理各种可能的 content 类型。
    #
    # 参数：
    #   content: str | list | None，AIMessage 的 content 字段
    #   text: str，要追加的文本
    #
    # 返回值：
    #   str | list：更新后的 content（类型与输入相同）
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

    # _apply：应用循环检测逻辑
    #
    # 方法作用：
    #   根据 _track_and_check 的结果，应用相应的动作
    #   （注入警告消息或强制停止）。
    #
    # 参数：
    #   state: AgentState，当前 agent 状态
    #   runtime: Runtime，LangGraph 运行时上下文
    #
    # 返回值：
    #   dict | None：状态更新字典，如果不需要更新则返回 None
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
            # 原因：某些模型（如 Anthropic）要求系统消息只在对话开始时出现
            # 中间注入系统消息会导致格式错误
            # HumanMessage 与所有 provider 兼容
            return {"messages": [HumanMessage(content=warning)]}

        return None

    # ============================================================
    # LangChain AgentMiddleware 钩子方法
    # ============================================================

    # after_model：同步版本的模型调用后钩子
    #
    # 方法作用：
    #   LangChain AgentMiddleware 提供的扩展点，
    #   在模型执行后同步执行循环检测。
    #
    # 参数：
    #   state: AgentState，当前 agent 状态
    #   runtime: Runtime，LangGraph 运行时上下文
    #
    # 返回值：
    #   dict | None：状态更新字典，如果不需要更新则返回 None
    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    # aafter_model：异步版本的模型调用后钩子
    #
    # 方法作用：
    #   与 after_model 相同，但用于异步调用。
    #
    # 参数：
    #   state: AgentState，当前 agent 状态
    #   runtime: Runtime，LangGraph 运行时上下文
    #
    # 返回值：
    #   dict | None：状态更新字典
    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._apply(state, runtime)

    # ============================================================
    # 公共方法
    # ============================================================

    # reset：清除跟踪状态
    #
    # 方法作用：
    #   清除循环检测的跟踪状态。
    #
    # 参数：
    #   thread_id: str | None，如果提供，只清除该线程的跟踪；否则清除所有
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
