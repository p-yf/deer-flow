"""MemoryMiddleware - 记忆机制中间件。

功能概述：
  在每次 agent 执行完成后，将对话内容加入记忆更新队列，实现长期记忆。

工作流程：
  1. after_agent 钩子触发（agent 执行完成后）
  2. 过滤消息，只保留用户输入和最终 AI 响应（排除工具调用过程）
  3. 去除 <uploaded_files> 块（文件路径是会话作用域，不能持久化）
  4. 检测纠错/强化信号（用于影响记忆更新方式）
  5. 将对话加入 MemoryUpdateQueue（防抖队列）

消息过滤规则：
  - 保留：用户消息（HumanMessage）
  - 保留：没有 tool_calls 的 AI 消息（最终响应）
  - 排除：ToolMessage（工具调用结果）
  - 排除：有 tool_calls 的 AI 消息（中间步骤）
  - 排除：<uploaded_files> 块

信号检测：
  - 纠错信号：用户明确指出 agent 错误（"不对"，"你理解错了"等）
  - 强化信号：用户确认 agent 做法正确（"对，就是这样"等）

防抖机制：
  MemoryUpdateQueue 使用 30 秒防抖，延迟执行记忆更新，
  避免频繁更新，同时合并短时间内多次更新请求。

执行位置：紧接 TitleMiddleware 之后。
"""
"""Middleware for memory mechanism."""

# 导入标准库 logging，用于记录日志
import logging
# re 模块：用于正则表达式匹配
# memory_middleware 使用正则来检测纠错和强化信号，以及去除上传文件块
import re
# typing 导入 Any（任意类型）和 override（方法重写标记）
from typing import Any, override

# 从 langchain.agents 导入 AgentState（agent 基础状态类）
from langchain.agents import AgentState
# 从 langchain.agents.middleware 导入 AgentMiddleware（中间件基类）
from langchain.agents.middleware import AgentMiddleware
# 从 langgraph.config 导入 get_config，用于获取 LangGraph 的运行时配置
from langgraph.config import get_config
# 从 langgraph.runtime 导入 Runtime，表示 LangGraph 的运行时上下文
from langgraph.runtime import Runtime

# 从 deerflow.agents.memory.queue 导入 get_memory_queue
# MemoryUpdateQueue 是记忆更新的防抖队列，记忆更新不会立即执行，而是延迟等待
from deerflow.agents.memory.queue import get_memory_queue
# 从 deerflow.config.memory_config 导入 get_memory_config
# 获取记忆系统的配置（如是否启用、debounce 秒数等）
from deerflow.config.memory_config import get_memory_config

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)

# 模块级正则表达式：用于匹配和去除消息中的 <uploaded_files> 块
# 这个块是由 UploadsMiddleware 注入的，包含文件列表信息
# 文件路径是会话作用域的，不能持久化到长期记忆
# re.IGNORECASE 使匹配不区分大小写，[\s\S]*? 匹配任意字符（包括换行）
_UPLOAD_BLOCK_RE = re.compile(r"<uploaded_files>[\s\S]*?</uploaded_files>\n*", re.IGNORECASE)

# 模块级元组：存储检测"纠错信号"的正则表达式
# 当用户明确指出 agent 的错误时触发（如 "that's wrong", "you misunderstood" 等中英文表达）
_CORRECTION_PATTERNS = (
    # 英文纠错模式
    re.compile(r"\bthat(?:'s| is) (?:wrong|incorrect)\b", re.IGNORECASE),  # "that's wrong", "that is incorrect"
    re.compile(r"\byou misunderstood\b", re.IGNORECASE),                     # "you misunderstood"
    re.compile(r"\btry again\b", re.IGNORECASE),                           # "try again"
    re.compile(r"\bredo\b", re.IGNORECASE),                                # "redo"
    # 中文纠错模式
    re.compile(r"不对"),      # "不对"
    re.compile(r"你理解错了"),  # "你理解错了"
    re.compile(r"你理解有误"),  # "你理解有误"
    re.compile(r"重试"),       # "重试"
    re.compile(r"重新来"),     # "重新来"
    re.compile(r"换一种"),     # "换一种"
    re.compile(r"改用"),       # "改用"
)

# 模块级元组：存储检测"正向强化信号"的正则表达式
# 当用户确认 agent 的做法正确或有帮助时触发
# 这让记忆系统记录什么方法有效，而不仅仅是记录错误
_REINFORCEMENT_PATTERNS = (
    # 英文强化模式
    re.compile(r"\byes[,.]?\s+(?:exactly|perfect|that(?:'s| is) (?:right|correct|it))\b", re.IGNORECASE),  # "yes, exactly", "yes, perfect"
    re.compile(r"\bperfect(?:[.!?]|$)", re.IGNORECASE),  # "perfect."
    re.compile(r"\bexactly\s+(?:right|correct)\b", re.IGNORECASE),  # "exactly right"
    re.compile(r"\bthat(?:'s| is)\s+(?:exactly\s+)?(?:right|correct|what i (?:wanted|needed|meant))\b", re.IGNORECASE),  # "that's exactly what I wanted"
    re.compile(r"\bkeep\s+(?:doing\s+)?that\b", re.IGNORECASE),  # "keep doing that"
    re.compile(r"\bjust\s+(?:like\s+)?(?:that|this)\b", re.IGNORECASE),  # "just like that"
    re.compile(r"\bthis is (?:great|helpful)\b(?:[.!?]|$)", re.IGNORECASE),  # "this is great."
    re.compile(r"\bthis is what i wanted\b(?:[.!?]|$)", re.IGNORECASE),  # "this is what I wanted."
    # 中文强化模式
    re.compile(r"对[，,]?\s*就是这样(?:[。！？!?.]|$)"),  # "对，就是这样"
    re.compile(r"完全正确(?:[。！？!?.]|$)"),  # "完全正确。"
    re.compile(r"(?:对[，,]?\s*)?就是这个意思(?:[。！？!?.]|$)"),  # "就是这个意思"
    re.compile(r"正是我想要的(?:[。！？!?.]|$)"),  # "正是我想要的。"
    re.compile(r"继续保持(?:[。！？!?.]|$)"),  # "继续保持。"
)


# MemoryMiddlewareState 类：定义中间件使用的状态 schema
# 继承自 AgentState
# 这个中间件不在状态中添加额外字段，所以是空的
class MemoryMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    pass


# 模块级函数：从消息中提取纯文本内容
#
# 消息的 content 字段可能是字符串或列表（列表通常是 {type: "text", text: "..."} 格式）
# 这个函数将 content 统一转换成字符串，便于后续处理
#
# 参数：
#   message: 任意消息对象
#
# 返回值：
#   字符串，消息的纯文本内容
def _extract_message_text(message: Any) -> str:
    """Extract plain text from message content for filtering and signal detection."""
    # 获取消息的 content 字段，默认为空字符串
    content = getattr(message, "content", "")

    # 如果 content 是列表（如 [{type: "text", text: "..."}] 格式）
    if isinstance(content, list):
        text_parts: list[str] = []
        # 遍历列表中的每个块
        for part in content:
            if isinstance(part, str):
                # 如果是字符串，直接添加
                text_parts.append(part)
            elif isinstance(part, dict):
                # 如果是字典，尝试获取 text 字段
                text_val = part.get("text")
                if isinstance(text_val, str):
                    text_parts.append(text_val)
        # 将所有文本部分用空格连接
        return " ".join(text_parts)

    # 如果 content 是字符串，直接返回
    return str(content)


# 模块级函数：过滤消息，只保留用于记忆更新的消息
#
# 过滤规则：
#   保留：
#     - 人类消息（用户输入）
#     - 没有 tool_calls 的 AI 消息（最终响应）
#   排除：
#     - 工具消息（中间的工具调用结果）
#     - 有 tool_calls 的 AI 消息（中间步骤，不是最终响应）
#     - <uploaded_files> 块（文件路径是会话作用域的，不能持久化）
#
# 特殊情况处理：
#   如果一个用户消息内容完全由 <uploaded_files> 块组成（剥离后为空）
#   则这个消息和对应的 AI 响应都被跳过
#
# 参数：
#   messages: 完整的消息列表
#
# 返回值：
#   过滤后的消息列表，只包含用户输入和最终 AI 响应
def _filter_messages_for_memory(messages: list[Any]) -> list[Any]:
    """Filter messages to keep only user inputs and final assistant responses.

    This filters out:
    - Tool messages (intermediate tool call results)
    - AI messages with tool_calls (intermediate steps, not final responses)
    - The <uploaded_files> block injected by UploadsMiddleware into human messages
      (file paths are session-scoped and must not persist in long-term memory).
      The user's actual question is preserved; only turns whose content is entirely
      the upload block (nothing remains after stripping) are dropped along with
      their paired assistant response.

    Only keeps:
    - Human messages (with the ephemeral upload block removed)
    - AI messages without tool_calls (final assistant responses), unless the
      paired human turn was upload-only and had no real user text.

    Args:
        messages: List of all conversation messages.

    Returns:
        Filtered list containing only user inputs and final assistant responses.
    """
    filtered = []  # 过滤后的消息列表
    skip_next_ai = False  # 是否跳过下一个 AI 消息（用于处理上传专用消息）

    for msg in messages:
        # 获取消息类型
        msg_type = getattr(msg, "type", None)

        # 处理人类消息
        if msg_type == "human":
            # 提取消息的文本内容
            content_str = _extract_message_text(msg)
            # 检查是否包含上传文件块
            if "<uploaded_files>" in content_str:
                # 去除上传文件块，只保留用户的真实问题
                stripped = _UPLOAD_BLOCK_RE.sub("", content_str).strip()
                if not stripped:
                    # 内容完全由上传文件块组成（剥离后为空）
                    # 这意味着整个消息只是上传文件的 bookkeeping
                    # 跳过这个消息，同时也要跳过对应的 AI 响应
                    skip_next_ai = True
                    continue
                # 有真实用户问题，复制消息并更新内容
                from copy import copy
                clean_msg = copy(msg)
                clean_msg.content = stripped
                filtered.append(clean_msg)
                skip_next_ai = False
            else:
                # 没有上传文件块，直接添加
                filtered.append(msg)
                skip_next_ai = False

        # 处理 AI 消息
        elif msg_type == "ai":
            # 获取 tool_calls 字段
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                # 没有 tool_calls，说明是最终响应
                if skip_next_ai:
                    # 上一个用户消息只是上传 bookkeeping，跳过这个 AI 响应
                    skip_next_ai = False
                    continue
                # 添加最终响应
                filtered.append(msg)

    return filtered


# 模块级函数：检测对话中的"纠错信号"
#
# 纠错信号：当用户明确指出 agent 的错误时触发
# 检测方法：在最近 6 条用户消息中搜索纠错正则模式
#
# 参数：
#   messages: 过滤后的消息列表
#
# 返回值：
#   bool，True 表示检测到纠错信号
def detect_correction(messages: list[Any]) -> bool:
    """Detect explicit user corrections in recent conversation turns.

    The queue keeps only one pending context per thread, so callers pass the
    latest filtered message list. Checking only recent user turns keeps signal
    detection conservative while avoiding stale corrections from long histories.
    """
    # 只检查最近 6 条用户消息（避免过时的纠错）
    recent_user_msgs = [msg for msg in messages[-6:] if getattr(msg, "type", None) == "human"]

    for msg in recent_user_msgs:
        # 提取消息文本并去除首尾空白
        content = _extract_message_text(msg).strip()
        if not content:
            continue
        # 检查是否匹配任何纠错模式
        if any(pattern.search(content) for pattern in _CORRECTION_PATTERNS):
            return True

    return False


# 模块级函数：检测对话中的"正向强化信号"
#
# 强化信号：当用户确认 agent 的做法正确或有帮助时触发
# 这与 detect_correction 互补，检测什么方法有效
#
# 参数：
#   messages: 过滤后的消息列表
#
# 返回值：
#   bool，True 表示检测到强化信号
def detect_reinforcement(messages: list[Any]) -> bool:
    """Detect explicit positive reinforcement signals in recent conversation turns.

    Complements detect_correction() by identifying when the user confirms the
    agent's approach was correct. This allows the memory system to record what
    worked well, not just what went wrong.

    The queue keeps only one pending context per thread, so callers pass the
    latest filtered message list. Checking only recent user turns keeps signal
    detection conservative while avoiding stale signals from long histories.
    """
    # 只检查最近 6 条用户消息
    recent_user_msgs = [msg for msg in messages[-6:] if getattr(msg, "type", None) == "human"]

    for msg in recent_user_msgs:
        content = _extract_message_text(msg).strip()
        if not content:
            continue
        # 检查是否匹配任何强化模式
        if any(pattern.search(content) for pattern in _REINFORCEMENT_PATTERNS):
            return True

    return False


# MemoryMiddleware 类：记忆机制中间件
#
# 工作流程：
#   1. 在 agent 执行完成后（after_agent），将对话加入记忆更新队列
#   2. 队列使用防抖机制，延迟 30 秒后才实际执行更新（避免频繁更新）
#   3. 队列会合并短时间内多次更新请求
#   4. MemoryUpdater 异步调用 LLM 提取记忆信息
#
# 关键设计：
#   - 只包含用户输入和最终 AI 响应（不包括工具调用过程）
#   - 检测纠错和强化信号，影响记忆更新的方式
#   - <uploaded_files> 块会被去除（文件路径是会话作用域的）
class MemoryMiddleware(AgentMiddleware[MemoryMiddlewareState]):
    """Middleware that queues conversation for memory update after agent execution.

    This middleware:
    1. After each agent execution, queues the conversation for memory update
    2. Only includes user inputs and final assistant responses (ignores tool calls)
    3. The queue uses debouncing to batch multiple updates together
    4. Memory is updated asynchronously via LLM summarization
    """

    # state_schema 类变量，指定该中间件使用的状态类型
    state_schema = MemoryMiddlewareState

    # 构造函数
    #
    # 参数：
    #   agent_name: 可选的 agent 名称
    #               如果提供，记忆按 agent 分离存储
    #               如果为 None，使用全局记忆
    def __init__(self, agent_name: str | None = None):
        """Initialize the MemoryMiddleware.

        Args:
            agent_name: If provided, memory is stored per-agent. If None, uses global memory.
        """
        # 调用父类构造函数
        super().__init__()
        # 保存 agent_name
        self._agent_name = agent_name

    # after_agent 钩子方法：在 agent 执行完成后被调用
    #
    # 工作流程：
    #   1. 检查记忆系统是否启用
    #   2. 获取 thread_id（用于标识记忆的归属）
    #   3. 获取消息列表并过滤
    #   4. 检测纠错和强化信号
    #   5. 将对话加入记忆更新队列
    #
    # 参数：
    #   state: 当前 agent 状态（MemoryMiddlewareState）
    #   runtime: LangGraph 运行时上下文
    #
    # 返回值：
    #   总是返回 None（这个中间件不修改状态）
    @override
    def after_agent(self, state: MemoryMiddlewareState, runtime: Runtime) -> dict | None:
        """Queue conversation for memory update after agent completes.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            None (no state changes needed from this middleware).
        """
        # 获取记忆系统配置
        config = get_memory_config()
        # 如果记忆系统未启用，直接返回
        if not config.enabled:
            return None

        # 获取 thread_id
        # 首先尝试从 runtime.context 获取
        thread_id = runtime.context.get("thread_id") if runtime.context else None
        if thread_id is None:
            # 如果没有，尝试从 LangGraph 全局配置获取
            config_data = get_config()
            thread_id = config_data.get("configurable", {}).get("thread_id")
        if not thread_id:
            # 仍然没有 thread_id，记录日志并跳过
            logger.debug("No thread_id in context, skipping memory update")
            return None

        # 从状态中获取消息列表
        messages = state.get("messages", [])
        if not messages:
            logger.debug("No messages in state, skipping memory update")
            return None

        # 过滤消息，只保留用户输入和最终 AI 响应
        filtered_messages = _filter_messages_for_memory(messages)

        # 检查是否有足够的消息（至少一条用户消息和一条 AI 响应）
        user_messages = [m for m in filtered_messages if getattr(m, "type", None) == "human"]
        assistant_messages = [m for m in filtered_messages if getattr(m, "type", None) == "ai"]

        if not user_messages or not assistant_messages:
            # 没有足够的对话内容，跳过
            return None

        # 检测纠错和强化信号
        # 注意：纠错检测优先，如果检测到纠错就不检测强化
        correction_detected = detect_correction(filtered_messages)
        reinforcement_detected = not correction_detected and detect_reinforcement(filtered_messages)

        # 获取记忆更新队列（防抖队列）
        queue = get_memory_queue()
        # 将对话加入队列，延迟执行
        # 队列会在 debounce_seconds（默认 30 秒）后调用 MemoryUpdater
        queue.add(
            thread_id=thread_id,  # 线程 ID，标识记忆的归属
            messages=filtered_messages,  # 过滤后的对话消息
            agent_name=self._agent_name,  # 可选的 agent 名称
            correction_detected=correction_detected,  # 是否检测到纠错
            reinforcement_detected=reinforcement_detected,  # 是否检测到强化
        )

        return None