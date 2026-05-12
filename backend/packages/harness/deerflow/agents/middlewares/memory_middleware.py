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

# ============================================================
# 导入标准库
# ============================================================

# logging：标准库日志模块，用于记录中间件运行日志
import logging

# re：标准库正则表达式模块，用于匹配纠错和强化信号
# 以及去除消息中的 <uploaded_files> 块
import re

# typing 导入：
#   - Any：任意类型，用于消息处理的类型注解
#   - override：方法重写标记，用于明确表示重写父类方法
from typing import Any, override

# ============================================================
# 导入 LangChain / LangGraph 相关模块
# ============================================================

# langchain.agents.AgentState：
#   LangChain agent 的基础状态类，所有自定义状态类继承自此
from langchain.agents import AgentState

# langchain.agents.middleware.AgentMiddleware：
#   LangChain 的中间件基类，所有自定义中间件必须继承此类
from langchain.agents.middleware import AgentMiddleware

# langgraph.config.get_config：
#   获取 LangGraph 的运行时配置
#   用于在无法从 runtime.context 获取 thread_id 时作为后备
from langgraph.config import get_config

# langgraph.runtime.Runtime：
#   LangGraph 运行时上下文
#   在钩子方法中作为参数传入
from langgraph.runtime import Runtime

# ============================================================
# 导入 DeerFlow 项目内部模块
# ============================================================

# deerflow.agents.memory.queue.get_memory_queue：
#   获取记忆更新队列（MemoryUpdateQueue 实例）
#   记忆更新不会立即执行，而是延迟等待（防抖机制）
#   来自：本项目 packages/harness/deerflow/agents/memory/queue.py
from deerflow.agents.memory.queue import get_memory_queue

# deerflow.config.memory_config.get_memory_config：
#   获取记忆系统的配置（如是否启用、debounce 秒数等）
#   来自：本项目 packages/harness/deerflow/config/memory_config.py
from deerflow.config.memory_config import get_memory_config

# ============================================================
# 模块级变量初始化
# ============================================================

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)

# ============================================================
# 模块级正则表达式和模式定义
# ============================================================

# _UPLOAD_BLOCK_RE：用于匹配和去除消息中的 <uploaded_files> 块
#
# 这个块是由 UploadsMiddleware 注入的，包含文件列表信息
# 文件路径是会话作用域的，不能持久化到长期记忆
#
# re.IGNORECASE：匹配时不区分大小写
# [\s\S]*?：非贪婪匹配任意字符（包括换行）
# \n*：匹配零个或多个换行符
_UPLOAD_BLOCK_RE = re.compile(r"<uploaded_files>[\s\S]*?</uploaded_files>\n*", re.IGNORECASE)

# _CORRECTION_PATTERNS：检测"纠错信号"的正则表达式元组
#
# 当用户明确指出 agent 的错误时触发（如 "that's wrong", "你理解错了" 等中英文表达）
# 用于让记忆系统知道这次对话中有错误被纠正
_CORRECTION_PATTERNS = (
    # 英文纠错模式
    re.compile(r"\bthat(?:'s| is) (?:wrong|incorrect)\b", re.IGNORECASE),
    re.compile(r"\byou misunderstood\b", re.IGNORECASE),
    re.compile(r"\btry again\b", re.IGNORECASE),
    re.compile(r"\bredo\b", re.IGNORECASE),
    # 中文纠错模式
    re.compile(r"不对"),
    re.compile(r"你理解错了"),
    re.compile(r"你理解有误"),
    re.compile(r"重试"),
    re.compile(r"重新来"),
    re.compile(r"换一种"),
    re.compile(r"改用"),
)

# _REINFORCEMENT_PATTERNS：检测"正向强化信号"的正则表达式元组
#
# 当用户确认 agent 的做法正确或有帮助时触发
# 这让记忆系统记录什么方法有效，而不仅仅是记录错误
_REINFORCEMENT_PATTERNS = (
    # 英文强化模式
    re.compile(r"\byes[,.]?\s+(?:exactly|perfect|that(?:'s| is) (?:right|correct|it))\b", re.IGNORECASE),
    re.compile(r"\bperfect(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bexactly\s+(?:right|correct)\b", re.IGNORECASE),
    re.compile(r"\bthat(?:'s| is)\s+(?:exactly\s+)?(?:right|correct|what i (?:wanted|needed|meant))\b", re.IGNORECASE),
    re.compile(r"\bkeep\s+(?:doing\s+)?that\b", re.IGNORECASE),
    re.compile(r"\bjust\s+(?:like\s+)?(?:that|this)\b", re.IGNORECASE),
    re.compile(r"\bthis is (?:great|helpful)\b(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bthis is what i wanted\b(?:[.!?]|$)", re.IGNORECASE),
    # 中文强化模式
    re.compile(r"对[，,]?\s*就是这样(?:[。！？!?.]|$)"),
    re.compile(r"完全正确(?:[。！？!?.]|$)"),
    re.compile(r"(?:对[，,]?\s*)?就是这个意思(?:[。！？!?.]|$)"),
    re.compile(r"正是我想要的(?:[。！？!?.]|$)"),
    re.compile(r"继续保持(?:[。！？!?.]|$)"),
)


# ============================================================
# 状态类型定义
# ============================================================

# MemoryMiddlewareState 类：定义中间件使用的状态 schema
#
# 作用说明：
#   继承自 AgentState，作为 MemoryMiddleware 的状态类型。
#   这个中间件不在状态中添加额外字段，所以类是空的（只有 pass）。
#   定义这个类只是为了满足 LangChain AgentMiddleware 的接口要求。
class MemoryMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    # pass 表示没有额外的状态字段需要定义
    pass


# ============================================================
# 模块级辅助函数
# ============================================================

# _extract_message_text：从消息中提取纯文本内容
#
# 方法作用：
#   将消息的 content 字段（可能是字符串或列表）统一转换为纯文本字符串。
#   用于后续的过滤和信号检测。
#
# 参数：
#   message: 任意消息对象
#
# 返回值：
#   str：消息的纯文本内容
#
# 特殊情况：
#   content 可能是 [{type: "text", text: "..."}] 格式的列表
def _extract_message_text(message: Any) -> str:
    """Extract plain text from message content for filtering and signal detection."""
    # getattr(message, "content", "") 获取消息的 content 字段，默认为空字符串
    content = getattr(message, "content", "")

    # 如果 content 是列表（如 [{type: "text", text: "..."}] 格式）
    if isinstance(content, list):
        # text_parts 存储提取的文本部分
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


# _filter_messages_for_memory：过滤消息，只保留用于记忆更新的消息
#
# 方法作用：
#   从完整消息列表中过滤出对记忆更新有用的消息。
#   排除工具调用过程和会话作用域的内容。
#
# 参数：
#   messages: list[Any]，完整的消息列表
#
# 返回值：
#   list[Any]：过滤后的消息列表
#
# 过滤规则：
#   保留：
#     - 人类消息（用户输入）
#     - 没有 tool_calls 的 AI 消息（最终响应）
#   排除：
#     - ToolMessage（工具调用结果）
#     - 有 tool_calls 的 AI 消息（中间步骤）
#     - <uploaded_files> 块（文件路径是会话作用域的）
#
# 特殊情况处理：
#   如果一个用户消息内容完全由 <uploaded_files> 块组成，
#   则这个消息和对应的 AI 响应都被跳过
def _filter_messages_for_memory(messages: list[Any]) -> list[Any]:
    """Filter messages to keep only user inputs and final assistant responses.

    This filters out:
    - Tool messages (intermediate tool call results)
    - AI messages with tool_calls (intermediate steps, not final responses)
    - The <uploaded_files> block injected by UploadsMiddleware (file paths are session-scoped)
    """
    # filtered 存储过滤后的消息列表
    filtered: list[Any] = []
    # skip_next_ai 标记是否跳过下一个 AI 消息
    skip_next_ai = False

    # 遍历所有消息
    for msg in messages:
        # 获取消息类型
        msg_type = getattr(msg, "type", None)

        # ---- 处理人类消息 ----
        if msg_type == "human":
            # 提取消息的文本内容
            content_str = _extract_message_text(msg)

            # 检查是否包含上传文件块
            if "<uploaded_files>" in content_str:
                # _UPLOAD_BLOCK_RE.sub("", content_str) 去除上传文件块
                # .strip() 去除首尾空白
                stripped = _UPLOAD_BLOCK_RE.sub("", content_str).strip()
                if not stripped:
                    # 内容完全由上传文件块组成（剥离后为空）
                    # 跳过这个消息，同时也要跳过对应的 AI 响应
                    skip_next_ai = True
                    continue
                # 有真实用户问题
                # 使用 copy 复制消息，避免修改原始消息
                from copy import copy
                clean_msg = copy(msg)
                # 更新消息内容为去除上传文件块后的内容
                clean_msg.content = stripped
                filtered.append(clean_msg)
                skip_next_ai = False
            else:
                # 没有上传文件块，直接添加
                filtered.append(msg)
                skip_next_ai = False

        # ---- 处理 AI 消息 ----
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


# detect_correction：检测对话中的"纠错信号"
#
# 方法作用：
#   在最近的用户消息中搜索纠错正则模式。
#   当用户明确指出 agent 的错误时触发。
#
# 参数：
#   messages: list[Any]，过滤后的消息列表
#
# 返回值：
#   bool：True 表示检测到纠错信号
#
# 注意：
#   只检查最近 6 条用户消息，避免过时的纠错影响记忆更新
def detect_correction(messages: list[Any]) -> bool:
    """Detect explicit user corrections in recent conversation turns."""
    # 只检查最近 6 条用户消息
    # messages[-6:] 获取最后 6 条消息
    # [msg for msg in ... if getattr(msg, "type", None) == "human"] 过滤出用户消息
    recent_user_msgs = [msg for msg in messages[-6:] if getattr(msg, "type", None) == "human"]

    for msg in recent_user_msgs:
        # 提取消息文本并去除首尾空白
        content = _extract_message_text(msg).strip()
        if not content:
            continue
        # any(pattern.search(content) for pattern in _CORRECTION_PATTERNS)
        # 检查是否匹配任何纠错模式
        if any(pattern.search(content) for pattern in _CORRECTION_PATTERNS):
            return True

    return False


# detect_reinforcement：检测对话中的"正向强化信号"
#
# 方法作用：
#   在最近的用户消息中搜索强化正则模式。
#   当用户确认 agent 的做法正确或有帮助时触发。
#
# 参数：
#   messages: list[Any]，过滤后的消息列表
#
# 返回值：
#   bool：True 表示检测到强化信号
#
# 注意：
#   与 detect_correction 互补，检测什么方法有效
def detect_reinforcement(messages: list[Any]) -> bool:
    """Detect explicit positive reinforcement signals in recent conversation turns."""
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


# ============================================================
# MemoryMiddleware 主类
# ============================================================

# MemoryMiddleware 类：记忆机制中间件
#
# 核心作用：
#   在每次 agent 执行完成后（after_agent），
#   将对话内容加入记忆更新队列，实现长期记忆。
#
# 工作流程：
#   1. 在 after_agent 钩子中触发
#   2. 检查记忆系统是否启用
#   3. 过滤消息，只保留用户输入和最终 AI 响应
#   4. 检测纠错和强化信号
#   5. 将对话加入 MemoryUpdateQueue（防抖队列）
#
# 关键设计：
#   - 只包含用户输入和最终 AI 响应（不包括工具调用过程）
#   - 检测纠错和强化信号，影响记忆更新的方式
#   - <uploaded_files> 块会被去除（文件路径是会话作用域的）
#   - 队列使用防抖机制，延迟 30 秒后执行更新
class MemoryMiddleware(AgentMiddleware[MemoryMiddlewareState]):
    """Middleware that queues conversation for memory update after agent execution.

    This middleware:
    1. After each agent execution, queues the conversation for memory update
    2. Only includes user inputs and final assistant responses (ignores tool calls)
    3. The queue uses debouncing to batch multiple updates together
    4. Memory is updated asynchronously via LLM summarization
    """

    # state_schema：类变量，指定该中间件使用的状态类型
    state_schema = MemoryMiddlewareState

    # 构造函数
    #
    # 参数：
    #   agent_name: str | None，可选的 agent 名称
    #     如果提供，记忆按 agent 分离存储
    #     如果为 None，使用全局记忆
    def __init__(self, agent_name: str | None = None):
        """Initialize the MemoryMiddleware.

        Args:
            agent_name: If provided, memory is stored per-agent. If None, uses global memory.
        """
        # 调用父类构造函数
        super().__init__()
        # 保存 agent_name
        self._agent_name = agent_name

    # ============================================================
    # LangChain AgentMiddleware 钩子方法
    # ============================================================

    # after_agent：agent 执行完成后的钩子
    #
    # 方法作用：
    #   LangChain AgentMiddleware 定义的钩子之一，
    #   在每次 agent 执行完成后同步执行。
    #
    # 调用时机：
    #   当 agent 执行完成时（无论是正常完成还是异常），
    #   这个方法都会被调用。
    #
    # 工作流程：
    #   1. 检查记忆系统是否启用
    #   2. 获取 thread_id（用于标识记忆的归属）
    #   3. 获取消息列表并过滤
    #   4. 检测纠错和强化信号
    #   5. 将对话加入记忆更新队列
    #
    # 参数：
    #   state: MemoryMiddlewareState，当前 agent 状态
    #   runtime: Runtime，LangGraph 运行时上下文
    #
    # 返回值：
    #   dict | None：总是返回 None（这个中间件不修改状态）
    @override
    def after_agent(self, state: MemoryMiddlewareState, runtime: Runtime) -> dict | None:
        """Queue conversation for memory update after agent completes.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            None (no state changes needed from this middleware).
        """
        # get_memory_config() 获取记忆系统配置
        config = get_memory_config()
        # 如果记忆系统未启用，直接返回
        if not config.enabled:
            return None

        # 获取 thread_id
        # 首先尝试从 runtime.context 获取
        # runtime.context 是 LangGraph 传递的上下文信息字典
        thread_id = runtime.context.get("thread_id") if runtime.context else None
        if thread_id is None:
            # 如果 runtime.context 中没有，尝试从 LangGraph 全局配置获取
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
        # getattr(msg, "type", None) 获取消息类型，如果消息没有 type 属性则返回 None
        user_messages = [m for m in filtered_messages if getattr(m, "type", None) == "human"]
        assistant_messages = [m for m in filtered_messages if getattr(m, "type", None) == "ai"]

        if not user_messages or not assistant_messages:
            # 没有足够的对话内容，跳过
            return None

        # 检测纠错和强化信号
        # 注意：纠错检测优先，如果检测到纠错就不检测强化
        correction_detected = detect_correction(filtered_messages)
        reinforcement_detected = not correction_detected and detect_reinforcement(filtered_messages)

        # get_memory_queue() 获取记忆更新队列（防抖队列）
        queue = get_memory_queue()
        # queue.add(...) 将对话加入队列
        # 队列会在 debounce_seconds（默认 30 秒）后调用 MemoryUpdater
        queue.add(
            thread_id=thread_id,
            messages=filtered_messages,
            agent_name=self._agent_name,
            correction_detected=correction_detected,
            reinforcement_detected=reinforcement_detected,
        )

        return None