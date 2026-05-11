"""ClarificationMiddleware - 澄清请求拦截中间件。

功能概述：
  拦截 agent 的澄清请求，中断执行并呈现问题给用户，等待用户响应。

工作流程：
  1. 包装工具调用（wrap_tool_call / awrap_tool_call）
  2. 检测到 ask_clarification 工具调用
  3. 从参数中提取澄清问题、类型、上下文、选项等信息
  4. 格式化为用户友好的消息
  5. 返回 Command(goto=END)，中断执行并跳转结束
  6. 前端检测到 ask_clarification 工具消息，显示给用户
  7. 用户响应后，执行继续

澄清类型与图标：
  - missing_info（❓）：缺失信息
  - ambiguous_requirement（🤔）：需求模糊
  - approach_choice（🔀）：方法选择
  - risk_confirmation（⚠️）：风险确认
  - suggestion（💡）：建议

实现考虑：
  使用 HumanMessage 而不是 SystemMessage 注入消息，
  因为某些模型（如 Anthropic）要求系统消息只在对话开始时出现。
  使用 Command(goto=END) 而非简单的 ToolMessage，
  确保执行流能够正确中断和恢复。

执行位置：中间件链最后一位（最后执行，拦截所有澄清请求）。
"""
"""Middleware for intercepting clarification requests and presenting them to the user."""

# 导入标准库 json，用于解析 JSON 字符串（有些模型将数组序列化为字符串）
import json
# 导入标准库 logging，用于记录日志
import logging
# collections.abc 导入 Callable（可调用对象）
from collections.abc import Callable
# typing 导入 override（方法重写标记）
from typing import override

# 从 langchain.agents 导入 AgentState（agent 基础状态类）
from langchain.agents import AgentState
# 从 langchain.agents.middleware 导入 AgentMiddleware（中间件基类）
from langchain.agents.middleware import AgentMiddleware
# 从 langchain_core.messages 导入 ToolMessage（工具消息）
# 用于创建包含澄清问题的 ToolMessage
from langchain_core.messages import ToolMessage
# 从 langgraph.graph 导入 END
# END 是 LangGraph 的特殊节点，表示图执行结束
# 用于 Command 中断执行，跳转到 END
from langgraph.graph import END
# 从 langgraph.prebuilt.tool_node 导入 ToolCallRequest
# 这是工具调用的请求对象，包含 tool_call（工具调用信息）
from langgraph.prebuilt.tool_node import ToolCallRequest
# 从 langgraph.types 导入 Command
# Command 是 LangGraph 的命令类型，用于控制执行流程
from langgraph.types import Command

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)


# ClarificationMiddlewareState 类：定义中间件使用的状态 schema
# 继承自 AgentState
# 这个中间件不在状态中添加额外字段，所以是空的
class ClarificationMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    pass


# ClarificationMiddleware 类：拦截澄清请求并呈现给用户
#
# 工作流程：
#   1. 当模型调用 `ask_clarification` 工具时，中间件拦截这个调用
#   2. 从工具参数中提取澄清问题和相关元数据
#   3. 格式化成一个用户友好的消息
#   4. 返回一个 Command，执行以下操作：
#      a. 将格式化后的消息作为 ToolMessage 添加到消息历史
#      b. 中断执行，跳转到 END（结束）
#   5. 前端检测到 ask_clarification 工具消息，显示给用户
#   6. 用户响应后，执行继续
#
# 这个中间件替代了之前"工具执行后继续对话流"的方式
# 现在是"工具调用时立即中断，等待用户响应"
class ClarificationMiddleware(AgentMiddleware[ClarificationMiddlewareState]):
    """Intercepts clarification tool calls and interrupts execution to present questions to the user.

    When the model calls the `ask_clarification` tool, this middleware:
    1. Intercepts the tool call before execution
    2. Extracts the clarification question and metadata
    3. Formats a user-friendly message
    4. Returns a Command that interrupts execution and presents the question
    5. Waits for user response before continuing

    This replaces the tool-based approach where clarification continued the conversation flow.
    """

    # state_schema 类变量，指定该中间件使用的状态类型
    state_schema = ClarificationMiddlewareState

    # 内部方法：检查文本是否包含中文字符
    #
    # 中文字符的 Unicode 范围是 \u4e00 到 \u9fff
    # 这个方法用于检测语言环境，决定消息格式
    #
    # 参数：
    #   text: 要检查的文本
    #
    # 返回值：
    #   bool，如果包含中文字符返回 True
    def _is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters.

        Args:
            text: Text to check

        Returns:
            True if text contains Chinese characters
        """
        # 遍历文本中的每个字符
        # 检查是否在中文 Unicode 范围内
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    # 内部方法：格式化澄清消息
    #
    # 将工具调用参数中的澄清信息格式化成用户友好的消息
    #
    # 参数：
    #   args: 工具调用参数，包含 question、clarification_type、context、options 等
    #
    # 返回值：
    #   格式化后的字符串
    def _format_clarification_message(self, args: dict) -> str:
        """Format the clarification arguments into a user-friendly message.

        Args:
            args: The tool call arguments containing clarification details

        Returns:
            Formatted message string
        """
        # 从参数中提取各个字段
        question = args.get("question", "")  # 澄清问题
        clarification_type = args.get("clarification_type", "missing_info")  # 类型
        context = args.get("context")  # 上下文背景
        options = args.get("options", [])  # 选项列表

        # 兼容处理：有些模型（如 Qwen3-Max）将数组参数序列化为 JSON 字符串
        # 需要反序列化并标准化为列表
        if isinstance(options, str):
            try:
                # 尝试将 JSON 字符串解析为列表
                options = json.loads(options)
            except (json.JSONDecodeError, TypeError):
                # 解析失败，转换成单元素列表
                options = [options]

        # 确保 options 是列表
        if options is None:
            options = []
        elif not isinstance(options, list):
            # 非列表类型，转换成单元素列表
            options = [options]

        # 根据澄清类型选择图标
        # 类型映射到 emoji 图标，用于视觉区分
        type_icons = {
            "missing_info": "❓",          # 缺失信息
            "ambiguous_requirement": "🤔",  # 模糊需求
            "approach_choice": "🔀",        # 方法选择
            "risk_confirmation": "⚠️",       # 风险确认
            "suggestion": "💡",             # 建议
        }
        # 获取对应类型的图标，如果找不到则使用默认的 ❓
        icon = type_icons.get(clarification_type, "❓")

        # 构建消息内容
        message_parts = []

        # 如果有上下文，先显示上下文作为背景
        if context:
            message_parts.append(f"{icon} {context}")
            message_parts.append(f"\n{question}")
        else:
            # 没有上下文，直接显示问题和图标
            message_parts.append(f"{icon} {question}")

        # 如果有选项，添加选项列表
        if options and len(options) > 0:
            message_parts.append("")  # 空行用于分隔
            # 为每个选项添加编号
            for i, option in enumerate(options, 1):
                message_parts.append(f"  {i}. {option}")

        # 将所有部分合并成单个字符串
        return "\n".join(message_parts)

    # 内部方法：处理澄清请求并返回中断命令
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求
    #
    # 返回值：
    #   Command，包含状态更新和跳转指令
    def _handle_clarification(self, request: ToolCallRequest) -> Command:
        """Handle clarification request and return command to interrupt execution.

        Args:
            request: Tool call request

        Returns:
            Command that interrupts execution with the formatted clarification message
        """
        # 从请求中提取澄清参数
        args = request.tool_call.get("args", {})
        question = args.get("question", "")

        # 记录日志
        logger.info("Intercepted clarification request")
        logger.debug("Clarification question: %s", question)

        # 格式化澄清消息
        formatted_message = self._format_clarification_message(args)

        # 获取工具调用 ID
        tool_call_id = request.tool_call.get("id", "")

        # 创建包含格式化消息的 ToolMessage
        # 这个消息会被添加到消息历史中
        tool_message = ToolMessage(
            content=formatted_message,  # 格式化的澄清消息
            tool_call_id=tool_call_id,  # 与原始工具调用对应
            name="ask_clarification",    # 工具名称
        )

        # 返回 Command，执行以下操作：
        #   1. update: 添加 ToolMessage 到消息历史
        #   2. goto: 跳转到 END（结束执行），实现中断
        #
        # 注意：这里不添加额外的 AIMessage
        # 前端会直接检测并显示 ask_clarification 工具消息
        return Command(
            update={"messages": [tool_message]},  # 更新消息历史
            goto=END,  # 跳转到 END，中断执行
        )

    # wrap_tool_call 钩子方法：同步版本的工具调用包装
    #
    # 工作流程：
    #   1. 检查是否是 ask_clarification 工具调用
    #   2. 如果是，拦截并调用 _handle_clarification
    #   3. 如果不是，正常执行 handler(request)
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求
    #   handler: 原始的工具执行处理器（同步函数）
    #
    # 返回值：
    #   ToolMessage | Command
    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept ask_clarification tool calls and interrupt execution (sync version).

        Args:
            request: Tool call request
            handler: Original tool execution handler

        Returns:
            Command that interrupts execution with the formatted clarification message
        """
        # 检查工具名称是否是 ask_clarification
        if request.tool_call.get("name") != "ask_clarification":
            # 不是澄清工具调用，正常执行
            return handler(request)

        # 是澄清工具调用，拦截处理
        return self._handle_clarification(request)

    # awrap_tool_call 钩子方法：异步版本的工具调用包装
    #
    # 功能与 wrap_tool_call 相同，但支持异步 handler
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求
    #   handler: 异步的工具执行处理器（返回 Awaitable）
    #
    # 返回值：
    #   ToolMessage | Command
    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Intercept ask_clarification tool calls and interrupt execution (async version).

        Args:
            request: Tool call request
            handler: Original tool execution handler (async)

        Returns:
            Command that interrupts execution with the formatted clarification message
        """
        # 检查工具名称
        if request.tool_call.get("name") != "ask_clarification":
            # 不是澄清工具调用，正常执行
            return await handler(request)

        # 是澄清工具调用，拦截处理
        return self._handle_clarification(request)