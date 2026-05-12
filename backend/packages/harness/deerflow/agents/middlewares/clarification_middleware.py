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

# ============================================================
# 导入标准库
# ============================================================

# json：标准库 JSON 模块，用于解析 JSON 字符串
# 有些模型（如 Qwen3-Max）将数组参数序列化为 JSON 字符串，需要反序列化
import json

# logging：标准库日志模块，用于记录中间件运行日志
import logging

# collections.abc.Callable：标准库可调用对象类型，用于类型注解
# handler 参数的类型是 Callable，表示原始的工具执行处理器
from collections.abc import Callable

# typing.override：方法重写标记，用于明确表示重写父类方法
from typing import override

# ============================================================
# 导入 LangChain / LangGraph 相关模块
# ============================================================

# langchain.agents.AgentState：
#   LangChain agent 的基础状态类，所有自定义状态类继承自此
#   定义了中间件需要使用的状态结构（如 messages 列表）
from langchain.agents import AgentState

# langchain.agents.middleware.AgentMiddleware：
#   LangChain 的中间件基类，所有自定义中间件必须继承此类
#   提供了 before_agent、after_agent、wrap_tool_call 等钩子方法
from langchain.agents.middleware import AgentMiddleware

# langchain_core.messages.ToolMessage：
#   工具消息类型，用于创建包含澄清问题的 ToolMessage
#   当工具执行完成后返回此类型的消息
from langchain_core.messages import ToolMessage

# langgraph.graph.END：
#   LangGraph 的特殊节点，表示图执行结束
#   用于 Command 中断执行时跳转到 END，实现流程中断
from langgraph.graph import END

# langgraph.prebuilt.tool_node.ToolCallRequest：
#   LangGraph 预建的工具节点使用的请求对象
#   包含 tool_call（工具调用信息：名称、参数、ID 等）
from langgraph.prebuilt.tool_node import ToolCallRequest

# langgraph.types.Command：
#   LangGraph 的命令类型，用于控制图的执行流程
#   可以指定 goto（跳转）和 update（状态更新）
from langgraph.types import Command

# ============================================================
# 模块级变量初始化
# ============================================================

# 创建模块级 logger，用于记录中间件运行日志
# 使用 __name__ 可以显示日志来源模块名
logger = logging.getLogger(__name__)


# ============================================================
# 状态类型定义
# ============================================================

# ClarificationMiddlewareState 类：定义中间件使用的状态 schema
#
# 作用说明：
#   继承自 AgentState，作为 ClarificationMiddleware 的状态类型。
#   这个中间件不在状态中添加额外字段，所以类是空的（只有 pass）。
#   定义这个类只是为了满足 LangChain AgentMiddleware 的接口要求。
class ClarificationMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    # pass 表示没有额外的状态字段需要定义
    pass


# ============================================================
# ClarificationMiddleware 主类
# ============================================================

# ClarificationMiddleware 类：拦截澄清请求并呈现给用户
#
# 核心作用：
#   当模型调用 ask_clarification 工具时，拦截这个调用并格式化澄清问题，
#   然后通过 Command(goto=END) 中断执行，让前端显示问题给用户，
#   等待用户响应后再继续执行流程。
#
# 工作流程详解：
#   1. 在 wrap_tool_call / awrap_tool_call 中拦截工具调用
#   2. 检查是否是 ask_clarification 工具
#   3. 如果是，从工具参数中提取 question、clarification_type、context、options
#   4. 调用 _format_clarification_message 格式化成用户友好的消息
#   5. 创建 ToolMessage 和 Command(goto=END)
#   6. 前端检测到 ask_clarification 类型的 ToolMessage，显示给用户
#   7. 用户响应后，继续执行（因为已经跳转到 END，所以会从 END 节点继续）
#
# 设计考虑：
#   - 使用 ToolMessage 而不是直接返回文本，因为前端是按照工具调用的流程处理
#   - 使用 Command(goto=END) 而不是普通返回，确保执行流正确中断
#   - 支持中英文的 clarification_type 检测
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

    # state_schema：类变量，指定该中间件使用的状态类型
    # LangChain AgentMiddleware 使用此属性进行状态类型检查和验证
    state_schema = ClarificationMiddlewareState

    # ============================================================
    # 内部辅助方法
    # ============================================================

    # _is_chinese：检查文本是否包含中文字符
    #
    # 方法作用：
    #   检测给定文本中是否包含中文字符。
    #   用于决定消息格式化和图标选择。
    #
    # 参数：
    #   text: str，要检查的文本
    #
    # 返回值：
    #   bool：如果包含中文字符返回 True，否则返回 False
    #
    # 实现逻辑：
    #   Unicode 标准中，中文字符的码点范围是 \u4e00 到 \u9fff
    #   遍历文本中每个字符，检查是否落在这个范围内
    def _is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters.

        Args:
            text: Text to check

        Returns:
            True if text contains Chinese characters
        """
        # any() 检查是否有任意一个字符满足条件
        # 遍历文本中每个字符
        # "\u4e00" <= char <= "\u9fff" 检查字符是否在中文 Unicode 范围内
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    # _format_clarification_message：格式化澄清消息
    #
    # 方法作用：
    #   将 ask_clarification 工具的参数格式化成用户友好的消息。
    #   消息包含澄清问题、上下文（如果有的话）、以及选项列表（如果有的话）。
    #
    # 参数：
    #   args: dict，工具调用参数，包含：
    #     - question: str，澄清问题（必需）
    #     - clarification_type: str，澄清类型，默认为 "missing_info"（可选）
    #     - context: str，上下文背景（可选）
    #     - options: list，选项列表（可选）
    #
    # 返回值：
    #   str：格式化的消息字符串
    #
    # 格式化规则：
    #   1. 根据 clarification_type 选择图标（❓/🤔/🔀/⚠️/💡）
    #   2. 如果有 context，先显示图标和 context，再显示问题
    #   3. 如果有 options，添加编号列表
    def _format_clarification_message(self, args: dict) -> str:
        """Format the clarification arguments into a user-friendly message.

        Args:
            args: The tool call arguments containing clarification details

        Returns:
            Formatted message string
        """
        # args.get(key, default) 获取字典中的值，如果不存在则使用默认值
        # question：澄清问题，默认为空字符串
        question = args.get("question", "")
        # clarification_type：澄清类型，用于选择图标
        clarification_type = args.get("clarification_type", "missing_info")
        # context：上下文背景，帮助用户理解为什么需要澄清
        context = args.get("context")
        # options：选项列表，让用户可以选择
        options = args.get("options", [])

        # ---- 兼容处理：有些模型（如 Qwen3-Max）将数组参数序列化为 JSON 字符串 ----
        # isinstance(options, str) 检查 options 是否是字符串类型
        if isinstance(options, str):
            try:
                # 尝试将 JSON 字符串解析为 Python 列表
                options = json.loads(options)
            except (json.JSONDecodeError, TypeError):
                # 解析失败（无效 JSON），将字符串作为单个选项
                options = [options]

        # ---- 确保 options 是列表类型 ----
        # 如果 options 是 None，设置为空列表
        if options is None:
            options = []
        # 如果 options 不是列表（如整数、字典等），转换成单元素列表
        elif not isinstance(options, list):
            options = [options]

        # ---- 根据澄清类型选择图标 ----
        # type_icons 是字典，key 是澄清类型，value 是 emoji 图标
        type_icons = {
            # missing_info：缺失信息，需要用户提供更多信息
            "missing_info": "❓",
            # ambiguous_requirement：需求模糊，需要澄清
            "ambiguous_requirement": "🤔",
            # approach_choice：方法选择，需要用户决定方法
            "approach_choice": "🔀",
            # risk_confirmation：风险确认，需要用户确认风险
            "risk_confirmation": "⚠️",
            # suggestion：建议，提供建议供用户参考
            "suggestion": "💡",
        }
        # dict.get(key, default) 获取值，如果 key 不存在则返回 default
        # 找不到 clarification_type 时使用默认的 ❓
        icon = type_icons.get(clarification_type, "❓")

        # ---- 构建消息内容 ----
        # message_parts 是列表，用于存储消息的各个部分
        message_parts = []

        # 如果有上下文，先显示上下文作为背景
        if context:
            # f-string 格式化字符串，将图标和上下文组合
            message_parts.append(f"{icon} {context}")
            # 换行后显示问题
            message_parts.append(f"\n{question}")
        else:
            # 没有上下文，直接显示问题和图标
            message_parts.append(f"{icon} {question}")

        # 如果有选项，添加选项列表
        # 选项不为空且长度大于 0 时才添加
        if options and len(options) > 0:
            # 空行用于分隔问题和建议
            message_parts.append("")
            # enumerate(options, 1) 遍历选项，从 1 开始编号
            # i 是编号，option 是选项内容
            for i, option in enumerate(options, 1):
                # 每个选项前面加两个空格和编号
                message_parts.append(f"  {i}. {option}")

        # "\n".join(message_parts) 用换行符连接所有部分
        return "\n".join(message_parts)

    # _handle_clarification：处理澄清请求并返回中断命令
    #
    # 方法作用：
    #   实际处理澄清请求的核心逻辑。
    #   格式化消息并创建返回给 LangGraph 的 Command。
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求
    #     包含 tool_call 字段，其中有所需的 args、id、name 等
    #
    # 返回值：
    #   Command：包含状态更新和跳转指令的命令
    #     - update: 更新消息历史，添加 ToolMessage
    #     - goto: 跳转到 END，停止当前执行
    #
    # 实现逻辑：
    #   1. 从 request.tool_call 中提取参数
    #   2. 记录日志
    #   3. 格式化澄清消息
    #   4. 获取 tool_call_id
    #   5. 创建 ToolMessage
    #   6. 返回 Command(update={"messages": [tool_message]}, goto=END)
    def _handle_clarification(self, request: ToolCallRequest) -> Command:
        """Handle clarification request and return command to interrupt execution.

        Args:
            request: Tool call request

        Returns:
            Command that interrupts execution with the formatted clarification message
        """
        # request.tool_call 是字典，包含工具调用的详细信息
        # args.get("question", "") 获取 question 参数，默认为空字符串
        args = request.tool_call.get("args", {})
        question = args.get("question", "")

        # 记录信息日志：拦截到澄清请求
        logger.info("Intercepted clarification request")
        # 记录调试日志：包含澄清问题内容
        logger.debug("Clarification question: %s", question)

        # 调用 _format_clarification_message 格式化消息
        formatted_message = self._format_clarification_message(args)

        # request.tool_call.get("id", "") 获取工具调用 ID
        # 用于将 ToolMessage 与原始调用关联
        tool_call_id = request.tool_call.get("id", "")

        # 创建 ToolMessage，包含格式化的澄清消息
        # ToolMessage 的参数：
        #   - content: 消息内容（格式化的澄清问题）
        #   - tool_call_id: 与原始工具调用对应的 ID
        #   - name: 工具名称（设为 "ask_clarification"）
        tool_message = ToolMessage(
            content=formatted_message,
            tool_call_id=tool_call_id,
            name="ask_clarification",
        )

        # 返回 Command，执行以下操作：
        #   1. update: 更新 LangGraph 状态，添加 ToolMessage 到 messages
        #   2. goto=END: 跳转到 END 节点，停止当前执行流程
        #
        # 这样前端可以检测到 ask_clarification 类型的 ToolMessage，
        # 显示给用户，等待用户响应后再继续。
        return Command(
            update={"messages": [tool_message]},
            goto=END,
        )

    # ============================================================
    # LangChain AgentMiddleware 钩子方法
    # ============================================================

    # wrap_tool_call：同步版本的工具调用包装钩子
    #
    # 方法作用：
    #   LangChain AgentMiddleware 提供的扩展点，
    #   在工具调用执行前后进行拦截和处理。
    #   这个方法检查是否是 ask_clarification 工具调用，
    #   如果是则拦截并处理，否则正常执行。
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求
    #     包含 tool_call（工具名称、参数、ID 等）
    #   handler: Callable[[ToolCallRequest], ToolMessage | Command]
    #     原始的工具执行处理器，是一个可调用对象
    #
    # 返回值：
    #   ToolMessage | Command：
    #     - 如果是 ask_clarification：返回中断命令
    #     - 否则：返回原始 handler 的执行结果
    #
    # 调用时机：
    #   当 agent 以同步方式调用工具时（agent.invoke()）
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
        # request.tool_call.get("name") 获取工具名称
        # 如果不是 ask_clarification，正常执行 handler
        if request.tool_call.get("name") != "ask_clarification":
            return handler(request)

        # 是 ask_clarification 工具调用，拦截处理
        return self._handle_clarification(request)

    # awrap_tool_call：异步版本的工具调用包装钩子
    #
    # 方法作用：
    #   与 wrap_tool_call 相同，但支持异步 handler。
    #   用于当工具执行是异步操作时。
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求
    #   handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    #     异步的工具执行处理器，返回一个 Awaitable
    #
    # 返回值：
    #   ToolMessage | Command：与同步版本相同
    #
    # 调用时机：
    #   当 agent 以异步方式调用工具时（await agent.ainvoke()）
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
            # 不是 ask_clarification，使用 await 调用异步 handler
            return await handler(request)

        # 是澄清工具调用，拦截处理
        return self._handle_clarification(request)