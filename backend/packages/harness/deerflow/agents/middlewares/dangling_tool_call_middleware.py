"""DanglingToolCallMiddleware - 悬空工具调用修复中间件。

功能概述：
  修复消息历史中的悬空工具调用，确保 LLM 看到格式完整的对话历史。

问题背景：
  当用户中断 agent 执行或请求取消时，可能出现"AIMessage 有 tool_calls
  但没有对应的 ToolMessage"的情况。这是因为中断发生在线程还没来得及
  执行工具并返回结果时。

问题影响：
  这种"悬空的工具调用"会导致 LLM 收到格式不完整的对话历史，引发错误。

解决方案：
  在 wrap_model_call 钩子中扫描消息历史，检测悬空的工具调用，
  并在相应的 AIMessage 后面立即插入一个合成的错误 ToolMessage。

实现细节：
  使用 wrap_model_call 而不是 before_model，因为 before_model + add_messages
  reducer 会在消息列表末尾追加，而不能在正确的位置（悬空的 AIMessage 后面）插入。

执行位置：紧接 SandboxMiddleware 之后，在模型调用前修复消息格式。
"""
"""Middleware to fix dangling tool calls in message history.

A dangling tool call occurs when an AIMessage contains tool_calls but there are
no corresponding ToolMessages in the history (e.g., due to user interruption or
request cancellation). This causes LLM errors due to incomplete message format.

This middleware intercepts the model call to detect and patch such gaps by
inserting synthetic ToolMessages with an error indicator immediately after the
AIMessage that made the tool calls, ensuring correct message ordering.

Note: Uses wrap_model_call instead of before_model to ensure patches are inserted
at the correct positions (immediately after each dangling AIMessage), not appended
to the end of the message list as before_model + add_messages reducer would do.
"""

# ============================================================
# 导入标准库
# ============================================================

# logging：标准库日志模块，用于记录中间件运行日志
import logging

# collections.abc 导入：
#   - Awaitable：异步可等待对象类型，用于异步 handler 的返回类型注解
#   - Callable：可调用对象类型，用于 handler 参数的类型注解
from collections.abc import Awaitable, Callable

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

# langchain.agents.middleware.types 导入中间件相关的类型定义：
#   - ModelCallResult：模型调用的结果（包含请求和响应）
#   - ModelRequest：模型调用的请求（包含 messages、tools 等）
#   - ModelResponse：模型调用的响应
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse

# langchain_core.messages.ToolMessage：
#   工具消息类型，用于创建合成的错误 ToolMessage
#   当工具执行完成后返回此类型的消息
from langchain_core.messages import ToolMessage

# ============================================================
# 模块级变量初始化
# ============================================================

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)


# ============================================================
# DanglingToolCallMiddleware 主类
# ============================================================

# DanglingToolCallMiddleware 类：修复消息历史中的悬空工具调用
#
# 核心作用：
#   在模型调用前扫描消息历史，检测"AIMessage 有 tool_calls 但没有对应 ToolMessage"
#   的情况（悬空工具调用），并插入合成的错误 ToolMessage 来修复格式。
#
# 问题场景：
#   - 用户点击"停止"按钮中断 agent 执行
#   - 请求超时被取消
#   - 任何导致工具调用未能完成的情况
#
# 工作流程：
#   1. 在 wrap_model_call / awrap_model_call 中拦截模型调用
#   2. 扫描所有消息，收集已有 ToolMessage 的 tool_call_id
#   3. 检查是否有 AIMessage 的 tool_calls 在已有列表中不存在
#   4. 如果有悬空的 tool_call，在对应 AIMessage 后插入合成的错误 ToolMessage
#   5. 用修复后的消息列表替换原始请求中的消息列表
#
# 设计考虑：
#   - 使用 wrap_model_call 而不是 before_model，因为后者与 add_messages reducer
#     配合会在列表末尾追加消息，而不能在正确位置（悬空 AIMessage 后）插入
#   - 合成的 ToolMessage 包含 status="error"，让 LLM 知道这个调用失败了
class DanglingToolCallMiddleware(AgentMiddleware[AgentState]):
    """Inserts placeholder ToolMessages for dangling tool calls before model invocation.

    Scans the message history for AIMessages whose tool_calls lack corresponding
    ToolMessages, and injects synthetic error responses immediately after the
    offending AIMessage so the LLM receives a well-formed conversation.
    """

    # state_schema：类变量，指定该中间件使用的状态类型
    # 这个中间件不在状态中添加额外字段，所以使用基础的 AgentState
    state_schema = AgentState

    # ============================================================
    # 内部辅助方法
    # ============================================================

    # _build_patched_messages：构建打了补丁的消息列表
    #
    # 方法作用：
    #   扫描消息历史，检测悬空的工具调用，并插入合成的错误 ToolMessage。
    #
    # 参数：
    #   messages: list，原始消息列表
    #
    # 返回值：
    #   list | None：如果有补丁，返回修复后的新消息列表；如果没有悬空工具调用，返回 None
    #
    # 工作流程：
    #   1. 遍历所有消息，收集所有已有 ToolMessage 的 tool_call_id
    #   2. 检查是否有 AIMessage 的 tool_calls 在已有列表中不存在（悬空的）
    #   3. 如果有悬空的 tool_call，创建合成的错误 ToolMessage
    #   4. 将这些合成消息插入到正确的位置（AIMessage 之后）
    def _build_patched_messages(self, messages: list) -> list | None:
        # ---- 第一步：收集所有已有的 ToolMessage 的 tool_call_id ----
        #
        # set 用于快速查找，存储所有已有的 tool_call_id
        # 这样可以 O(1) 地判断一个 tool_call_id 是否有对应的 ToolMessage
        existing_tool_msg_ids: set[str] = set()
        for msg in messages:
            # isinstance(msg, ToolMessage) 检查消息是否是 ToolMessage 类型
            if isinstance(msg, ToolMessage):
                # ToolMessage.tool_call_id 标识它对应的工具调用
                existing_tool_msg_ids.add(msg.tool_call_id)

        # ---- 第二步：检查是否有悬空的工具调用需要补丁 ----
        needs_patch = False  # 标记是否需要打补丁
        for msg in messages:
            # getattr(msg, "type", None) 获取消息类型，如果消息没有 type 属性则返回 None
            if getattr(msg, "type", None) != "ai":
                continue  # 不是 AI 消息，跳过

            # 获取消息的 tool_calls 字段
            # tool_calls 是列表，每个元素是一个工具调用（包含 id、name 等）
            # getattr(msg, "tool_calls", None) 如果没有 tool_calls 字段返回 None
            # or [] 确保结果是列表而不是 None
            for tc in getattr(msg, "tool_calls", None) or []:
                # tc 是字典，tc.get("id") 获取工具调用的 ID
                tc_id = tc.get("id")
                # 如果这个 ID 不在已有的 ToolMessage IDs 中，说明是悬空的
                if tc_id and tc_id not in existing_tool_msg_ids:
                    needs_patch = True  # 标记需要打补丁
                    break  # 找到一个悬空的就够了，退出内层循环
            if needs_patch:
                break  # 已确定需要补丁，退出外层循环

        # 如果不需要补丁，直接返回 None（不修改消息列表）
        if not needs_patch:
            return None

        # ---- 第三步：构建打补丁后的消息列表 ----
        patched: list = []  # 存储修复后的消息列表
        patched_ids: set[str] = set()  # 记录已处理的 tool_call_id，避免重复插入
        patch_count = 0  # 统计打了多少补丁

        for msg in messages:
            # 首先添加原始消息到结果列表
            patched.append(msg)

            # 如果不是 AI 消息，跳过（不需要检查 tool_calls）
            if getattr(msg, "type", None) != "ai":
                continue

            # 遍历 AI 消息中的所有 tool_calls
            for tc in getattr(msg, "tool_calls", None) or []:
                tc_id = tc.get("id")
                # 检查是否需要创建合成消息（同时满足三个条件）：
                #   1. tc_id 存在（非空）
                #   2. 不在已有的 ToolMessage IDs 中（悬空的）
                #   3. 还没处理过（避免重复插入）
                if tc_id and tc_id not in existing_tool_msg_ids and tc_id not in patched_ids:
                    # 创建合成的错误 ToolMessage
                    # 使用 ToolMessage 构造函数，参数包括：
                    patched.append(
                        ToolMessage(
                            # content：错误消息内容，告知 LLM 这个调用被中断了
                            content="[Tool call was interrupted and did not return a result.]",
                            # tool_call_id：工具调用 ID，与 AIMessage 中的 tool_call.id 对应
                            tool_call_id=tc_id,
                            # name：工具名称，从原始 tool_call 中获取，默认为 "unknown"
                            name=tc.get("name", "unknown"),
                            # status：状态设为 "error"，标识这是一个错误的响应
                            # 这让 LLM 知道工具调用没有成功完成
                            status="error",
                        )
                    )
                    # 记录已处理的 ID，避免重复
                    patched_ids.add(tc_id)
                    # 计数器 +1
                    patch_count += 1

        # 记录警告日志：插入了多少个占位符 ToolMessage
        # 这是 warning 级别，因为悬空工具调用说明有异常情况发生
        logger.warning(f"Injecting {patch_count} placeholder ToolMessage(s) for dangling tool calls")

        # 返回打补丁后的消息列表
        return patched

    # ============================================================
    # LangChain AgentMiddleware 钩子方法
    # ============================================================

    # wrap_model_call：同步版本的模型调用包装钩子
    #
    # 方法作用：
    #   LangChain AgentMiddleware 提供的扩展点，
    #   在模型调用前后可以对请求和响应进行处理。
    #   这个方法检查并修复悬空的工具调用。
    #
    # 参数：
    #   request: ModelRequest，模型调用请求（包含 messages、tools 等）
    #   handler: Callable[[ModelRequest], ModelResponse]，原始模型调用处理器
    #
    # 返回值：
    #   ModelCallResult，模型调用的结果
    #
    # 工作流程：
    #   1. 调用 _build_patched_messages 检查是否需要补丁
    #   2. 如果需要，用修复后的消息列表替换请求中的消息
    #   3. 调用原始 handler 处理请求
    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        # 调用内部方法检查消息列表是否需要补丁
        patched = self._build_patched_messages(request.messages)
        if patched is not None:
            # 用修复后的消息列表创建新的请求
            # request.override() 创建一个浅拷贝，用 patched 替换原始的 messages
            request = request.override(messages=patched)

        # 调用原始 handler 处理请求，返回 ModelCallResult
        return handler(request)

    # awrap_model_call：异步版本的模型调用包装钩子
    #
    # 方法作用：
    #   与 wrap_model_call 相同，但支持异步 handler。
    #   用于当 agent 以异步方式调用时（await agent.ainvoke()）。
    #
    # 参数：
    #   request: ModelRequest，模型调用请求
    #   handler: Callable[[ModelRequest], Awaitable[ModelResponse]]，异步模型调用处理器
    #
    # 返回值：
    #   ModelCallResult，模型调用的结果
    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        # 调用内部方法检查消息列表是否需要补丁
        patched = self._build_patched_messages(request.messages)
        if patched is not None:
            # 用修复后的消息列表创建新的请求
            request = request.override(messages=patched)

        # 使用 await 调用异步 handler，返回 ModelCallResult
        return await handler(request)
