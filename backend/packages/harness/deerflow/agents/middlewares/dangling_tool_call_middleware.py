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

# 导入标准库 logging，用于记录日志
import logging
# collections.abc 导入 Awaitable（异步可调用）和 Callable（可调用对象）
from collections.abc import Awaitable, Callable
# typing 导入 override（方法重写标记）
from typing import override

# 从 langchain.agents 导入 AgentState（agent 基础状态类）
from langchain.agents import AgentState
# 从 langchain.agents.middleware 导入 AgentMiddleware（中间件基类）
from langchain.agents.middleware import AgentMiddleware
# 导入中间件相关的类型定义：
#   - ModelCallResult: 模型调用的结果（包含请求和响应）
#   - ModelRequest: 模型调用的请求（包含 messages 等）
#   - ModelResponse: 模型调用的响应
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
# 从 langchain_core.messages 导入 ToolMessage（工具消息）
# 用于创建合成的错误 ToolMessage
from langchain_core.messages import ToolMessage

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)


# DanglingToolCallMiddleware 类：修复消息历史中的悬空工具调用
#
# 工作流程：
#   1. 在模型调用前，扫描消息历史中的所有消息
#   2. 找出所有已有的 ToolMessage，记录它们的 tool_call_id
#   3. 检查是否有 AIMessage 的 tool_calls 在这个列表中不存在
#   4. 如果有，为每个悬空的 tool_call 在对应的 AIMessage 后面
#      插入一个合成的错误 ToolMessage
#   5. 用修复后的消息列表替换原始的请求消息列表
#
# 使用场景：
#   - 用户点击"停止"按钮中断 agent 执行
#   - 请求超时被取消
#   - 任何导致工具调用未能完成的情况
class DanglingToolCallMiddleware(AgentMiddleware[AgentState]):
    """Inserts placeholder ToolMessages for dangling tool calls before model invocation.

    Scans the message history for AIMessages whose tool_calls lack corresponding
    ToolMessages, and injects synthetic error responses immediately after the
    offending AIMessage so the LLM receives a well-formed conversation.
    """

    # 内部方法：构建打了补丁的消息列表
    #
    # 工作原理：
    #   1. 首先遍历所有消息，收集所有已有 ToolMessage 的 tool_call_id
    #   2. 然后检查是否有 AIMessage 的 tool_calls 在已有列表中
    #   3. 如果有悬空的 tool_call，创建合成的错误 ToolMessage
    #   4. 将这些合成消息插入到正确的位置（AIMessage 之后）
    #
    # 参数：
    #   messages: 原始消息列表
    #
    # 返回值：
    #   如果有补丁，返回修复后的新消息列表
    #   如果不需要补丁（即没有悬空工具调用），返回 None
    def _build_patched_messages(self, messages: list) -> list | None:
        # 第一步：收集所有已有的 ToolMessage 的 tool_call_id
        #
        # set 用于快速查找，避免重复
        existing_tool_msg_ids: set[str] = set()
        for msg in messages:
            # 检查是否是 ToolMessage 类型
            if isinstance(msg, ToolMessage):
                # ToolMessage 有一个 tool_call_id 字段，标识它对应的工具调用
                existing_tool_msg_ids.add(msg.tool_call_id)

        # 第二步：检查是否有悬空的工具调用需要补丁
        needs_patch = False
        for msg in messages:
            # 只检查 AI 类型的消息
            if getattr(msg, "type", None) != "ai":
                continue

            # 获取消息的 tool_calls 字段
            # tool_calls 是列表，每个元素是一个工具调用（包含 id、name 等）
            for tc in getattr(msg, "tool_calls", None) or []:
                # 获取工具调用的 ID
                tc_id = tc.get("id")
                # 如果这个 ID 不在已有的 ToolMessage IDs 中，说明是悬空的
                if tc_id and tc_id not in existing_tool_msg_ids:
                    needs_patch = True  # 需要打补丁
                    break
            if needs_patch:
                break

        # 如果不需要补丁，直接返回 None（不修改消息列表）
        if not needs_patch:
            return None

        # 第三步：构建打补丁后的消息列表
        patched: list = []
        patched_ids: set[str] = set()  # 记录已处理的 tool_call_id，避免重复
        patch_count = 0  # 统计打了多少补丁

        for msg in messages:
            # 首先添加原始消息
            patched.append(msg)

            # 如果不是 AI 消息，跳过
            if getattr(msg, "type", None) != "ai":
                continue

            # 遍历 AI 消息中的 tool_calls
            for tc in getattr(msg, "tool_calls", None) or []:
                tc_id = tc.get("id")
                # 检查是否需要创建合成消息：
                #   1. tc_id 存在
                #   2. 不在已有的 ToolMessage IDs 中（悬空的）
                #   3. 还没处理过（避免重复）
                if tc_id and tc_id not in existing_tool_msg_ids and tc_id not in patched_ids:
                    # 创建合成的错误 ToolMessage
                    patched.append(
                        ToolMessage(
                            # 错误消息内容
                            content="[Tool call was interrupted and did not return a result.]",
                            # 工具调用 ID，与 AIMessage 中的 tool_call.id 对应
                            tool_call_id=tc_id,
                            # 工具名称，从原始 tool_call 中获取
                            name=tc.get("name", "unknown"),
                            # 状态设为 "error"，标识这是一个错误的响应
                            status="error",
                        )
                    )
                    # 记录已处理的 ID
                    patched_ids.add(tc_id)
                    # 计数器 +1
                    patch_count += 1

        # 记录警告日志：插入了多少个占位符 ToolMessage
        logger.warning(f"Injecting {patch_count} placeholder ToolMessage(s) for dangling tool calls")

        # 返回打补丁后的消息列表
        return patched

    # wrap_model_call 钩子方法：同步版本的模型调用包装
    #
    # 这个方法是 AgentMiddleware 提供的扩展点
    # 在模型调用前后可以对请求和响应进行处理
    #
    # 参数：
    #   request: ModelRequest 模型调用请求（包含 messages 等）
    #   handler: 原始的模型调用处理器（一个 Callable）
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
        # 检查消息列表是否需要补丁
        patched = self._build_patched_messages(request.messages)
        if patched is not None:
            # 用修复后的消息列表创建新的请求
            # request.override() 创建一个新的请求，用 patched 替换原始 messages
            request = request.override(messages=patched)

        # 调用原始 handler 处理请求
        return handler(request)

    # awrap_model_call 钩子方法：异步版本的模型调用包装
    #
    # 功能与 wrap_model_call 相同，但是支持异步 handler
    #
    # 参数：
    #   request: ModelRequest 模型调用请求
    #   handler: 异步的模型调用处理器（返回 Awaitable）
    #
    # 返回值：
    #   ModelCallResult，模型调用的结果
    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        # 检查消息列表是否需要补丁
        patched = self._build_patched_messages(request.messages)
        if patched is not None:
            # 用修复后的消息列表创建新的请求
            request = request.override(messages=patched)

        # 使用 await 调用异步 handler
        return await handler(request)