"""ViewImageMiddleware - 图像查看详情注入中间件。

功能概述：
  在 LLM 调用前将图像详情注入对话，让模型自动接收和分析图像。

工作流程：
  1. before_model 钩子中检查条件
  2. 获取最后一条助手消息，检查是否包含 view_image 工具调用
  3. 验证所有工具调用都已完成（有 ToolMessage）
  4. 如果条件满足，从 state.viewed_images 获取图像详情（包含 base64 数据）
  5. 创建混合内容的人类消息（文本描述 + base64 图像）
  6. 返回状态更新，将消息注入消息历史

注入条件（全部满足才注入）：
  - 最后一条消息是助手消息
  - 助手消息包含 view_image 工具调用
  - 所有工具调用都已完成
  - 还没有注入过图像详情消息

图像格式：
  - 文本描述：图像路径和 MIME 类型
  - 图像数据：data:{mime_type};base64,{base64_data}
  - 通过 image_url 类型内容块发送给 LLM

执行位置：紧接 MemoryMiddleware 之后（当模型支持 vision 时才添加）。
"""
"""
3. 如果条件满足，创建一个包含所有已查看图像详情的人类消息
4. 将消息添加到状态中，让 LLM 可以看到和分析这些图像

这样 LLM 可以自动接收和分析通过 view_image 工具加载的图像，
无需用户显式提示描述图像。
"""

# 标准库 logging：用于记录中间件运行日志
import logging
# typing 导入 override（方法重写标记）
from typing import override

# langchain.agents.middleware：AgentMiddleware 是所有中间件的基类
from langchain.agents.middleware import AgentMiddleware
# langchain_core.messages：导入 AIMessage、HumanMessage、ToolMessage
# - AIMessage：助手消息
# - HumanMessage：用于注入图像详情的人类消息
# - ToolMessage：工具调用结果
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
# langgraph.runtime：导入 Runtime，LangGraph 运行时上下文
from langgraph.runtime import Runtime

# 从 deerflow.agents.thread_state 导入 ThreadState，用于状态类型定义
from deerflow.agents.thread_state import ThreadState

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)


# ViewImageMiddlewareState 类：中间件使用的状态 schema
#
# 继承自 ThreadState，复用其完整的状态定义
# 确保使用 reducer 支持的键（如 viewed_images）保留正确的类型注解
class ViewImageMiddlewareState(ThreadState):
    """Reuse the thread state so reducer-backed keys keep their annotations."""


# ViewImageMiddleware 类：图像查看中间件
#
# 工作流程：
#   1. 在 before_model（模型调用前）钩子中检查条件
#   2. 获取最后一条助手消息，检查是否包含 view_image 工具调用
#   3. 验证该消息中的所有工具调用都已完成（有 ToolMessage）
#   4. 如果条件满足，从 state.viewed_images 中获取图像详情（包含 base64 数据）
#   5. 创建一个混合内容的人类消息（文本 + 图像 URL）
#   6. 返回状态更新，将消息注入消息历史
#
# 设计考虑：
#   - 只在 view_image 工具调用完成后才注入消息（避免 LLM 看到不完整的图像）
#   - 检查是否已经注入过，避免重复注入
#   - 支持文本描述和 base64 编码的图像数据同时发送
#   - 图像数据通过 data URL 格式（data:mime_type;base64,base64_data）发送
class ViewImageMiddleware(AgentMiddleware[ViewImageMiddlewareState]):
    """Injects image details as a human message before LLM calls when view_image tools have completed.

    This middleware:
    1. Runs before each LLM call
    2. Checks if the last assistant message contains view_image tool calls
    3. Verifies all tool calls in that message have been completed (have corresponding ToolMessages)
    4. If conditions are met, creates a human message with all viewed image details (including base64 data)
    5. Adds the message to state so the LLM can see and analyze the images

    This enables the LLM to automatically receive and analyze images that were loaded via view_image tool,
    without requiring explicit user prompts to describe the images.
    """

    # state_schema 类变量，指定该中间件使用的状态类型
    state_schema = ViewImageMiddlewareState

    # 内部方法：获取消息列表中的最后一条助手消息
    #
    # 参数：
    #   messages: list，消息列表
    #
    # 返回值：
    #   AIMessage | None：最后一条 AIMessage，如果没有则返回 None
    #
    # 实现：反向遍历消息列表，找到第一个 AIMessage 即为最后一条
    def _get_last_assistant_message(self, messages: list) -> AIMessage | None:
        """Get the last assistant message from the message list.

        Args:
            messages: List of messages

        Returns:
            Last AIMessage or None if not found
        """
        # 反向遍历消息
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg
        return None

    # 内部方法：检查助手消息是否包含 view_image 工具调用
    #
    # 参数：
    #   message: AIMessage，要检查的助手消息
    #
    # 返回值：
    #   bool：如果包含 view_image 工具调用则返回 True
    def _has_view_image_tool(self, message: AIMessage) -> bool:
        """Check if the assistant message contains view_image tool calls.

        Args:
            message: Assistant message to check

        Returns:
            True if message contains view_image tool calls
        """
        # 检查消息是否有 tool_calls 属性且非空
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return False

        # 检查是否有任何 tool_call 的名称为 "view_image"
        return any(tool_call.get("name") == "view_image" for tool_call in message.tool_calls)

    # 内部方法：检查助手消息中的所有工具调用是否都已完成
    #
    # 参数：
    #   messages: list，所有消息列表
    #   assistant_msg: AIMessage，包含工具调用的助手消息
    #
    # 返回值：
    #   bool：如果所有工具调用都有对应的 ToolMessage 则返回 True
    #
    # 工作原理：
    #   1. 获取助手消息中的所有 tool_call IDs
    #   2. 在助手消息之后的消息中查找 ToolMessage
    #   3. 比较：所有 tool_call_ids 是否都在 completed_tool_ids 中
    def _all_tools_completed(self, messages: list, assistant_msg: AIMessage) -> bool:
        """Check if all tool calls in the assistant message have been completed.

        Args:
            messages: List of all messages
            assistant_msg: The assistant message containing tool calls

        Returns:
            True if all tool calls have corresponding ToolMessages
        """
        # 检查助手消息是否有 tool_calls
        if not hasattr(assistant_msg, "tool_calls") or not assistant_msg.tool_calls:
            return False

        # 获取助手消息中所有 tool_call 的 ID
        tool_call_ids = {tool_call.get("id") for tool_call in assistant_msg.tool_calls if tool_call.get("id")}

        # 找到助手消息在列表中的索引
        try:
            assistant_idx = messages.index(assistant_msg)
        except ValueError:
            return False

        # 在助手消息之后的消息中查找所有 ToolMessage
        completed_tool_ids = set()
        for msg in messages[assistant_idx + 1:]:
            if isinstance(msg, ToolMessage) and msg.tool_call_id:
                completed_tool_ids.add(msg.tool_call_id)

        # 检查所有工具调用是否都已完成
        return tool_call_ids.issubset(completed_tool_ids)

    # 内部方法：创建包含所有已查看图像详情的消息内容
    #
    # 参数：
    #   state: ViewImageMiddlewareState，当前状态
    #
    # 返回值：
    #   list[str | dict]：内容块列表（文本和图像），用于 HumanMessage
    #
    # 消息格式：
    #   - 文本块：描述图像列表
    #   - 图像块：包含 base64 数据的图像 URL（data:mime_type;base64,...）
    def _create_image_details_message(self, state: ViewImageMiddlewareState) -> list[str | dict]:
        """Create a formatted message with all viewed image details.

        Args:
            state: Current state containing viewed_images

        Returns:
            List of content blocks (text and images) for the HumanMessage
        """
        # 从状态中获取 viewed_images（这是一个 dict，key 是图像路径）
        viewed_images = state.get("viewed_images", {})
        if not viewed_images:
            # 如果没有已查看的图像，返回提示文本
            # 注意：返回正确格式的内容块，而不是普通字符串数组
            return [{"type": "text", "text": "No images have been viewed."}]

        # 构建消息内容
        content_blocks: list[str | dict] = [{"type": "text", "text": "Here are the images you've viewed:"}]

        # 遍历所有已查看的图像
        for image_path, image_data in viewed_images.items():
            # 获取 MIME 类型和 base64 数据
            mime_type = image_data.get("mime_type", "unknown")
            base64_data = image_data.get("base64", "")

            # 添加文本描述（图像路径和类型）
            content_blocks.append({"type": "text", "text": f"\n- **{image_path}** ({mime_type})"})

            # 添加实际的图像数据，让 LLM 可以"看到"图像
            if base64_data:
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
                    }
                )

        return content_blocks

    # 内部方法：判断是否应该注入图像详情消息
    #
    # 参数：
    #   state: ViewImageMiddlewareState，当前状态
    #
    # 返回值：
    #   bool：如果应该注入则返回 True
    #
    # 条件检查（全部满足才注入）：
    #   1. 有消息列表
    #   2. 最后一条消息是助手消息
    #   3. 助手消息包含 view_image 工具调用
    #   4. 所有工具调用都已完成
    #   5. 还没有注入过图像详情消息（避免重复）
    def _should_inject_image_message(self, state: ViewImageMiddlewareState) -> bool:
        """Determine if we should inject an image details message.

        Args:
            state: Current state

        Returns:
            True if we should inject the message
        """
        # 获取消息列表
        messages = state.get("messages", [])
        if not messages:
            return False

        # 获取最后一条助手消息
        last_assistant_msg = self._get_last_assistant_message(messages)
        if not last_assistant_msg:
            return False

        # 检查是否包含 view_image 工具调用
        if not self._has_view_image_tool(last_assistant_msg):
            return False

        # 检查所有工具调用是否已完成
        if not self._all_tools_completed(messages, last_assistant_msg):
            return False

        # 检查是否已经注入过图像详情消息
        # 查找助手消息之后是否有包含特定内容的人类消息
        assistant_idx = messages.index(last_assistant_msg)
        for msg in messages[assistant_idx + 1:]:
            if isinstance(msg, HumanMessage):
                content_str = str(msg.content)
                # 如果已经注入过，不再重复注入
                if "Here are the images you've viewed" in content_str or "Here are the details of the images you've viewed" in content_str:
                    return False

        return True

    # 内部方法：注入图像详情消息（内部辅助方法）
    #
    # 参数：
    #   state: ViewImageMiddlewareState，当前状态
    #
    # 返回值：
    #   dict | None：状态更新（包含新消息），如果不需要更新则返回 None
    def _inject_image_message(self, state: ViewImageMiddlewareState) -> dict | None:
        """Internal helper to inject image details message.

        Args:
            state: Current state

        Returns:
            State update with additional human message, or None if no update needed
        """
        # 检查是否应该注入
        if not self._should_inject_image_message(state):
            return None

        # 创建图像详情消息（包含文本和图像内容）
        image_content = self._create_image_details_message(state)

        # 创建新的人类消息，混合内容（文本 + 图像）
        human_msg = HumanMessage(content=image_content)

        logger.debug("Injecting image details message with images before LLM call")

        # 返回状态更新，包含新消息
        return {"messages": [human_msg]}

    # before_model：同步版本的模型调用前钩子
    #
    # 参数：
    #   state: ViewImageMiddlewareState，当前状态
    #   runtime: Runtime，运行时上下文（接口要求但未使用）
    #
    # 返回值：
    #   dict | None：状态更新（包含注入的消息），如果不需要更新则返回 None
    @override
    def before_model(self, state: ViewImageMiddlewareState, runtime: Runtime) -> dict | None:
        """Inject image details message before LLM call if view_image tools have completed (sync version).

        This runs before each LLM call, checking if the previous turn included view_image
        tool calls that have all completed. If so, it injects a human message with the image
        details so the LLM can see and analyze the images.

        Args:
            state: Current state
            runtime: Runtime context (unused but required by interface)

        Returns:
            State update with additional human message, or None if no update needed
        """
        return self._inject_image_message(state)

    # abefore_model：异步版本的模型调用前钩子
    #
    # 功能与同步版本相同
    @override
    async def abefore_model(self, state: ViewImageMiddlewareState, runtime: Runtime) -> dict | None:
        """Inject image details message before LLM call if view_image tools have completed (async version).

        This runs before each LLM call, checking if the previous turn included view_image
        tool calls that have all completed. If so, it injects a human message with the image
        details so the LLM can see and analyze the images.

        Args:
            state: Current state
            runtime: Runtime context (unused but required by interface)

        Returns:
            State update with additional human message, or None if no update needed
        """
        return self._inject_image_message(state)