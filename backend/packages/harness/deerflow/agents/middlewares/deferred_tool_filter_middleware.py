"""DeferredToolFilterMiddleware - 延迟工具过滤中间件。

功能概述：
  从模型绑定中过滤延迟工具的模式，节省上下文 tokens。

问题背景：
  当启用 tool_search 时，MCP 工具被注册到 DeferredToolRegistry 中，
  并传递给 ToolNode 执行，但它们的模式不应该通过 bind_tools 发送给 LLM。
  这就是延迟的目的 — 节省上下文 tokens。

工作流程：
  1. 拦截 wrap_model_call / awrap_model_call
  2. 从 deerflow.tools.builtins.tool_search 获取 DeferredRegistry
  3. 从 request.tools 中过滤掉所有延迟工具
  4. 返回更新后的 ModelRequest（只包含活跃工具）

架构设计：
  - ToolNode 仍然持有所有工具（包括延迟的），用于执行路由
  - LLM 只看到活跃工具的模式
  - 延迟工具通过 tool_search 工具在运行时被发现

执行位置：当 tool_search.enabled 时添加，在 SubagentLimitMiddleware 之后。
"""

# 标准库 logging：用于记录日志
import logging
# collections.abc 导入 Awaitable（异步可等待对象）和 Callable（可调用对象）
from collections.abc import Awaitable, Callable
# typing 导入 override（方法重写标记）
from typing import override

# LangChain agents 相关：AgentState 是 agent 基础状态类
from langchain.agents import AgentState
# LangChain agents.middleware：AgentMiddleware 是所有中间件的基类
from langchain.agents.middleware import AgentMiddleware
# LangChain agents.middleware.types：中间件使用的请求/响应类型
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)


# DeferredToolFilterMiddleware 类：延迟工具过滤中间件
#
# 工作原理：
#   1. 在 wrap_model_call / awrap_model_call 中拦截模型调用
#   2. 从 deerflow.tools.builtins.tool_search 获取 DeferredRegistry
#   3. 从 request.tools 中过滤掉所有延迟工具
#   4. 返回更新后的 ModelRequest（只包含活跃工具）
#
# 设计考虑：
#   - ToolNode 仍然持有所有工具（包括延迟的），用于执行路由
#   - 但 LLM 只看到活跃工具的模式 — 延迟工具通过 tool_search 在运行时发现
#   - 这样可以节省大量上下文 tokens（大型 MCP 工具模式可能有几千 token）
class DeferredToolFilterMiddleware(AgentMiddleware[AgentState]):
    """Remove deferred tools from request.tools before model binding.

    ToolNode still holds all tools (including deferred) for execution routing,
    but the LLM only sees active tool schemas — deferred tools are discoverable
    via tool_search at runtime.
    """

    # 内部方法：过滤延迟工具
    #
    # 参数：
    #   request: ModelRequest，原始模型调用请求
    #
    # 返回值：
    #   ModelRequest：更新后的请求（移除了延迟工具）
    #
    # 工作流程：
    #   1. 获取延迟工具注册表
    #   2. 如果注册表为空，直接返回原始请求（没有延迟工具需要过滤）
    #   3. 构建延迟工具名称集合
    #   4. 过滤掉所有名称在延迟集合中的工具
    #   5. 如果有工具被过滤，记录 debug 日志
    #   6. 返回更新后的请求（使用 request.override(tools=active_tools)）
    def _filter_tools(self, request: ModelRequest) -> ModelRequest:
        # 延迟导入，避免循环依赖
        from deerflow.tools.builtins.tool_search import get_deferred_registry

        # 获取延迟工具注册表
        registry = get_deferred_registry()
        # 如果没有注册表（或为空），不需要过滤
        if not registry:
            return request

        # 构建延迟工具名称集合
        deferred_names = {e.name for e in registry.entries}
        # 过滤：只保留不在延迟集合中的工具
        active_tools = [t for t in request.tools if getattr(t, "name", None) not in deferred_names]

        # 如果有工具被过滤，记录日志
        if len(active_tools) < len(request.tools):
            logger.debug(f"Filtered {len(request.tools) - len(active_tools)} deferred tool schema(s) from model binding")

        # 使用 request.override 创建更新后的请求（保留其他字段不变）
        return request.override(tools=active_tools)

    # wrap_model_call：同步版本的模型调用包装
    #
    # 参数：
    #   request: ModelRequest，模型调用请求
    #   handler: Callable[[ModelRequest], ModelResponse]，原始模型调用处理器
    #
    # 返回值：
    #   ModelCallResult：模型调用结果
    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        # 先过滤延迟工具，再调用 handler
        return handler(self._filter_tools(request))

    # awrap_model_call：异步版本的模型调用包装
    #
    # 参数：
    #   request: ModelRequest，模型调用请求
    #   handler: Callable[[ModelRequest], Awaitable[ModelResponse]]，异步模型调用处理器
    #
    # 返回值：
    #   ModelCallResult：模型调用结果
    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        # 先过滤延迟工具，再调用 handler
        return await handler(self._filter_tools(request))