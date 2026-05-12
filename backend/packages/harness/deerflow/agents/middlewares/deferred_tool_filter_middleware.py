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
"""Middleware to filter deferred tools from model binding.

When tool_search is enabled, MCP tools are registered in a DeferredToolRegistry
and passed to ToolNode for execution routing, but their schemas should NOT be
sent to the LLM via bind_tools. Deferred tools are designed to save context tokens.

This middleware intercepts wrap_model_call to remove deferred tools from
request.tools before the model processes them, while ToolNode still holds
all tools (including deferred ones) for execution routing.
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
#   - ModelCallResult：模型调用的结果
#   - ModelRequest：模型调用的请求（包含 messages、tools 等）
#   - ModelResponse：模型调用的响应
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse

# ============================================================
# 模块级变量初始化
# ============================================================

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)


# ============================================================
# DeferredToolFilterMiddleware 主类
# ============================================================

# DeferredToolFilterMiddleware 类：延迟工具过滤中间件
#
# 核心作用：
#   在模型调用前从 request.tools 中过滤掉延迟工具，只保留活跃工具。
#   这样 LLM 不会看到延迟工具的模式，节省上下文 tokens。
#
# 工作流程：
#   1. 在 wrap_model_call / awrap_model_call 中拦截模型调用
#   2. 从 deerflow.tools.builtins.tool_search 获取 DeferredRegistry
#   3. 从 request.tools 中过滤掉所有名称在延迟注册表中的工具
#   4. 返回更新后的 ModelRequest（只包含活跃工具）
#
# 设计考虑：
#   - ToolNode 仍然持有所有工具（包括延迟的），用于执行路由
#   - 但 LLM 只看到活跃工具的模式
#   - 延迟工具通过 tool_search 工具在运行时被发现
#   - 这样可以节省大量上下文 tokens（大型 MCP 工具模式可能有几千 token）
class DeferredToolFilterMiddleware(AgentMiddleware[AgentState]):
    """Remove deferred tools from request.tools before model binding.

    ToolNode still holds all tools (including deferred) for execution routing,
    but the LLM only sees active tool schemas — deferred tools are discoverable
    via tool_search at runtime.
    """

    # state_schema：类变量，指定该中间件使用的状态类型
    state_schema = AgentState

    # ============================================================
    # 内部辅助方法
    # ============================================================

    # _filter_tools：过滤延迟工具
    #
    # 方法作用：
    #   从请求中移除所有延迟工具，返回只包含活跃工具的更新请求。
    #
    # 参数：
    #   request: ModelRequest，原始模型调用请求
    #
    # 返回值：
    #   ModelRequest：更新后的请求（移除了延迟工具）
    #
    # 工作流程：
    #   1. 延迟导入 DeferredRegistry（避免循环依赖）
    #   2. 获取延迟工具注册表，如果为空则不处理
    #   3. 构建延迟工具名称集合
    #   4. 过滤掉所有名称在延迟集合中的工具
    #   5. 如果有工具被过滤，记录 debug 日志
    #   6. 使用 request.override() 创建更新后的请求
    def _filter_tools(self, request: ModelRequest) -> ModelRequest:
        # 延迟导入，避免循环依赖
        # DeferredRegistry 在 deerflow.tools.builtins.tool_search 模块中定义
        from deerflow.tools.builtins.tool_search import get_deferred_registry

        # 获取延迟工具注册表
        registry = get_deferred_registry()
        # 如果没有注册表（或注册表为空），不需要过滤
        if not registry:
            return request

        # 构建延迟工具名称集合
        # registry.entries 是延迟工具条目的列表
        # 每个条目有 name 属性
        deferred_names = {e.name for e in registry.entries}

        # 过滤：只保留不在延迟集合中的工具
        # getattr(t, "name", None) 获取工具的名称，如果工具没有 name 属性则返回 None
        # 这样可以处理各种类型的工具对象
        active_tools = [t for t in request.tools if getattr(t, "name", None) not in deferred_names]

        # 如果有工具被过滤，记录 debug 日志
        # 显示过滤了多少个延迟工具模式
        if len(active_tools) < len(request.tools):
            logger.debug(f"Filtered {len(request.tools) - len(active_tools)} deferred tool schema(s) from model binding")

        # 使用 request.override 创建更新后的请求
        # override() 创建一个浅拷贝，用 active_tools 替换原始的 tools
        # 保留其他字段（如 messages）不变
        return request.override(tools=active_tools)

    # ============================================================
    # LangChain AgentMiddleware 钩子方法
    # ============================================================

    # wrap_model_call：同步版本的模型调用包装钩子
    #
    # 方法作用：
    #   LangChain AgentMiddleware 提供的扩展点，
    #   在模型调用前过滤掉延迟工具。
    #
    # 参数：
    #   request: ModelRequest，模型调用请求
    #   handler: Callable[[ModelRequest], ModelResponse]，原始模型调用处理器
    #
    # 返回值：
    #   ModelCallResult，模型调用结果
    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        # 先过滤延迟工具，再调用原始 handler
        return handler(self._filter_tools(request))

    # awrap_model_call：异步版本的模型调用包装钩子
    #
    # 方法作用：
    #   与 wrap_model_call 相同，但支持异步 handler。
    #
    # 参数：
    #   request: ModelRequest，模型调用请求
    #   handler: Callable[[ModelRequest], Awaitable[ModelResponse]]，异步模型调用处理器
    #
    # 返回值：
    #   ModelCallResult，模型调用结果
    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        # 先过滤延迟工具，再调用原始 handler
        return await handler(self._filter_tools(request))
