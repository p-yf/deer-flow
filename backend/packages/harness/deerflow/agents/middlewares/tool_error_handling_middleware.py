"""ToolErrorHandlingMiddleware - 工具错误处理中间件。

功能概述：
  将工具执行过程中的异常转换为格式良好的错误 ToolMessage，确保 LLM 能正确处理。

工作流程：
  1. 包装工具调用（wrap_tool_call / awrap_tool_call）
  2. 捕获工具执行过程中的异常
  3. 将异常分类：超时、授权错误、语法错误、无效工具等
  4. 构建包含错误信息的 ToolMessage 返回给 LLM
  5. 某些异常（如 KeyboardInterrupt）会重新抛出

错误类型处理：
  - TimeoutError → "Tool call timed out" ToolMessage
  - AuthorizationError → "Authorization error" ToolMessage
  - 语法错误 → 详细的语法错误信息 ToolMessage
  - 无效工具 → "Unknown tool" ToolMessage
  - 其他异常 → 通用错误 ToolMessage

注意：GraphBubbleUp 等控制流信号会正确向上传播，不会被截获。

执行位置：紧接 DanglingToolCallMiddleware 之后。
"""
"""Tool error handling middleware and shared runtime middleware builders."""

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

# langchain_core.messages.ToolMessage：
#   工具消息类型，当工具执行失败时返回此类型的消息
#   包含错误信息供 LLM 理解工具调用失败的原因
from langchain_core.messages import ToolMessage

# langgraph.errors.GraphBubbleUp：
#   LangGraph 的控制流信号（如 interrupt、pause、resume）
#   这些信号需要被保留而不是被转换成错误消息
from langgraph.errors import GraphBubbleUp

# langgraph.prebuilt.tool_node.ToolCallRequest：
#   工具调用的请求对象，包含 tool_call（工具名称、参数、ID 等）
from langgraph.prebuilt.tool_node import ToolCallRequest

# langgraph.types.Command：
#   LangGraph 的命令类型，用于控制图执行流程（如中断、跳转）
from langgraph.types import Command

# ============================================================
# 模块级变量初始化
# ============================================================

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)

# ============================================================
# 模块级常量定义
# ============================================================

# _MISSING_TOOL_CALL_ID：当工具调用没有 ID 时使用的默认值
# 这发生在工具调用的 id 字段缺失时
# 用于确保 ToolMessage 有正确的 tool_call_id
_MISSING_TOOL_CALL_ID = "missing_tool_call_id"


# ============================================================
# ToolErrorHandlingMiddleware 主类
# ============================================================

# ToolErrorHandlingMiddleware 类：工具错误处理中间件
#
# 核心作用：
#   拦截工具调用，捕获执行过程中的异常，
#   将异常转换为格式良好的错误 ToolMessage，让 agent 可以优雅地继续执行。
#
# 工作原理：
#   1. 拦截工具调用（wrap_tool_call / awrap_tool_call）
#   2. 用 try-except 包装原始 handler 的调用
#   3. 如果执行成功，返回正常结果
#   4. 如果抛出 GraphBubbleUp（LangGraph 控制流信号），重新抛出
#   5. 如果是其他异常，将异常转换为包含错误信息的 ToolMessage
#
# 关键设计：
#   - 不直接让异常传播，而是转换成 ToolMessage
#   - 这样 agent 的执行不会因为工具失败而中断
#   - Agent 可以用错误信息作为上下文，决定下一步做什么
class ToolErrorHandlingMiddleware(AgentMiddleware[AgentState]):
    """Convert tool exceptions into error ToolMessages so the run can continue."""

    # state_schema：类变量，指定该中间件使用的状态类型
    state_schema = AgentState

    # ============================================================
    # 内部辅助方法
    # ============================================================

    # _build_error_message：构建错误 ToolMessage
    #
    # 方法作用：
    #   将异常对象转换为一个格式良好的错误 ToolMessage。
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求
    #   exc: Exception，捕获的异常对象
    #
    # 返回值：
    #   ToolMessage，包含格式化后的错误信息
    #
    # 错误消息格式：
    #   "Error: Tool '{tool_name}' failed with {exception_class}: {detail}. Continue..."
    #
    # 设计考虑：
    #   - 错误消息被截断到 500 字符以避免上下文溢出
    #   - 消息指示 agent 可以继续使用可用上下文或选择其他工具
    def _build_error_message(self, request: ToolCallRequest, exc: Exception) -> ToolMessage:
        # 获取工具名称，如果不存在则使用 "unknown_tool"
        tool_name = str(request.tool_call.get("name") or "unknown_tool")

        # 获取工具调用 ID，如果不存在则使用默认值
        tool_call_id = str(request.tool_call.get("id") or _MISSING_TOOL_CALL_ID)

        # 获取异常详情字符串，去除首尾空白
        # 如果异常没有详情（空字符串），使用异常类名作为后备
        detail = str(exc).strip() or exc.__class__.__name__

        # 如果详情超过 500 字符，截断到 497 字符并添加 "..."
        # 避免错误消息占用过多上下文空间
        if len(detail) > 500:
            detail = detail[:497] + "..."

        # 构建错误消息内容
        # 格式：Error: Tool '{tool_name}' failed with {exception_class}: {detail}. Continue...
        content = f"Error: Tool '{tool_name}' failed with {exc.__class__.__name__}: {detail}. Continue with available context, or choose an alternative tool."

        # 返回包含错误信息的 ToolMessage
        return ToolMessage(
            content=content,        # 错误消息内容
            tool_call_id=tool_call_id,  # 工具调用 ID，与请求对应
            name=tool_name,          # 工具名称
            status="error",         # 状态设为 "error"，标识这是一个错误响应
        )

    # ============================================================
    # LangChain AgentMiddleware 钩子方法
    # ============================================================

    # wrap_tool_call：同步版本的工具调用包装钩子
    #
    # 方法作用：
    #   LangChain AgentMiddleware 提供的扩展点，
    #   在工具调用执行前后进行拦截和处理。
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求（包含工具名称、参数、ID 等）
    #   handler: Callable[[ToolCallRequest], ToolMessage | Command]，原始工具执行处理器
    #
    # 返回值：
    #   ToolMessage | Command：
    #     - 成功：返回原始执行结果
    #     - 控制流信号：重新抛出 GraphBubbleUp
    #     - 执行失败：返回包含错误信息的 ToolMessage
    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        try:
            # 调用原始 handler 执行工具
            return handler(request)

        except GraphBubbleUp:
            # GraphBubbleUp 是 LangGraph 的控制流信号
            # 包括 interrupt（中断）、pause（暂停）、resume（恢复）等
            # 这些信号必须被保留，不能转换成错误消息
            raise

        except Exception as exc:
            # 其他异常（工具执行失败）
            # 记录异常日志（包含工具名称和 ID）
            logger.exception("Tool execution failed (sync): name=%s id=%s", request.tool_call.get("name"), request.tool_call.get("id"))

            # 将异常转换为错误 ToolMessage 并返回
            # 这样 agent 可以继续执行，用错误信息作为上下文
            return self._build_error_message(request, exc)

    # awrap_tool_call：异步版本的工具调用包装钩子
    #
    # 方法作用：
    #   与 wrap_tool_call 相同，但支持异步 handler。
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求
    #   handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]，异步工具执行处理器
    #
    # 返回值：
    #   ToolMessage | Command（与同步版本相同）
    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        try:
            # 使用 await 调用异步 handler
            return await handler(request)

        except GraphBubbleUp:
            # 控制流信号，重新抛出
            raise

        except Exception as exc:
            # 其他异常
            # 记录异常日志（工具名称和 ID）
            logger.exception("Tool execution failed (async): name=%s id=%s", request.tool_call.get("name"), request.tool_call.get("id"))

            # 将异常转换为错误 ToolMessage
            return self._build_error_message(request, exc)


# ============================================================
# 模块级中间件构建器函数
# ============================================================

# _build_runtime_middlewares：构建共享的基础运行时中间件列表
#
# 方法作用：
#   根据配置决定包含哪些中间件，返回按执行顺序排列的中间件列表。
#   所有 agent（lead agent 和 subagent）共享的基础中间件都在这里构建。
#
# 参数：
#   include_uploads: bool，是否包含 UploadsMiddleware（注入上传文件信息）
#   include_dangling_tool_call_patch: bool，是否包含 DanglingToolCallMiddleware（修复悬空工具调用）
#   lazy_init: bool，是否延迟初始化（影响 ThreadDataMiddleware 和 SandboxMiddleware）
#
# 返回值：
#   list[AgentMiddleware]：中间件列表，按执行顺序排列
#
# 中间件顺序：
#   1. ThreadDataMiddleware - 创建线程数据目录
#   2. UploadsMiddleware - 注入上传文件信息（可选）
#   3. SandboxMiddleware - 获取沙箱环境
#   4. DanglingToolCallMiddleware - 修复悬空工具调用（可选）
#   5. LLMErrorHandlingMiddleware - 处理 LLM 调用错误
#   6. GuardrailMiddleware - 工具调用授权检查（可选）
#   7. SandboxAuditMiddleware - Bash 命令安全审计
#   8. ToolErrorHandlingMiddleware - 工具错误处理
def _build_runtime_middlewares(
    *,
    include_uploads: bool,
    include_dangling_tool_call_patch: bool,
    lazy_init: bool = True,
) -> list[AgentMiddleware]:
    # 内部导入（在函数内部导入避免循环依赖）
    # LLMErrorHandlingMiddleware：处理 LLM 调用中的错误
    from deerflow.agents.middlewares.llm_error_handling_middleware import LLMErrorHandlingMiddleware
    # ThreadDataMiddleware：创建线程数据目录
    from deerflow.agents.middlewares.thread_data_middleware import ThreadDataMiddleware
    # SandboxMiddleware：获取沙箱环境
    from deerflow.sandbox.middleware import SandboxMiddleware

    # 初始化中间件列表
    # 这是共享的基础中间件，所有 agent 都会使用
    middlewares: list[AgentMiddleware] = [
        # ThreadDataMiddleware 第一个执行，为线程创建数据目录
        ThreadDataMiddleware(lazy_init=lazy_init),
        # SandboxMiddleware 第二个执行，获取沙箱环境
        SandboxMiddleware(lazy_init=lazy_init),
    ]

    # 如果配置了 include_uploads，在第二个位置插入 UploadsMiddleware
    # UploadsMiddleware 负责将上传文件信息注入到 agent 上下文
    if include_uploads:
        from deerflow.agents.middlewares.uploads_middleware import UploadsMiddleware
        # insert(1, ...) 将 UploadsMiddleware 插入到列表开头之后的位置
        # 这样它就在 ThreadDataMiddleware 之后、SandboxMiddleware 之前
        middlewares.insert(1, UploadsMiddleware())

    # 如果配置了 include_dangling_tool_call_patch，追加 DanglingToolCallMiddleware
    # 这个中间件修复消息历史中的悬空工具调用
    if include_dangling_tool_call_patch:
        from deerflow.agents.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware
        middlewares.append(DanglingToolCallMiddleware())

    # 追加 LLMErrorHandlingMiddleware
    # 这个中间件处理 LLM 调用中的错误
    middlewares.append(LLMErrorHandlingMiddleware())

    # Guardrail 中间件（如果配置了的话）
    # Guardrail 用于在工具调用前进行授权检查
    from deerflow.config.guardrails_config import get_guardrails_config
    guardrails_config = get_guardrails_config()

    # 如果 guardrails 启用且配置了 provider
    if guardrails_config.enabled and guardrails_config.provider:
        import inspect
        from deerflow.guardrails.middleware import GuardrailMiddleware
        from deerflow.reflection import resolve_variable

        # 使用 resolve_variable 动态导入 provider 类
        # provider.use 是一个字符串，如 "module.path:ClassName"
        provider_cls = resolve_variable(guardrails_config.provider.use)
        # 从配置中获取 provider 的参数
        provider_kwargs = dict(guardrails_config.provider.config) if guardrails_config.provider.config else {}

        # 如果 provider 的构造函数接受 framework 参数或 **kwargs
        # 则注入 framework="deerflow" 提示
        if "framework" not in provider_kwargs:
            try:
                # 检查构造函数的签名
                sig = inspect.signature(provider_cls.__init__)
                # 如果接受 framework 参数或 **kwargs
                if "framework" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    # 注入 framework 参数
                    provider_kwargs["framework"] = "deerflow"
            except (ValueError, TypeError):
                # 某些类可能无法获取签名（如 C 扩展），忽略错误
                pass

        # 创建 provider 实例
        provider = provider_cls(**provider_kwargs)
        # 创建并追加 GuardrailMiddleware
        # fail_closed：拒绝模式，为 True 时默认拒绝
        # passport：认证配置
        middlewares.append(GuardrailMiddleware(provider, fail_closed=guardrails_config.fail_closed, passport=guardrails_config.passport))

    # 导入并追加 SandboxAuditMiddleware
    # 这个中间件负责沙箱的审计日志
    from deerflow.agents.middlewares.sandbox_audit_middleware import SandboxAuditMiddleware
    middlewares.append(SandboxAuditMiddleware())

    # 最后追加 ToolErrorHandlingMiddleware
    # 这个中间件将工具执行中的异常转换为错误 ToolMessage
    middlewares.append(ToolErrorHandlingMiddleware())

    # 返回完整的中间件列表
    return middlewares


# build_lead_runtime_middlewares：构建 Lead Agent 专用的运行时中间件
#
# 方法作用：
#   Lead Agent（主代理）使用的中间件链构建函数。
#   它调用 _build_runtime_middlewares 并指定 Lead Agent 的配置。
#
# 参数：
#   lazy_init: bool，是否延迟初始化
#
# 返回值：
#   list[AgentMiddleware]：中间件列表
#
# Lead Agent 的配置：
#   - include_uploads=True：需要注入上传文件信息
#   - include_dangling_tool_call_patch=True：需要修复悬空工具调用
def build_lead_runtime_middlewares(*, lazy_init: bool = True) -> list[AgentMiddleware]:
    """Middlewares shared by lead agent runtime before lead-only middlewares."""
    return _build_runtime_middlewares(
        include_uploads=True,  # Lead Agent 需要处理上传文件
        include_dangling_tool_call_patch=True,  # 需要修复悬空工具调用
        lazy_init=lazy_init,
    )


# build_subagent_runtime_middlewares：构建 Subagent 专用的运行时中间件
#
# 方法作用：
#   Subagent 使用的中间件链构建函数。
#   子代理与主代理的中间件链略有不同。
#
# 参数：
#   lazy_init: bool，是否延迟初始化
#
# 返回值：
#   list[AgentMiddleware]：中间件列表
#
# Subagent 的配置：
#   - include_uploads=False：子代理不需要处理上传文件
#   - include_dangling_tool_call_patch=True：需要修复悬空工具调用
def build_subagent_runtime_middlewares(*, lazy_init: bool = True) -> list[AgentMiddleware]:
    """Middlewares shared by subagent runtime before subagent-only middlewares."""
    return _build_runtime_middlewares(
        include_uploads=False,  # 子代理不需要处理上传文件
        include_dangling_tool_call_patch=True,  # 需要修复悬空工具调用
        lazy_init=lazy_init,
    )
