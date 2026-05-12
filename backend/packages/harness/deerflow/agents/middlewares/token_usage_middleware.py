"""TokenUsageMiddleware - LLM Token 使用量日志中间件。

功能概述：
  在每次模型调用后记录 token 使用量信息，供监控和调试。

工作流程：
  1. after_model / aafter_model 钩子触发
  2. 获取消息列表中的最后一条消息
  3. 检查是否有 usage_metadata 属性
  4. 如果有，记录 input_tokens、output_tokens、total_tokens

日志格式：
  "LLM token usage: input={input} output={output} total={total}"

特点：
  - 纯日志中间件，不修改状态
  - 使用标准 logger（可见于 langgraph.log）
  - 不存在 usage_metadata 时静默跳过（正常情况）

执行位置：紧接 build_lead_runtime_middlewares 之后（当 token_usage.enabled 时才添加）。
"""
"""Middleware for logging LLM token usage after each model call."""

# ============================================================
# 导入标准库
# ============================================================

# logging：标准库日志模块，用于记录 token 使用量日志
import logging

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

# langgraph.runtime.Runtime：
#   LangGraph 运行时上下文，在钩子方法中作为参数传入
from langgraph.runtime import Runtime

# ============================================================
# 模块级变量初始化
# ============================================================

# 创建模块级 logger，用于记录 token 使用量日志
# 使用 __name__ 可以显示日志来源模块名
logger = logging.getLogger(__name__)


# ============================================================
# TokenUsageMiddleware 主类
# ============================================================

# TokenUsageMiddleware 类：Token 使用量日志中间件
#
# 核心作用：
#   在每次模型调用后记录 token 使用量信息，供监控和调试。
#   这是一个纯日志中间件，不修改任何状态。
#
# 工作流程：
#   1. 在 after_model（模型调用后）钩子中触发
#   2. 获取消息列表中的最后一条消息
#   3. 检查消息是否有 usage_metadata 属性
#   4. 如果有，提取 input_tokens、output_tokens、total_tokens 并记录日志
#
# 设计考虑：
#   - 这是一个纯日志中间件，不修改状态
#   - 使用标准 logger（visible in langgraph.log）
#   - 不存在 usage_metadata 时静默跳过（正常情况，不是错误）
class TokenUsageMiddleware(AgentMiddleware):
    """Logs token usage from model response usage_metadata."""

    # state_schema：类变量，指定该中间件使用的状态类型
    # 使用基础的 AgentState，因为这个中间件不修改状态
    state_schema = AgentState

    # ============================================================
    # LangChain AgentMiddleware 钩子方法
    # ============================================================

    # after_model：同步版本的模型调用后钩子
    #
    # 方法作用：
    #   LangChain AgentMiddleware 提供的扩展点，
    #   在模型执行后同步执行日志记录。
    #
    # 参数：
    #   state: AgentState，当前 agent 状态
    #   runtime: Runtime，LangGraph 运行时上下文
    #
    # 返回值：
    #   dict | None：始终返回 None（此中间件不修改状态）
    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        # 委托给 _log_usage 方法
        return self._log_usage(state)

    # aafter_model：异步版本的模型调用后钩子
    #
    # 方法作用：
    #   与 after_model 相同，但用于异步调用。
    #
    # 参数：
    #   state: AgentState，当前 agent 状态
    #   runtime: Runtime，LangGraph 运行时上下文
    #
    # 返回值：
    #   dict | None：始终返回 None
    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        # 委托给 _log_usage 方法
        return self._log_usage(state)

    # ============================================================
    # 内部辅助方法
    # ============================================================

    # _log_usage：记录 token 使用量的内部方法
    #
    # 方法作用：
    #   从状态中获取最后一条消息，提取 usage_metadata 并记录日志。
    #
    # 参数：
    #   state: AgentState，当前 agent 状态
    #
    # 返回值：
    #   None（始终返回 None，不修改状态）
    #
    # 工作流程：
    #   1. 获取消息列表
    #   2. 如果消息列表为空，直接返回
    #   3. 获取最后一条消息
    #   4. 尝试获取 usage_metadata 属性
    #   5. 如果存在，记录 input_tokens、output_tokens、total_tokens
    def _log_usage(self, state: AgentState) -> None:
        # 从状态中获取消息列表
        messages = state.get("messages", [])

        # 如果消息列表为空，直接返回
        if not messages:
            return  # 注意：这里直接 return None，但没有接收返回值

        # 获取最后一条消息
        last = messages[-1]

        # 尝试获取 usage_metadata（模型响应附加的元数据）
        # 不同的模型 provider 可能有不同的格式
        # 使用 getattr 而不是直接访问，避免属性不存在时抛出异常
        usage = getattr(last, "usage_metadata", None)

        # 如果存在 usage_metadata，记录使用量
        if usage:
            logger.info(
                "LLM token usage: input=%s output=%s total=%s",
                # 提取 input_tokens，输入 token 数，缺失时显示 "?"
                usage.get("input_tokens", "?"),
                # 提取 output_tokens，输出 token 数，缺失时显示 "?"
                usage.get("output_tokens", "?"),
                # 提取 total_tokens，总 token 数，缺失时显示 "?"
                usage.get("total_tokens", "?"),
            )

        # 始终返回 None（此中间件不修改状态）
        # LangGraph 会忽略这个返回值
        return None
