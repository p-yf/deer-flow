"""SubagentLimitMiddleware - 子代理并发限制中间件。

功能概述：
  限制单次模型响应中的最大并发子代理（task 工具）调用数。

问题背景：
  基于 prompt 的限制（如"不要同时调用超过 3 个子代理"）不可靠，
  LLM 可能忽略这些指令。这个中间件直接在响应级别强制执行。

工作流程：
  1. after_model 钩子中，检查最后一条 AI 消息
  2. 获取 tool_calls 列表，统计有多少个 "task" 调用
  3. 如果超过 max_concurrent（默认 3，范围 [2, 4]），保留前 N 个，丢弃其余
  4. 记录警告日志
  5. 返回更新后的消息

限制范围：
  - MIN_SUBAGENT_LIMIT = 2（最小值）
  - MAX_SUBAGENT_LIMIT = 4（最大值）
  - 任何传入值都会被钳制到这个范围内

设计考虑：
  - 丢弃多余的 task 调用，而不是重新排序
  - 保留前 N 个，丢弃后面的
  - 这是一种简单的"截断"策略

执行位置：紧接 ViewImageMiddleware 之后（当 subagent_enabled 时才添加）。
"""
"""Middleware to enforce maximum concurrent subagent tool calls per model response."""

# 导入标准库 logging，用于记录日志
import logging
# typing 导入 override（方法重写标记）
from typing import override

# 从 langchain.agents 导入 AgentState（agent 基础状态类）
from langchain.agents import AgentState
# 从 langchain.agents.middleware 导入 AgentMiddleware（中间件基类）
from langchain.agents.middleware import AgentMiddleware
# 从 langgraph.runtime 导入 Runtime（LangGraph 运行时上下文）
from langgraph.runtime import Runtime

# 从 deerflow.subagents.executor 导入 MAX_CONCURRENT_SUBAGENTS
# 这是子代理执行器定义的最大并发数（值为 3）
from deerflow.subagents.executor import MAX_CONCURRENT_SUBAGENTS

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)

# 模块级常量：max_concurrent_subagents 的有效范围
MIN_SUBAGENT_LIMIT = 2  # 最小值：2
MAX_SUBAGENT_LIMIT = 4  # 最大值：4


# 模块级函数：将子代理限制值限制在有效范围内
#
# 参数：
#   value: 原始的限制值
#
# 返回值：
#   限制在 [2, 4] 范围内的值
#
# 设计考虑：
#   子代理执行器使用 3 个工作线程
#   限制范围 [2, 4] 是为了平衡并发和资源消耗
def _clamp_subagent_limit(value: int) -> int:
    """Clamp subagent limit to valid range [2, 4]."""
    # 使用 max 和 min 将值限制在 [MIN_SUBAGENT_LIMIT, MAX_SUBAGENT_LIMIT] 范围内
    return max(MIN_SUBAGENT_LIMIT, min(MAX_SUBAGENT_LIMIT, value))


# SubagentLimitMiddleware 类：限制并发子代理调用
#
# 工作原理：
#   当 LLM 在单次响应中生成超过 max_concurrent 个并行的 "task" 工具调用时
#   此中间件保留前 max_concurrent 个，丢弃其余的
#
# 为什么需要这个中间件？
#   - 基于 prompt 的限制（如"不要同时调用超过 3 个子代理"）不可靠
#   - LLM 可能忽略这些指令
#   - 这个中间件直接在响应级别强制执行，更可靠
#
# 设计考虑：
#   - 丢弃多余的 task 调用，而不是重新排序
#   - 保留前 N 个，丢弃后面的
#   - 这是一种简单的"截断"策略
class SubagentLimitMiddleware(AgentMiddleware[AgentState]):
    """Truncates excess 'task' tool calls from a single model response.

    When an LLM generates more than max_concurrent parallel task tool calls
    in one response, this middleware keeps only the first max_concurrent and
    discards the rest. This is more reliable than prompt-based limits.

    Args:
        max_concurrent: Maximum number of concurrent subagent calls allowed.
            Defaults to MAX_CONCURRENT_SUBAGENTS (3). Clamped to [2, 4].
    """

    # 构造函数
    #
    # 参数：
    #   max_concurrent: 最大并发子代理调用数，默认为 MAX_CONCURRENT_SUBAGENTS（3）
    #                  会被限制在 [2, 4] 范围内
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_SUBAGENTS):
        # 调用父类构造函数
        super().__init__()
        # 将限制值限制在有效范围内
        self.max_concurrent = _clamp_subagent_limit(max_concurrent)

    # 内部方法：截断超量的 task 调用
    #
    # 工作流程：
    #   1. 获取消息列表，检查最后一条是否是 AI 消息
    #   2. 获取 AI 消息的 tool_calls 列表
    #   3. 统计有多少个 task 调用
    #   4. 如果超过限制，保留前 N 个，丢弃其余的
    #   5. 返回更新的消息列表
    #
    # 参数：
    #   state: AgentState，当前 agent 状态
    #
    # 返回值：
    #   状态更新字典（包含更新后的消息），如果不需要截断则返回 None
    def _truncate_task_calls(self, state: AgentState) -> dict | None:
        # 从状态中获取消息列表
        messages = state.get("messages", [])
        if not messages:
            return None

        # 获取最后一条消息
        last_msg = messages[-1]
        # 确保是 AI 消息
        if getattr(last_msg, "type", None) != "ai":
            return None

        # 获取工具调用列表
        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return None

        # 找出所有 task 调用的索引
        # task_indices 是一个列表，包含所有 "task" 工具调用的位置索引
        task_indices = [i for i, tc in enumerate(tool_calls) if tc.get("name") == "task"]

        # 如果 task 调用数量不超过限制，不需要截断
        if len(task_indices) <= self.max_concurrent:
            return None

        # 计算要丢弃的索引
        # 例如：max_concurrent=3，有 5 个 task 调用
        # task_indices[3:] 给出索引 3, 4（从第 4 个开始丢弃）
        indices_to_drop = set(task_indices[self.max_concurrent:])

        # 构建截断后的 tool_calls 列表
        # 只保留不在 indices_to_drop 中的调用
        truncated_tool_calls = [tc for i, tc in enumerate(tool_calls) if i not in indices_to_drop]

        # 计算丢弃数量
        dropped_count = len(indices_to_drop)

        # 记录警告日志
        logger.warning(f"Truncated {dropped_count} excess task tool call(s) from model response (limit: {self.max_concurrent})")

        # 复制消息并更新 tool_calls
        # 注意：这里用 model_copy 创建新消息，保留相同的 id
        # LangGraph 会用 id 来识别和替换消息
        updated_msg = last_msg.model_copy(update={"tool_calls": truncated_tool_calls})

        # 返回状态更新，只包含被修改的消息
        return {"messages": [updated_msg]}

    # after_model 钩子方法：同步版本
    #
    # 在模型执行后被调用，截断超量的 task 调用
    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._truncate_task_calls(state)

    # aafter_model 钩子方法：异步版本
    #
    # 功能与同步版本相同
    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._truncate_task_calls(state)