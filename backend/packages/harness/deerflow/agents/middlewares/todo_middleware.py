"""TodoMiddleware - TodoList 上下文丢失检测中间件。

功能概述：
  扩展 LangChain 的 TodoListMiddleware，检测 write_todos 上下文丢失并注入提醒。

问题背景：
  当消息历史被截断（如 SummarizationMiddleware summarization 后），
  原始的 write_todos 工具调用和其 ToolMessage 可能从活跃上下文窗口中滚出，
  导致模型忘记当前的待办列表状态。

解决方案：
  在 before_model 钩子中检测这种情况，并注入一条提醒消息：

触发条件（全部满足才注入）：
  - state.todos 非空
  - 消息历史中没有 write_todos 调用
  - 消息历史中没有已存在的 todo_reminder 提醒

提醒消息格式：
  - 使用 <system_reminder> 标签包裹
  - 包含当前待办列表的完整状态
  - 提示模型继续跟踪和更新待办列表

执行位置：紧接 SummarizationMiddleware 之后（当 is_plan_mode 时才添加）。
"""
"""Middleware to detect and inject reminders when write_todos context is lost.

Extends LangChain's TodoListMiddleware to detect when the original write_todos
tool call has been truncated from message history (e.g., after summarization)
and injects a reminder message so the model can continue tracking progress.
"""

# ============================================================
# 导入标准库
# ============================================================

# __future__ 导入 annotations，使类型注解可以引用尚未定义的类（向前引用）
from __future__ import annotations

# typing 导入：
#   - Any：任意类型，用于消息处理的类型注解
#   - override：方法重写标记，用于明确表示重写父类方法
from typing import Any, override

# ============================================================
# 导入 LangChain / LangGraph 相关模块
# ============================================================

# langchain.agents.middleware.TodoListMiddleware：
#   LangChain 提供的标准待办列表中间件
#   TodoMiddleware 继承此类以扩展功能
from langchain.agents.middleware import TodoListMiddleware

# langchain.agents.middleware.todo：
#   - PlanningState：规划状态类型，用于 before_model 参数
#   - Todo：待办项类型，用于处理待办列表
from langchain.agents.middleware.todo import PlanningState, Todo

# langchain_core.messages：
#   - AIMessage：AI 消息类型，用于检查是否有 write_todos 调用
#   - HumanMessage：人类消息类型，用于注入提醒消息
from langchain_core.messages import AIMessage, HumanMessage

# langgraph.runtime.Runtime：
#   LangGraph 运行时上下文，在钩子方法中作为参数传入
from langgraph.runtime import Runtime


# ============================================================
# 模块级辅助函数
# ============================================================

# _todos_in_messages：检查消息列表中是否存在 write_todos 工具调用
#
# 方法作用：
#   遍历所有消息，找到所有 AIMessage，检查其 tool_calls 中是否有 write_todos。
#
# 参数：
#   messages: list[Any]，消息列表
#
# 返回值：
#   bool：如果任何 AIMessage 包含 write_todos 工具调用则返回 True
def _todos_in_messages(messages: list[Any]) -> bool:
    """Return True if any AIMessage in *messages* contains a write_todos tool call."""
    # 遍历所有消息
    for msg in messages:
        # isinstance(msg, AIMessage) 检查是否是 AI 消息
        # msg.tool_calls 检查是否有工具调用
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # 遍历所有工具调用
            for tc in msg.tool_calls:
                # tc.get("name") 获取工具名称
                if tc.get("name") == "write_todos":
                    return True  # 找到了 write_todos 调用
    # 没有找到任何 write_todos 调用
    return False


# _reminder_in_messages：检查消息列表中是否已经存在 todo_reminder 提醒消息
#
# 方法作用：
#   遍历所有消息，找到所有 HumanMessage，检查其 name 属性是否为 "todo_reminder"。
#   这用于避免重复注入提醒消息。
#
# 参数：
#   messages: list[Any]，消息列表
#
# 返回值：
#   bool：如果已经存在 todo_reminder 人类消息则返回 True
def _reminder_in_messages(messages: list[Any]) -> bool:
    """Return True if a todo_reminder HumanMessage is already present in *messages*."""
    # 遍历所有消息
    for msg in messages:
        # isinstance(msg, HumanMessage) 检查是否是人类消息
        # getattr(msg, "name", None) 获取消息名称，如果不存在则返回 None
        if isinstance(msg, HumanMessage) and getattr(msg, "name", None) == "todo_reminder":
            return True  # 已经存在提醒消息
    # 没有找到提醒消息
    return False


# _format_todos：将 Todo 项列表格式化为可读字符串
#
# 方法作用：
#   将待办列表转换为人类可读的字符串格式。
#
# 参数：
#   todos: list[Todo]，待办项列表
#
# 返回值：
#   str：格式化的待办列表字符串
#
# 格式示例：
#   - [in_progress] 完成任务 A
#   - [pending] 完成任务 B
def _format_todos(todos: list[Todo]) -> str:
    """Format a list of Todo items into a human-readable string."""
    lines: list[str] = []  # 存储格式化后的每一行

    # 遍历每个待办项
    for todo in todos:
        # todo.get("status", "pending") 获取状态，默认为 "pending"
        status = todo.get("status", "pending")
        # todo.get("content", "") 获取内容，默认为空字符串
        content = todo.get("content", "")
        # 格式化为 "- [status] content" 格式
        lines.append(f"- [{status}] {content}")

    # 用换行符连接所有行
    return "\n".join(lines)


# ============================================================
# TodoMiddleware 主类
# ============================================================

# TodoMiddleware 类：待办列表中间件（扩展版本）
#
# 核心作用：
#   继承自 TodoListMiddleware（LangChain 提供的标准待办列表中间件），
#   在 before_model 钩子中检测 write_todos 上下文丢失并注入提醒。
#
# 工作原理：
#   1. 继承自 TodoListMiddleware（LangChain 提供的标准待办列表中间件）
#   2. 在 before_model 钩子中检测 write_todos 上下文丢失
#   3. 如果待办列表存在于状态中，但原始 write_todos 调用已不在消息历史中
#   4. 注入一条 HumanMessage 提醒，让模型继续跟踪待办进度
#
# 触发条件（全部满足才注入提醒）：
#   - state.todos 非空
#   - 消息历史中没有 write_todos 调用
#   - 消息历史中没有已存在的 todo_reminder 提醒
#
# 设计考虑：
#   当 SummarizationMiddleware 截断消息历史时，原始的 write_todos 调用
#   可能被移除，导致模型忘记当前有哪些待办事项。这个中间件通过注入
#   提醒消息来解决这个问题，让模型即使在上下文窗口缩小的情况下也能
#   继续正确跟踪待办列表状态。
class TodoMiddleware(TodoListMiddleware):
    """Extends TodoListMiddleware with `write_todos` context-loss detection.

    When the original `write_todos` tool call has been truncated from the message
    history (e.g., after summarization), the model loses awareness of the current
    todo list. This middleware detects that gap in `before_model` / `abefore_model`
    and injects a reminder message so the model can continue tracking progress.
    """

    # ============================================================
    # LangChain AgentMiddleware 钩子方法
    # ============================================================

    # before_model：同步版本的模型调用前钩子
    #
    # 方法作用：
    #   LangChain AgentMiddleware 提供的扩展点，
    #   在模型执行前同步执行，检测待办列表上下文丢失。
    #
    # 参数：
    #   state: PlanningState，当前规划状态
    #   runtime: Runtime，运行时上下文（未使用，但接口需要）
    #
    # 返回值：
    #   dict[str, Any] | None：状态更新（包含提醒消息），如果不需要则返回 None
    #
    # 工作流程：
    #   1. 获取待办列表（todos）
    #   2. 如果为空，不需要提醒
    #   3. 检查消息历史中是否还有 write_todos 调用
    #   4. 如果有，说明上下文完整，不需要提醒
    #   5. 检查是否已经注入过提醒消息
    #   6. 如果都没有，注入新的提醒消息
    @override
    def before_model(
        self,
        state: PlanningState,
        runtime: Runtime,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Inject a todo-list reminder when write_todos has left the context window."""
        # 从状态中获取待办列表
        # state.get("todos") 获取 todos 字段，如果不存在返回 None
        # or [] 确保结果是列表而不是 None
        todos: list[Todo] = state.get("todos") or []  # type: ignore[assignment]

        # 如果没有待办事项，不需要提醒
        if not todos:
            return None

        # 获取消息列表
        messages = state.get("messages") or []

        # 检查消息历史中是否还有 write_todos 调用
        if _todos_in_messages(messages):
            # write_todos 仍然在上下文中可见 — 不需要做任何事
            return None

        # 检查是否已经注入了提醒消息
        if _reminder_in_messages(messages):
            # 提醒已经被注入且还没有被截断
            return None

        # 待办列表存在于状态中，但原始 write_todos 调用已经消失
        # 注入一条提醒消息，让模型保持对待办列表的认知
        formatted = _format_todos(todos)  # 格式化待办列表

        # 创建提醒消息
        # 使用 HumanMessage 而不是 SystemMessage
        # 原因：Anthropic 模型要求系统消息只在对话开始时出现
        # 中间注入系统消息会导致格式错误
        reminder = HumanMessage(
            name="todo_reminder",  # 标记消息名称，用于去重检查
            content=(
                # 使用 <system_reminder> 标签包裹内容
                "<system_reminder>\n"
                "Your todo list from earlier is no longer visible in the current context window, "
                "but it is still active. Here is the current state:\n\n"
                f"{formatted}\n\n"
                "Continue tracking and updating this todo list as you work. "
                "Call `write_todos` whenever the status of any item changes.\n"
                "</system_reminder>"
            ),
        )

        # 返回状态更新，包含新消息
        # 这会让 LangGraph 的 reducer 将提醒消息添加到 messages 列表
        return {"messages": [reminder]}

    # abefore_model：异步版本的模型调用前钩子
    #
    # 方法作用：
    #   与 before_model 相同，但用于异步调用。
    #   功能与同步版本相同，直接委托给 before_model。
    #
    # 参数：
    #   state: PlanningState，当前规划状态
    #   runtime: Runtime，运行时上下文
    #
    # 返回值：
    #   dict[str, Any] | None：状态更新字典
    @override
    async def abefore_model(
        self,
        state: PlanningState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async version of before_model."""
        # 直接委托给 before_model
        return self.before_model(state, runtime)
