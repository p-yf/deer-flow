"""任务委托工具（task tool）。

功能概述：
  这是将工作委托给子智能体的核心工具。当主智能体（Lead Agent）调用 task 工具时，
  该工具会创建一个子智能体来在独立上下文中执行任务，实现任务的并行分解和委托。

工作流程：
  1. 验证子智能体类型是否可用
  2. 应用配置覆盖（超时、最大轮次、skills 等）
  3. 提取父级上下文（sandbox_state, thread_data, thread_id 等）
  4. 创建 SubagentExecutor 并启动后台执行
  5. 轮询任务状态，发送 SSE 事件（task_started, task_running, task_completed 等）
  6. 返回最终结果给调用者

SSE 事件流：
  - task_started：任务开始执行
  - task_running：子智能体生成了新的 AI 消息（包含消息内容）
  - task_completed：任务成功完成（包含结果）
  - task_failed：任务执行失败（包含错误信息）
  - task_cancelled：任务被取消
  - task_timed_out：任务执行超时

轮询机制：
  - 每 5 秒检查一次任务状态
  - 轮询超时 = 配置超时时间 + 60 秒
  - 通过 get_background_task_result() 获取实时状态
"""

import asyncio
import logging
import uuid
from dataclasses import replace
from typing import Annotated

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain.graph.config import get_stream_writer
from langgraph.typing import ContextT

from deerflow.agents.lead_agent.prompt import get_skills_prompt_section
from deerflow.agents.thread_state import ThreadState
from deerflow.sandbox.security import LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE, is_host_bash_allowed
from deerflow.subagents import SubagentExecutor, get_available_subagent_names, get_subagent_config
from deerflow.subagents.executor import SubagentStatus, cleanup_background_task, get_background_task_result, request_cancel_background_task

logger = logging.getLogger(__name__)


@tool("task", parse_docstring=True)
async def task_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,              # 任务简短描述（3-5 词），用于日志显示和追踪
    prompt: str,                   # 子智能体的任务描述，应该清晰明确
    subagent_type: str,            # 子智能体类型，如 "general-purpose" 或 "bash"
    tool_call_id: Annotated[str, InjectedToolCallId],  # 工具调用ID，用于追踪和去重
    max_turns: int | None = None,  # 可选的最大执行轮次，覆盖子智能体默认配置
) -> str:
    """Delegate a task to a specialized subagent that runs in its own context.

    子智能体帮助实现：
      - 通过分离探索和实现来保持上下文
      - 处理复杂的多个步骤的任务
      - 在隔离上下文中执行命令或操作

    可用的子智能体类型取决于活跃的沙箱配置：
      - **general-purpose**: 用于复杂的多步骤任务，需要探索和操作。
        适用于需要复杂推理、多个依赖步骤或隔离上下文的任务。
      - **bash**: 命令执行专家，用于运行 bash 命令。
        仅在明确允许主机 bash 或使用隔离 shell 沙箱（如 AioSandboxProvider）时可用。

    使用场景：
      - 需要多步骤的复杂任务
      - 产生大量输出的任务
      - 需要与主对话隔离上下文的任务
      - 并行研究或探索任务

    不使用场景：
      - 简单的单步操作（直接使用工具）
      - 需要用户交互或澄清的任务

    参数说明：
      description: 任务的简短描述（3-5 词），用于日志和显示。始终放在第一位。
      prompt: 子智能体的任务描述。要具体清晰。始终放在第二位。
      subagent_type: 子智能体类型。始终放在第三位。
      max_turns: 可选的最大智能体轮数。默认使用子智能体的配置。
    """
    # 获取当前可用的子智能体列表
    available_subagent_names = get_available_subagent_names()

    # ========== 第 1 步：验证子智能体类型 ==========

    # 获取子智能体配置（包括 config.yaml 的覆盖）
    config = get_subagent_config(subagent_type)
    if config is None:
        # 子智能体类型不存在，返回错误信息
        available = ", ".join(available_subagent_names)
        return f"Error: Unknown subagent type '{subagent_type}'. Available: {available}"

    # 检查 bash 子智能体是否可用（需要主机 bash 允许）
    if subagent_type == "bash" and not is_host_bash_allowed():
        return f"Error: {LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE}"

    # ========== 第 2 步：应用配置覆盖 ==========

    # 构建配置覆盖字典
    overrides: dict = {}

    # 添加 skills 部分到系统提示词
    # 这允许子智能体访问相同的 skills 配置
    skills_section = get_skills_prompt_section()
    if skills_section:
        overrides["system_prompt"] = config.system_prompt + "\n\n" + skills_section

    # 如果提供了 max_turns，覆盖默认配置
    if max_turns is not None:
        overrides["max_turns"] = max_turns

    # 应用覆盖到配置
    if overrides:
        config = replace(config, **overrides)

    # ========== 第 3 步：提取父级上下文 ==========

    sandbox_state = None   # 沙箱状态，用于子智能体继承相同的沙箱访问权限
    thread_data = None     # 线程数据，用于上下文传递
    thread_id = None       # 线程ID，用于沙箱操作和路径隔离
    parent_model = None    # 父级模型，用于子智能体模型选择
    trace_id = None        # 追踪ID，用于日志关联

    if runtime is not None:
        # 从 runtime.state 获取沙箱和线程数据
        sandbox_state = runtime.state.get("sandbox")
        thread_data = runtime.state.get("thread_data")

        # 从 runtime.context 获取 thread_id
        thread_id = runtime.context.get("thread_id") if runtime.context else None
        if thread_id is None:
            # 备选：从 configurable 中获取
            thread_id = runtime.config.get("configurable", {}).get("thread_id")

        # 从 metadata 中获取父级模型和追踪ID
        metadata = runtime.config.get("metadata", {})
        parent_model = metadata.get("model_name")
        trace_id = metadata.get("trace_id") or str(uuid.uuid4())[:8]

    # ========== 第 4 步：获取可用工具 ==========

    # 懒导入避免循环依赖
    from deerflow.tools import get_available_tools

    # 子智能体不应该启用子智能体工具，防止递归嵌套
    # 这确保 task 工具不在子智能体的工具列表中
    tools = get_available_tools(model_name=parent_model, subagent_enabled=False)

    # ========== 第 5 步：创建执行器并启动后台执行 ==========

    executor = SubagentExecutor(
        config=config,
        tools=tools,
        parent_model=parent_model,
        sandbox_state=sandbox_state,
        thread_data=thread_data,
        thread_id=thread_id,
        trace_id=trace_id,
    )

    # 启动后台执行（始终异步，防止阻塞主智能体）
    # 使用 tool_call_id 作为 task_id，便于追踪
    task_id = executor.execute_async(prompt, task_id=tool_call_id)

    # ========== 第 6 步：轮询任务状态 ==========

    poll_count = 0           # 轮询次数计数器
    last_status = None       # 上次状态，用于检测状态变化
    last_message_count = 0   # 上次 AI 消息数量，用于检测新消息

    # 计算最大轮询次数：超时时间 + 60 秒缓冲，每 5 秒轮询一次
    max_poll_count = (config.timeout_seconds + 60) // 5

    logger.info(f"[trace={trace_id}] Started background task {task_id} (subagent={subagent_type}, timeout={config.timeout_seconds}s, polling_limit={max_poll_count} polls)")

    # 获取 SSE writer 用于发送事件
    writer = get_stream_writer()

    # 发送任务开始事件
    writer({"type": "task_started", "task_id": task_id, "description": description})

    try:
        while True:
            # 获取任务结果
            result = get_background_task_result(task_id)

            if result is None:
                # 任务消失（不应该发生）
                logger.error(f"[trace={trace_id}] Task {task_id} not found in background tasks")
                writer({"type": "task_failed", "task_id": task_id, "error": "Task disappeared from background tasks"})
                cleanup_background_task(task_id)
                return f"Error: Task {task_id} disappeared from background tasks"

            # 记录状态变化
            if result.status != last_status:
                logger.info(f"[trace={trace_id}] Task {task_id} status: {result.status.value}")
                last_status = result.status

            # ========== 检查并发送新 AI 消息事件 ==========

            current_message_count = len(result.ai_messages)
            if current_message_count > last_message_count:
                # 有新消息，发送 task_running 事件
                for i in range(last_message_count, current_message_count):
                    message = result.ai_messages[i]
                    writer(
                        {
                            "type": "task_running",
                            "task_id": task_id,
                            "message": message,
                            "message_index": i + 1,  # 1-based 索引用于显示
                            "total_messages": current_message_count,
                        }
                    )
                    logger.info(f"[trace={trace_id}] Task {task_id} sent message #{i + 1}/{current_message_count}")
                last_message_count = current_message_count

            # ========== 检查终态 ==========

            if result.status == SubagentStatus.COMPLETED:
                writer({"type": "task_completed", "task_id": task_id, "result": result.result})
                logger.info(f"[trace={trace_id}] Task {task_id} completed after {poll_count} polls")
                cleanup_background_task(task_id)
                return f"Task Succeeded. Result: {result.result}"

            elif result.status == SubagentStatus.FAILED:
                writer({"type": "task_failed", "task_id": task_id, "error": result.error})
                logger.error(f"[trace={trace_id}] Task {task_id} failed: {result.error}")
                cleanup_background_task(task_id)
                return f"Task failed. Error: {result.error}"

            elif result.status == SubagentStatus.CANCELLED:
                writer({"type": "task_cancelled", "task_id": task_id, "error": result.error})
                logger.info(f"[trace={trace_id}] Task {task_id} cancelled: {result.error}")
                cleanup_background_task(task_id)
                return "Task cancelled by user."

            elif result.status == SubagentStatus.TIMED_OUT:
                writer({"type": "task_timed_out", "task_id": task_id, "error": result.error})
                logger.warning(f"[trace={trace_id}] Task {task_id} timed out: {result.error}")
                cleanup_background_task(task_id)
                return f"Task timed out. Error: {result.error}"

            # ========== 继续轮询 ==========

            # 等待 5 秒后继续轮询
            await asyncio.sleep(5)
            poll_count += 1

            # 轮询超时保护
            # 这是最后的安全网，处理线程池超时未能正常工作的情况
            if poll_count > max_poll_count:
                timeout_minutes = config.timeout_seconds // 60
                logger.error(f"[trace={trace_id}] Task {task_id} polling timed out after {poll_count} polls (should have been caught by thread pool timeout)")
                writer({"type": "task_timed_out", "task_id": task_id})
                return f"Task polling timed out after {timeout_minutes} minutes. This may indicate the background task is stuck. Status: {result.status.value}"

    except asyncio.CancelledError:
        # ========== 取消处理 ==========

        # 当主智能体被取消时（如用户中断），此异常被抛出
        # 协作取消：信号后台子智能体线程停止

        # 重要说明：
        # 如果没有这个处理，ThreadPoolExecutor 中的子智能体线程
        #（使用 asyncio.run() 运行自己的事件循环）会继续执行
        # 因为 ThreadPoolExecutor 无法通过 Future.cancel() 强制终止线程
        request_cancel_background_task(task_id)

        # 定义延迟清理函数：在取消后等待任务达到终态
        async def cleanup_when_done() -> None:
            max_cleanup_polls = max_poll_count
            cleanup_poll_count = 0

            while True:
                result = get_background_task_result(task_id)
                if result is None:
                    # 任务已清理
                    return

                # 检查是否达到终态
                if result.status in {SubagentStatus.COMPLETED, SubagentStatus.FAILED, SubagentStatus.CANCELLED, SubagentStatus.TIMED_OUT} or getattr(result, "completed_at", None) is not None:
                    cleanup_background_task(task_id)
                    return

                # 超时保护
                if cleanup_poll_count > max_cleanup_polls:
                    logger.warning(f"[trace={trace_id}] Deferred cleanup for task {task_id} timed out after {cleanup_poll_count} polls")
                    return

                await asyncio.sleep(5)
                cleanup_poll_count += 1

        # 定义清理失败回调
        def log_cleanup_failure(cleanup_task: asyncio.Task[None]) -> None:
            if cleanup_task.cancelled():
                return

            exc = cleanup_task.exception()
            if exc is not None:
                logger.error(f"[trace={trace_id}] Deferred cleanup failed for task {task_id}: {exc}")

        logger.debug(f"[trace={trace_id}] Scheduling deferred cleanup for cancelled task {task_id}")

        # 创建异步清理任务
        asyncio.create_task(cleanup_when_done()).add_done_callback(log_cleanup_failure)
        raise
