"""子智能体（Subagent）执行引擎。

功能概述：
  本模块是 DeerFlow 子智能体系统的核心执行引擎，负责在后台线程中运行子智能体任务。

核心组件：
  - SubagentStatus：子智能体执行状态枚举
  - SubagentResult：子智能体执行结果数据类
  - SubagentExecutor：子智能体执行器，负责创建和管理子智能体运行

线程池架构：
  - _scheduler_pool：调度线程池（3个工作线程），用于任务调度和编排
  - _execution_pool：执行线程池（3个工作线程），用于实际子智能体执行，带超时支持
  - _isolated_loop_pool：隔离循环线程池（3个工作线程），用于从已运行的事件循环中调用同步方法

执行流程：
  1. 调用 execute_async() 启动后台任务 → 返回 task_id
  2. 任务提交到 _scheduler_pool，状态设为 PENDING
  3. 调度器将任务提交到 _execution_pool，状态设为 RUNNING
  4. 如果检测到运行中的事件循环，使用 _isolated_loop_pool 创建独立事件循环
  5. 任务结果更新到全局 _background_tasks 字典
  6. task_tool 轮询获取结果，返回给主智能体
"""

import asyncio
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from deerflow.agents.thread_state import SandboxState, ThreadDataState, ThreadState
from deerflow.models import create_chat_model
from deerflow.subagents.config import SubagentConfig

logger = logging.getLogger(__name__)


class SubagentStatus(Enum):
    """子智能体执行状态枚举。

    状态转换流程：
      PENDING → RUNNING → COMPLETED / FAILED / CANCELLED / TIMED_OUT

    各状态含义：
      - PENDING：任务已创建并提交到调度器，等待调度执行
      - RUNNING：任务正在执行中
      - COMPLETED：任务成功完成
      - FAILED：任务执行失败（异常）
      - CANCELLED：任务被用户取消
      - TIMED_OUT：任务执行超时
    """

    PENDING = "pending"      # 待调度状态
    RUNNING = "running"      # 执行中状态
    COMPLETED = "completed"  # 成功完成状态
    FAILED = "failed"        # 执行失败状态
    CANCELLED = "cancelled"  # 被取消状态
    TIMED_OUT = "timed_out"  # 执行超时状态


@dataclass
class SubagentResult:
    """子智能体执行结果数据类。

    用于存储子智能体执行过程中的所有信息和最终结果。

    属性说明：
      task_id：唯一任务标识符，用于追踪和检索任务结果
      trace_id：分布式追踪ID，用于关联父级和子级日志，便于调试
      status：当前执行状态，类型为 SubagentStatus 枚举
      result：最终结果字符串（如果成功完成），否则为 None
      error：错误信息字符串（如果执行失败），否则为 None
      started_at：任务开始时间，用于计算执行时长
      completed_at：任务完成时间，用于判断超时和清理
      ai_messages：执行过程中生成的所有 AI 消息列表（用于 SSE 事件推送）
      cancel_event：协作取消事件，用于响应外部取消请求

    线程安全说明：
      所有字段修改都需要在持有 _background_tasks_lock 锁的情况下进行
    """

    task_id: str                           # 唯一任务标识符
    trace_id: str                          # 分布式追踪ID，关联父级和子级日志
    status: SubagentStatus                 # 当前执行状态
    result: str | None = None              # 最终结果（成功完成时）
    error: str | None = None               # 错误信息（执行失败时）
    started_at: datetime | None = None     # 开始执行时间
    completed_at: datetime | None = None   # 完成时间（终态时设置）
    ai_messages: list[dict[str, Any]] | None = None  # AI消息列表（用于实时推送）
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)  # 协作取消事件

    def __post_init__(self):
        """初始化可变默认值。

        使用 __post_init__ 而非 field default_factory 是为了确保 ai_messages 在每次实例化时都有独立列表，
        避免多个 SubagentResult 实例共享同一列表导致的竞态条件。
        """
        if self.ai_messages is None:
            self.ai_messages = []


# ============================================================
# 全局变量：后台任务存储和线程池
# ============================================================

# _background_tasks：进程级全局字典，存储所有后台任务的结果
# key: task_id (str)，value: SubagentResult
# 线程安全通过 _background_tasks_lock 保护
_background_tasks: dict[str, SubagentResult] = {}

# _background_tasks_lock：保护 _background_tasks 的线程锁
_background_tasks_lock = threading.Lock()

# _scheduler_pool：调度线程池，用于任务调度和编排
# - 接收 execute_async() 的任务提交
# - 设置任务状态为 RUNNING
# - 提交到 _execution_pool 执行
# - 等待执行结果并更新状态
# - 处理超时情况
_scheduler_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent-scheduler-")

# _execution_pool：执行线程池，用于实际子智能体执行
# - 接收调度器提交的执行任务
# - 调用 SubagentExecutor.execute() 运行子智能体
# - 支持超时控制（通过 execution_future.result(timeout=...) 实现）
_execution_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent-exec-")

# _isolated_loop_pool：隔离循环线程池，用于处理同步调用
# - 当从已运行的事件循环中调用 execute() 时使用
# - 创建独立的新事件循环，避免与父事件循环冲突
# - 适用于 async 上下文中的同步子智能体执行
_isolated_loop_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent-isolated-")


# ============================================================
# 辅助函数：工具过滤和模型名称解析
# ============================================================

def _filter_tools(
    all_tools: list[BaseTool],
    allowed: list[str] | None,
    disallowed: list[str] | None,
) -> list[BaseTool]:
    """根据子智能体配置过滤工具列表。

    工具过滤规则（按顺序应用）：
      1. 如果 allowed 不为 None，只保留允许列表中的工具
      2. 如果 disallowed 不为 None，排除禁用列表中的工具

    参数说明：
      all_tools：所有可用工具的完整列表
      allowed：允许的工具名称列表，None 表示不过滤（全部允许）
      disallowed：禁用的工具名称列表，None 表示不排除任何工具

    返回值：
      过滤后的工具列表

    设计考虑：
      - disallowed 优先级高于 allowed（先限制再允许）
      - 返回新列表，不修改原列表
    """
    filtered = all_tools

    # 应用允许列表过滤
    # allowed=None 表示不限制，所有工具都允许
    if allowed is not None:
        allowed_set = set(allowed)
        filtered = [t for t in filtered if t.name in allowed_set]

    # 应用禁用列表过滤
    # disallowed 用于排除特定工具，如 "task"（防止嵌套）
    if disallowed is not None:
        disallowed_set = set(disallowed)
        filtered = [t for t in filtered if t.name not in disallowed_set]

    return filtered


def _get_model_name(config: SubagentConfig, parent_model: str | None) -> str | None:
    """解析子智能体使用的模型名称。

    参数说明：
      config：子智能体配置对象
      parent_model：父级智能体使用的模型名称

    返回值：
      如果 config.model == "inherit"，返回 parent_model
      否则返回 config.model 中指定的模型名称

    设计考虑：
      - "inherit" 策略允许子智能体使用与父级相同的模型
      - 可以通过 config.yaml 覆盖为其他模型
      - 返回 None 表示使用默认模型（由 create_chat_model 处理）
    """
    if config.model == "inherit":
        return parent_model
    return config.model


# ============================================================
# 主类：SubagentExecutor
# ============================================================

class SubagentExecutor:
    """子智能体执行器。

    负责在后台线程中创建和运行子智能体。核心职责：
      1. 根据 SubagentConfig 创建子智能体实例
      2. 过滤工具列表（应用 allowlist 和 denylist）
      3. 管理执行上下文（sandbox_state, thread_data 等）
      4. 提供同步/异步/后台执行方法

    使用方式：
      1. 创建执行器实例：executor = SubagentExecutor(config, tools, ...)
      2. 同步执行：result = executor.execute(task)
      3. 异步执行：result = await executor._aexecute(task)
      4. 后台执行：task_id = executor.execute_async(task)

    线程安全说明：
      - 内部状态（config, tools 等）只读，创建后不修改
      - 结果通过 SubagentResult 存储在全局 _background_tasks 中
      - 协作取消通过 cancel_event 实现
    """

    def __init__(
        self,
        config: SubagentConfig,                     # 子智能体配置
        tools: list[BaseTool],                       # 所有可用工具（会被过滤）
        parent_model: str | None = None,            # 父级智能体模型名称（用于继承）
        sandbox_state: SandboxState | None = None,  # 沙箱状态（从父级传递）
        thread_data: ThreadDataState | None = None,  # 线程数据（从父级传递）
        thread_id: str | None = None,               # 线程ID（用于沙箱操作）
        trace_id: str | None = None,                # 追踪ID（用于日志关联）
    ):
        """初始化子智能体执行器。

        参数说明：
          config：子智能体配置，包含名称、描述、系统提示词、工具限制等
          tools：完整工具列表，会根据 config.tools 和 config.disallowed_tools 过滤
          parent_model：父级智能体使用的模型，子智能体可选择继承
          sandbox_state：父级智能体的沙箱状态，子智能体继承相同的沙箱访问权限
          thread_data：父级智能体的线程数据，用于上下文传递
          thread_id：线程唯一标识符，用于沙箱操作和路径隔离
          trace_id：分布式追踪ID，用于关联父级和子智能体的日志条目

        设计考虑：
          - 所有参数都是从父级智能体上下文传递，确保子智能体可以访问相同的资源
          - trace_id 用于在日志中追踪整个调用链
        """
        self.config = config
        self.parent_model = parent_model
        self.sandbox_state = sandbox_state
        self.thread_data = thread_data
        self.thread_id = thread_id
        # 如果没有提供 trace_id，生成一个 8 字符的短 ID 用于日志追踪
        self.trace_id = trace_id or str(uuid.uuid4())[:8]

        # 根据配置过滤工具
        # - config.tools=None 表示继承所有工具（除了 disallowed）
        # - config.disallowed_tools 默认包含 ["task"]，防止子智能体嵌套调用 task 工具
        self.tools = _filter_tools(
            tools,
            config.tools,           # 允许列表，None 表示全部允许
            config.disallowed_tools,  # 禁用列表
        )

        logger.info(f"[trace={self.trace_id}] SubagentExecutor initialized: {config.name} with {len(self.tools)} tools")

    def _create_agent(self):
        """创建子智能体实例。

        使用 LangChain 的 create_agent 工厂函数创建子智能体。
        模型使用 config.model（或继承 parent_model），工具使用过滤后的 self.tools。

        返回值：
          LangChain Agent 实例，可以调用 invoke/stream/astream 执行任务

        中间件说明：
          使用 build_subagent_runtime_middlewares 构建中间件链，
          包含工具错误处理等必要的中间件，与主智能体共享。

        状态模式说明：
          使用 ThreadState 作为状态模式，包含 messages, sandbox, thread_data 等字段
        """
        model_name = _get_model_name(self.config, self.parent_model)
        model = create_chat_model(name=model_name, thinking_enabled=False)

        from deerflow.agents.middlewares.tool_error_handling_middleware import build_subagent_runtime_middlewares

        # 复用与主智能体共享的中间件组合，确保一致的错误处理和行为
        middlewares = build_subagent_runtime_middlewares(lazy_init=True)

        return create_agent(
            model=model,
            tools=self.tools,
            middleware=middlewares,
            system_prompt=self.config.system_prompt,
            state_schema=ThreadState,
        )

    def _build_initial_state(self, task: str) -> dict[str, Any]:
        """构建子智能体初始状态。

        参数说明：
          task：任务描述字符串，作为 HumanMessage 发送给子智能体

        返回值：
          初始状态字典，包含 messages 字段（可选包含 sandbox 和 thread_data）

        设计说明：
          - 初始消息是 HumanMessage，包含任务描述
          - sandbox 和 thread_data 从父级传递，保持上下文一致性
          - 子智能体可以看到相同的沙箱状态和线程数据
        """
        state: dict[str, Any] = {
            "messages": [HumanMessage(content=task)],
        }

        # 传递父级的沙箱和线程数据，确保子智能体可以访问相同的资源
        if self.sandbox_state is not None:
            state["sandbox"] = self.sandbox_state
        if self.thread_data is not None:
            state["thread_data"] = self.thread_data

        return state

    async def _aexecute(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """异步执行任务。

        这是子智能体执行的核心方法，使用 astream 实现流式执行和实时状态更新。

        参数说明：
          task：任务描述字符串
          result_holder：可选的预创建结果对象，用于在执行过程中实时更新状态

        返回值：
          包含执行结果的 SubagentResult 对象

        执行流程：
          1. 创建或使用提供的 SubagentResult
          2. 创建子智能体实例
          3. 构建初始状态（包含任务消息和父级上下文）
          4. 使用 astream 流式执行，实时更新 ai_messages
          5. 检查 cancel_event，支持协作取消
          6. 提取最终结果并更新状态为 COMPLETED

        异常处理：
          - 捕获所有异常并设置 FAILED 状态
          - 确保 completed_at 时间戳始终被设置
        """
        # 使用提供的 result_holder（用于异步执行时实时更新）
        # 或者创建新的结果对象（用于同步执行）
        if result_holder is not None:
            result = result_holder
        else:
            task_id = str(uuid.uuid4())[:8]
            result = SubagentResult(
                task_id=task_id,
                trace_id=self.trace_id,
                status=SubagentStatus.RUNNING,
                started_at=datetime.now(),
            )

        try:
            # 创建子智能体实例
            agent = self._create_agent()
            # 构建初始状态
            state = self._build_initial_state(task)

            # 构建运行配置
            run_config: RunnableConfig = {
                "recursion_limit": self.config.max_turns,  # 最大执行轮次，防止无限循环
            }
            context = {}
            if self.thread_id:
                # 配置 thread_id 用于沙箱访问
                run_config["configurable"] = {"thread_id": self.thread_id}
                context["thread_id"] = self.thread_id

            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} starting async execution with max_turns={self.config.max_turns}")

            # 使用 astream 而非 invoke 以获取实时更新
            # stream_mode="values" 每次迭代返回完整状态快照
            final_state = None

            # 预检查：如果在流式开始前已取消，立即返回
            if result.cancel_event.is_set():
                logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} cancelled before streaming")
                with _background_tasks_lock:
                    if result.status == SubagentStatus.RUNNING:
                        result.status = SubagentStatus.CANCELLED
                        result.error = "Cancelled by user"
                        result.completed_at = datetime.now()
                return result

            # 流式执行主循环
            # 每次迭代检查 cancel_event，支持协作取消
            async for chunk in agent.astream(state, config=run_config, context=context, stream_mode="values"):  # type: ignore[arg-type]
                # 协作取消检查：在每次迭代边界检查取消事件
                # 注意：长时间运行的工具调用在单次迭代中不会被中断，
                # 只能在下一次迭代开始时检测取消请求
                if result.cancel_event.is_set():
                    logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} cancelled by parent")
                    with _background_tasks_lock:
                        if result.status == SubagentStatus.RUNNING:
                            result.status = SubagentStatus.CANCELLED
                            result.error = "Cancelled by user"
                            result.completed_at = datetime.now()
                    return result

                final_state = chunk

                # 从当前状态提取 AI 消息，用于实时推送
                messages = chunk.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    # 只处理 AIMessage 类型的消息
                    if isinstance(last_message, AIMessage):
                        message_dict = last_message.model_dump()
                        # 去重检查：避免重复添加相同消息
                        message_id = message_dict.get("id")
                        is_duplicate = False
                        if message_id:
                            # 通过 ID 检查重复
                            is_duplicate = any(msg.get("id") == message_id for msg in result.ai_messages)
                        else:
                            # 如果没有 ID，通过完整字典比较
                            is_duplicate = message_dict in result.ai_messages

                        if not is_duplicate:
                            result.ai_messages.append(message_dict)
                            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} captured AI message #{len(result.ai_messages)}")

            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} completed async execution")

            # 处理最终状态，提取结果
            if final_state is None:
                logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} no final state")
                result.result = "No response generated"
            else:
                # 从最终状态中提取消息列表
                messages = final_state.get("messages", [])
                logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} final messages count: {len(messages)}")

                # 找到最后一个 AIMessage
                last_ai_message = None
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        last_ai_message = msg
                        break

                # 处理最终消息内容
                if last_ai_message is not None:
                    content = last_ai_message.content
                    # 内容可能是 str 或 list（多模态内容块）
                    if isinstance(content, str):
                        result.result = content
                    elif isinstance(content, list):
                        # 从内容块列表中提取文本
                        # 处理混合类型：str 块和 dict 块（如 {"type": "text", "text": "..."}）
                        text_parts = []
                        pending_str_parts = []
                        for block in content:
                            if isinstance(block, str):
                                pending_str_parts.append(block)
                            elif isinstance(block, dict):
                                # 遇到 dict 块，先处理累积的字符串
                                if pending_str_parts:
                                    text_parts.append("".join(pending_str_parts))
                                    pending_str_parts.clear()
                                text_val = block.get("text")
                                if isinstance(text_val, str):
                                    text_parts.append(text_val)
                        if pending_str_parts:
                            text_parts.append("".join(pending_str_parts))
                        result.result = "\n".join(text_parts) if text_parts else "No text content in response"
                    else:
                        result.result = str(content)
                elif messages:
                    # 降级处理：没有找到 AIMessage，使用最后一条消息
                    last_message = messages[-1]
                    logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} no AIMessage found, using last message: {type(last_message)}")
                    raw_content = last_message.content if hasattr(last_message, "content") else str(last_message)
                    # 处理各种内容类型
                    if isinstance(raw_content, str):
                        result.result = raw_content
                    elif isinstance(raw_content, list):
                        parts = []
                        pending_str_parts = []
                        for block in raw_content:
                            if isinstance(block, str):
                                pending_str_parts.append(block)
                            elif isinstance(block, dict):
                                if pending_str_parts:
                                    parts.append("".join(pending_str_parts))
                                    pending_str_parts.clear()
                                text_val = block.get("text")
                                if isinstance(text_val, str):
                                    parts.append(text_val)
                        if pending_str_parts:
                            parts.append("".join(pending_str_parts))
                        result.result = "\n".join(parts) if parts else "No text content in response"
                    else:
                        result.result = str(raw_content)
                else:
                    logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} no messages in final state")
                    result.result = "No response generated"

            # 标记完成
            result.status = SubagentStatus.COMPLETED
            result.completed_at = datetime.now()

        except Exception as e:
            # 异常处理：记录异常并标记为失败
            logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} async execution failed")
            result.status = SubagentStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.now()

        return result

    def _execute_in_isolated_loop(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """在完全独立的新事件循环中执行子智能体。

        此方法专门用于从已运行的事件循环（如 async FastAPI 路由）中调用同步方法。
        创建全新的事件循环，确保子智能体的 asyncio 原语（如 httpx 客户端）不会与父事件循环冲突。

        参数说明：
          task：任务描述字符串
          result_holder：可选的预创建结果对象

        返回值：
          包含执行结果的 SubagentResult 对象

        执行流程：
          1. 保存当前事件循环（如果有）
          2. 创建新的事件循环
          3. 设置为当前事件循环
          4. 运行 _aexecute
          5. 清理：新事件循环中取消所有待处理任务、关闭异步生成器、关闭执行器
          6. 恢复原事件循环

        设计考虑：
          - 避免事件循环冲突：httpx 等库绑定到特定事件循环
          - 隔离执行：子智能体完全独立，不影响父事件循环
        """
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None

        # 创建新事件循环，确保与父事件循环隔离
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._aexecute(task, result_holder))
        finally:
            # 清理流程：
            # 1. 取消所有待处理任务
            # 2. 关闭异步生成器（shutdown_asyncgens）
            # 3. 关闭默认执行器（shutdown_default_executor）
            # 4. 关闭事件循环
            # 5. 恢复原事件循环
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task_obj in pending:
                        task_obj.cancel()
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                logger.debug(
                    f"[trace={self.trace_id}] Failed while cleaning up isolated event loop for subagent {self.config.name}",
                    exc_info=True,
                )
            finally:
                try:
                    loop.close()
                finally:
                    asyncio.set_event_loop(previous_loop)

    def execute(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """同步执行任务。

        这是 execute_async 内部调用的实际执行方法，
        包装了异步执行逻辑，提供同步接口。

        参数说明：
          task：任务描述字符串
          result_holder：可选的预创建结果对象

        返回值：
          包含执行结果的 SubagentResult 对象

        执行策略：
          1. 检测是否有运行中的事件循环
          2. 如果有（异步上下文），使用隔离线程池和独立事件循环
          3. 如果没有（同步上下文），使用 asyncio.run 直接运行

        异常处理：
          - 捕获所有异常，创建失败结果的 SubagentResult
          - 确保 status、error、completed_at 被正确设置
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                # 检测到运行中的事件循环（异步上下文）
                # 使用隔离线程池执行，避免事件循环冲突
                logger.debug(f"[trace={self.trace_id}] Subagent {self.config.name} detected running event loop, using isolated thread")
                future = _isolated_loop_pool.submit(self._execute_in_isolated_loop, task, result_holder)
                return future.result()

            # 标准路径：没有运行中的事件循环（同步上下文）
            # 使用 asyncio.run 创建新事件循环并执行
            return asyncio.run(self._aexecute(task, result_holder))
        except Exception as e:
            logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} execution failed")
            # 创建失败结果对象
            if result_holder is not None:
                result = result_holder
            else:
                result = SubagentResult(
                    task_id=str(uuid.uuid4())[:8],
                    trace_id=self.trace_id,
                    status=SubagentStatus.FAILED,
                )
            result.status = SubagentStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.now()
            return result

    def execute_async(self, task: str, task_id: str | None = None) -> str:
        """启动后台异步任务执行。

        这是子智能体执行的入口方法，被 task_tool 调用。
        将任务提交到线程池立即返回，后台执行。

        参数说明：
          task：任务描述字符串
          task_id：可选的任务ID，如果不提供则自动生成

        返回值：
          task_id 字符串，可用于后续查询任务状态和取消任务

        执行流程：
          1. 生成或使用提供的 task_id
          2. 创建初始 PENDING 状态的 SubagentResult
          3. 提交到 _scheduler_pool
          4. 调度器设置状态为 RUNNING
          5. 提交到 _execution_pool 执行（带超时）
          6. 返回 task_id

        超时处理：
          - 使用 execution_future.result(timeout=...) 设置执行超时
          - 超时后设置 cancel_event，协作取消子智能体
          - 状态设为 TIMED_OUT
        """
        # 使用提供的 task_id 或生成新的
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]

        # 创建初始待处理结果
        result = SubagentResult(
            task_id=task_id,
            trace_id=self.trace_id,
            status=SubagentStatus.PENDING,
        )

        logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} starting async execution, task_id={task_id}, timeout={self.config.timeout_seconds}s")

        with _background_tasks_lock:
            _background_tasks[task_id] = result

        # 提交到调度线程池
        def run_task():
            with _background_tasks_lock:
                _background_tasks[task_id].status = SubagentStatus.RUNNING
                _background_tasks[task_id].started_at = datetime.now()
                result_holder = _background_tasks[task_id]

            try:
                # 提交到执行线程池，带超时
                # result_holder 会被 execute() 实时更新
                execution_future: Future = _execution_pool.submit(self.execute, task, result_holder)
                try:
                    # 等待执行结果，设置超时
                    exec_result = execution_future.result(timeout=self.config.timeout_seconds)
                    # 更新最终结果
                    with _background_tasks_lock:
                        _background_tasks[task_id].status = exec_result.status
                        _background_tasks[task_id].result = exec_result.result
                        _background_tasks[task_id].error = exec_result.error
                        _background_tasks[task_id].completed_at = datetime.now()
                        _background_tasks[task_id].ai_messages = exec_result.ai_messages
                except FuturesTimeoutError:
                    # 超时处理
                    logger.error(f"[trace={self.trace_id}] Subagent {self.config.name} execution timed out after {self.config.timeout_seconds}s")
                    with _background_tasks_lock:
                        if _background_tasks[task_id].status == SubagentStatus.RUNNING:
                            _background_tasks[task_id].status = SubagentStatus.TIMED_OUT
                            _background_tasks[task_id].error = f"Execution timed out after {self.config.timeout_seconds} seconds"
                            _background_tasks[task_id].completed_at = datetime.now()
                    # 设置取消事件，协作取消子智能体线程
                    result_holder.cancel_event.set()
                    execution_future.cancel()
            except Exception as e:
                logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} async execution failed")
                with _background_tasks_lock:
                    _background_tasks[task_id].status = SubagentStatus.FAILED
                    _background_tasks[task_id].error = str(e)
                    _background_tasks[task_id].completed_at = datetime.now()

        _scheduler_pool.submit(run_task)
        return task_id


# ============================================================
# 全局常量和辅助函数
# ============================================================

# MAX_CONCURRENT_SUBAGENTS：最大并发子智能体数量常量
#
# 作用说明：
#   限制同时运行的子智能体任务的最大数量。
#   用于防止子智能体系统过载，确保系统资源合理分配。
#
# 调用位置：
#   SubagentLimitMiddleware 调用此常量检查是否需要截断多余的 task 工具调用
#   来源文件：deerflow/agents/middlewares/subagent_limit_middleware.py
#
# 设计考虑：
#   - 这是一个进程级全局常量，简化了并发控制逻辑
#   - 与 SubagentExecutor 内部的执行池大小配合使用（_execution_pool max_workers=3）
#   - 值被设置为 3，与执行池大小匹配，确保不会超过系统承载能力
MAX_CONCURRENT_SUBAGENTS = 3


def request_cancel_background_task(task_id: str) -> None:
    """请求取消后台任务。

    通过设置 cancel_event 来协作地请求子智能体停止执行。
    注意：这不会强制终止线程，只会在下次迭代边界检查时生效。

    参数说明：
      task_id：要取消的任务ID

    取消机制：
      - 设置 result.cancel_event
      - _aexecute 中的 astream 迭代会检查 cancel_event
      - 如果设置，则在下次迭代时停止执行并返回 CANCELLED 状态

    设计考虑：
      - 协作取消：子智能体有机会在安全点停止（如迭代边界）
      - 无法强制中断：长时间运行的工具调用可能不会被立即中断
      - 适用于 graceful shutdown 场景
    """
    with _background_tasks_lock:
        result = _background_tasks.get(task_id)
        if result is not None:
            result.cancel_event.set()
            logger.info("Requested cancellation for background task %s", task_id)


def get_background_task_result(task_id: str) -> SubagentResult | None:
    """获取后台任务结果。

    参数说明：
      task_id：任务ID，由 execute_async 返回

    返回值：
      SubagentResult 对象（如果找到），否则 None

    线程安全：
      通过 _background_tasks_lock 保护读取
    """
    with _background_tasks_lock:
        return _background_tasks.get(task_id)


def list_background_tasks() -> list[SubagentResult]:
    """列出所有后台任务。

    返回值：
      所有 SubagentResult 实例的列表

    用途：
      - 调试和监控
      - 列出当前运行的任务
    """
    with _background_tasks_lock:
        return list(_background_tasks.values())


def cleanup_background_task(task_id: str) -> None:
    """清理完成的后台任务。

    从 _background_tasks 中移除已完成的任务，防止内存泄漏。
    只清理终态任务（COMPLETED/FAILED/CANCELLED/TIMED_OUT），避免与后台执行器的竞态。

    参数说明：
      task_id：要清理的任务ID

    设计考虑：
      - 防止内存泄漏：已完成任务的结果不再需要保留在内存中
      - 只清理终态：避免在后台执行器更新状态时删除任务
      - 可能被多次调用：task_tool 完成轮询后会调用，scheduler 超时后也可能调用
    """
    with _background_tasks_lock:
        result = _background_tasks.get(task_id)
        if result is None:
            # 任务已不存在，可能已被清理
            logger.debug("Requested cleanup for unknown background task %s", task_id)
            return

        # 只清理终态任务，避免与后台执行器的竞态条件
        # 终态包括：COMPLETED, FAILED, CANCELLED, TIMED_OUT
        is_terminal_status = result.status in {
            SubagentStatus.COMPLETED,
            SubagentStatus.FAILED,
            SubagentStatus.CANCELLED,
            SubagentStatus.TIMED_OUT,
        }
        if is_terminal_status or result.completed_at is not None:
            del _background_tasks[task_id]
            logger.debug("Cleaned up background task: %s", task_id)
        else:
            # 非终态任务不清理，可能还在执行中
            logger.debug(
                "Skipping cleanup for non-terminal background task %s (status=%s)",
                task_id,
                result.status.value if hasattr(result.status, "value") else result.status,
            )
