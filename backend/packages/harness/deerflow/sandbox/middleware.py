"""SandboxMiddleware - 沙箱生命周期管理中间件。

功能概述：
  管理沙箱的获取和释放，确保每次线程执行都有可用的沙箱环境。

工作流程：
  1. before_agent：从 SandboxProvider 获取沙箱实例
     - 将 sandbox 对象存入状态，sandbox_id 写入配置
     - 沙箱可以是本地文件系统（LocalSandboxProvider）或 Docker 容器（AioSandboxProvider）
  2. after_agent：释放沙箱
     - 通知 SandboxProvider 该沙箱已使用完毕，可以回收或清理
     - 如果是容器模式，容器可能被停止或删除

虚拟路径系统：
  agent 看到的路径是虚拟路径（如 /mnt/user-data/workspace）
  实际路径是物理路径（如 backend/.deer-flow/threads/{thread_id}/user-data/workspace）
  路径转换由 deerflow.config.paths 模块处理

状态更新：
  - 将 sandbox 对象存入状态，供后续工具使用
  - 将 sandbox_id 写入配置，供运行时上下文使用

执行位置：中间件链第三位（紧接 UploadsMiddleware 之后）。
"""

# ============================================================
# 导入标准库
# ============================================================

# logging：标准库日志模块，用于记录中间件运行日志
import logging

# typing 导入：
#   - NotRequired：可选字段标记，用于状态字段定义
#   - override：方法重写标记，用于明确表示重写父类方法
from typing import NotRequired, override

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
# 导入 DeerFlow 项目内部模块
# ============================================================

# deerflow.agents.thread_state.SandboxState 和 ThreadDataState：
#   - SandboxState：包含 sandbox_id 等沙箱状态信息
#   - ThreadDataState：包含线程数据路径等信息（用于获取 thread_id）
#   来自：本项目 packages/harness/deerflow/agents/thread_state.py
from deerflow.agents.thread_state import SandboxState, ThreadDataState

# deerflow.sandbox.get_sandbox_provider：
#   这个函数返回全局的 SandboxProvider 实例，用于获取和释放沙箱
#   来自：本项目 packages/harness/deerflow/sandbox/__init__.py
from deerflow.sandbox import get_sandbox_provider

# ============================================================
# 模块级变量初始化
# ============================================================

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)


# ============================================================
# 状态类型定义
# ============================================================

# SandboxMiddlewareState 类：定义中间件使用的状态 schema
#
# 作用说明：
#   继承自 AgentState，作为 SandboxMiddleware 的状态类型。
#   定义了中间件需要访问和修改的状态字段。
class SandboxMiddlewareState(AgentState):
    # sandbox 字段：沙箱状态信息
    # 类型是 SandboxState | None，表示可选字段
    # SandboxState 包含 sandbox_id（沙箱唯一标识）等信息
    sandbox: NotRequired[SandboxState | None]

    # thread_data 字段：线程数据路径信息
    # 类型是 ThreadDataState | None，表示可选字段
    # 这里需要 thread_data 是因为可能需要从中获取 thread_id
    thread_data: NotRequired[ThreadDataState | None]


# ============================================================
# SandboxMiddleware 主类
# ============================================================

# SandboxMiddleware 类：沙箱生命周期管理中间件
#
# 核心作用：
#   管理沙箱的获取和释放，确保每次线程执行都有可用的沙箱环境。
#
# 沙箱的作用：
#   沙箱为 agent 提供了隔离的文件系统和命令执行环境。
#   agent 可以读写文件、执行命令，但只能访问沙箱内的资源。
#   不同线程（不同对话）使用不同的沙箱，相互隔离。
#
# 生命周期管理：
#   - lazy_init=True（默认）：沙箱在第一次工具调用时才获取（延迟初始化）
#     这样可以避免 agent 只需要简单对话时的不必要开销
#   - lazy_init=False：在 before_agent 时立即获取沙箱
#   - 沙箱在同一线程的多次交互中会被重用，不会每次都重新创建
#   - 沙箱不是在每个 agent 调用后释放，而是在应用关闭时通过
#     SandboxProvider.shutdown() 统一清理
#
# 注意：
#   这个中间件的释放逻辑在 after_agent 中，
#   但只有非 lazy_init 模式或显式设置的沙箱会被释放。
#   lazy_init 模式的沙箱获取和释放由工具执行时触发。
class SandboxMiddleware(AgentMiddleware[SandboxMiddlewareState]):
    # state_schema：类变量，指定该中间件使用的状态类型
    state_schema = SandboxMiddlewareState

    # ============================================================
    # 构造函数
    # ============================================================

    # __init__：构造函数
    #
    # 参数：
    #   lazy_init: bool，控制沙箱获取时机的布尔值
    #              - True（默认）：延迟获取，沙箱在第一次工具调用时才获取
    #              - False：立即获取，在 before_agent 时就获取沙箱
    def __init__(self, lazy_init: bool = True):
        # 调用父类 AgentMiddleware 的构造函数
        super().__init__()

        # 保存 lazy_init 配置
        self._lazy_init = lazy_init

    # ============================================================
    # 内部辅助方法
    # ============================================================

    # _acquire_sandbox：获取沙箱
    #
    # 方法作用：
    #   从 SandboxProvider 获取一个沙箱实例。
    #
    # 参数：
    #   thread_id: str，线程的唯一标识符
    #
    # 返回值：
    #   str：沙箱的唯一标识（sandbox_id）
    #
    # 工作流程：
    #   1. 获取全局的 SandboxProvider（沙箱提供者）
    #   2. 调用 provider.acquire(thread_id) 获取一个沙箱实例
    #   3. 返回沙箱的 ID
    #
    # 注意：
    #   acquire 语义取决于具体的 provider 实现：
    #   - LocalSandboxProvider：返回单例 "local"
    #   - AioSandboxProvider：可能创建新的 Docker 容器
    def _acquire_sandbox(self, thread_id: str) -> str:
        # 获取全局的 SandboxProvider 实例
        provider = get_sandbox_provider()

        # 调用 provider 的 acquire 方法，传入 thread_id
        sandbox_id = provider.acquire(thread_id)

        # 记录日志：获取了哪个沙箱
        logger.info(f"Acquiring sandbox {sandbox_id}")

        # 返回沙箱 ID
        return sandbox_id

    # ============================================================
    # LangChain AgentMiddleware 钩子方法
    # ============================================================

    # before_agent：agent 执行前的钩子
    #
    # 方法作用：
    #   LangChain AgentMiddleware 提供的扩展点，
    #   在 agent 执行前同步执行，获取沙箱环境。
    #
    # 工作逻辑：
    #   1. 如果 lazy_init=True（默认），跳过获取（沙箱会在工具调用时获取）
    #   2. 如果 lazy_init=False 且 state 中没有 sandbox，立即获取沙箱
    #   3. 返回包含 sandbox_id 的状态更新
    #
    # 参数：
    #   state: SandboxMiddlewareState，当前 agent 状态
    #   runtime: Runtime，LangGraph 运行时上下文（包含 thread_id）
    #
    # 返回值：
    #   dict | None：状态更新字典，包含 sandbox 信息
    @override
    def before_agent(self, state: SandboxMiddlewareState, runtime: Runtime) -> dict | None:
        # 如果 lazy_init=True，跳过获取操作
        # 沙箱会在第一次工具调用时由工具获取
        if self._lazy_init:
            return super().before_agent(state, runtime)

        # lazy_init=False：立即获取沙箱
        # 检查 state 中是否已有 sandbox 信息
        if "sandbox" not in state or state["sandbox"] is None:
            # 从 runtime.context 获取 thread_id
            thread_id = (runtime.context or {}).get("thread_id")
            if thread_id is None:
                # 如果没有 thread_id，无法获取沙箱
                # 调用父类方法（不做任何事）
                return super().before_agent(state, runtime)

            # 调用内部方法获取沙箱
            sandbox_id = self._acquire_sandbox(thread_id)

            # 记录日志：沙箱分配给哪个线程
            logger.info(f"Assigned sandbox {sandbox_id} to thread {thread_id}")

            # 返回状态更新，包含 sandbox 信息
            return {"sandbox": {"sandbox_id": sandbox_id}}

        # state 中已有 sandbox 信息，不需要再次获取
        return super().before_agent(state, runtime)

    # after_agent：agent 执行后的钩子
    #
    # 方法作用：
    #   LangChain AgentMiddleware 提供的扩展点，
    #   在 agent 执行后同步执行，释放沙箱环境。
    #
    # 工作逻辑：
    #   1. 检查 state 中是否有 sandbox 信息
    #   2. 如果有，调用 provider.release() 释放沙箱
    #   3. 也检查 runtime.context 中是否有 sandbox_id（备选来源）
    #
    # 参数：
    #   state: SandboxMiddlewareState，当前 agent 状态
    #   runtime: Runtime，LangGraph 运行时上下文
    #
    # 返回值：
    #   dict | None：总是返回 None（这个中间件不修改状态）
    #
    # 注意：
    #   - 对于 lazy_init=True 的情况，沙箱获取和释放通常由工具处理
    #   - 这里只处理非 lazy_init 模式下在 before_agent 获取的沙箱
    #   - LocalSandboxProvider 的 release() 是空操作（单例模式）
    @override
    def after_agent(self, state: SandboxMiddlewareState, runtime: Runtime) -> dict | None:
        # 从 state 中获取 sandbox 信息
        sandbox = state.get("sandbox")
        if sandbox is not None:
            # 从 sandbox 字典中获取 sandbox_id
            sandbox_id = sandbox["sandbox_id"]

            # 记录日志：释放沙箱
            logger.info(f"Releasing sandbox {sandbox_id}")

            # 调用 provider.release() 释放沙箱
            get_sandbox_provider().release(sandbox_id)
            return None

        # 备选：从 runtime.context 中获取 sandbox_id
        if (runtime.context or {}).get("sandbox_id") is not None:
            sandbox_id = runtime.context.get("sandbox_id")
            logger.info(f"Releasing sandbox {sandbox_id} from context")
            get_sandbox_provider().release(sandbox_id)
            return None

        # 没有沙箱需要释放
        return super().after_agent(state, runtime)