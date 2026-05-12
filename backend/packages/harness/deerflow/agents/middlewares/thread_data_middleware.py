"""

  中间件执行顺序（从 first 到 last）：

  ┌──────┬──────────────────────────────┬─────────────────────────────────────────┐
  │ 顺序 │            中间件            │                  作用                   │
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 1    │ ThreadDataMiddleware         │ 创建线程数据目录结构                    │
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 2    │ UploadsMiddleware            │ 注入上传文件信息                      重点***│
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 3    │ SandboxMiddleware            │ 获取/释放沙箱                        重点**│
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 4    │ DanglingToolCallMiddleware   │ 修复悬空的工具调用                      │
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 5    │ ToolErrorHandlingMiddleware  │ 工具异常转错误消息                      │
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 6    │ SummarizationMiddleware      │ 上下文截断/摘要（可选，LangChain 内置） │
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 7    │ TodoMiddleware               │ TodoList 上下文丢失检测（可选）       重点***│
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 8    │ TokenUsageMiddleware         │ Token 使用量日志（可选）                   │
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 9    │ TitleMiddleware              │ 自动生成线程标题                        │
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 10   │ MemoryMiddleware             │ 记忆更新队列                        重点***│
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 11   │ ViewImageMiddleware          │ 图像详情注入（可选，需 vision 支持）    重点**│
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 12   │ DeferredToolFilterMiddleware │ 过滤延迟工具模式（可选）（需要详细看看延迟工具注册表实现） │
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 13   │ SubagentLimitMiddleware      │ 限制并发子代理数（可选）                │
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 14   │ LoopDetectionMiddleware      │ 检测/打破循环调用                    重点***│
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 15   │ SandboxAuditMiddleware       │ Bash 命令安全审计                       │
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 16   │ LLMErrorHandlingMiddleware   │ LLM 错误重试                            │
  ├──────┼──────────────────────────────┼─────────────────────────────────────────┤
  │ 17   │ ClarificationMiddleware      │ 拦截澄清请求（最后）                  重点***│
  └──────┴──────────────────────────────┴─────────────────────────────────────────┘
"""
"""ThreadDataMiddleware - 线程数据目录创建中间件。

功能概述：
  在每次线程执行前，为该线程创建独立的数据目录结构。

目录结构：
  {base_dir}/threads/{thread_id}/user-data/
    ├── workspace/    # agent 主要工作目录，常规文件操作在此进行
    ├── uploads/      # 用户上传文件存放目录
    └── outputs/     # agent 生成的文件产物目录

生命周期管理：
  - lazy_init=True（默认）：只计算路径，目录在首次使用时才创建（延迟初始化，性能优化）
  - lazy_init=False：在 before_agent() 时立即创建目录

状态更新：
  将 thread_data 字段写入状态，包含：
    - workspace_path：工作目录的绝对路径
    - uploads_path：上传目录的绝对路径
    - outputs_path：输出目录的绝对路径

执行位置：中间件链第一个执行，确保后续中间件可以使用 thread_id。
"""

# 导入标准库 logging，用于记录日志
import logging
# typing 模块导入 NotRequired（表示字段可选）和 override（标记方法重写）
from typing import NotRequired, override

# 从 langchain.agents 导入 AgentState，这是 LangChain agent 的基础状态类
# 中间件通过扩展这个类来定义自己的状态 schema
from langchain.agents import AgentState
# 从 langchain.agents.middleware 导入 AgentMiddleware，这是所有中间件的基类
# 中间件继承这个类并实现 before_agent/after_agent 等钩子方法来拦截 agent 执行流程
from langchain.agents.middleware import AgentMiddleware
# 从 langgraph.config 导入 get_config，用于获取 LangGraph 的运行时配置
from langgraph.config import get_config
# 从 langgraph.runtime 导入 Runtime，表示 LangGraph 的运行时上下文
from langgraph.runtime import Runtime

# 从 deerflow.agents.thread_state 导入 ThreadDataState，这是线程数据的状态类型定义
from deerflow.agents.thread_state import ThreadDataState
# 从 deerflow.config.paths 导入 Paths 和 get_paths
# Paths 类用于解析和构建各种路径，get_paths() 返回全局的 Paths 实例
from deerflow.config.paths import Paths, get_paths

# 创建模块级别的 logger，用于记录中间件的运行日志
logger = logging.getLogger(__name__)


# 定义中间件使用的状态类型，继承自 AgentState
# 这个类定义了该中间件需要添加到 agent state 中的字段
# ThreadDataState 包含 workspace_path、uploads_path、outputs_path 三个路径字段
class ThreadDataMiddlewareState(AgentState):
    # thread_data 字段，类型是 ThreadDataState | None
    # 使用 NotRequired 表示这个字段在状态中不是必需的（可选字段）
    thread_data: NotRequired[ThreadDataState | None]


# ThreadDataMiddleware 类，继承自 AgentMiddleware
# 它的作用是在每次线程执行前创建该线程的数据目录结构
# 目录结构包括：
#   - {base_dir}/threads/{thread_id}/user-data/workspace    （工作目录，agent 在此读写文件）
#   - {base_dir}/threads/{thread_id}/user-data/uploads      （上传文件目录，存放用户上传的文件）
#   - {base_dir}/threads/{thread_id}/user-data/outputs      （输出文件目录，agent 生成的产物放这里）
#
# 生命周期管理：
#   - lazy_init=True（默认）：只计算路径，实际目录在首次使用时才创建（延迟初始化）
#   - lazy_init=False：在 before_agent() 时立即创建目录（ eager 初始化）
class ThreadDataMiddleware(AgentMiddleware[ThreadDataMiddlewareState]):
    # state_schema 类变量，指定该中间件使用的状态类型
    # LangChain agent 用这个来验证和类型检查中间件返回的状态更新
    state_schema = ThreadDataMiddlewareState

    # 构造函数，初始化中间件实例
    #
    # 参数：
    #   base_dir: 线程数据的根目录，默认为 None（使用 Paths 默认解析）
    #   lazy_init: 为 True 时延迟创建目录（性能优化），为 False 时立即创建
    def __init__(self, base_dir: str | None = None, lazy_init: bool = True):
        # 调用父类 AgentMiddleware 的构造函数
        super().__init__()
        # 如果提供了 base_dir，用它创建 Paths 实例；否则使用全局的 Paths 实例
        # Paths 实例负责解析和构建各种沙箱相关路径
        self._paths = Paths(base_dir) if base_dir else get_paths()
        # 保存 lazy_init 配置，决定目录创建的时机
        self._lazy_init = lazy_init

    # 内部方法：获取给定 thread_id 的三个数据目录路径
    # 这个方法只计算路径，不创建任何目录
    #
    # 参数：
    #   thread_id: 线程的唯一标识符
    #
    # 返回值：
    #   一个字典，包含三个路径：
    #     - workspace_path：工作目录的绝对路径（字符串形式）
    #     - uploads_path：上传目录的绝对路径
    #     - outputs_path：输出目录的绝对路径
    #
    # 具体路径由 self._paths 的三个方法生成：
    #   - sandbox_work_dir(thread_id) 返回 Path 对象，转字符串
    #   - sandbox_uploads_dir(thread_id) 同上
    #   - sandbox_outputs_dir(thread_id) 同上
    def _get_thread_paths(self, thread_id: str) -> dict[str, str]:
        return {
            # workspace_path：agent 主要的工作目录，用于常规文件操作
            "workspace_path": str(self._paths.sandbox_work_dir(thread_id)),
            # uploads_path：用户上传文件的存放目录
            "uploads_path": str(self._paths.sandbox_uploads_dir(thread_id)),
            # outputs_path：agent 生成的输出产物目录（如生成的代码、文档等）
            "outputs_path": str(self._paths.sandbox_outputs_dir(thread_id)),
        }

    # 内部方法：创建线程的三个数据目录
    # 这个方法会实际在文件系统上创建目录
    #
    # 参数：
    #   thread_id: 线程的唯一标识符
    #
    # 返回值：
    #   与 _get_thread_paths 相同的字典结构，但目录已实际创建
    #
    # 注意：目录是通过 self._paths.ensure_thread_dirs(thread_id) 创建的
    # 这个方法内部会创建所有三个子目录
    def _create_thread_directories(self, thread_id: str) -> dict[str, str]:
        # ensure_thread_dirs 方法负责确保线程的所有目录存在（不存在则创建）
        self._paths.ensure_thread_dirs(thread_id)
        # 返回路径字典（与 _get_thread_paths 相同）
        return self._get_thread_paths(thread_id)

    # before_agent 钩子方法：在 agent 执行前被调用
    # 这个方法是 AgentMiddleware 定义的扩展点
    #
    # 参数：
    #   state: 当前 agent 的状态（ThreadDataMiddlewareState 类型）
    #   runtime: LangGraph 的运行时上下文（包含 thread_id 等信息）
    #
    # 返回值：
    #   返回一个字典，用于更新 agent 状态
    #   返回 None 表示不需要更新状态
    @override
    def before_agent(self, state: ThreadDataMiddlewareState, runtime: Runtime) -> dict | None:
        # 首先尝试从 runtime.context 获取 thread_id
        # runtime.context 是 LangGraph 传递的上下文信息字典
        context = runtime.context or {}
        thread_id = context.get("thread_id")

        # 如果 runtime.context 中没有 thread_id，尝试从 LangGraph 的全局配置获取
        # get_config() 返回配置对象，config.get("configurable", {}) 获取可配置参数
        if thread_id is None:
            config = get_config()
            thread_id = config.get("configurable", {}).get("thread_id")

        # 如果仍然没有 thread_id，抛出异常（无法确定线程就无法创建目录）
        if thread_id is None:
            raise ValueError("Thread ID is required in runtime context or config.configurable")

        # 根据 lazy_init 配置决定是计算路径还是创建目录
        if self._lazy_init:
            # lazy_init=True：只计算路径，不实际创建目录
            # 这是一种性能优化，避免不必要的文件系统操作
            # 目录会在首次使用时由工具自动创建
            paths = self._get_thread_paths(thread_id)
        else:
            # lazy_init=False：立即创建目录
            # 这确保目录在使用前一定存在
            paths = self._create_thread_directories(thread_id)
            # 记录日志：创建了哪些目录
            logger.debug("Created thread data directories for thread %s", thread_id)

        # 返回状态更新字典
        # 这个更新会被合并到 agent 的状态中
        # state 中的 thread_data 字段会包含三个路径：
        #   - workspace_path：工作目录路径
        #   - uploads_path：上传目录路径
        #   - outputs_path：输出目录路径
        # 后续的中间件和工具可以通过 state["thread_data"] 访问这些路径
        return {
            "thread_data": {
                **paths,
            }
        }
