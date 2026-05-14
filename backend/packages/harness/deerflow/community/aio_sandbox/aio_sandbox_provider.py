"""AIO 沙箱提供者模块 - 管理 Docker 容器沙箱的生命周期。

本模块实现 AioSandboxProvider 类，组合 SandboxBackend（如何提供沙箱）：
- LocalContainerBackend：本地 Docker/Apple Container 模式
- RemoteSandboxBackend：远程 K8s/Provisioner 模式

提供者本身负责：
- 进程内缓存：快速重复访问
- 预热池：释放的容器保持运行，可快速回收
- 空闲超时管理：自动清理长时间空闲的容器
- 优雅关闭：信号处理确保容器被正确清理
- 挂载计算：线程特定的目录、技能目录

核心概念：
  - 确定性 sandbox_id：SHA256(thread_id)[:8]，跨进程共享
  - 两层一致性：进程内缓存（Layer 1）+ 后端发现（Layer 2）
  - 文件锁：跨进程创建同一沙箱时的序列化
  - 预热池：released 容器保持运行，idle_timeout 后销毁
  - 启动协调：进程启动时收养孤儿容器

配置示例（config.yaml）：
  sandbox:
    use: deerflow.community.aio_sandbox:AioSandboxProvider
    image: enterprise-public-cn-beijing.cr.volces.com/vefaas-public/all-in-one-sandbox:latest
    port: 8080
    idle_timeout: 600
    replicas: 3
"""

# 导入 atexit 模块，注册程序退出时的回调函数
# 使用 atexit.register() 注册 shutdown()，确保程序退出时清理所有沙箱
import atexit

# 导入 hashlib 模块，SHA256 哈希计算
# _deterministic_sandbox_id() 使用 SHA256 从 thread_id 生成确定性 ID
import hashlib

# 导入 logging 模块，记录日志
import logging

# 导入 os 模块，操作系统功能
# _resolve_env_vars() 使用 os.environ 获取环境变量
import os

# 导入 signal 模块，信号处理
# _register_signal_handlers() 注册 SIGTERM/SIGINT/SIGHUP 处理器
import signal

# 导入 threading 模块，线程编程
# _lock：保护内部状态
# _thread_locks：每个线程的锁
# _idle_checker_thread：空闲检查线程
import threading

# 导入 time 模块，时间相关
# _last_activity 使用时间戳跟踪沙箱最后活动时间
import time

# 导入 uuid 模块，生成唯一标识符
# 无 thread_id 时使用 uuid 生成随机 sandbox_id
import uuid

# 尝试导入 fcntl 模块（Unix 文件锁）
# Windows 上不可用，使用 msvcrt 作为后备
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    # Windows 上 fcntl 不可用，设为 None
    fcntl = None  # type: ignore[assignment]
    # 导入 Windows 的文件锁模块
    import msvcrt

# 从 deerflow.config 导入 get_app_config 函数
# _load_config() 使用此函数获取应用配置
from deerflow.config import get_app_config

# 导入路径相关的配置和工具函数
# VIRTUAL_PATH_PREFIX：虚拟路径前缀（如 /mnt/user-data）
# get_paths()：获取路径配置对象
from deerflow.config.paths import VIRTUAL_PATH_PREFIX, get_paths

# 从同一包的 sandbox 模块导入 Sandbox 抽象基类
# AioSandbox 继承自 Sandbox
from deerflow.sandbox.sandbox import Sandbox

# 导入 SandboxProvider 抽象基类
# AioSandboxProvider 继承自 SandboxProvider
from deerflow.sandbox.sandbox_provider import SandboxProvider

# 从当前包导入 AioSandbox 客户端类
from .aio_sandbox import AioSandbox

# 导入 SandboxBackend 抽象基类和 wait_for_sandbox_ready 函数
from .backend import SandboxBackend, wait_for_sandbox_ready

# 导入本地容器后端
from .local_backend import LocalContainerBackend

# 导入远程 K8s 后端
from .remote_backend import RemoteSandboxBackend

# 导入沙箱信息数据类
from .sandbox_info import SandboxInfo

# 获取本模块的 logger 实例
logger = logging.getLogger(__name__)

# 默认配置常量
# 默认容器镜像
DEFAULT_IMAGE = "enterprise-public-cn-beijing.cr.volces.com/vefaas-public/all-in-one-sandbox:latest"

# 默认基础端口（本地容器模式）
DEFAULT_PORT = 8080

# 默认容器名前缀
DEFAULT_CONTAINER_PREFIX = "deer-flow-sandbox"

# 默认空闲超时（秒）：10 分钟
# 沙箱空闲超过此时间后被销毁
DEFAULT_IDLE_TIMEOUT = 600

# 默认最大并发沙箱数量
# 当运行中的沙箱超过此数量时，最旧的预热池沙箱会被驱逐
DEFAULT_REPLICAS = 3

# 空闲检查间隔（秒）：每 60 秒检查一次
IDLE_CHECK_INTERVAL = 60


# _lock_file_exclusive：获取文件排他锁
#
# 参数：
#   lock_file：打开的文件对象
#
# 实现逻辑：
#   - Unix 系统：使用 fcntl.flock 获取排他锁
#   - Windows 系统：使用 msvcrt.locking 锁定文件
def _lock_file_exclusive(lock_file) -> None:
    # Unix 系统
    if fcntl is not None:
        # LOCK_EX = 排他锁（非阻塞）
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return

    # Windows 系统
    # LK_LOCK：锁定文件，如果不可用则阻塞等待
    lock_file.seek(0)
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)


# _unlock_file：释放文件锁
#
# 参数：
#   lock_file：打开的文件对象
def _unlock_file(lock_file) -> None:
    # Unix 系统
    if fcntl is not None:
        # LOCK_UN：解锁
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        return

    # Windows 系统
    lock_file.seek(0)
    # LK_UNLCK：解锁
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


# AioSandboxProvider 类：AIO 沙箱提供者
#
# 作用说明：
#   管理 Docker 容器沙箱的生命周期，提供 acquire/get/release 接口。
#   组合 SandboxBackend 实现，可以是本地 Docker 或远程 K8s。
#
# 继承关系：
#   继承自 SandboxProvider，必须实现 acquire/get/release 方法
#
# 关键特性：
#   - 两层一致性：进程内缓存 + 后端发现
#   - 预热池：released 容器保持运行，快速回收
#   - 空闲超时：自动清理长时间空闲的容器
#   - 信号处理：程序退出时正确清理
#   - 启动协调：进程启动时收养孤儿容器
#
# 配置选项（config.yaml）：
#   - image：容器镜像
#   - port：基础端口
#   - container_prefix：容器名前缀
#   - idle_timeout：空闲超时
#   - replicas：最大并发数
#   - mounts：卷挂载
#   - environment：环境变量
#   - provisioner_url：远程模式（可选）
class AioSandboxProvider(SandboxProvider):
    """Sandbox provider that manages containers running the AIO sandbox.

    Architecture:
        This provider composes a SandboxBackend (how to provision), enabling:
        - Local Docker/Apple Container mode (auto-start containers)
        - Remote/K8s mode (connect to pre-existing sandbox URL)

    Configuration options in config.yaml under sandbox:
        use: deerflow.community.aio_sandbox:AioSandboxProvider
        image: <container image>
        port: 8080                      # Base port for local containers
        container_prefix: deer-flow-sandbox
        idle_timeout: 600               # Idle timeout in seconds (0 to disable)
        replicas: 3                     # Max concurrent sandbox containers (LRU eviction when exceeded)
        mounts:                         # Volume mounts for local containers
          - host_path: /path/on/host
            container_path: /path/in/container
            read_only: false
        environment:                    # Environment variables for containers
          NODE_ENV: production
          API_KEY: $MY_API_KEY
    """

    # __init__：构造函数
    #
    # 初始化所有内部状态，创建后端，注册信号处理器，
    # 执行启动协调，启动空闲检查器。
    def __init__(self):
        # 保护内部状态的锁
        self._lock = threading.Lock()

        # 进程内沙箱缓存：sandbox_id -> AioSandbox 实例
        # 活跃使用的沙箱
        self._sandboxes: dict[str, AioSandbox] = {}

        # 沙箱信息缓存：sandbox_id -> SandboxInfo（用于 destroy）
        self._sandbox_infos: dict[str, SandboxInfo] = {}

        # 线程到沙箱的映射：thread_id -> sandbox_id
        # 用于同一线程多次调用的快速查找
        self._thread_sandboxes: dict[str, str] = {}

        # 线程锁：thread_id -> threading.Lock
        # 每个线程有独立的锁，防止同一线程的并发问题
        self._thread_locks: dict[str, threading.Lock] = {}

        # 最后活动时间：sandbox_id -> timestamp
        # 用于空闲超时检测
        self._last_activity: dict[str, float] = {}

        # 预热池：released 但容器仍在运行的沙箱
        # 格式：sandbox_id -> (SandboxInfo, release_timestamp)
        # 这些容器可以快速回收（无冷启动）或在容量不足时被驱逐
        self._warm_pool: dict[str, tuple[SandboxInfo, float]] = {}

        # 关闭标志：防止多次关闭
        self._shutdown_called = False

        # 空闲检查器停止事件
        self._idle_checker_stop = threading.Event()

        # 空闲检查器线程
        self._idle_checker_thread: threading.Thread | None = None

        # 加载配置
        self._config = self._load_config()

        # 创建后端（本地 Docker 或远程 K8s）
        self._backend: SandboxBackend = self._create_backend()

        # 注册退出时的 shutdown 处理器
        atexit.register(self.shutdown)

        # 注册信号处理器（SIGTERM/SIGINT/SIGHUP）
        self._register_signal_handlers()

        # 启动协调：收养孤儿容器
        self._reconcile_orphans()

        # 启动空闲检查器（如果启用）
        if self._config.get("idle_timeout", DEFAULT_IDLE_TIMEOUT) > 0:
            self._start_idle_checker()

    # ── Factory methods ──────────────────────────────────────────────────

    # _create_backend：创建后端
    #
    # 返回值：
    #   SandboxBackend：本地或远程后端实例
    #
    # 实现逻辑：
    #   1. 如果配置了 provisioner_url，使用 RemoteSandboxBackend
    #   2. 否则，使用 LocalContainerBackend
    def _create_backend(self) -> SandboxBackend:
        """Create the appropriate backend based on configuration.

        Selection logic (checked in order):
        1. ``provisioner_url`` set → RemoteSandboxBackend (provisioner mode)
              Provisioner dynamically creates Pods + Services in k3s.
        2. Default → LocalContainerBackend (local mode)
              Local provider manages container lifecycle directly (start/stop).
        """
        # 获取 provisioner URL
        provisioner_url = self._config.get("provisioner_url")

        # 如果有 provisioner URL，使用远程后端
        if provisioner_url:
            logger.info(f"Using remote sandbox backend with provisioner at {provisioner_url}")
            return RemoteSandboxBackend(provisioner_url=provisioner_url)

        # 否则使用本地容器后端
        logger.info("Using local container sandbox backend")
        return LocalContainerBackend(
            image=self._config["image"],
            base_port=self._config["port"],
            container_prefix=self._config["container_prefix"],
            config_mounts=self._config["mounts"],
            environment=self._config["environment"],
        )

    # ── Configuration ────────────────────────────────────────────────────

    # _load_config：加载配置
    #
    # 返回值：
    #   dict：沙箱配置字典
    #
    # 实现逻辑：
    #   从应用配置读取沙箱相关配置，提供默认值
    def _load_config(self) -> dict:
        """Load sandbox configuration from app config."""
        # 获取应用配置
        config = get_app_config()

        # 获取沙箱配置
        sandbox_config = config.sandbox

        # 读取空闲超时配置
        idle_timeout = getattr(sandbox_config, "idle_timeout", None)

        # 读取副本数配置
        replicas = getattr(sandbox_config, "replicas", None)

        # 构建配置字典，提供默认值
        return {
            # 容器镜像，默认值
            "image": sandbox_config.image or DEFAULT_IMAGE,

            # 基础端口，默认值
            "port": sandbox_config.port or DEFAULT_PORT,

            # 容器名前缀，默认值
            "container_prefix": sandbox_config.container_prefix or DEFAULT_CONTAINER_PREFIX,

            # 空闲超时，默认值
            "idle_timeout": idle_timeout if idle_timeout is not None else DEFAULT_IDLE_TIMEOUT,

            # 最大并发数，默认值
            "replicas": replicas if replicas is not None else DEFAULT_REPLICAS,

            # 卷挂载列表
            "mounts": sandbox_config.mounts or [],

            # 环境变量（解析 $ 开头的值）
            "environment": self._resolve_env_vars(sandbox_config.environment or {}),

            # provisioner URL（远程模式）
            "provisioner_url": getattr(sandbox_config, "provisioner_url", None) or "",
        }

    # _resolve_env_vars：解析环境变量引用
    #
    # 参数：
    #   env_config: dict[str, str]，环境变量配置
    #
    # 返回值：
    #   dict[str, str]，解析后的环境变量
    #
    # 实现逻辑：
    #   如果值以 $ 开头，从环境变量读取
    #   否则转换为字符串
    @staticmethod
    def _resolve_env_vars(env_config: dict[str, str]) -> dict[str, str]:
        """Resolve environment variable references (values starting with $)."""
        resolved = {}

        # 遍历所有环境变量
        for key, value in env_config.items():
            # 如果值以 $ 开头，从环境变量读取
            if isinstance(value, str) and value.startswith("$"):
                env_name = value[1:]  # 去掉 $ 前缀
                resolved[key] = os.environ.get(env_name, "")
            else:
                # 否则转换为字符串
                resolved[key] = str(value)

        return resolved

    # ── Startup reconciliation ────────────────────────────────────────────

    # _reconcile_orphans：协调孤儿容器
    #
    # 实现逻辑：
    #   进程启动时，枚举所有正在运行的容器，
    #   将它们收养到预热池中。
    #   空闲检查器会决定是回收还是销毁。
    #
    # 重要说明：
    #   所有容器都会被无条件收养，因为我们无法仅凭年龄
    #   判断容器是"孤儿的"还是"被其他进程活跃使用的"。
    #   idle_timeout 代表的是不活跃，而非运行时间。
    #   通过收养到预热池并让空闲检查器决定，
    #   避免了销毁可能仍在使用的容器。
    #
    # 这个方法解决了进程崩溃或被 kill 时容器永远运行的问题。
    def _reconcile_orphans(self) -> None:
        """Reconcile orphaned containers left by previous process lifecycles.

        On startup, enumerate all running containers matching our prefix
        and adopt them all into the warm pool.  The idle checker will reclaim
        containers that nobody re-acquires within ``idle_timeout``.

        All containers are adopted unconditionally because we cannot
        distinguish "orphaned" from "actively used by another process"
        based on age alone — ``idle_timeout`` represents inactivity, not
        uptime.  Adopting into the warm pool and letting the idle checker
        decide avoids destroying containers that a concurrent process may
        still be using.

        This closes the fundamental gap where in-memory state loss (process
        restart, crash, SIGKILL) leaves Docker containers running forever.
        """
        try:
            # 调用后端列出所有运行中的沙箱
            running = self._backend.list_running()
        except Exception as e:
            logger.warning(f"Failed to enumerate running containers during startup reconciliation: {e}")
            return

        # 如果没有运行中的容器，直接返回
        if not running:
            return

        current_time = time.time()
        adopted = 0

        # 遍历所有运行中的容器
        for info in running:
            # 计算容器年龄
            age = current_time - info.created_at if info.created_at > 0 else float("inf")

            # 单次锁获取：原子性的检查并插入
            # 避免 "已经跟踪？" 检查和 "加入预热池" 之间的 TOCTOU 窗口
            with self._lock:
                # 如果已经在追踪中，跳过
                if info.sandbox_id in self._sandboxes or info.sandbox_id in self._warm_pool:
                    continue

                # 加入预热池
                self._warm_pool[info.sandbox_id] = (info, current_time)

            adopted += 1
            logger.info(f"Adopted container {info.sandbox_id} into warm pool (age: {age:.0f}s)")

        logger.info(f"Startup reconciliation complete: {adopted} adopted into warm pool, {len(running)} total found")

    # ── Deterministic ID ─────────────────────────────────────────────────

    # _deterministic_sandbox_id：生成确定性沙箱 ID
    #
    # 参数：
    #   thread_id: str，线程 ID
    #
    # 返回值：
    #   str，SHA256 前 8 位十六进制字符串
    #
    # 实现逻辑：
    #   使用 SHA256 哈希 thread_id，取前 8 位
    #   确保相同 thread_id 在任何进程中生成相同的 sandbox_id
    #   这使得跨进程沙箱发现成为可能（无需共享状态）
    @staticmethod
    def _deterministic_sandbox_id(thread_id: str) -> str:
        """Generate a deterministic sandbox ID from a thread ID.

        Ensures all processes derive the same sandbox_id for a given thread,
        enabling cross-process sandbox discovery without shared memory.
        """
        # SHA256 哈希，取前 8 个十六进制字符
        return hashlib.sha256(thread_id.encode()).hexdigest()[:8]

    # ── Mount helpers ────────────────────────────────────────────────────

    # _get_extra_mounts：获取额外挂载列表
    #
    # 参数：
    #   thread_id: str | None，线程 ID
    #
    # 返回值：
    #   list[tuple[str, str, bool]]，挂载列表
    #   每个元素是 (主机路径, 容器路径, 只读标志)
    #
    # 实现逻辑：
    #   1. 如果有 thread_id，获取线程特定挂载
    #   2. 获取技能目录挂载
    def _get_extra_mounts(self, thread_id: str | None) -> list[tuple[str, str, bool]]:
        """Collect all extra mounts for a sandbox (thread-specific + skills)."""
        mounts: list[tuple[str, str, bool]] = []

        # 如果有 thread_id，添加线程特定挂载
        if thread_id:
            mounts.extend(self._get_thread_mounts(thread_id))
            logger.info(f"Adding thread mounts for thread {thread_id}: {mounts}")

        # 添加技能目录挂载
        skills_mount = self._get_skills_mount()
        if skills_mount:
            mounts.append(skills_mount)
            logger.info(f"Adding skills mount: {skills_mount}")

        return mounts

    # _get_thread_mounts：获取线程特定挂载
    #
    # 参数：
    #   thread_id: str，线程 ID
    #
    # 返回值：
    #   list[tuple[str, str, bool]]，挂载列表
    #
    # 实现逻辑：
    #   挂载以下目录到容器：
    #   - workspace：读写
    #   - uploads：读写
    #   - outputs：读写
    #   - acp-workspace：只读
    #
    # 注意：
    #   挂载源使用 host_base_dir，这样在 Docker 中使用 Docker socket (DooD) 时，
    #   主机 Docker daemon 可以解析路径。
    @staticmethod
    def _get_thread_mounts(thread_id: str) -> list[tuple[str, str, bool]]:
        """Get volume mounts for a thread's data directories.

        Creates directories if they don't exist (lazy initialization).
        Mount sources use host_base_dir so that when running inside Docker with a
        mounted Docker socket (DooD), the host Docker daemon can resolve the paths.
        """
        # 获取路径配置
        paths = get_paths()

        # 确保线程目录存在（惰性创建）
        paths.ensure_thread_dirs(thread_id)

        # 返回挂载列表
        # (主机路径, 容器路径, 只读标志)
        return [
            # workspace 目录
            (paths.host_sandbox_work_dir(thread_id), f"{VIRTUAL_PATH_PREFIX}/workspace", False),
            # uploads 目录
            (paths.host_sandbox_uploads_dir(thread_id), f"{VIRTUAL_PATH_PREFIX}/uploads", False),
            # outputs 目录
            (paths.host_sandbox_outputs_dir(thread_id), f"{VIRTUAL_PATH_PREFIX}/outputs", False),
            # ACP workspace：容器内只读
            # ACP 子进程从主机端写入，不是从容器内
            (paths.host_acp_workspace_dir(thread_id), "/mnt/acp-workspace", True),
        ]

    # _get_skills_mount：获取技能目录挂载
    #
    # 返回值：
    #   tuple[str, str, bool] | None：挂载元组或 None
    #
    # 实现逻辑：
    #   从配置读取技能路径，验证存在后返回挂载配置
    #   在 Docker DooD 模式下使用 DEER_FLOW_HOST_SKILLS_PATH
    @staticmethod
    def _get_skills_mount() -> tuple[str, str, bool] | None:
        """Get the skills directory mount configuration.

        Mount source uses DEER_FLOW_HOST_SKILLS_PATH when running inside Docker (DooD)
        so the host Docker daemon can resolve the path.
        """
        try:
            # 获取配置
            config = get_app_config()
            skills_path = config.skills.get_skills_path()
            container_path = config.skills.container_path

            # 验证技能目录存在
            if skills_path.exists():
                # 在 Docker DooD 模式下使用主机技能路径
                host_skills = os.environ.get("DEER_FLOW_HOST_SKILLS_PATH") or str(skills_path)
                # 返回只读挂载（安全）
                return (host_skills, container_path, True)
        except Exception as e:
            logger.warning(f"Could not setup skills mount: {e}")

        return None

    # ── Idle timeout management ──────────────────────────────────────────

    # _start_idle_checker：启动空闲检查器
    #
    # 实现逻辑：
    #   创建后台线程运行 _idle_checker_loop
    #   线程是 daemon 线程，程序退出时自动终止
    def _start_idle_checker(self) -> None:
        """Start the background thread that checks for idle sandboxes."""
        self._idle_checker_thread = threading.Thread(
            target=self._idle_checker_loop,
            name="sandbox-idle-checker",
            daemon=True,  # daemon 线程
        )
        self._idle_checker_thread.start()
        logger.info(f"Started idle checker thread (timeout: {self._config.get('idle_timeout', DEFAULT_IDLE_TIMEOUT)}s)")

    # _idle_checker_loop：空闲检查循环
    #
    # 实现逻辑：
    #   每 IDLE_CHECK_INTERVAL 秒检查一次空闲沙箱
    #   持续运行直到 _idle_checker_stop 事件被设置
    def _idle_checker_loop(self) -> None:
        """Background loop that checks for idle sandboxes."""
        # 获取空闲超时配置
        idle_timeout = self._config.get("idle_timeout", DEFAULT_IDLE_TIMEOUT)

        # 循环直到停止
        while not self._idle_checker_stop.wait(timeout=IDLE_CHECK_INTERVAL):
            try:
                # 清理空闲沙箱
                self._cleanup_idle_sandboxes(idle_timeout)
            except Exception as e:
                logger.error(f"Error in idle checker loop: {e}")

    # _cleanup_idle_sandboxes：清理空闲沙箱
    #
    # 参数：
    #   idle_timeout: float，空闲超时（秒）
    #
    # 实现逻辑：
    #   1. 遍历活跃沙箱，标记超时的
    #   2. 遍历预热池，标记超时的
    #   3. 在锁外销毁沙箱（避免长时间持锁）
    def _cleanup_idle_sandboxes(self, idle_timeout: float) -> None:
        """Check and destroy idle sandboxes."""
        current_time = time.time()
        active_to_destroy = []  # 待销毁的活跃沙箱
        warm_to_destroy: list[tuple[str, SandboxInfo]] = []  # 待销毁的预热池沙箱

        # 首先在锁内收集待销毁的沙箱
        with self._lock:
            # 活跃沙箱：检查最后活动时间
            for sandbox_id, last_activity in self._last_activity.items():
                idle_duration = current_time - last_activity
                if idle_duration > idle_timeout:
                    active_to_destroy.append(sandbox_id)
                    logger.info(f"Sandbox {sandbox_id} idle for {idle_duration:.1f}s, marking for destroy")

            # 预热池沙箱：检查 release 时间戳
            for sandbox_id, (info, release_ts) in list(self._warm_pool.items()):
                warm_duration = current_time - release_ts
                if warm_duration > idle_timeout:
                    warm_to_destroy.append((sandbox_id, info))
                    del self._warm_pool[sandbox_id]
                    logger.info(f"Warm-pool sandbox {sandbox_id} idle for {warm_duration:.1f}s, marking for destroy")

        # 销毁活跃沙箱（锁外执行）
        for sandbox_id in active_to_destroy:
            try:
                # 在销毁前重新验证仍然空闲
                # 在快照和此处之间，沙箱可能已被重新获取或释放/销毁
                with self._lock:
                    last_activity = self._last_activity.get(sandbox_id)
                    if last_activity is None:
                        # 已经释放或销毁，跳过
                        logger.info(f"Sandbox {sandbox_id} already gone before idle destroy, skipping")
                        continue
                    if (time.time() - last_activity) < idle_timeout:
                        # 已被重新获取，跳过
                        logger.info(f"Sandbox {sandbox_id} was re-acquired before idle destroy, skipping")
                        continue

                logger.info(f"Destroying idle sandbox {sandbox_id}")
                self.destroy(sandbox_id)
            except Exception as e:
                logger.error(f"Failed to destroy idle sandbox {sandbox_id}: {e}")

        # 销毁预热池沙箱（已经在锁内移除）
        for sandbox_id, info in warm_to_destroy:
            try:
                self._backend.destroy(info)
                logger.info(f"Destroyed idle warm-pool sandbox {sandbox_id}")
            except Exception as e:
                logger.error(f"Failed to destroy idle warm-pool sandbox {sandbox_id}: {e}")

    # ── Signal handling ──────────────────────────────────────────────────

    # _register_signal_handlers：注册信号处理器
    #
    # 实现逻辑：
    #   注册 SIGTERM、SIGINT、SIGHUP 处理器
    #   收到信号时调用 shutdown()，然后执行原始处理器
    #
    # 目的：
    #   确保即使用户关闭终端，容器也被正确清理
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown.

        Handles SIGTERM, SIGINT, and SIGHUP (terminal close) to ensure
        sandbox containers are cleaned up even when the user closes the terminal.
        """
        # 保存原始信号处理器
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sighup = signal.getsignal(signal.SIGHUP) if hasattr(signal, "SIGHUP") else None

        # 定义信号处理器
        def signal_handler(signum, frame):
            # 首先调用 shutdown
            self.shutdown()

            # 根据信号类型确定原始处理器
            if signum == signal.SIGTERM:
                original = self._original_sigterm
            elif hasattr(signal, "SIGHUP") and signum == signal.SIGHUP:
                original = self._original_sighup
            else:
                original = self._original_sigint

            # 执行原始处理器
            if callable(original):
                original(signum, frame)
            elif original == signal.SIG_DFL:
                # 恢复默认行为并重新引发信号
                signal.signal(signum, signal.SIG_DFL)
                signal.raise_signal(signum)

        try:
            # 注册信号处理器
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            if hasattr(signal, "SIGHUP"):
                signal.signal(signal.SIGHUP, signal_handler)
        except ValueError:
            # 非主线程无法注册信号，跳过
            logger.debug("Could not register signal handlers (not main thread)")

    # ── Thread locking (in-process) ──────────────────────────────────────

    # _get_thread_lock：获取线程锁
    #
    # 参数：
    #   thread_id: str，线程 ID
    #
    # 返回值：
    #   threading.Lock，该线程的锁
    #
    # 实现逻辑：
    #   每个线程有独立的锁，防止同一线程的并发问题
    def _get_thread_lock(self, thread_id: str) -> threading.Lock:
        """Get or create an in-process lock for a specific thread_id."""
        with self._lock:
            if thread_id not in self._thread_locks:
                self._thread_locks[thread_id] = threading.Lock()
            return self._thread_locks[thread_id]

    # ── Core: acquire / get / release / shutdown ─────────────────────────

    # acquire：获取沙箱
    #
    # 参数：
    #   thread_id: str | None，线程 ID（可选）
    #
    # 返回值：
    #   str，沙箱 ID
    #
    # 实现逻辑：
    #   对于相同 thread_id，此方法在多次调用、多个进程之间返回相同 sandbox_id
    #   线程安全，支持进程内和跨进程锁定
    def acquire(self, thread_id: str | None = None) -> str:
        """Acquire a sandbox environment and return its ID.

        For the same thread_id, this method will return the same sandbox_id
        across multiple turns, multiple processes, and (with shared storage)
        multiple pods.

        Thread-safe with both in-process and cross-process locking.

        Args:
            thread_id: Optional thread ID for thread-specific configurations.

        Returns:
            The ID of the acquired sandbox environment.
        """
        # 如果有 thread_id，先获取线程锁
        if thread_id:
            thread_lock = self._get_thread_lock(thread_id)
            with thread_lock:
                return self._acquire_internal(thread_id)
        else:
            # 无 thread_id，直接内部获取
            return self._acquire_internal(thread_id)

    # _acquire_internal：内部获取逻辑
    #
    # 参数：
    #   thread_id: str | None
    #
    # 返回值：
    #   str，沙箱 ID
    #
    # 两层一致性：
    #   Layer 1：进程内缓存（最快，覆盖同进程重复访问）
    #   Layer 2：后端发现（覆盖其他进程创建的容器）
    def _acquire_internal(self, thread_id: str | None) -> str:
        """Internal sandbox acquisition with two-layer consistency.

        Layer 1: In-process cache (fastest, covers same-process repeated access)
        Layer 2: Backend discovery (covers containers started by other processes;
                 sandbox_id is deterministic from thread_id so no shared state file
                 is needed — any process can derive the same container name)
        """
        # ── Layer 1: 进程内缓存（快速路径）──
        if thread_id:
            with self._lock:
                # 如果线程已有沙箱，检查是否仍然存在
                if thread_id in self._thread_sandboxes:
                    existing_id = self._thread_sandboxes[thread_id]
                    if existing_id in self._sandboxes:
                        # 复用现有沙箱
                        logger.info(f"Reusing in-process sandbox {existing_id} for thread {thread_id}")
                        self._last_activity[existing_id] = time.time()
                        return existing_id
                    else:
                        # 沙箱已销毁，清理映射
                        del self._thread_sandboxes[thread_id]

        # 确定性 ID（有 thread_id）或随机 ID（无 thread_id）
        sandbox_id = (
            self._deterministic_sandbox_id(thread_id)
            if thread_id
            else str(uuid.uuid4())[:8]
        )

        # ── Layer 1.5: 预热池（容器仍在运行，无冷启动）──
        if thread_id:
            with self._lock:
                if sandbox_id in self._warm_pool:
                    # 从预热池回收
                    info, _ = self._warm_pool.pop(sandbox_id)
                    sandbox = AioSandbox(id=sandbox_id, base_url=info.sandbox_url)
                    self._sandboxes[sandbox_id] = sandbox
                    self._sandbox_infos[sandbox_id] = info
                    self._last_activity[sandbox_id] = time.time()
                    self._thread_sandboxes[thread_id] = sandbox_id
                    logger.info(f"Reclaimed warm-pool sandbox {sandbox_id} for thread {thread_id} at {info.sandbox_url}")
                    return sandbox_id

        # ── Layer 2: 后端发现 + 创建（受跨进程文件锁保护）──
        # 使用文件锁，这样两个进程竞争为同一 thread_id 创建沙箱时会序列化
        # 第二个进程会发现第一个进程创建的容器，而不是产生名称冲突
        if thread_id:
            return self._discover_or_create_with_lock(thread_id, sandbox_id)

        # 无 thread_id，直接创建
        return self._create_sandbox(thread_id, sandbox_id)

    # _discover_or_create_with_lock：发现或创建沙箱（带跨进程锁）
    #
    # 参数：
    #   thread_id: str，线程 ID
    #   sandbox_id: str，沙箱 ID
    #
    # 返回值：
    #   str，沙箱 ID
    #
    # 实现逻辑：
    #   1. 获取文件锁
    #   2. 在锁内重新检查进程内缓存
    #   3. 尝试后端发现
    #   4. 如果未发现，创建新沙箱
    def _discover_or_create_with_lock(self, thread_id: str, sandbox_id: str) -> str:
        """Discover an existing sandbox or create a new one under a cross-process file lock.

        The file lock serializes concurrent sandbox creation for the same thread_id
        across multiple processes, preventing container-name conflicts.
        """
        # 获取路径配置
        paths = get_paths()

        # 确保线程目录存在
        paths.ensure_thread_dirs(thread_id)

        # 锁文件路径
        lock_path = paths.thread_dir(thread_id) / f"{sandbox_id}.lock"

        # 打开锁文件
        with open(lock_path, "a", encoding="utf-8") as lock_file:
            locked = False
            try:
                # 获取排他锁
                _lock_file_exclusive(lock_file)
                locked = True

                # 在文件锁下重新检查进程内缓存
                # （在等待锁期间，其他线程可能已创建）
                with self._lock:
                    if thread_id in self._thread_sandboxes:
                        existing_id = self._thread_sandboxes[thread_id]
                        if existing_id in self._sandboxes:
                            logger.info(f"Reusing in-process sandbox {existing_id} for thread {thread_id} (post-lock check)")
                            self._last_activity[existing_id] = time.time()
                            return existing_id

                    # 重新检查预热池
                    if sandbox_id in self._warm_pool:
                        info, _ = self._warm_pool.pop(sandbox_id)
                        sandbox = AioSandbox(id=sandbox_id, base_url=info.sandbox_url)
                        self._sandboxes[sandbox_id] = sandbox
                        self._sandbox_infos[sandbox_id] = info
                        self._last_activity[sandbox_id] = time.time()
                        self._thread_sandboxes[thread_id] = sandbox_id
                        logger.info(f"Reclaimed warm-pool sandbox {sandbox_id} for thread {thread_id} (post-lock check)")
                        return sandbox_id

                # 后端发现：其他进程可能已创建容器
                discovered = self._backend.discover(sandbox_id)
                if discovered is not None:
                    # 发现现有沙箱
                    sandbox = AioSandbox(id=discovered.sandbox_id, base_url=discovered.sandbox_url)
                    with self._lock:
                        self._sandboxes[discovered.sandbox_id] = sandbox
                        self._sandbox_infos[discovered.sandbox_id] = discovered
                        self._last_activity[discovered.sandbox_id] = time.time()
                        self._thread_sandboxes[thread_id] = discovered.sandbox_id
                    logger.info(f"Discovered existing sandbox {discovered.sandbox_id} for thread {thread_id} at {discovered.sandbox_url}")
                    return discovered.sandbox_id

                # 未发现，创建新沙箱
                return self._create_sandbox(thread_id, sandbox_id)
            finally:
                # 释放锁
                if locked:
                    _unlock_file(lock_file)

    # _evict_oldest_warm：驱逐最旧的预热池沙箱
    #
    # 返回值：
    #   str | None，被驱逐的 sandbox_id 或 None
    #
    # 实现逻辑：
    #   找到预热池中最老的沙箱（最早 release 的）并销毁
    #   用于在达到 replicas 限制时腾出容量
    def _evict_oldest_warm(self) -> str | None:
        """Destroy the oldest container in the warm pool to free capacity.

        Returns:
            The evicted sandbox_id, or None if warm pool is empty.
        """
        with self._lock:
            if not self._warm_pool:
                return None

            # 找到 release 时间最早的
            oldest_id = min(self._warm_pool, key=lambda sid: self._warm_pool[sid][1])
            info, _ = self._warm_pool.pop(oldest_id)

        try:
            # 销毁容器
            self._backend.destroy(info)
            logger.info(f"Destroyed warm-pool sandbox {oldest_id}")
        except Exception as e:
            logger.error(f"Failed to destroy warm-pool sandbox {oldest_id}: {e}")
            return None

        return oldest_id

    # _create_sandbox：创建新沙箱
    #
    # 参数：
    #   thread_id: str | None，线程 ID
    #   sandbox_id: str，沙箱 ID
    #
    # 返回值：
    #   str，沙箱 ID
    #
    # 错误：
    #   RuntimeError：沙箱创建或就绪检查失败
    def _create_sandbox(self, thread_id: str | None, sandbox_id: str) -> str:
        """Create a new sandbox via the backend.

        Args:
            thread_id: Optional thread ID.
            sandbox_id: The sandbox ID to use.

        Returns:
            The sandbox_id.

        Raises:
            RuntimeError: If sandbox creation or readiness check fails.
        """
        # 获取挂载配置
        extra_mounts = self._get_extra_mounts(thread_id)

        # 强制执行 replicas 限制
        # 只有预热池容器计入驱逐预算
        # 活跃沙箱正在服务线程，不能强制停止
        replicas = self._config.get("replicas", DEFAULT_REPLICAS)
        with self._lock:
            total = len(self._sandboxes) + len(self._warm_pool)

        if total >= replicas:
            # 尝试驱逐最旧的预热池沙箱
            evicted = self._evict_oldest_warm()
            if evicted:
                logger.info(f"Evicted warm-pool sandbox {evicted} to stay within replicas={replicas}")
            else:
                # 所有槽位都被活跃沙箱占用，继续创建并记录警告
                # replicas 是一个软限制；我们从不强制停止正在服务的容器
                logger.warning(f"All {replicas} replica slots are in active use; creating sandbox {sandbox_id} beyond the soft limit")

        # 调用后端创建沙箱
        info = self._backend.create(thread_id, sandbox_id, extra_mounts=extra_mounts or None)

        # 等待沙箱就绪
        if not wait_for_sandbox_ready(info.sandbox_url, timeout=60):
            # 就绪检查失败，销毁已创建的沙箱
            self._backend.destroy(info)
            raise RuntimeError(f"Sandbox {sandbox_id} failed to become ready within timeout at {info.sandbox_url}")

        # 创建 AioSandbox 客户端
        sandbox = AioSandbox(id=sandbox_id, base_url=info.sandbox_url)

        # 更新内部状态
        with self._lock:
            self._sandboxes[sandbox_id] = sandbox
            self._sandbox_infos[sandbox_id] = info
            self._last_activity[sandbox_id] = time.time()
            if thread_id:
                self._thread_sandboxes[thread_id] = sandbox_id

        logger.info(f"Created sandbox {sandbox_id} for thread {thread_id} at {info.sandbox_url}")
        return sandbox_id

    # get：获取沙箱实例
    #
    # 参数：
    #   sandbox_id: str，沙箱 ID
    #
    # 返回值：
    #   Sandbox | None，沙箱实例或 None
    #
    # 注意：
    #   更新最后活动时间
    def get(self, sandbox_id: str) -> Sandbox | None:
        """Get a sandbox by ID. Updates last activity timestamp.

        Args:
            sandbox_id: The ID of the sandbox.

        Returns:
            The sandbox instance if found, None otherwise.
        """
        with self._lock:
            sandbox = self._sandboxes.get(sandbox_id)
            if sandbox is not None:
                # 更新最后活动时间
                self._last_activity[sandbox_id] = time.time()
            return sandbox

    # release：释放沙箱到预热池
    #
    # 参数：
    #   sandbox_id: str，要释放的沙箱 ID
    #
    # 实现逻辑：
    #   将沙箱从活跃状态移到预热池
    #   容器保持运行，可以快速回收
    #   只在 replicas 限制强制驱逐或关闭时停止
    def release(self, sandbox_id: str) -> None:
        """Release a sandbox from active use into the warm pool.

        The container is kept running so it can be reclaimed quickly by the same
        thread on its next turn without a cold-start.  The container will only be
        stopped when the replicas limit forces eviction or during shutdown.

        Args:
            sandbox_id: The ID of the sandbox to release.
        """
        info = None
        thread_ids_to_remove: list[str] = []

        with self._lock:
            # 从活跃沙箱移除
            self._sandboxes.pop(sandbox_id, None)

            # 获取并移除沙箱信息
            info = self._sandbox_infos.pop(sandbox_id, None)

            # 找到关联的线程并移除
            thread_ids_to_remove = [tid for tid, sid in self._thread_sandboxes.items() if sid == sandbox_id]
            for tid in thread_ids_to_remove:
                del self._thread_sandboxes[tid]

            # 移除最后活动时间
            self._last_activity.pop(sandbox_id, None)

            # 加入预热池（容器保持运行）
            if info and sandbox_id not in self._warm_pool:
                self._warm_pool[sandbox_id] = (info, time.time())

        logger.info(f"Released sandbox {sandbox_id} to warm pool (container still running)")

    # destroy：销毁沙箱
    #
    # 参数：
    #   sandbox_id: str，要销毁的沙箱 ID
    #
    # 实现逻辑：
    #   与 release() 不同，destroy() 真正停止容器
    #   用于显式清理、容量驱动驱逐或关闭
    def destroy(self, sandbox_id: str) -> None:
        """Destroy a sandbox: stop the container and free all resources.

        Unlike release(), this actually stops the container.  Use this for
        explicit cleanup, capacity-driven eviction, or shutdown.

        Args:
            sandbox_id: The ID of the sandbox to destroy.
        """
        info = None
        thread_ids_to_remove: list[str] = []

        with self._lock:
            # 从所有数据结构移除
            self._sandboxes.pop(sandbox_id, None)
            info = self._sandbox_infos.pop(sandbox_id, None)
            thread_ids_to_remove = [tid for tid, sid in self._thread_sandboxes.items() if sid == sandbox_id]
            for tid in thread_ids_to_remove:
                del self._thread_sandboxes[tid]
            self._last_activity.pop(sandbox_id, None)

            # 如果 info 为 None，可能是预热池中的
            if info is None and sandbox_id in self._warm_pool:
                info, _ = self._warm_pool.pop(sandbox_id)
            else:
                self._warm_pool.pop(sandbox_id, None)

        # 调用后端销毁
        if info:
            self._backend.destroy(info)
            logger.info(f"Destroyed sandbox {sandbox_id}")

    # shutdown：关闭所有沙箱
    #
    # 线程安全，可多次调用（幂等）
    def shutdown(self) -> None:
        """Shutdown all sandboxes. Thread-safe and idempotent."""
        with self._lock:
            # 幂等检查
            if self._shutdown_called:
                return
            self._shutdown_called = True

            # 收集所有沙箱
            sandbox_ids = list(self._sandboxes.keys())
            warm_items = list(self._warm_pool.items())

            # 清空预热池
            self._warm_pool.clear()

        # 停止空闲检查器
        self._idle_checker_stop.set()
        if self._idle_checker_thread is not None and self._idle_checker_thread.is_alive():
            self._idle_checker_thread.join(timeout=5)
            logger.info("Stopped idle checker thread")

        logger.info(f"Shutting down {len(sandbox_ids)} active + {len(warm_items)} warm-pool sandbox(es)")

        # 销毁活跃沙箱
        for sandbox_id in sandbox_ids:
            try:
                self.destroy(sandbox_id)
            except Exception as e:
                logger.error(f"Failed to destroy sandbox {sandbox_id} during shutdown: {e}")

        # 销毁预热池沙箱
        for sandbox_id, (info, _) in warm_items:
            try:
                self._backend.destroy(info)
                logger.info(f"Destroyed warm-pool sandbox {sandbox_id} during shutdown")
            except Exception as e:
                logger.error(f"Failed to destroy warm-pool sandbox {sandbox_id} during shutdown: {e}")