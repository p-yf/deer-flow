"""沙箱后端抽象基类模块 - 定义沙箱提供者的接口。

本模块定义 SandboxBackend 抽象基类，描述沙箱的创建、销毁、发现接口。
具体实现包括 LocalContainerBackend（本地 Docker）和 RemoteSandboxBackend（远程 K8s）。

核心概念：
  - 沙箱生命周期：create（创建）-> discover（发现）-> destroy（销毁）
  - 跨进程发现：通过确定性 ID 或容器名发现其他进程创建的沙箱
  - 后端实现差异：
    * LocalContainerBackend：直接管理本地 Docker 容器
    * RemoteSandboxBackend：通过 HTTP 与 provisioner 服务通信

架构图：
    ┌──────────────────┐
    │  SandboxBackend  │  抽象基类
    └────────┬─────────┘
             │
    ┌────────┴─────────┐
    │                  │
    ▼                  ▼
    ┌──────────────┐  ┌──────────────────┐
    │LocalContainer│  │RemoteSandboxBackend│
    │   Backend    │  │  (provisioner)   │
    └──────────────┘  └──────────────────┘
"""

# 未来类型注解支持
# 允许在类型注解中使用尚未定义的类名
from __future__ import annotations

# 导入 logging 模块，用于记录日志
# SandboxBackend 的创建、销毁等操作会记录日志
import logging

# 导入 time 模块，用于时间相关操作
# wait_for_sandbox_ready 函数使用 sleep 实现轮询等待
import time

# 导入抽象基类和抽象方法装饰器
# ABC 是抽象基类元类，abstractmethod 标记抽象方法
from abc import ABC, abstractmethod

# 导入 requests 库，用于 HTTP 请求
# wait_for_sandbox_ready 函数发送 HTTP 健康检查
import requests

# 从同一包的 sandbox_info 模块导入 SandboxInfo 数据类
# SandboxInfo 包含沙箱的连接信息（ID、URL）
from .sandbox_info import SandboxInfo

# 获取本模块的 logger 实例
logger = logging.getLogger(__name__)


# wait_for_sandbox_ready：等待沙箱就绪
#
# 参数：
#   sandbox_url: str，沙箱的 URL（如 http://k3s:30001）
#   timeout: int，最大等待时间（秒），默认 30
#
# 返回值：
#   bool：沙箱就绪返回 True，超时返回 False
#
# 实现逻辑：
#   每秒轮询沙箱的 /v1/sandbox 健康检查端点
#   如果返回 200 认为沙箱就绪
#   如果超时时间到达仍未就绪，返回 False
#
# 使用场景：
#   在创建沙箱后调用，等待容器完全启动
#   确保后续操作不会因为容器未就绪而失败
def wait_for_sandbox_ready(sandbox_url: str, timeout: int = 30) -> bool:
    """Poll sandbox health endpoint until ready or timeout.

    Args:
        sandbox_url: URL of the sandbox (e.g. http://k3s:30001).
        timeout: Maximum time to wait in seconds.

    Returns:
        True if sandbox is ready, False otherwise.
    """
    # 记录开始时间
    start_time = time.time()

    # 循环直到超时
    while time.time() - start_time < timeout:
        try:
            # 发送 GET 请求到沙箱的健康检查端点
            response = requests.get(f"{sandbox_url}/v1/sandbox", timeout=5)

            # 如果返回 200，认为沙箱就绪
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            # 请求失败（网络错误、超时等），继续等待
            pass

        # 等待 1 秒后重试
        time.sleep(1)

    # 超时，返回 False
    return False


# SandboxBackend 类：沙箱后端抽象基类
#
# 作用说明：
#   定义沙箱提供者的底层接口，管理沙箱的创建、销毁、发现。
#   AioSandboxProvider 使用后端来管理 Docker/K8s 容器。
#
# 两种实现：
#   - LocalContainerBackend：本地 Docker/Apple Container
#   - RemoteSandboxBackend：远程 K8s（通过 provisioner 服务）
#
# 核心方法：
#   - create：创建沙箱
#   - destroy：销毁沙箱
#   - is_alive：检查沙箱是否存活
#   - discover：发现已存在的沙箱
#   - list_running：列出所有运行中的沙箱
class SandboxBackend(ABC):
    """Abstract base for sandbox provisioning backends.

    Two implementations:
    - LocalContainerBackend: starts Docker/Apple Container locally, manages ports
    - RemoteSandboxBackend: connects to a pre-existing URL (K8s service, external)
    """

    # create：创建沙箱
    #
    # 参数：
    #   thread_id: str，线程 ID（用于组织沙箱，可选）
    #   sandbox_id: str，确定性的沙箱标识符
    #   extra_mounts: list[tuple[str, str, bool]] | None额外的卷挂载
    #     格式为 (主机路径, 容器路径, 只读标志) 元组列表
    #     仅被管理本地容器的后端使用（如 LocalContainerBackend）
    #     RemoteSandboxBackend 忽略此参数
    #
    # 返回值：
    #   SandboxInfo：包含连接详情（沙箱 ID、URL 等）
    #
    # 注意：
    #   这是一个抽象方法，子类必须实现
    @abstractmethod
    def create(
        self,
        thread_id: str,
        sandbox_id: str,
        extra_mounts: list[tuple[str, str, bool]] | None = None,
    ) -> SandboxInfo:
        """Create/provision a new sandbox.

        Args:
            thread_id: Thread ID for which the sandbox is being created. Useful for backends that want to organize sandboxes by thread.
            sandbox_id: Deterministic sandbox identifier.
            extra_mounts: Additional volume mounts as (host_path, container_path, read_only) tuples.
                Ignored by backends that don't manage containers (e.g., remote).

        Returns:
            SandboxInfo with connection details.
        """
        # ... 表示抽象方法，子类必须实现
        ...

    # destroy：销毁沙箱
    #
    # 参数：
    #   info: SandboxInfo，要销毁的沙箱元数据
    #
    # 注意：
    #   这是一个抽象方法，子类必须实现
    @abstractmethod
    def destroy(self, info: SandboxInfo) -> None:
        """Destroy/cleanup a sandbox and release its resources.

        Args:
            info: The sandbox metadata to destroy.
        """
        ...

    # is_alive：检查沙箱是否存活
    #
    # 参数：
    #   info: SandboxInfo，要检查的沙箱元数据
    #
    # 返回值：
    #   bool：如果沙箱看起来是存活的返回 True
    #
    # 实现注意：
    #   这应该是轻量级检查（如容器 inspect）
    #   而不是完整的健康检查
    @abstractmethod
    def is_alive(self, info: SandboxInfo) -> bool:
        """Quick check whether a sandbox is still alive.

        This should be a lightweight check (e.g., container inspect)
        rather than a full health check.

        Args:
            info: The sandbox metadata to check.

        Returns:
            True if the sandbox appears to be alive.
        """
        ...

    # discover：发现已存在的沙箱
    #
    # 参数：
    #   sandbox_id: str，要查找的确定性沙箱 ID
    #
    # 返回值：
    #   SandboxInfo | None：如果找到且健康则返回，否则返回 None
    #
    # 使用场景：
    #   跨进程恢复：当另一个进程创建了沙箱，
    #   当前进程可以通过确定性容器名或 URL 发现它
    @abstractmethod
    def discover(self, sandbox_id: str) -> SandboxInfo | None:
        """Try to discover an existing sandbox by its deterministic ID.

        Used for cross-process recovery: when another process started a sandbox,
        this process can discover it by the deterministic container name or URL.

        Args:
            sandbox_id: The deterministic sandbox ID to look for.

        Returns:
            SandboxInfo if found and healthy, None otherwise.
        """
        ...

    # list_running：列出所有运行中的沙箱
    #
    # 返回值：
    #   list[SandboxInfo]：所有当前运行中的沙箱列表
    #
    # 使用场景：
    #   启动协调：当进程重启时，需要发现之前进程启动的容器，
    #   以便将它们加入预热池或在空闲太久后销毁
    #
    # 默认实现：
    #   返回空列表，这对不管理本地容器的后端是正确的
    #   （如 RemoteSandboxBackend 将生命周期委托给 provisioner，
    #   它自己处理清理）
    def list_running(self) -> list[SandboxInfo]:
        """Enumerate all running sandboxes managed by this backend.

        Used for startup reconciliation: when the process restarts, it needs
        to discover containers started by previous processes so they can be
        adopted into the warm pool or destroyed if idle too long.

        The default implementation returns an empty list, which is correct
        for backends that don't manage local containers (e.g., RemoteSandboxBackend
        delegates lifecycle to the provisioner which handles its own cleanup).

        Returns:
            A list of SandboxInfo for all currently running sandboxes.
        """
        # 默认返回空列表
        # 子类（如 LocalContainerBackend）会重写此方法
        return []