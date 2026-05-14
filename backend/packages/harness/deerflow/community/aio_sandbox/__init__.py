"""AIO 沙箱模块 - Docker 容器沙箱实现。

本模块提供基于 Docker 容器的沙箱实现，用于生产环境隔离。

核心组件：
  - AioSandbox：连接运行中的 AIO 沙箱容器的客户端
  - AioSandboxProvider：管理容器生命周期、预热池、空闲超时的提供者
  - SandboxBackend：抽象基类，定义沙箱的创建/销毁/发现接口
  - LocalContainerBackend：本地 Docker/Apple Container 后端
  - RemoteSandboxBackend：远程 K8s provisioner 后端
  - SandboxInfo：沙箱元数据（ID、URL、容器信息）

沙箱类型：
  AioSandbox（Docker 容器）：
    * 在隔离的容器中执行命令
    * 每个线程可能有独立的容器
    * 通过 agent_sandbox 库与容器通信
    * 更强的隔离性，但更重量级

配置示例（config.yaml）：
  sandbox:
    use: deerflow.community.aio_sandbox:AioSandboxProvider
    image: enterprise-public-cn-beijing.cr.volces.com/vefaas-public/all-in-one-sandbox:latest
    port: 8080
    idle_timeout: 600
    replicas: 3
"""

from .aio_sandbox import AioSandbox
from .aio_sandbox_provider import AioSandboxProvider
from .backend import SandboxBackend
from .local_backend import LocalContainerBackend
from .remote_backend import RemoteSandboxBackend
from .sandbox_info import SandboxInfo

__all__ = [
    "AioSandbox",
    "AioSandboxProvider",
    "LocalContainerBackend",
    "RemoteSandboxBackend",
    "SandboxBackend",
    "SandboxInfo",
]
