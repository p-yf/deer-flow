"""本地沙箱提供者模块 - 管理本地文件系统沙箱的生命周期。

本模块实现 LocalSandboxProvider，负责：
1. 管理 LocalSandbox 单例实例
2. 配置虚拟路径到实际路径的映射（PathMapping）
3. 处理技能目录和自定义挂载的配置

核心概念：
  - 单例模式：整个进程只有一个 LocalSandbox 实例
  - PathMapping：虚拟路径（如 /mnt/skills）到实际主机路径的映射
  - 只读标志：技能目录标记为只读，防止 agent 修改

沙箱类型：
  LocalSandbox（本地文件系统）：
    * 使用宿主机的文件系统
    * 单例模式，整个进程共享一个沙箱实例
    * 轻量级，无额外开销
    * 不是真正的隔离沙箱（无容器隔离）

使用场景：
  - 开发/测试：轻量快速
  - 生产环境：使用 AioSandboxProvider 提供更强隔离
"""

# 导入 logging 模块，用于记录日志
# LocalSandboxProvider 在路径映射配置失败时记录警告日志
import logging

# 导入 Path 对象，用于处理文件系统路径
from pathlib import Path

# 从本地 local_sandbox 模块导入 LocalSandbox 类和 PathMapping 数据类
# LocalSandbox 是具体的沙箱实现
# PathMapping 定义虚拟路径到物理路径的映射规则
from deerflow.sandbox.local.local_sandbox import LocalSandbox, PathMapping

# 从同一包的 sandbox_provider 模块导入 SandboxProvider 抽象基类
# LocalSandboxProvider 继承自 SandboxProvider，必须实现 acquire/get/release 方法
from deerflow.sandbox.sandbox import Sandbox

# 导入 SandboxProvider 抽象基类
from deerflow.sandbox.sandbox_provider import SandboxProvider

# 获取本模块的 logger 实例
# 用于记录路径映射配置过程中的警告信息
logger = logging.getLogger(__name__)

# _singleton：全局 LocalSandbox 单例实例
# 初始化为 None，在首次 acquire() 时创建
# 使用模块级变量实现单例模式
_singleton: LocalSandbox | None = None


# LocalSandboxProvider 类：本地沙箱提供者
#
# 作用说明：
#   管理 LocalSandbox 的生命周期，提供线程安全的单例访问。
#   配置路径映射，允许访问技能目录和自定义挂载。
#
# 设计考虑：
#   - 单例模式：避免重复创建沙箱实例
#   - 延迟初始化：沙箱在实际需要时才创建
#   - 路径映射：支持虚拟路径到实际路径的转换
class LocalSandboxProvider(SandboxProvider):
    def __init__(self):
        """初始化本地沙箱提供者，设置路径映射。

        在构造函数中设置路径映射，包括：
        1. 技能目录映射（只读）
        2. 自定义挂载映射
        """
        # 调用 _setup_path_mappings 设置路径映射列表
        # 这些映射会被传递给 LocalSandbox 实例
        self._path_mappings = self._setup_path_mappings()

    # _setup_path_mappings：设置路径映射
    #
    # 方法作用：
    #   配置虚拟路径到实际主机路径的映射，包括技能目录和自定义挂载。
    #
    # 返回值：
    #   list[PathMapping]：路径映射列表
    #
    # 实现逻辑：
    #   1. 从应用配置获取技能目录路径和容器路径
    #   2. 如果技能目录存在，添加只读映射
    #   3. 遍历自定义挂载配置，过滤无效挂载并添加映射
    #
    # 安全考虑：
    #   - 保留的容器路径前缀（/mnt/acp-workspace, /mnt/user-data）不能被自定义挂载覆盖
    #   - 路径必须存在才能添加映射
    def _setup_path_mappings(self) -> list[PathMapping]:
        """
        Setup path mappings for local sandbox.

        Maps container paths to actual local paths, including skills directory
        and any custom mounts configured in config.yaml.

        Returns:
            List of path mappings
        """
        # 初始化空列表，存储 PathMapping 对象
        mappings: list[PathMapping] = []

        # Map skills container path to local skills directory
        try:
            # 在函数内部导入，避免顶层循环依赖
            from deerflow.config import get_app_config

            # 获取应用配置对象
            config = get_app_config()

            # 获取技能目录的本地路径
            # config.skills.get_skills_path() 返回 Path 对象
            skills_path = config.skills.get_skills_path()

            # 获取容器内的技能路径（虚拟路径）
            # 例如：/mnt/skills
            container_path = config.skills.container_path

            # Only add mapping if skills directory exists
            if skills_path.exists():
                # 添加技能目录的路径映射
                # 只读标志为 True，agent 不能修改技能文件
                mappings.append(
                    PathMapping(
                        container_path=container_path,  # 虚拟路径
                        local_path=str(skills_path),   # 实际本地路径
                        read_only=True,  # Skills directory is always read-only
                    )
                )

            # Map custom mounts from sandbox config
            # 定义保留的容器路径前缀，不能被自定义挂载覆盖
            # 这些路径用于系统功能，agent 不能修改
            _RESERVED_CONTAINER_PREFIXES = [container_path, "/mnt/acp-workspace", "/mnt/user-data"]

            # 获取沙箱配置
            sandbox_config = config.sandbox

            # 如果沙箱配置存在且有自定义挂载
            if sandbox_config and sandbox_config.mounts:
                # 遍历每个挂载配置
                for mount in sandbox_config.mounts:
                    # host_path：主机上的实际路径
                    host_path = Path(mount.host_path)

                    # container_path：容器内的虚拟路径
                    # rstrip("/") 移除末尾的斜杠，如果为空则默认为 "/"
                    container_path = mount.container_path.rstrip("/") or "/"

                    # 验证 host_path 必须是绝对路径
                    if not host_path.is_absolute():
                        logger.warning(
                            "Mount host_path must be absolute, skipping: %s -> %s",
                            mount.host_path,
                            mount.container_path,
                        )
                        # 跳过无效挂载，继续处理下一个
                        continue

                    # 验证 container_path 必须是绝对路径
                    if not container_path.startswith("/"):
                        logger.warning(
                            "Mount container_path must be absolute, skipping: %s -> %s",
                            mount.host_path,
                            mount.container_path,
                        )
                        continue

                    # Reject mounts that conflict with reserved container paths
                    # 检查是否与保留路径冲突
                    if any(container_path == p or container_path.startswith(p + "/") for p in _RESERVED_CONTAINER_PREFIXES):
                        logger.warning(
                            "Mount container_path conflicts with reserved prefix, skipping: %s",
                            mount.container_path,
                        )
                        continue

                    # Ensure the host path exists before adding mapping
                    # 只有主机路径存在时才添加映射
                    if host_path.exists():
                        mappings.append(
                            PathMapping(
                                container_path=container_path,  # 虚拟路径
                                local_path=str(host_path.resolve()),  # 解析后的绝对路径
                                read_only=mount.read_only,  # 从配置读取只读标志
                            )
                        )
                    else:
                        # 主机路径不存在，记录警告并跳过
                        logger.warning(
                            "Mount host_path does not exist, skipping: %s -> %s",
                            mount.host_path,
                            mount.container_path,
                        )
        except Exception as e:
            # Log but don't fail if config loading fails
            # 配置加载失败时记录警告但不抛出异常
            # 这样可以让沙箱在没有完整配置的情况下也能工作
            logger.warning("Could not setup path mappings: %s", e, exc_info=True)

        # 返回路径映射列表
        return mappings

    # acquire：获取沙箱
    #
    # 方法作用：
    #   获取本地沙箱实例并返回其标识符。
    #
    # 参数：
    #   thread_id: str | None，线程 ID（LocalSandbox 使用单例，忽略此参数）
    #
    # 返回值：
    #   str：沙箱标识符（固定为 "local"）
    #
    # 实现逻辑：
    #   - 检查单例是否已创建
    #   - 如果没有，创建新的 LocalSandbox 实例
    #   - 返回沙箱 ID（"local"）
    def acquire(self, thread_id: str | None = None) -> str:
        # 声明要修改模块级变量 _singleton
        # 注意：这里修改的是模块级变量，不是实例属性
        global _singleton

        # 检查单例是否已创建
        if _singleton is None:
            # 如果单例为空，创建新的 LocalSandbox 实例
            # 传入 "local" 作为沙箱 ID
            # path_mappings 包含技能目录和自定义挂载的映射
            _singleton = LocalSandbox("local", path_mappings=self._path_mappings)

        # 返回沙箱 ID
        # LocalSandbox 的 ID 固定为 "local"
        return _singleton.id

    # get：获取沙箱实例
    #
    # 方法作用：
    #   根据 sandbox_id 获取沙箱实例。
    #
    # 参数：
    #   sandbox_id: str，沙箱标识符
    #
    # 返回值：
    #   Sandbox | None：如果 ID 为 "local" 则返回单例，否则返回 None
    def get(self, sandbox_id: str) -> Sandbox | None:
        # 只有 "local" ID 才是本地沙箱
        if sandbox_id == "local":
            # 如果单例还没创建，先调用 acquire 创建
            if _singleton is None:
                self.acquire()

            # 返回单例实例
            return _singleton

        # 其他 ID 返回 None
        return None

    # release：释放沙箱
    #
    # 方法作用：
    #   释放沙箱资源（LocalSandbox 单例不需要清理）。
    #
    # 参数：
    #   sandbox_id: str，要释放的沙箱 ID
    #
    # 注意：
    #   LocalSandbox 使用单例模式，release() 是空操作。
    #   沙箱会被多个 turn 复用，不会被清理。
    #   真正的清理只在应用关闭时通过 shutdown_sandbox_provider() 进行。
    #   对于 Docker 容器提供者（如 AioSandboxProvider），清理在 shutdown() 时进行。
    def release(self, sandbox_id: str) -> None:
        # LocalSandbox uses singleton pattern - no cleanup needed.
        # Note: This method is intentionally not called by SandboxMiddleware
        # to allow sandbox reuse across multiple turns in a thread.
        # For Docker-based providers (e.g., AioSandboxProvider), cleanup
        # happens at application shutdown via the shutdown() method.
        pass