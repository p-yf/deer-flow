"""沙箱提供者抽象基类 - 管理沙箱的获取和释放。

本模块定义了 SandboxProvider 抽象基类，负责沙箱的 lifecycle 管理。

核心概念：
  - 提供者模式：SandboxProvider 是沙箱的工厂，通过 acquire/release 方法管理沙箱
  - 单例模式：全局只有一个 SandboxProvider 实例，通过 get_sandbox_provider() 获取
  - 实现差异：
    * LocalSandboxProvider：返回本地文件系统沙箱（单例模式）
    * AioSandboxProvider：返回 Docker 容器沙箱（每次获取创建新容器）

沙箱类型：
  - LocalSandbox（本地文件系统）：
    * 使用宿主机的文件系统
    * 单例模式，整个进程共享一个沙箱实例
    * 轻量级，无额外开销
  - AioSandbox（Docker 容器）：
    * 在隔离的容器中执行命令
    * 每个线程可能有独立的容器
    * 更强的隔离性，但更重量级

使用场景：
  - 开发/测试：使用 LocalSandboxProvider，轻量快速
  - 生产环境：使用 AioSandboxProvider，提供更强隔离
"""

# 从 abc 模块导入抽象基类 ABC 和抽象方法装饰器 abstractmethod
# ABC 是 Python 内置的抽象基类元类，用于定义接口
# @abstractmethod 装饰器标记子类必须实现的方法
from abc import ABC, abstractmethod

# 从 deerflow.config 导入 get_app_config 函数
# 用于获取应用配置，读取 config.yaml 中的 sandbox.use 配置
from deerflow.config import get_app_config

# 从 deerflow.reflection 导入 resolve_class 函数
# 通过字符串路径动态加载类，例如 "deerflow.sandbox.local:LocalSandboxProvider"
from deerflow.reflection import resolve_class

# 从同一包的 sandbox 模块导入 Sandbox 抽象基类
# Sandbox 是所有沙箱实现（LocalSandbox、AioSandbox）的基类
from deerflow.sandbox.sandbox import Sandbox


# SandboxProvider 类：沙箱提供者抽象基类
#
# 作用：定义沙箱的获取和释放接口。所有沙箱提供者（如 LocalSandboxProvider）
# 必须继承此类并实现 acquire、get、release 方法。
#
# 设计模式：
#   - 提供者模式：隐藏沙箱创建细节，调用者不需要知道是本地沙箱还是容器沙箱
#   - 延迟初始化：沙箱在实际需要时才创建
#   - 单例模式：全局只有一个提供者实例
class SandboxProvider(ABC):
    """Abstract base class for sandbox providers."""

    # acquire 方法：获取沙箱
    #
    # 作用：获取一个沙箱环境并返回其唯一标识符。
    # 调用者使用返回的 sandbox_id 调用 get() 获取实际的沙箱对象。
    #
    # 参数：
    #   thread_id: str | None，线程 ID（用于关联沙箱和线程）
    #
    # 返回值：str，沙箱的唯一标识符（sandbox_id）
    #
    # 注意：
    #   - LocalSandboxProvider.acquire() 返回 "local"（单例）
    #   - AioSandboxProvider.acquire() 可能创建新的 Docker 容器
    @abstractmethod
    def acquire(self, thread_id: str | None = None) -> str:
        """Acquire a sandbox environment and return its ID.

        Returns:
            The ID of the acquired sandbox environment.
        """
        pass

    # get 方法：获取沙箱实例
    #
    # 作用：根据 sandbox_id 获取实际的沙箱实例。
    #
    # 参数：
    #   sandbox_id: str，沙箱的唯一标识符
    #
    # 返回值：Sandbox | None，如果找到则返回沙箱实例，否则返回 None
    @abstractmethod
    def get(self, sandbox_id: str) -> Sandbox | None:
        """Get a sandbox environment by ID.

        Args:
            sandbox_id: The ID of the sandbox environment to retain.
        """
        pass

    # release 方法：释放沙箱
    #
    # 作用：释放沙箱资源。对于容器沙箱，这可能意味着停止/删除容器。
    # 对于本地沙箱（单例），通常是空操作。
    #
    # 参数：
    #   sandbox_id: str，要释放的沙箱 ID
    #
    # 注意：
    #   - LocalSandboxProvider.release() 是空操作（单例模式不清理）
    #   - AioSandboxProvider.release() 可能停止 Docker 容器
    @abstractmethod
    def release(self, sandbox_id: str) -> None:
        """Release a sandbox environment.

        Args:
            sandbox_id: The ID of the sandbox environment to destroy.
        """
        pass


# 模块级变量：全局沙箱提供者单例
# 初始化为 None，在首次调用 get_sandbox_provider() 时创建
_default_sandbox_provider: SandboxProvider | None = None


# get_sandbox_provider 函数：获取沙箱提供者单例
#
# 作用：返回全局的 SandboxProvider 实例。
# 首次调用时根据配置（config.sandbox.use）创建实际提供者（本地或 Docker），
# 后续调用返回缓存的实例。
#
# 参数：**kwargs，传递给提供者构造函数的额外参数
#
# 返回值：SandboxProvider，沙箱提供者实例
#
# 工作流程：
#   1. 检查 _default_sandbox_provider 是否已有缓存实例
#   2. 如果没有，从应用配置获取沙箱提供者类路径
#   3. 使用 resolve_class 加载并实例化提供者类
#   4. 缓存实例并返回
#
# 相关函数：
#   - reset_sandbox_provider()：重置单例（不清除沙箱）
#   - shutdown_sandbox_provider()：关闭并重置单例（清除所有沙箱）
#   - set_sandbox_provider()：设置自定义提供者（用于测试）
def get_sandbox_provider(**kwargs) -> SandboxProvider:
    """Get the sandbox provider singleton.

    Returns a cached singleton instance. Use `reset_sandbox_provider()` to clear
    the cache, or `shutdown_sandbox_provider()` to properly shutdown and clear.

    Returns:
        A sandbox provider instance.
    """
    # 使用 global 关键字声明要修改模块级变量
    global _default_sandbox_provider

    # 检查缓存是否已有提供者实例（懒加载模式）
    # 如果为 None，说明还没创建过，需要创建
    if _default_sandbox_provider is None:
        # 获取应用配置对象
        config = get_app_config()

        # 使用反射加载沙箱提供者类
        # config.sandbox.use 是一个字符串，如 "deerflow.sandbox.local:LocalSandboxProvider"
        # resolve_class 会解析这个字符串，导入模块，获取类，并验证是 SandboxProvider 的子类
        cls = resolve_class(config.sandbox.use, SandboxProvider)

        # 创建提供者实例，传入额外参数
        # cls 是通过反射加载的类，这里是它的构造函数调用
        _default_sandbox_provider = cls(**kwargs)

    # 返回缓存的提供者实例
    return _default_sandbox_provider


# reset_sandbox_provider 函数：重置沙箱提供者单例
#
# 作用：清除缓存的提供者实例，但不调用 shutdown。
# 下次调用 get_sandbox_provider() 会创建新实例。
#
# 使用场景：
#   - 测试时切换配置
#   - 重置提供者状态
#
# 注意：
#   如果提供者有活跃的沙箱，它们会变成孤儿（orphaned）状态。
#   如果需要正确清理，使用 shutdown_sandbox_provider()。
def reset_sandbox_provider() -> None:
    """Reset the sandbox provider singleton.

    This clears the cached instance without calling shutdown.
    The next call to `get_sandbox_provider()` will create a new instance.
    Useful for testing or when switching configurations.

    Note: If the provider has active sandboxes, they will be orphaned.
    Use `shutdown_sandbox_provider()` for proper cleanup.
    """
    # 声明要修改模块级变量
    global _default_sandbox_provider
    # 直接设置为 None，清除缓存
    # 不会调用任何 shutdown 方法，已有的沙箱会变成孤儿
    _default_sandbox_provider = None


# shutdown_sandbox_provider 函数：关闭并重置沙箱提供者
#
# 作用：正确关闭提供者（释放所有沙箱资源），然后清除单例。
# 调用此函数后，所有活跃的沙箱都会被正确清理。
#
# 使用场景：
#   - 应用关闭时
#   - 需要完全重置沙箱系统时
#
# 工作流程：
#   1. 检查 _default_sandbox_provider 是否为 None
#   2. 如果不是，调用其 shutdown() 方法（如果有）
#   3. 清除单例引用
def shutdown_sandbox_provider() -> None:
    """Shutdown and reset the sandbox provider.

    This properly shuts down the provider (releasing all sandboxes)
    before clearing the singleton. Call this when the application
    is shutting down or when you need to completely reset the sandbox system.
    """
    # 声明要修改模块级变量
    global _default_sandbox_provider

    # 如果存在提供者实例（非 None），才进行清理
    if _default_sandbox_provider is not None:
        # 检查提供者是否有 shutdown 方法
        # 使用 hasattr 而不是直接调用，因为不是所有提供者都有 shutdown 方法
        if hasattr(_default_sandbox_provider, "shutdown"):
            # 调用 shutdown 方法清理所有沙箱资源
            # 这会停止所有 Docker 容器或清理其他资源
            _default_sandbox_provider.shutdown()

        # 清除单例引用
        _default_sandbox_provider = None


# set_sandbox_provider 函数：设置自定义沙箱提供者
#
# 作用：允许注入自定义或模拟的提供者，用于测试目的。
#
# 参数：
#   provider: SandboxProvider，要使用的提供者实例
#
# 使用场景：
#   - 单元测试时注入模拟提供者
#   - 特殊场景下使用自定义提供者
def set_sandbox_provider(provider: SandboxProvider) -> None:
    """Set a custom sandbox provider instance.

    This allows injecting a custom or mock provider for testing purposes.

    Args:
        provider: The SandboxProvider instance to use.
    """
    # 声明要修改模块级变量
    global _default_sandbox_provider
    # 直接设置为传入的提供者，替换缓存的单例
    _default_sandbox_provider = provider