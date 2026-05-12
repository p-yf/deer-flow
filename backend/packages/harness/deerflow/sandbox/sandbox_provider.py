from abc import ABC, abstractmethod

from deerflow.config import get_app_config
from deerflow.reflection import resolve_class
from deerflow.sandbox.sandbox import Sandbox


class SandboxProvider(ABC):
    """Abstract base class for sandbox providers"""

    @abstractmethod
    def acquire(self, thread_id: str | None = None) -> str:
        """Acquire a sandbox environment and return its ID.

        Returns:
            The ID of the acquired sandbox environment.
        """
        pass

    @abstractmethod
    def get(self, sandbox_id: str) -> Sandbox | None:
        """Get a sandbox environment by ID.

        Args:
            sandbox_id: The ID of the sandbox environment to retain.
        """
        pass

    @abstractmethod
    def release(self, sandbox_id: str) -> None:
        """Release a sandbox environment.

        Args:
            sandbox_id: The ID of the sandbox environment to destroy.
        """
        pass


_default_sandbox_provider: SandboxProvider | None = None


# get_sandbox_provider：获取沙箱提供者单例
#
# 作用说明：
#   返回全局的 SandboxProvider 单例实例。
#   首次调用时根据配置创建实际提供者（本地或 Docker），后续调用返回缓存实例。
#
# 参数：
#   - **kwargs：传递给提供者构造函数的额外参数
#
# 返回值：
#   SandboxProvider：沙箱提供者实例
#
# 调用位置：
#   SandboxMiddleware 在 before_agent 中调用以获取沙箱
#   SandboxMiddleware 在 after_agent 中调用以释放沙箱
#   来源文件：deerflow/sandbox/middleware.py
#
# 工作流程：
#   1. 检查是否已有缓存的提供者实例
#   2. 如果没有，从应用配置获取沙箱提供者类路径
#   3. 使用反射机制 resolve_class 加载并实例化提供者类
#   4. 返回实例
#
# 相关函数：
#   - reset_sandbox_provider()：重置单例（不清除沙箱）
#   - shutdown_sandbox_provider()：关闭并重置单例（清除所有沙箱）
def get_sandbox_provider(**kwargs) -> SandboxProvider:
    """Get the sandbox provider singleton.

    Returns a cached singleton instance. Use `reset_sandbox_provider()` to clear
    the cache, or `shutdown_sandbox_provider()` to properly shutdown and clear.

    Returns:
        A sandbox provider instance.
    """
    global _default_sandbox_provider
    # 检查缓存是否已有提供者实例（懒加载）
    if _default_sandbox_provider is None:
        # 获取应用配置
        config = get_app_config()
        # 使用反射加载沙箱提供者类（根据 config.sandbox.use 配置）
        cls = resolve_class(config.sandbox.use, SandboxProvider)
        # 创建提供者实例，传入额外参数
        _default_sandbox_provider = cls(**kwargs)
    return _default_sandbox_provider


def reset_sandbox_provider() -> None:
    """Reset the sandbox provider singleton.

    This clears the cached instance without calling shutdown.
    The next call to `get_sandbox_provider()` will create a new instance.
    Useful for testing or when switching configurations.

    Note: If the provider has active sandboxes, they will be orphaned.
    Use `shutdown_sandbox_provider()` for proper cleanup.
    """
    global _default_sandbox_provider
    _default_sandbox_provider = None


def shutdown_sandbox_provider() -> None:
    """Shutdown and reset the sandbox provider.

    This properly shuts down the provider (releasing all sandboxes)
    before clearing the singleton. Call this when the application
    is shutting down or when you need to completely reset the sandbox system.
    """
    global _default_sandbox_provider
    if _default_sandbox_provider is not None:
        if hasattr(_default_sandbox_provider, "shutdown"):
            _default_sandbox_provider.shutdown()
        _default_sandbox_provider = None


def set_sandbox_provider(provider: SandboxProvider) -> None:
    """Set a custom sandbox provider instance.

    This allows injecting a custom or mock provider for testing purposes.

    Args:
        provider: The SandboxProvider instance to use.
    """
    global _default_sandbox_provider
    _default_sandbox_provider = provider
