"""沙箱安全辅助函数 - 提供沙箱能力 gating 和安全检查。

本模块提供沙箱安全相关的辅助函数，用于：
1. 判断是否使用本地沙箱提供者
2. 判断是否允许在宿主机上执行 bash 命令
3. 生成安全相关的错误消息

安全考虑：
  - LocalSandboxProvider 提供的不是真正的沙箱隔离
  - 宿主机 bash 执行默认禁用，除非明确配置 sandbox.allow_host_bash: true
  - 子代理的 bash 执行也受同样限制
"""

# 导入应用配置获取函数
# 用于读取 config.yaml 中的沙箱配置
from deerflow.config import get_app_config


# _LOCAL_SANDBOX_PROVIDER_MARKERS：本地沙箱提供者类路径标记
#
# 这些字符串用于识别 LocalSandboxProvider 的各种可能配置格式：
#   - "deerflow.sandbox.local:LocalSandboxProvider"（新格式，简洁）
#   - "deerflow.sandbox.local.local_sandbox_provider:LocalSandboxProvider"（完整模块路径）
#
# 使用元组存储这些标记，支持快速的成员检查（in 操作）
_LOCAL_SANDBOX_PROVIDER_MARKERS = (
    "deerflow.sandbox.local:LocalSandboxProvider",
    "deerflow.sandbox.local.local_sandbox_provider:LocalSandboxProvider",
)


# LOCAL_HOST_BASH_DISABLED_MESSAGE：宿主机 bash 执行被禁用的错误消息
#
# 当用户尝试在 LocalSandboxProvider 下执行 bash 命令时会显示此消息。
# 提示用户切换到 AioSandboxProvider 或在配置中启用 allow_host_bash。
LOCAL_HOST_BASH_DISABLED_MESSAGE = (
    "Host bash execution is disabled for LocalSandboxProvider because it is not a secure "
    "sandbox boundary. Switch to AioSandboxProvider for isolated bash access, or set "
    "sandbox.allow_host_bash: true only in a fully trusted local environment."
)


# LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE：子代理 bash 执行被禁用的错误消息
#
# 与 LOCAL_HOST_BASH_DISABLED_MESSAGE 类似，但专门针对子代理场景。
# 子代理使用独立的 bash 执行器，也需要同样的安全检查。
LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE = (
    "Bash subagent is disabled for LocalSandboxProvider because host bash execution is not "
    "a secure sandbox boundary. Switch to AioSandboxProvider for isolated bash access, or "
    "set sandbox.allow_host_bash: true only in a fully trusted local environment."
)


# uses_local_sandbox_provider：判断是否使用本地沙箱提供者
#
# 参数：
#   config：应用配置（可选，默认为 None 表示使用全局配置）
#
# 返回值：
#   bool：如果使用 LocalSandboxProvider 则返回 True
#
# 实现逻辑：
#   检查 config.sandbox.use 是否匹配本地提供者的类路径标记
def uses_local_sandbox_provider(config=None) -> bool:
    """Return True when the active sandbox provider is the host-local provider."""
    # 如果没有提供配置，使用全局配置
    # get_app_config() 从 config.yaml 读取配置
    if config is None:
        config = get_app_config()

    # 获取沙箱配置
    # getattr 带默认值，防止属性不存在时抛出异常
    sandbox_cfg = getattr(config, "sandbox", None)

    # 获取提供者类路径
    # config.sandbox.use 应该是类似 "deerflow.sandbox.local:LocalSandboxProvider" 的字符串
    sandbox_use = getattr(sandbox_cfg, "use", "")

    # 直接匹配标记列表
    # 检查是否与已知的所有本地提供者路径匹配
    if sandbox_use in _LOCAL_SANDBOX_PROVIDER_MARKERS:
        return True

    # 备用检查：路径以 ":LocalSandboxProvider" 结尾且包含 "deerflow.sandbox.local"
    # 这种模糊匹配可以处理一些变体路径
    return sandbox_use.endswith(":LocalSandboxProvider") and "deerflow.sandbox.local" in sandbox_use


# is_host_bash_allowed：判断是否允许宿主机 bash 执行
#
# 参数：
#   config：应用配置（可选，默认为 None 表示使用全局配置）
#
# 返回值：
#   bool：是否允许在宿主机上执行 bash 命令
#
# 安全逻辑：
#   - 如果是 AioSandboxProvider（容器隔离），始终允许
#   - 如果是 LocalSandboxProvider（本地沙箱）：
#     * 默认不允许（返回 False）
#     * 除非明确配置 sandbox.allow_host_bash: true
def is_host_bash_allowed(config=None) -> bool:
    """Return whether host bash execution is explicitly allowed."""
    # 如果没有提供配置，使用全局配置
    if config is None:
        config = get_app_config()

    # 获取沙箱配置
    sandbox_cfg = getattr(config, "sandbox", None)

    # 如果没有沙箱配置，默认允许（安全策略：信任配置）
    # 这种情况不太可能发生，因为沙箱配置是必需的
    if sandbox_cfg is None:
        return True

    # 如果不是本地沙箱，提供者本身已经提供了隔离，始终允许
    # AioSandboxProvider 使用 Docker 容器隔离，bash 执行在容器内是安全的
    if not uses_local_sandbox_provider(config):
        return True

    # 本地沙箱：检查 allow_host_bash 配置
    # 默认为 False（不安全），必须明确启用
    return bool(getattr(sandbox_cfg, "allow_host_bash", False))