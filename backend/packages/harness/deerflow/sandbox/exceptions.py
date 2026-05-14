"""沙箱相关异常类 - 定义沙箱操作的错误类型。

本模块定义了沙箱操作中可能遇到的各类异常，支持结构化错误信息。
每个异常包含 message（错误消息）和 details（详细属性），方便调试和错误处理。

异常层次结构：
  SandboxError（基类）
    ├── SandboxNotFoundError     # 沙箱未找到
    ├── SandboxRuntimeError      # 运行时错误
    ├── SandboxCommandError       # 命令执行错误
    ├── SandboxFileError          # 文件操作错误
    │     ├── SandboxPermissionError   # 权限错误
    │     └── SandboxFileNotFoundError # 文件未找到
"""


# SandboxError 类：沙箱错误基类
#
# 作用说明：
#   所有沙箱相关异常的基类，支持结构化错误信息。
#
# 属性：
#   - message: str，错误消息
#   - details: dict，额外详细信息（可选）
#
# 格式：
#   - 无 details 时：只显示 message
#   - 有 details 时：message (key=value, ...)
class SandboxError(Exception):
    """Base exception for all sandbox-related errors."""

    # __init__：构造函数
    #
    # 参数：
    #   message: str，错误消息
    #   details: dict | None，额外详细信息
    def __init__(self, message: str, details: dict | None = None):
        # 调用父类（Exception）的构造函数
        # 将 message 传递给父类，父类存储在 args 属性中
        super().__init__(message)

        # 保存错误消息到实例属性
        # self.message 用于程序化访问错误消息
        self.message = message

        # 保存详细信息，默认为空字典
        # 使用 or 运算符，如果 details 为 None 或 falsy，则使用空字典
        self.details = details or {}

    # __str__：字符串表示
    #
    # 返回值：
    #   格式化后的错误字符串
    #   - 无 details 时：只显示 message
    #   - 有 details 时：message (key=value, key=value, ...)
    def __str__(self) -> str:
        # 如果有详细信息，格式化为 "message (key=value, ...)"
        if self.details:
            # 将 details 字典格式化为 "key=value" 字符串
            # 遍历字典 items，用 f-string 格式化每个键值对
            # 然后用 ", " 连接所有格式化后的字符串
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        # 没有 details 时，直接返回消息
        return self.message


# SandboxNotFoundError：沙箱未找到异常
#
# 作用说明：
#   当沙箱不存在或无法找到时抛出。
#
# 属性：
#   - sandbox_id: str | None，尝试查找的沙箱 ID
class SandboxNotFoundError(SandboxError):
    """Raised when a sandbox cannot be found or is not available."""

    # __init__：构造函数
    #
    # 参数：
    #   message: str，错误消息（默认为 "Sandbox not found"）
    #   sandbox_id: str | None，尝试查找的沙箱 ID
    def __init__(self, message: str = "Sandbox not found", sandbox_id: str | None = None):
        # 如果提供了 sandbox_id，构建 details 字典
        # 否则 details 为 None
        details = {"sandbox_id": sandbox_id} if sandbox_id else None

        # 调用父类构造函数
        # 传递 message 和 details
        super().__init__(message, details)

        # 保存 sandbox_id 属性
        # 这样可以方便地通过异常对象访问沙箱 ID
        self.sandbox_id = sandbox_id


# SandboxRuntimeError：沙箱运行时错误
#
# 作用说明：
#   当沙箱运行时不可用或配置错误时抛出。
#   例如：Docker 未运行、配置无效、环境问题等。
class SandboxRuntimeError(SandboxError):
    """Raised when sandbox runtime is not available or misconfigured."""

    # 使用 pass 表示没有额外的构造函数逻辑
    # 直接使用父类的构造函数，使用默认错误消息
    pass


# SandboxCommandError：沙箱命令执行错误
#
# 作用说明：
#   当沙箱中的命令执行失败时抛出。
#
# 属性：
#   - command: str | None，执行失败的命令
#   - exit_code: int | None，命令的退出码
class SandboxCommandError(SandboxError):
    """Raised when a command execution fails in the sandbox."""

    # __init__：构造函数
    #
    # 参数：
    #   message: str，错误消息
    #   command: str | None，执行失败的命令（只显示前 100 字符）
    #   exit_code: int | None，命令的退出码
    def __init__(self, message: str, command: str | None = None, exit_code: int | None = None):
        # 初始化空字典用于存储详细信息
        details = {}

        # 如果提供了命令
        if command:
            # 如果命令超过 100 字符，截断并添加 "..."
            # 这样可以避免错误消息过长
            details["command"] = command[:100] + "..." if len(command) > 100 else command

        # 如果提供了退出码
        if exit_code is not None:
            # 将退出码添加到详细信息中
            details["exit_code"] = exit_code

        # 调用父类构造函数
        # 传递消息和可能为空的 details 字典
        super().__init__(message, details)

        # 保存额外属性到实例
        # self.command 用于程序化访问失败的命令
        self.command = command

        # self.exit_code 用于程序化访问退出码
        self.exit_code = exit_code


# SandboxFileError：沙箱文件操作错误
#
# 作用说明：
#   当沙箱中的文件操作失败时抛出。
#
# 属性：
#   - path: str | None，操作失败的文件路径
#   - operation: str | None，操作类型（如 "read", "write"）
class SandboxFileError(SandboxError):
    """Raised when a file operation fails in the sandbox."""

    # __init__：构造函数
    #
    # 参数：
    #   message: str，错误消息
    #   path: str | None，操作失败的文件路径
    #   operation: str | None，操作类型
    def __init__(self, message: str, path: str | None = None, operation: str | None = None):
        # 初始化空字典用于存储详细信息
        details = {}

        # 如果提供了路径
        if path:
            # 添加 path 到详细信息
            details["path"] = path

        # 如果提供了操作类型
        if operation:
            # 添加 operation 到详细信息
            details["operation"] = operation

        # 调用父类构造函数
        super().__init__(message, details)

        # 保存额外属性到实例
        # self.path 用于程序化访问失败的文件路径
        self.path = path

        # self.operation 用于程序化访问操作类型
        self.operation = operation


# SandboxPermissionError：沙箱权限错误
#
# 作用说明：
#   当文件操作权限不足时抛出（SandboxFileError 的子类）。
#   例如：尝试写入只读路径、无法读取文件等。
class SandboxPermissionError(SandboxFileError):
    """Raised when a permission error occurs during file operations."""

    # 使用 pass 表示没有额外的构造函数逻辑
    # 直接继承 SandboxFileError 的构造函数
    pass


# SandboxFileNotFoundError：沙箱文件未找到错误
#
# 作用说明：
#   当文件或目录不存在时抛出（SandboxFileError 的子类）。
class SandboxFileNotFoundError(SandboxFileError):
    """Raised when a file or directory is not found."""

    # 使用 pass 表示没有额外的构造函数逻辑
    # 直接继承 SandboxFileError 的构造函数
    pass