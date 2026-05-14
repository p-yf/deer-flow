"""AIO 沙箱客户端模块 - 连接运行中的 Docker 容器沙箱。

本模块实现 AioSandbox 类，通过 HTTP API 与运行中的 AIO 沙箱容器通信。
AIO 沙箱容器镜像预先配置了执行环境，客户端通过 agent_sandbox 库调用容器 API。

核心概念：
  - agent_sandbox：与容器通信的客户端库（来自 agent-infra/sandbox）
  - 线程锁：execute_command 使用锁防止并发请求破坏容器的单会话状态
  - 错误恢复：检测 ErrorObservation 错误并重试新会话

容器通信：
  - HTTP API：容器的 /v1/sandbox/* 端点
  - Shell 执行：/shell/exec_command
  - 文件操作：/file/read_file, /file/write_file, /file/find_files, /file/search_in_file
  - 错误处理：捕获异常并返回格式化错误消息
"""

# 导入 base64 模块，用于二进制文件传输
# update_file 使用 base64 编码传输二进制内容
import base64

# 导入 logging 模块，用于记录日志
# 记录错误、警告等信息
import logging

# 导入 shlex 模块，用于 shell 命令转义
# list_dir 中使用 shlex.quote() 安全引用路径
import shlex

# 导入 threading 模块，提供线程编程原语
# 使用 Lock 序列化并发命令执行
import threading

# 导入 uuid 模块，生成唯一标识符
# ErrorObservation 错误恢复时生成新的会话 ID
import uuid

# 从 agent_sandbox 库导入 Sandbox 客户端类
# agent_sandbox 是与 AIO 沙箱容器通信的客户端库
# 这个库封装了 HTTP API 调用
from agent_sandbox import Sandbox as AioSandboxClient

# 从同一包的 sandbox 模块导入 Sandbox 抽象基类
# AioSandbox 继承自 Sandbox，必须实现所有抽象方法
from deerflow.sandbox.sandbox import Sandbox

# 从 search 模块导入搜索相关函数和数据类
# GrepMatch：grep 结果数据结构
# path_matches：检查路径是否匹配 glob 模式
# should_ignore_path：检查路径是否应被忽略
# truncate_line：截断过长的行
from deerflow.sandbox.search import GrepMatch, path_matches, should_ignore_path, truncate_line

# 获取本模块的 logger 实例
logger = logging.getLogger(__name__)

# ErrorObservation 错误特征字符串
# 当容器输出包含此字符串时，说明会话损坏，需要重试
_ERROR_OBSERVATION_SIGNATURE = "'ErrorObservation' object has no attribute 'exit_code'"


# AioSandbox 类：AIO 沙箱实现
#
# 作用说明：
#   通过 HTTP API 连接运行中的 AIO 沙箱容器。
#   容器由 AioSandboxProvider 管理，此客户端类负责与容器通信。
#
# 继承关系：
#   继承自 Sandbox 抽象基类，实现所有抽象方法
#
# 关键特性：
#   - 线程锁：序列化 shell 命令，防止并发请求破坏容器的单会话状态
#   - 错误恢复：检测 ErrorObservation 错误并用新会话重试
#   - HTTP API：通过 agent_sandbox 库调用容器 API
#
# 已知问题：
#   #1433：容器的单持久会话在并发请求时会损坏
#   使用锁来防止这个问题，但仍保留了检测和恢复机制
class AioSandbox(Sandbox):
    """Sandbox implementation using the agent-infra/sandbox Docker container.

    This sandbox connects to a running AIO sandbox container via HTTP API.
    A threading lock serializes shell commands to prevent concurrent requests
    from corrupting the container's single persistent session (see #1433).
    """

    # __init__：构造函数
    #
    # 参数：
    #   id: str，沙箱唯一标识符
    #   base_url: str，沙箱 API 的 URL（如 http://localhost:8080）
    #   home_dir: str | None，容器内的主目录（可选，自动获取）
    def __init__(self, id: str, base_url: str, home_dir: str | None = None):
        """Initialize the AIO sandbox.

        Args:
            id: Unique identifier for this sandbox instance.
            base_url: URL of the sandbox API (e.g., http://localhost:8080).
            home_dir: Home directory inside the sandbox. If None, will be fetched from the sandbox.
        """
        # 调用父类构造函数，保存沙箱 ID
        super().__init__(id)

        # 保存沙箱 API 的基础 URL
        self._base_url = base_url

        # 创建 agent_sandbox 客户端
        # timeout=600 秒（10 分钟），命令执行可能有较长超时
        self._client = AioSandboxClient(base_url=base_url, timeout=600)

        # 保存主目录（可选）
        self._home_dir = home_dir

        # 创建线程锁，序列化并发命令执行
        # 防止多个线程同时执行命令导致会话损坏
        self._lock = threading.Lock()

    # base_url 属性：获取沙箱 API 的基础 URL
    @property
    def base_url(self) -> str:
        return self._base_url

    # home_dir 属性：获取容器内的主目录
    #
    # 实现逻辑：
    #   如果 _home_dir 未设置，通过 API 获取容器上下文
    #   然后从上下文中提取 home_dir
    @property
    def home_dir(self) -> str:
        """Get the home directory inside the sandbox."""
        if self._home_dir is None:
            # 调用 API 获取容器上下文
            context = self._client.sandbox.get_context()
            # 从上下文中提取 home_dir 并缓存
            self._home_dir = context.home_dir
        return self._home_dir

    # execute_command：执行 shell 命令
    #
    # 参数：
    #   command: str，要执行的命令
    #
    # 返回值：
    #   str，命令的输出
    #
    # 实现细节：
    #   - 使用锁序列化并发请求
    #   - 检测 ErrorObservation 错误并重试
    #   - 错误时返回格式化错误消息
    def execute_command(self, command: str) -> str:
        """Execute a shell command in the sandbox.

        Uses a lock to serialize concurrent requests. The AIO sandbox
        container maintains a single persistent shell session that
        corrupts when hit with concurrent exec_command calls (returns
        ``ErrorObservation`` instead of real output). If corruption is
        detected despite the lock (e.g. multiple processes sharing a
        sandbox), the command is retried on a fresh session.

        Args:
            command: The command to execute.

        Returns:
            The output of the command.
        """
        # 获取锁
        with self._lock:
            try:
                # 执行命令
                result = self._client.shell.exec_command(command=command)

                # 提取输出
                output = result.data.output if result.data else ""

                # 检测 ErrorObservation 错误
                if output and _ERROR_OBSERVATION_SIGNATURE in output:
                    logger.warning(
                        "ErrorObservation detected in sandbox output, retrying with a fresh session"
                    )
                    # 生成新的会话 ID
                    fresh_id = str(uuid.uuid4())
                    # 用新会话重试命令
                    result = self._client.shell.exec_command(command=command, id=fresh_id)
                    output = result.data.output if result.data else ""

                # 如果没有输出，返回 "(no output)"
                return output if output else "(no output)"
            except Exception as e:
                # 记录错误并返回错误消息
                logger.error(f"Failed to execute command in sandbox: {e}")
                return f"Error: {e}"

    # read_file：读取文件内容
    #
    # 参数：
    #   path: str，文件的绝对路径
    #
    # 返回值：
    #   str，文件内容
    def read_file(self, path: str) -> str:
        """Read the content of a file in the sandbox.

        Args:
            path: The absolute path of the file to read.

        Returns:
            The content of the file.
        """
        try:
            # 调用 API 读取文件
            result = self._client.file.read_file(file=path)
            return result.data.content if result.data else ""
        except Exception as e:
            logger.error(f"Failed to read file in sandbox: {e}")
            return f"Error: {e}"

    # list_dir：列出目录内容
    #
    # 参数：
    #   path: str，目录的绝对路径
    #   max_depth: int，最大遍历深度（默认 2）
    #
    # 返回值：
    #   list[str]，目录内容的路径列表
    #
    # 实现细节：
    #   使用 find 命令列出文件和目录
    #   通过锁保护，因为使用了 exec_command
    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        """List the contents of a directory in the sandbox.

        Args:
            path: The absolute path of the directory to list.
            max_depth: The maximum depth to traverse. Default is 2.

        Returns:
            The contents of the directory.
        """
        # 使用锁保护，因为内部调用了 exec_command
        with self._lock:
            try:
                # 使用 find 命令列出目录内容
                # -maxdepth 限制深度
                # -type f 只文件，-type d 只目录
                # 2>/dev/null 忽略错误输出
                # head -500 限制结果数量
                result = self._client.shell.exec_command(
                    command=f"find {shlex.quote(path)} -maxdepth {max_depth} -type f -o -type d 2>/dev/null | head -500"
                )
                output = result.data.output if result.data else ""

                # 解析输出为行列表
                if output:
                    # 分割、去除空白、过滤空行
                    return [line.strip() for line in output.strip().split("\n") if line.strip()]
                return []
            except Exception as e:
                logger.error(f"Failed to list directory in sandbox: {e}")
                return []

    # write_file：写入文件内容
    #
    # 参数：
    #   path: str，文件的绝对路径
    #   content: str，要写入的文本内容
    #   append: bool，是否追加模式（默认 False = 覆盖）
    #
    # 实现细节：
    #   追加模式：先读取现有内容，拼接后再写入
    def write_file(self, path: str, content: str, append: bool = False) -> None:
        """Write content to a file in the sandbox.

        Args:
            path: The absolute path of the file to write to.
            content: The text content to write to the file.
            append: Whether to append the content to the file.
        """
        with self._lock:
            try:
                # 如果是追加模式，先读取现有内容
                if append:
                    existing = self.read_file(path)
                    # 如果读取成功（非错误），拼接内容
                    if not existing.startswith("Error:"):
                        content = existing + content

                # 调用 API 写入文件
                self._client.file.write_file(file=path, content=content)
            except Exception as e:
                logger.error(f"Failed to write file in sandbox: {e}")
                raise

    # glob：查找匹配 glob 模式的文件
    #
    # 参数：
    #   path: str，搜索根目录
    #   pattern: str，glob 模式（如 **/*.py）
    #   include_dirs: bool，是否包含目录（默认 False）
    #   max_results: int，最大返回结果数（默认 200）
    #
    # 返回值：
    #   tuple[list[str], bool]：(匹配路径列表, 是否截断)
    def glob(
        self,
        path: str,
        pattern: str,
        *,
        include_dirs: bool = False,
        max_results: int = 200,
    ) -> tuple[list[str], bool]:
        # 如果不包含目录，使用 find_files API
        if not include_dirs:
            # 调用 API 查找匹配的文件
            result = self._client.file.find_files(path=path, glob=pattern)
            files = result.data.files if result.data and result.data.files else []

            # 过滤掉需要忽略的路径
            filtered = [file_path for file_path in files if not should_ignore_path(file_path)]

            # 检查是否截断
            truncated = len(filtered) > max_results
            return filtered[:max_results], truncated

        # 如果包含目录，列出所有文件再过滤
        result = self._client.file.list_path(path=path, recursive=True, show_hidden=False)
        entries = result.data.files if result.data and result.data.files else []

        matches: list[str] = []

        # 计算路径前缀用于过滤
        root_path = path.rstrip("/") or "/"
        root_prefix = root_path if root_path == "/" else f"{root_path}/"

        # 遍历条目，查找匹配
        for entry in entries:
            # 跳过根目录本身和非子路径
            if entry.path != root_path and not entry.path.startswith(root_prefix):
                continue

            # 跳过需要忽略的路径
            if should_ignore_path(entry.path):
                continue

            # 计算相对路径
            rel_path = entry.path[len(root_path) :].lstrip("/")

            # 检查是否匹配模式
            if path_matches(pattern, rel_path):
                matches.append(entry.path)

                # 检查是否达到最大结果数
                if len(matches) >= max_results:
                    return matches, True

        return matches, False

    # grep：在文件中搜索匹配模式
    #
    # 参数：
    #   path: str，搜索根目录
    #   pattern: str，搜索模式（正则表达式或字面字符串）
    #   glob: str | None，可选的 glob 过滤器
    #   literal: bool，是否作为字面字符串搜索
    #   case_sensitive: bool，是否区分大小写
    #   max_results: int，最大返回结果数（默认 100）
    #
    # 返回值：
    #   tuple[list[GrepMatch], bool]：(匹配结果列表, 是否截断)
    def grep(
        self,
        path: str,
        pattern: str,
        *,
        glob: str | None = None,
        literal: bool = False,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> tuple[list[GrepMatch], bool]:
        import re as _re

        # 处理模式
        if literal:
            # 字面字符串：转义所有正则特殊字符
            regex_source = _re.escape(pattern)
        else:
            regex_source = pattern

        # 本地验证正则表达式
        # 这样无效的正则会在本地抛出 re.error
        # 而不是远程 API 的通用错误
        _re.compile(regex_source, 0 if case_sensitive else _re.IGNORECASE)

        # 构建正则表达式
        # 如果不区分大小写，添加 (?i) 前缀
        regex = regex_source if case_sensitive else f"(?i){regex_source}"

        # 获取候选文件列表
        if glob is not None:
            # 有 glob 过滤器，使用 find_files
            find_result = self._client.file.find_files(path=path, glob=glob)
            candidate_paths = find_result.data.files if find_result.data and find_result.data.files else []
        else:
            # 无过滤器，列出所有文件
            list_result = self._client.file.list_path(path=path, recursive=True, show_hidden=False)
            entries = list_result.data.files if list_result.data and list_result.data.files else []
            # 只保留文件，跳过目录
            candidate_paths = [entry.path for entry in entries if not entry.is_directory]

        matches: list[GrepMatch] = []
        truncated = False

        # 遍历候选文件
        for file_path in candidate_paths:
            # 跳过忽略的路径
            if should_ignore_path(file_path):
                continue

            # 在文件中搜索
            search_result = self._client.file.search_in_file(file=file_path, regex=regex)
            data = search_result.data
            if data is None:
                continue

            # 提取行号和匹配行
            line_numbers = data.line_numbers or []
            matched_lines = data.matches or []

            # 构建 GrepMatch 结果
            for line_number, line in zip(line_numbers, matched_lines):
                matches.append(
                    GrepMatch(
                        path=file_path,
                        # 确保行号是整数
                        line_number=line_number if isinstance(line_number, int) else 0,
                        line=truncate_line(line),
                    )
                )

                # 检查是否达到最大结果数
                if len(matches) >= max_results:
                    truncated = True
                    return matches, truncated

        return matches, truncated

    # update_file：更新二进制文件
    #
    # 参数：
    #   path: str，文件的绝对路径
    #   content: bytes，二进制内容
    #
    # 实现细节：
    #   使用 base64 编码传输二进制内容
    #   agent_sandbox 的 write_file 支持 encoding="base64"
    def update_file(self, path: str, content: bytes) -> None:
        """Update a file with binary content in the sandbox.

        Args:
            path: The absolute path of the file to update.
            content: The binary content to write to the file.
        """
        with self._lock:
            try:
                # base64 编码二进制内容
                base64_content = base64.b64encode(content).decode("utf-8")

                # 写入文件，指定 base64 编码
                self._client.file.write_file(file=path, content=base64_content, encoding="base64")
            except Exception as e:
                logger.error(f"Failed to update file in sandbox: {e}")
                raise