"""本地沙箱实现模块 - 基于宿主文件系统的沙箱实现。

本模块实现 LocalSandbox 类，提供：
1. 文件系统操作的虚拟路径到实际路径的转换
2. 本地 bash 命令执行
3. 文件读写、目录列表、glob 搜索、grep 搜索

核心概念：
  - 虚拟路径 vs 物理路径：
    * 虚拟路径：agent 使用的路径（如 /mnt/user-data/workspace）
    * 物理路径：实际主机文件系统路径（如 backend/.deer-flow/threads/123/user-data/workspace）
  - PathMapping：定义虚拟路径到物理路径的转换规则
  - 读写权限：技能目录等只读路径不能写入

重要提示：
  所有路径操作方法（execute_command、read_file、write_file 等）都使用虚拟路径。
  LocalSandbox 负责将虚拟路径转换为物理路径后再执行实际文件系统操作。

安全考虑：
  - 只读路径检查：_is_read_only_path() 防止写入只读挂载
  - 路径遍历检查：所有路径必须解析后在允许的根目录下
  - 输出路径反向转换：显示给用户时将物理路径转回虚拟路径
"""

# 导入 errno 模块，提供标准 Unix 错误码
# 使用 errno.EROFS (Read-only file system) 标识只读错误
import errno

# 导入 ntpath 模块，Windows 路径处理
# 用于在 Windows 上处理路径分隔符
import ntpath

# 导入 os 模块，提供操作系统功能
import os

# 导入 shutil 模块，提供高级文件操作
# _find_first_available_shell 使用 shutil.which() 查找可执行文件
import shutil

# 导入 subprocess 模块，执行子进程/命令
# execute_command 使用 subprocess.run() 执行 bash 命令
import subprocess

# 从 dataclasses 导入 dataclass 装饰器
# 用于创建 PathMapping 数据类
from dataclasses import dataclass

# 导入 Path 对象，提供面向对象的文件系统路径操作
from pathlib import Path

# 从本地 list_dir 模块导入 list_dir 函数
# LocalSandbox.list_dir() 使用此函数进行目录遍历
from deerflow.sandbox.local.list_dir import list_dir

# 从同一包的 sandbox 模块导入 Sandbox 抽象基类
# LocalSandbox 继承自 Sandbox，必须实现所有抽象方法
from deerflow.sandbox.sandbox import Sandbox

# 从 search 模块导入搜索相关函数和数据类
# GrepMatch：grep 结果的数据结构
# find_glob_matches：glob 模式匹配查找文件
# find_grep_matches：在文件中搜索匹配内容
from deerflow.sandbox.search import GrepMatch, find_glob_matches, find_grep_matches


# PathMapping 数据类：虚拟路径到实际主机路径的映射
#
# 属性：
#   - container_path: str，容器内虚拟路径（如 /mnt/skills）
#   - local_path: str，主机实际路径（如 /path/to/skills）
#   - read_only: bool，是否只读（默认 False）
#
# 使用场景：
#   - 技能目录：/mnt/skills -> /path/to/skills, read_only=True
#   - 自定义挂载：用户配置的主机路径到容器路径的映射
#
# @dataclass(frozen=True) 说明：
#   - frozen=True 使得实例不可变，创建后不能修改属性
#   - 这确保 PathMapping 在多线程环境下是安全的
@dataclass(frozen=True)
class PathMapping:
    """A path mapping from a container path to a local path with optional read-only flag."""

    # 容器内虚拟路径，如 /mnt/skills
    container_path: str

    # 主机实际路径，如 /path/to/skills
    local_path: str

    # 是否只读，默认 False
    read_only: bool = False


# LocalSandbox 类：本地文件系统沙箱实现
#
# 作用说明：
#   提供基于宿主文件系统的沙箱功能，用于开发/测试环境。
#   不是真正的隔离沙箱，无容器隔离。
#
# 继承关系：
#   继承自 Sandbox 抽象基类，实现所有抽象方法
#
# 关键特性：
#   - 路径映射：虚拟路径到物理路径的双向转换
#   - 只读保护：防止写入标记为只读的路径
#   - Shell 检测：自动检测可用的 shell（zsh/bash/sh 或 Windows PowerShell/cmd）
#   - 线程锁：保护命令执行的并发安全
class LocalSandbox(Sandbox):
    # _shell_name：静态方法，从 shell 路径提取可执行文件名
    #
    # 参数：
    #   shell: str，shell 的完整路径或命令
    #
    # 返回值：
    #   str，小写的 shell 可执行文件名
    #
    # 示例：
    #   _shell_name("/bin/zsh") -> "zsh"
    #   _shell_name("C:\\Windows\\System32\\powershell.exe") -> "powershell.exe"
    @staticmethod
    def _shell_name(shell: str) -> str:
        """Return the executable name for a shell path or command."""
        # 将反斜杠替换为正斜杠，统一路径分隔符
        # 然后按 "/" 分割并取最后一部分（文件名）
        # 最后转为小写
        return shell.replace("\\", "/").rsplit("/", 1)[-1].lower()

    # _is_powershell：静态方法，判断 shell 是否为 PowerShell
    @staticmethod
    def _is_powershell(shell: str) -> bool:
        """Return whether the selected shell is a PowerShell executable."""
        # 检查 shell 名是否在 PowerShell 相关名称集合中
        return LocalSandbox._shell_name(shell) in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}

    # _is_cmd_shell：静态方法，判断 shell 是否为 cmd.exe
    @staticmethod
    def _is_cmd_shell(shell: str) -> bool:
        """Return whether the selected shell is cmd.exe."""
        return LocalSandbox._shell_name(shell) in {"cmd", "cmd.exe"}

    # _find_first_available_shell：静态方法，从候选列表中找到第一个可用的 shell
    #
    # 参数：
    #   candidates: tuple[str, ...]，shell 候选路径/命令元组
    #
    # 返回值：
    #   str | None：找到的第一个可用 shell 路径，或 None
    #
    # 实现逻辑：
    #   1. 如果候选是绝对路径，检查文件是否存在且可执行
    #   2. 如果是命令（如 "zsh"），用 shutil.which() 在 PATH 中查找
    @staticmethod
    def _find_first_available_shell(candidates: tuple[str, ...]) -> str | None:
        """Return the first executable shell path or command found from candidates."""
        # 遍历所有候选 shell
        for shell in candidates:
            # 如果是绝对路径（Unix 以 / 开头，Windows 以盘符开头）
            if os.path.isabs(shell):
                # 检查文件是否存在且有执行权限
                if os.path.isfile(shell) and os.access(shell, os.X_OK):
                    return shell
                # 绝对路径但文件不存在或不可执行，跳过
                continue

            # 如果是命令，在 PATH 中查找
            shell_from_path = shutil.which(shell)
            if shell_from_path is not None:
                return shell_from_path

        # 所有候选都不可用，返回 None
        return None

    # __init__：构造函数
    #
    # 参数：
    #   id: str，沙箱标识符
    #   path_mappings: list[PathMapping] | None，路径映射列表
    def __init__(self, id: str, path_mappings: list[PathMapping] | None = None):
        """
        Initialize local sandbox with optional path mappings.

        Args:
            id: Sandbox identifier
            path_mappings: List of path mappings with optional read-only flag.
                          Skills directory is read-only by default.
        """
        # 调用父类构造函数，保存沙箱 ID
        super().__init__(id)

        # 保存路径映射列表，默认为空列表
        # 这些映射用于虚拟路径到物理路径的转换
        self.path_mappings = path_mappings or []

    # _is_read_only_path：检查路径是否在只读挂载下
    #
    # 参数：
    #   resolved_path: str，已解析的物理路径
    #
    # 返回值：
    #   bool：如果路径在只读挂载下则返回 True
    #
    # 实现逻辑：
    #   1. 解析路径为绝对路径
    #   2. 找到最匹配的 PathMapping（最长前缀匹配）
    #   3. 返回该映射的 read_only 标志
    #
    # 嵌套挂载处理：
    #   当有嵌套的挂载时（如 /mnt 和 /mnt/user-data），选择最具体的映射
    def _is_read_only_path(self, resolved_path: str) -> bool:
        """Check if a resolved path is under a read-only mount.

        When multiple mappings match (nested mounts), prefer the most specific
        mapping (i.e. the one whose local_path is the longest prefix of the
        resolved path), similar to how ``_resolve_path`` handles container paths.
        """
        # 将路径解析为绝对路径字符串
        resolved = str(Path(resolved_path).resolve())

        # 初始化最佳映射为空
        best_mapping: PathMapping | None = None

        # 最长前缀长度，初始化为 -1
        best_prefix_len = -1

        # 遍历所有路径映射
        for mapping in self.path_mappings:
            # 解析映射中的本地路径为绝对路径
            local_resolved = str(Path(mapping.local_path).resolve())

            # 检查 resolved 是否与 local_resolved 相同或是其子目录
            # 使用 os.sep 确保目录边界正确
            if resolved == local_resolved or resolved.startswith(local_resolved + os.sep):
                # 计算前缀长度
                prefix_len = len(local_resolved)

                # 选择最长前缀的映射（最具体的匹配）
                if prefix_len > best_prefix_len:
                    best_prefix_len = prefix_len
                    best_mapping = mapping

        # 如果没有匹配的映射，路径可写
        if best_mapping is None:
            return False

        # 返回最佳映射的只读标志
        return best_mapping.read_only

    # _resolve_path：将虚拟路径转换为物理路径
    #
    # 参数：
    #   path: str，容器内虚拟路径
    #
    # 返回值：
    #   str，解析后的本地物理路径
    #
    # 实现逻辑：
    #   1. 按容器路径长度排序（长的优先）进行最具体匹配
    #   2. 如果路径以映射的容器路径开头，替换为本地路径
    #   3. 如果没有匹配，返回原路径
    def _resolve_path(self, path: str) -> str:
        """
        Resolve container path to actual local path using mappings.

        Args:
            path: Path that might be a container path

        Returns:
            Resolved local path
        """
        # 确保 path 是字符串
        path_str = str(path)

        # 按容器路径长度排序（长的优先，这样更具体的匹配优先）
        for mapping in sorted(self.path_mappings, key=lambda m: len(m.container_path), reverse=True):
            container_path = mapping.container_path
            local_path = mapping.local_path

            # 检查路径是否与容器路径匹配（完全匹配或子路径）
            if path_str == container_path or path_str.startswith(container_path + "/"):
                # 计算相对路径
                # 例如：path_str = "/mnt/skills/python", container_path = "/mnt/skills"
                # 则 relative = "python"
                relative = path_str[len(container_path) :].lstrip("/")

                # 拼接本地路径
                # 如果 relative 为空，返回本地路径本身（根目录映射）
                # 否则拼接 local_path + relative
                resolved = str(Path(local_path) / relative) if relative else local_path
                return resolved

        # 没有映射匹配，返回原始路径
        return path_str

    # _reverse_resolve_path：将物理路径转换回虚拟路径
    #
    # 参数：
    #   path: str，本地物理路径
    #
    # 返回值：
    #   str，容器虚拟路径，如果没有映射则返回原路径
    #
    # 实现逻辑：
    #   与 _resolve_path 相反，将本地路径转回容器路径
    def _reverse_resolve_path(self, path: str) -> str:
        """
        Reverse resolve local path back to container path using mappings.

        Args:
            path: Local path that might need to be mapped to container path

        Returns:
            Container path if mapping exists, otherwise original path
        """
        # 统一路径分隔符为正斜杠
        normalized_path = path.replace("\\", "/")

        # 解析为绝对路径
        path_str = str(Path(normalized_path).resolve())

        # 按本地路径长度排序（长的优先，最具体的匹配优先）
        for mapping in sorted(self.path_mappings, key=lambda m: len(m.local_path), reverse=True):
            local_path_resolved = str(Path(mapping.local_path).resolve())

            # 检查路径是否与本地路径匹配
            if path_str == local_path_resolved or path_str.startswith(local_path_resolved + "/"):
                # 计算相对路径
                relative = path_str[len(local_path_resolved) :].lstrip("/")

                # 拼接容器路径
                resolved = f"{mapping.container_path}/{relative}" if relative else mapping.container_path
                return resolved

        # 没有映射匹配，返回原始路径
        return path_str

    # _reverse_resolve_paths_in_output：在输出字符串中将本地路径转回虚拟路径
    #
    # 参数：
    #   output: str，可能包含本地路径的输出字符串
    #
    # 返回值：
    #   str，本地路径已转换为虚拟路径的输出
    #
    # 实现逻辑：
    #   使用正则表达式匹配输出中的本地路径，替换为虚拟路径
    def _reverse_resolve_paths_in_output(self, output: str) -> str:
        """
        Reverse resolve local paths back to container paths in output string.

        Args:
            output: Output string that may contain local paths

        Returns:
            Output with local paths resolved to container paths
        """
        import re

        # 按本地路径长度排序（长的优先，正确的前缀匹配）
        sorted_mappings = sorted(self.path_mappings, key=lambda m: len(m.local_path), reverse=True)

        # 如果没有映射，直接返回原输出
        if not sorted_mappings:
            return output

        # 用于存储替换结果
        result = output

        # 遍历每个映射
        for mapping in sorted_mappings:
            # 转义本地路径用于正则表达式
            escaped_local = re.escape(str(Path(mapping.local_path).resolve()))

            # 构建匹配模式：本地路径 + 可选的路径组件
            # (?:[/\\][^\s\"';&|<>()]*)? 匹配可选的路径部分
            # 不匹配空白字符和特殊 shell 字符
            pattern = re.compile(escaped_local + r"(?:[/\\][^\s\"';&|<>()]*)?)

            # 定义替换函数
            def replace_match(match: re.Match) -> str:
                matched_path = match.group(0)
                return self._reverse_resolve_path(matched_path)

            # 执行替换
            result = pattern.sub(replace_match, result)

        return result

    # _resolve_paths_in_command：在命令字符串中将虚拟路径转换为物理路径
    #
    # 参数：
    #   command: str，可能包含虚拟路径的命令字符串
    #
    # 返回值：
    #   str，虚拟路径已转换为物理路径的命令
    #
    # 实现逻辑：
    #   使用正则表达式匹配命令中的虚拟路径，替换为物理路径
    #   只在合理的路径边界匹配（如 / 后面、空格后面、引号前面等）
    def _resolve_paths_in_command(self, command: str) -> str:
        """
        Resolve container paths to local paths in a command string.

        Args:
            command: Command string that may contain container paths

        Returns:
            Command with container paths resolved to local paths
        """
        import re

        # 按容器路径长度排序（长的优先）
        sorted_mappings = sorted(self.path_mappings, key=lambda m: len(m.container_path), reverse=True)

        # 如果没有映射，直接返回原命令
        if not sorted_mappings:
            return command

        # 构建正则表达式模式
        # 前瞻断言 (?=/|$|[\s\"';&|<>()]) 确保只在路径边界匹配
        # 这防止 /mnt/skills 匹配到 /mnt/skills-extra
        patterns = [
            re.escape(m.container_path) + r"(?=/|$|[\s\"';&|<>()])(?:/[^\s\"';&|<>()]*)?"
            for m in sorted_mappings
        ]
        pattern = re.compile("|".join(f"({p})" for p in patterns))

        # 定义替换函数
        def replace_match(match: re.Match) -> str:
            matched_path = match.group(0)
            return self._resolve_path(matched_path)

        # 执行替换
        return pattern.sub(replace_match, command)

    # _get_shell：静态方法，检测可用的 shell
    #
    # 返回值：
    #   str，可用 shell 的路径
    #
    # 实现逻辑：
    #   1. Unix 系统：尝试 /bin/zsh, /bin/bash, /bin/sh
    #   2. Windows 系统：尝试 PowerShell、pwsh、cmd.exe
    #   3. 都找不到则抛出 RuntimeError
    @staticmethod
    def _get_shell() -> str:
        """Detect available shell executable with fallback."""
        # 首先尝试 Unix shell
        shell = LocalSandbox._find_first_available_shell(("/bin/zsh", "/bin/bash", "/bin/sh", "sh"))
        if shell is not None:
            return shell

        # Windows 系统
        if os.name == "nt":
            # 获取 Windows 系统目录
            system_root = os.environ.get("SystemRoot", r"C:\Windows")

            # 尝试各种 Windows shell
            shell = LocalSandbox._find_first_available_shell(
                (
                    "pwsh",                    # PowerShell Core
                    "pwsh.exe",
                    "powershell",             # Windows PowerShell
                    "powershell.exe",
                    # Windows PowerShell v1.0 经典路径
                    ntpath.join(system_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
                    "cmd.exe",                # 命令提示符
                )
            )
            if shell is not None:
                return shell

            # 所有 Windows shell 都不可用
            raise RuntimeError(
                "No suitable shell executable found. Tried /bin/zsh, /bin/bash, /bin/sh, `sh` on PATH, then PowerShell and cmd.exe fallbacks for Windows."
            )

        # Unix 系统但所有 shell 都不可用
        raise RuntimeError(
            "No suitable shell executable found. Tried /bin/zsh, /bin/bash, /bin/sh, and `sh` on PATH."
        )

    # execute_command：执行 bash 命令
    #
    # 参数：
    #   command: str，要执行的命令（使用虚拟路径）
    #
    # 返回值：
    #   str，命令的标准输出（本地路径已转回虚拟路径）
    def execute_command(self, command: str) -> str:
        # 在执行前将命令中的虚拟路径转换为物理路径
        resolved_command = self._resolve_paths_in_command(command)

        # 检测可用的 shell
        shell = self._get_shell()

        # Windows 系统
        if os.name == "nt":
            # PowerShell
            if self._is_powershell(shell):
                args = [shell, "-NoProfile", "-Command", resolved_command]
            # cmd.exe
            elif self._is_cmd_shell(shell):
                args = [shell, "/c", resolved_command]
            # 其他 shell（如 Git Bash）
            else:
                args = [shell, "-c", resolved_command]

            # 执行命令，捕获输出
            result = subprocess.run(
                args,
                shell=False,           # 不使用 shell 解析参数
                capture_output=True,   # 捕获 stdout 和 stderr
                text=True,             # 返回字符串而非字节
                timeout=600,           # 10 分钟超时
            )
        else:
            # Unix 系统：使用 shell=True 让系统 shell 解析命令
            result = subprocess.run(
                resolved_command,
                executable=shell,      # 指定 shell 执行
                shell=True,             # 让 shell 解析命令字符串
                capture_output=True,
                text=True,
                timeout=600,
            )

        # 获取标准输出
        output = result.stdout

        # 如果有标准错误，附加到输出
        if result.stderr:
            output += f"\nStd Error:\n{result.stderr}" if output else result.stderr

        # 如果退出码非零，附加退出码信息
        if result.returncode != 0:
            output += f"\nExit Code: {result.returncode}"

        # 如果没有输出，返回 "(no output)"
        final_output = output if output else "(no output)"

        # 在返回前将输出中的本地路径转回虚拟路径
        return self._reverse_resolve_paths_in_output(final_output)

    # list_dir：列出目录内容
    #
    # 参数：
    #   path: str，目录的虚拟路径
    #   max_depth: int，最大遍历深度（默认 2）
    #
    # 返回值：
    #   list[str]，目录内容的绝对虚拟路径列表（目录以 "/" 结尾）
    def list_dir(self, path: str, max_depth=2) -> list[str]:
        # 将虚拟路径转换为物理路径
        resolved_path = self._resolve_path(path)

        # 调用 list_dir 函数获取物理路径列表
        entries = list_dir(resolved_path, max_depth)

        # 将物理路径转换回虚拟路径后返回
        return [self._reverse_resolve_paths_in_output(entry) for entry in entries]

    # read_file：读取文件内容
    #
    # 参数：
    #   path: str，文件的虚拟路径
    #
    # 返回值：
    #   str，文件内容（UTF-8 编码）
    def read_file(self, path: str) -> str:
        # 将虚拟路径转换为物理路径
        resolved_path = self._resolve_path(path)
        try:
            # 以 UTF-8 编码打开文件读取内容
            with open(resolved_path, encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            # 重新抛出异常，但使用原始虚拟路径
            # 这样错误消息对用户更清晰，不暴露内部路径
            raise type(e)(e.errno, e.strerror, path) from None

    # write_file：写入文件内容
    #
    # 参数：
    #   path: str，文件的虚拟路径
    #   content: str，要写入的内容
    #   append: bool，是否追加模式（默认 False = 覆盖）
    def write_file(self, path: str, content: str, append: bool = False) -> None:
        # 将虚拟路径转换为物理路径
        resolved_path = self._resolve_path(path)

        # 检查路径是否只读
        if self._is_read_only_path(resolved_path):
            # 抛出只读文件系统错误
            raise OSError(errno.EROFS, "Read-only file system", path)

        try:
            # 获取目录路径
            dir_path = os.path.dirname(resolved_path)

            # 如果目录存在，创建目录树
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            # 确定写入模式：追加或覆盖
            mode = "a" if append else "w"

            # 写入文件
            with open(resolved_path, mode, encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            # 重新抛出异常，使用原始虚拟路径
            raise type(e)(e.errno, e.strerror, path) from None

    # glob：查找匹配 glob 模式的文件
    #
    # 参数：
    #   path: str，搜索根目录的虚拟路径
    #   pattern: str，glob 模式（如 **/*.py）
    #   include_dirs: bool，是否包含目录（默认 False）
    #   max_results: int，最大返回结果数（默认 200）
    #
    # 返回值：
    #   tuple[list[str], bool]：(匹配的文件虚拟路径列表, 是否截断)
    def glob(
        self,
        path: str,
        pattern: str,
        *,
        include_dirs: bool = False,
        max_results: int = 200,
    ) -> tuple[list[str], bool]:
        # 转换根目录虚拟路径为物理路径
        resolved_path = Path(self._resolve_path(path))

        # 调用 search 模块的 find_glob_matches 查找匹配
        matches, truncated = find_glob_matches(
            resolved_path,
            pattern,
            include_dirs=include_dirs,
            max_results=max_results,
        )

        # 将物理路径转换回虚拟路径后返回
        return [self._reverse_resolve_path(match) for match in matches], truncated

    # grep：在文件中搜索匹配模式
    #
    # 参数：
    #   path: str，搜索根目录的虚拟路径
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
        # 转换根目录虚拟路径为物理路径
        resolved_path = Path(self._resolve_path(path))

        # 调用 search 模块的 find_grep_matches 搜索
        matches, truncated = find_grep_matches(
            resolved_path,
            pattern,
            glob_pattern=glob,
            literal=literal,
            case_sensitive=case_sensitive,
            max_results=max_results,
        )

        # 将结果中的物理路径转换回虚拟路径
        return [
            GrepMatch(
                path=self._reverse_resolve_path(match.path),
                line_number=match.line_number,
                line=match.line,
            )
            for match in matches
        ], truncated

    # update_file：更新二进制文件
    #
    # 参数：
    #   path: str，文件的虚拟路径
    #   content: bytes，二进制内容
    def update_file(self, path: str, content: bytes) -> None:
        # 转换虚拟路径为物理路径
        resolved_path = self._resolve_path(path)

        # 检查只读
        if self._is_read_only_path(resolved_path):
            raise OSError(errno.EROFS, "Read-only file system", path)

        try:
            # 获取目录路径
            dir_path = os.path.dirname(resolved_path)

            # 创建目录树
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            # 以二进制模式写入
            with open(resolved_path, "wb") as f:
                f.write(content)
        except OSError as e:
            # 重新抛出异常，使用原始虚拟路径
            raise type(e)(e.errno, e.strerror, path) from None