"""搜索工具模块 - 提供 glob 文件查找和 grep 内容搜索功能。

本模块实现文件系统搜索功能，用于：
1. glob 模式匹配：find_glob_matches() 在目录树中查找匹配的文件
2. grep 内容搜索：find_grep_matches() 在文件中搜索匹配的行

核心概念：
  - IGNORE_PATTERNS：忽略的目录和文件模式（如 .git, node_modules 等）
  - 路径匹配：path_matches() 处理 glob 模式，包括 ** 递归匹配
  - 二进制文件过滤：is_binary_file() 跳过二进制文件不进行 grep
  - 结果截断：max_results 参数限制返回结果数量

使用场景：
  - LocalSandbox 使用 os.walk 实现本地文件系统搜索
  - AioSandbox 远程调用容器 API 实现搜索（见 aio_sandbox.py）
"""

# 导入 fnmatch 模块，文件名模式匹配
# 支持通配符匹配，如 *.py, node_modules 等
import fnmatch

# 导入 os 模块，提供操作系统功能
# os.walk 用于遍历目录树
import os

# 导入 re 模块，正则表达式
# 用于 grep 的模式匹配
import re

# 从 dataclasses 导入 dataclass 装饰器
# 用于创建 GrepMatch 数据类
from dataclasses import dataclass

# 导入 Path 和 PurePosixPath
# Path 用于文件系统路径操作
# PurePosixPath 用于跨平台处理 POSIX 风格路径（始终使用正斜杠）
from pathlib import Path, PurePosixPath


# IGNORE_PATTERNS：忽略的目录和文件模式列表
#
# 这些模式用于在文件搜索时排除不相关的文件和目录：
#   - 版本控制：.git, .svn, .hg, .bzr
#   - 依赖目录：node_modules, .venv, venv, site-packages
#   - 缓存目录：__pycache__, .pytest_cache, .ruff_cache
#   - 构建输出：dist, build, target, out, .next, .nuxt
#   - IDE 配置：.idea, .vscode, .project, .classpath
#   - 临时文件：*.swp, *.swo, *~, *.tmp, *.bak, *.log
#   - 系统文件：.DS_Store, Thumbs.db, desktop.ini, *.lnk
IGNORE_PATTERNS = [
    ".git",
    ".svn",
    ".hg",
    ".bzr",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    "env",
    ".tox",
    ".nox",
    ".eggs",
    "*.egg-info",
    "site-packages",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".output",
    ".turbo",
    "target",
    "out",
    ".idea",
    ".vscode",
    "*.swp",
    "*.swo",
    "*~",
    ".project",
    ".classpath",
    ".settings",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "*.lnk",
    "*.log",
    "*.tmp",
    "*.temp",
    "*.bak",
    "*.cache",
    ".cache",
    "logs",
    ".coverage",
    "coverage",
    ".nyc_output",
    "htmlcov",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
]

# 默认最大文件大小：1MB
# 超过此大小的文件在 grep 时会被跳过
DEFAULT_MAX_FILE_SIZE_BYTES = 1_000_000

# 默认行截断长度：200 字符
# 超过此长度的匹配行会被截断
DEFAULT_LINE_SUMMARY_LENGTH = 200


# GrepMatch 数据类：grep 搜索结果的数据结构
#
# 属性：
#   - path: str，匹配的文件绝对路径
#   - line_number: int，匹配的行号（1 起始）
#   - line: str，匹配的文本行（可能被截断）
#
# 使用场景：
#   - find_grep_matches() 返回 GrepMatch 列表
#   - 工具函数格式化输出时使用
#
# @dataclass(frozen=True)：
#   frozen=True 使得实例不可变，线程安全
@dataclass(frozen=True)
class GrepMatch:
    """A single grep match result."""

    # 匹配的文件路径
    path: str

    # 行号（1 起始）
    line_number: int

    # 匹配的文本行（可能被截断）
    line: str


# should_ignore_name：检查文件名是否应被忽略
#
# 参数：
#   name: str，要检查的文件/目录名
#
# 返回值：
#   bool：如果文件名匹配 IGNORE_PATTERNS 中的任何模式则返回 True
#
# 实现逻辑：
#   遍历 IGNORE_PATTERNS，使用 fnmatch.fnmatch 进行模式匹配
#   fnmatch 支持通配符：* 匹配任意字符，? 匹配单个字符
#
# 示例：
#   should_ignore_name(".git") -> True
#   should_ignore_name("node_modules") -> True
#   should_ignore_name("test.py") -> False
def should_ignore_name(name: str) -> bool:
    # 遍历所有忽略模式
    for pattern in IGNORE_PATTERNS:
        # 使用 fnmatch 进行模式匹配
        if fnmatch.fnmatch(name, pattern):
            return True
    # 没有匹配，返回 False
    return False


# should_ignore_path：检查路径是否应被忽略
#
# 参数：
#   path: str，要检查的路径（使用正斜杠分隔）
#
# 返回值：
#   bool：如果路径的任何组成部分匹配 IGNORE_PATTERNS 则返回 True
#
# 实现逻辑：
#   将路径按 "/" 分割成各个组成部分
#   检查每个部分是否匹配忽略模式
#
# 示例：
#   should_ignore_path("/home/user/.git/config") -> True（因为包含 .git）
#   should_ignore_path("/node_modules/pkg/index.js") -> True（因为包含 node_modules）
def should_ignore_path(path: str) -> bool:
    # 将反斜杠替换为正斜杠（统一分隔符）
    # 按 "/" 分割路径
    # 过滤掉空字符串
    # 检查每个组成部分是否应被忽略
    return any(should_ignore_name(segment) for segment in path.replace("\\", "/").split("/") if segment)


# path_matches：检查相对路径是否匹配 glob 模式
#
# 参数：
#   pattern: str，glob 模式（如 *.py, **/*.js）
#   rel_path: str，相对于搜索根目录的路径（使用正斜杠）
#
# 返回值：
#   bool：如果路径匹配模式则返回 True
#
# 实现逻辑：
#   使用 PurePosixPath.match() 进行匹配
#   处理 ** 递归模式：允许跨越目录边界
#
# 示例：
#   path_matches("*.py", "test.py") -> True
#   path_matches("**/*.js", "src/app.js") -> True
#   path_matches("**/*.js", "src/nested/deep/app.js") -> True
def path_matches(pattern: str, rel_path: str) -> bool:
    # 将相对路径转为 PurePosixPath（确保使用正斜杠）
    path = PurePosixPath(rel_path)

    # 首先尝试直接匹配
    if path.match(pattern):
        return True

    # 处理 ** 递归模式
    # 如果模式以 **/ 开头，移除 **/ 后再匹配
    # 这样 src/**.js 可以匹配 src/nested/app.js
    if pattern.startswith("**/"):
        return path.match(pattern[3:])

    # 不匹配
    return False


# truncate_line：截断过长的行
#
# 参数：
#   line: str，要截断的行
#   max_chars: int，最大字符数（默认 200）
#
# 返回值：
#   str：如果行长度超过 max_chars，截断并在末尾添加 "..."
#
# 注意：
#   行末的换行符（\n, \r）会被移除后再比较长度
def truncate_line(line: str, max_chars: int = DEFAULT_LINE_SUMMARY_LENGTH) -> str:
    # 移除行末的换行符
    line = line.rstrip("\n\r")

    # 如果行长度在限制内，直接返回
    if len(line) <= max_chars:
        return line

    # 截断并添加 "..." 表示省略
    return line[: max_chars - 3] + "..."


# is_binary_file：检查文件是否为二进制文件
#
# 参数：
#   path: Path，要检查的文件路径
#   sample_size: int，读取的样本大小（默认 8192 字节）
#
# 返回值：
#   bool：如果文件包含空字节则视为二进制文件
#
# 实现逻辑：
#   读取文件的前 sample_size 字节
#   检查是否包含 \0 字符（空字节）
#   如果读取失败（OSError），默认返回 True（安全考虑：跳过无法读取的文件）
#
# 安全考虑：
#   二进制文件不应进行文本 grep，返回 True 跳过
def is_binary_file(path: Path, sample_size: int = 8192) -> bool:
    try:
        # 以二进制模式打开文件
        with path.open("rb") as handle:
            # 读取样本数据，检查是否包含空字节
            return b"\0" in handle.read(sample_size)
    except OSError:
        # 读取失败，默认当作二进制文件处理（安全）
        return True


# find_glob_matches：在目录树中查找匹配 glob 模式的文件
#
# 参数：
#   root: Path，搜索的根目录
#   pattern: str，glob 模式（如 **/*.py）
#   include_dirs: bool，是否包含目录（默认 False）
#   max_results: int，最大返回结果数（默认 200）
#
# 返回值：
#   tuple[list[str], bool]：元组 (匹配的文件路径列表, 是否截断)
#
# 实现逻辑：
#   - 使用 os.walk 遍历目录树（深度优先）
#   - 在遍历时过滤掉匹配的目录（减少遍历）
#   - 按路径字母顺序排序结果
#   - 超过 max_results 时截断并返回 truncated=True
#
# 错误处理：
#   - 根目录不存在：抛出 FileNotFoundError
#   - 根目录不是目录：抛出 NotADirectoryError
def find_glob_matches(
    root: Path,
    pattern: str,
    *,
    include_dirs: bool = False,
    max_results: int = 200,
) -> tuple[list[str], bool]:
    # 存储匹配结果
    matches: list[str] = []
    # 是否截断标志
    truncated = False

    # 解析根目录为绝对路径
    root = root.resolve()

    # 验证根目录存在且是目录
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    # 使用 os.walk 遍历目录树
    # current_root: str，当前遍历的目录
    # dirs: list[str]，当前目录下的子目录列表
    # files: list[str]，当前目录下的文件列表
    for current_root, dirs, files in os.walk(root):
        # 过滤掉需要忽略的目录
        # 直接修改 dirs 列表会影响 os.walk 的后续遍历
        # 这是 os.walk 的正确用法
        dirs[:] = [name for name in dirs if not should_ignore_name(name)]

        # 计算当前目录相对于根目录的路径
        # root 已经解析为绝对路径；os.walk 的 current_root 是基于 root 构建的
        # 所以 relative_to() 不需要额外的 stat()/resolve() 就能工作
        rel_dir = Path(current_root).relative_to(root)

        # 如果包含目录，检查目录是否匹配
        if include_dirs:
            for name in dirs:
                # 构建相对路径
                rel_path = (rel_dir / name).as_posix()
                if path_matches(pattern, rel_path):
                    # 添加完整路径到结果
                    matches.append(str(Path(current_root) / name))
                    # 检查是否达到最大结果数
                    if len(matches) >= max_results:
                        truncated = True
                        return matches, truncated

        # 检查文件是否匹配
        for name in files:
            # 跳过忽略的文件
            if should_ignore_name(name):
                continue

            # 构建相对路径
            rel_path = (rel_dir / name).as_posix()
            if path_matches(pattern, rel_path):
                # 添加完整路径到结果
                matches.append(str(Path(current_root) / name))
                # 检查是否达到最大结果数
                if len(matches) >= max_results:
                    truncated = True
                    return matches, truncated

    # 返回结果和截断标志
    return matches, truncated


# find_grep_matches：在文件中搜索匹配的模式
#
# 参数：
#   root: Path，搜索的根目录
#   pattern: str，要搜索的模式（正则表达式或字面字符串）
#   glob_pattern: str | None，可选的 glob 过滤器（如 **/*.py）
#   literal: bool，是否将模式作为字面字符串而非正则表达式
#   case_sensitive: bool，是否区分大小写
#   max_results: int，最大返回结果数（默认 100）
#   max_file_size: int，跳过大于此大小的文件（默认 1MB）
#   line_summary_length: int，行截断长度（默认 200）
#
# 返回值：
#   tuple[list[GrepMatch], bool]：元组 (匹配结果列表, 是否截断)
#
# 实现逻辑：
#   1. 遍历目录树，使用 glob_pattern 过滤候选文件
#   2. 跳过二进制文件和大文件
#   3. 使用正则表达式搜索匹配的行
#   4. 超过 max_results 时截断返回
#
# 安全考虑：
#   - 超长行（> 10 * line_summary_length）会被跳过，防止 ReDoS 攻击
#   - 二进制文件会被跳过
#   - 文件大小限制防止读取过大的文件
#   - 使用 errors="replace" 处理编码错误
def find_grep_matches(
    root: Path,
    pattern: str,
    *,
    glob_pattern: str | None = None,
    literal: bool = False,
    case_sensitive: bool = False,
    max_results: int = 100,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    line_summary_length: int = DEFAULT_LINE_SUMMARY_LENGTH,
) -> tuple[list[GrepMatch], bool]:
    # 存储匹配结果
    matches: list[GrepMatch] = []
    # 是否截断标志
    truncated = False

    # 解析根目录为绝对路径
    root = root.resolve()

    # 验证根目录存在且是目录
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    # 构建正则表达式
    if literal:
        # 如果是字面字符串搜索，转义所有正则特殊字符
        regex_source = re.escape(pattern)
    else:
        # 否则直接使用提供的模式
        regex_source = pattern

    # 确定正则标志
    flags = 0 if case_sensitive else re.IGNORECASE

    # 编译正则表达式
    regex = re.compile(regex_source, flags)

    # 超长行阈值：超过此长度的行会被跳过，防止 ReDoS
    # 在压缩或无换行的文件中，超长行可能导致正则表达式慢
    _max_line_chars = line_summary_length * 10

    # 遍历目录树
    for current_root, dirs, files in os.walk(root):
        # 过滤需要忽略的目录
        dirs[:] = [name for name in dirs if not should_ignore_name(name)]

        # 计算当前目录相对路径
        rel_dir = Path(current_root).relative_to(root)

        # 遍历文件
        for name in files:
            # 跳过忽略的文件
            if should_ignore_name(name):
                continue

            # 构建候选路径
            candidate_path = Path(current_root) / name
            rel_path = (rel_dir / name).as_posix()

            # 如果有 glob_pattern，先检查是否匹配
            if glob_pattern is not None and not path_matches(glob_pattern, rel_path):
                continue

            try:
                # 跳过符号链接
                if candidate_path.is_symlink():
                    continue

                # 解析为绝对路径
                file_path = candidate_path.resolve()

                # 确保文件在搜索根目录下（防止符号链接逃逸）
                if not file_path.is_relative_to(root):
                    continue

                # 检查文件大小和是否为二进制
                file_stat = file_path.stat()
                if file_stat.st_size > max_file_size or is_binary_file(file_path):
                    continue

                # 打开文件进行搜索
                # errors="replace" 用替换字符替换无法解码的字节
                with file_path.open(encoding="utf-8", errors="replace") as handle:
                    # 逐行读取
                    for line_number, line in enumerate(handle, start=1):
                        # 跳过超长行（防止 ReDoS）
                        if len(line) > _max_line_chars:
                            continue

                        # 检查是否匹配
                        if regex.search(line):
                            matches.append(
                                GrepMatch(
                                    path=str(file_path),  # 使用绝对路径
                                    line_number=line_number,
                                    line=truncate_line(line, line_summary_length),
                                )
                            )
                            # 检查是否达到最大结果数
                            if len(matches) >= max_results:
                                truncated = True
                                return matches, truncated
            except OSError:
                # 忽略无法读取的文件（如权限问题）
                continue

    # 返回结果和截断标志
    return matches, truncated