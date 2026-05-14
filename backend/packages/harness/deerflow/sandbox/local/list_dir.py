"""目录列表模块 - 提供递归目录遍历功能。

本模块实现 list_dir 函数，用于：
1. 遍历目录树到指定深度
2. 排除匹配 IGNORE_PATTERNS 的目录和文件
3. 返回排序后的绝对路径列表（目录以 "/" 结尾）

核心概念：
  - max_depth：最大遍历深度
    * 1 = 仅直接子项
    * 2 = 子项 + 孙项，以此类推
  - IGNORE_PATTERNS：排除的文件/目录模式（来自 search.py）
  - 权限错误处理：遇到 PermissionError 时静默跳过

使用场景：
  - LocalSandbox.list_dir() 使用此函数列出目录内容
  - 返回的路径都是物理路径，需要外部调用者转换
"""

# 导入 Path 对象，提供面向对象的文件系统路径操作
from pathlib import Path

# 从 search 模块导入 should_ignore_name 函数
# 用于判断文件名是否应该被忽略（匹配 IGNORE_PATTERNS）
from deerflow.sandbox.search import should_ignore_name


# list_dir：列出目录内容
#
# 参数：
#   path: str，根目录路径
#   max_depth: int，最大遍历深度（默认 2）
#              1 = 仅直接子项，2 = 子项 + 孙项，以此类推
#
# 返回值：
#   list[str]：目录内容的绝对路径列表
#   - 以 "/" 结尾的表示目录
#   - 结果按字母顺序排序
#   - 排除匹配 IGNORE_PATTERNS 的项
#
# 实现逻辑：
#   1. 验证路径是有效目录
#   2. 使用递归函数 _traverse 遍历
#   3. 每次递归检查深度是否超限
#   4. 跳过匹配的文件/目录
#   5. 对目录进行递归（如果深度允许）
#
# 错误处理：
#   - PermissionError：静默跳过无权访问的目录
#   - 其他错误由调用者处理
def list_dir(path: str, max_depth: int = 2) -> list[str]:
    """
    List files and directories up to max_depth levels deep.

    Args:
        path: The root directory path to list.
        max_depth: Maximum depth to traverse (default: 2).
                   1 = only direct children, 2 = children + grandchildren, etc.

    Returns:
        A list of absolute paths for files and directories,
        excluding items matching IGNORE_PATTERNS.
    """
    # 初始化结果列表，用于存储遍历得到的路径
    result: list[str] = []

    # 将输入路径转为 Path 对象并解析为绝对路径
    # resolve() 会解析符号链接并返回标准化的绝对路径
    root_path = Path(path).resolve()

    # 如果根路径不是有效目录，直接返回空列表
    if not root_path.is_dir():
        return result

    # 定义内部递归遍历函数
    def _traverse(current_path: Path, current_depth: int) -> None:
        """Recursively traverse directories up to max_depth.

        参数：
          current_path: Path，当前遍历的目录路径
          current_depth: int，当前深度（1 表示根目录级别）
        """
        # 如果当前深度超过最大深度，停止递归
        if current_depth > max_depth:
            return

        try:
            # 遍历当前目录的所有直接子项
            for item in current_path.iterdir():
                # 如果应该忽略此文件/目录（匹配 IGNORE_PATTERNS），跳过
                if should_ignore_name(item.name):
                    continue

                # 如果是目录，添加 "/" 后缀；否则无后缀
                post_fix = "/" if item.is_dir() else ""

                # 将解析后的绝对路径添加到结果列表
                # resolve() 获取标准化的绝对路径
                result.append(str(item.resolve()) + post_fix)

                # 如果是目录且未达到最大深度，递归进入
                if item.is_dir() and current_depth < max_depth:
                    _traverse(item, current_depth + 1)

        # 捕获权限错误：静默跳过无权访问的目录
        # 这是为了避免整个遍历因为某个子目录无权限而失败
        except PermissionError:
            pass

    # 从深度 1 开始遍历
    _traverse(root_path, 1)

    # 返回排序后的结果（按字母顺序）
    return sorted(result)