"""沙箱抽象基类 - 定义沙箱环境的通用接口。

本模块定义了 Sandbox 抽象基类，所有具体沙箱实现（如 LocalSandbox）都必须继承此类。
Sandbox 为 agent 提供了隔离的文件系统和命令执行环境。

核心概念：
  - 虚拟路径 vs 物理路径：agent 使用虚拟路径（如 /mnt/user-data/workspace），
    但实际文件系统操作使用物理路径（如 backend/.deer-flow/threads/123/user-data/workspace）
  - 路径映射（PathMapping）：定义虚拟路径到物理路径的转换规则
  - 读写权限：某些路径（如 skills）是只读的，不能写入

重要提示：
  所有路径操作方法（execute_command、read_file、write_file 等）都使用虚拟路径。
  子类（如 LocalSandbox）负责将虚拟路径转换为物理路径后再执行实际文件系统操作。
"""

# 导入抽象基类和抽象方法装饰器，用于定义接口
# ABC 是 Python 内置的抽象基类元类，用于定义抽象基类
# abstractmethod 装饰器标记子类必须实现的方法
from abc import ABC, abstractmethod

# 导入 GrepMatch，用于 grep 方法的返回值类型定义
# GrepMatch 是 search.py 中定义的数据类，包含文件的匹配结果信息
from deerflow.sandbox.search import GrepMatch


# Sandbox 类：沙箱环境的抽象基类
#
# 作用说明：
#   定义沙箱环境的通用接口，所有具体沙箱实现必须实现以下方法：
#   - execute_command：执行 bash 命令
#   - read_file：读取文件内容
#   - write_file：写入文件内容
#   - list_dir：列出目录内容
#   - glob：查找匹配 glob 模式的文件
#   - grep：在文件中搜索匹配模式
#   - update_file：更新二进制文件
#
# 设计考虑：
#   - 这是一个抽象基类，不能直接实例化
#   - 子类需要实现所有抽象方法
#   - 虚拟路径系统允许 agent 在隔离环境中操作文件，同时保持与宿主机的隔离
class Sandbox(ABC):
    """Abstract base class for sandbox environments."""

    # _id：沙箱的唯一标识符
    # 由子类在构造函数中设置
    # 这是一个类属性的类型注解，子类需要设置这个属性
    _id: str

    # __init__：构造函数
    #
    # 参数：
    #   id: str，沙箱的唯一标识符
    def __init__(self, id: str):
        # 保存沙箱 ID 到实例属性
        # 使用 _id 作为存储，下划线前缀表示这是内部属性
        self._id = id

    # id 属性：获取沙箱的唯一标识符
    # @property 装饰器将方法转为属性访问
    # 外部代码可以访问 sandbox.id 而不是 sandbox.id()
    @property
    def id(self) -> str:
        """Return the sandbox ID."""
        # 返回存储的沙箱 ID
        return self._id

    # execute_command：执行 bash 命令
    #
    # 参数：
    #   command: str，要执行的 bash 命令
    #
    # 返回值：
    #   str：命令的标准输出或错误输出
    #
    # 注意：
    #   - 命令使用虚拟路径，子类负责转换为物理路径
    #   - 如果命令执行失败，错误信息也会包含在返回值中
    @abstractmethod
    def execute_command(self, command: str) -> str:
        """Execute bash command in sandbox.

        Args:
            command: The command to execute.

        Returns:
            The standard or error output of the command.
        """
        # pass 表示抽象方法没有实现
        # 子类必须实现此方法，否则不能实例化
        pass

    # read_file：读取文件内容
    #
    # 参数：
    #   path: str，文件的绝对路径（虚拟路径）
    #
    # 返回值：
    #   str：文件内容（UTF-8 编码的文本）
    #
    # 注意：
    #   - 如果文件不存在，应抛出异常
    #   - 路径使用虚拟路径，子类负责转换
    @abstractmethod
    def read_file(self, path: str) -> str:
        """Read the content of a file.

        Args:
            path: The absolute path of the file to read.

        Returns:
            The content of the file.
        """
        pass

    # list_dir：列出目录内容
    #
    # 参数：
    #   path: str，目录的绝对路径（虚拟路径）
    #   max_depth: int，最大遍历深度，默认为 2
    #              1 = 仅直接子项，2 = 子项 + 孙项，以此类推
    #
    # 返回值：
    #   list[str]：目录内容的绝对路径列表
    #
    # 注意：
    #   - 返回的路径以 "/" 结尾表示目录
    #   - 结果按字母顺序排序
    @abstractmethod
    def list_dir(self, path: str, max_depth=2) -> list[str]:
        """List the contents of a directory.

        Args:
            path: The absolute path of the directory to list.
            max_depth: The maximum depth to traverse. Default is 2.

        Returns:
            The contents of the directory.
        """
        pass

    # write_file：写入文件内容
    #
    # 参数：
    #   path: str，文件的绝对路径（虚拟路径）
    #   content: str，要写入的文本内容
    #   append: bool，如果为 True，则追加到文件末尾；否则创建或覆盖文件
    #
    # 注意：
    #   - 如果目录不存在，应自动创建
    #   - 如果路径是只读的，应抛出异常
    @abstractmethod
    def write_file(self, path: str, content: str, append: bool = False) -> None:
        """Write content to a file.

        Args:
            path: The absolute path of the file to write to.
            content: The text content to write to the file.
            append: Whether to append the content to the file. If False, the file will be created or overwritten.
        """
        pass

    # glob：查找匹配 glob 模式的文件
    #
    # 参数：
    #   path: str，搜索的根目录（虚拟路径）
    #   pattern: str，glob 模式，如 **/*.py
    #   include_dirs: bool，是否包含目录，默认为 False
    #   max_results: int，最大返回结果数，默认为 200
    #
    # 返回值：
    #   tuple[list[str], bool]：元组 (匹配的文件路径列表, 是否截断)
    #
    # 注意：
    #   - 结果按路径字母顺序排序
    #   - 如果结果被截断，第二个返回值为 True
    @abstractmethod
    def glob(self, path: str, pattern: str, *, include_dirs: bool = False, max_results: int = 200) -> tuple[list[str], bool]:
        """Find paths that match a glob pattern under a root directory."""
        pass

    # grep：在文件中搜索匹配模式
    #
    # 参数：
    #   path: str，搜索的根目录（虚拟路径）
    #   pattern: str，要搜索的模式（字符串或正则表达式）
    #   glob: str | None，可选的 glob 过滤器，如 **/*.py
    #   literal: bool，是否将模式作为字面字符串而非正则表达式
    #   case_sensitive: bool，是否区分大小写
    #   max_results: int，最大返回结果数，默认为 100
    #
    # 返回值：
    #   tuple[list[GrepMatch], bool]：元组 (匹配结果列表, 是否截断)
    #
    # GrepMatch 结构：
    #   - path: str，文件路径
    #   - line_number: int，行号（1 起始）
    #   - line: str，匹配的文本行（可能被截断）
    @abstractmethod
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
        """Search for matches inside text files under a directory."""
        pass

    # update_file：更新二进制文件
    #
    # 参数：
    #   path: str，文件的绝对路径（虚拟路径）
    #   content: bytes，二进制内容
    #
    # 注意：
    #   - 用于写入二进制文件（如图片）
    #   - 如果路径是只读的，应抛出异常
    @abstractmethod
    def update_file(self, path: str, content: bytes) -> None:
        """Update a file with binary content.

        Args:
            path: The absolute path of the file to update.
            content: The binary content to write to the file.
        """
        pass