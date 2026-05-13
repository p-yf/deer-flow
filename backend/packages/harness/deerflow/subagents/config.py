"""子智能体配置数据类。

功能概述：
  定义子智能体的配置结构，包括名称、描述、系统提示词、工具限制等。

设计考虑：
  - 使用 @dataclass 简化数据类定义
  - 提供默认值允许部分参数省略
  - 工具限制通过 allowlist 和 denylist 实现灵活控制
"""

from dataclasses import dataclass, field


@dataclass
class SubagentConfig:
    """子智能体配置数据类。

    用于存储子智能体的所有配置信息，创建子智能体时传递。

    属性说明：
      name: 唯一标识符，用于注册表查找
      description: 描述何时应该委托给此子智能体（用于文档和 prompt）
      system_prompt: 系统提示词，指导子智能体的行为
      tools: 允许的工具名称列表，None 表示继承所有工具
      disallowed_tools: 禁用的工具名称列表，默认包含 ["task"] 防止嵌套
      model: 使用的模型，"inherit" 表示使用父级模型
      max_turns: 最大执行轮次，防止无限循环
      timeout_seconds: 最大执行时间（秒），超时后强制终止

    工具过滤规则：
      1. 如果 tools 不为 None，只允许列表中的工具
      2. 如果 disallowed_tools 不为 None，排除列表中的工具
      3. 通常设置 tools=None（继承全部），disallowed_tools=["task"]（防止嵌套）

    常用组合：
      - general-purpose: tools=None, disallowed_tools=["task", "ask_clarification", "present_files"]
      - bash: tools=["bash", "ls", "read_file", "write_file", "str_replace"], disallowed_tools=["task", ...]
    """

    name: str  # 子智能体唯一标识符

    description: str  # 描述何时应该使用此子智能体

    system_prompt: str  # 系统提示词，指导子智能体行为

    # tools: 允许的工具列表
    # None 表示继承父级所有工具（除了 disallowed_tools 中列出的）
    # 非 None 表示只允许列表中的工具
    tools: list[str] | None = None

    # disallowed_tools: 禁用的工具列表
    # 默认值是 ["task"]，防止子智能体调用 task 工具导致无限递归
    # 还可以添加其他工具如 "ask_clarification", "present_files"
    disallowed_tools: list[str] | None = field(default_factory=lambda: ["task"])

    # model: 使用的模型名称
    # "inherit" 表示使用父级智能体的模型
    # 具体模型名称如 "gpt-4", "claude-3" 表示使用指定模型
    model: str = "inherit"

    # max_turns: 最大执行轮次
    # 超过此轮次后强制终止，防止无限循环
    # general-purpose 默认 100（复杂任务需要更多轮次）
    # bash 默认 60（命令执行通常较快）
    max_turns: int = 50

    # timeout_seconds: 最大执行时间（秒）
    # 超过此时间后强制终止（通过 ThreadPoolExecutor 的 timeout 实现）
    # 默认 900 秒 = 15 分钟
    timeout_seconds: int = 900