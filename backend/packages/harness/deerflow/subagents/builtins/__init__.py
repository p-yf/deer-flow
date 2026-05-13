"""内置子智能体注册表。

功能概述：
  导出内置子智能体配置，提供统一的访问入口。

内置子智能体：
  - general-purpose: 通用目的子智能体，适用于复杂多步骤任务
  - bash: 命令执行子智能体，适用于 bash 命令执行

使用方式：
  from deerflow.subagents.builtins import BUILTIN_SUBAGENTS, GENERAL_PURPOSE_CONFIG, BASH_AGENT_CONFIG

设计说明：
  - 使用延迟导入避免循环依赖
  - BUILTIN_SUBAGENTS 是主注册表字典
  - GENERAL_PURPOSE_CONFIG 和 BASH_AGENT_CONFIG 单独导出便于访问
"""

from .bash_agent import BASH_AGENT_CONFIG
from .general_purpose import GENERAL_PURPOSE_CONFIG

__all__ = [
    "GENERAL_PURPOSE_CONFIG",
    "BASH_AGENT_CONFIG",
]

# 主注册表：子智能体名称 → 配置对象
# 用于 registry.py 中的查询
BUILTIN_SUBAGENTS = {
    "general-purpose": GENERAL_PURPOSE_CONFIG,
    "bash": BASH_AGENT_CONFIG,
}