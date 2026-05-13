"""子智能体注册表模块。

功能概述：
  管理所有可用的子智能体配置，提供查询、列表和可用性检查功能。

核心功能：
  1. get_subagent_config(name)：获取指定名称的子智能体配置（包括 config.yaml 覆盖）
  2. list_subagents()：列出所有可用的子智能体配置
  3. get_subagent_names()：获取所有子智能体名称
  4. get_available_subagent_names()：获取在当前沙箱配置下可用的子智能体

设计考虑：
  - 配置分离：内置配置（BUILTIN_SUBAGENTS）与运行时覆盖（config.yaml）分离
  - 懒导入：避免循环依赖
  - 沙箱感知：某些子智能体（如 bash）可能在特定沙箱配置下不可用
"""

import logging
from dataclasses import replace

from deerflow.sandbox.security import is_host_bash_allowed
from deerflow.subagents.builtins import BUILTIN_SUBAGENTS
from deerflow.subagents.config import SubagentConfig

logger = logging.getLogger(__name__)


def get_subagent_config(name: str) -> SubagentConfig | None:
    """获取指定名称的子智能体配置（已应用 config.yaml 覆盖）。

    参数说明：
      name: str，子智能体名称

    返回值：
      SubagentConfig 如果找到（已应用 config.yaml 覆盖），否则 None

    覆盖逻辑：
      1. 首先从 BUILTIN_SUBAGENTS 获取内置配置
      2. 然后从 config.yaml 读取覆盖配置
      3. 应用超时和最大轮次覆盖
      4. 使用 dataclasses.replace 创建新实例（不可变）

    懒导入说明：
      使用懒导入避免循环依赖，因为 config.yaml 加载可能导入子模块
    """
    # 先从内置字典获取配置
    config = BUILTIN_SUBAGENTS.get(name)
    if config is None:
        return None

    # 应用 config.yaml 中的覆盖配置（懒导入避免循环依赖）
    from deerflow.config.subagents_config import get_subagents_app_config

    # 获取应用配置
    app_config = get_subagents_app_config()

    # 获取有效超时（可能已被 config.yaml 覆盖）
    effective_timeout = app_config.get_timeout_for(name)

    # 获取有效最大轮次（可能已被 config.yaml 覆盖）
    effective_max_turns = app_config.get_max_turns_for(name, config.max_turns)

    # 构建覆盖字典
    overrides = {}

    # 如果超时被覆盖，记录日志并应用
    if effective_timeout != config.timeout_seconds:
        logger.debug(
            "Subagent '%s': timeout overridden by config.yaml (%ss -> %ss)",
            name,
            config.timeout_seconds,
            effective_timeout,
        )
        overrides["timeout_seconds"] = effective_timeout

    # 如果最大轮次被覆盖，记录日志并应用
    if effective_max_turns != config.max_turns:
        logger.debug(
            "Subagent '%s': max_turns overridden by config.yaml (%s -> %s)",
            name,
            config.max_turns,
            effective_max_turns,
        )
        overrides["max_turns"] = effective_max_turns

    # 如果有覆盖，使用 replace 创建新实例（保持不可变性）
    if overrides:
        config = replace(config, **overrides)

    return config


def list_subagents() -> list[SubagentConfig]:
    """列出所有可用的子智能体配置（已应用 config.yaml 覆盖）。

    返回值：
      所有 SubagentConfig 实例的列表

    用途：
      - 获取完整配置列表用于管理界面
      - 每个配置都已应用 config.yaml 的覆盖
    """
    return [get_subagent_config(name) for name in BUILTIN_SUBAGENTS]


def get_subagent_names() -> list[str]:
    """获取所有子智能体名称。

    返回值：
      所有子智能体名称的列表（不考虑可用性）

    注意：
      此方法返回所有注册的子智能体，不考虑当前沙箱配置
      使用 get_available_subagent_names() 获取实际可用的子智能体
    """
    return list(BUILTIN_SUBAGENTS.keys())


def get_available_subagent_names() -> list[str]:
    """获取在当前沙箱配置下可用的子智能体名称。

    返回值：
      在当前沙箱配置下可用的子智能体名称列表

    设计说明：
      某些子智能体（如 bash）可能在特定沙箱配置下不可用
      例如：如果主机 bash 不被允许，bash 子智能体将被过滤掉

    实现逻辑：
      1. 获取所有子智能体名称
      2. 尝试检查主机 bash 是否允许
      3. 如果不允许，从列表中移除 "bash" 子智能体
      4. 如果检查失败，返回所有子智能体（保守策略）

    用途：
      - 在 task_tool 文档中显示可用子智能体
      - 在 prompt.py 中构建子智能体使用指南
    """
    names = list(BUILTIN_SUBAGENTS.keys())
    try:
        # 检查主机 bash 是否允许
        host_bash_allowed = is_host_bash_allowed()
    except Exception:
        # 无法确定时，返回所有子智能体（保守策略）
        logger.debug("Could not determine host bash availability; exposing all built-in subagents")
        return names

    # 如果主机 bash 不允许，过滤掉 bash 子智能体
    if not host_bash_allowed:
        names = [name for name in names if name != "bash"]
    return names