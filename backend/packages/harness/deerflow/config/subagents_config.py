"""子智能体配置文件模块。

功能概述：
  定义 config.yaml 中子智能体配置的结构，提供配置加载和查询功能。

配置结构：
  subagents:
    timeout_seconds: 900        # 全局默认超时（15分钟）
    max_turns: null              # 全局最大轮次覆盖（null 表示使用内置默认）
    agents:
      general-purpose:           # 按子智能体名称的配置覆盖
        timeout_seconds: 1200    # 覆盖 general-purpose 的超时
        max_turns: 150           # 覆盖 general-purpose 的最大轮次
      bash:
        timeout_seconds: 600    # 覆盖 bash 的超时

设计考虑：
  - 支持全局默认值和按子智能体的覆盖
  - 使用 Pydantic 进行验证
  - 懒加载避免循环依赖
"""

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SubagentOverrideConfig(BaseModel):
    """单个子智能体的配置覆盖。

    用于覆盖内置默认配置的值。
    所有字段都是可选的，不提供则使用默认值或全局配置。

    属性说明：
      timeout_seconds: 超时时间（秒），覆盖此子智能体的默认超时
      max_turns: 最大轮次，覆盖此子智能体的默认最大轮次

    验证：
      - ge=1 确保值 >= 1
    """

    timeout_seconds: int | None = Field(
        default=None,
        ge=1,  # 必须 >= 1
        description="Timeout in seconds for this subagent (None = use global default)",
    )
    max_turns: int | None = Field(
        default=None,
        ge=1,  # 必须 >= 1
        description="Maximum turns for this subagent (None = use global or builtin default)",
    )


class SubagentsAppConfig(BaseModel):
    """子智能体系统配置。

    包含全局默认值和按子智能体的覆盖配置。

    属性说明：
      timeout_seconds: 所有子智能体的默认超时时间（秒）
      max_turns: 所有子智能体的默认最大轮次（可选，全局覆盖）
      agents: 按子智能体名称的配置覆盖字典

    设计说明：
      - timeout_seconds 有默认值 900（15分钟）
      - max_turns 默认 None，表示使用各子智能体的内置默认
      - agents 是可选的字典，允许按名称覆盖特定子智能体的配置
    """

    timeout_seconds: int = Field(
        default=900,  # 默认 15 分钟
        ge=1,         # 必须 >= 1
        description="Default timeout in seconds for all subagents (default: 900 = 15 minutes)",
    )
    max_turns: int | None = Field(
        default=None,  # 默认 None，表示使用内置默认
        ge=1,          # 必须 >= 1
        description="Optional default max-turn override for all subagents (None = keep builtin defaults)",
    )
    # agents: 按子智能体名称的配置覆盖
    # key: 子智能体名称（如 "general-purpose", "bash"）
    # value: SubagentOverrideConfig 实例
    agents: dict[str, SubagentOverrideConfig] = Field(
        default_factory=dict,
        description="Per-agent configuration overrides keyed by agent name",
    )

    def get_timeout_for(self, agent_name: str) -> int:
        """获取指定子智能体的有效超时时间。

        优先级：
          1. agents[agent_name].timeout_seconds（如果提供）
          2. 全局 timeout_seconds

        参数说明：
          agent_name: 子智能体名称

        返回值：
          超时时间（秒）
        """
        # 先检查是否有针对此子智能体的覆盖
        override = self.agents.get(agent_name)
        if override is not None and override.timeout_seconds is not None:
            return override.timeout_seconds
        # 否则使用全局默认
        return self.timeout_seconds

    def get_max_turns_for(self, agent_name: str, builtin_default: int) -> int:
        """获取指定子智能体的有效最大轮次。

        优先级：
          1. agents[agent_name].max_turns（如果提供）
          2. 全局 max_turns（如果提供）
          3. 内置默认 builtin_default

        参数说明：
          agent_name: 子智能体名称
          builtin_default: 该子智能体的内置默认最大轮次

        返回值：
          最大轮次数
        """
        # 先检查是否有针对此子智能体的覆盖
        override = self.agents.get(agent_name)
        if override is not None and override.max_turns is not None:
            return override.max_turns
        # 然后检查全局覆盖
        if self.max_turns is not None:
            return self.max_turns
        # 最后使用内置默认
        return builtin_default


# 模块级配置实例（单例模式）
_subagents_config: SubagentsAppConfig = SubagentsAppConfig()


def get_subagents_app_config() -> SubagentsAppConfig:
    """获取当前子智能体配置。

    返回值：
      SubagentsAppConfig 单例实例
    """
    return _subagents_config


def load_subagents_config_from_dict(config_dict: dict) -> None:
    """从字典加载子智能体配置。

    用于从 config.yaml 读取子智能体配置段。

    参数说明：
      config_dict: 包含子智能体配置的字典

    副作用：
      更新全局 _subagents_config 单例

    日志：
      记录加载的配置和所有覆盖
    """
    global _subagents_config
    _subagents_config = SubagentsAppConfig(**config_dict)

    # 构建覆盖摘要用于日志
    overrides_summary = {}
    for name, override in _subagents_config.agents.items():
        parts = []
        if override.timeout_seconds is not None:
            parts.append(f"timeout={override.timeout_seconds}s")
        if override.max_turns is not None:
            parts.append(f"max_turns={override.max_turns}")
        if parts:
            overrides_summary[name] = ", ".join(parts)

    # 记录配置加载情况
    if overrides_summary:
        logger.info(
            "Subagents config loaded: default timeout=%ss, default max_turns=%s, per-agent overrides=%s",
            _subagents_config.timeout_seconds,
            _subagents_config.max_turns,
            overrides_summary,
        )
    else:
        logger.info(
            "Subagents config loaded: default timeout=%ss, default max_turns=%s, no per-agent overrides",
            _subagents_config.timeout_seconds,
            _subagents_config.max_turns,
        )