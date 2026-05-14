"""沙箱信息数据类模块 - 沙箱元数据用于跨进程发现和状态持久化。

本模块定义 SandboxInfo 数据类，用于存储沙箱的连接信息和元数据。
支持序列化和反序列化，便于跨进程传递和持久化。

核心概念：
  - 沙箱 ID：确定性 ID，从 thread_id 派生（SHA256 前 8 位）
  - 沙箱 URL：容器的 HTTP API 地址
  - 容器信息：容器名称和 ID（仅本地后端使用）

跨进程发现机制：
  - AioSandboxProvider 使用确定性 sandbox_id
  - 同一 thread_id 在任何进程中生成相同的 sandbox_id
  - 通过容器名（prefix-sandbox_id）发现其他进程的容器
  - 文件锁保证多进程创建同一沙箱时的序列化
"""

# 未来类型注解支持
# 允许在类型注解中使用尚未定义的类名
from __future__ import annotations

# 导入 time 模块，获取时间戳
# created_at 属性使用 time.time() 获取当前时间
import time

# 从 dataclasses 导入 dataclass 和 field
# dataclass 装饰器自动生成 __init__, __repr__, __eq__ 等方法
# field 用于自定义字段行为，如 default_factory
from dataclasses import dataclass, field


# SandboxInfo 数据类：沙箱元数据
#
# 作用说明：
#   存储沙箱的连接信息和元数据，支持跨进程发现。
#   包含沙箱 ID、URL、容器名、容器 ID、创建时间。
#
# 设计考虑：
#   - 不可变设计（dataclass 默认是可变的，但这里不提供 setter）
#   - 支持序列化（to_dict）和反序列化（from_dict）
#   - 跨进程传递：通过 URL 连接容器，通过名称发现容器
#
# 跨进程发现：
#   - sandbox_id 是确定性的（SHA256(thread_id)[:8]）
#   - 容器名是 deterministic：container_prefix-sandbox_id
#   - 任何进程都可以通过容器名发现已存在的容器
@dataclass
class SandboxInfo:
    """Persisted sandbox metadata that enables cross-process discovery.

    This dataclass holds all the information needed to reconnect to an
    existing sandbox from a different process (e.g., gateway vs langgraph,
    multiple workers, or across K8s pods with shared storage).
    """

    # 沙箱 ID：确定性标识符
    # 从 thread_id 派生，同一 thread_id 生成相同 ID
    sandbox_id: str

    # 沙箱 URL：容器的 HTTP API 地址
    # 格式：http://localhost:8080 或 http://k3s:30001
    sandbox_url: str

    # 容器名称（仅本地后端使用）
    # 格式：container_prefix-sandbox_id
    # 用于 docker ps/inspect 等操作
    container_name: str | None = None

    # 容器 ID（仅本地后端使用）
    # Docker 容器分配的唯一 ID
    container_id: str | None = None

    # 创建时间戳：Unix 时间戳（秒）
    # 使用 field(default_factory=time.time) 在创建时自动生成
    # 用于计算容器年龄，判断是否为孤儿容器
    created_at: float = field(default_factory=time.time)

    # to_dict：转换为字典
    #
    # 返回值：
    #   dict：包含所有字段的字典
    #
    # 用途：
    #   - 序列化用于存储或传输
    #   - JSON 序列化时使用
    def to_dict(self) -> dict:
        return {
            "sandbox_id": self.sandbox_id,
            "sandbox_url": self.sandbox_url,
            "container_name": self.container_name,
            "container_id": self.container_id,
            "created_at": self.created_at,
        }

    # from_dict：从字典创建
    #
    # 参数：
    #   data: dict，包含沙箱信息的字典
    #
    # 返回值：
    #   SandboxInfo：新的 SandboxInfo 实例
    #
    # 实现逻辑：
    #   - sandbox_id 和 sandbox_url 是必需字段
    #   - container_name/container_id 是可选字段
    #   - created_at 默认为当前时间
    #   - 向后兼容：sandbox_url 可能是 base_url（旧字段名）
    @classmethod
    def from_dict(cls, data: dict) -> SandboxInfo:
        return cls(
            # 必需字段
            sandbox_id=data["sandbox_id"],
            # sandbox_url 或 base_url（旧字段名，向后兼容）
            sandbox_url=data.get("sandbox_url", data.get("base_url", "")),
            # 可选字段
            container_name=data.get("container_name"),
            container_id=data.get("container_id"),
            # 创建时间，默认为当前时间
            created_at=data.get("created_at", time.time()),
        )