"""Stream bridge — decouples agent workers from SSE endpoints.

作用说明：
  StreamBridge 位于后台任务（agent worker，生产者）和 HTTP 端点（SSE，消费者）之间
  解耦了 agent 的流式输出与 HTTP 传输层

核心组件：
  - StreamBridge：抽象基类，定义 publish/subscribe 协议
  - MemoryStreamBridge：内存实现（默认），基于 asyncio.Queue
  - make_stream_bridge：异步上下文管理器工厂

使用场景：
  - Gateway 模式下 agent 嵌入 Gateway 进程内
  - 通过 StreamBridge 实现生产者和消费者的解耦
  - 支持多 run_id 多路复用和断线重连
"""

# 从 async_provider 模块导入流桥接器工厂函数
# make_stream_bridge 是异步上下文管理器，返回 StreamBridge 实例
# 使用方式：async with make_stream_bridge() as bridge: ...
from .async_provider import make_stream_bridge

# 从 base 模块导入抽象基类和核心数据类型
# - StreamBridge：抽象基类
# - StreamEvent：单个事件的数据类（id, event, data）
# - HEARTBEAT_SENTINEL：心跳哨兵事件
# - END_SENTINEL：流结束哨兵事件
from .base import END_SENTINEL, HEARTBEAT_SENTINEL, StreamBridge, StreamEvent

# 从 memory 模块导入内存实现
# MemoryStreamBridge 是 StreamBridge 的默认实现
# 使用内存中的事件列表存储事件，支持多消费者和断线重连
from .memory import MemoryStreamBridge

# 公开导出：所有可在包外访问的名称
# 这定义了 stream_bridge 子包的公共 API
__all__ = [
    # 结束哨兵：流结束时发送给消费者
    "END_SENTINEL",
    # 心跳哨兵：保活信号，指定间隔无事件时发送
    "HEARTBEAT_SENTINEL",
    # 内存流桥接器实现（默认）
    "MemoryStreamBridge",
    # 抽象流桥接器基类
    "StreamBridge",
    # 单个流事件的数据类
    "StreamEvent",
    # 流桥接器工厂函数（异步上下文管理器）
    "make_stream_bridge",
]
