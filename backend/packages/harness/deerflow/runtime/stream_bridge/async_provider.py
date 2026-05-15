"""Async stream bridge factory.

作用说明：
  提供异步上下文管理器工厂函数 make_stream_bridge
  根据配置返回对应类型的 StreamBridge 实现

设计对齐：
  - 与 deerflow.agents.checkpointer.async_provider.make_checkpointer 对齐
  - 两者都是异步上下文管理器，用于管理有生命周期资源

使用方式（FastAPI lifespan）：
  from deerflow.runtime.stream_bridge import make_stream_bridge

  async with make_stream_bridge() as bridge:
      app.state.stream_bridge = bridge

支持类型：
  - memory：内存实现（默认），MemoryStreamBridge
  - redis：Redis 实现（计划中，Phase 2）
"""

# 类型注解前瞻引用，兼容 Python 3.9+
from __future__ import annotations

# 导入 contextlib，提供异步上下文管理器装饰器
# @contextlib.asynccontextmanager 将生成器转换为异步上下文管理器
# 使得 make_stream_bridge 可以作为 async with 的入口点
import contextlib

# 导入 logging，记录初始化信息
import logging

# 导入 AsyncIterator 类型，用于注解返回类型
from collections.abc import AsyncIterator

# 从配置模块导入流桥配置获取函数
# get_stream_bridge_config() 读取 config.yaml 中的 stream_bridge 配置
from deerflow.config.stream_bridge_config import get_stream_bridge_config

# 从当前包导入 StreamBridge 抽象基类
# 工厂函数返回的具体类型都继承自此类
from .base import StreamBridge

# 获取本模块的 logger 实例
logger = logging.getLogger(__name__)


# make_stream_bridge：异步上下文管理器工厂函数
#
# 参数：
#   config: StreamBridgeConfig | None，配置对象（默认从 config.yaml 读取）
#
# 返回值：
#   AsyncIterator[StreamBridge]，异步迭代器，yield StreamBridge 实例
#
# 实现逻辑：
#   1. 如果 config 为 None，从全局配置读取
#   2. 如果 type 是 "memory" 或无配置，返回 MemoryStreamBridge
#   3. 如果 type 是 "redis"，抛出 NotImplementedError（Phase 2）
#   4. 其他 type 抛出 ValueError
#
# 生命周期：
#   - 进入时：创建并初始化 StreamBridge 实例
#   - 退出时：自动调用 bridge.close() 清理资源
#
# 使用示例：
#   async with make_stream_bridge() as bridge:
#       app.state.stream_bridge = bridge
#       # 使用 bridge 进行发布和订阅
@contextlib.asynccontextmanager
async def make_stream_bridge(config=None) -> AsyncIterator[StreamBridge]:
    """Async context manager that yields a :class:`StreamBridge`.

    Falls back to :class:`MemoryStreamBridge` when no configuration is
    provided and nothing is set globally.
    """
    # 如果 config 为 None，从全局配置获取
    # get_stream_bridge_config() 读取 config.yaml
    if config is None:
        config = get_stream_bridge_config()

    # 判断配置类型
    if config is None or config.type == "memory":
        # 延迟导入 MemoryStreamBridge（避免循环导入）
        # 在这里才导入，因为 base.py 已经导入过了
        from deerflow.runtime.stream_bridge.memory import MemoryStreamBridge

        # 获取队列最大容量（无配置时默认 256）
        # 每个 run_id 的事件缓冲最多存储此数量的事件
        maxsize = config.queue_maxsize if config is not None else 256

        # 创建内存流桥接器实例
        bridge = MemoryStreamBridge(queue_maxsize=maxsize)

        # 记录初始化日志
        logger.info("Stream bridge initialised: memory (queue_maxsize=%d)", maxsize)

        try:
            # yield 实例给调用者
            # 调用者可以使用 bridge 进行发布和订阅
            yield bridge
        finally:
            # 退出时自动关闭桥接器，清理资源
            # 即使发生异常也会执行
            await bridge.close()
        return

    # Redis 类型（计划中，Phase 2）
    # 尚未实现，抛出异常
    if config.type == "redis":
        raise NotImplementedError("Redis stream bridge planned for Phase 2")

    # 未知类型，抛出 ValueError
    raise ValueError(f"Unknown stream bridge type: {config.type!r}")