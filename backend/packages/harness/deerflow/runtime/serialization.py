"""Canonical serialization for LangChain / LangGraph objects.

作用说明：
  提供单一事实来源，将 LangChain 消息对象、Pydantic 模型、
  LangGraph state dicts 转换为普通 JSON 可序列化的 Python 结构

使用方：
  - deerflow.runtime.runs.worker（SSE 发布）
  - app.gateway.routers.threads（REST 响应）

核心函数：
  - serialize_lc_object()：递归序列化 LangChain 对象
  - serialize_channel_values()：序列化 channel values，剥离内部 key
  - serialize_messages_tuple()：序列化 messages 模式元组
  - serialize()：根据模式选择序列化方式
"""

# 类型注解前瞻引用，兼容 Python 3.9+
from __future__ import annotations

# 导入 Any，用于注解接受任意类型
from typing import Any


# serialize_lc_object：递归序列化 LangChain 对象
#
# 参数：
#   obj: Any，要序列化的对象
#     可能是 LangChain 消息、Pydantic 模型、普通 dict/list 等
#
# 返回值：
#   Any，JSON 可序列化的 Python 结构
#
# 实现逻辑：
#   1. None -> None
#   2. 原始类型（str, int, float, bool）-> 直接返回
#   3. dict -> 递归序列化每个 value
#   4. list/tuple -> 递归序列化每个元素
#   5. Pydantic v2 对象 -> 使用 model_dump()
#   6. Pydantic v1 对象 -> 使用 dict()
#   7. 最后 resorts -> str() 或 repr()
#
# 设计说明：
#   这是递归下降的序列化器，处理嵌套的 LangChain 对象
#   例如：{"messages": [HumanMessage(content="hello")]}
#   被转换为 {"messages": [{"type": "human", "content": "hello", "additional_kwargs": {}}]}
def serialize_lc_object(obj: Any) -> Any:
    """Recursively serialize a LangChain object to a JSON-serialisable dict."""
    # None 直接返回
    if obj is None:
        return None

    # 原始类型直接返回（JSON 原生支持）
    # 这些类型可以直接被 json.dumps() 处理
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # dict：递归序列化每个 value
    # 保留键名不变，只序列化值
    if isinstance(obj, dict):
        return {k: serialize_lc_object(v) for k, v in obj.items()}

    # list/tuple：递归序列化每个元素
    # 用于序列化消息列表等
    if isinstance(obj, (list, tuple)):
        return [serialize_lc_object(item) for item in obj]

    # Pydantic v2：优先使用 model_dump()
    # Pydantic v2 模型（如 HumanMessage）有 model_dump() 方法
    # 用于将模型转换为 dict
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass

    # Pydantic v1 / older objects：使用 dict()
    # Pydantic v1 模型使用 dict() 方法
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass

    # 最后 resorts：尝试 str()，失败则 repr()
    # 对于无法识别类型的对象，尝试转换为字符串
    # 这确保即使遇到未知类型也不会抛出序列化异常
    try:
        return str(obj)
    except Exception:
        return repr(obj)


# serialize_channel_values：序列化 channel values，剥离内部 key
#
# 参数：
#   channel_values: dict[str, Any]，LangGraph channel 值字典
#     来自 checkpoint["channel_values"]
#
# 返回值：
#   dict[str, Any]，清理后的 channel 值
#
# 设计说明：
#   移除 __pregel_* 开头的内部 key 和 __interrupt__
#   使输出与 LangGraph Platform API 返回格式一致
#   这些内部 key 是 LangGraph 运行时使用的，不应暴露给前端
#
# 示例：
#   输入：{"messages": [...], "title": "Hello", "__pregel_readonly": True}
#   输出：{"messages": [...], "title": "Hello"}
def serialize_channel_values(channel_values: dict[str, Any]) -> dict[str, Any]:
    """Serialize channel values, stripping internal LangGraph keys.

    Internal keys like ``__pregel_*`` and ``__interrupt__`` are removed
    to match what the LangGraph Platform API returns.
    """
    result: dict[str, Any] = {}

    # 遍历 channel values
    for key, value in channel_values.items():
        # 跳过内部 key（以 __pregel_ 开头或 __interrupt__）
        # 这些是 LangGraph 内部使用的 key
        # __pregel_readonly, __pregel_version 等
        if key.startswith("__pregel_") or key == "__interrupt__":
            continue

        # 序列化其余 key
        # 调用 serialize_lc_object 处理嵌套对象
        result[key] = serialize_lc_object(value)

    return result


# serialize_messages_tuple：序列化 messages 模式元组
#
# 参数：
#   obj: Any，messages 模式的对象
#     messages 模式下，astream 返回的是 (message_chunk, metadata_dict) 元组
#
# 返回值：
#   Any，序列化后的对象
#
# 说明：
#   messages 模式的 item 是 (message_chunk, metadata_dict) 二元组
#   需要分别序列化 chunk，metadata 保持原样（如果已是 dict）
#
# 示例：
#   输入：(AIMessageChunk(content="Hello", id="ai-1"), {"usage": {...}})
#   输出：[{"type": "ai", "content": "Hello", "id": "ai-1"}, {"usage": {...}}]
def serialize_messages_tuple(obj: Any) -> Any:
    """Serialize a messages-mode tuple ``(chunk, metadata_dict)``."""
    # 如果是 (chunk, metadata) 二元组
    if isinstance(obj, tuple) and len(obj) == 2:
        chunk, metadata = obj

        # 返回 [序列化后的chunk, metadata字典或空字典]
        # chunk 被序列化为普通 dict
        # metadata 保持原样（如果已是 dict），否则转为空 dict
        return [serialize_lc_object(chunk), metadata if isinstance(metadata, dict) else {}]

    # 否则直接序列化（降级处理）
    return serialize_lc_object(obj)


# serialize：模式特定的 LangChain 对象序列化
#
# 参数：
#   obj: Any，要序列化的对象
#   mode: str，序列化模式
#     - "messages"：obj 是 (message_chunk, metadata_dict) 元组
#     - "values"：obj 是完整 state dict，剥离 __pregel_* keys
#     - 其他：递归 model_dump()/dict() 回退
#
# 返回值：
#   Any，序列化后的对象
#
# 设计说明：
#   根据 mode 选择不同的序列化策略
#   messages 模式需要保持元组结构
#   values 模式需要剥离内部 key
#
# 调用方：
#   worker.py 中的 run_agent() 调用
#   每个从 agent.astream() 收到的 chunk 都会被序列化
def serialize(obj: Any, *, mode: str = "") -> Any:
    """Serialize LangChain objects with mode-specific handling.

    * ``messages`` — obj is ``(message_chunk, metadata_dict)``
    * ``values`` — obj is the full state dict; ``__pregel_*`` keys stripped
    * everything else — recursive ``model_dump()`` / ``dict()`` fallback
    """
    # messages 模式：序列化消息元组
    # 保持 [chunk, metadata] 元组结构
    if mode == "messages":
        return serialize_messages_tuple(obj)

    # values 模式：序列化 channel values（剥离内部 key）
    # 只有 dict 类型的才剥离 key，其他类型递归序列化
    if mode == "values":
        return serialize_channel_values(obj) if isinstance(obj, dict) else serialize_lc_object(obj)

    # 其他模式：递归序列化
    # 包括 "custom", "updates", "checkpoints" 等模式
    return serialize_lc_object(obj)