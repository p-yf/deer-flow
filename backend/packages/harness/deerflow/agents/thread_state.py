from typing import Annotated, NotRequired, TypedDict

from langchain.agents import AgentState


# SandboxState：沙箱状态数据类型
#
# 作用说明：
#   定义 agent 状态中沙箱相关的数据结构。
#   用于在中间件之间传递沙箱标识信息。
#
# 字段说明：
#   - sandbox_id：沙箱的唯一标识符，None 表示尚未获取沙箱
#
# 调用位置：
#   SandboxMiddleware 在 before_agent 中设置此状态
#   SandboxMiddleware 在 after_agent 中读取此状态释放沙箱
#   来源文件：deerflow/sandbox/middleware.py
class SandboxState(TypedDict):
    sandbox_id: NotRequired[str | None]


# ThreadDataState：线程数据路径状态数据类型
#
# 作用说明：
#   定义 agent 状态中线程数据路径相关的数据结构。
#   用于在中间件之间传递线程目录路径信息，实现文件隔离。
#
# 字段说明：
#   - workspace_path：工作区目录路径，对应 /mnt/user-data/workspace/
#   - uploads_path：上传文件目录路径，对应 /mnt/user-data/uploads/
#   - outputs_path：输出文件目录路径，对应 /mnt/user-data/outputs/
#
# 调用位置：
#   ThreadDataMiddleware 在 before_agent 中创建并设置此状态
#   上传/下载工具使用这些路径访问线程隔离的目录
#   来源文件：deerflow/agents/middlewares/thread_data_middleware.py
class ThreadDataState(TypedDict):
    workspace_path: NotRequired[str | None]
    uploads_path: NotRequired[str | None]
    outputs_path: NotRequired[str | None]


class ViewedImageData(TypedDict):
    base64: str
    mime_type: str


# merge_artifacts：制品列表归约函数
#
# 作用说明：
#   LangGraph 状态归约函数，用于合并多个制品列表。
#   在 agent 状态更新时自动被调用，合并来自不同来源的制品列表。
#
# 参数：
#   - existing：当前已存在的制品列表（可为 None）
#   - new：新添加的制品列表（可为 None）
#
# 返回值：
#   合并去重后的制品列表，保持原始顺序
#
# 设计考虑：
#   - 使用 dict.fromkeys() 实现去重同时保持顺序（Python 3.7+ 保证 dict 顺序）
#   - 避免使用 set 保持顺序的传统方式
def merge_artifacts(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Reducer for artifacts list - merges and deduplicates artifacts."""
    # 如果已存在列表为空，返回新列表（或空列表）
    if existing is None:
        return new or []
    # 如果新列表为空，返回已存在列表
    if new is None:
        return existing
    # 使用 dict.fromkeys 合并并去重，保持顺序
    return list(dict.fromkeys(existing + new))


# merge_viewed_images：已查看图片字典归约函数
#
# 作用说明：
#   LangGraph 状态归约函数，用于合并图片数据字典。
#   在 agent 状态更新时自动被调用，合并来自不同来源的图片数据。
#
# 参数：
#   - existing：当前已存在的图片字典（可为 None）
#   - new：新添加的图片字典（可为 None）
#
# 返回值：
#   合并后的图片字典
#
# 特殊处理：
#   - 如果 new 为空字典 {}，则清空所有已查看的图片
#   - 这允许中间件在处理完成后清除图片状态以释放内存
def merge_viewed_images(existing: dict[str, ViewedImageData] | None, new: dict[str, ViewedImageData] | None) -> dict[str, ViewedImageData]:
    """Reducer for viewed_images dict - merges image dictionaries.

    Special case: If new is an empty dict {}, it clears the existing images.
    This allows middlewares to clear the viewed_images state after processing.
    """
    # 如果已存在字典为空，返回新字典（或空字典）
    if existing is None:
        return new or {}
    # 如果新字典为空，返回已存在字典
    if new is None:
        return existing
    # 特殊处理：空字典意味着清空所有已查看的图片
    if len(new) == 0:
        return {}
    # 合并字典，新值覆盖已存在相同键的值
    return {**existing, **new}


# ThreadState：主代理线程状态类
#
# 作用说明：
#   扩展 LangChain AgentState，定义主代理的完整状态结构。
#   包含所有中间件和工具需要访问的状态字段。
#
# 字段说明：
#   - sandbox：沙箱状态，包含 sandbox_id
#   - thread_data：线程数据路径状态，包含 workspace_path、uploads_path、outputs_path
#   - title：自动生成的线程标题
#   - artifacts：制品列表（文件路径），使用 merge_artifacts 归约函数自动去重合并
#   - todos：任务列表（用于 plan 模式）
#   - uploaded_files：已上传文件列表
#   - viewed_images：已查看图片字典（image_path -> {base64, mime_type}），使用 merge_viewed_images 归约函数
#
# 调用位置：
#   create_agent() 使用此状态模式创建 agent
#   所有中间件和工具通过状态传递数据
class ThreadState(AgentState):
    sandbox: NotRequired[SandboxState | None]
    thread_data: NotRequired[ThreadDataState | None]
    title: NotRequired[str | None]
    artifacts: Annotated[list[str], merge_artifacts]
    todos: NotRequired[list | None]
    uploaded_files: NotRequired[list[dict] | None]
    viewed_images: Annotated[dict[str, ViewedImageData], merge_viewed_images]  # image_path -> {base64, mime_type}
