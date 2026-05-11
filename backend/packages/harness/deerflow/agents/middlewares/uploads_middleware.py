"""UploadsMiddleware - 上传文件信息注入中间件。

功能概述：
  当用户上传文件后，将文件信息注入到 agent 上下文中，让模型知道有哪些文件可以操作。

工作流程：
  1. 在 before_agent 钩子中检查最后一条人类消息
  2. 从消息的 additional_kwargs.files 中提取上传文件元数据
  3. 获取文件的虚拟路径、大小、类型等信息
  4. 将文件列表以 <uploaded_files> 标签的形式添加到用户消息内容前面
  5. 同时更新状态中的 uploaded_files 字段

文件信息来源：
  - additional_kwargs.files：前端通过此字段传递上传文件元数据
  - 消息格式：list[dict{"path": str, "size": int, "mime_type": str}]

处理细节：
  - 如果文件是 PDF/PPT/Excel/Word 等格式，会先转换为 Markdown
  - 转换后的文件内容可以用 <outline> 标签嵌入，提供文档结构预览
  - 每个上传文件映射到虚拟路径：/mnt/user-data/uploads/{filename}

状态更新：
  更新 uploaded_files 字段（list[dict]），包含每个文件的路径和信息。

执行位置：中间件链第二位（紧接 ThreadDataMiddleware 之后）。
"""

# 导入标准库 logging，用于记录日志
import logging
# pathlib.Path 用于处理文件路径
from pathlib import Path
# typing 导入 NotRequired（可选字段）和 override（方法重写标记）
from typing import NotRequired, override

# 从 langchain.agents 导入 AgentState（agent 基础状态类）
from langchain.agents import AgentState
# 从 langchain.agents.middleware 导入 AgentMiddleware（中间件基类）
from langchain.agents.middleware import AgentMiddleware
# 从 langchain_core.messages 导入 HumanMessage（人类消息类型）
# 中间件需要检查最后一条人类消息是否包含上传文件信息
from langchain_core.messages import HumanMessage
# 从 langgraph.runtime 导入 Runtime（LangGraph 运行时上下文）
from langgraph.runtime import Runtime

# 从 deerflow.config.paths 导入 Paths 和 get_paths
# 用于解析上传目录的路径
from deerflow.config.paths import Paths, get_paths
# 从 deerflow.utils.file_conversion 导入 extract_outline 函数
# 这个函数从 Markdown 文件中提取标题大纲（用于文档结构展示）
from deerflow.utils.file_conversion import extract_outline

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)

# 模块级常量：定义预览行数
# 当文档没有标题时，读取转换后的 .md 文件的前几行作为内容预览
_OUTLINE_PREVIEW_LINES = 5


# 模块级函数：为一个上传的文件提取文档大纲和预览
#
# 参数：
#   file_path: Path 对象，指向实际的上传文件（如 PDF、Word 等）
#
# 工作原理：
#   上传的文件会通过转换管道转换成同名的 .md 文件
#   例如 file.pdf 会转换成 file.pdf.md
#   这个函数查找这个 .md 文件并提取标题信息
#
# 返回值：
#   一个元组 (outline, preview)
#   - outline: 标题列表，每个元素是 {title, line} 字典
#              表示标题文字和标题所在的行号
#              如果没有标题或 .md 文件不存在则为空列表
#   - preview: 前几行非空内容
#              当 outline 为空时使用，作为备选内容提示
#              当 outline 非空时为空列表（不需要 preview）
def _extract_outline_for_file(file_path: Path) -> tuple[list[dict], list[str]]:
    # 将文件路径的后缀替换为 .md
    # 例如 /path/to/doc.pdf -> /path/to/doc.pdf.md
    md_path = file_path.with_suffix(".md")

    # 如果 .md 文件不存在，返回空的大纲和预览
    if not md_path.is_file():
        return [], []

    # 调用 extract_outline 函数从 .md 文件中提取标题
    # 这个函数返回标题列表
    outline = extract_outline(md_path)

    # 如果成功提取到标题，直接返回（preview 为空列表）
    if outline:
        logger.debug("Extracted %d outline entries from %s", len(outline), file_path.name)
        return outline, []

    # outline 为空，说明没有标题
    # 此时读取 .md 文件的前几行作为内容预览
    preview: list[str] = []
    try:
        # 以 UTF-8 编码打开 .md 文件
        with md_path.open(encoding="utf-8") as f:
            # 逐行读取
            for line in f:
                stripped = line.strip()
                # 跳过空行
                if stripped:
                    preview.append(stripped)
                # 达到预览行数上限（_OUTLINE_PREVIEW_LINES = 5）停止读取
                if len(preview) >= _OUTLINE_PREVIEW_LINES:
                    break
    except Exception:
        # 文件读取失败，记录日志但继续执行
        logger.debug("Failed to read preview lines from %s", md_path, exc_info=True)

    # 返回空的大纲和预览内容
    return [], preview


# UploadsMiddlewareState 类：定义中间件使用的状态 schema
# 继承自 AgentState
class UploadsMiddlewareState(AgentState):
    # uploaded_files 字段：本次新上传的文件列表
    # 类型是 list[dict] | None，表示可选字段
    # 每个字典包含 filename、size、path、extension 等信息
    uploaded_files: NotRequired[list[dict] | None]


# UploadsMiddleware 类：负责将上传文件信息注入到 agent 上下文
#
# 工作流程：
#   1. 在 before_agent 时检查最后一条人类消息是否包含上传文件信息
#   2. 从消息的 additional_kwargs.files 中读取新上传的文件元数据
#   3. 扫描上传目录获取历史文件列表（排除本次新上传的）
#   4. 为每个文件提取文档大纲（如果有的话）
#   5. 创建一个格式化的 <uploaded_files> 信息块
#   6. 将这个信息块添加到用户消息内容的前面
#   7. 同时在状态中返回 uploaded_files 列表，供后续使用
#
# 注意：
#   - 这个中间件只修改最后一条人类消息
#   - additional_kwargs 被保留，让前端可以从流式消息中读取文件元数据
#   - 文件路径使用虚拟路径 /mnt/user-data/uploads/，不是物理路径
class UploadsMiddleware(AgentMiddleware[UploadsMiddlewareState]):
    # state_schema 类变量，指定该中间件使用的状态类型
    state_schema = UploadsMiddlewareState

    # 构造函数
    #
    # 参数：
    #   base_dir: 线程数据的根目录，默认为 None（使用 Paths 默认解析）
    def __init__(self, base_dir: str | None = None):
        # 调用父类构造函数
        super().__init__()
        # 如果提供了 base_dir，用它创建 Paths 实例；否则使用全局 Paths 实例
        self._paths = Paths(base_dir) if base_dir else get_paths()

    # 内部方法：为单个文件格式化条目信息，追加到 lines 列表中
    #
    # 这个方法生成类似如下的格式化文本：
    #   - report.pdf (245.6 KB)
    #     Path: /mnt/user-data/uploads/report.pdf
    #     Document outline (use `read_file` with line ranges to read sections):
    #       L45: Executive Summary
    #       L78: Financial Analysis
    #       ...
    #
    # 参数：
    #   file: 文件信息字典，包含 filename、size、path、outline 等字段
    #   lines: 用于追加格式化文本的列表（就地修改）
    def _format_file_entry(self, file: dict, lines: list[str]) -> None:
        # 计算文件大小字符串
        # size 是字节数，除以 1024 得到 KB
        size_kb = file["size"] / 1024
        # 如果小于 1024 KB 显示 "X.X KB"，否则显示 "X.X MB"
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"

        # 第一行：文件名称和大小
        lines.append(f"- {file['filename']} ({size_str})")
        # 第二行：虚拟路径（agent 使用的路径格式）
        lines.append(f"  Path: {file['path']}")

        # 获取文档大纲（如果有的话）
        outline = file.get("outline") or []

        if outline:
            # 有大纲的情况：
            # 检查大纲是否被截断（最后一条记录有 truncated=True 表示截断）
            truncated = outline[-1].get("truncated", False)
            # 过滤掉 truncated 的条目，只显示完整的标题
            visible = [e for e in outline if not e.get("truncated")]

            # 添加说明文字和大纲内容
            lines.append("  Document outline (use `read_file` with line ranges to read sections):")
            for entry in visible:
                # 格式：L{行号}: {标题}
                lines.append(f"    L{entry['line']}: {entry['title']}")

            # 如果大纲被截断，添加说明
            if truncated:
                lines.append(f"    ... (showing first {len(visible)} headings; use `read_file` to explore further)")
        else:
            # 没有大纲的情况：
            # 检查是否有内容预览（当没有标题时使用）
            preview = file.get("outline_preview") or []
            if preview:
                # 添加说明和预览内容
                lines.append("  No structural headings detected. Document begins with:")
                for text in preview:
                    # 预览内容用 "> " 前缀
                    lines.append(f"    > {text}")

            # 添加使用 grep 搜索的提示
            lines.append("  Use `grep` to search for keywords (e.g. `grep(pattern='keyword', path='/mnt/user-data/uploads/')`).")

        # 添加空行分隔
        lines.append("")

    # 内部方法：创建格式化的文件列表消息
    #
    # 参数：
    #   new_files: 本次新上传的文件列表
    #   historical_files: 历史文件列表（之前上传的）
    #
    # 返回值：
    #   格式化的字符串，包含在 <uploaded_files> 和 </uploaded_files> 标签之间
    #   内容结构：
    #     - 本次上传的文件列表
    #     - 历史文件列表
    #     - 文件使用提示
    def _create_files_message(self, new_files: list[dict], historical_files: list[dict]) -> str:
        # 初始化 lines 列表，以 <uploaded_files> 标签开始
        lines = ["<uploaded_files>"]

        # 添加"本次上传的文件"部分标题
        lines.append("The following files were uploaded in this message:")
        lines.append("")

        # 遍历本次上传的文件，格式化并添加到 lines
        if new_files:
            for file in new_files:
                self._format_file_entry(file, lines)
        else:
            # 没有新文件的情况
            lines.append("(empty)")
            lines.append("")

        # 添加历史文件部分（如果有的话）
        if historical_files:
            lines.append("The following files were uploaded in previous messages and are still available:")
            lines.append("")
            for file in historical_files:
                self._format_file_entry(file, lines)

        # 添加文件使用提示
        lines.append("To work with these files:")
        lines.append("- Read from the file first — use the outline line numbers and `read_file` to locate relevant sections.")
        lines.append("- Use `grep` to search for keywords when you are not sure which section to look at")
        lines.append("  (e.g. `grep(pattern='revenue', path='/mnt/user-data/uploads/')`).")
        lines.append("- Use `glob` to find files by name pattern")
        lines.append("  (e.g. `glob(pattern='**/*.md', path='/mnt/user-data/uploads/')`).")
        lines.append("- Only fall back to web search if the file content is clearly insufficient to answer the question.")
        lines.append("</uploaded_files>")

        # 将所有行合并成单个字符串，用换行符分隔
        return "\n".join(lines)

    # 内部方法：从消息的 additional_kwargs 中提取文件信息
    #
    # 工作原理：
    #   前端上传文件后，会在消息的 additional_kwargs 中设置 files 字段
    #   这个字段是一个列表，每个元素包含 filename、size、path、status 等
    #
    # 参数：
    #   message: HumanMessage，要检查的人类消息
    #   uploads_dir: 可选的 Path 对象，用于验证文件是否真实存在
    #                如果文件不存在，该条目会被跳过
    #
    # 返回值：
    #   文件信息字典列表，每个字典包含：
    #     - filename: 文件名
    #     - size: 文件大小（字节）
    #     - path: 虚拟路径（/mnt/user-data/uploads/{filename}）
    #     - extension: 文件扩展名
    #   如果没有文件信息或字段为空，返回 None
    def _files_from_kwargs(self, message: HumanMessage, uploads_dir: Path | None = None) -> list[dict] | None:
        # 从消息的 additional_kwargs 中获取 files 字段
        # additional_kwargs 是消息的附加元数据，文件上传时由前端设置
        kwargs_files = (message.additional_kwargs or {}).get("files")

        # 如果不是列表或列表为空，返回 None
        if not isinstance(kwargs_files, list) or not kwargs_files:
            return None

        files = []
        # 遍历所有文件条目
        for f in kwargs_files:
            # 确保是字典类型
            if not isinstance(f, dict):
                continue

            # 获取文件名
            filename = f.get("filename") or ""
            # 跳过无效的文件名（空或包含路径）
            # Path(filename).name == filename 检查是否只是文件名而非路径
            if not filename or Path(filename).name != filename:
                continue

            # 如果提供了 uploads_dir，验证文件是否真实存在
            if uploads_dir is not None and not (uploads_dir / filename).is_file():
                continue

            # 构建文件信息字典
            files.append(
                {
                    "filename": filename,  # 文件名
                    "size": int(f.get("size") or 0),  # 大小（字节），转换为 int
                    # 虚拟路径：agent 使用这个路径访问文件
                    "path": f"/mnt/user-data/uploads/{filename}",
                    "extension": Path(filename).suffix,  # 文件扩展名（如 .pdf、.docx）
                }
            )

        # 如果有文件返回列表，否则返回 None
        return files if files else None

    # before_agent 钩子方法：在 agent 执行前被调用
    #
    # 工作流程：
    #   1. 获取消息列表，检查最后一条是否是 HumanMessage
    #   2. 从消息的 additional_kwargs.files 提取新上传的文件
    #   3. 扫描上传目录获取历史文件（排除新上传的）
    #   4. 为所有文件提取文档大纲
    #   5. 创建文件信息消息并添加到用户消息内容前面
    #   6. 返回状态更新（messages 和 uploaded_files）
    #
    # 参数：
    #   state: 当前 agent 状态（UploadsMiddlewareState）
    #   runtime: LangGraph 运行时上下文（包含 thread_id）
    #
    # 返回值：
    #   状态更新字典，包含：
    #     - messages: 更新后的消息列表（最后一条 HumanMessage 内容被修改）
    #     - uploaded_files: 本次新上传的文件列表
    #   如果没有文件或消息列表为空，返回 None
    @override
    def before_agent(self, state: UploadsMiddlewareState, runtime: Runtime) -> dict | None:
        # 从 state 中获取消息列表
        messages = list(state.get("messages", []))
        # 如果没有消息，直接返回 None
        if not messages:
            return None

        # 获取最后一条消息
        last_message_index = len(messages) - 1
        last_message = messages[last_message_index]

        # 确保最后一条消息是人类消息（只有人类消息可以包含上传文件）
        if not isinstance(last_message, HumanMessage):
            return None

        # 获取 thread_id（用于定位上传目录）
        # 首先尝试从 runtime.context 获取
        thread_id = (runtime.context or {}).get("thread_id")
        if thread_id is None:
            try:
                # 如果 runtime.context 中没有，尝试从 LangGraph 全局配置获取
                from langgraph.config import get_config
                thread_id = get_config().get("configurable", {}).get("thread_id")
            except RuntimeError:
                # get_config() 在非运行上下文中（如单元测试）会抛出异常
                pass

        # 解析上传目录路径（用于文件存在性检查）
        uploads_dir = self._paths.sandbox_uploads_dir(thread_id) if thread_id else None

        # 从消息的 additional_kwargs.files 中提取新上传的文件
        new_files = self._files_from_kwargs(last_message, uploads_dir) or []

        # 收集历史文件（之前上传的，还在上传目录中的）
        new_filenames = {f["filename"] for f in new_files}  # 用于排除本次新上传的文件
        historical_files: list[dict] = []

        if uploads_dir and uploads_dir.exists():
            # 遍历上传目录中的所有文件
            for file_path in sorted(uploads_dir.iterdir()):
                # 只处理文件，且排除本次新上传的文件
                if file_path.is_file() and file_path.name not in new_filenames:
                    stat = file_path.stat()
                    # 提取文档大纲（如果有的话）
                    outline, preview = _extract_outline_for_file(file_path)
                    # 构建文件信息字典
                    historical_files.append(
                        {
                            "filename": file_path.name,
                            "size": stat.st_size,  # 文件大小（字节）
                            "path": f"/mnt/user-data/uploads/{file_path.name}",  # 虚拟路径
                            "extension": file_path.suffix,  # 扩展名
                            "outline": outline,  # 文档大纲
                            "outline_preview": preview,  # 内容预览
                        }
                    )

        # 为新上传的文件也提取大纲
        if uploads_dir:
            for file in new_files:
                phys_path = uploads_dir / file["filename"]
                outline, preview = _extract_outline_for_file(phys_path)
                file["outline"] = outline
                file["outline_preview"] = preview

        # 如果既没有新文件也没有历史文件，直接返回 None（不需要修改消息）
        if not new_files and not historical_files:
            return None

        # 记录日志：有哪些新文件、历史文件
        logger.debug(f"New files: {[f['filename'] for f in new_files]}, historical: {[f['filename'] for f in historical_files]}")

        # 创建格式化的文件信息消息
        files_message = self._create_files_message(new_files, historical_files)

        # 提取原始消息内容
        # 消息内容可能是字符串或列表（列表通常是 {type: "text", text: "..."} 格式）
        original_content = ""
        if isinstance(last_message.content, str):
            # 字符串格式：直接使用
            original_content = last_message.content
        elif isinstance(last_message.content, list):
            # 列表格式：提取所有 text 类型的块并合并
            text_parts = []
            for block in last_message.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            original_content = "\n".join(text_parts)

        # 创建更新后的消息
        # 内容 = 文件信息 + 原内容
        # additional_kwargs 被保留，让前端可以从流式消息中读取文件元数据
        updated_message = HumanMessage(
            content=f"{files_message}\n\n{original_content}",
            id=last_message.id,  # 保持消息 ID 不变
            additional_kwargs=last_message.additional_kwargs,  # 保留文件元数据
        )

        # 将更新后的消息放回消息列表
        messages[last_message_index] = updated_message

        # 返回状态更新
        # - messages: 更新后的消息列表（文件信息已添加到用户消息前）
        # - uploaded_files: 本次新上传的文件列表（用于后续处理）
        return {
            "uploaded_files": new_files,
            "messages": messages,
        }