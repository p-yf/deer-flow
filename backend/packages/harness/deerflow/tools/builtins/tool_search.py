"""Tool search — deferred tool discovery at runtime.

Contains:
- DeferredToolRegistry: stores deferred tools and handles regex search
- tool_search: the LangChain tool the agent calls to discover deferred tools

The agent sees deferred tool names in <available-deferred-tools> but cannot
call them until it fetches their full schema via the tool_search tool.
Source-agnostic: no mention of MCP or tool origin.
"""

# ============================================================
# 导入标准库
# ============================================================

# contextvars：Python 标准库上下文变量模块
# 用于在异步环境中存储请求级别的数据
# 这样每个并发请求有独立的注册表，互不干扰
import contextvars

# json：标准库 JSON 模块，用于序列化和反序列化 JSON 数据
import json

# logging：标准库日志模块，用于记录运行日志
import logging

# re：标准库正则表达式模块，用于编译和匹配搜索模式
import re

# dataclasses：标准库数据类模块，用于定义 DeferredToolEntry 数据类
from dataclasses import dataclass

# ============================================================
# 导入 LangChain 相关模块
# ============================================================

# langchain.tools.BaseTool：LangChain 工具基类
# 所有工具都继承自此基类
from langchain.tools import BaseTool

# langchain_core.tools.tool：LangChain 装饰器，用于定义工具函数
from langchain_core.tools import tool

# langchain_core.utils.function_calling.convert_to_openai_function：
# LangChain 工具，用于将工具转换为 OpenAI 函数格式
from langchain_core.utils.function_calling import convert_to_openai_function

# ============================================================
# 模块级变量初始化
# ============================================================

# 创建模块级 logger，用于记录运行日志
logger = logging.getLogger(__name__)

# MAX_RESULTS：每次搜索返回的最大工具数量
# 限制返回结果数量，避免过大的上下文开销
MAX_RESULTS = 5


# ============================================================
# 数据结构定义
# ============================================================

# DeferredToolEntry：延迟工具条目数据类
#
# 作用说明：
#   这是一个轻量级的数据结构，用于存储延迟工具的元数据。
#   延迟工具指的是在 <available-deferred-tools> 中出现名称、
#   但还没有获取完整模式（schema）的工具。
#
# 字段说明：
#   - name: str，工具名称
#   - description: str，工具描述
#   - tool: BaseTool，完整的工具对象，在搜索匹配时返回
@dataclass
class DeferredToolEntry:
    """Lightweight metadata for a deferred tool (no full schema in context)."""

    # 工具名称
    name: str
    # 工具描述
    description: str
    # 完整工具对象，只在搜索匹配时返回
    # 这允许我们只存储元数据而不需要在上下文加载完整模式
    tool: BaseTool


# DeferredToolRegistry：延迟工具注册表
#
# 作用说明：
#   管理所有延迟工具的注册、搜索、提升和删除。
#   支持三种查询模式（与 Claude Code 对齐）：
#   1. "select:name1,name2" — 精确名称匹配
#   2. "+keyword rest" — 名称必须包含关键词，rest 用于排序
#   3. "keyword query" — 正则表达式匹配名称 + 描述
#
# 设计考虑：
#   - 使用列表存储条目，保持插入顺序
#   - 支持上下文级别的隔离（通过 ContextVar）
#   - 提供 promote 机制，工具获取完整模式后从注册表移除
class DeferredToolRegistry:
    """Registry of deferred tools, searchable by regex pattern."""

    # 构造函数
    #
    # 初始化一个空的注册表
    def __init__(self):
        # _entries：内部存储的延迟工具条目列表
        self._entries: list[DeferredToolEntry] = []

    # register：注册一个延迟工具
    #
    # 参数：
    #   tool: BaseTool，要注册的完整工具对象
    #
    # 工作流程：
    #   创建一个 DeferredToolEntry，添加到内部列表
    def register(self, tool: BaseTool) -> None:
        # 创建条目并添加到列表
        self._entries.append(
            DeferredToolEntry(
                # 工具名称
                name=tool.name,
                # 工具描述，默认为空字符串
                description=tool.description or "",
                # 完整的工具对象
                tool=tool,
            )
        )

    # promote：将工具从延迟注册表提升为活跃状态
    #
    # 参数：
    #   names: set[str]，要提升的工具名称集合
    #
    # 作用说明：
    #   当 tool_search 返回工具的完整模式后，调用此方法从注册表中移除。
    #   这样 DeferredToolFilterMiddleware 就不会再从 bind_tools 中过滤它们。
    #
    # 工作流程：
    #   1. 如果 names 为空，直接返回
    #   2. 过滤掉所有名称在 names 中的条目
    #   3. 记录日志
    def promote(self, names: set[str]) -> None:
        """Remove tools from the deferred registry so they pass through the filter.

        Called after tool_search returns a tool's schema — the LLM now knows
        the full definition, so the DeferredToolFilterMiddleware should stop
        stripping it from bind_tools on subsequent calls.
        """
        # 如果 names 为空，直接返回
        if not names:
            return

        # 记录提升前的条目数量
        before = len(self._entries)

        # 过滤：只保留名称不在 names 中的条目
        self._entries = [e for e in self._entries if e.name not in names]

        # 计算提升了多少个工具
        promoted = before - len(self._entries)

        # 如果有工具被提升，记录调试日志
        if promoted:
            logger.debug(f"Promoted {promoted} tool(s) from deferred to active: {names}")

    # search：搜索延迟工具
    #
    # 参数：
    #   query: str，查询字符串
    #
    # 返回值：
    #   list[BaseTool]：匹配的工具列表（最多 MAX_RESULTS 个）
    #
    # 支持三种查询模式：
    #   1. "select:name1,name2" — 精确名称匹配
    #   2. "+keyword rest" — 名称必须包含 keyword，rest 用于排序
    #   3. "keyword query" — 正则表达式匹配名称 + 描述
    def search(self, query: str) -> list[BaseTool]:
        """Search deferred tools by regex pattern against name + description.

        Supports three query forms (aligned with Claude Code):
          - "select:name1,name2" — exact name match
          - "+keyword rest" — name must contain keyword, rank by rest
          - "keyword query" — regex match against name + description

        Returns:
            List of matched BaseTool objects (up to MAX_RESULTS).
        """
        # ---- 模式 1：精确选择 ----
        if query.startswith("select:"):
            # 解析名称列表
            names = {n.strip() for n in query[7:].split(",")}
            # 返回名称匹配的工具体
            return [e.tool for e in self._entries if e.name in names][:MAX_RESULTS]

        # ---- 模式 2：关键词 + 排序 ----
        if query.startswith("+"):
            # 分割关键词和排序词
            parts = query[1:].split(None, 1)
            # 关键词必须出现在名称中（转为小写）
            required = parts[0].lower()
            # 筛选包含关键词的条目
            candidates = [e for e in self._entries if required in e.name.lower()]
            # 如果有排序词，使用正则表达式得分排序
            if len(parts) > 1:
                candidates.sort(
                    key=lambda e: _regex_score(parts[1], e),
                    reverse=True,
                )
            return [e.tool for e in candidates][:MAX_RESULTS]

        # ---- 模式 3：正则表达式搜索 ----
        try:
            # 尝试编译正则表达式
            regex = re.compile(query, re.IGNORECASE)
        except re.error:
            # 正则表达式无效时，使用转义后的字面量
            regex = re.compile(re.escape(query), re.IGNORECASE)

        # 评分：名称匹配得 2 分，描述匹配得 1 分
        scored = []
        for entry in self._entries:
            # 可搜索文本 = 名称 + 描述
            searchable = f"{entry.name} {entry.description}"
            if regex.search(searchable):
                # 名称匹配权重更高
                score = 2 if regex.search(entry.name) else 1
                scored.append((score, entry))

        # 按分数降序排序
        scored.sort(key=lambda x: x[0], reverse=True)
        # 返回工具列表
        return [entry.tool for _, entry in scored][:MAX_RESULTS]

    # entries 属性：获取所有条目的副本
    #
    # 返回值：
    #   list[DeferredToolEntry]：所有延迟工具条目的列表
    @property
    def entries(self) -> list[DeferredToolEntry]:
        # 返回列表的副本，避免外部修改
        return list(self._entries)

    # __len__：返回注册表中的工具数量
    def __len__(self) -> int:
        return len(self._entries)


# _regex_score：计算正则表达式在条目中匹配的次数
#
# 参数：
#   pattern: str，正则表达式模式
#   entry: DeferredToolEntry，延迟工具条目
#
# 返回值：
#   int：在条目名称和描述中找到的匹配次数
#
# 工作原理：
#   尝试将 pattern 作为正则表达式编译，
#   如果失败则转义为字面量。
#   然后在名称和描述中查找所有匹配。
def _regex_score(pattern: str, entry: DeferredToolEntry) -> int:
    try:
        # 尝试编译正则表达式
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        # 编译失败，使用转义后的字面量
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
    # 返回在名称和描述中找到的匹配次数
    return len(regex.findall(f"{entry.name} {entry.description}"))


# ============================================================
# 上下文级别的注册表管理（ContextVar）
# ============================================================

# _registry_var：上下文变量，用于存储当前请求的注册表
#
# 设计说明：
#   使用 ContextVar 而非模块级全局变量，防止并发请求相互干扰。
#   在基于 asyncio 的 LangGraph 中，每个图执行在独立的异步上下文，
#   因此每个请求获得独立的注册表值。
#   对于通过 loop.run_in_executor 运行的同步工具，
#   Python 会将当前上下文复制到工作线程，上下文变量值也会正确继承。
_registry_var: contextvars.ContextVar[DeferredToolRegistry | None] = contextvars.ContextVar(
    "deferred_tool_registry",
    default=None  # 默认值为 None，表示没有注册表
)


# get_deferred_registry：获取当前异步上下文的注册表
#
# 返回值：
#   DeferredToolRegistry | None：当前上下文的注册表，如果没有则返回 None
#
# 使用位置：
#   DeferredToolFilterMiddleware._filter_tools() 调用此函数
#   来源文件：deerflow/agents/middlewares/deferred_tool_filter_middleware.py
def get_deferred_registry() -> DeferredToolRegistry | None:
    # 获取当前上下文中的注册表值
    return _registry_var.get()


# set_deferred_registry：设置当前异步上下文的注册表
#
# 参数：
#   registry: DeferredToolRegistry，要设置的注册表
#
# 使用场景：
#   在请求开始时创建注册表并设置到上下文
def set_deferred_registry(registry: DeferredToolRegistry) -> None:
    # 设置当前上下文的注册表值
    _registry_var.set(registry)


# reset_deferred_registry：重置当前异步上下文的注册表
#
# 作用说明：
#   将注册表设置为 None，用于清理或重置状态
#
# 使用场景：
#   在请求结束时清理注册表
def reset_deferred_registry() -> None:
    """Reset the deferred registry for the current async context."""
    # 将注册表设置为 None
    _registry_var.set(None)


# ============================================================
# Tool 定义
# ============================================================

# tool_search：延迟工具发现工具
#
# 作用说明：
#   获取延迟工具的完整模式定义，使它们可以被调用。
#
# 工作流程：
#   延迟工具在系统提示的 <available-deferred-tools> 中显示名称。
#   在获取之前只知道名称，没有参数模式，工具无法被调用。
#   此工具接收查询，在延迟工具列表中匹配，返回匹配工具的完整定义。
#   一旦工具的模式出现在结果中，它就可以被调用了。
#
# 查询形式：
#   - "select:Read,Edit,Grep" — 按名称获取这些工具
#   - "notebook jupyter" — 关键词搜索，最多 max_results 个最佳匹配
#   - "+slack send" — 要求 "slack" 在名称中，其余术语用于排序
#
# 参数：
#   query: str，查询字符串，用于查找延迟工具
#
# 返回值：
#   str，匹配工具定义的 JSON 数组
@tool
def tool_search(query: str) -> str:
    """Fetches full schema definitions for deferred tools so they can be called.

    Deferred tools appear by name in <available-deferred-tools> in the system
    prompt. Until fetched, only the name is known — there is no parameter
    schema, so the tool cannot be invoked. This tool takes a query, matches
    it against the deferred tool list, and returns the matched tools' complete
    definitions. Once a tool's schema appears in that result, it is callable.

    Query forms:
      - "select:Read,Edit,Grep" — fetch these exact tools by name
      - "notebook jupyter" — keyword search, up to max_results best matches
      - "+slack send" — require "slack" in the name, rank by remaining terms

    Args:
        query: Query to find deferred tools. Use "select:<tool_name>" for
               direct selection, or keywords to search.

    Returns:
        Matched tool definitions as JSON array.
    """
    # 获取当前上下文的注册表
    registry = get_deferred_registry()

    # 如果没有注册表，返回提示信息
    if not registry:
        return "No deferred tools available."

    # 搜索匹配的工县
    matched_tools = registry.search(query)

    # 如果没有匹配的工具，返回提示信息
    if not matched_tools:
        return f"No tools found matching: {query}"

    # 使用 LangChain 内置的序列化生成 OpenAI 函数格式
    # 这是模型无关的：所有 LLM 都理解这种标准模式
    tool_defs = [convert_to_openai_function(t) for t in matched_tools[:MAX_RESULTS]]

    # 提升匹配的工县，使 DeferredToolFilterMiddleware 不再过滤
    # LLM 现在有了完整的模式，可以调用它们了
    registry.promote({t.name for t in matched_tools[:MAX_RESULTS]})

    # 返回 JSON 格式的工具定义
    return json.dumps(tool_defs, indent=2, ensure_ascii=False)
