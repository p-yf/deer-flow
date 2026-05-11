"""TitleMiddleware - 自动线程标题生成中间件。

功能概述：
  在第一次完整对话交换后（1 条用户消息 + 至少 1 条助手响应），自动生成线程标题。

生成时机：
  - 需要恰好 1 条用户消息 + 至少 1 条助手响应
  - 标题生成功能必须已启用（config.enabled）
  - 状态中还没有标题
  - 已经有标题时不重复生成

生成方式：
  - 异步版本（aafter_model）：调用 LLM 生成标题
    * 使用配置中的模型（config.model_name）或默认模型
    * 使用 {max_words}、{user_msg}、{assistant_msg} 填充 prompt 模板
    * 添加 "middleware:title" 标签，确保 RunJournal 正确标识
  - 同步版本（after_model）：使用本地回退
    * 直接取用户消息的前 50 字符作为标题

标题格式：
  - 提取用户消息和助手消息的前 500 字符
  - 使用 prompt 模板生成（受 max_words 和 max_chars 限制）
  - 最终截断到 max_chars（默认 50）字符

执行位置：紧接 TokenUsageMiddleware 之后（如果有的话），在 MemoryMiddleware 之前。
"""
"""Middleware for automatic thread title generation."""

# 导入标准库 logging，用于记录日志
import logging
# typing 导入 Any（任意类型）、NotRequired（可选字段标记）、override（方法重写标记）
from typing import Any, NotRequired, override

# 从 langchain.agents 导入 AgentState（agent 基础状态类）
from langchain.agents import AgentState
# 从 langchain.agents.middleware 导入 AgentMiddleware（中间件基类）
from langchain.agents.middleware import AgentMiddleware
# 从 langgraph.config 导入 get_config，用于获取 LangGraph 的运行时配置
from langgraph.config import get_config
# 从 langgraph.runtime 导入 Runtime（LangGraph 运行时上下文）
from langgraph.runtime import Runtime

# 从 deerflow.config.title_config 导入 get_title_config
# 获取标题生成配置（如是否启用、最大词数、最大字符数、prompt 模板等）
from deerflow.config.title_config import get_title_config
# 从 deerflow.models 导入 create_chat_model
# 用于创建标题生成的 LLM 模型实例
from deerflow.models import create_chat_model

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)


# TitleMiddlewareState 类：定义中间件使用的状态 schema
# 继承自 AgentState
class TitleMiddlewareState(AgentState):
    # title 字段：线程标题
    # 类型是 str | None，表示可选字段
    title: NotRequired[str | None]


# TitleMiddleware 类：自动生成线程标题
#
# 工作流程：
#   1. 在 after_model（模型执行后）钩子中检查是否需要生成标题
#   2. 首次交换完成时（第一条用户消息 + 第一条助手响应）触发标题生成
#   3. 异步调用 LLM 生成标题（或失败时使用本地回退）
#   4. 将标题存入状态
#
# 生成时机：
#   - 只在第一个完整交换后生成（1 条用户消息 + 至少 1 条助手响应）
#   - 已经存在标题时不重复生成
#
# 实现方式：
#   - 主要使用异步 LLM 调用（ainvoke）
#   - 失败时使用本地回退（取用户消息的前 50 字符）
class TitleMiddleware(AgentMiddleware[TitleMiddlewareState]):
    """Automatically generate a title for the thread after the first user message."""

    # state_schema 类变量，指定该中间件使用的状态类型
    state_schema = TitleMiddlewareState

    # 内部方法：标准化消息内容为纯文本
    #
    # 消息的 content 字段可能是：
    #   - str：直接返回
    #   - list：递归处理每个元素，合并非空文本
    #   - dict：尝试获取 text 字段或递归处理 content 字段
    #   - 其他：返回空字符串
    #
    # 参数：
    #   content: 消息的 content 字段
    #
    # 返回值：
    #   字符串，标准化后的文本内容
    def _normalize_content(self, content: object) -> str:
        # 如果是字符串，直接返回
        if isinstance(content, str):
            return content

        # 如果是列表，递归处理每个元素
        if isinstance(content, list):
            # 处理每个元素，合并非空结果
            parts = [self._normalize_content(item) for item in content]
            return "\n".join(part for part in parts if part)

        # 如果是字典
        if isinstance(content, dict):
            # 尝试获取 text 字段
            text_value = content.get("text")
            if isinstance(text_value, str):
                return text_value

            # 尝试获取 content 字段并递归处理
            nested_content = content.get("content")
            if nested_content is not None:
                return self._normalize_content(nested_content)

        # 其他类型，返回空字符串
        return ""

    # 内部方法：检查是否应该生成标题
    #
    # 生成条件：
    #   1. 标题生成功能已启用（config.enabled）
    #   2. 状态中还没有标题
    #   3. 是第一个完整交换（1 条用户消息 + 至少 1 条助手响应）
    #
    # 参数：
    #   state: TitleMiddlewareState，当前 agent 状态
    #
    # 返回值：
    #   bool，True 表示应该生成标题
    def _should_generate_title(self, state: TitleMiddlewareState) -> bool:
        # 获取标题配置
        config = get_title_config()

        # 检查是否启用
        if not config.enabled:
            return False

        # 检查是否已有标题
        if state.get("title"):
            return False

        # 获取消息列表
        messages = state.get("messages", [])
        # 至少需要两条消息（用户 + 助手）
        if len(messages) < 2:
            return False

        # 统计用户消息和助手消息数量
        user_messages = [m for m in messages if m.type == "human"]
        assistant_messages = [m for m in messages if m.type == "ai"]

        # 在第一次完整交换后生成标题
        # 条件：恰好 1 条用户消息 + 至少 1 条助手响应
        return len(user_messages) == 1 and len(assistant_messages) >= 1

    # 内部方法：构建标题生成的 prompt
    #
    # 参数：
    #   state: TitleMiddlewareState，当前 agent 状态
    #
    # 返回值：
    #   (prompt_string, user_msg) 元组
    #   - prompt_string: 用于调用 LLM 的 prompt
    #   - user_msg: 用户消息内容（标准化后），用于回退标题
    def _build_title_prompt(self, state: TitleMiddlewareState) -> tuple[str, str]:
        # 获取标题配置
        config = get_title_config()
        # 获取消息列表
        messages = state.get("messages", [])

        # 提取第一条用户消息的内容
        user_msg_content = next((m.content for m in messages if m.type == "human"), "")
        # 提取第一条助手消息的内容
        assistant_msg_content = next((m.content for m in messages if m.type == "ai"), "")

        # 标准化内容为纯文本
        user_msg = self._normalize_content(user_msg_content)
        assistant_msg = self._normalize_content(assistant_msg_content)

        # 使用配置中的 prompt 模板构建 prompt
        # 模板中 {max_words}、{user_msg}、{assistant_msg} 被替换
        prompt = config.prompt_template.format(
            max_words=config.max_words,  # 最大词数
            user_msg=user_msg[:500],     # 用户消息（截断到 500 字符）
            assistant_msg=assistant_msg[:500],  # 助手消息（截断到 500 字符）
        )
        return prompt, user_msg

    # 内部方法：解析 LLM 输出为标题字符串
    #
    # 参数：
    #   content: LLM 返回的 content（可能是 str、list 或 dict）
    #
    # 返回值：
    #   字符串，处理后的标题
    #
    # 处理步骤：
    #   1. 标准化为纯文本
    #   2. 去除首尾空白和引号
    #   3. 截断到最大字符数
    def _parse_title(self, content: object) -> str:
        # 获取标题配置
        config = get_title_config()
        # 标准化 LLM 输出
        title_content = self._normalize_content(content)
        # 去除首尾空白和引号
        title = title_content.strip().strip('"').strip("'")
        # 如果超过最大字符数，截断
        return title[: config.max_chars] if len(title) > config.max_chars else title

    # 内部方法：生成本地回退标题
    #
    # 当 LLM 调用失败时使用
    # 直接取用户消息的前 N 个字符作为标题
    #
    # 参数：
    #   user_msg: 用户消息内容
    #
    # 返回值：
    #   字符串，回退标题
    def _fallback_title(self, user_msg: str) -> str:
        # 获取标题配置
        config = get_title_config()
        # 回退标题最大字符数（配置值和 50 中较小者）
        fallback_chars = min(config.max_chars, 50)

        # 如果用户消息超过回退长度，截断并添加 "..."
        if len(user_msg) > fallback_chars:
            return user_msg[:fallback_chars].rstrip() + "..."

        # 否则直接返回用户消息（如果非空）
        return user_msg if user_msg else "New Conversation"

    # 内部方法：获取 RunnableConfig 并添加中间件标签
    #
    # 这个方法继承父级 RunnableConfig 并添加 "middleware:title" 标签
    # 确保 RunJournal 将来自此中间件的 LLM 调用标识为 "middleware:title"
    # 而不是 "lead_agent"
    #
    # 返回值：
    #   字典，包含合并后的配置
    def _get_runnable_config(self) -> dict[str, Any]:
        """Inherit the parent RunnableConfig and add middleware tag.

        This ensures RunJournal identifies LLM calls from this middleware
        as ``middleware:title`` instead of ``lead_agent``.
        """
        try:
            # 尝试获取父级配置
            parent = get_config()
        except Exception:
            # 获取失败，使用空字典
            parent = {}

        # 合并配置
        config = {**parent}
        # 添加中间件标签
        config["tags"] = [*(config.get("tags") or []), "middleware:title"]
        return config

    # 内部方法：同步版本生成标题结果
    #
    # 注意：这是同步方法，内部实际使用本地回退
    # 真正的 LLM 调用在异步版本中
    #
    # 参数：
    #   state: TitleMiddlewareState，当前 agent 状态
    #
    # 返回值：
    #   状态更新字典（包含 title），如果不需要生成则返回 None
    def _generate_title_result(self, state: TitleMiddlewareState) -> dict | None:
        """Generate a local fallback title without blocking on an LLM call."""
        # 检查是否需要生成
        if not self._should_generate_title(state):
            return None

        # 获取用户消息，构建回退标题
        _, user_msg = self._build_title_prompt(state)
        return {"title": self._fallback_title(user_msg)}

    # 内部方法：异步版本生成标题结果
    #
    # 工作流程：
    #   1. 检查是否需要生成
    #   2. 构建 prompt
    #   3. 调用 LLM 生成标题
    #   4. 成功：返回解析后的标题
    #   5. 失败：返回本地回退标题
    #
    # 参数：
    #   state: TitleMiddlewareState，当前 agent 状态
    #
    # 返回值：
    #   状态更新字典（包含 title）
    async def _agenerate_title_result(self, state: TitleMiddlewareState) -> dict | None:
        """Generate a title asynchronously and fall back locally on failure."""
        # 检查是否需要生成
        if not self._should_generate_title(state):
            return None

        # 获取配置和 prompt
        config = get_title_config()
        prompt, user_msg = self._build_title_prompt(state)

        try:
            # 创建 LLM 模型
            if config.model_name:
                # 使用配置的模型名
                model = create_chat_model(name=config.model_name, thinking_enabled=False)
            else:
                # 使用默认模型
                model = create_chat_model(thinking_enabled=False)

            # 调用 LLM 生成标题
            # 使用 _get_runnable_config() 获取带标签的配置
            response = await model.ainvoke(prompt, config=self._get_runnable_config())

            # 解析标题
            title = self._parse_title(response.content)
            if title:
                return {"title": title}

        except Exception:
            # LLM 调用失败，记录日志
            logger.debug("Failed to generate async title; falling back to local title", exc_info=True)

        # 返回本地回退标题
        return {"title": self._fallback_title(user_msg)}

    # after_model 钩子方法：同步版本
    #
    # 注意：同步版本使用本地回退，不阻塞等待 LLM
    # 异步版本才能真正调用 LLM 生成标题
    @override
    def after_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        return self._generate_title_result(state)

    # aafter_model 钩子方法：异步版本
    #
    # 这是实际的标题生成逻辑
    @override
    async def aafter_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        return await self._agenerate_title_result(state)