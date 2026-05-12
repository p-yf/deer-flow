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

# ============================================================
# 导入标准库
# ============================================================

# logging：标准库日志模块，用于记录中间件运行日志
import logging

# typing 导入：
#   - Any：任意类型，用于 _get_runnable_config 的返回类型
#   - NotRequired：可选字段标记，用于状态字段定义
#   - override：方法重写标记，用于明确表示重写父类方法
from typing import Any, NotRequired, override

# ============================================================
# 导入 LangChain / LangGraph 相关模块
# ============================================================

# langchain.agents.AgentState：
#   LangChain agent 的基础状态类，所有自定义状态类继承自此
#   TitleMiddlewareState 继承此类
from langchain.agents import AgentState

# langchain.agents.middleware.AgentMiddleware：
#   LangChain 的中间件基类，所有自定义中间件必须继承此类
from langchain.agents.middleware import AgentMiddleware

# langgraph.config.get_config：
#   获取 LangGraph 的运行时配置
#   用于获取父级 RunnableConfig，合并后添加中间件标签
from langgraph.config import get_config

# langgraph.runtime.Runtime：
#   LangGraph 运行时上下文
#   在钩子方法中作为参数传入
from langgraph.runtime import Runtime

# ============================================================
# 导入 DeerFlow 项目内部模块
# ============================================================

# deerflow.config.title_config.get_title_config：
#   获取标题生成配置（TitleConfig 实例）
#   包含 enabled、max_words、max_chars、prompt_template、model_name 等
#   来自：本项目 packages/harness/deerflow/config/title_config.py
from deerflow.config.title_config import get_title_config

# deerflow.models.create_chat_model：
#   创建 LLM 模型实例的工厂函数
#   用于创建标题生成的 LLM 模型
#   来自：本项目 packages/harness/deerflow/models/__init__.py
from deerflow.models import create_chat_model

# ============================================================
# 模块级变量初始化
# ============================================================

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)


# ============================================================
# 状态类型定义
# ============================================================

# TitleMiddlewareState 类：定义中间件使用的状态 schema
#
# 作用说明：
#   继承自 AgentState，作为 TitleMiddleware 的状态类型。
#   添加了 title 字段，用于存储自动生成的线程标题。
class TitleMiddlewareState(AgentState):
    # title 字段：线程标题
    # 类型是 str | None，表示可选字段
    # 如果已经有标题（如用户手动设置），则不生成新的
    title: NotRequired[str | None]


# ============================================================
# TitleMiddleware 主类
# ============================================================

# TitleMiddleware 类：自动生成线程标题
#
# 核心作用：
#   在第一次完整对话交换后（1 条用户消息 + 至少 1 条助手响应），
#   自动生成一个简洁的线程标题。
#
# 工作流程：
#   1. 在 after_model / aafter_model 钩子中检查是否需要生成标题
#   2. 检查标题生成功能是否启用、状态中是否已有标题
#   3. 检查是否是第一次完整交换（恰好 1 条用户消息 + 至少 1 条助手响应）
#   4. 如果需要生成：
#      - 同步版本：使用本地回退（取用户消息的前 50 字符）
#      - 异步版本：调用 LLM 生成更智能的标题
#   5. 将标题存入状态（state.title）
#
# 设计考虑：
#   - 同步版本是 fast fallback，不会阻塞，不调用 LLM
#   - 异步版本可以生成更智能的标题，但会有额外延迟
#   - 标题只生成一次（如果已有标题则不再生成）
class TitleMiddleware(AgentMiddleware[TitleMiddlewareState]):
    """Automatically generate a title for the thread after the first user message."""

    # state_schema：类变量，指定该中间件使用的状态类型
    state_schema = TitleMiddlewareState

    # ============================================================
    # 内部辅助方法
    # ============================================================

    # _normalize_content：标准化消息内容为纯文本
    #
    # 方法作用：
    #   将消息的 content 字段（可能是 str、list、dict 等）统一转换为纯文本字符串。
    #   用于提取用户消息和助手消息的文本内容，传递给 LLM 或用于标题生成。
    #
    # 参数：
    #   content: object，消息的 content 字段
    #     可能是：
    #       - str：直接返回
    #       - list：[...] 格式，递归处理每个元素
    #       - dict：{"type": "text", "text": "..."} 或 {"type": "image_url", ...} 格式
    #       - 其他：返回空字符串
    #
    # 返回值：
    #   str：标准化后的纯文本内容
    def _normalize_content(self, content: object) -> str:
        # 如果 content 已经是字符串，直接返回
        if isinstance(content, str):
            return content

        # 如果 content 是列表，递归处理每个元素并合并
        # 用于处理 [{type: "text", text: "..."}, ...] 格式
        if isinstance(content, list):
            # 递归调用 _normalize_content 处理每个元素
            parts = [self._normalize_content(item) for item in content]
            # 用换行符合并非空部分
            return "\n".join(part for part in parts if part)

        # 如果 content 是字典，尝试获取 text 字段
        if isinstance(content, dict):
            # dict.get(key, default) 获取指定键的值
            text_value = content.get("text")
            # 如果 text_value 是字符串，返回它
            if isinstance(text_value, str):
                return text_value

            # 某些格式是 {"type": "text", "text": "..."} 但 content 字段又嵌套了
            nested_content = content.get("content")
            if nested_content is not None:
                # 递归处理嵌套的 content
                return self._normalize_content(nested_content)

        # 其他类型（数字、None 等），返回空字符串
        return ""

    # _should_generate_title：检查是否应该生成标题
    #
    # 方法作用：
    #   综合检查多个条件，判断当前是否应该生成标题。
    #
    # 参数：
    #   state: TitleMiddlewareState，当前 agent 状态
    #
    # 返回值：
    #   bool：如果应该生成标题返回 True
    #
    # 生成条件（全部满足才返回 True）：
    #   1. 标题生成功能已启用（config.enabled）
    #   2. 状态中还没有标题（state.title 为空）
    #   3. 消息数量 >= 2（至少 1 条用户消息 + 1 条助手响应）
    #   4. 恰好有 1 条用户消息且至少有一条助手消息
    def _should_generate_title(self, state: TitleMiddlewareState) -> bool:
        # get_title_config() 获取标题配置（全局单例）
        config = get_title_config()

        # 检查 1：标题生成功能是否启用
        if not config.enabled:
            return False

        # 检查 2：状态中是否已有标题
        # state.get("title") 获取 title 字段，如果不存在返回 None
        if state.get("title"):
            return False

        # 获取消息列表
        messages = state.get("messages", [])
        # 检查 3：消息数量是否足够（至少 2 条）
        if len(messages) < 2:
            return False

        # 统计用户消息和助手消息数量
        # m.type 获取消息类型，"human" 表示用户消息
        user_messages = [m for m in messages if m.type == "human"]
        # m.type == "ai" 表示助手消息
        assistant_messages = [m for m in messages if m.type == "ai"]

        # 检查 4：恰好 1 条用户消息 + 至少 1 条助手响应
        # 这是"第一次完整交换"的判断条件
        return len(user_messages) == 1 and len(assistant_messages) >= 1

    # _build_title_prompt：构建标题生成的 prompt
    #
    # 方法作用：
    #   从状态中提取用户消息和助手消息，构造成 LLM 调用的 prompt。
    #
    # 参数：
    #   state: TitleMiddlewareState，当前 agent 状态
    #
    # 返回值：
    #   tuple[str, str]：元组
    #     - prompt_string: 用于调用 LLM 的 prompt 字符串
    #     - user_msg: 用户消息内容（标准化后），用于回退标题
    #
    # 实现逻辑：
    #   1. 从配置中获取 prompt_template
    #   2. 提取第一条用户消息和第一条助手消息
    #   3. 标准化内容为纯文本（调用 _normalize_content）
    #   4. 用 max_words、user_msg（截断到 500 字符）、assistant_msg（截断到 500 字符）填充模板
    def _build_title_prompt(self, state: TitleMiddlewareState) -> tuple[str, str]:
        # 获取标题配置
        config = get_title_config()
        # 获取消息列表
        messages = state.get("messages", [])

        # next(...) 获取第一条匹配的消息的 content
        # 如果没有找到，返回空字符串作为默认值
        user_msg_content = next((m.content for m in messages if m.type == "human"), "")
        assistant_msg_content = next((m.content for m in messages if m.type == "ai"), "")

        # 标准化内容为纯文本
        # 这样可以处理 list 或 dict 格式的 content
        user_msg = self._normalize_content(user_msg_content)
        assistant_msg = self._normalize_content(assistant_msg_content)

        # str.format() 格式化字符串
        # config.prompt_template 是类似 "生成标题：{max_words} {user_msg} {assistant_msg}" 的模板
        # max_words: 配置的最大词数
        # user_msg[:500]: 用户消息截断到 500 字符
        # assistant_msg[:500]: 助手消息截断到 500 字符
        prompt = config.prompt_template.format(
            max_words=config.max_words,
            user_msg=user_msg[:500],
            assistant_msg=assistant_msg[:500],
        )
        # 返回 (prompt, user_msg) 元组
        # prompt 用于 LLM 调用，user_msg 用于回退标题
        return prompt, user_msg

    # _parse_title：解析 LLM 输出为标题字符串
    #
    # 方法作用：
    #   将 LLM 返回的 content 标准化为标题字符串。
    #
    # 参数：
    #   content: object，LLM 返回的 content（可能是 str、list 或 dict）
    #
    # 返回值：
    #   str：处理后的标题字符串
    #
    # 处理步骤：
    #   1. 调用 _normalize_content 标准化为纯文本
    #   2. strip() 去除首尾空白
    #   3. strip('"') 去除首尾的双引号
    #   4. strip("'") 去除首尾的单引号（处理 LLM 可能添加的引号）
    #   5. 如果超过 max_chars，截断
    def _parse_title(self, content: object) -> str:
        # 获取配置
        config = get_title_config()
        # 标准化 LLM 输出
        title_content = self._normalize_content(content)
        # strip() 去除首尾空白和引号
        title = title_content.strip().strip('"').strip("'")
        # 如果标题超过最大字符数限制，截断
        # config.max_chars 默认是 50
        if len(title) > config.max_chars:
            return title[: config.max_chars]
        return title

    # _fallback_title：生成本地回退标题
    #
    # 方法作用：
    #   当 LLM 调用失败时，使用本地策略生成标题。
    #   直接取用户消息的前 N 个字符作为标题。
    #
    # 参数：
    #   user_msg: str，用户消息内容（标准化后）
    #
    # 返回值：
    #   str：回退标题
    #     - 如果用户消息 <= fallback_chars，返回用户消息
    #     - 如果用户消息 > fallback_chars，截断并添加 "..."
    #     - 如果用户消息为空，返回 "New Conversation"
    def _fallback_title(self, user_msg: str) -> str:
        # 获取配置
        config = get_title_config()
        # 回退标题的最大字符数
        # 取配置值和 50 中较小者，确保回退标题不会太长
        fallback_chars = min(config.max_chars, 50)

        # 如果用户消息超过回退长度
        if len(user_msg) > fallback_chars:
            # rstrip() 去除末尾空白（避免截断后末尾有空格）
            # 然后添加 "..." 表示截断
            return user_msg[:fallback_chars].rstrip() + "..."

        # 用户消息非空，返回用户消息
        if user_msg:
            return user_msg

        # 用户消息为空，返回默认标题
        return "New Conversation"

    # _get_runnable_config：获取 RunnableConfig 并添加中间件标签
    #
    # 方法作用：
    #   继承父级 RunnableConfig 并添加 "middleware:title" 标签。
    #   确保 RunJournal 将来自此中间件的 LLM 调用标识为 "middleware:title"
    #   而不是 "lead_agent"。
    #
    # 返回值：
    #   dict[str, Any]：包含合并后配置的字典
    #
    # 实现逻辑：
    #   1. 调用 get_config() 获取父级配置
    #   2. 解包父级配置到新字典
    #   3. 合并 tags，添加 "middleware:title"
    def _get_runnable_config(self) -> dict[str, Any]:
        """Inherit the parent RunnableConfig and add middleware tag.

        This ensures RunJournal identifies LLM calls from this middleware
        as ``middleware:title`` instead of ``lead_agent``.
        """
        try:
            # 尝试获取父级 RunnableConfig
            parent = get_config()
        except Exception:
            # 获取失败（可能在某些测试环境中），使用空字典
            parent = {}

        # {**parent} 解包并创建新字典（浅拷贝）
        config = {**parent}
        # config.get("tags", []) 获取 tags，如果不存在则返回空列表
        # [*(...), "middleware:title"] 创建新列表，添加 "middleware:title"
        config["tags"] = [*(config.get("tags") or []), "middleware:title"]
        return config

    # _generate_title_result：同步版本生成标题结果
    #
    # 方法作用：
    #   同步生成标题，使用本地回退策略（不调用 LLM）。
    #   这是 fast fallback，用于不希望阻塞等待 LLM 的场景。
    #
    # 参数：
    #   state: TitleMiddlewareState，当前 agent 状态
    #
    # 返回值：
    #   dict | None：状态更新字典（包含 title），如果不需要生成则返回 None
    #
    # 实现逻辑：
    #   1. 调用 _should_generate_title 检查是否需要生成
    #   2. 如果不需要，返回 None
    #   3. 如果需要，调用 _build_title_prompt 获取用户消息
    #   4. 调用 _fallback_title 生成本地回退标题
    #   5. 返回 {"title": title}
    def _generate_title_result(self, state: TitleMiddlewareState) -> dict | None:
        """Generate a local fallback title without blocking on an LLM call."""
        # 检查是否需要生成
        if not self._should_generate_title(state):
            return None

        # 构建 prompt 并获取用户消息
        # _ 返回值被忽略（prompt 不用于本地回退）
        _, user_msg = self._build_title_prompt(state)
        # 生成本地回退标题并返回状态更新
        return {"title": self._fallback_title(user_msg)}

    # _agenerate_title_result：异步版本生成标题结果
    #
    # 方法作用：
    #   异步生成标题，调用 LLM 生成智能标题，失败时回退到本地策略。
    #   这是主要的标题生成逻辑。
    #
    # 参数：
    #   state: TitleMiddlewareState，当前 agent 状态
    #
    # 返回值：
    #   dict：状态更新字典（包含 title）
    #
    # 实现逻辑：
    #   1. 调用 _should_generate_title 检查是否需要生成
    #   2. 如果不需要，返回 None
    #   3. 调用 _build_title_prompt 构建 LLM prompt
    #   4. 创建 LLM 模型（使用配置或默认）
    #   5. 调用 model.ainvoke(prompt) 生成标题
    #   6. 解析标题并返回
    #   7. 如果任何步骤失败，返回本地回退标题
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
            # create_chat_model(name=..., thinking_enabled=False) 创建聊天模型
            # thinking_enabled=False 禁用思考模式，因为标题生成不需要
            if config.model_name:
                # 使用配置中指定的模型名
                model = create_chat_model(name=config.model_name, thinking_enabled=False)
            else:
                # 使用默认模型
                model = create_chat_model(thinking_enabled=False)

            # 调用 LLM 生成标题
            # self._get_runnable_config() 获取带标签的配置
            # await model.ainvoke(prompt, config=...) 异步调用模型
            response = await model.ainvoke(prompt, config=self._get_runnable_config())

            # 解析标题
            # response.content 是 LLM 返回的内容
            title = self._parse_title(response.content)
            if title:
                return {"title": title}

        except Exception:
            # LLM 调用失败（包括网络错误、超时、解析错误等）
            # 记录调试日志，包含异常堆栈
            logger.debug("Failed to generate async title; falling back to local title", exc_info=True)

        # 返回本地回退标题
        return {"title": self._fallback_title(user_msg)}

    # ============================================================
    # LangChain AgentMiddleware 钩子方法
    # ============================================================

    # after_model：同步版本的模型调用后钩子
    #
    # 方法作用：
    #   LangChain AgentMiddleware 定义的钩子之一，
    #   在每次模型执行后同步执行。
    #
    # 调用时机：
    #   当 agent 以同步方式调用时（agent.invoke()）
    #
    # 参数：
    #   state: TitleMiddlewareState，当前状态
    #   runtime: Runtime，LangGraph 运行时上下文
    #
    # 返回值：
    #   dict | None：状态更新（包含 title），如果不需要更新则返回 None
    #
    # 注意：
    #   同步版本使用本地回退，不阻塞等待 LLM
    @override
    def after_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        # 委托给 _generate_title_result
        return self._generate_title_result(state)

    # aafter_model：异步版本的模型调用后钩子
    #
    # 方法作用：
    #   LangChain AgentMiddleware 定义的钩子之一，
    #   在每次模型执行后异步执行。
    #
    # 调用时机：
    #   当 agent 以异步方式调用时（await agent.ainvoke()）
    #
    # 参数：
    #   state: TitleMiddlewareState，当前状态
    #   runtime: Runtime，LangGraph 运行时上下文
    #
    # 返回值：
    #   dict | None：状态更新（包含 title）
    #
    # 注意：
    #   这是实际的标题生成逻辑，会调用 LLM
    @override
    async def aafter_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        # 委托给 _agenerate_title_result
        return await self._agenerate_title_result(state)