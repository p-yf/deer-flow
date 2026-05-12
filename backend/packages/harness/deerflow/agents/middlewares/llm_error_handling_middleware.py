"""LLMErrorHandlingMiddleware - LLM 错误处理与重试中间件。

功能概述：
  包装 LLM 模型调用，捕获临时性错误，自动进行指数退避重试，失败时返回友好消息。

工作流程：
  1. wrap_model_call / awrap_model_call 包装模型调用
  2. 捕获异常并分类：
     - quota（配额错误）：不重试，账户问题
     - auth（认证错误）：不重试，凭据问题
     - transient（临时错误）：可重试（超时、连接错误、5xx）
     - busy（服务器繁忙）：可重试
  3. 对于可重试类型，进行指数退避重试（最多 3 次）
  4. 重试期间通过 StreamWriter 发送 llm_retry 事件（供前端显示进度）
  5. 重试耗尽或非重试类型错误，返回友好的用户消息（AIMessage）
  6. GraphBubbleUp 等 LangGraph 控制流信号正确向上传播

重试配置：
  - retry_max_attempts = 3（默认）
  - retry_base_delay_ms = 1000（默认，1 秒）
  - retry_cap_delay_ms = 8000（默认，上限 8 秒）
  - 延迟计算：min(base_delay * 2^(attempt-1), cap_delay)
  - 如果异常包含 Retry-After 头，使用服务器指定的时间

错误分类模式：
  - 可重试状态码：408, 409, 425, 429, 500, 502, 503, 504
  - 繁忙模式：server busy, 负载较高, 服务繁忙, rate limit 等
  - 配额模式：insufficient_quota, 余额不足, 欠费 等
  - 认证模式：invalid api key, unauthorized, 无权 等

执行位置：在模型调用链的最外层包装（第一个或最后一个中间件）。
"""
"""Middleware for LLM error handling and retry with exponential backoff.

Wraps LLM model calls to:
1. Catch and classify exceptions (quota, auth, transient, busy)
2. Retry transient errors with exponential backoff (max 3 attempts)
3. Emit llm_retry events via StreamWriter during retries
4. Return user-friendly error messages on failure
5. Properly propagate LangGraph control flow signals (GraphBubbleUp)
"""

# ============================================================
# 导入标准库
# ============================================================

# __future__ 导入 annotations，使类型注解可以引用尚未定义的类（向前引用）
# 这允许在类定义之前引用该类作为类型注解
from __future__ import annotations

# asyncio：标准库异步模块，用于异步 sleep（重试等待）
import asyncio

# logging：标准库日志模块，用于记录重试和错误信息
import logging

# time：标准库时间模块，用于同步 sleep（同步版本的重试等待）
import time

# collections.abc 导入：
#   - Awaitable：异步可等待对象类型，用于异步 handler 的返回类型注解
#   - Callable：可调用对象类型，用于 handler 参数的类型注解
from collections.abc import Awaitable, Callable

# email.utils.parsedate_to_datetime：解析 HTTP Retry-After 头的时间戳
# 这是 email 模块的标准库函数，用于解析 HTTP 日期字符串
from email.utils import parsedate_to_datetime

# typing 导入：
#   - Any：任意类型，用于错误码等类型不确定的情况
#   - override：方法重写标记，用于明确表示重写父类方法
from typing import Any, override

# ============================================================
# 导入 LangChain / LangGraph 相关模块
# ============================================================

# langchain.agents.AgentState：
#   LangChain agent 的基础状态类，所有自定义状态类继承自此
from langchain.agents import AgentState

# langchain.agents.middleware.AgentMiddleware：
#   LangChain 的中间件基类，所有自定义中间件必须继承此类
from langchain.agents.middleware import AgentMiddleware

# langchain.agents.middleware.types 导入中间件相关的类型定义：
#   - ModelCallResult：模型调用的结果（包含生成的 AIMessage）
#   - ModelRequest：模型调用的请求（包含 model、messages 等）
#   - ModelResponse：模型调用响应（由 handler 返回）
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

# langchain_core.messages.AIMessage：
#   AI 消息类型，用于构建降级响应（当重试耗尽时返回的错误消息）
from langchain_core.messages import AIMessage

# langgraph.errors.GraphBubbleUp：
#   LangGraph 的控制流信号，用于中断/暂停/恢复
#   中间件必须正确传播这些信号，不能被错误捕获逻辑拦截
from langgraph.errors import GraphBubbleUp

# ============================================================
# 模块级变量初始化
# ============================================================

# 创建模块级 logger，用于记录中间件运行日志
logger = logging.getLogger(__name__)

# ============================================================
# 模块级常量定义
# ============================================================

# _RETRIABLE_STATUS_CODES：可重试的 HTTP 状态码集合
# 这些状态码表示临时性服务器错误，可以安全地重试
# 408: 请求超时, 409: 冲突, 425: 过于早请求, 429: 请求过多
# 500-504: 服务器错误（内部错误、服务不可用、网关超时等）
_RETRIABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

# _BUSY_PATTERNS：检测"服务器繁忙"错误的模式元组（英文 + 中文）
# 这些模式出现在错误消息中，表示临时性过载，可以重试
# 使用元组而不是列表，因为元组是不可变的，更适合作为常量
_BUSY_PATTERNS = (
    "server busy",              # 服务器繁忙
    "temporarily unavailable",  # 暂时不可用
    "try again later",          # 稍后重试
    "please retry",             # 请重试
    "please try again",         # 请再试一次
    "overloaded",               # 过载
    "high demand",              # 高需求
    "rate limit",               # 速率限制
    # 中文繁忙模式
    "负载较高",                 # 服务器负载较高
    "服务繁忙",                 # 服务繁忙
    "稍后重试",                 # 稍后重试
    "请稍后重试",               # 请稍后重试
)

# _QUOTA_PATTERNS：检测"配额/计费"错误的模式元组
# 这类错误不应该重试，因为不是临时性的（配额耗尽、欠费等）
_QUOTA_PATTERNS = (
    "insufficient_quota",   # 配额不足
    "quota",                # 配额
    "billing",              # 计费
    "credit",               # 信用/额度
    "payment",              # 支付
    # 中文配额模式
    "余额不足",             # 账户余额不足
    "超出限额",             # 超出限制
    "额度不足",             # 额度不足
    "欠费",                 # 欠费
)

# _AUTH_PATTERNS：检测"认证/权限"错误的模式元组
# 这类错误不应该重试，是配置问题而非临时性问题
_AUTH_PATTERNS = (
    "authentication",       # 认证
    "unauthorized",         # 未授权
    "invalid api key",      # 无效 API 密钥
    "invalid_api_key",      # 无效 API 密钥（下划线版本）
    "permission",           # 权限
    "forbidden",            # 禁止访问
    "access denied",        # 访问被拒绝
    # 中文权限模式
    "无权",                 # 无权访问
    "未授权",               # 未授权
)


# ============================================================
# LLMErrorHandlingMiddleware 主类
# ============================================================

# LLMErrorHandlingMiddleware 类：LLM 错误处理与重试中间件
#
# 核心作用：
#   包装 LLM 模型调用，捕获临时性错误，自动进行指数退避重试，
#   失败时返回用户友好的错误消息，同时正确传播 LangGraph 控制流信号。
#
# 工作流程：
#   1. 在 wrap_model_call / awrap_model_call 中包装模型调用
#   2. 捕获异常并分类：transient（临时性）/ busy（繁忙）/ quota（配额）/ auth（认证）/ generic（通用）
#   3. 对于 transient/busy 类型，进行指数退避重试（最多 retry_max_attempts 次）
#   4. 重试期间通过 StreamWriter 发送 llm_retry 事件，供前端显示进度
#   5. 如果重试耗尽或非重试类型错误，返回一条友好的用户消息（AIMessage）
#   6. GraphBubbleUp 等 LangGraph 控制信号会被正确向上传播，不被截获
#
# 设计考虑：
#   - 同步和异步版本逻辑相同，异步版本使用 await asyncio.sleep
#   - 所有异常都通过 _build_user_message 转换为用户友好的消息
#   - 通过 Retry-After 头实现精确等待时间（如果服务器指定了的话）
#   - 正确的错误分类避免无意义的重试（如配额问题不会重试）
class LLMErrorHandlingMiddleware(AgentMiddleware[AgentState]):
    """Retry transient LLM errors and surface graceful assistant messages."""

    # state_schema：类变量，指定该中间件使用的状态类型
    state_schema = AgentState

    # retry_max_attempts：最大重试次数（类属性）
    # 可以通过构造函数参数覆盖
    retry_max_attempts: int = 3

    # retry_base_delay_ms：基础退避延迟（毫秒）
    # 第一次重试等待 base_delay * 2^0 = 1000ms
    retry_base_delay_ms: int = 1000

    # retry_cap_delay_ms：最大延迟上限（毫秒）
    # 避免等待时间过长，默认 8 秒
    retry_cap_delay_ms: int = 8000

    # ============================================================
    # 内部辅助方法
    # ============================================================

    # _classify_error：对异常进行分类
    #
    # 方法作用：
    #   分析异常的特征，判断它是什么类型的错误，以及是否应该重试。
    #
    # 参数：
    #   exc: BaseException，要分类的异常
    #
    # 返回值：
    #   tuple[bool, str]：元组 (是否可重试, 错误原因)
    #   - 第一个元素：如果为 True 表示这是临时性错误，可以重试
    #   - 第二个元素：错误原因分类，可能是 "transient"/"busy"/"quota"/"auth"/"generic"
    #
    # 分类逻辑：
    #   1. 配额错误（quota）：不重试，直接返回友好消息
    #   2. 认证错误（auth）：不重试，需要检查凭据
    #   3. 超时/连接/服务器错误：可重试（transient）
    #   4. 可重试状态码：可重试（transient）
    #   5. 服务器繁忙模式：可重试（busy）
    #   6. 其他：通用错误，不重试
    def _classify_error(self, exc: BaseException) -> tuple[bool, str]:
        # 提取错误详情（用于模式匹配）
        # _extract_error_detail 返回字符串形式的错误信息
        detail = _extract_error_detail(exc)

        # 转换为小写，用于不区分大小写的模式匹配
        lowered = detail.lower()

        # 提取错误码（如果存在）
        error_code = _extract_error_code(exc)

        # 提取 HTTP 状态码（如果存在）
        status_code = _extract_status_code(exc)

        # ---- 检查是否是配额错误 ----
        # 匹配降低的错误详情，或者错误码本身
        if _matches_any(lowered, _QUOTA_PATTERNS) or _matches_any(str(error_code).lower(), _QUOTA_PATTERNS):
            return False, "quota"  # 不可重试，是配额问题

        # ---- 检查是否是认证错误 ----
        if _matches_any(lowered, _AUTH_PATTERNS):
            return False, "auth"    # 不可重试，是认证问题

        # 获取异常类名，用于精确匹配某些已知异常类型
        exc_name = exc.__class__.__name__

        # ---- 检查是否是已知的可重试异常类型 ----
        if exc_name in {
            "APITimeoutError",      # API 超时错误
            "APIConnectionError",   # API 连接错误
            "InternalServerError",  # 内部服务器错误
        }:
            return True, "transient"  # 可重试，临时性问题

        # ---- 检查状态码是否在可重试列表中 ----
        if status_code in _RETRIABLE_STATUS_CODES:
            return True, "transient"  # 可重试，HTTP 层面临时性错误

        # ---- 检查是否是服务器繁忙模式 ----
        if _matches_any(lowered, _BUSY_PATTERNS):
            return True, "busy"  # 可重试，服务器繁忙

        # ---- 所有其他错误 ----
        return False, "generic"  # 不可重试，通用错误

    # _build_retry_delay_ms：计算重试延迟时间
    #
    # 方法作用：
    #   根据当前尝试次数和异常信息，计算需要等待的时间。
    #
    # 参数：
    #   attempt: int，当前尝试次数（从 1 开始）
    #   exc: BaseException，触发的异常（可能包含 Retry-After 信息）
    #
    # 返回值：
    #   int：延迟时间（毫秒）
    #
    # 延迟计算策略：
    #   1. 如果异常中包含 Retry-After 信息，使用服务器指定的时间
    #   2. 否则使用指数退避：base_delay * 2^(attempt-1)，上限为 cap_delay
    #   例如：attempt=1 → 1000ms, attempt=2 → 2000ms, attempt=3 → 4000ms
    def _build_retry_delay_ms(self, attempt: int, exc: BaseException) -> int:
        # 首先尝试从异常中提取 Retry-After 信息
        retry_after = _extract_retry_after_ms(exc)
        if retry_after is not None:
            return retry_after  # 使用服务器指定的时间

        # 计算指数退避延迟
        # base_delay * 2^(attempt-1)
        # max(0, attempt - 1) 确保指数不会变成负数（虽然不会发生）
        backoff = self.retry_base_delay_ms * (2 ** max(0, attempt - 1))

        # 返回退避延迟和最大上限中的较小值
        return min(backoff, self.retry_cap_delay_ms)

    # _build_retry_message：构建重试消息字符串
    #
    # 方法作用：
    #   格式化一个用于日志和事件的字符串，描述当前重试状态。
    #
    # 参数：
    #   attempt: int，当前尝试次数
    #   wait_ms: int，等待时间（毫秒）
    #   reason: str，错误原因（"busy" 或其他）
    #
    # 返回值：
    #   str：格式化的重试消息
    #
    # 消息格式："LLM request retry 1/3: provider is busy. Retrying in 2s."
    def _build_retry_message(self, attempt: int, wait_ms: int, reason: str) -> str:
        # 将毫秒转换为秒数，最小为 1 秒（避免显示 0 秒）
        seconds = max(1, round(wait_ms / 1000))

        # 如果是服务器繁忙，原因文本为"provider is busy"，否则为"provider request failed temporarily"
        reason_text = "provider is busy" if reason == "busy" else "provider request failed temporarily"

        # 格式化完整消息
        return f"LLM request retry {attempt}/{self.retry_max_attempts}: {reason_text}. Retrying in {seconds}s."

    # _build_user_message：构建面向用户的降级消息
    #
    # 方法作用：
    #   当重试耗尽或遇到不可重试的错误时，将异常转换为用户友好的错误消息。
    #
    # 参数：
    #   exc: BaseException，触发的异常
    #   reason: str，错误原因分类
    #
    # 返回值：
    #   str：用户友好的错误消息，用于替换原始异常作为 AI 响应
    #
    # 设计考虑：
    #   - 不暴露技术细节（如 API 密钥错误、具体异常信息）
    #   - 提供清晰的后续操作指导
    #   - 区分不同错误类型，给出针对性建议
    def _build_user_message(self, exc: BaseException, reason: str) -> str:
        # 提取错误详情
        detail = _extract_error_detail(exc)

        # 根据错误原因返回对应的友好消息
        if reason == "quota":
            # 配额问题：建议用户检查账户和计费
            return "The configured LLM provider rejected the request because the account is out of quota, billing is unavailable, or usage is restricted. Please fix the provider account and try again."
        if reason == "auth":
            # 认证问题：建议用户检查凭据配置
            return "The configured LLM provider rejected the request because authentication or access is invalid. Please check the provider credentials and try again."
        if reason in {"busy", "transient"}:
            # 服务器繁忙或临时性错误：建议用户稍后重试
            return "The configured LLM provider is temporarily unavailable after multiple retries. Please wait a moment and continue the conversation."
        # 通用错误：返回错误详情的简化版本
        return f"LLM request failed: {detail}"

    # _emit_retry_event：发送重试事件到前端
    #
    # 方法作用：
    #   通过 StreamWriter 发送 llm_retry 类型的自定义事件，
    #   前端可以监听此事件显示"正在重试..."的用户提示。
    #
    # 参数：
    #   attempt: int，当前尝试次数
    #   wait_ms: int，等待时间（毫秒）
    #   reason: str，错误原因
    #
    # 注意：
    #   如果发送失败，只记录 debug 日志，不影响主流程
    def _emit_retry_event(self, attempt: int, wait_ms: int, reason: str) -> None:
        try:
            # 从 langgraph.config 导入 get_stream_writer，用于获取流式写入器
            from langgraph.config import get_stream_writer

            # 获取流式写入器
            writer = get_stream_writer()

            # 发送 llm_retry 事件，包含重试信息
            writer(
                {
                    "type": "llm_retry",                    # 事件类型：LLM 重试
                    "attempt": attempt,                     # 当前尝试次数
                    "max_attempts": self.retry_max_attempts,  # 最大尝试次数
                    "wait_ms": wait_ms,                    # 等待时间（毫秒）
                    "reason": reason,                      # 重试原因
                    "message": self._build_retry_message(attempt, wait_ms, reason),  # 格式化消息
                }
            )
        except Exception:
            # 发送失败不影响主流程，只记录 debug 日志
            logger.debug("Failed to emit llm_retry event", exc_info=True)

    # ============================================================
    # LangChain AgentMiddleware 钩子方法
    # ============================================================

    # wrap_model_call：同步版本的模型调用包装钩子
    #
    # 方法作用：
    #   LangChain AgentMiddleware 提供的扩展点，
    #   在模型调用前后添加重试逻辑。
    #   包装 handler(request)，捕获异常并进行重试。
    #
    # 参数：
    #   request: ModelRequest，模型调用请求
    #   handler: Callable[[ModelRequest], ModelResponse]，原始模型调用处理器
    #
    # 返回值：
    #   ModelCallResult：模型调用结果（成功时是 handler 的结果，失败时是降级的 AIMessage）
    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        attempt = 1  # 当前尝试次数，从 1 开始

        while True:  # 无限循环，通过 continue 或 return 退出
            try:
                # 调用原始 handler，执行实际的模型调用
                return handler(request)

            except GraphBubbleUp:
                # 捕获 LangGraph 控制流信号（interrupt/pause/resume）
                # 这些信号必须向上传播，不能被重试逻辑拦截
                raise

            except Exception as exc:
                # 捕获其他所有异常（API 错误、超时等）

                # 对错误进行分类：retriable=是否可重试，reason=错误原因
                retriable, reason = self._classify_error(exc)

                # 如果可重试且还有尝试次数
                if retriable and attempt < self.retry_max_attempts:
                    # 计算延迟时间
                    wait_ms = self._build_retry_delay_ms(attempt, exc)

                    # 记录警告日志
                    logger.warning(
                        "Transient LLM error on attempt %d/%d; retrying in %dms: %s",
                        attempt,
                        self.retry_max_attempts,
                        wait_ms,
                        _extract_error_detail(exc),
                    )

                    # 发送重试事件到前端
                    self._emit_retry_event(attempt, wait_ms, reason)

                    # 同步等待（使用 time.sleep 而不是 asyncio.sleep）
                    time.sleep(wait_ms / 1000)

                    attempt += 1  # 增加尝试次数
                    continue     # 继续下一次循环

                # 不可重试或重试次数已耗尽
                logger.warning(
                    "LLM call failed after %d attempt(s): %s",
                    attempt,
                    _extract_error_detail(exc),
                    exc_info=exc,
                )

                # 返回降级的 AIMessage，包含用户友好的错误消息
                return AIMessage(content=self._build_user_message(exc, reason))

    # awrap_model_call：异步版本的模型调用包装钩子
    #
    # 方法作用：
    #   与 wrap_model_call 相同，但支持异步 handler。
    #   使用 await asyncio.sleep 实现异步等待。
    #
    # 参数：
    #   request: ModelRequest，模型调用请求
    #   handler: Callable[[ModelRequest], Awaitable[ModelResponse]]，异步模型调用处理器
    #
    # 返回值：
    #   ModelCallResult：模型调用结果
    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        attempt = 1  # 当前尝试次数

        while True:
            try:
                # 异步调用原始 handler
                return await handler(request)

            except GraphBubbleUp:
                # 正确传播 LangGraph 控制流信号
                raise

            except Exception as exc:
                # 分类错误
                retriable, reason = self._classify_error(exc)

                # 如果可重试且还有尝试次数
                if retriable and attempt < self.retry_max_attempts:
                    # 计算延迟时间
                    wait_ms = self._build_retry_delay_ms(attempt, exc)

                    # 记录警告日志
                    logger.warning(
                        "Transient LLM error on attempt %d/%d; retrying in %dms: %s",
                        attempt,
                        self.retry_max_attempts,
                        wait_ms,
                        _extract_error_detail(exc),
                    )

                    # 发送重试事件到前端
                    self._emit_retry_event(attempt, wait_ms, reason)

                    # 异步等待（这里是同步和异步版本的主要区别）
                    await asyncio.sleep(wait_ms / 1000)

                    attempt += 1
                    continue

                # 重试耗尽，返回降级消息
                logger.warning(
                    "LLM call failed after %d attempt(s): %s",
                    attempt,
                    _extract_error_detail(exc),
                    exc_info=exc,
                )

                return AIMessage(content=self._build_user_message(exc, reason))


# ============================================================
# 模块级辅助函数
# ============================================================

# _matches_any：检查错误详情是否匹配任意一个模式
#
# 方法作用：
#   在错误详情中搜索预定义的模式列表，用于快速判断错误类型。
#
# 参数：
#   detail: str，降低的错误详情（用于不区分大小写的匹配）
#   patterns: tuple[str, ...]，要匹配的模式元组
#
# 返回值：
#   bool：如果详情中包含任意一个模式则返回 True
def _matches_any(detail: str, patterns: tuple[str, ...]) -> bool:
    # 使用 any() 检查是否有任意一个模式出现在详情中
    return any(pattern in detail for pattern in patterns)


# _extract_error_code：从异常中提取错误码
#
# 方法作用：
#   从异常对象中提取错误码，用于分类和诊断。
#
# 参数：
#   exc: BaseException，异常对象
#
# 返回值：
#   Any：错误码的值，如果不存在则返回 None
#
# 提取策略：
#   1. 尝试从异常属性获取 code 或 error_code
#   2. 尝试从异常的 body（通常是响应体）中获取 error.code 或 error.type
#   3. 如果都找不到，返回 None
def _extract_error_code(exc: BaseException) -> Any:
    # 首先尝试从异常属性获取
    for attr in ("code", "error_code"):
        value = getattr(exc, attr, None)
        # 排除 None 和空字符串
        if value not in (None, ""):
            return value

    # 尝试从 body 中获取（一些 provider 将错误信息放在 body 中）
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        # body 中可能有 error 字段
        error = body.get("error")
        if isinstance(error, dict):
            # error 中可能有 code 或 type
            for key in ("code", "type"):
                value = error.get(key)
                if value not in (None, ""):
                    return value

    # 找不到任何错误码
    return None


# _extract_status_code：从异常中提取 HTTP 状态码
#
# 方法作用：
#   从异常对象中提取 HTTP 状态码，用于判断是否可重试。
#
# 参数：
#   exc: BaseException，异常对象
#
# 返回值：
#   int | None：HTTP 状态码，如果不存在则返回 None
#
# 提取策略：
#   1. 尝试从异常属性获取 status_code 或 status
#   2. 尝试从异常的 response 属性中获取 status_code
#   3. 只返回整数类型的状态码，其他类型忽略
def _extract_status_code(exc: BaseException) -> int | None:
    # 首先尝试从异常属性获取
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        # 只接受整数类型
        if isinstance(value, int):
            return value

    # 尝试从异常的 response 属性中获取
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)

    # 确保返回的是整数类型
    return status if isinstance(status, int) else None


# _extract_retry_after_ms：从异常中提取 Retry-After 时间
#
# 方法作用：
#   从 HTTP 异常中提取 Retry-After 头的信息，用于精确控制重试等待时间。
#
# 参数：
#   exc: BaseException，异常对象
#
# 返回值：
#   int | None：Retry-After 时间（毫秒），如果不存在则返回 None
#
# 提取策略：
#   1. 从异常的 response.headers 中查找 Retry-After 相关头
#   2. 支持多种格式：
#      - Retry-After-MS: 毫秒数（如 "500"）
#      - Retry-After: 秒数（如 "30"）或 HTTP 日期
#   3. 如果是毫秒后缀，直接乘以 1；如果是秒数，乘以 1000
#   4. 如果是 HTTP 日期，计算距离现在的时间差
#   5. 所有计算结果最小为 0
def _extract_retry_after_ms(exc: BaseException) -> int | None:
    # 获取异常的 response 对象
    response = getattr(exc, "response", None)

    # 获取响应头
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    # 尝试查找 Retry-After 相关头
    raw = None
    header_name = ""

    # 支持多种大小写变体（不同的 HTTP 库可能使用不同的大小写）
    for key in ("retry-after-ms", "Retry-After-Ms", "retry-after", "Retry-After"):
        header_name = key
        if hasattr(headers, "get"):
            raw = headers.get(key)
        if raw:
            break  # 找到就停止搜索

    if not raw:
        return None  # 没有找到 Retry-After 头

    # 解析 Retry-After 值
    try:
        # 确定是毫秒还是秒数（毫秒后缀用 ms）
        multiplier = 1 if "ms" in header_name.lower() else 1000

        # 转换为浮点数再乘以倍数
        return max(0, int(float(raw) * multiplier))
    except (TypeError, ValueError):
        # 不是数字格式，尝试解析为 HTTP 日期
        try:
            # parsedate_to_datetime 解析 HTTP 日期字符串为 datetime
            target = parsedate_to_datetime(str(raw))

            # 计算目标时间距离现在的时间差（秒）
            delta = target.timestamp() - time.time()

            # 转换为毫秒，最小为 0
            return max(0, int(delta * 1000))
        except (TypeError, ValueError, OverflowError):
            # 解析失败
            return None


# _extract_error_detail：从异常中提取错误详情
#
# 方法作用：
#   从异常对象中提取人类可读的错误描述，用于日志和用户消息。
#
# 参数：
#   exc: BaseException，异常对象
#
# 返回值：
#   str：格式化的错误详情字符串
#
# 提取策略（按优先级）：
#   1. 尝试 str(exc).strip()，去除首尾空白
#   2. 如果为空，尝试 getattr(exc, "message", None).strip()
#   3. 如果都为空，返回异常类名作为最后回退
#
# 设计考虑：
#   不同 provider 的异常格式差异很大：
#   - 有些用 str(exc) 包含完整消息
#   - 有些把消息放在单独的 message 属性中
#   - 有些只有异常类名（如某些内部错误）
def _extract_error_detail(exc: BaseException) -> str:
    # 首先尝试 str(exc)
    detail = str(exc).strip()
    if detail:
        return detail

    # 尝试从 message 属性获取
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()

    # 最后回退：使用异常类名
    return exc.__class__.__name__
