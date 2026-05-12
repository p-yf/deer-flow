"""SandboxAuditMiddleware - Bash 命令安全审计中间件。

功能概述：
  对所有 bash 工具调用进行安全审计，防止危险命令执行。

命令分类：
  - 高风险（block）：阻止执行，返回错误 ToolMessage
    * rm -rf /（递归删除）
    * curl url | bash（管道到 shell）
    * dd if=（磁盘写入）
    * mkfs（格式化文件系统）
    * 读取 /etc/shadow（密码文件）
    * LD_PRELOAD/LD_LIBRARY_PATH（动态链接器劫持）
    * Fork 炸弹（:(){ :|:& };:）
    * /dev/tcp/（bash 内置网络）
  - 中风险（warn）：正常执行，但追加警告
    * chmod 777
    * pip install / apt install
    * sudo/su
    * PATH 修改
  - 安全（pass）：正常执行

检测策略：
  1. 整体高风险扫描：正则匹配跨越多个语句的攻击（如 Fork 炸弹）
  2. 分割复合命令：按 ;、&&、|| 分割
  3. 逐个子命令分类，最坏结果胜出
  4. 使用 shlex 辅助解析，排除引号内干扰

输入验证：
  - 空命令拒绝
  - 超长命令拒绝（> 10000 字符）
  - 空字节拒绝

审计日志：
  - 每次调用记录结构化 JSON（timestamp、thread_id、command、verdict）
  - 通过标准 logger 记录（可见于 langgraph.log）

执行位置：在 LLMErrorHandlingMiddleware 之后。
"""
"""Bash command security auditing middleware."""

# ============================================================
# 导入标准库
# ============================================================

# json：标准库 JSON 模块，用于序列化为 JSON 格式的审计日志
import json

# logging：标准库日志模块，用于记录审计日志
import logging

# re：标准库正则表达式模块，用于编译和匹配高风险/中风险命令模式
import re

# shlex：标准库 shell 词法分析模块，用于安全地解析 shell 命令 token
import shlex

# collections.abc 导入：
#   - Awaitable：异步可等待对象类型，用于异步 handler 的返回类型注解
#   - Callable：可调用对象类型，用于 handler 参数的类型注解
from collections.abc import Awaitable, Callable

# datetime 导入：
#   - UTC：协调世界时，用于生成 ISO 格式的时间戳
#   - datetime：日期时间类，用于生成审计日志的时间戳
from datetime import UTC, datetime

# typing 导入：
#   - override：方法重写标记，用于明确表示重写父类方法
from typing import override

# ============================================================
# 导入 LangChain / LangGraph 相关模块
# ============================================================

# langchain.agents.middleware.AgentMiddleware：
#   LangChain 的中间件基类，所有自定义中间件必须继承此类
from langchain.agents.middleware import AgentMiddleware

# langchain_core.messages.ToolMessage：
#   工具消息类型，用于构建阻止消息和追加警告
from langchain_core.messages import ToolMessage

# langgraph.prebuilt.tool_node.ToolCallRequest：
#   工具调用的请求对象，包含 tool_call（工具名称、参数、ID 等）
from langgraph.prebuilt.tool_node import ToolCallRequest

# langgraph.types.Command：
#   LangGraph 的命令类型，用于控制图执行流程
from langgraph.types import Command

# ============================================================
# 导入 DeerFlow 项目内部模块
# ============================================================

# deerflow.agents.thread_state.ThreadState：
#   线程状态类型，用于中间件的状态类型定义
#   来自：本项目 packages/harness/deerflow/agents/thread_state.py
from deerflow.agents.thread_state import ThreadState

# ============================================================
# 模块级变量初始化
# ============================================================

# 创建模块级 logger，用于记录审计日志
logger = logging.getLogger(__name__)

# ============================================================
# 命令分类规则（正则表达式）
# ============================================================

# 高风险模式列表：匹配的命令会被直接阻止（block）
# 每个模式在导入时编译一次，避免重复编译开销
_HIGH_RISK_PATTERNS: list[re.Pattern[str]] = [
    # rm -rf 递归删除系统关键目录
    re.compile(r"rm\s+-[^\s]*r[^\s]*\s+(/\*?|~/?\*?|/home\b|/root\b)\s*$"),
    # dd 命令（常用于磁盘写入，可能用于破坏数据）
    re.compile(r"dd\s+if="),
    # mkfs 命令（格式化文件系统）
    re.compile(r"mkfs"),
    # 读取 /etc/shadow（密码文件）
    re.compile(r"cat\s+/etc/shadow"),
    # 覆盖 /etc/ 下的文件（可能用于修改系统配置）
    re.compile(r">+\s*/etc/"),
    # 管道到 sh/bash（通用化，任何命令输出通过管道传给 shell 执行）
    re.compile(r"\|\s*(ba)?sh\b"),
    # 命令替换（仅针对危险的可执行文件）
    re.compile(r"[`$]\(?\s*(curl|wget|bash|sh|python|ruby|perl|base64)"),
    # base64 解码后执行
    re.compile(r"base64\s+.*-d.*\|"),
    # 覆盖系统二进制文件
    re.compile(r">+\s*(/usr/bin/|/bin/|/sbin/)"),
    # 覆盖 shell 启动文件
    re.compile(r">+\s*~/?\.(bashrc|profile|zshrc|bash_profile)"),
    # 读取 /proc 环境变量
    re.compile(r"/proc/[^/]+/environ"),
    # 动态链接器劫持（单步提权）
    re.compile(r"\b(LD_PRELOAD|LD_LIBRARY_PATH)\s*="),
    # bash 内置网络（绕过工具白名单）
    re.compile(r"/dev/tcp/"),
    # Fork 炸弹（两种形式）
    re.compile(r"\S+\(\)\s*\{[^}]*\|\s*\S+\s*&"),  # :(){ :|:& };:
    re.compile(r"while\s+true.*&\s*done"),  # while true; do bash & done
]

# 中风险模式列表：匹配的命令会执行但追加警告（warn）
_MEDIUM_RISK_PATTERNS: list[re.Pattern[str]] = [
    # chmod 777（过于宽松的权限）
    re.compile(r"chmod\s+777"),
    # pip 安装包
    re.compile(r"pip3?\s+install"),
    # apt 安装包
    re.compile(r"apt(-get)?\s+install"),
    # sudo/su：在 Docker root 下无操作，但警告让 LLM 知道
    re.compile(r"\b(sudo|su)\b"),
    # PATH 修改：需要较长的攻击链，警告而非阻止
    re.compile(r"\bPATH\s*="),
]


# ============================================================
# 模块级辅助函数
# ============================================================

# _split_compound_command：分割复合命令（考虑引号）
#
# 方法作用：
#   将复合命令（如 "cmd1 && cmd2 ; cmd3"）分割成子命令列表。
#   识别引号（单引号、双引号）和转义字符，在未引用的控制操作符处分割。
#
# 参数：
#   command: str，原始命令字符串
#
# 返回值：
#   list[str]：分割后的子命令列表
#
# 工作原理：
#   1. 逐字符扫描命令字符串
#   2. 识别引号和转义字符
#   3. 在未引用的控制操作符（;、&&、||）处分割
#   4. 引号内的操作符不参与分割
#
# 边界情况：
#   如果命令以未闭合的引号或悬空转义符结束，采用"失败关闭"策略，
#   返回整个命令而不分割（避免部分命令执行导致安全问题）
def _split_compound_command(command: str) -> list[str]:
    """Split a compound command into sub-commands (quote-aware)."""
    parts: list[str] = []      # 存储分割后的子命令
    current: list[str] = []    # 当前正在处理的字符缓冲区
    in_single_quote = False    # 是否在单引号内
    in_double_quote = False    # 是否在双引号内
    escaping = False           # 是否处于转义状态（遇到反斜杠后）
    index = 0                  # 当前扫描位置

    while index < len(command):
        char = command[index]

        # 如果处于转义状态，当前字符被当作字面量处理
        if escaping:
            current.append(char)
            escaping = False
            index += 1
            continue

        # 处理反斜杠转义
        if char == "\\" and not in_single_quote:
            current.append(char)
            escaping = True
            index += 1
            continue

        # 处理单引号（双引号内不处理）
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current.append(char)
            index += 1
            continue

        # 处理双引号（单引号内不处理）
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current.append(char)
            index += 1
            continue

        # 只有在未引用状态下才识别控制操作符
        if not in_single_quote and not in_double_quote:
            # 检查 && 或 ||
            if command.startswith("&&", index) or command.startswith("||", index):
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                index += 2  # 跳过操作符
                continue
            # 检查分号
            if char == ";":
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                index += 1
                continue

        # 普通字符，添加到当前缓冲区
        current.append(char)
        index += 1

    # 检查未闭合的引号或悬空转义符
    if in_single_quote or in_double_quote or escaping:
        return [command]  # 失败关闭：返回整个命令

    # 处理最后一个缓冲区中的内容
    part = "".join(current).strip()
    if part:
        parts.append(part)

    return parts if parts else [command]


# _classify_single_command：对单个（非复合）命令进行分类
#
# 方法作用：
#   对一个单独的命令进行风险分类。
#
# 参数：
#   command: str，单个命令字符串
#
# 返回值：
#   str：分类结果，可能是 "block"（阻止）/ "warn"（警告）/ "pass"（通过）
def _classify_single_command(command: str) -> str:
    """Classify a single (non-compound) command. Return 'block', 'warn', or 'pass'."""
    # 规范化：合并多个空白字符为单个空格
    normalized = " ".join(command.split())

    # 首先检查高风险模式（直接匹配）
    for pattern in _HIGH_RISK_PATTERNS:
        if pattern.search(normalized):
            return "block"

    # 也要尝试用 shlex 解析后的 token 进行高风险检测
    try:
        tokens = shlex.split(command)
        joined = " ".join(tokens)
        for pattern in _HIGH_RISK_PATTERNS:
            if pattern.search(joined):
                return "block"
    except ValueError:
        # shlex.split 在未闭合引号时失败 — 当作可疑命令处理
        return "block"

    # 检查中风险模式
    for pattern in _MEDIUM_RISK_PATTERNS:
        if pattern.search(normalized):
            return "warn"

    # 所有检查通过，安全命令
    return "pass"


# _classify_command：对完整命令进行分类（处理复合命令）
#
# 方法作用：
#   对一个完整命令（可能是复合命令）进行风险分类。
#
# 参数：
#   command: str，完整命令字符串（可能是复合命令）
#
# 返回值：
#   str：分类结果，可能是 "block"（阻止）/ "warn"（警告）/ "pass"（通过）
#
# 工作原理（两遍策略）：
#   1. 第一遍：整体高风险扫描（捕获跨越多个语句的结构性攻击）
#   2. 第二遍：分割后逐个子命令分类，取最坏结果
def _classify_command(command: str) -> str:
    """Return 'block', 'warn', or 'pass'."""
    # 第一遍：整体高风险扫描
    normalized = " ".join(command.split())
    for pattern in _HIGH_RISK_PATTERNS:
        if pattern.search(normalized):
            return "block"

    # 第二遍：按子命令分类
    sub_commands = _split_compound_command(command)
    worst = "pass"
    for sub in sub_commands:
        verdict = _classify_single_command(sub)
        if verdict == "block":
            return "block"  # 短路：block 是最坏结果
        if verdict == "warn":
            worst = "warn"   # 记录较坏的结果
    return worst


# ============================================================
# SandboxAuditMiddleware 主类
# ============================================================

# SandboxAuditMiddleware 类：bash 命令安全审计中间件
#
# 核心作用：
#   对所有 bash 工具调用进行安全审计，防止危险命令执行。
#
# 工作流程：
#   1. 在 wrap_tool_call 中拦截 bash 工具调用
#   2. 调用 _pre_process 进行输入验证和命令分类
#   3. 如果是高风险命令，返回阻止消息（不调用 handler）
#   4. 如果是中风险命令，调用 handler 后在结果中追加警告
#   5. 如果是安全命令，正常执行
#
# 设计考虑：
#   - 高风险命令：阻止执行，返回错误消息，让 agent 可以优雅地继续
#   - 中风险命令：执行但警告，让 LLM 知道这个命令可能有风险
#   - 所有命令都记录审计日志，可用于安全审计
class SandboxAuditMiddleware(AgentMiddleware[ThreadState]):
    """Bash command security auditing middleware."""

    # state_schema：类变量，指定该中间件使用的状态类型
    state_schema = ThreadState

    # _AUDIT_COMMAND_LIMIT：审计命令的最大长度限制（类属性）
    _AUDIT_COMMAND_LIMIT = 200

    # _MAX_COMMAND_LENGTH：最大命令长度限制
    # 任何更长的命令几乎肯定是 payload 注入或 base64 编码的攻击字符串
    _MAX_COMMAND_LENGTH = 10_000

    # ============================================================
    # 内部辅助方法
    # ============================================================

    # _get_thread_id：从 ToolCallRequest 中提取 thread_id
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求
    #
    # 返回值：
    #   str | None：thread_id 字符串，如果找不到则返回 None
    def _get_thread_id(self, request: ToolCallRequest) -> str | None:
        runtime = request.runtime  # ToolRuntime; may be None-like in tests
        if runtime is None:
            return None
        # 尝试从 runtime.context 获取
        ctx = getattr(runtime, "context", None) or {}
        thread_id = ctx.get("thread_id") if isinstance(ctx, dict) else None
        # 尝试从 runtime.config 获取
        if thread_id is None:
            cfg = getattr(runtime, "config", None) or {}
            thread_id = cfg.get("configurable", {}).get("thread_id")
        return thread_id

    # _write_audit：写入审计日志
    #
    # 方法作用：
    #   记录命令审计信息到日志。
    #
    # 参数：
    #   thread_id: str | None，线程 ID
    #   command: str，执行的命令
    #   verdict: str，判决结果（block/warn/pass）
    #   truncate: bool，是否截断命令（默认为 False）
    def _write_audit(self, thread_id: str | None, command: str, verdict: str, *, truncate: bool = False) -> None:
        # 如果需要截断且命令过长
        audited_command = command
        if truncate and len(command) > self._AUDIT_COMMAND_LIMIT:
            # 截断到限制长度，并标注原始长度
            audited_command = f"{command[: self._AUDIT_COMMAND_LIMIT]}... ({len(command)} chars)"

        # 构建审计记录（结构化 JSON）
        record = {
            "timestamp": datetime.now(UTC).isoformat(),  # ISO 格式时间戳
            "thread_id": thread_id or "unknown",          # 线程 ID
            "command": audited_command,                    # 执行的命令（可能截断）
            "verdict": verdict,                            # 判决结果
        }
        # 通过 logger 记录
        logger.info("[SandboxAudit] %s", json.dumps(record, ensure_ascii=False))

    # _build_block_message：构建阻止消息
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求
    #   reason: str，阻止原因
    #
    # 返回值：
    #   ToolMessage：错误消息
    def _build_block_message(self, request: ToolCallRequest, reason: str) -> ToolMessage:
        tool_call_id = str(request.tool_call.get("id") or "missing_id")
        return ToolMessage(
            content=f"Command blocked: {reason}. Please use a safer alternative approach.",
            tool_call_id=tool_call_id,
            name="bash",
            status="error",
        )

    # _append_warn_to_result：向工具结果追加警告
    #
    # 方法作用：
    #   向 ToolMessage 的内容追加警告消息。
    #
    # 参数：
    #   result: ToolMessage | Command，原始工具结果
    #   command: str，执行的命令
    #
    # 返回值：
    #   ToolMessage | Command：追加警告后的结果
    def _append_warn_to_result(self, result: ToolMessage | Command, command: str) -> ToolMessage | Command:
        """Append a warning note to the tool result for medium-risk commands."""
        # 只处理 ToolMessage 类型
        if not isinstance(result, ToolMessage):
            return result
        # 构建警告消息
        warning = f"\n\n⚠️ Warning: `{command}` is a medium-risk command that may modify the runtime environment."
        # 追加到内容的适当位置
        if isinstance(result.content, list):
            new_content = list(result.content) + [{"type": "text", "text": warning}]
        else:
            new_content = str(result.content) + warning
        return ToolMessage(
            content=new_content,
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
        )

    # _validate_input：验证输入命令是否可接受
    #
    # 参数：
    #   command: str，要验证的命令
    #
    # 返回值：
    #   str | None：如果不可接受，返回拒绝原因；如果可接受，返回 None
    def _validate_input(self, command: str) -> str | None:
        """Return ``None`` if *command* is acceptable, else a rejection reason."""
        # 检查空命令
        if not command.strip():
            return "empty command"
        # 检查命令长度
        if len(command) > self._MAX_COMMAND_LENGTH:
            return "command too long"
        # 检查空字节（常见于攻击 payload）
        if "\x00" in command:
            return "null byte detected"
        # 所有检查通过
        return None

    # _pre_process：预处理命令（验证 + 分类 + 审计）
    #
    # 方法作用：
    #   执行命令的验证、分类和审计。
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求
    #
    # 返回值：
    #   tuple[str, str | None, str, str | None]：
    #     - command: 经过验证的命令
    #     - thread_id: 线程 ID
    #     - verdict: 分类结果
    #     - reject_reason: 拒绝原因
    def _pre_process(self, request: ToolCallRequest) -> tuple[str, str | None, str, str | None]:
        """Returns (command, thread_id, verdict, reject_reason)."""
        # 从请求中提取命令参数
        args = request.tool_call.get("args", {})
        raw_command = args.get("command")
        # 确保命令是字符串类型
        command = raw_command if isinstance(raw_command, str) else ""
        # 获取线程 ID
        thread_id = self._get_thread_id(request)

        # ① 输入验证
        reject_reason = self._validate_input(command)
        if reject_reason:
            self._write_audit(thread_id, command, "block", truncate=True)
            logger.warning("[SandboxAudit] INVALID INPUT thread=%s reason=%s", thread_id, reject_reason)
            return command, thread_id, "block", reject_reason

        # ② 命令分类
        verdict = _classify_command(command)

        # ③ 审计日志
        self._write_audit(thread_id, command, verdict)

        # 记录阻止或警告日志
        if verdict == "block":
            logger.warning("[SandboxAudit] BLOCKED thread=%s cmd=%r", thread_id, command)
        elif verdict == "warn":
            logger.warning("[SandboxAudit] WARN (medium-risk) thread=%s cmd=%r", thread_id, command)

        return command, thread_id, verdict, None

    # ============================================================
    # LangChain AgentMiddleware 钩子方法
    # ============================================================

    # wrap_tool_call：同步版本的工具调用包装钩子
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求
    #   handler: Callable[[ToolCallRequest], ToolMessage | Command]，原始工具处理器
    #
    # 返回值：
    #   ToolMessage | Command：处理结果
    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        # 只处理 bash 工具调用，其他工具直接放行
        if request.tool_call.get("name") != "bash":
            return handler(request)

        # 预处理命令
        command, _, verdict, reject_reason = self._pre_process(request)

        # 如果是高风险命令或输入验证失败，返回阻止消息
        if verdict == "block":
            reason = reject_reason or "security violation detected"
            return self._build_block_message(request, reason)

        # 执行命令
        result = handler(request)

        # 如果是中风险命令，向结果追加警告
        if verdict == "warn":
            result = self._append_warn_to_result(result, command)
        return result

    # awrap_tool_call：异步版本的工具调用包装钩子
    #
    # 参数：
    #   request: ToolCallRequest，工具调用请求
    #   handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]，异步工具处理器
    #
    # 返回值：
    #   ToolMessage | Command：处理结果
    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        # 只处理 bash 工具调用
        if request.tool_call.get("name") != "bash":
            return await handler(request)

        # 预处理命令
        command, _, verdict, reject_reason = self._pre_process(request)

        # 高风险命令：返回阻止消息
        if verdict == "block":
            reason = reject_reason or "security violation detected"
            return self._build_block_message(request, reason)

        # 执行命令（异步）
        result = await handler(request)

        # 中风险命令：追加警告
        if verdict == "warn":
            result = self._append_warn_to_result(result, command)
        return result
