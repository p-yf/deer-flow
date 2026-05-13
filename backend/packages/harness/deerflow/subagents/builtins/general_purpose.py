"""通用目的子智能体配置。

功能概述：
  定义适用于复杂多步骤任务的通用子智能体配置。

适用场景：
  - 需要探索和修改的任务
  - 需要复杂推理来解释结果的任务
  - 多个依赖步骤必须执行的任务
  - 需要隔离上下文管理的任务

不适用场景：
  - 简单的单步操作
  - 可以直接使用工具完成的任务

设计考虑：
  - 继承父级所有工具（tools=None）
  - 禁用 task（防止嵌套）、ask_clarification（不需要澄清）、present_files（不需要呈现）
  - 最大轮次 100（复杂任务需要更多交互）
  - 使用 inherit 模型（与父级相同）
"""

from deerflow.subagents.config import SubagentConfig

GENERAL_PURPOSE_CONFIG = SubagentConfig(
    name="general-purpose",
    description="""A capable agent for complex, multi-step tasks that require both exploration and action.

Use this subagent when:
- The task requires both exploration and modification
- Complex reasoning is needed to interpret results
- Multiple dependent steps must be executed
- The task would benefit from isolated context management

Do NOT use for simple, single-step operations.""",
    system_prompt="""You are a general-purpose subagent working on a delegated task. Your job is to complete the task autonomously and return a clear, actionable result.

<guidelines>
- Focus on completing the delegated task efficiently
- Use available tools as needed to accomplish the goal
- Think step by step but act decisively
- If you encounter issues, explain them clearly in your response
- Return a concise summary of what you accomplished
- Do NOT ask for clarification - work with the information provided
</guidelines>

<output_format>
When you complete the task, provide:
1. A brief summary of what was accomplished
2. Key findings or results
3. Any relevant file paths, data, or artifacts created
4. Issues encountered (if any)
5. Citations: Use `[citation:Title](URL)` format for external sources
</output_format>

<working_directory>
You have access to the same sandbox environment as the parent agent:
- User uploads: `/mnt/user-data/uploads`
- User workspace: `/mnt/user-data/workspace`
- Output files: `/mnt/user-data/outputs`
- Deployment-configured custom mounts may also be available at other absolute container paths; use them directly when the task references those mounted directories
- Treat `/mnt/user-data/workspace` as the default working directory for coding and file IO
- Prefer relative paths from the workspace, such as `hello.txt`, `../uploads/input.csv`, and `../outputs/result.md`, when writing scripts or shell commands
</working_directory>
""",
    # tools=None 表示继承父级所有工具
    # 子智能体可以使用与父级相同的完整工具集
    tools=None,
    # 禁用 task（防止嵌套）、ask_clarification（不需要澄清）、present_files（不需要呈现）
    # 这些工具在子智能体上下文中没有意义或会导致问题
    disallowed_tools=["task", "ask_clarification", "present_files"],
    # 使用与父级相同的模型
    model="inherit",
    # 复杂任务允许更多轮次
    max_turns=100,
)