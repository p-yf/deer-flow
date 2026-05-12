"""Configuration for automatic thread title generation."""

# ============================================================
# 导入 Pydantic 相关模块
# ============================================================

# pydantic.BaseModel：Pydantic 基类，用于数据验证和配置管理
from pydantic import BaseModel, Field


# ============================================================
# 配置类定义
# ============================================================

# TitleConfig：自动线程标题生成配置类
#
# 作用说明：
#   定义自动标题生成的各种配置参数，如启用开关、最大词数/字符数、模型选择、提示模板等。
#   使用 Pydantic 进行数据验证。
#
# 字段说明：
#   - enabled：是否启用自动标题生成
#   - max_words：生成标题的最大词数
#   - max_chars：生成标题的最大字符数
#   - model_name：用于标题生成的模型名称
#   - prompt_template：生成标题的提示模板
class TitleConfig(BaseModel):
    """Configuration for automatic thread title generation."""

    # enabled：是否启用自动标题生成
    enabled: bool = Field(
        default=True,
        description="Whether to enable automatic title generation",
    )

    # max_words：生成标题的最大词数
    max_words: int = Field(
        default=6,
        ge=1,  # 最小值 1
        le=20,  # 最大值 20
        description="Maximum number of words in the generated title",
    )

    # max_chars：生成标题的最大字符数
    max_chars: int = Field(
        default=60,
        ge=10,  # 最小值 10
        le=200,  # 最大值 200
        description="Maximum number of characters in the generated title",
    )

    # model_name：用于标题生成的模型名称
    # 如果为 None，则使用默认模型
    model_name: str | None = Field(
        default=None,
        description="Model name to use for title generation (None = use default model)",
    )

    # prompt_template：生成标题的提示模板
    # 模板中使用 {max_words}、{user_msg}、{assistant_msg} 作为占位符
    prompt_template: str = Field(
        default=(
            "Generate a concise title (max {max_words} words) for this conversation.\n"
            "User: {user_msg}\n"
            "Assistant: {assistant_msg}\n"
            "\n"
            "Return ONLY the title, no quotes, no explanation."
        ),
        description="Prompt template for title generation",
    )


# ============================================================
# 模块级变量初始化
# ============================================================

# _title_config：模块级全局配置实例
# 初始化为默认配置的 TitleConfig 实例
_title_config: TitleConfig = TitleConfig()


# ============================================================
# 配置访问函数
# ============================================================

# get_title_config：获取当前标题配置
#
# 作用说明：
#   返回全局的 TitleConfig 实例。
#
# 调用位置：
#   TitleMiddleware._agenerate_title_result() 调用此函数
#   来源文件：deerflow/agents/middlewares/title_middleware.py
#
# 返回值：
#   TitleConfig：当前的标题配置实例
def get_title_config() -> TitleConfig:
    """Get the current title configuration."""
    return _title_config


# set_title_config：设置标题配置
#
# 参数：
#   config: TitleConfig，新的配置实例
#
# 作用说明：
#   将全局的 _title_config 设置为传入的配置对象
def set_title_config(config: TitleConfig) -> None:
    """Set the title configuration."""
    global _title_config
    _title_config = config


# load_title_config_from_dict：从字典加载配置
#
# 参数：
#   config_dict: dict，配置字典
#
# 作用说明：
#   使用 Pydantic 的 **解包语法，将字典转换为 TitleConfig 实例
#   并设置全局配置
def load_title_config_from_dict(config_dict: dict) -> None:
    """Load title configuration from a dictionary."""
    global _title_config
    _title_config = TitleConfig(**config_dict)
