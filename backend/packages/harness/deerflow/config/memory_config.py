"""Configuration for memory mechanism."""

# ============================================================
# 导入 Pydantic 相关模块
# ============================================================

# pydantic.BaseModel：Pydantic 基类，用于数据验证和配置管理
from pydantic import BaseModel, Field


# ============================================================
# 配置类定义
# ============================================================

# MemoryConfig：记忆机制配置类
#
# 作用说明：
#   定义记忆系统的各种配置参数，如启用开关、存储路径、debounce 时间等。
#   使用 Pydantic 进行数据验证。
#
# 字段说明：
#   - enabled：是否启用记忆机制
#   - storage_path：记忆数据存储路径
#   - storage_class：记忆存储提供者类路径
#   - debounce_seconds：防抖等待秒数
#   - model_name：用于记忆更新的模型名称
#   - max_facts：最大存储事实数量
#   - fact_confidence_threshold：事实存储的最小置信度
#   - injection_enabled：是否启用记忆注入到系统提示
#   - max_injection_tokens：记忆注入的最大 token 数
class MemoryConfig(BaseModel):
    """Configuration for global memory mechanism."""

    # enabled：是否启用记忆机制
    # 默认值为 True
    enabled: bool = Field(
        default=True,
        description="Whether to enable memory mechanism",
    )

    # storage_path：记忆数据存储路径
    # 支持绝对路径和相对路径
    storage_path: str = Field(
        default="",
        description=(
            "Path to store memory data. "
            "If empty, defaults to `{base_dir}/memory.json` (see Paths.memory_file). "
            "Absolute paths are used as-is. "
            "Relative paths are resolved against `Paths.base_dir` "
            "(not the backend working directory). "
            "Note: if you previously set this to `.deer-flow/memory.json`, "
            "the file will now be resolved as `{base_dir}/.deer-flow/memory.json`; "
            "migrate existing data or use an absolute path to preserve the old location."
        ),
    )

    # storage_class：记忆存储提供者类路径
    # 用于指定使用哪种存储后端
    storage_class: str = Field(
        default="deerflow.agents.memory.storage.FileMemoryStorage",
        description="The class path for memory storage provider",
    )

    # debounce_seconds：防抖等待秒数
    # 在这段时间内的多次更新请求会被合并
    debounce_seconds: int = Field(
        default=30,
        ge=1,  # 最小值 1 秒
        le=300,  # 最大值 300 秒
        description="Seconds to wait before processing queued updates (debounce)",
    )

    # model_name：用于记忆更新的模型名称
    # 如果为 None，则使用默认模型
    model_name: str | None = Field(
        default=None,
        description="Model name to use for memory updates (None = use default model)",
    )

    # max_facts：最大存储事实数量
    # 超过此数量的最旧事实会被丢弃
    max_facts: int = Field(
        default=100,
        ge=10,  # 最小值 10
        le=500,  # 最大值 500
        description="Maximum number of facts to store",
    )

    # fact_confidence_threshold：事实存储的最小置信度
    # 置信度低于此值的事实不会被存储
    fact_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,  # 最小值 0.0
        le=1.0,  # 最大值 1.0
        description="Minimum confidence threshold for storing facts",
    )

    # injection_enabled：是否启用记忆注入到系统提示
    # 如果为 True，记忆内容会被注入到 <memory> 标签中
    injection_enabled: bool = Field(
        default=True,
        description="Whether to inject memory into system prompt",
    )

    # max_injection_tokens：记忆注入的最大 token 数
    # 超过此限制的记忆内容会被截断
    max_injection_tokens: int = Field(
        default=2000,
        ge=100,  # 最小值 100
        le=8000,  # 最大值 8000
        description="Maximum tokens to use for memory injection",
    )


# ============================================================
# 模块级变量初始化
# ============================================================

# _memory_config：模块级全局配置实例
# 初始化为默认配置的 MemoryConfig 实例
_memory_config: MemoryConfig = MemoryConfig()


# ============================================================
# 配置访问函数
# ============================================================

# get_memory_config：获取当前记忆配置
#
# 作用说明：
#   返回全局的 MemoryConfig 实例。
#
# 调用位置：
#   MemoryMiddleware.after_agent() 调用此函数
#   来源文件：deerflow/agents/middlewares/memory_middleware.py
#
# 返回值：
#   MemoryConfig：当前的记忆配置实例
def get_memory_config() -> MemoryConfig:
    """Get the current memory configuration."""
    return _memory_config


# set_memory_config：设置记忆配置
#
# 参数：
#   config: MemoryConfig，新的配置实例
#
# 作用说明：
#   将全局的 _memory_config 设置为传入的配置对象
def set_memory_config(config: MemoryConfig) -> None:
    """Set the memory configuration."""
    global _memory_config
    _memory_config = config


# load_memory_config_from_dict：从字典加载配置
#
# 参数：
#   config_dict: dict，配置字典
#
# 作用说明：
#   使用 Pydantic 的 **解包语法，将字典转换为 MemoryConfig 实例
#   并设置全局配置
def load_memory_config_from_dict(config_dict: dict) -> None:
    """Load memory configuration from a dictionary."""
    global _memory_config
    _memory_config = MemoryConfig(**config_dict)
