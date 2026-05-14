"""文件操作锁模块 - 为并发文件访问提供线程安全的锁机制。

本模块实现基于沙箱和路径的文件操作锁，用于：
1. 防止多个线程同时写入同一文件导致数据损坏
2. 维护沙箱隔离：不同沙箱实例不会相互干扰

核心概念：
  - 锁键 (LockKey)：(sandbox_id, path) 元组，唯一标识一个文件操作
  - WeakValueDictionary：当锁不再被任何线程引用时自动清理，防止内存泄漏
  - 线程锁：threading.Lock 用于保护锁字典的创建和访问

使用场景：
  - write_file 和 str_replace 工具使用此锁保证写入安全
  - 每个沙箱实例有独立的锁空间，不会相互阻塞

线程安全实现：
  - 使用 threading.Lock 保护锁字典的创建和访问
  - 每次获取锁时使用 with 语句确保自动释放
"""

# 导入 threading 模块，提供线程编程原语
# Lock 用于创建互斥锁，实现线程间的互斥访问
import threading

# 导入 weakref 模块，提供弱引用
# WeakValueDictionary 是一种特殊的字典，当值对象不再被其他对象引用时自动移除
import weakref

# 从同一包的 sandbox 模块导入 Sandbox 抽象基类
# get_file_operation_lock_key 函数需要 Sandbox 类型提示
from deerflow.sandbox.sandbox import Sandbox

# 类型别名：_LockKey 定义锁键的类型
# 锁键是 (sandbox_id, path) 元组
# sandbox_id 是字符串，path 是字符串
_LockKey = tuple[str, str]


# _FILE_OPERATION_LOCKS：文件操作锁的存储
#
# 使用 WeakValueDictionary 的原因：
#   当锁不再被任何线程引用时，自动从字典中移除，避免内存泄漏
#   这对于长时间运行的进程很重要，因为锁对象不会无限累积
#
# 键类型：_LockKey = tuple[str, str] = (sandbox_id, path)
# 值类型：threading.Lock
#
# 示例：
#   ("local", "/mnt/user-data/workspace/test.py") -> threading.Lock()
#
# WeakValueDictionary vs 普通字典：
#   - 普通字典：键值对永久保存，直到手动删除
#   - WeakValueDictionary：当值对象（这里是 Lock）不再被其他对象引用时，自动被垃圾回收
#   - 这样可以避免创建大量锁对象导致的内存泄漏
_FILE_OPERATION_LOCKS: weakref.WeakValueDictionary[_LockKey, threading.Lock] = weakref.WeakValueDictionary()


# _FILE_OPERATION_LOCKS_GUARD：保护锁字典的全局锁
#
# 因为 WeakValueDictionary 的操作不是线程安全的，
# 需要一个全局锁来保护创建和访问锁的过程
#
# 线程安全问题：
#   - 在多线程环境中，多个线程可能同时访问 _FILE_OPERATION_LOCKS
#   - get() 操作虽然本身是原子的，但"检查-创建"组合不是原子的
#   - 例如：线程 A 检查发现锁不存在，线程 B 也检查发现锁不存在，两者都会创建锁
#   - 使用全局锁保护整个"检查-创建"过程，确保线程安全
_FILE_OPERATION_LOCKS_GUARD = threading.Lock()


# get_file_operation_lock_key：生成文件操作锁的键
#
# 参数：
#   sandbox: Sandbox，沙箱实例
#   path: str，文件路径
#
# 返回值：
#   tuple[str, str]：(sandbox_id, path) 元组
#
# 实现逻辑：
#   - 使用 sandbox.id 作为沙箱标识符
#   - 如果 sandbox 没有 id 属性，使用对象的内存地址作为后备
#   - 这样确保每个沙箱实例有独立的锁空间
def get_file_operation_lock_key(sandbox: Sandbox, path: str) -> tuple[str, str]:
    # 尝试获取 sandbox.id
    # getattr 带默认值，如果属性不存在则返回 None
    sandbox_id = getattr(sandbox, "id", None)

    # 如果 sandbox_id 为空（None 或空字符串）
    if not sandbox_id:
        # 使用对象的内存地址作为后备标识符
        # id(sandbox) 返回对象的内存地址（整数）
        # f-string 格式化为 "instance:地址"
        sandbox_id = f"instance:{id(sandbox)}"

    # 返回元组：(sandbox_id, path)
    # 这个元组用作锁字典的键
    return sandbox_id, path


# get_file_operation_lock：获取或创建文件操作锁
#
# 参数：
#   sandbox: Sandbox，沙箱实例
#   path: str，文件路径
#
# 返回值：
#   threading.Lock：用于保护该文件操作的锁
#
# 实现逻辑：
#   1. 生成锁键 (sandbox_id, path)
#   2. 在全局锁保护下检查是否已存在锁
#   3. 如果不存在，创建新锁并存储
#   4. 返回锁对象
#
# 线程安全：
#   - 使用 _FILE_OPERATION_LOCKS_GUARD 保护整个检查-创建过程
#   - 确保多个线程不会创建重复的锁
#
# 使用示例：
#   with get_file_operation_lock(sandbox, path):
#       # 执行文件操作
#       sandbox.write_file(path, content)
def get_file_operation_lock(sandbox: Sandbox, path: str) -> threading.Lock:
    # 第一步：生成锁键
    # 使用 sandbox.id 和 path 组成唯一的键
    lock_key = get_file_operation_lock_key(sandbox, path)

    # 第二步：在全局锁保护下获取或创建锁
    # with 语句确保锁在块结束时自动释放
    # 这里使用的是 _FILE_OPERATION_LOCKS_GUARD，不是要返回的锁
    with _FILE_OPERATION_LOCKS_GUARD:
        # 尝试从字典中获取已有的锁
        lock = _FILE_OPERATION_LOCKS.get(lock_key)

        # 如果锁不存在
        if lock is None:
            # 创建新的 Lock 对象
            lock = threading.Lock()

            # 将新锁存储到字典中
            _FILE_OPERATION_LOCKS[lock_key] = lock

        # 返回锁对象
        # 注意：返回时仍然持有 _FILE_OPERATION_LOCKS_GUARD
        # 但这没关系，因为 with 块会在返回前释放它
        return lock