"""本地容器后端模块 - 使用 Docker/Apple Container 管理本地沙箱容器。

本模块实现 LocalContainerBackend 类，使用本地 Docker 或 Apple Container 管理沙箱容器。
自动检测容器运行时（macOS 上优先 Apple Container），处理容器生命周期、端口分配、跨进程发现。

核心概念：
  - 容器运行时检测：macOS 优先 Apple Container，Windows/Linux 使用 Docker
  - 确定性容器命名：container_prefix-sandbox_id，便于跨进程发现
  - 端口分配：自动查找空闲端口，绑定到容器 8080
  - 批量操作：list_running 使用单次 docker ps + docker inspect 调用

架构特点：
  - 双层一致性：进程内缓存 + 后端发现（文件锁保证跨进程）
  - 启动协调：进程启动时枚举并收养孤儿容器进入预热池
  - 端口重用处理：Docker 释放端口可能有延迟，自动尝试下一个端口
"""

# 未来类型注解支持
from __future__ import annotations

# 导入 json 模块，JSON 编解码
# _batch_inspect 使用 json.loads 解析 docker inspect 输出
import json

# 导入 logging 模块，记录日志
import logging

# 导入 os 模块，操作系统功能
# 获取环境变量 DEER_FLOW_SANDBOX_HOST
import os

# 导入 subprocess 模块，执行子进程
# 使用 subprocess.run() 执行 docker/container 命令
import subprocess

# 导入 datetime 模块，日期时间处理
# _parse_docker_timestamp 使用 datetime 解析 Docker 时间戳
from datetime import datetime

# 从 deerflow.utils.network 导入网络工具函数
# get_free_port：获取空闲端口
# release_port：释放端口
from deerflow.utils.network import get_free_port, release_port

# 导入 SandboxBackend 抽象基类和 wait_for_sandbox_ready 函数
from .backend import SandboxBackend, wait_for_sandbox_ready

# 导入 SandboxInfo 数据类
from .sandbox_info import SandboxInfo

# 获取本模块的 logger 实例
logger = logging.getLogger(__name__)


# _parse_docker_timestamp：解析 Docker 时间戳
#
# 参数：
#   raw: str，Docker 返回的 ISO 8601 时间戳
#        例如：2026-04-08T01:22:50.123456789Z
#
# 返回值：
#   float，Unix 时间戳（秒）
#
# 实现逻辑：
#   Docker 返回纳秒级精度的 ISO 8601 时间戳，带尾部 Z
#   Python 的 fromisoformat 最多支持微秒，且 3.11 前不支持 Z
#   所以需要规范化字符串后再解析
#
# 特殊情况处理：
#   - 空输入：返回 0.0
#   - 解析失败：返回 0.0（调用方用 0.0 作为"未知年龄"的标记）
def _parse_docker_timestamp(raw: str) -> float:
    """Parse Docker's ISO 8601 timestamp into a Unix epoch float.

    Docker returns timestamps with nanosecond precision and a trailing ``Z``
    (e.g. ``2026-04-08T01:22:50.123456789Z``).  Python's ``fromisoformat``
    accepts at most microseconds and (pre-3.11) does not accept ``Z``, so the
    string is normalized before parsing.  Returns ``0.0`` on empty input or
    parse failure so callers can use ``0.0`` as a sentinel for "unknown age".
    """
    # 空输入处理
    if not raw:
        return 0.0

    try:
        # 去除首尾空白
        s = raw.strip()

        # 处理小数部分：截断到微秒（6位）
        # Docker 格式：2026-04-08T01:22:50.123456789Z
        # Python 需要：2026-04-08T01:22:50.123456+00:00
        if "." in s:
            dot_pos = s.index(".")
            tz_start = dot_pos + 1

            # 找到小数部分结束的索引（第一个非数字字符）
            while tz_start < len(s) and s[tz_start].isdigit():
                tz_start += 1

            # 截断小数部分到 6 位（微秒）
            frac = s[dot_pos + 1 : tz_start][:6]
            tz_suffix = s[tz_start:]
            s = s[: dot_pos + 1] + frac + tz_suffix

        # 替换尾部的 Z 为 +00:00（UTC 时区）
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"

        # 解析为 datetime 并转换为时间戳
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError) as e:
        # 解析失败，记录调试日志并返回 0.0
        logger.debug(f"Could not parse docker timestamp {raw!r}: {e}")
        return 0.0


# _extract_host_port：从 docker inspect 结果提取主机端口
#
# 参数：
#   inspect_entry: dict，docker inspect 返回的单个容器数据
#   container_port: int，容器端口（通常是 8080）
#
# 返回值：
#   int | None，主机端口或 None（如果没有端口映射）
#
# 实现逻辑：
#   从 NetworkSettings.Ports[container_port/tcp] 获取主机端口绑定
def _extract_host_port(inspect_entry: dict, container_port: int) -> int | None:
    """Extract the host port mapped to ``container_port/tcp`` from a docker inspect entry.

    Returns None if the container has no port mapping for that port.
    """
    try:
        # 获取端口映射
        ports = (inspect_entry.get("NetworkSettings") or {}).get("Ports") or {}

        # 获取 container_port/tcp 的绑定
        # 格式：[{HostIp: "0.0.0.0", HostPort: "8080"}, ...]
        bindings = ports.get(f"{container_port}/tcp") or []
        if bindings:
            host_port = bindings[0].get("HostPort")
            if host_port:
                return int(host_port)
    except (ValueError, TypeError, AttributeError):
        pass

    return None


# _format_container_mount：格式化容器挂载参数
#
# 参数：
#   runtime: str，容器运行时（"docker" 或 "container"）
#   host_path: str，主机路径
#   container_path: str，容器内路径
#   read_only: bool，是否只读
#
# 返回值：
#   list[str]，docker/container 命令参数片段
#
# 实现逻辑：
#   Docker：使用 --mount type=bind,... 格式（避免 Windows 路径歧义）
#   Apple Container：使用 -v host:container:ro 格式
#
# Windows 路径问题：
#   Docker 的 -v 语法对 Windows 路径（如 D:/...）有歧义
#   因为冒号既用于盘符也用于卷分隔符
#   使用 --mount 可以避免这个问题
def _format_container_mount(runtime: str, host_path: str, container_path: str, read_only: bool) -> list[str]:
    """Format a bind-mount argument for the selected runtime.

    Docker's ``-v host:container`` syntax is ambiguous for Windows drive-letter
    paths like ``D:/...`` because ``:`` is both the drive separator and the
    volume separator. Use ``--mount type=bind,...`` for Docker to avoid that
    parsing ambiguity. Apple Container keeps using ``-v``.
    """
    # Docker：使用 --mount 格式
    if runtime == "docker":
        mount_spec = f"type=bind,src={host_path},dst={container_path}"
        if read_only:
            mount_spec += ",readonly"
        return ["--mount", mount_spec]

    # Apple Container：使用 -v 格式
    mount_spec = f"{host_path}:{container_path}"
    if read_only:
        mount_spec += ":ro"
    return ["-v", mount_spec]


# LocalContainerBackend 类：本地容器后端
#
# 作用说明：
#   使用本地 Docker 或 Apple Container 管理沙箱容器。
#   自动检测容器运行时，macOS 优先 Apple Container。
#
# 继承关系：
#   继承自 SandboxBackend，必须实现 create/destroy/is_alive/discover 方法
#
# 关键特性：
#   - 确定性容器命名：container_prefix-sandbox_id，便于跨进程发现
#   - 端口分配：从基础端口开始自动查找空闲端口
#   - 容器生命周期：--rm 确保容器停止时自动删除
#   - 批量操作：list_running 使用最小化 subprocess 调用
class LocalContainerBackend(SandboxBackend):
    """Backend that manages sandbox containers locally using Docker or Apple Container.

    On macOS, automatically prefers Apple Container if available, otherwise falls back to Docker.
    On other platforms, uses Docker.

    Features:
    - Deterministic container naming for cross-process discovery
    - Port allocation with thread-safe utilities
    - Container lifecycle management (start/stop with --rm)
    - Support for volume mounts and environment variables
    """

    # __init__：构造函数
    #
    # 参数：
    #   image: str，容器镜像
    #   base_port: int，基础端口号（搜索空闲端口的起始位置）
    #   container_prefix: str，容器名前缀（如 "deer-flow-sandbox"）
    #   config_mounts: list，从配置加载的卷挂载列表
    #   environment: dict[str, str]，注入容器的环境变量
    def __init__(
        self,
        *,
        image: str,
        base_port: int,
        container_prefix: str,
        config_mounts: list,
        environment: dict[str, str],
    ):
        """Initialize the local container backend.

        Args:
            image: Container image to use.
            base_port: Base port number to start searching for free ports.
            container_prefix: Prefix for container names (e.g., "deer-flow-sandbox").
            config_mounts: Volume mount configurations from config (list of VolumeMountConfig).
            environment: Environment variables to inject into containers.
        """
        # 保存容器镜像
        self._image = image

        # 保存基础端口
        self._base_port = base_port

        # 保存容器名前缀
        self._container_prefix = container_prefix

        # 保存配置挂载
        self._config_mounts = config_mounts

        # 保存环境变量
        self._environment = environment

        # 检测容器运行时
        self._runtime = self._detect_runtime()

    # runtime 属性：获取检测到的容器运行时
    @property
    def runtime(self) -> str:
        """The detected container runtime ("docker" or "container")."""
        return self._runtime

    # _detect_runtime：检测容器运行时
    #
    # 返回值：
    #   str，"container" 表示 Apple Container，"docker" 表示 Docker
    #
    # 实现逻辑：
    #   - macOS (Darwin)：优先检查 Apple Container 是否可用
    #   - 其他平台：直接使用 Docker
    def _detect_runtime(self) -> str:
        """Detect which container runtime to use.

        On macOS, prefer Apple Container if available, otherwise fall back to Docker.
        On other platforms, use Docker.

        Returns:
            "container" for Apple Container, "docker" for Docker.
        """
        import platform

        # macOS 检测
        if platform.system() == "Darwin":
            try:
                # 尝试运行 container --version
                result = subprocess.run(
                    ["container", "--version"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                logger.info(f"Detected Apple Container: {result.stdout.strip()}")
                return "container"
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # Apple Container 不可用，回退到 Docker
                logger.info("Apple Container not available, falling back to Docker")

        # 默认使用 Docker
        return "docker"

    # ── SandboxBackend interface ──────────────────────────────────────────

    # create：创建沙箱
    #
    # 参数：
    #   thread_id: str，线程 ID
    #   sandbox_id: str，确定性沙箱标识符
    #   extra_mounts: list[tuple[str, str, bool]] | None，额外的卷挂载
    #
    # 返回值：
    #   SandboxInfo，包含容器详情
    #
    # 错误：
    #   RuntimeError：容器启动失败
    #
    # 实现逻辑：
    #   1. 构建容器名（prefix-sandbox_id）
    #   2. 分配端口（重试直到成功或用尽重试次数）
    #   3. 启动容器
    #   4. 处理端口冲突和名称冲突
    def create(self, thread_id: str, sandbox_id: str, extra_mounts: list[tuple[str, str, bool]] | None = None) -> SandboxInfo:
        """Start a new container and return its connection info.

        Args:
            thread_id: Thread ID for which the sandbox is being created. Useful for backends that want to organize sandboxes by thread.
            sandbox_id: Deterministic sandbox identifier (used in container name).
            extra_mounts: Additional volume mounts as (host_path, container_path, read_only) tuples.

        Returns:
            SandboxInfo with container details.

        Raises:
            RuntimeError: If the container fails to start.
        """
        # 构建容器名：prefix-sandbox_id
        # 例如：deer-flow-sandbox-a1b2c3d4
        container_name = f"{self._container_prefix}-{sandbox_id}"

        # 重试循环：如果 Docker 拒绝端口（如进程重启后旧容器仍持有绑定）
        # 跳过一个端口并尝试下一个
        _next_start = self._base_port
        container_id: str | None = None
        port: int = 0

        for _attempt in range(10):
            # 获取空闲端口
            port = get_free_port(start_port=_next_start)
            try:
                # 尝试启动容器
                container_id = self._start_container(container_name, port, extra_mounts)
                break
            except RuntimeError as exc:
                # 启动失败，释放端口
                release_port(port)
                err = str(exc)
                err_lower = err.lower()

                # 端口已被分配：跳过此端口并重试
                if "port is already allocated" in err or "address already in use" in err_lower:
                    logger.warning(f"Port {port} rejected by Docker (already allocated), retrying with next port")
                    _next_start = port + 1
                    continue

                # 容器名冲突：其他进程可能已启动同名容器
                # 尝试发现并收养现有容器，而不是失败
                if "is already in use by container" in err_lower or "conflict. the container name" in err_lower:
                    logger.warning(f"Container name {container_name} already in use, attempting to discover existing sandbox instance")
                    existing = self.discover(sandbox_id)
                    if existing is not None:
                        return existing

                # 其他错误：直接抛出
                raise
        else:
            # 所有候选端口都已分配
            raise RuntimeError("Could not start sandbox container: all candidate ports are already allocated by Docker")

        # 在 Docker 内部运行（DooD）时，通过 host.docker.internal 访问主机
        # 而不是 localhost（容器在主机 Docker daemon 上运行）
        sandbox_host = os.environ.get("DEER_FLOW_SANDBOX_HOST", "localhost")

        return SandboxInfo(
            sandbox_id=sandbox_id,
            sandbox_url=f"http://{sandbox_host}:{port}",
            container_name=container_name,
            container_id=container_id,
        )

    # destroy：销毁沙箱
    #
    # 参数：
    #   info: SandboxInfo，要销毁的沙箱信息
    def destroy(self, info: SandboxInfo) -> None:
        """Stop the container and release its port."""
        # 优先使用 container_id，回退到 container_name
        # 两者 docker stop 都接受
        # 这确保通过 list_running() 发现的容器（只有名称）也能被停止
        stop_target = info.container_id or info.container_name
        if stop_target:
            self._stop_container(stop_target)

        # 从 sandbox_url 提取端口并释放
        try:
            from urllib.parse import urlparse

            port = urlparse(info.sandbox_url).port
            if port:
                release_port(port)
        except Exception:
            pass

    # is_alive：检查沙箱是否存活
    #
    # 参数：
    #   info: SandboxInfo，要检查的沙箱
    #
    # 返回值：
    #   bool，容器是否正在运行
    #
    # 注意：
    #   这是轻量级检查，不发送 HTTP 请求
    def is_alive(self, info: SandboxInfo) -> bool:
        """Check if the container is still running (lightweight, no HTTP)."""
        if info.container_name:
            return self._is_container_running(info.container_name)
        return False

    # discover：发现已存在的沙箱
    #
    # 参数：
    #   sandbox_id: str，确定性沙箱 ID
    #
    # 返回值：
    #   SandboxInfo | None，找到则返回，否则返回 None
    #
    # 实现逻辑：
    #   1. 检查容器是否运行
    #   2. 获取主机端口
    #   3. 验证容器健康（HTTP 请求）
    def discover(self, sandbox_id: str) -> SandboxInfo | None:
        """Discover an existing container by its deterministic name.

        Checks if a container with the expected name is running, retrieves its
        port, and verifies it responds to health checks.

        Args:
            sandbox_id: The deterministic sandbox ID (determines container name).

        Returns:
            SandboxInfo if container found and healthy, None otherwise.
        """
        # 构建容器名
        container_name = f"{self._container_prefix}-{sandbox_id}"

        # 检查容器是否运行
        if not self._is_container_running(container_name):
            return None

        # 获取主机端口
        port = self._get_container_port(container_name)
        if port is None:
            return None

        # 获取沙箱主机地址
        sandbox_host = os.environ.get("DEER_FLOW_SANDBOX_HOST", "localhost")
        sandbox_url = f"http://{sandbox_host}:{port}"

        # 验证容器健康（5 秒超时）
        if not wait_for_sandbox_ready(sandbox_url, timeout=5):
            return None

        return SandboxInfo(
            sandbox_id=sandbox_id,
            sandbox_url=sandbox_url,
            container_name=container_name,
        )

    # list_running：列出所有运行中的沙箱
    #
    # 返回值：
    #   list[SandboxInfo]，所有运行中的沙箱列表
    #
    # 实现逻辑：
    #   步骤 1：docker ps 列出容器名
    #   步骤 2：批量 docker inspect 获取创建时间和端口映射
    #   总共 2 次 subprocess 调用（而不是 naive 方案的 2N+1 次）
    #
    # 注意：
    #   Docker 的 --filter name= 是子串匹配，所以需要额外的前缀检查
    #   没有端口映射的容器也会包含（sandbox_url 为空），以便收养孤儿
    def list_running(self) -> list[SandboxInfo]:
        """Enumerate all running containers matching the configured prefix.

        Uses a single ``docker ps`` call to list container names, then a
        single batched ``docker inspect`` call to retrieve creation timestamp
        and port mapping for all containers at once.  Total subprocess calls:
        2 (down from 2N+1 in the naive per-container approach).

        Note: Docker's ``--filter name=`` performs *substring* matching,
        so a secondary ``startswith`` check is applied to ensure only
        containers with the exact prefix are included.

        Containers without port mappings are still included (with empty
        sandbox_url) so that startup reconciliation can adopt orphans
        regardless of their port state.
        """
        # 步骤 1：通过 docker ps 枚举容器名
        try:
            result = subprocess.run(
                [
                    self._runtime,
                    "ps",
                    "--filter",
                    f"name={self._container_prefix}-",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                logger.warning(
                    "Failed to list running containers with %s ps (returncode=%s, stderr=%s)",
                    self._runtime,
                    result.returncode,
                    stderr or "<empty>",
                )
                return []
            if not result.stdout.strip():
                return []
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning(f"Failed to list running containers: {e}")
            return []

        # 过滤：确保只有精确匹配前缀的容器
        # （docker filter 是子串匹配）
        container_names = [
            name.strip()
            for name in result.stdout.strip().splitlines()
            if name.strip().startswith(self._container_prefix + "-")
        ]
        if not container_names:
            return []

        # 步骤 2：批量 docker inspect — 单次 subprocess 调用
        inspections = self._batch_inspect(container_names)

        # 构建 SandboxInfo 列表
        infos: list[SandboxInfo] = []
        sandbox_host = os.environ.get("DEER_FLOW_SANDBOX_HOST", "localhost")

        for container_name in container_names:
            data = inspections.get(container_name)
            if data is None:
                # 容器在 ps 和 inspect 之间消失，或 inspect 失败
                continue

            created_at, host_port = data

            # 从容器名提取 sandbox_id
            # 格式：container_prefix-sandbox_id
            sandbox_id = container_name[len(self._container_prefix) + 1 :]

            # 构建 URL（如果有端口映射）
            sandbox_url = f"http://{sandbox_host}:{host_port}" if host_port else ""

            infos.append(
                SandboxInfo(
                    sandbox_id=sandbox_id,
                    sandbox_url=sandbox_url,
                    container_name=container_name,
                    created_at=created_at,
                )
            )

        logger.info(f"Found {len(infos)} running sandbox container(s)")
        return infos

    # _batch_inspect：批量检查容器
    #
    # 参数：
    #   container_names: list[str]，容器名列表
    #
    # 返回值：
    #   dict[str, tuple[float, int | None]]，容器名 -> (创建时间, 主机端口)
    #
    # 实现逻辑：
    #   单次 docker inspect 调用获取所有容器信息
    #   解析 JSON 输出，提取创建时间和端口映射
    def _batch_inspect(self, container_names: list[str]) -> dict[str, tuple[float, int | None]]:
        """Batch-inspect containers in a single subprocess call.

        Returns a mapping of ``container_name -> (created_at, host_port)``.
        Missing containers or parse failures are silently dropped from the result.
        """
        if not container_names:
            return {}

        try:
            # 单次 subprocess 调用，inspect 多个容器
            result = subprocess.run(
                [self._runtime, "inspect", *container_names],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning(f"Failed to batch-inspect containers: {e}")
            return {}

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            logger.warning(
                "Failed to batch-inspect containers with %s inspect (returncode=%s, stderr=%s)",
                self._runtime,
                result.returncode,
                stderr or "<empty>",
            )
            return {}

        # 解析 JSON 输出
        try:
            payload = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse docker inspect output as JSON: {e}")
            return {}

        # 解析每个容器的信息
        out: dict[str, tuple[float, int | None]] = {}
        for entry in payload:
            # docker inspect 返回的 Name 字段以 / 开头
            name = (entry.get("Name") or "").lstrip("/")
            if not name:
                continue

            # 解析创建时间
            created_at = _parse_docker_timestamp(entry.get("Created", ""))

            # 提取主机端口（容器端口 8080）
            host_port = _extract_host_port(entry, 8080)

            out[name] = (created_at, host_port)

        return out

    # ── Container operations ─────────────────────────────────────────────

    # _start_container：启动容器
    #
    # 参数：
    #   container_name: str，容器名称
    #   port: int，主机端口（映射到容器 8080）
    #   extra_mounts: list[tuple[str, str, bool]] | None，额外的挂载
    #
    # 返回值：
    #   str，容器 ID
    #
    # 错误：
    #   RuntimeError：容器启动失败
    def _start_container(
        self,
        container_name: str,
        port: int,
        extra_mounts: list[tuple[str, str, bool]] | None = None,
    ) -> str:
        """Start a new container.

        Args:
            container_name: Name for the container.
            port: Host port to map to container port 8080.
            extra_mounts: Additional volume mounts.

        Returns:
            The container ID.

        Raises:
            RuntimeError: If container fails to start.
        """
        # 构建 docker/container run 命令
        cmd = [self._runtime, "run"]

        # Docker 安全选项
        if self._runtime == "docker":
            # seccomp=unconfined：禁用安全沙箱（允许更多系统调用）
            cmd.extend(["--security-opt", "seccomp=unconfined"])

        # 添加标准参数
        cmd.extend(
            [
                "--rm",              # 容器停止时自动删除
                "-d",                # 后台运行
                "-p", f"{port}:8080",  # 端口映射
                "--name", container_name,  # 容器名称
            ]
        )

        # 环境变量
        for key, value in self._environment.items():
            cmd.extend(["-e", f"{key}={value}"])

        # 配置级卷挂载
        for mount in self._config_mounts:
            cmd.extend(
                _format_container_mount(
                    self._runtime,
                    mount.host_path,
                    mount.container_path,
                    mount.read_only,
                )
            )

        # 额外挂载（线程特定、技能等）
        if extra_mounts:
            for host_path, container_path, read_only in extra_mounts:
                cmd.extend(
                    _format_container_mount(
                        self._runtime,
                        host_path,
                        container_path,
                        read_only,
                    )
                )

        # 添加镜像
        cmd.append(self._image)

        # 记录启动命令
        logger.info(f"Starting container using {self._runtime}: {' '.join(cmd)}")

        try:
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # 提取容器 ID（stdout 是容器 ID）
            container_id = result.stdout.strip()
            logger.info(f"Started container {container_name} (ID: {container_id}) using {self._runtime}")
            return container_id
        except subprocess.CalledProcessError as e:
            # 启动失败
            logger.error(f"Failed to start container using {self._runtime}: {e.stderr}")
            raise RuntimeError(f"Failed to start sandbox container: {e.stderr}")

    # _stop_container：停止容器
    #
    # 参数：
    #   container_id: str，容器 ID 或名称
    def _stop_container(self, container_id: str) -> None:
        """Stop a container (--rm ensures automatic removal)."""
        try:
            subprocess.run(
                [self._runtime, "stop", container_id],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"Stopped container {container_id} using {self._runtime}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to stop container {container_id}: {e.stderr}")

    # _is_container_running：检查容器是否运行
    #
    # 参数：
    #   container_name: str，容器名称
    #
    # 返回值：
    #   bool，容器是否正在运行
    #
    # 实现逻辑：
    #   使用 docker/container inspect -f "{{.State.Running}}"
    #   检查容器状态是否为 Running
    def _is_container_running(self, container_name: str) -> bool:
        """Check if a named container is currently running.

        This enables cross-process container discovery — any process can detect
        containers started by another process via the deterministic container name.
        """
        try:
            result = subprocess.run(
                [self._runtime, "inspect", "-f", "{{.State.Running}}", container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and result.stdout.strip().lower() == "true"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    # _get_container_port：获取容器的主机端口
    #
    # 参数：
    #   container_name: str，容器名称
    #
    # 返回值：
    #   int | None，主机端口或 None
    #
    # 实现逻辑：
    #   使用 docker/container port 获取端口映射
    #   输出格式：0.0.0.0:PORT 或 :::PORT
    def _get_container_port(self, container_name: str) -> int | None:
        """Get the host port of a running container.

        Args:
            container_name: The container name to inspect.

        Returns:
            The host port mapped to container port 8080, or None if not found.
        """
        try:
            result = subprocess.run(
                [self._runtime, "port", container_name, "8080"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # 输出格式：0.0.0.0:PORT 或 :::PORT
                port_str = result.stdout.strip().split(":")[-1]
                return int(port_str)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            pass
        return None