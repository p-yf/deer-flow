"""远程沙箱后端模块 - 通过 provisioner 服务管理远程沙箱生命周期。

本模块实现 RemoteSandboxBackend 类，通过 HTTP API 与 provisioner 服务通信，
由 provisioner 在 k3s 集群中动态创建/销毁沙箱 Pod 和 Service。

架构说明：
  - provisioner 在 k3s 中为每个 sandbox_id 创建 Pod + NodePort Service
  - 后端直接通过 k3s:{NodePort} 访问沙箱 Pod

使用场景：
  - config.yaml 中配置 provisioner_url
  - 例如：sandbox.provisioner_url: http://provisioner:8002

限制：
  - 沙箱发现需要 provisioner 运行并可访问
  - destroy 是幂等的（404 被视为成功）
"""

from __future__ import annotations

import logging

import requests

from .backend import SandboxBackend
from .sandbox_info import SandboxInfo

logger = logging.getLogger(__name__)


class RemoteSandboxBackend(SandboxBackend):
    """Backend that delegates sandbox lifecycle to the provisioner service.

    All Pod creation, destruction, and discovery are handled by the
    provisioner.  This backend is a thin HTTP client.

    Typical config.yaml::

        sandbox:
          use: deerflow.community.aio_sandbox:AioSandboxProvider
          provisioner_url: http://provisioner:8002
    """

    def __init__(self, provisioner_url: str):
        """Initialize with the provisioner service URL.

        Args:
            provisioner_url: URL of the provisioner service
                             (e.g., ``http://provisioner:8002``).
        """
        self._provisioner_url = provisioner_url.rstrip("/")

    @property
    def provisioner_url(self) -> str:
        return self._provisioner_url

    # ── SandboxBackend interface ──────────────────────────────────────────

    def create(
        self,
        thread_id: str,
        sandbox_id: str,
        extra_mounts: list[tuple[str, str, bool]] | None = None,
    ) -> SandboxInfo:
        """Create a sandbox Pod + Service via the provisioner.

        Calls ``POST /api/sandboxes`` which creates a dedicated Pod +
        NodePort Service in k3s.
        """
        return self._provisioner_create(thread_id, sandbox_id, extra_mounts)

    def destroy(self, info: SandboxInfo) -> None:
        """Destroy a sandbox Pod + Service via the provisioner."""
        self._provisioner_destroy(info.sandbox_id)

    def is_alive(self, info: SandboxInfo) -> bool:
        """Check whether the sandbox Pod is running."""
        return self._provisioner_is_alive(info.sandbox_id)

    def discover(self, sandbox_id: str) -> SandboxInfo | None:
        """Discover an existing sandbox via the provisioner.

        Calls ``GET /api/sandboxes/{sandbox_id}`` and returns info if
        the Pod exists.
        """
        return self._provisioner_discover(sandbox_id)

    # ── Provisioner API calls ─────────────────────────────────────────────

    def _provisioner_create(self, thread_id: str, sandbox_id: str, extra_mounts: list[tuple[str, str, bool]] | None = None) -> SandboxInfo:
        """POST /api/sandboxes → create Pod + Service."""
        try:
            resp = requests.post(
                f"{self._provisioner_url}/api/sandboxes",
                json={
                    "sandbox_id": sandbox_id,
                    "thread_id": thread_id,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"Provisioner created sandbox {sandbox_id}: sandbox_url={data['sandbox_url']}")
            return SandboxInfo(
                sandbox_id=sandbox_id,
                sandbox_url=data["sandbox_url"],
            )
        except requests.RequestException as exc:
            logger.error(f"Provisioner create failed for {sandbox_id}: {exc}")
            raise RuntimeError(f"Provisioner create failed: {exc}") from exc

    def _provisioner_destroy(self, sandbox_id: str) -> None:
        """DELETE /api/sandboxes/{sandbox_id} → destroy Pod + Service."""
        try:
            resp = requests.delete(
                f"{self._provisioner_url}/api/sandboxes/{sandbox_id}",
                timeout=15,
            )
            if resp.ok:
                logger.info(f"Provisioner destroyed sandbox {sandbox_id}")
            else:
                logger.warning(f"Provisioner destroy returned {resp.status_code}: {resp.text}")
        except requests.RequestException as exc:
            logger.warning(f"Provisioner destroy failed for {sandbox_id}: {exc}")

    def _provisioner_is_alive(self, sandbox_id: str) -> bool:
        """GET /api/sandboxes/{sandbox_id} → check Pod phase."""
        try:
            resp = requests.get(
                f"{self._provisioner_url}/api/sandboxes/{sandbox_id}",
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                return data.get("status") == "Running"
            return False
        except requests.RequestException:
            return False

    def _provisioner_discover(self, sandbox_id: str) -> SandboxInfo | None:
        """GET /api/sandboxes/{sandbox_id} → discover existing sandbox."""
        try:
            resp = requests.get(
                f"{self._provisioner_url}/api/sandboxes/{sandbox_id}",
                timeout=10,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            return SandboxInfo(
                sandbox_id=sandbox_id,
                sandbox_url=data["sandbox_url"],
            )
        except requests.RequestException as exc:
            logger.debug(f"Provisioner discover failed for {sandbox_id}: {exc}")
            return None