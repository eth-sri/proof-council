"""Docker-backed sandbox for configurable CLI agents.

Wraps each CLI invocation in ``docker run --rm ...`` with:

- capability drop (``--cap-drop ALL``; optionally
  ``--security-opt no-new-privileges``)
- resource limits (``--memory``, ``--cpus``, ``--pids-limit``)
- writable bind mount ONLY for the per-invocation workdir
- ``tmpfs`` for ``/tmp``
- non-root uid (matching the host's default 1000 so bind-mount file
  ownership stays sane)
- environment scrubbed to an explicit allowlist + provider keys

The image is expected to be built locally via ``deploy/sandbox/Dockerfile``;
we do not pull from a registry.

Use this backend when a CLI node should run inside the local sandbox
image from ``deploy/sandbox/Dockerfile``.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Iterable, Mapping

from proofstack.sandbox.base import CommandResult, Sandbox, SandboxSpec
from proofstack.sandbox.subprocess import _StreamingProcess


class DockerSandboxError(RuntimeError):
    """Raised when docker is missing / image is not built / etc."""


_DOCKER_CONTROL_TIMEOUT_S = 5.0
_DOCKER_OWNER_LABEL = "proofstack.workspace-owner"


async def _docker_control(*args: str) -> tuple[int, str, str]:
    """Run a bounded Docker control command and retain its diagnostics."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise DockerSandboxError("docker binary not found") from e
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(),
            timeout=_DOCKER_CONTROL_TIMEOUT_S,
        )
    except asyncio.TimeoutError as e:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except (asyncio.TimeoutError, ProcessLookupError, ValueError):
            pass
        raise DockerSandboxError(
            f"docker {' '.join(args)} timed out"
        ) from e
    return (
        proc.returncode if proc.returncode is not None else -1,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )


async def _docker_container_identity(
    container_name: str,
) -> tuple[str, str | None, str] | None:
    """Return an exact container's status/owner/id, or ``None`` if absent."""
    returncode, stdout, stderr = await _docker_control(
        "container",
        "inspect",
        "--format",
        (
            "{{.Id}}\t{{.State.Status}}\t"
            f'{{{{index .Config.Labels "{_DOCKER_OWNER_LABEL}"}}}}'
        ),
        container_name,
    )
    if returncode == 0:
        fields = stdout.rstrip("\r\n").split("\t", maxsplit=2)
        if len(fields) != 3 or not fields[0].strip():
            raise DockerSandboxError(
                f"Docker returned malformed identity for {container_name!r}"
            )
        container_id, status, owner = fields
        normalized_owner = owner.strip()
        if normalized_owner in {"", "<no value>"}:
            normalized_owner = None
        return (
            status.strip() or "unknown",
            normalized_owner,
            container_id.strip(),
        )
    detail = stderr.strip()
    lowered = detail.lower()
    if "no such object" in lowered or "no such container" in lowered:
        return None
    failure_detail = detail or f"exit {returncode}"
    raise DockerSandboxError(
        f"could not inspect Docker container {container_name!r}: "
        f"{failure_detail}"
    )


async def _docker_container_status(container_name: str) -> str | None:
    identity = await _docker_container_identity(container_name)
    return identity[0] if identity is not None else None


async def _docker_kill(container_name: str, *, owner: str) -> bool:
    """Stop and remove a container, returning whether absence was verified.

    Killing the ``docker run`` CLI client does NOT propagate to the
    container — dockerd sees a detached client but PID 1 in the
    container keeps running until it exits on its own, holding on to
    the per-container resource caps. Termination therefore addresses
    the named container and verifies that it no longer exists before
    the workspace is treated as idle.
    """
    try:
        identity = await _docker_container_identity(container_name)
    except DockerSandboxError:
        return False
    if identity is None:
        return True
    if identity[1] != owner:
        return False
    container_id = identity[2]

    for command in (
        ("kill", container_id),
        ("rm", "--force", container_id),
    ):
        try:
            await _docker_control(*command)
        except DockerSandboxError:
            # The final inspect below is authoritative. A command can fail
            # simply because --rm already removed the container.
            continue
    try:
        return await _docker_container_identity(container_name) is None
    except DockerSandboxError:
        return False


CONTAINER_WORKDIR = "/work"
CONTAINER_TMPFS = "/tmp"
# Deliberate, minimal PATH inside the container. Does not inherit from
# the host (which may have random user-installed shims). Caller-supplied
# extra_path entries (typically the per-invocation shim dir under
# sandbox.root) are prepended to this by _build_docker_cmd after host-
# to-container translation.
CONTAINER_BASE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class DockerSandbox(Sandbox):
    """Run each command in a fresh ``docker run --rm`` container.

    Host <-> container path translation: anything the caller passes
    that contains ``str(self.root)`` (typically
    ``FINISH_DONE_PATH`` and ``extra_path``) is rewritten to
    ``/work`` before being handed to the container. The container sees
    ``/work``; the orchestrator on the host polls
    ``<self.root>/done.json`` via the same bind mount.
    """

    def __init__(
        self,
        spec: SandboxSpec,
        *,
        root: Path | None = None,
        inherited_fds: Iterable[int] = (),
    ) -> None:
        super().__init__(spec, root=root, inherited_fds=inherited_fds)
        self.container_owner = uuid.uuid4().hex

    @property
    def container_name(self) -> str:
        return _container_name_for_root(self.root)

    async def ensure_workspace_available(self) -> None:
        status = await _docker_container_status(self.container_name)
        if status is not None:
            raise DockerSandboxError(
                "persistent workspace still has a Docker sandbox "
                f"({self.container_name}, status={status}); refusing to overlap "
                "a resumed invocation"
            )

    async def run_command(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        timeout_s: int | None = None,
        env_extra: Mapping[str, str] | None = None,
        extra_path: Iterable[Path] = (),
        input_data: str | bytes | None = None,
    ) -> CommandResult:
        input_bytes = (
            input_data.encode("utf-8") if isinstance(input_data, str) else input_data
        )
        container_name = self.container_name
        docker_cmd = self._build_docker_cmd(
            cmd,
            env_extra=env_extra,
            extra_path=list(extra_path),
            cwd=cwd,
            interactive=input_bytes is not None,
            container_name=container_name,
        )
        timeout = timeout_s if timeout_s is not None else self.spec.timeout_s
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdin=(asyncio.subprocess.PIPE if input_bytes is not None else None),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                pass_fds=self.inherited_fds,
            )
        except FileNotFoundError as e:
            raise DockerSandboxError(
                "docker binary not found — install Docker Desktop (Windows) "
                "or the Docker engine (Linux), or set "
                "PROOFSTACK_SANDBOX_BACKEND=subprocess to bypass."
            ) from e
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=input_bytes), timeout=timeout
            )
            returncode = proc.returncode if proc.returncode is not None else -1
            if returncode != 0:
                container_stopped = await _docker_kill(
                    container_name,
                    owner=self.container_owner,
                )
                if not container_stopped:
                    suffix = (
                        "Docker container could not be confirmed stopped or "
                        "belongs to another invocation"
                    ).encode("utf-8")
                    stderr_b = stderr_b.rstrip()
                    stderr_b += (b"\n" if stderr_b else b"") + suffix
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.communicate()
            except ProcessLookupError:
                pass
            # Stop the client first so it cannot create the named container
            # after an early "not found" result from the control commands.
            container_stopped = await _docker_kill(
                container_name,
                owner=self.container_owner,
            )
            returncode = -9
            stdout_b = b""
            detail = f"timeout after {timeout}s"
            if not container_stopped:
                detail += "; Docker container could not be confirmed stopped"
            stderr_b = detail.encode("utf-8")
        elapsed = time.monotonic() - start
        return CommandResult(
            cmd=cmd,
            returncode=returncode,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            duration_s=elapsed,
        )

    async def stream_command(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        timeout_s: int | None = None,
        env_extra: Mapping[str, str] | None = None,
        extra_path: Iterable[Path] = (),
    ) -> "_StreamingProcess":
        container_name = self.container_name
        docker_cmd = self._build_docker_cmd(
            cmd,
            env_extra=env_extra,
            extra_path=list(extra_path),
            cwd=cwd,
            interactive=True,
            container_name=container_name,
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                pass_fds=self.inherited_fds,
            )
        except FileNotFoundError as e:
            raise DockerSandboxError(
                "docker binary not found — install Docker Desktop (Windows) "
                "or the Docker engine (Linux), or set "
                "PROOFSTACK_SANDBOX_BACKEND=subprocess to bypass."
            ) from e
        deadline = (
            time.monotonic() + (timeout_s if timeout_s is not None else self.spec.timeout_s)
        )
        return _DockerStreamingProcess(
            proc=proc,
            cmd=docker_cmd,
            deadline=deadline,
            container_name=container_name,
            container_owner=self.container_owner,
        )

    # --- helpers ----------------------------------------------------------

    def _build_docker_cmd(
        self,
        inner_cmd: list[str],
        *,
        env_extra: Mapping[str, str] | None,
        extra_path: list[Path],
        cwd: str | None,
        interactive: bool,
        container_name: str,
    ) -> list[str]:
        args: list[str] = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--label",
            f"{_DOCKER_OWNER_LABEL}={self.container_owner}",
        ]
        if interactive:
            args += ["-i"]
        args += [
            "--memory", f"{self.spec.memory_gb}g",
            "--cpus", str(self.spec.cpu_limit),
            "--pids-limit", str(self.spec.docker_pids_limit),
            "--cap-drop", "ALL",
            "--network", self.spec.docker_network,
            "--user", f"{os.getuid()}:{os.getgid()}",
            # Docker treats relative source paths as named volumes, so
            # resolve to an absolute host path. RunContext.create can keep
            # the workdir relative (e.g. `--output outputs`), which would
            # otherwise yield `-v outputs/...:/work` and fail.
            "-v", f"{self.root.resolve()}:{CONTAINER_WORKDIR}",
            "-w",
            (CONTAINER_WORKDIR + "/" + cwd) if cwd else CONTAINER_WORKDIR,
            # tmpfs so /tmp is writable without escaping the mount
            "--tmpfs", f"{CONTAINER_TMPFS}:size=1g,mode=1777",
        ]
        if self.spec.docker_no_new_privileges:
            args += ["--security-opt", "no-new-privileges"]

        # --- Env forwarding -------------------------------------------
        # Pinned container env (HOME, TMPDIR, PATH) — the host's values
        # are NEVER leaked into the container. extra_path entries are
        # translated from host -> container paths and prepended to PATH
        # so the per-invocation shim dir (e.g. the CLIAgent finish
        # bin) resolves first. Mirrors SubprocessSandbox behaviour.
        translated_extra = [
            self._translate_path(str(p.resolve() if p.is_absolute() else p))
            for p in extra_path
        ]
        path_value = ":".join([*translated_extra, CONTAINER_BASE_PATH]) if translated_extra else CONTAINER_BASE_PATH
        fixed_env = {
            "HOME": CONTAINER_WORKDIR,
            "TMPDIR": CONTAINER_TMPFS,
            "PATH": path_value,
        }
        for k, v in fixed_env.items():
            args += ["-e", f"{k}={v}"]

        # Provider keys: pass names only; docker reads the value from
        # the orchestrator's env at spawn time and sets them in the
        # container. Values are never written to disk.
        for key in self.spec.provider_keys:
            if os.environ.get(key) is not None:
                args += ["-e", key]

        # Caller-supplied env_extra — translate host paths -> container paths.
        if env_extra:
            for k, v in env_extra.items():
                args += ["-e", f"{k}={self._translate_path(str(v))}"]

        # spec.extra_env last, so a SandboxSpec-level override wins.
        for k, v in self.spec.extra_env.items():
            args += ["-e", f"{k}={self._translate_path(str(v))}"]

        # --- Any extra user-supplied docker args ----------------------
        args += list(self.spec.docker_extra_args)

        # --- Image + inner command -----------------------------------
        args += [self.spec.docker_image]
        args += inner_cmd
        return args

    def _translate_path(self, value: str) -> str:
        """Rewrite any occurrence of ``str(self.root)`` with ``/work``.

        The bind mount aliases these two paths. Callers like CLIAgent
        pass host paths (``sandbox.root / "done.json"``) via env; we
        translate once here so every backend sees identical semantics.
        """
        root_str = str(self.root)
        if root_str in value:
            return value.replace(root_str, CONTAINER_WORKDIR)
        return value


def _container_name_for_root(root: Path) -> str:
    """Return the stable Docker identity protecting one workspace path."""
    resolved = Path(root).resolve()
    digest = hashlib.sha256(os.fsencode(str(resolved))).hexdigest()[:24]
    return f"proofstack-sbx-{digest}"


class _DockerStreamingProcess(_StreamingProcess):
    """Streaming handle that also knows how to signal its container.

    ``_StreamingProcess.terminate`` only kills the Docker CLI client,
    which can leave the container running. We then remove the exact
    owner-labelled container and verify that the workspace is idle.
    """

    def __init__(
        self,
        *,
        container_name: str,
        container_owner: str,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.container_name = container_name
        self.container_owner = container_owner
        self._container_stopped = False

    @property
    def worker_stopped(self) -> bool:
        return self._container_stopped

    async def terminate(self) -> None:
        try:
            # Stop the client first so no new container can appear after the
            # final inspect performed by _docker_kill.
            await super().terminate()
        finally:
            self._container_stopped = await _docker_kill(
                self.container_name,
                owner=self.container_owner,
            )


def check_image_available(image: str = "proofstack-sandbox:latest") -> bool:
    """Returns True if the given docker image is built locally.

    Non-async because it's called at startup / construction time.
    Uses ``docker image inspect`` which is cheap.
    """
    import subprocess as _sp
    try:
        res = _sp.run(
            ["docker", "image", "inspect", image],
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
            timeout=5,
        )
        return res.returncode == 0
    except (FileNotFoundError, _sp.TimeoutExpired):
        return False


__all__ = ["DockerSandbox", "DockerSandboxError", "check_image_available"]
