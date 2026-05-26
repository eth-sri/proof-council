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
import os
import time
import uuid
from pathlib import Path
from typing import Iterable, Mapping

from proofstack.sandbox.base import CommandResult, Sandbox
from proofstack.sandbox.subprocess import _StreamingProcess


async def _docker_kill(container_name: str) -> None:
    """Best-effort ``docker kill <name>``.

    Killing the ``docker run`` CLI client does NOT propagate to the
    container — dockerd sees a detached client but PID 1 in the
    container keeps running until it exits on its own, holding on to
    the per-container resource caps. On timeout we issue an explicit
    ``docker kill`` against the named container so nothing orphans.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "kill", container_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5)
    except (FileNotFoundError, asyncio.TimeoutError, Exception):
        pass


CONTAINER_WORKDIR = "/work"
CONTAINER_TMPFS = "/tmp"
# Deliberate, minimal PATH inside the container. Does not inherit from
# the host (which may have random user-installed shims). Caller-supplied
# extra_path entries (typically the per-invocation shim dir under
# sandbox.root) are prepended to this by _build_docker_cmd after host-
# to-container translation.
CONTAINER_BASE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class DockerSandboxError(RuntimeError):
    """Raised when docker is missing / image is not built / etc."""


class DockerSandbox(Sandbox):
    """Run each command in a fresh ``docker run --rm`` container.

    Host <-> container path translation: anything the caller passes
    that contains ``str(self.root)`` (typically
    ``FINISH_DONE_PATH`` and ``extra_path``) is rewritten to
    ``/work`` before being handed to the container. The container sees
    ``/work``; the orchestrator on the host polls
    ``<self.root>/done.json`` via the same bind mount.
    """

    async def run_command(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        timeout_s: int | None = None,
        env_extra: Mapping[str, str] | None = None,
        extra_path: Iterable[Path] = (),
    ) -> CommandResult:
        container_name = _new_container_name()
        docker_cmd = self._build_docker_cmd(
            cmd,
            env_extra=env_extra,
            extra_path=list(extra_path),
            cwd=cwd,
            interactive=False,
            container_name=container_name,
        )
        timeout = timeout_s if timeout_s is not None else self.spec.timeout_s
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise DockerSandboxError(
                "docker binary not found — install Docker Desktop (Windows) "
                "or the Docker engine (Linux), or set "
                "PROOFSTACK_SANDBOX_BACKEND=subprocess to bypass."
            ) from e
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            returncode = proc.returncode if proc.returncode is not None else -1
        except asyncio.TimeoutError:
            await _docker_kill(container_name)
            try:
                proc.kill()
                await proc.communicate()
            except ProcessLookupError:
                pass
            returncode = -9
            stdout_b = b""
            stderr_b = f"timeout after {timeout}s".encode("utf-8")
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
        container_name = _new_container_name()
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
        args: list[str] = ["docker", "run", "--rm", "--name", container_name]
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


def _new_container_name() -> str:
    """Per-invocation container name; used with ``docker kill`` on timeout."""
    return f"proofstack-sbx-{uuid.uuid4().hex[:12]}"


class _DockerStreamingProcess(_StreamingProcess):
    """Streaming handle that also knows how to signal its container.

    ``_StreamingProcess.terminate`` only kills the docker CLI client,
    which leaves the container running. We override to ``docker kill``
    the named container first so nothing orphans.
    """

    def __init__(self, *, container_name: str, **kw) -> None:
        super().__init__(**kw)
        self.container_name = container_name

    async def terminate(self) -> None:
        if self.proc.returncode is None:
            await _docker_kill(self.container_name)
        await super().terminate()


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
