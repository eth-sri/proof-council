"""Local-subprocess sandbox backend.

Per-invocation isolation:
- fresh ``tempfile.mkdtemp`` workdir;
- env stripped to the spec's allowlist plus declared provider keys;
- wallclock timeout enforced by the orchestrator;
- soft CPU/memory limits via ``setrlimit`` (best-effort);
- new POSIX session (``start_new_session=True``) so descendents stay in
  one process group and ``os.killpg`` can clean them up on teardown.
"""
from __future__ import annotations

import asyncio
import os
import resource
import shlex
import signal
import time
from pathlib import Path
from typing import Iterable, Mapping

from proofstack.sandbox.base import CommandResult, Sandbox


def _make_preexec(memory_gb: int, cpu_limit: int, cpu_seconds: int):
    """Returns a preexec_fn that applies soft setrlimit limits.

    Linux-only; returns ``None`` on platforms without ``resource``.
    Limits are best-effort — the host container is the actual security
    boundary per SPEC §3.3.1.

    ``cpu_limit`` is the number of CPU cores the task is allowed to use
    in parallel; ``cpu_seconds`` is the wall-clock timeout in seconds
    that the orchestrator will enforce. The actual CPU-time ceiling is
    ``cpu_limit * cpu_seconds`` (with a 60s floor for very short runs),
    which is the most CPU-time a perfectly parallel task could consume
    inside its wall budget. The previous formula was ``cpu_limit * 60``,
    which dimensionally treated ``cpu_limit`` as minutes and killed
    multi-minute CAS/codex runs at 4 minutes of CPU-time regardless of
    the configured wall timeout.
    """

    def _apply() -> None:
        try:
            mem_bytes = memory_gb * 1024 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
        try:
            rlimit_cpu = max(int(cpu_limit) * int(cpu_seconds), 60)
            resource.setrlimit(resource.RLIMIT_CPU, (rlimit_cpu, rlimit_cpu))
        except (ValueError, OSError):
            pass

    return _apply


async def _terminate_process_group(proc: asyncio.subprocess.Process, *, grace_s: float = 5.0) -> None:
    """Best-effort SIGTERM-then-SIGKILL of the child's process group.

    With ``start_new_session=True``, the child is its own session leader
    so ``os.killpg(pid, ...)`` reaches every descendant. Without that,
    CAS / codex subprocesses can outlive the main worker until container
    exit.

    Crucially: we attempt the group kill even when the direct child has
    *already* exited. A short-lived launcher process (e.g. a shell or
    npm wrapper) can exit while leaving long-running descendants (codex,
    a CAS subprocess) alive in the same pgid. Returning early on
    ``proc.returncode is not None`` would skip the group SIGTERM and
    leak those descendants until container teardown.
    """
    pid = proc.pid
    # Phase 1: SIGTERM the entire group. This reaches descendants even
    # when the direct child is gone.
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        # No process group (somehow not a session leader) or already
        # gone. Fall back to single-process terminate.
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    # If the direct child is still around, give it a window to exit
    # cleanly before we escalate.
    if proc.returncode is None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_s)
        except asyncio.TimeoutError:
            pass
    # Phase 2: SIGKILL the group. Always attempted — descendants may
    # still be alive even after the direct child finishes.
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    if proc.returncode is None:
        try:
            await proc.wait()
        except ProcessLookupError:
            pass


class SubprocessSandbox(Sandbox):
    """Run commands in the sandbox root via ``asyncio.create_subprocess_exec``."""

    async def run_command(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        timeout_s: int | None = None,
        env_extra: Mapping[str, str] | None = None,
        extra_path: Iterable[Path] = (),
    ) -> CommandResult:
        cwd_path = self.root / cwd if cwd else self.root
        env = self.spec.build_env(sandbox_root=self.root, extra_path=extra_path)
        if env_extra:
            env.update(env_extra)
        timeout = timeout_s if timeout_s is not None else self.spec.timeout_s

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd_path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_make_preexec(self.spec.memory_gb, self.spec.cpu_limit, int(timeout)),
                start_new_session=True,
            )
        except FileNotFoundError as e:
            return CommandResult(cmd=cmd, returncode=127, stdout="", stderr=str(e), duration_s=0.0)

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            returncode = proc.returncode if proc.returncode is not None else -1
        except asyncio.TimeoutError:
            await _terminate_process_group(proc, grace_s=5.0)
            try:
                stdout_b, stderr_b = await proc.communicate()
            except (ProcessLookupError, ValueError):
                stdout_b = b""
                stderr_b = b""
            returncode = -9
            if not stderr_b:
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
        """Spawn a long-running command and return a handle.

        Used by CLIAgent so the orchestrator can poll for ``done.json``
        and emit ``cli.heartbeat`` events without blocking on the child.
        ``stdin`` is piped so CLIAgent can write the prompt to it; the
        DockerSandbox equivalent does the same. Without this, codex
        inherits the parent's stdin, sees EOF immediately, and exits
        with code 1 before doing any work.
        """
        cwd_path = self.root / cwd if cwd else self.root
        env = self.spec.build_env(sandbox_root=self.root, extra_path=extra_path)
        if env_extra:
            env.update(env_extra)
        timeout = timeout_s if timeout_s is not None else self.spec.timeout_s
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd_path),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=_make_preexec(self.spec.memory_gb, self.spec.cpu_limit, int(timeout)),
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        return _StreamingProcess(proc=proc, cmd=cmd, deadline=deadline)


class _StreamingProcess:
    def __init__(self, *, proc: asyncio.subprocess.Process, cmd: list[str], deadline: float):
        self.proc = proc
        self.cmd = cmd
        self.deadline = deadline
        self._stdout_buf: list[str] = []
        self._stderr_buf: list[str] = []
        self._stdout_task = asyncio.create_task(self._drain(proc.stdout, self._stdout_buf))
        self._stderr_task = asyncio.create_task(self._drain(proc.stderr, self._stderr_buf))

    @staticmethod
    async def _drain(stream, sink: list[str]) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            sink.append(chunk.decode("utf-8", errors="replace"))

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    @property
    def done(self) -> bool:
        return self.proc.returncode is not None

    async def wait(self, timeout_s: float | None = None) -> int:
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return -1
        await self._drain_pipes(timeout_s=5.0)
        return self.proc.returncode or 0

    async def terminate(self) -> None:
        await _terminate_process_group(self.proc, grace_s=5.0)
        await self._drain_pipes(timeout_s=5.0)

    async def _drain_pipes(self, *, timeout_s: float) -> None:
        """Drain stdout/stderr with a hard cap.

        When the main CLI process exits but spawns a background child
        that inherited stdout/stderr, the pipes never see EOF and
        ``asyncio.gather`` on the drain tasks would hang indefinitely.
        Cap the wait, then cancel the still-pending drain tasks so the
        caller can return promptly with whatever buffered output we
        already collected.
        """
        try:
            await asyncio.wait_for(
                asyncio.gather(self._stdout_task, self._stderr_task, return_exceptions=True),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            for task in (self._stdout_task, self._stderr_task):
                if not task.done():
                    task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(self._stdout_task, self._stderr_task, return_exceptions=True),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                pass

    @property
    def stdout(self) -> str:
        return "".join(self._stdout_buf)

    @property
    def stderr(self) -> str:
        return "".join(self._stderr_buf)


__all__ = ["SubprocessSandbox"]
