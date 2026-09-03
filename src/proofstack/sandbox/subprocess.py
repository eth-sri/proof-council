"""Local-subprocess sandbox backend.

Per-invocation isolation:
- fresh ``tempfile.mkdtemp`` workdir;
- env stripped to the spec's allowlist plus declared provider keys;
- wallclock timeout enforced by the orchestrator;
- soft CPU/memory limits via ``setrlimit`` (best-effort);
- new POSIX session (``start_new_session=True``) so ordinary descendants
  stay in one process group and ``os.killpg`` can clean them up on teardown;
- a per-invocation environment marker to find descendants that create a new
  session. If the host process table cannot be inspected, cleanup reports an
  unknown state instead of claiming the worker stopped.
"""
from __future__ import annotations

import asyncio
import json
import os
import resource
import shlex
import signal
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import psutil  # type: ignore[import-untyped]

from proofstack.sandbox.base import (
    CommandResult,
    Sandbox,
    SandboxSpawnError,
    WorkerStopState,
)


STREAM_CAPTURE_MAX_CHARS = 16 * 1024 * 1024
USAGE_CAPTURE_MAX_CHARS = 16 * 1024 * 1024
USAGE_CAPTURE_MAX_LINE_CHARS = 2 * 1024 * 1024
PROCESS_GROUP_EXIT_TIMEOUT_S = 5.0
PROCESS_GROUP_EXIT_POLL_S = 0.05
PROCESS_MARKER_ENV = "PROOFSTACK_PROCESS_TOKEN"
PROCESS_MARKER_CLOCK_SKEW_S = 2.0
PROCESS_MARKER_TERM_GRACE_S = 1.0
PROCESS_MARKER_EXIT_TIMEOUT_S = 1.0
PROCESS_MARKER_EMPTY_SCANS = 3


@dataclass(frozen=True)
class _ProcessMarker:
    """Marker inherited by the descendants of one subprocess invocation."""

    token: str
    created_at: float


def _new_process_marker() -> _ProcessMarker:
    return _ProcessMarker(token=uuid.uuid4().hex, created_at=time.time())


def _find_marked_processes(
    marker: _ProcessMarker,
) -> tuple[list[psutil.Process], bool]:
    """Find same-user processes that inherited ``marker``.

    The subprocess backend is not a security boundary: a hostile child can
    remove the environment marker. It does, however, let cleanup find normal
    descendants that call ``setsid()`` and leave the original process group.
    Any process-table inspection gap makes the result incomplete so callers
    fail closed instead of claiming that the worker stopped.
    """
    try:
        current_uid = os.getuid()
        candidates = psutil.process_iter(
            attrs=["pid", "uids", "create_time"],
            ad_value=None,
        )
    except (AttributeError, OSError, psutil.Error):
        return [], False

    found: list[psutil.Process] = []
    complete = True
    try:
        for candidate in candidates:
            try:
                info = candidate.info
                if info.get("pid") == os.getpid():
                    continue
                uids = info.get("uids")
                if uids is None:
                    complete = False
                    continue
                if getattr(uids, "real", None) != current_uid:
                    continue
                created_at = info.get("create_time")
                if not isinstance(created_at, (int, float)):
                    complete = False
                    continue
                if created_at < marker.created_at - PROCESS_MARKER_CLOCK_SKEW_S:
                    continue
                environment = candidate.environ()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (OSError, psutil.AccessDenied, psutil.Error):
                complete = False
                continue
            if environment.get(PROCESS_MARKER_ENV) == marker.token:
                found.append(candidate)
    except (OSError, psutil.Error):
        return found, False
    return found, complete


def _signal_marked_processes(
    processes: Iterable[psutil.Process],
    *,
    kill: bool,
) -> None:
    for process in processes:
        try:
            process.kill() if kill else process.terminate()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except (OSError, psutil.AccessDenied, psutil.Error):
            # The verification scans below decide whether cleanup succeeded.
            continue


async def _terminate_marked_processes(marker: _ProcessMarker) -> bool:
    """Stop escaped descendants and require repeated complete empty scans."""
    processes, _ = _find_marked_processes(marker)
    _signal_marked_processes(processes, kill=False)

    term_deadline = time.monotonic() + PROCESS_MARKER_TERM_GRACE_S
    while processes and time.monotonic() < term_deadline:
        await asyncio.sleep(PROCESS_GROUP_EXIT_POLL_S)
        processes, _ = _find_marked_processes(marker)

    _signal_marked_processes(processes, kill=True)
    verify_deadline = time.monotonic() + PROCESS_MARKER_EXIT_TIMEOUT_S
    empty_scans = 0
    while True:
        processes, complete = _find_marked_processes(marker)
        if not complete or processes:
            empty_scans = 0
            if processes:
                _signal_marked_processes(processes, kill=True)
        else:
            empty_scans += 1
            if empty_scans >= PROCESS_MARKER_EMPTY_SCANS:
                return True
        remaining = verify_deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(PROCESS_GROUP_EXIT_POLL_S, remaining))


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


async def _process_group_stopped(
    pgid: int,
    *,
    timeout_s: float = PROCESS_GROUP_EXIT_TIMEOUT_S,
) -> bool:
    """Wait briefly until no process remains in ``pgid``."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except (PermissionError, OSError):
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(PROCESS_GROUP_EXIT_POLL_S, remaining))


async def _terminate_process_group(
    proc: asyncio.subprocess.Process,
    *,
    grace_s: float = 5.0,
    process_marker: _ProcessMarker | None = None,
) -> bool:
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
    marked_processes_stopped = (
        await _terminate_marked_processes(process_marker)
        if process_marker is not None
        else True
    )
    process_group_stopped = await _process_group_stopped(pid)
    return process_group_stopped and marked_processes_stopped


async def _terminate_process_group_uninterruptibly(
    proc: asyncio.subprocess.Process,
    *,
    grace_s: float = 5.0,
    process_marker: _ProcessMarker | None = None,
) -> bool:
    """Drain process-group cleanup despite cancellation of the caller."""
    cleanup = asyncio.create_task(
        _terminate_process_group(
            proc,
            grace_s=grace_s,
            process_marker=process_marker,
        )
    )
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    try:
        return bool(cleanup.result())
    except (asyncio.CancelledError, Exception):
        return False


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
        input_data: str | bytes | None = None,
    ) -> CommandResult:
        cwd_path = self.root / cwd if cwd else self.root
        env = self.spec.build_env(sandbox_root=self.root, extra_path=extra_path)
        if env_extra:
            env.update(env_extra)
        process_marker = _new_process_marker()
        env[PROCESS_MARKER_ENV] = process_marker.token
        timeout = timeout_s if timeout_s is not None else self.spec.timeout_s
        input_bytes = (
            input_data.encode("utf-8") if isinstance(input_data, str) else input_data
        )

        start = time.monotonic()
        self._mark_worker_launch_pending()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd_path),
                env=env,
                stdin=(asyncio.subprocess.PIPE if input_bytes is not None else None),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_make_preexec(self.spec.memory_gb, self.spec.cpu_limit, int(timeout)),
                start_new_session=True,
                pass_fds=self.inherited_fds,
            )
        except (OSError, ValueError) as e:
            self._set_worker_lifecycle(
                WorkerStopState.STOPPED,
                launch_settled=True,
            )
            return CommandResult(cmd=cmd, returncode=127, stdout="", stderr=str(e), duration_s=0.0)

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=input_bytes), timeout=timeout
            )
            returncode = proc.returncode if proc.returncode is not None else -1
            cleanup_succeeded = await _terminate_process_group(
                proc,
                grace_s=0.0,
                process_marker=process_marker,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            cleanup_succeeded = await _terminate_process_group_uninterruptibly(
                proc,
                grace_s=5.0,
                process_marker=process_marker,
            )
            self._set_worker_lifecycle(
                (
                    WorkerStopState.STOPPED
                    if cleanup_succeeded
                    else WorkerStopState.UNKNOWN
                ),
                launch_settled=True,
            )
            if isinstance(e, asyncio.CancelledError):
                raise
            if cleanup_succeeded:
                try:
                    stdout_b, stderr_b = await proc.communicate()
                except (ProcessLookupError, ValueError):
                    stdout_b = b""
                    stderr_b = b""
            else:
                stdout_b = b""
                stderr_b = b""
            returncode = -9
            if not stderr_b:
                stderr_b = f"timeout after {timeout}s".encode("utf-8")
        except BaseException:
            cleanup_succeeded = await _terminate_process_group_uninterruptibly(
                proc,
                grace_s=0.0,
                process_marker=process_marker,
            )
            self._set_worker_lifecycle(
                (
                    WorkerStopState.STOPPED
                    if cleanup_succeeded
                    else WorkerStopState.UNKNOWN
                ),
                launch_settled=True,
            )
            raise

        stop_state = (
            WorkerStopState.STOPPED
            if cleanup_succeeded
            else WorkerStopState.UNKNOWN
        )
        self._set_worker_lifecycle(stop_state, launch_settled=True)
        if stop_state is not WorkerStopState.STOPPED:
            suffix = b"subprocess process-tree stop state is unknown"
            stderr_b = stderr_b.rstrip()
            stderr_b += (b"\n" if stderr_b else b"") + suffix
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
        process_marker = _new_process_marker()
        env[PROCESS_MARKER_ENV] = process_marker.token
        timeout = timeout_s if timeout_s is not None else self.spec.timeout_s
        self._mark_worker_launch_pending()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd_path),
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_make_preexec(
                    self.spec.memory_gb,
                    self.spec.cpu_limit,
                    int(timeout),
                ),
                start_new_session=True,
                pass_fds=self.inherited_fds,
            )
        except (OSError, ValueError) as e:
            self._set_worker_lifecycle(
                WorkerStopState.STOPPED,
                launch_settled=True,
            )
            executable = cmd[0] if cmd else "<empty command>"
            raise SandboxSpawnError(
                f"could not spawn sandbox command {executable!r}: {e}"
            ) from e
        deadline = time.monotonic() + timeout
        self._set_worker_lifecycle(
            WorkerStopState.SURVIVING,
            launch_settled=True,
        )
        return _StreamingProcess(
            proc=proc,
            cmd=cmd,
            deadline=deadline,
            process_marker=process_marker,
            lifecycle_callback=self._set_worker_lifecycle,
        )


class _BoundedTextBuffer:
    """Retain a bounded, complete-line tail of an unbounded stream.

    Dropping at an arbitrary character boundary can retain only the suffix of a
    credential, which exact-value redaction cannot recognize. Once truncation
    occurs, discard through the next newline so persisted output never starts
    in the middle of a token or other logical record.
    """

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max(1, int(max_chars))
        self._chunks: deque[str] = deque()
        self.retained_chars = 0
        self.dropped_chars = 0
        self._discard_until_newline = False

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        if self._discard_until_newline:
            newline = chunk.find("\n")
            if newline < 0:
                self.dropped_chars += len(chunk)
                return
            discarded = newline + 1
            self.dropped_chars += discarded
            chunk = chunk[discarded:]
            self._discard_until_newline = False
            if not chunk:
                return
        self._chunks.append(chunk)
        self.retained_chars += len(chunk)
        trimmed = False
        cut_at_line_boundary = False
        while self.retained_chars > self.max_chars and self._chunks:
            overflow = self.retained_chars - self.max_chars
            head = self._chunks[0]
            if len(head) <= overflow:
                self._chunks.popleft()
                removed_text = head
                removed = len(head)
            else:
                removed_text = head[:overflow]
                self._chunks[0] = head[overflow:]
                removed = overflow
            self.retained_chars -= removed
            self.dropped_chars += removed
            trimmed = True
            cut_at_line_boundary = removed_text.endswith("\n")
        if trimmed and not cut_at_line_boundary:
            self._discard_partial_first_line()

    def _discard_partial_first_line(self) -> None:
        while self._chunks:
            head = self._chunks[0]
            newline = head.find("\n")
            if newline >= 0:
                removed = newline + 1
                if removed == len(head):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = head[removed:]
                self.retained_chars -= removed
                self.dropped_chars += removed
                return
            self._chunks.popleft()
            removed = len(head)
            self.retained_chars -= removed
            self.dropped_chars += removed
        self._discard_until_newline = True

    def text(self) -> str:
        return "".join(self._chunks)


class _JsonUsageCapture:
    """Retain compact JSONL usage records independently of output tails.

    Codex and Claude emit usage alongside potentially large transcript events.
    Keeping only a bounded transcript tail must not silently discard earlier
    billing records, so recognized events are reduced to the fields consumed
    by ``cli_usage`` and stored in a separate bounded buffer.
    """

    def __init__(
        self,
        *,
        max_chars: int = USAGE_CAPTURE_MAX_CHARS,
        max_line_chars: int = USAGE_CAPTURE_MAX_LINE_CHARS,
    ) -> None:
        self._records = _BoundedTextBuffer(max_chars)
        self._pending = ""
        self._discard_line = False
        self.max_line_chars = max(1, int(max_line_chars))
        self.events = 0
        self.oversized_lines = 0

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        pending = self._pending + chunk
        self._pending = ""
        while True:
            newline = pending.find("\n")
            if newline < 0:
                if self._discard_line:
                    return
                if len(pending) > self.max_line_chars:
                    self._discard_line = True
                    self.oversized_lines += 1
                    return
                self._pending = pending
                return
            line, pending = pending[:newline], pending[newline + 1 :]
            if self._discard_line:
                self._discard_line = False
                continue
            self._capture_line(line)

    def finish(self) -> None:
        if not self._discard_line and self._pending:
            self._capture_line(self._pending)
        self._pending = ""
        self._discard_line = False

    def _capture_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            return
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return

        compact: dict[str, object] | None = None
        event_type = event.get("type")
        usage = event.get("usage")
        if event_type == "turn.completed" and isinstance(usage, dict):
            compact = {"type": event_type, "usage": usage}
        elif event_type == "result" and isinstance(usage, dict):
            compact = {
                "type": event_type,
                "usage": usage,
                "num_turns": event.get("num_turns"),
                "total_cost_usd": event.get("total_cost_usd"),
            }
        elif event_type == "assistant":
            message = event.get("message")
            if isinstance(message, dict):
                message_usage = message.get("usage")
                if isinstance(message_usage, dict):
                    compact = {
                        "type": event_type,
                        "message": {
                            "id": message.get("id"),
                            "usage": message_usage,
                        },
                    }
        elif isinstance(usage, dict):
            compact = {
                "type": event_type,
                "usage": usage,
                "num_turns": event.get("num_turns"),
                "total_cost_usd": event.get("total_cost_usd"),
            }

        if compact is None:
            return
        rendered = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        self._records.append(rendered + "\n")
        self.events += 1

    def text(self) -> str:
        return self._records.text()

    @property
    def dropped_chars(self) -> int:
        return self._records.dropped_chars


class _StreamingProcess:
    def __init__(
        self,
        *,
        proc: asyncio.subprocess.Process,
        cmd: list[str],
        deadline: float,
        max_capture_chars: int = STREAM_CAPTURE_MAX_CHARS,
        process_marker: _ProcessMarker | None = None,
        lifecycle_callback: Callable[..., None] | None = None,
    ):
        self.proc = proc
        self.cmd = cmd
        self.deadline = deadline
        self._process_marker = process_marker
        self._stdout_buf = _BoundedTextBuffer(max_capture_chars)
        self._stderr_buf = _BoundedTextBuffer(max_capture_chars)
        self._process_group_stop_state = WorkerStopState.SURVIVING
        self._process_lifecycle_callback = lifecycle_callback
        self._stdout_usage = _JsonUsageCapture()
        self._stdout_task = asyncio.create_task(
            self._drain(proc.stdout, self._stdout_buf, self._stdout_usage)
        )
        self._stderr_task = asyncio.create_task(self._drain(proc.stderr, self._stderr_buf))

    @staticmethod
    async def _drain(
        stream,
        sink: _BoundedTextBuffer,
        usage_capture: _JsonUsageCapture | None = None,
    ) -> None:
        if stream is None:
            return
        try:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                sink.append(text)
                if usage_capture is not None:
                    usage_capture.feed(text)
        finally:
            if usage_capture is not None:
                usage_capture.finish()

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    @property
    def done(self) -> bool:
        return self.proc.returncode is not None

    @property
    def worker_stop_state(self) -> WorkerStopState:
        """What is known about the process tree protected by the lease."""
        return self.process_group_stop_state

    @property
    def process_group_stop_state(self) -> WorkerStopState:
        """Terminal-state knowledge for the host-side client process group."""
        if (
            self._process_group_stop_state is WorkerStopState.SURVIVING
            and self.done
        ):
            return WorkerStopState.UNKNOWN
        return self._process_group_stop_state

    @property
    def worker_stopped(self) -> bool:
        return self.worker_stop_state is WorkerStopState.STOPPED

    async def wait(self, timeout_s: float | None = None) -> int:
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return -1
        await self._drain_pipes(timeout_s=5.0)
        return self.proc.returncode or 0

    async def terminate(self) -> None:
        if self._process_group_stop_state is WorkerStopState.STOPPED:
            await self._drain_pipes(timeout_s=5.0)
            return
        cleanup_succeeded = await _terminate_process_group_uninterruptibly(
            self.proc,
            grace_s=5.0,
            process_marker=self._process_marker,
        )
        self._process_group_stop_state = (
            WorkerStopState.STOPPED
            if cleanup_succeeded
            else WorkerStopState.UNKNOWN
        )
        if self._process_lifecycle_callback is not None:
            self._process_lifecycle_callback(
                self._process_group_stop_state,
                launch_settled=True,
            )
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
        return self._stdout_buf.text()

    @property
    def stderr(self) -> str:
        return self._stderr_buf.text()

    @property
    def stdout_chars(self) -> int:
        return self._stdout_buf.retained_chars + self._stdout_buf.dropped_chars

    @property
    def stderr_chars(self) -> int:
        return self._stderr_buf.retained_chars + self._stderr_buf.dropped_chars

    @property
    def stdout_dropped_chars(self) -> int:
        return self._stdout_buf.dropped_chars

    @property
    def stderr_dropped_chars(self) -> int:
        return self._stderr_buf.dropped_chars

    @property
    def metering_stdout(self) -> str:
        if self._stdout_buf.dropped_chars and self._stdout_usage.events:
            return self._stdout_usage.text()
        return self.stdout

    @property
    def usage_events_captured(self) -> int:
        return self._stdout_usage.events

    @property
    def usage_capture_dropped_chars(self) -> int:
        return self._stdout_usage.dropped_chars

    @property
    def usage_capture_oversized_lines(self) -> int:
        return self._stdout_usage.oversized_lines


__all__ = ["SubprocessSandbox"]
