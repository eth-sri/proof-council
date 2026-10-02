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
from proofstack.sandbox.memory import MemoryLease, markers_rss


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
MEMORY_GAP_GRACE_S = 3.0


@dataclass(frozen=True)
class _ProcessMarker:
    """Marker inherited by the descendants of one subprocess invocation."""

    token: str
    created_at: float


def _new_process_marker() -> _ProcessMarker:
    return _ProcessMarker(token=uuid.uuid4().hex, created_at=time.time())


def _process_exited(process: psutil.Process) -> bool:
    try:
        return not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return True
    except (OSError, psutil.Error):
        return False


def _foreign_process(process: psutil.Process, current_uid: int,
                     markers: dict[str, _ProcessMarker]) -> bool:
    """True when a live same-uid process we may not inspect cannot be ours.

    Reading another process's environment needs ptrace access, which a
    user-namespace sandbox (``codex-linux-sandbox``, ``bwrap``), a setgid
    binary (``ssh-agent``, ``crontab``) or a privilege drop (``sshd-session``
    for a fresh login) all deny, even to the same user. Such a process is only
    an accounting gap for this worker when it may descend from the worker: it
    has a marked ancestor, or it was orphaned and adopted by pid 1 or by a
    same-uid daemon that may be a subreaper. A process whose parent is an
    ordinary live process outside our tree, or another user's process, is a
    bystander from a different session.
    """
    try:
        parent = process.parent()
        if parent is None:
            return True
        ancestor = parent
        while ancestor is not None:
            if ancestor.pid == os.getpid():
                return False
            # Shared admission also scans workers owned by sibling controllers.
            # Their marked descendants are not bystanders of this scan.
            if ancestor.uids().real == current_uid:
                marker = markers.get(ancestor.environ().get(PROCESS_MARKER_ENV))
                if marker is not None and ancestor.create_time() >= marker.created_at - PROCESS_MARKER_CLOCK_SKEW_S:
                    return False
            ancestor = ancestor.parent()
        if parent.pid == 1:
            return False
        if parent.uids().real != current_uid:
            return True
        return parent.ppid() != 1
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return _process_exited(process)
    except (AttributeError, OSError, psutil.Error):
        return False


def _best_effort(read):
    try:
        return read()
    except Exception:
        return None


def _scan_gap(process, error: str) -> dict:
    info = _best_effort(lambda: process.info) or {}
    return {
        "pid": _best_effort(lambda: process.pid) or info.get("pid"),
        "name": _best_effort(lambda: process.name()),
        "ppid": _best_effort(lambda: process.ppid()),
        "status": _best_effort(lambda: process.status()),
        "create_time": info.get("create_time") or _best_effort(lambda: process.create_time()),
        "error": error[:200],
    }


def _exc_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


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
    found, complete, _ = _find_processes_by_marker([marker])
    return found[marker.token], complete


def _find_processes_by_marker(
    markers: Iterable[_ProcessMarker],
) -> tuple[dict[str, list[psutil.Process]], bool, list[dict]]:
    """Inspect the process table once for all requested invocation markers.

    ``gaps`` describes each process that made the scan incomplete.
    """
    markers_by_token = {marker.token: marker for marker in markers}
    found: dict[str, list[psutil.Process]] = {token: [] for token in markers_by_token}
    gaps: list[dict] = []
    if not markers_by_token:
        return found, True, gaps
    earliest = min(marker.created_at for marker in markers_by_token.values())
    try:
        current_uid = os.getuid()
        candidates = psutil.process_iter(
            attrs=["pid", "uids", "create_time"],
            ad_value=None,
        )
    except (AttributeError, OSError, psutil.Error) as exc:
        gaps.append(_scan_gap(None, _exc_text(exc)))
        return found, False, gaps

    complete = True
    try:
        for candidate in candidates:
            try:
                info = candidate.info
                if info.get("pid") == os.getpid():
                    continue
                uids = info.get("uids")
                if uids is None:
                    if not _process_exited(candidate):
                        complete = False
                        gaps.append(_scan_gap(candidate, "uids unavailable"))
                    continue
                if getattr(uids, "real", None) != current_uid:
                    continue
                created_at = info.get("create_time")
                if not isinstance(created_at, (int, float)):
                    if not _process_exited(candidate):
                        complete = False
                        gaps.append(_scan_gap(candidate, "create_time unavailable"))
                    continue
                if created_at < earliest - PROCESS_MARKER_CLOCK_SKEW_S:
                    continue
                environment = candidate.environ()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (OSError, psutil.AccessDenied, psutil.Error) as exc:
                # process_iter's cached attributes can outlive a process. A
                # disappearing sibling is not an accounting gap for this worker.
                if _process_exited(candidate):
                    continue
                if isinstance(exc, psutil.AccessDenied) and _foreign_process(candidate, current_uid, markers_by_token):
                    continue
                complete = False
                gaps.append(_scan_gap(candidate, _exc_text(exc)))
                continue
            marker = markers_by_token.get(environment.get(PROCESS_MARKER_ENV))
            if marker is not None and created_at >= marker.created_at - PROCESS_MARKER_CLOCK_SKEW_S:
                found[marker.token].append(candidate)
    except (OSError, psutil.Error) as exc:
        gaps.append(_scan_gap(None, _exc_text(exc)))
        return found, False, gaps
    return found, complete, gaps


async def _memory_operation(operation):
    # Cancellation cannot stop a thread. Settle admission before its caller
    # closes the lease, otherwise the thread could acquire a slot after cleanup.
    task = asyncio.create_task(asyncio.to_thread(operation))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


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


def _make_preexec(memory_gb: int, cpu_limit: int, cpu_seconds: int, *, limit_address_space: bool = True):
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
            if limit_address_space:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
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
                preexec_fn=_make_preexec(self.spec.memory_gb, self.spec.cpu_limit, int(timeout), limit_address_space=self.spec.limit_address_space),
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
        queue_timeout_s: float | None = None,
        wallclock_deadline: float | None = None,
    ) -> "_StreamingProcess":
        """Spawn a long-running command and return a handle.

        Used by CLIAgent so the orchestrator can poll for ``done.json``
        and emit ``cli.heartbeat`` events without blocking on the child.
        ``stdin`` is piped so CLIAgent can write the prompt to it; the
        DockerSandbox equivalent does the same. Without this, codex
        inherits the parent's stdin, sees EOF immediately, and exits
        with code 1 before doing any work.

        Queue waiting is bounded separately (by ``timeout_s`` unless overridden).
        ``wallclock_deadline`` caps both admission and execution in monotonic time.
        """
        cwd_path = self.root / cwd if cwd else self.root
        env = self.spec.build_env(sandbox_root=self.root, extra_path=extra_path)
        if env_extra:
            env.update(env_extra)
        process_marker = _new_process_marker()
        env[PROCESS_MARKER_ENV] = process_marker.token
        timeout = timeout_s if timeout_s is not None else self.spec.timeout_s
        memory_lease = None
        queue_deadline = time.monotonic() + (
            timeout if queue_timeout_s is None else queue_timeout_s
        )
        if wallclock_deadline is not None:
            queue_deadline = min(queue_deadline, wallclock_deadline)
        emit_memory = getattr(self, "emit_memory_event", None)
        if self.spec.memory_policy is not None:
            memory_lease = MemoryLease(self.spec.memory_policy, process_marker)
            queued = False
            try:
                while True:
                    if time.monotonic() >= queue_deadline:
                        raise SandboxSpawnError("timed out waiting for a shared Compute slot")
                    acquired = await _memory_operation(memory_lease.try_acquire)
                    if emit_memory is not None and (event := memory_lease.quarantine_event()) is not None:
                        await emit_memory("cli.memory_registry_quarantine", event)
                    if acquired:
                        break
                    if not queued and emit_memory is not None:
                        await emit_memory("cli.memory_queued", {
                            "max_workers": self.spec.memory_policy.max_workers,
                            "worker_limit_bytes": self.spec.memory_policy.worker_bytes,
                            "queue_timeout_s": timeout if queue_timeout_s is None else queue_timeout_s,
                        })
                    queued = True
                    await asyncio.sleep(min(1.0, max(0.0, queue_deadline - time.monotonic())))
                if emit_memory is not None:
                    await emit_memory("cli.memory_admitted", {
                        "max_workers": self.spec.memory_policy.max_workers,
                        "worker_limit_bytes": self.spec.memory_policy.worker_bytes,
                        "reserve_bytes": self.spec.memory_policy.reserve_bytes,
                    })
                if time.monotonic() >= queue_deadline:
                    raise SandboxSpawnError("Compute deadline expired before worker launch")
            except BaseException:
                memory_lease.close()
                raise
        started_at = time.monotonic()
        deadline = started_at + timeout
        if wallclock_deadline is not None:
            deadline = min(deadline, wallclock_deadline)
        if deadline <= started_at:
            if memory_lease is not None:
                memory_lease.close()
            raise SandboxSpawnError("Compute deadline expired before worker launch")
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
                    int(deadline - started_at),
                    limit_address_space=self.spec.limit_address_space,
                ),
                start_new_session=True,
                pass_fds=self.inherited_fds + (
                    (memory_lease.fd,) if memory_lease is not None else ()
                ),
            )
        except (OSError, ValueError) as e:
            if memory_lease is not None:
                memory_lease.close()
            self._set_worker_lifecycle(
                WorkerStopState.STOPPED,
                launch_settled=True,
            )
            executable = cmd[0] if cmd else "<empty command>"
            raise SandboxSpawnError(
                f"could not spawn sandbox command {executable!r}: {e}"
            ) from e
        except BaseException:
            if memory_lease is not None:
                memory_lease.close()
            raise
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
            memory_lease=memory_lease,
            resident_limit_bytes=(self.spec.memory_gb * 1024**3 if not self.spec.limit_address_space else None),
            emit_memory=emit_memory,
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
        elif event_type == "result":
            compact = {
                "type": event_type,
                "usage": usage,
                "num_turns": event.get("num_turns"),
                "total_cost_usd": event.get("total_cost_usd"),
                "modelUsage": event.get("modelUsage"),
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
                "modelUsage": event.get("modelUsage"),
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
        memory_lease: MemoryLease | None = None,
        resident_limit_bytes: int | None = None,
        emit_memory=None,
    ):
        self.proc = proc
        self.cmd = cmd
        self.deadline = deadline
        self._process_marker = process_marker
        self._stdout_buf = _BoundedTextBuffer(max_capture_chars)
        self._stderr_buf = _BoundedTextBuffer(max_capture_chars)
        self._process_group_stop_state = WorkerStopState.SURVIVING
        self._process_lifecycle_callback = lifecycle_callback
        self._memory_lease = memory_lease
        self._resident_limit_bytes = resident_limit_bytes
        self._emit_memory = emit_memory
        self.memory_failure: str | None = None
        self.memory_sample: dict = {}
        self._terminate_lock = asyncio.Lock()
        self._stdout_usage = _JsonUsageCapture()
        self._stdout_task = asyncio.create_task(
            self._drain(proc.stdout, self._stdout_buf, self._stdout_usage)
        )
        self._stderr_task = asyncio.create_task(self._drain(proc.stderr, self._stderr_buf))
        self._memory_task = (
            asyncio.create_task(self._watch_memory()) if memory_lease is not None or resident_limit_bytes is not None else None
        )

    async def _watch_memory(self) -> None:
        last_event = 0.0
        last_gap_event = 0.0
        gap_since: float | None = None
        try:
            while self._process_group_stop_state is not WorkerStopState.STOPPED:
                try:
                    if self._memory_lease is not None:
                        self.memory_sample = await _memory_operation(self._memory_lease.sample)
                    else:
                        marker = self._process_marker
                        usage = await _memory_operation(lambda: markers_rss([
                            {"token": marker.token, "created_at": marker.created_at}
                        ]))
                        rss = usage[marker.token][0]
                        self.memory_sample = {
                            "rss_bytes": rss,
                            "worker_limit_bytes": self._resident_limit_bytes,
                            "reason": "worker_memory_limit" if rss > self._resident_limit_bytes else None,
                        }
                    reason = self.memory_sample["reason"]
                    gap_since = None
                except Exception as exc:
                    reason = "memory_accounting_unavailable"
                    self.memory_sample = {
                        "reason": reason, "error_type": type(exc).__name__, "error": str(exc)[:600],
                    }
                    now = time.monotonic()
                    if gap_since is None:
                        gap_since = now
                    # A process hidden for under MEMORY_GAP_GRACE_S cannot have
                    # grown far past the last good sample, while a spurious kill
                    # wastes an hour-long paid editor.
                    if now - gap_since < MEMORY_GAP_GRACE_S:
                        if self._emit_memory is not None and now - last_gap_event >= 5:
                            await self._emit_memory("cli.memory_accounting_gap", self.memory_sample)
                            last_gap_event = now
                        await asyncio.sleep(self._memory_lease.policy.poll_seconds if self._memory_lease is not None else 1.0)
                        continue
                if reason:
                    self.memory_failure = reason
                    # Kill before logging and without the normal five-second
                    # grace; an allocating child can consume that headroom fast.
                    await self.terminate()
                if (self._emit_memory is not None and self._memory_lease is not None
                        and self.memory_sample.get("unreadable_slots")):
                    event = self._memory_lease.quarantine_event()
                    if event is not None:
                        await self._emit_memory("cli.memory_registry_quarantine", event)
                if reason:
                    if self._emit_memory is not None:
                        await self._emit_memory("cli.memory_limit_exceeded", self.memory_sample)
                    return
                if self._emit_memory is not None and time.monotonic() - last_event >= 30:
                    await self._emit_memory("cli.memory_usage", self.memory_sample)
                    last_event = time.monotonic()
                await asyncio.sleep(self._memory_lease.policy.poll_seconds if self._memory_lease is not None else 1.0)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.memory_failure = "memory_monitor_failed"
            await self.terminate()

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
        async with self._terminate_lock:
            await self._terminate_locked()

    async def _terminate_locked(self) -> None:
        if self._process_group_stop_state is WorkerStopState.STOPPED:
            await self._drain_pipes(timeout_s=5.0)
            return
        cleanup_succeeded = await _terminate_process_group_uninterruptibly(
            self.proc,
            grace_s=0.0 if self.memory_failure else 5.0,
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
        if self._memory_task is not None and self._memory_task is not asyncio.current_task():
            self._memory_task.cancel()
            # Do not await while holding the terminate lock: the monitor may
            # itself be waiting for that lock to enforce a pressure stop.
        if self._memory_lease is not None:
            self._memory_lease.close()
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
