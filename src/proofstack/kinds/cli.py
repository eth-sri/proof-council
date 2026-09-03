"""CLIAgent — drive an external coding CLI with the ``finish`` stop signal."""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from proofstack.agent import Agent
from proofstack.budget import BudgetExhausted
from proofstack.context import RunContext
from proofstack.events import new_call_id
from proofstack.sandbox import make_sandbox, resolve_backend
from proofstack.sandbox.base import (
    Sandbox,
    SandboxSpawnError,
    SandboxSpec,
    WorkerStopState,
)
from proofstack.storage_reservation import (
    StorageReservationError,
    StorageReservationLease,
)


FINISH_SCRIPT = """\
#!/bin/sh
# finish — active stop signal for proofstack CLIAgent runs.
# Writes a `done.json` to $FINISH_DONE_PATH and exits 0 so the
# orchestrator knows the model is finished.
set -eu
TARGET="${FINISH_DONE_PATH:-${PWD}/done.json}"
if [ "${1:-}" != "" ]; then
    if [ -f "$1" ]; then
        cp "$1" "$TARGET"
    else
        printf '%s' "$1" > "$TARGET"
    fi
elif [ ! -t 0 ]; then
    cat > "$TARGET"
else
    printf '{"status": "done", "summary": "(no body supplied)"}' > "$TARGET"
fi
echo "finish: wrote $TARGET" >&2
exit 0
"""
_SHELL_START_BLOCK_BEGIN = "# proofstack finish shim begin"
_SHELL_START_BLOCK_END = "# proofstack finish shim end"
_SENSITIVE_STATE_QUARANTINE = "SENSITIVE_STATE_UNTRUSTED"
_WORKSPACE_RECOVERY_STATE_DIR = ".proofcouncil-workspace-recovery"
_WORKSPACE_QUARANTINE_DIR = ".proofcouncil-untrusted-workspaces"
_WORKSPACE_ACTIVE_GUARD_VERSION = 2
_WORKSPACE_ACTIVE_GUARD_MAX_BYTES = 4096
_WORKSPACE_GUARD_IDLE = "idle"
_WORKSPACE_GUARD_LAUNCH_PENDING = "launch_pending"
_WORKSPACE_GUARD_LAUNCH_SETTLED = "launch_settled"
_WORKSPACE_GUARD_PHASES = {
    _WORKSPACE_GUARD_IDLE,
    _WORKSPACE_GUARD_LAUNCH_PENDING,
    _WORKSPACE_GUARD_LAUNCH_SETTLED,
}


DoneStatus = Literal["done", "partial", "blocked", "timeout", "error"]


@dataclass(frozen=True)
class _WorkspaceRecoveryClaim:
    workspace_root: Path
    attempts: int
    lock_fd: int


@dataclass(frozen=True)
class _GuardedWorkspaceRecovery:
    action: Literal["cleared", "quarantined"]
    backend: str
    quarantine_path: Path | None = None


class _WorkspaceLockBusy(RuntimeError):
    """Another process currently holds a lock for this workspace."""


class _WorkerStopUnconfirmed(RuntimeError):
    """The external worker may still be able to mutate its workspace."""

    def __init__(self, state: WorkerStopState, *, phase: str) -> None:
        self.state = state
        self.phase = phase
        super().__init__(
            "CLI worker could not be confirmed stopped "
            f"(state={state.value}, phase={phase}); response and handoff "
            "artifacts were suppressed. Workspace availability will be "
            "rechecked before the next invocation, and uncertain prior data "
            "will be preserved in quarantine rather than deleted"
        )


def _reported_worker_stop_state(worker: Any) -> WorkerStopState | None:
    raw_state = getattr(worker, "worker_stop_state", None)
    if isinstance(raw_state, WorkerStopState):
        return raw_state
    if isinstance(raw_state, str):
        try:
            return WorkerStopState(raw_state)
        except ValueError:
            pass
    return None


def _worker_stop_state(stream: Any) -> WorkerStopState:
    """Normalize old and backend-specific streaming handle state."""
    if (reported := _reported_worker_stop_state(stream)) is not None:
        return reported
    stopped = bool(
        getattr(stream, "worker_stopped", getattr(stream, "done", False))
    )
    return WorkerStopState.STOPPED if stopped else WorkerStopState.SURVIVING


def _sandbox_worker_stop_state(sandbox: Any) -> WorkerStopState:
    """Return backend lifecycle state, defaulting legacy sandboxes to idle."""
    state = _reported_worker_stop_state(sandbox) or WorkerStopState.STOPPED
    if state is WorkerStopState.STOPPED and not bool(
        getattr(sandbox, "worker_launch_settled", True)
    ):
        return WorkerStopState.UNKNOWN
    return state


def _require_worker_stopped(stream: Any, *, phase: str) -> None:
    state = _worker_stop_state(stream)
    if state is not WorkerStopState.STOPPED:
        raise _WorkerStopUnconfirmed(state, phase=phase)


async def _release_storage_lease(lease: StorageReservationLease) -> None:
    """Release a filesystem lease without blocking the shared event loop."""
    await asyncio.to_thread(lease.release)


async def _retain_storage_lease(lease: StorageReservationLease) -> str | None:
    """Persist a filesystem lease for a worker whose stop is unconfirmed."""
    return await _run_in_thread_uninterruptibly(lease.retain)


async def _run_in_thread_uninterruptibly(
    func: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Finish a safety-critical thread operation before propagating cancel."""
    task = asyncio.ensure_future(asyncio.to_thread(func, *args, **kwargs))
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as e:
            cancellation = e
            continue
    result = task.result()
    if cancellation is not None:
        raise cancellation
    return result


_SENSITIVE_STATE_QUARANTINE_MESSAGE = (
    "Credential refresh failed after an external CLI ran. Remove this marker "
    "only after discarding or manually sanitizing the workspace.\n"
)


def _path_exists_without_following(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        # An unreadable marker location is not safe to treat as clean.
        return True
    return True


def _write_sensitive_quarantine_marker(path: Path) -> bool:
    """Create one marker without following a model-created symlink."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        return _path_exists_without_following(path)
    except OSError:
        return False
    try:
        os.write(fd, _SENSITIVE_STATE_QUARANTINE_MESSAGE.encode("utf-8"))
        os.fsync(fd)
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass
        return False
    finally:
        os.close(fd)
    return True


def _mark_sensitive_workspace_untrusted(
    workspace_marker: Path,
    external_marker: Path,
    *,
    recovery_sources: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """Persist a quarantine marker in at least one independent location.

    The external sidecar survives deletion or corruption of the model-writable
    runtime directory. If that filesystem is completely full, renaming an
    existing runtime record gives the workspace marker an allocation-free
    fallback on filesystems where rename remains possible.
    """
    persisted: list[Path] = []
    for marker in (workspace_marker, external_marker):
        if _write_sensitive_quarantine_marker(marker):
            persisted.append(marker)

    if workspace_marker not in persisted:
        for source in recovery_sources:
            try:
                metadata = source.lstat()
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                os.replace(source, workspace_marker)
            except OSError:
                continue
            persisted.append(workspace_marker)
            break

    if not persisted:
        raise RuntimeError(
            "CLI credential state is untrusted and its quarantine marker "
            "could not be persisted"
        )
    return tuple(persisted)


def _workspace_active_guard_payload(
    root: Path,
    *,
    backend: str,
    phase: str,
    created_at: float | None = None,
) -> bytes:
    if phase not in _WORKSPACE_GUARD_PHASES:
        raise ValueError(f"invalid workspace active guard phase: {phase}")
    resolved = Path(root).resolve()
    metadata = resolved.stat()
    payload = json.dumps(
        {
            "version": _WORKSPACE_ACTIVE_GUARD_VERSION,
            "workspace": str(resolved),
            "workspace_identity": [int(metadata.st_dev), int(metadata.st_ino)],
            "backend": str(backend),
            "phase": phase,
            "created_at": time.time() if created_at is None else created_at,
            "updated_at": time.time(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > _WORKSPACE_ACTIVE_GUARD_MAX_BYTES:
        raise RuntimeError("workspace active guard payload is unexpectedly large")
    return payload


def _write_fd_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("could not write workspace active guard")
        offset += written
    os.fsync(fd)


def _write_workspace_active_guard(
    path: Path,
    root: Path,
    *,
    backend: str = "unknown",
    phase: str = _WORKSPACE_GUARD_LAUNCH_SETTLED,
) -> None:
    """Durably mark a workspace as unsafe until its worker is sanitized."""
    payload = _workspace_active_guard_payload(
        root,
        backend=backend,
        phase=phase,
    )
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as e:
        raise RuntimeError(
            f"workspace active guard already exists: {path}"
        ) from e
    try:
        _write_fd_all(fd, payload)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(fd)


def _validate_workspace_active_guard(path: Path, root: Path) -> dict[str, Any]:
    """Verify that a bounded, non-symlink guard belongs to this workspace."""
    try:
        metadata = path.lstat()
    except OSError as e:
        raise RuntimeError(f"cannot read workspace active guard: {path}") from e
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > _WORKSPACE_ACTIVE_GUARD_MAX_BYTES
    ):
        raise RuntimeError(f"workspace active guard is invalid: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
        try:
            raw = os.read(fd, _WORKSPACE_ACTIVE_GUARD_MAX_BYTES + 1)
        finally:
            os.close(fd)
        payload = json.loads(raw.decode("utf-8"))
        resolved = Path(root).resolve()
        root_metadata = resolved.stat()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RuntimeError(f"workspace active guard is unreadable: {path}") from e
    expected_identity = [int(root_metadata.st_dev), int(root_metadata.st_ino)]
    version = payload.get("version") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or version not in {1, _WORKSPACE_ACTIVE_GUARD_VERSION}
        or payload.get("workspace") != str(resolved)
        or payload.get("workspace_identity") != expected_identity
    ):
        raise RuntimeError(
            "workspace active guard does not match the current workspace; "
            f"refusing automatic recovery: {path}"
        )
    if version == 1:
        # Version 1 did not record whether a Docker create request had settled.
        # Treat it as pending so an upgrade cannot turn heuristic absence into
        # permission to quarantine or reuse a live workspace.
        payload = {
            **payload,
            "backend": "unknown",
            "phase": _WORKSPACE_GUARD_LAUNCH_PENDING,
        }
    if (
        not isinstance(payload.get("backend"), str)
        or payload.get("phase") not in _WORKSPACE_GUARD_PHASES
    ):
        raise RuntimeError(f"workspace active guard has invalid lifecycle data: {path}")
    return payload


def _update_workspace_active_guard(
    path: Path,
    root: Path,
    *,
    phase: str,
) -> None:
    """Atomically persist a lifecycle transition for an existing guard."""
    current = _validate_workspace_active_guard(path, root)
    payload = _workspace_active_guard_payload(
        root,
        backend=str(current["backend"]),
        phase=phase,
        created_at=float(current.get("created_at", time.time())),
    )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        _write_fd_all(fd, payload)
        os.close(fd)
        fd = -1
        os.replace(temporary, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _quarantine_workspace(root: Path) -> Path:
    """Atomically preserve an uncertain workspace and recreate its root.

    The quarantine lives under a private sibling directory, so Docker workers
    mounting only ``root`` cannot inspect it. The subprocess backend remains a
    trusted-host backend and does not provide a filesystem security boundary.
    """
    resolved = Path(root).resolve()
    metadata = resolved.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or resolved == resolved.parent:
        raise RuntimeError(f"refusing to quarantine unsafe workspace path: {resolved}")

    quarantine_parent = resolved.parent / _WORKSPACE_QUARANTINE_DIR
    quarantine_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_metadata = quarantine_parent.lstat()
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or getattr(parent_metadata, "st_uid", os.getuid()) != os.getuid()
    ):
        raise RuntimeError(
            "workspace quarantine parent is not a private owned directory: "
            f"{quarantine_parent}"
        )
    try:
        quarantine_parent.chmod(0o700)
    except OSError as e:
        raise RuntimeError(
            f"cannot secure workspace quarantine parent: {quarantine_parent}"
        ) from e

    digest = hashlib.sha256(os.fsencode(str(resolved))).hexdigest()[:16]
    quarantine_container = Path(
        tempfile.mkdtemp(prefix=f"{digest}-", dir=quarantine_parent)
    )
    quarantined = quarantine_container / "workspace"
    try:
        os.rename(resolved, quarantined)
    except OSError as e:
        try:
            quarantine_container.rmdir()
        except OSError:
            pass
        raise RuntimeError(f"could not quarantine workspace: {resolved}") from e

    try:
        resolved.mkdir(mode=stat.S_IMODE(metadata.st_mode))
    except OSError as create_error:
        try:
            os.rename(quarantined, resolved)
        except OSError as rollback_error:
            raise RuntimeError(
                "workspace was preserved in quarantine but its active path could "
                f"not be recreated or restored: {quarantined}"
            ) from rollback_error
        try:
            quarantine_container.rmdir()
        except OSError:
            pass
        raise RuntimeError(
            f"could not recreate quarantined workspace root: {resolved}"
        ) from create_error
    return quarantined


def _restore_quarantined_workspace(root: Path, quarantined: Path) -> None:
    """Roll back a quarantine transaction before its guard is cleared."""
    resolved = Path(root).resolve()
    try:
        resolved.rmdir()
        os.rename(quarantined, resolved)
    except OSError as e:
        raise RuntimeError(
            "workspace quarantine could not be committed or rolled back; "
            f"preserved data remains at {quarantined}"
        ) from e
    try:
        quarantined.parent.rmdir()
    except OSError:
        pass


def _clear_workspace_active_guard(path: Path) -> None:
    """Remove a guard without following a substituted path."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError as e:
        raise RuntimeError(f"cannot inspect workspace active guard: {path}") from e
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(f"workspace active guard is invalid: {path}")
    try:
        path.unlink()
    except OSError as e:
        raise RuntimeError(f"cannot clear workspace active guard: {path}") from e


def measure_workspace_usage(
    root: Path,
    *,
    largest_count: int = 10,
    stop_after_bytes: int = 0,
    stop_after_entries: int = 0,
) -> dict[str, Any]:
    """Measure workspace storage without reading contents or following links.

    Optional stop thresholds bound the monitor's own work when a runaway CLI
    creates an extreme number of entries. Reaching either threshold is enough
    for callers enforcing the corresponding hard limit.
    """
    root = Path(root)
    total_bytes = 0
    allocated_bytes = 0
    errors = 0
    transient_errors = 0
    try:
        allocated_bytes_supported = hasattr(root.stat(), "st_blocks")
    except OSError:
        allocated_bytes_supported = False
        errors += 1
    files = 0
    directories = 0
    broken_symlinks = 0
    files_over_100_mib = 0
    files_over_1_gib = 0
    largest: list[tuple[int, str]] = []
    scan_truncated = False
    stack = [Path(root)]
    while stack and not scan_truncated:
        directory = stack.pop()
        try:
            entries = os.scandir(directory)
        except (FileNotFoundError, NotADirectoryError):
            # A worker can remove or replace a directory while the monitor is
            # walking it. This makes the snapshot approximate, but it is not a
            # safety-scan failure worth killing the worker for.
            transient_errors += 1
            continue
        except OSError:
            errors += 1
            continue
        with entries:
            iterator = iter(entries)
            while True:
                try:
                    entry = next(iterator)
                except StopIteration:
                    break
                except (FileNotFoundError, NotADirectoryError):
                    transient_errors += 1
                    break
                except OSError:
                    errors += 1
                    break
                try:
                    if entry.is_dir(follow_symlinks=False):
                        directories += 1
                        if allocated_bytes_supported:
                            try:
                                directory_stat = entry.stat(follow_symlinks=False)
                            except (FileNotFoundError, NotADirectoryError):
                                transient_errors += 1
                            except OSError:
                                errors += 1
                            else:
                                allocated_bytes += (
                                    max(0, int(directory_stat.st_blocks)) * 512
                                )
                        stack.append(Path(entry.path))
                        if (
                            stop_after_entries > 0
                            and files + directories >= stop_after_entries
                        ):
                            scan_truncated = True
                            stack.clear()
                            break
                        continue
                    stat_result = entry.stat(follow_symlinks=False)
                except (FileNotFoundError, NotADirectoryError):
                    transient_errors += 1
                    continue
                except OSError:
                    errors += 1
                    continue
                files += 1
                size = max(0, stat_result.st_size)
                total_bytes += size
                if allocated_bytes_supported:
                    allocated_bytes += max(0, int(stat_result.st_blocks)) * 512
                if size >= 100 * 1024 * 1024:
                    files_over_100_mib += 1
                if size >= 1024 * 1024 * 1024:
                    files_over_1_gib += 1
                path = Path(entry.path)
                if entry.is_symlink() and not path.exists():
                    broken_symlinks += 1
                if largest_count > 0:
                    try:
                        relative = path.relative_to(root).as_posix()
                    except ValueError:
                        relative = path.name
                    largest.append((size, relative))
                    largest.sort(reverse=True)
                    del largest[largest_count:]
                if (
                    (
                        stop_after_bytes > 0
                        and (
                            allocated_bytes
                            if allocated_bytes_supported
                            else total_bytes
                        )
                        >= stop_after_bytes
                    )
                    or (
                        stop_after_entries > 0
                        and files + directories >= stop_after_entries
                    )
                ):
                    scan_truncated = True
                    stack.clear()
                    break
    try:
        filesystem_free_bytes = shutil.disk_usage(root).free
    except OSError:
        filesystem_free_bytes = None
    try:
        filesystem_stats = os.statvfs(root)
        filesystem_free_inodes = (
            filesystem_stats.f_favail if filesystem_stats.f_files > 0 else None
        )
    except (AttributeError, OSError):
        filesystem_free_inodes = None
    return {
        "bytes": total_bytes,
        "allocated_bytes": allocated_bytes,
        "allocated_bytes_supported": allocated_bytes_supported,
        "limit_bytes": (
            allocated_bytes if allocated_bytes_supported else total_bytes
        ),
        "files": files,
        "directories": directories,
        "entries": files + directories,
        "scan_truncated": scan_truncated,
        "errors": errors,
        "transient_errors": transient_errors,
        "broken_symlinks": broken_symlinks,
        "files_over_100_mib": files_over_100_mib,
        "files_over_1_gib": files_over_1_gib,
        "filesystem_free_bytes": filesystem_free_bytes,
        "filesystem_free_inodes": filesystem_free_inodes,
        "largest_files": [
            {"path": path, "bytes": size} for size, path in largest
        ],
    }


class CLIDoneRecord(BaseModel):
    """Schema of the ``done.json`` written by ``finish``."""

    status: DoneStatus = "done"
    summary: str = ""
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    diff_summary: str = ""


class CLIAgent(Agent):
    """Base class for agents that drive an external CLI tool.

    Subclasses set:
      - ``CLI_CMD``:        the command to invoke (e.g. ``["codex", "-q"]``).
      - ``SANDBOX``:        a ``SandboxSpec`` (sane defaults below).

    They also override ``setup`` (to write files into the sandbox) and
    ``collect`` (to harvest outputs after the CLI has exited). Override
    ``cli_input`` to write to the CLI's stdin.
    """

    description: ClassVar[str] = "Drive an external CLI tool in a sandbox."
    execution_mode: ClassVar[str] = "agent"

    CLI_CMD: ClassVar[list[str]] = []
    SANDBOX: ClassVar[SandboxSpec] = SandboxSpec()
    HEARTBEAT_INTERVAL_S: ClassVar[float] = 30.0
    POLL_INTERVAL_S: ClassVar[float] = 1.0
    CLEANUP_GRACE_S: ClassVar[float] = 30.0
    DONE_DRAIN_GRACE_S: ClassVar[float] = 30.0
    SOFT_TIMEOUT_S: ClassVar[int] = 0
    WORKSPACE_SOFT_LIMIT_BYTES: ClassVar[int] = 0
    WORKSPACE_HARD_LIMIT_BYTES: ClassVar[int] = 0
    WORKSPACE_SOFT_LIMIT_ENTRIES: ClassVar[int] = 0
    WORKSPACE_HARD_LIMIT_ENTRIES: ClassVar[int] = 0
    WORKSPACE_MIN_FREE_BYTES: ClassVar[int] = 0
    WORKSPACE_MIN_FREE_INODES: ClassVar[int] = 0
    WORKSPACE_REQUIRE_INODE_ACCOUNTING: ClassVar[bool] = False
    WORKSPACE_RESERVATION_BYTES: ClassVar[int] = 0
    WORKSPACE_RESERVATION_DIR: ClassVar[Path | None] = None
    WORKSPACE_CHECK_INTERVAL_S: ClassVar[float] = 60.0
    WORKSPACE_USAGE_EVENT_INTERVAL_S: ClassVar[float] = 300.0
    WORKSPACE_USAGE_EVENT_MIN_BYTES_DELTA: ClassVar[int] = 1024 * 1024 * 1024
    WORKSPACE_USAGE_EVENT_MIN_ENTRIES_DELTA: ClassVar[int] = 1_000
    WORKSPACE_RECOVERY_ENABLED: ClassVar[bool] = False
    WORKSPACE_RECOVERY_MAX_ATTEMPTS: ClassVar[int] = 1
    WORKSPACE_RECOVERY_GROWTH_BYTES: ClassVar[int] = 1024 * 1024 * 1024
    WORKSPACE_RECOVERY_GROWTH_ENTRIES: ClassVar[int] = 1_000
    WORKSPACE_PRESSURE_FILENAME: ClassVar[str] = "STORAGE_PRESSURE"
    # 'finish'/'file': the completion record (done.json) is REQUIRED — a CLI
    # exit without it (or with an unparseable one) is an error even on rc 0.
    # 'exit': the exit code is the completion signal (codex-style CLIs that
    # never call finish). Plain subclasses keep the legacy 'exit' semantics;
    # ConfigurableCLIAgent sets this from its completion_signal config.
    COMPLETION_SIGNAL: ClassVar[str] = "exit"
    MISSING_COMPLETION_STATUS: ClassVar[DoneStatus] = "error"

    def __init__(
        self,
        ctx: RunContext,
        *,
        sandbox_root: Path | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(ctx, **kw)
        self.sandbox_root = Path(sandbox_root) if sandbox_root is not None else None
        self._workspace_recovery_mode = False
        self._workspace_recovery_max_bytes: int | None = None
        self._workspace_recovery_max_entries: int | None = None
        self._workspace_recovery_attempts = 0

    # --- subclass hooks --------------------------------------------------------

    async def setup(self, sandbox: Sandbox, inp: BaseModel) -> None:
        """Write input files (e.g. ``main.tex``) into the sandbox."""

    async def collect(
        self,
        sandbox: Sandbox,
        inp: BaseModel,
        done: CLIDoneRecord,
    ) -> BaseModel:
        """Harvest outputs from the sandbox after CLI exit."""
        raise NotImplementedError

    def attach_workspace_recovery_notice(
        self,
        out: BaseModel,
        notice: str,
    ) -> BaseModel:
        """Let persistent agents expose a workspace replacement to consumers."""
        return out

    async def teardown(self, sandbox: Sandbox, inp: BaseModel) -> None:
        """Scrub per-invocation secrets from the sandbox workdir.

        Called from ``run()``'s finally block. The sandbox dir itself is
        kept on disk for artifact capture, so anything sensitive written
        by ``setup()`` (credentials, session tokens) must be removed here
        or it persists under ``outputs/`` and can leak when a run dir is
        shared.
        """

    async def record_cli_usage(
        self,
        stdout_text: str,
        stderr_text: str,
        done: CLIDoneRecord,
    ) -> None:
        """Optionally bill token/cost usage from a CLI transcript."""

    def cli_usage_must_succeed(self) -> bool:
        """Whether missing or invalid usage must fail the invocation."""
        return False

    def cli_input(self, inp: BaseModel) -> str:
        """Build the message piped into the CLI's stdin."""
        return ""

    def extra_env(self, sandbox: Sandbox, inp: BaseModel) -> dict[str, str]:
        """Subclass-extensible env vars passed to the sandbox.

        Merged *after* the framework's own vars (FINISH_DONE_PATH),
        so a subclass can override them if truly needed.
        """
        return {}

    def sanitize_cli_output(self, text: str) -> str:
        """Remove subclass-owned secrets before transcripts reach artifacts/events."""
        return text

    async def refresh_sensitive_state(
        self,
        sandbox: Sandbox,
        inp: BaseModel,
    ) -> None:
        """Refresh secrets that an external CLI may rotate while running."""

    def sandbox_root_for(self, inp: BaseModel) -> Path | None:
        """Return a persistent sandbox root for this invocation, if any."""
        return self.sandbox_root

    def workspace_reservation_dir_for(self, root: Path) -> Path:
        """Return the shared lease registry used for this filesystem."""
        configured = self.WORKSPACE_RESERVATION_DIR
        if configured is not None:
            configured_path = Path(configured)
            if not configured_path.is_absolute():
                configured_path = self.ctx.root_workdir.parent / configured_path
            return configured_path
        return self.ctx.root_workdir.parent / ".proofcouncil-storage-reservations"

    def sensitive_quarantine_path_for(self, root: Path) -> Path:
        """Return a model-inaccessible sidecar for a persistent workspace."""
        resolved = Path(root).resolve()
        digest = hashlib.sha256(os.fsencode(str(resolved))).hexdigest()
        return (
            self.ctx.root_workdir.parent
            / ".proofcouncil-sensitive-quarantine"
            / f"{digest}.marker"
        )

    def workspace_recovery_state_path_for(self, root: Path) -> Path:
        """Return host-side recovery state that survives workflow resumes."""
        resolved = Path(root).resolve()
        digest = hashlib.sha256(os.fsencode(str(resolved))).hexdigest()
        return (
            self.ctx.root_workdir.parent
            / _WORKSPACE_RECOVERY_STATE_DIR
            / f"{digest}.json"
        )

    def workspace_recovery_lock_path_for(self, root: Path) -> Path:
        """Return the stable lock file protecting one recovery state record."""
        return self.workspace_recovery_state_path_for(root).with_suffix(".lock")

    def workspace_invocation_lock_path_for(self, root: Path) -> Path:
        """Return the lock file serializing use of one persistent workspace."""
        resolved = Path(root).resolve()
        digest = hashlib.sha256(os.fsencode(str(resolved))).hexdigest()
        return (
            resolved.parent
            / _WORKSPACE_RECOVERY_STATE_DIR
            / f"{digest}.active.lock"
        )

    def workspace_active_guard_path_for(self, root: Path) -> Path:
        """Return the model-inaccessible guard for an active CLI workspace."""
        resolved = Path(root).resolve()
        digest = hashlib.sha256(os.fsencode(str(resolved))).hexdigest()
        return (
            resolved.parent
            / _WORKSPACE_RECOVERY_STATE_DIR
            / f"{digest}.active.json"
        )

    async def _recover_guarded_workspace(
        self,
        root: Path,
        guard: Path,
    ) -> _GuardedWorkspaceRecovery:
        """Preserve a guarded workspace before starting a clean invocation."""

        def recover() -> _GuardedWorkspaceRecovery:
            payload = _validate_workspace_active_guard(guard, root)
            phase = str(payload["phase"])
            backend = str(payload["backend"])
            if phase == _WORKSPACE_GUARD_LAUNCH_PENDING:
                raise RuntimeError(
                    "persistent workspace has an unresolved external launch; "
                    "automatic recovery is unsafe until an operator confirms "
                    f"that no worker can still start: {guard}"
                )
            if phase == _WORKSPACE_GUARD_IDLE:
                _clear_workspace_active_guard(guard)
                return _GuardedWorkspaceRecovery(
                    action="cleared",
                    backend=backend,
                )
            if backend != "docker":
                raise RuntimeError(
                    "persistent workspace has a stale active guard, but the "
                    f"{backend!r} backend cannot prove that every detached worker "
                    "is gone. The workspace was preserved in place; an operator "
                    f"must confirm it is idle before removing the guard: {guard}"
                )
            quarantined = _quarantine_workspace(root)
            try:
                self._clear_workspace_recovery_attempts(root)
                _clear_workspace_active_guard(guard)
            except BaseException:
                _restore_quarantined_workspace(root, quarantined)
                raise
            return _GuardedWorkspaceRecovery(
                action="quarantined",
                backend=backend,
                quarantine_path=quarantined,
            )

        # Cancelling ``to_thread`` does not stop its filesystem operation. Drain
        # the atomic move/recreate transaction before releasing the invocation
        # lock so a resumed worker cannot race workspace replacement.
        recovery = await _run_in_thread_uninterruptibly(recover)
        if recovery.action == "quarantined":
            await self.events.emit(
                "cli.workspace_quarantined_after_unconfirmed_stop",
                {
                    "workspace": str(Path(root).resolve()),
                    "quarantine_path": str(recovery.quarantine_path),
                    "reason": "stale_active_guard",
                    "backend": recovery.backend,
                    "fresh_workspace_created": True,
                },
            )
        else:
            await self.events.emit(
                "cli.workspace_idle_guard_cleared",
                {
                    "workspace": str(Path(root).resolve()),
                    "reason": "stale_idle_guard",
                    "backend": recovery.backend,
                },
            )
        return recovery

    @staticmethod
    def _workspace_identity(root: Path) -> tuple[int, int]:
        metadata = Path(root).resolve().stat()
        return int(metadata.st_dev), int(metadata.st_ino)

    @staticmethod
    def _ensure_workspace_recovery_state_parent(path: Path) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent_metadata = path.parent.lstat()
        if not stat.S_ISDIR(parent_metadata.st_mode):
            raise RuntimeError(
                f"workspace recovery state parent is not a directory: {path.parent}"
            )

    def _acquire_workspace_lock(self, path: Path, *, busy_message: str) -> int:
        self._ensure_workspace_recovery_state_parent(path)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = -1
        try:
            fd = os.open(path, flags, 0o600)
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeError(
                    f"workspace lock is not a private regular file: {path}"
                )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                raise _WorkspaceLockBusy(
                    f"{busy_message}: {path}"
                ) from e
            return fd
        except Exception:
            if fd >= 0:
                os.close(fd)
            raise

    def _acquire_workspace_recovery_lock(self, root: Path) -> int:
        return self._acquire_workspace_lock(
            self.workspace_recovery_lock_path_for(root),
            busy_message="workspace recovery state is busy",
        )

    def _acquire_workspace_invocation_lock(self, root: Path) -> int:
        return self._acquire_workspace_lock(
            self.workspace_invocation_lock_path_for(root),
            busy_message="persistent workspace is already active",
        )

    @staticmethod
    def _release_workspace_lock(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    @staticmethod
    def _close_workspace_invocation_lock(fd: int) -> None:
        """Drop only this process's reference to an inherited lease.

        Explicitly unlocking would also release the shared ``flock`` held by
        a worker that survived an orchestrator crash or failed termination.
        Closing releases the lease when no spawned process still owns it.
        """
        os.close(fd)

    def _read_workspace_recovery_attempts(self, root: Path) -> int:
        path = self.workspace_recovery_state_path_for(root)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return 0
        except OSError:
            return max(1, self.WORKSPACE_RECOVERY_MAX_ATTEMPTS)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4096:
            return max(1, self.WORKSPACE_RECOVERY_MAX_ATTEMPTS)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags)
            try:
                chunks: list[bytes] = []
                remaining = 4097
                while remaining > 0:
                    chunk = os.read(fd, remaining)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
            finally:
                os.close(fd)
            if len(raw) > 4096:
                return max(1, self.WORKSPACE_RECOVERY_MAX_ATTEMPTS)
            payload = json.loads(raw.decode("utf-8"))
            identity = self._workspace_identity(root)
            if payload.get("workspace_identity") != [identity[0], identity[1]]:
                return 0
            attempts = int(payload.get("attempts", 0))
            return max(0, attempts)
        except (OSError, UnicodeDecodeError, ValueError, TypeError, AttributeError):
            return max(1, self.WORKSPACE_RECOVERY_MAX_ATTEMPTS)

    def _write_workspace_recovery_attempts(self, root: Path, attempts: int) -> None:
        path = self.workspace_recovery_state_path_for(root)
        self._ensure_workspace_recovery_state_parent(path)
        identity = self._workspace_identity(root)
        payload = json.dumps(
            {
                "attempts": max(0, int(attempts)),
                "workspace_identity": [identity[0], identity[1]],
                "updated_at": time.time(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(fd, 0o600)
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("could not write workspace recovery state")
                offset += written
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(temporary, path)
        except Exception:
            if fd >= 0:
                os.close(fd)
            try:
                temporary.unlink()
            except OSError:
                pass
            raise

    def _claim_workspace_recovery_attempt(
        self,
        root: Path,
    ) -> _WorkspaceRecoveryClaim | None:
        lock_fd = self._acquire_workspace_recovery_lock(root)
        try:
            # The host-side record is canonical. An agent instance can serve
            # more than one workspace over its lifetime, so carrying the
            # in-memory count across roots would incorrectly exhaust a fresh
            # workspace (or one recreated at the same path).
            attempts = self._read_workspace_recovery_attempts(root)
            self._workspace_recovery_attempts = attempts
            if (
                self.WORKSPACE_RECOVERY_MAX_ATTEMPTS > 0
                and attempts >= self.WORKSPACE_RECOVERY_MAX_ATTEMPTS
            ):
                claim = None
            else:
                claim = _WorkspaceRecoveryClaim(
                    workspace_root=Path(root),
                    attempts=attempts + 1,
                    lock_fd=lock_fd,
                )
        except Exception:
            self._release_workspace_lock(lock_fd)
            raise
        if claim is None:
            self._release_workspace_lock(lock_fd)
        return claim

    def _commit_workspace_recovery_claim(
        self,
        claim: _WorkspaceRecoveryClaim,
    ) -> None:
        try:
            self._write_workspace_recovery_attempts(
                claim.workspace_root,
                claim.attempts,
            )
            self._workspace_recovery_attempts = claim.attempts
        finally:
            self._release_workspace_lock(claim.lock_fd)

    def _abort_workspace_recovery_claim(
        self,
        claim: _WorkspaceRecoveryClaim,
    ) -> None:
        self._release_workspace_lock(claim.lock_fd)

    def _clear_workspace_recovery_attempts(self, root: Path) -> None:
        path = self.workspace_recovery_state_path_for(root)
        lock_fd = self._acquire_workspace_recovery_lock(root)
        try:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as unlink_error:
                try:
                    self._write_workspace_recovery_attempts(root, 0)
                except Exception as reset_error:
                    raise RuntimeError(
                        "could not clear persisted workspace recovery state"
                    ) from reset_error
                if self._read_workspace_recovery_attempts(root) != 0:
                    raise RuntimeError(
                        "persisted workspace recovery state did not reset"
                    ) from unlink_error
            self._workspace_recovery_attempts = 0
        finally:
            self._release_workspace_lock(lock_fd)

    async def _clear_workspace_recovery_attempts_recorded(
        self,
        root: Path,
        *,
        phase: str,
        call_id: str | None = None,
    ) -> str | None:
        try:
            await asyncio.to_thread(self._clear_workspace_recovery_attempts, root)
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            await self.events.emit(
                "cli.workspace_recovery_state_clear_failed",
                {
                    "workspace": str(root),
                    "phase": phase,
                    "type": type(e).__name__,
                    "msg": str(e),
                },
                call_id=call_id,
            )
            return detail
        return None

    def _quarantine_sensitive_workspace(
        self,
        root: Path,
        workspace_marker: Path,
    ) -> tuple[Path, ...]:
        runtime_dir = workspace_marker.parent
        return _mark_sensitive_workspace_untrusted(
            workspace_marker,
            self.sensitive_quarantine_path_for(root),
            recovery_sources=(
                runtime_dir / "done.json",
                runtime_dir / "WRAP_UP",
            ),
        )

    def _workspace_monitor_enabled(self) -> bool:
        return any(
            value > 0
            for value in (
                self.WORKSPACE_HARD_LIMIT_BYTES,
                self.WORKSPACE_HARD_LIMIT_ENTRIES,
                self.WORKSPACE_MIN_FREE_BYTES,
                self.WORKSPACE_MIN_FREE_INODES,
            )
        )

    def _measure_workspace(self, root: Path) -> dict[str, Any]:
        stop_after_bytes = self.WORKSPACE_HARD_LIMIT_BYTES
        stop_after_entries = self.WORKSPACE_HARD_LIMIT_ENTRIES
        if self._workspace_recovery_mode:
            if self._workspace_recovery_max_bytes is not None:
                stop_after_bytes = self._workspace_recovery_max_bytes
            if self._workspace_recovery_max_entries is not None:
                stop_after_entries = self._workspace_recovery_max_entries
        return self._sanitize_workspace_usage(
            measure_workspace_usage(
                root,
                stop_after_bytes=stop_after_bytes,
                stop_after_entries=stop_after_entries,
            )
        )

    def _measure_workspace_recovery_baseline(self, root: Path) -> dict[str, Any]:
        # A recovery ceiling must be relative to the complete current
        # footprint, not the ordinary hard limit where the fast monitor
        # truncates. This exceptional scan can be expensive, but an arbitrary
        # cap would make sufficiently oversized workspaces impossible to
        # recover: the raised ceiling could still sit below existing usage.
        return self._sanitize_workspace_usage(
            measure_workspace_usage(root)
        )

    def _sanitize_workspace_usage(
        self,
        usage: dict[str, Any],
    ) -> dict[str, Any]:
        return usage

    def _workspace_limit_failure(
        self,
        usage: dict[str, Any],
        *,
        recovery_ceiling: bool = True,
    ) -> tuple[str, str] | None:
        entries = int(usage.get("entries", usage.get("files", 0)) or 0)
        used_bytes = self._workspace_used_bytes(usage)
        free_bytes = usage.get("filesystem_free_bytes")
        free_inodes = usage.get("filesystem_free_inodes")
        errors = int(usage.get("errors", 0) or 0)
        if errors > 0:
            return (
                "workspace_scan_error",
                f"workspace safety scan encountered {errors} filesystem error(s)",
            )
        hard_bytes = self.WORKSPACE_HARD_LIMIT_BYTES
        hard_entries = self.WORKSPACE_HARD_LIMIT_ENTRIES
        if recovery_ceiling and self._workspace_recovery_mode:
            if self._workspace_recovery_max_bytes is not None:
                hard_bytes = self._workspace_recovery_max_bytes
            if self._workspace_recovery_max_entries is not None:
                hard_entries = self._workspace_recovery_max_entries
        if (
            hard_bytes > 0
            and used_bytes >= hard_bytes
        ):
            return (
                "workspace_hard_limit",
                f"workspace allocated size {used_bytes} is at or above hard limit "
                f"{hard_bytes} bytes",
            )
        if (
            hard_entries > 0
            and entries >= hard_entries
        ):
            return (
                "workspace_entry_limit",
                f"workspace entry count {entries} is at or above hard limit "
                f"{hard_entries}",
            )
        if self.WORKSPACE_MIN_FREE_BYTES > 0 and not isinstance(free_bytes, int):
            return (
                "filesystem_free_unknown",
                "filesystem free space could not be measured",
            )
        if (
            self.WORKSPACE_MIN_FREE_BYTES > 0
            and isinstance(free_bytes, int)
            and free_bytes < self.WORKSPACE_MIN_FREE_BYTES
        ):
            return (
                "filesystem_min_free",
                f"filesystem free space {free_bytes} is below required reserve "
                f"{self.WORKSPACE_MIN_FREE_BYTES} bytes",
            )
        if (
            self.WORKSPACE_MIN_FREE_INODES > 0
            and self.WORKSPACE_REQUIRE_INODE_ACCOUNTING
            and not isinstance(free_inodes, int)
        ):
            return (
                "filesystem_free_inodes_unknown",
                "filesystem free inodes could not be measured for the explicitly "
                "configured reserve",
            )
        if (
            self.WORKSPACE_MIN_FREE_INODES > 0
            and isinstance(free_inodes, int)
            and free_inodes < self.WORKSPACE_MIN_FREE_INODES
        ):
            return (
                "filesystem_min_free_inodes",
                f"filesystem free inodes {free_inodes} are below required reserve "
                f"{self.WORKSPACE_MIN_FREE_INODES}",
            )
        return None

    @staticmethod
    def _workspace_used_bytes(usage: dict[str, Any]) -> int:
        if usage.get("allocated_bytes_supported") is True:
            return int(usage.get("allocated_bytes", 0) or 0)
        if "limit_bytes" in usage:
            return int(usage.get("limit_bytes", 0) or 0)
        return int(usage.get("bytes", 0) or 0)

    def _workspace_soft_pressure(self, usage: dict[str, Any]) -> bool:
        entries = int(usage.get("entries", usage.get("files", 0)) or 0)
        return bool(
            (
                self.WORKSPACE_SOFT_LIMIT_BYTES > 0
                and self._workspace_used_bytes(usage)
                >= self.WORKSPACE_SOFT_LIMIT_BYTES
            )
            or (
                self.WORKSPACE_SOFT_LIMIT_ENTRIES > 0
                and entries >= self.WORKSPACE_SOFT_LIMIT_ENTRIES
            )
        )

    def _workspace_recovery_target_satisfied(
        self,
        usage: dict[str, Any],
    ) -> bool:
        used_bytes = self._workspace_used_bytes(usage)
        entries = int(usage.get("entries", usage.get("files", 0)) or 0)
        byte_target = (
            self.WORKSPACE_SOFT_LIMIT_BYTES
            or self.WORKSPACE_HARD_LIMIT_BYTES
        )
        entry_target = (
            self.WORKSPACE_SOFT_LIMIT_ENTRIES
            or self.WORKSPACE_HARD_LIMIT_ENTRIES
        )
        return bool(
            (byte_target <= 0 or used_bytes < byte_target)
            and (entry_target <= 0 or entries < entry_target)
        )

    def _workspace_pressure_path(self, root: Path) -> Path:
        return root / ".pwc" / "runtime" / self.WORKSPACE_PRESSURE_FILENAME

    def _write_workspace_pressure(
        self,
        root: Path,
        usage: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        path = self._workspace_pressure_path(root)
        payload = {
            "reason": reason,
            "allocated_bytes": self._workspace_used_bytes(usage),
            "entries": int(usage.get("entries", usage.get("files", 0)) or 0),
            "soft_limit_bytes": self.WORKSPACE_SOFT_LIMIT_BYTES or None,
            "hard_limit_bytes": self.WORKSPACE_HARD_LIMIT_BYTES or None,
            "soft_limit_entries": self.WORKSPACE_SOFT_LIMIT_ENTRIES or None,
            "hard_limit_entries": self.WORKSPACE_HARD_LIMIT_ENTRIES or None,
            "created_at": time.time(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError:
            pass

    def _clear_workspace_pressure(self, root: Path) -> None:
        try:
            self._workspace_pressure_path(root).unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def _configure_workspace_recovery(self, usage: dict[str, Any]) -> None:
        used_bytes = self._workspace_used_bytes(usage)
        entries = int(usage.get("entries", usage.get("files", 0)) or 0)
        self._workspace_recovery_mode = True
        self._workspace_recovery_max_bytes = max(
            self.WORKSPACE_HARD_LIMIT_BYTES,
            used_bytes,
        ) + max(1, self.WORKSPACE_RECOVERY_GROWTH_BYTES)
        self._workspace_recovery_max_entries = max(
            self.WORKSPACE_HARD_LIMIT_ENTRIES,
            entries,
        ) + max(1, self.WORKSPACE_RECOVERY_GROWTH_ENTRIES)

    def _cancel_workspace_recovery(self) -> None:
        self._workspace_recovery_mode = False
        self._workspace_recovery_max_bytes = None
        self._workspace_recovery_max_entries = None

    def _prepare_workspace_recovery_claim(
        self,
        usage: dict[str, Any],
        workspace_root: Path,
    ) -> tuple[
        _WorkspaceRecoveryClaim | None,
        tuple[str, str] | None,
    ]:
        self._configure_workspace_recovery(usage)
        failure = self._workspace_limit_failure(
            usage,
            recovery_ceiling=True,
        )
        if failure is not None:
            self._cancel_workspace_recovery()
            return None, failure
        try:
            claim = self._claim_workspace_recovery_attempt(workspace_root)
        except Exception:
            self._cancel_workspace_recovery()
            raise
        if claim is None:
            self._cancel_workspace_recovery()
        return claim, None

    def _workspace_usage_materially_changed(
        self,
        previous: dict[str, Any] | None,
        current: dict[str, Any],
    ) -> bool:
        if previous is None:
            return True
        byte_delta = abs(
            self._workspace_used_bytes(current)
            - self._workspace_used_bytes(previous)
        )
        entry_delta = abs(
            int(current.get("entries", current.get("files", 0)) or 0)
            - int(previous.get("entries", previous.get("files", 0)) or 0)
        )
        return bool(
            byte_delta >= self.WORKSPACE_USAGE_EVENT_MIN_BYTES_DELTA
            or entry_delta >= self.WORKSPACE_USAGE_EVENT_MIN_ENTRIES_DELTA
        )

    async def _finalize_workspace_recovery(
        self,
        done: CLIDoneRecord,
        workspace_root: Path | None,
        *,
        spawn_call_id: str,
    ) -> CLIDoneRecord:
        if not self._workspace_recovery_mode or workspace_root is None:
            return done
        usage = await asyncio.to_thread(self._measure_workspace, workspace_root)
        failure = self._workspace_limit_failure(usage)
        recovered = failure is None and self._workspace_recovery_target_satisfied(
            usage
        )
        if recovered:
            self._cancel_workspace_recovery()
            state_error = await self._clear_workspace_recovery_attempts_recorded(
                workspace_root,
                phase="recovery_completed",
                call_id=spawn_call_id,
            )
            if state_error is not None:
                status: DoneStatus = "error" if done.status == "error" else "partial"
                summary = (done.summary or "").strip()
                suffix = (
                    "storage recovery completed, but its attempt state could not "
                    f"be reset: {state_error}"
                )
                self._write_workspace_pressure(
                    workspace_root,
                    usage,
                    reason="workspace_recovery_state_clear_failed",
                )
                return done.model_copy(
                    update={
                        "status": status,
                        "summary": f"{summary}; {suffix}" if summary else suffix,
                    }
                )
            self._clear_workspace_pressure(workspace_root)
            await self.events.emit(
                "cli.workspace_recovery_completed",
                usage,
                call_id=spawn_call_id,
            )
            return done

        reason = failure[0] if failure is not None else "still_above_soft_limit"
        self._write_workspace_pressure(
            workspace_root,
            usage,
            reason=reason,
        )
        await self.events.emit(
            "cli.workspace_recovery_incomplete",
            {**usage, "reason": reason},
            call_id=spawn_call_id,
        )
        detail = failure[1] if failure is not None else (
            "workspace remains at or above its configured soft limit"
        )
        status: DoneStatus = "error" if done.status == "error" else "partial"
        summary = (done.summary or "").strip()
        suffix = f"storage recovery incomplete: {detail}"
        return done.model_copy(
            update={
                "status": status,
                "summary": f"{summary}; {suffix}" if summary else suffix,
            }
        )

    # --- framework-managed -----------------------------------------------------

    async def run(self, inp: BaseModel) -> BaseModel:  # type: ignore[override]
        if not self.CLI_CMD:
            raise RuntimeError(f"{type(self).__name__}.CLI_CMD is empty")

        root = self.sandbox_root_for(inp)
        workspace_invocation_lock_fd: int | None = None
        if root is not None and self.WORKSPACE_RECOVERY_ENABLED:
            try:
                workspace_invocation_lock_fd = (
                    self._acquire_workspace_invocation_lock(root)
                )
            except _WorkspaceLockBusy as e:
                await self.events.emit(
                    "cli.workspace_invocation_busy",
                    {
                        "workspace": str(root),
                        "type": type(e).__name__,
                        "msg": str(e),
                    },
                )
                raise RuntimeError(str(e)) from e
        try:
            return await self._run_once(
                inp,
                root=root,
                workspace_invocation_lock_fd=workspace_invocation_lock_fd,
            )
        finally:
            if workspace_invocation_lock_fd is not None:
                try:
                    self._close_workspace_invocation_lock(
                        workspace_invocation_lock_fd
                    )
                except Exception as e:
                    try:
                        await self.events.emit(
                            "cli.workspace_invocation_release_failed",
                            {
                                "workspace": str(root),
                                "type": type(e).__name__,
                                "msg": str(e),
                            },
                        )
                    except Exception:
                        pass

    async def _run_once(
        self,
        inp: BaseModel,
        *,
        root: Path | None,
        workspace_invocation_lock_fd: int | None,
    ) -> BaseModel:
        if not self.CLI_CMD:
            raise RuntimeError(f"{type(self).__name__}.CLI_CMD is empty")

        self._cancel_workspace_recovery()

        await self._emit_budget_warnings(self.tracker.check())
        self.tracker.add_tool_call()
        await self._emit_budget_warnings(self.tracker.check())

        # Track the streaming process so the finally block can terminate
        # it unconditionally on cancellation. Without this, if the
        # surrounding task is cancelled while ``_wait_for_done`` is
        # awaiting, the codex (or other CLI) child keeps running until
        # its own timeout or until the container exits.
        stream = None
        # Set once the normal record_cli_usage below has been reached, so the
        # finally knows whether it still owes a partial-usage metering (see the
        # cancellation note in the finally block).
        usage_recorded = False
        # The retained metering task once the run produced a done record. The
        # finally awaits THIS task rather than re-running metering, so a cancel
        # landing mid-metering neither loses it nor double-counts it (A7).
        meter_task: asyncio.Task | None = None
        required_usage_error: BaseException | None = None
        output_truncation_emitted = False
        sensitive_state_trusted = True
        sensitive_state_ever_failed = False
        sensitive_suppression_reason = "credential_refresh_failed"
        worker_stop_failure_emitted = False
        worker_stop_workspace_quarantined = False
        workspace_active_guard: Path | None = None
        workspace_guard_armed = False
        workspace_guard_phase: str | None = None
        stream_spawn_attempted = False
        worker_stop_phase = "setup"
        spawn_call_id = new_call_id()
        workspace_guard_error: BaseException | None = None
        storage_lease_error: BaseException | None = None
        storage_retention_guard_state: str | None = None
        sensitive_quarantine_path: Path | None = None
        sensitive_quarantine_external_path: Path | None = None
        sandbox: Sandbox | None = None
        storage_lease: StorageReservationLease | None = None
        workspace_recovery_claim: _WorkspaceRecoveryClaim | None = None
        workspace_recovery_started_payload: dict[str, Any] | None = None
        workspace_recovery_notice: str | None = None

        def repair_storage_retention_guard() -> None:
            if workspace_active_guard is None or sandbox is None:
                raise StorageReservationError(
                    "cannot repair a storage reservation without its workspace guard"
                )
            phase = workspace_guard_phase
            if phase not in _WORKSPACE_GUARD_PHASES:
                phase = _WORKSPACE_GUARD_LAUNCH_PENDING
            _write_workspace_active_guard(
                workspace_active_guard,
                sandbox.root,
                backend=resolve_backend(self.SANDBOX),
                phase=phase,
            )

        try:
            if root is not None:
                root.mkdir(parents=True, exist_ok=True)
                inherited_fds = (
                    (workspace_invocation_lock_fd,)
                    if workspace_invocation_lock_fd is not None
                    else ()
                )
                sandbox = make_sandbox(
                    self.SANDBOX,
                    root=root,
                    inherited_fds=inherited_fds,
                )
                persistent = True
            else:
                sandbox = make_sandbox(self.SANDBOX, root=self.workdir / "sandbox")
                persistent = False
            ensure_workspace_available = getattr(
                sandbox,
                "ensure_workspace_available",
                None,
            )
            if ensure_workspace_available is not None:
                await ensure_workspace_available()
            if persistent:
                runtime_dir = sandbox.root / ".pwc" / "runtime"
                sensitive_quarantine_path = (
                    runtime_dir / _SENSITIVE_STATE_QUARANTINE
                )
                sensitive_quarantine_external_path = (
                    self.sensitive_quarantine_path_for(sandbox.root)
                )
                quarantine_paths = (
                    sensitive_quarantine_path,
                    sensitive_quarantine_external_path,
                )
                existing_quarantine_paths = [
                    path
                    for path in quarantine_paths
                    if _path_exists_without_following(path)
                ]
                if existing_quarantine_paths:
                    raise RuntimeError(
                        "persistent CLI workspace is quarantined after an "
                        "unreadable credential refresh; discard or manually "
                        "sanitize it before removing all quarantine markers: "
                        + ", ".join(str(path) for path in existing_quarantine_paths)
                    )
                if workspace_invocation_lock_fd is not None:
                    workspace_active_guard = self.workspace_active_guard_path_for(
                        sandbox.root
                    )
                    if _path_exists_without_following(workspace_active_guard):
                        guarded_recovery = await self._recover_guarded_workspace(
                            sandbox.root,
                            workspace_active_guard,
                        )
                        if guarded_recovery.action == "quarantined":
                            workspace_recovery_notice = (
                                "A previous persistent workspace was preserved in "
                                "host-side quarantine after its worker stop or "
                                "teardown could not be confirmed. This invocation "
                                "started with a fresh workspace; prior files were "
                                "not available to the Compute worker."
                            )
                runtime_dir.mkdir(parents=True, exist_ok=True)
                bin_dir = runtime_dir / ".bin"
                bin_dir.mkdir(parents=True, exist_ok=True)
                done_path = runtime_dir / "done.json"
                wrap_up_path: Path | None = runtime_dir / "WRAP_UP"
                for stale in (done_path, wrap_up_path):
                    try:
                        stale.unlink()
                    except FileNotFoundError:
                        pass
            else:
                bin_dir = sandbox.root / ".bin"
                bin_dir.mkdir(parents=True, exist_ok=True)
                done_path = sandbox.root / "done.json"
                wrap_up_path = None

            if persistent and self.WORKSPACE_RESERVATION_BYTES > 0:
                if workspace_active_guard is None:
                    raise RuntimeError(
                        "persistent CLI storage reservations require "
                        "WORKSPACE_RECOVERY_ENABLED so an unconfirmed worker "
                        "cannot lose its reservation after a process crash"
                    )
                reservation_dir = self.workspace_reservation_dir_for(sandbox.root)
                storage_lease = StorageReservationLease(
                    registry_dir=reservation_dir,
                    workspace_root=sandbox.root,
                    requested_bytes=self.WORKSPACE_RESERVATION_BYTES,
                    minimum_free_bytes=self.WORKSPACE_MIN_FREE_BYTES,
                    owner=f"{self.ctx.run_id}:{self.name}",
                    retention_guard_path=workspace_active_guard,
                    retention_guard_repair=repair_storage_retention_guard,
                )
                try:
                    reservation = await asyncio.to_thread(storage_lease.acquire)
                except StorageReservationError as e:
                    await self.events.emit(
                        "cli.storage_reservation_denied",
                        {
                            "workspace": str(sandbox.root),
                            "registry": str(reservation_dir),
                            "requested_bytes": self.WORKSPACE_RESERVATION_BYTES,
                            "minimum_free_bytes": self.WORKSPACE_MIN_FREE_BYTES,
                            "msg": str(e),
                        },
                    )
                    raise RuntimeError(str(e)) from e
                await self.events.emit(
                    "cli.storage_reserved",
                    {
                        "workspace": str(sandbox.root),
                        "registry": str(reservation_dir),
                        **reservation.__dict__,
                    },
                )

            if workspace_active_guard is not None:
                workspace_guard_armed = True
                try:
                    await _run_in_thread_uninterruptibly(
                        _write_workspace_active_guard,
                        workspace_active_guard,
                        sandbox.root,
                        backend=resolve_backend(self.SANDBOX),
                        phase=_WORKSPACE_GUARD_IDLE,
                    )
                    workspace_guard_phase = _WORKSPACE_GUARD_IDLE
                except asyncio.CancelledError:
                    # The write may have completed before cancellation was
                    # re-raised. Keep the in-memory guard armed so the finalizer
                    # can clear it only after teardown succeeds.
                    raise
                except BaseException:
                    workspace_guard_armed = False
                    raise
                await _run_in_thread_uninterruptibly(
                    _update_workspace_active_guard,
                    workspace_active_guard,
                    sandbox.root,
                    phase=_WORKSPACE_GUARD_LAUNCH_PENDING,
                )
                workspace_guard_phase = _WORKSPACE_GUARD_LAUNCH_PENDING

            # setup() may invoke a CLI and may copy mutable credentials into the
            # workspace. Do not trust artifacts again until backend cleanup and
            # credential refresh both complete.
            sensitive_state_trusted = False
            await self.setup(sandbox, inp)
            setup_stop_state = _sandbox_worker_stop_state(sandbox)
            if setup_stop_state is not WorkerStopState.STOPPED:
                raise _WorkerStopUnconfirmed(setup_stop_state, phase="setup")
            if workspace_active_guard is not None:
                await _run_in_thread_uninterruptibly(
                    _update_workspace_active_guard,
                    workspace_active_guard,
                    sandbox.root,
                    phase=_WORKSPACE_GUARD_LAUNCH_SETTLED,
                )
                workspace_guard_phase = _WORKSPACE_GUARD_LAUNCH_SETTLED

            if persistent and self._workspace_monitor_enabled():
                usage = await asyncio.to_thread(
                    self._measure_workspace,
                    sandbox.root,
                )
                failure = self._workspace_limit_failure(
                    usage,
                    recovery_ceiling=False,
                )
                recoverable_reasons = {
                    "workspace_hard_limit",
                    "workspace_entry_limit",
                }
                if (
                    self.WORKSPACE_RECOVERY_ENABLED
                    and failure is not None
                    and failure[0] in recoverable_reasons
                    and usage.get("scan_truncated")
                ):
                    usage = await asyncio.to_thread(
                        self._measure_workspace_recovery_baseline,
                        sandbox.root,
                    )
                    failure = self._workspace_limit_failure(
                        usage,
                        recovery_ceiling=False,
                    )

                pressure = self._workspace_soft_pressure(usage)
                recoverable_failure = (
                    failure is not None and failure[0] in recoverable_reasons
                )
                if self.WORKSPACE_RECOVERY_ENABLED and recoverable_failure:
                    reason = failure[0]
                    hard_failure = failure
                    self._write_workspace_pressure(
                        sandbox.root,
                        usage,
                        reason=reason,
                    )
                    # Workspace byte/entry limits use the bounded recovery
                    # ceiling. Global filesystem floors remain non-bypassable,
                    # and must be checked before an attempt is claimed.
                    try:
                        (
                            workspace_recovery_claim,
                            failure,
                        ) = self._prepare_workspace_recovery_claim(
                            usage,
                            sandbox.root,
                        )
                    except _WorkspaceLockBusy:
                        failure = hard_failure
                        await self.events.emit(
                            "cli.workspace_recovery_busy",
                            {
                                **usage,
                                "reason": reason,
                            },
                        )
                    if failure is not None:
                        self._cancel_workspace_recovery()
                        self._write_workspace_pressure(
                            sandbox.root,
                            usage,
                            reason=failure[0],
                        )
                    elif workspace_recovery_claim is not None:
                        workspace_recovery_started_payload = {
                            **usage,
                            "reason": reason,
                            "attempt": workspace_recovery_claim.attempts,
                            "max_attempts": self.WORKSPACE_RECOVERY_MAX_ATTEMPTS,
                            "recovery_max_bytes": self._workspace_recovery_max_bytes,
                            "recovery_max_entries": self._workspace_recovery_max_entries,
                        }
                    else:
                        self._cancel_workspace_recovery()
                        failure = hard_failure
                        await self.events.emit(
                            "cli.workspace_recovery_exhausted",
                            {
                                **usage,
                                "reason": reason,
                                "attempts": self._workspace_recovery_attempts,
                                "max_attempts": self.WORKSPACE_RECOVERY_MAX_ATTEMPTS,
                            },
                        )
                elif pressure:
                    self._write_workspace_pressure(
                        sandbox.root,
                        usage,
                        reason="soft_limit",
                    )
                    await self.events.emit(
                        "cli.workspace_limit_warning",
                        {
                            **usage,
                            "phase": "pre_spawn",
                            "soft_limit_bytes": self.WORKSPACE_SOFT_LIMIT_BYTES,
                            "hard_limit_bytes": self.WORKSPACE_HARD_LIMIT_BYTES,
                            "soft_limit_entries": self.WORKSPACE_SOFT_LIMIT_ENTRIES or None,
                            "hard_limit_entries": self.WORKSPACE_HARD_LIMIT_ENTRIES or None,
                        },
                    )
                elif not pressure:
                    if failure is None:
                        state_error = (
                            await self._clear_workspace_recovery_attempts_recorded(
                                sandbox.root,
                                phase="pre_spawn_below_pressure",
                            )
                        )
                        if state_error is not None:
                            failure = (
                                "workspace_recovery_state_clear_failed",
                                "persisted workspace recovery state could not be "
                                f"reset: {state_error}",
                            )
                    if failure is None:
                        self._clear_workspace_pressure(sandbox.root)
                    elif failure[0] == "workspace_recovery_state_clear_failed":
                        self._write_workspace_pressure(
                            sandbox.root,
                            usage,
                            reason=failure[0],
                        )
                await self.events.emit(
                    "cli.workspace_usage",
                    {
                        **usage,
                        "phase": "pre_spawn",
                        "soft_limit_bytes": self.WORKSPACE_SOFT_LIMIT_BYTES or None,
                        "hard_limit_bytes": self.WORKSPACE_HARD_LIMIT_BYTES,
                        "soft_limit_entries": self.WORKSPACE_SOFT_LIMIT_ENTRIES or None,
                        "hard_limit_entries": self.WORKSPACE_HARD_LIMIT_ENTRIES or None,
                        "min_free_bytes": self.WORKSPACE_MIN_FREE_BYTES or None,
                        "min_free_inodes": self.WORKSPACE_MIN_FREE_INODES or None,
                        "recovery_mode": self._workspace_recovery_mode,
                    },
                )
                if failure is not None:
                    reason, detail = failure
                    await self.events.emit(
                        "cli.workspace_limit_exceeded",
                        {
                            **usage,
                            "phase": "pre_spawn",
                            "hard_limit_bytes": self.WORKSPACE_HARD_LIMIT_BYTES,
                            "hard_limit_entries": self.WORKSPACE_HARD_LIMIT_ENTRIES or None,
                            "min_free_bytes": self.WORKSPACE_MIN_FREE_BYTES or None,
                            "min_free_inodes": self.WORKSPACE_MIN_FREE_INODES or None,
                            "reason": reason,
                        },
                    )
                    raise RuntimeError(
                        f"persistent CLI workspace cannot start: {detail}"
                    )

            # Install the finish shim into a private bin dir inside
            # the sandbox root. Both backends expose this dir to the CLI.
            shim = bin_dir / "finish"
            shim.write_text(FINISH_SCRIPT, encoding="utf-8")
            shim.chmod(0o755)

            extra_env: dict[str, str] = {
                "FINISH_DONE_PATH": str(done_path),
                "FINISH_BIN": str(shim),
            }
            extra_env.update(self.extra_env(sandbox, inp))
            self._install_shell_startup(sandbox, bin_dir=bin_dir, shim=shim, done_path=done_path)

            timeout_s = self._effective_timeout_s()
            soft_timeout_s = self._effective_soft_timeout_s(timeout_s)
            await self.events.emit(
                "cli.spawn",
                {
                    "cmd": self.CLI_CMD,
                    "sandbox": str(sandbox.root),
                    "backend": resolve_backend(self.SANDBOX),
                    "timeout_s": timeout_s,
                    "soft_timeout_s": soft_timeout_s or None,
                    "persistent_workspace": persistent,
                    "workspace_soft_limit_bytes": (
                        self.WORKSPACE_SOFT_LIMIT_BYTES or None
                    ),
                    "workspace_hard_limit_bytes": (
                        self.WORKSPACE_HARD_LIMIT_BYTES or None
                    ),
                    "workspace_soft_limit_entries": (
                        self.WORKSPACE_SOFT_LIMIT_ENTRIES or None
                    ),
                    "workspace_hard_limit_entries": (
                        self.WORKSPACE_HARD_LIMIT_ENTRIES or None
                    ),
                    "workspace_min_free_bytes": self.WORKSPACE_MIN_FREE_BYTES or None,
                    "workspace_min_free_inodes": self.WORKSPACE_MIN_FREE_INODES or None,
                    "workspace_reservation_bytes": (
                        self.WORKSPACE_RESERVATION_BYTES or None
                    ),
                    "workspace_reservation_dir": (
                        str(self.workspace_reservation_dir_for(sandbox.root))
                        if self.WORKSPACE_RESERVATION_BYTES > 0
                        else None
                    ),
                    "workspace_recovery_mode": self._workspace_recovery_mode,
                    "workspace_recovery_max_bytes": self._workspace_recovery_max_bytes,
                    "workspace_recovery_max_entries": self._workspace_recovery_max_entries,
                },
                call_id=spawn_call_id,
            )

            if workspace_active_guard is not None:
                await _run_in_thread_uninterruptibly(
                    _update_workspace_active_guard,
                    workspace_active_guard,
                    sandbox.root,
                    phase=_WORKSPACE_GUARD_LAUNCH_PENDING,
                )
                workspace_guard_phase = _WORKSPACE_GUARD_LAUNCH_PENDING
            worker_stop_phase = "spawn"
            stream_spawn_attempted = True
            try:
                stream = await sandbox.stream_command(
                    self.CLI_CMD,
                    env_extra=extra_env,
                    extra_path=[bin_dir],
                    timeout_s=timeout_s,
                )
            except SandboxSpawnError:
                # The backend guarantees that no worker was created. Treat the
                # workspace as idle so a missing executable cannot manufacture a
                # stale guard and quarantine useful state on the next invocation.
                stream_spawn_attempted = False
                raise
            worker_stop_phase = "finalize"
            if workspace_active_guard is not None:
                await _run_in_thread_uninterruptibly(
                    _update_workspace_active_guard,
                    workspace_active_guard,
                    sandbox.root,
                    phase=_WORKSPACE_GUARD_LAUNCH_SETTLED,
                )
                workspace_guard_phase = _WORKSPACE_GUARD_LAUNCH_SETTLED
            # Once the external process has started it may rotate credentials.
            # No transcript or workspace artifact is safe to persist until the
            # terminal credential state has been refreshed successfully.
            sensitive_state_trusted = False
            # Pipe the initial message to stdin if the process accepts it.
            if stream.proc.stdin is not None:
                payload = self.cli_input(inp).encode("utf-8")
                try:
                    stream.proc.stdin.write(payload)
                    await stream.proc.stdin.drain()
                    stream.proc.stdin.close()
                except (BrokenPipeError, ConnectionResetError, OSError, RuntimeError, ValueError) as e:
                    await self.events.emit(
                        "cli.stdin_closed",
                        {"type": type(e).__name__, "msg": str(e)},
                        call_id=spawn_call_id,
                    )
                    if workspace_recovery_claim is not None:
                        raise RuntimeError(
                            "workspace recovery prompt could not be delivered"
                        ) from e
            elif workspace_recovery_claim is not None:
                await self.events.emit(
                    "cli.stdin_closed",
                    {
                        "type": "MissingStdin",
                        "msg": "workspace recovery process has no stdin pipe",
                    },
                    call_id=spawn_call_id,
                )
                raise RuntimeError(
                    "workspace recovery prompt could not be delivered"
                )

            # A recovery attempt is spent only after the cleanup instructions
            # have reached the child. Startup and stdin failures remain retryable.
            if workspace_recovery_claim is not None:
                claim_to_commit = workspace_recovery_claim
                try:
                    self._commit_workspace_recovery_claim(claim_to_commit)
                finally:
                    # commit() releases the state-transaction lock even when its
                    # durable write fails, so the outer finally must not close it
                    # a second time.
                    workspace_recovery_claim = None
                if workspace_recovery_started_payload is not None:
                    await self.events.emit(
                        "cli.workspace_recovery_started",
                        workspace_recovery_started_payload,
                        call_id=spawn_call_id,
                    )

            try:
                done = await self._wait_for_done(
                    stream,
                    done_path,
                    spawn_call_id=spawn_call_id,
                    wrap_up_path=wrap_up_path,
                    soft_timeout_s=soft_timeout_s,
                    workspace_root=sandbox.root if persistent else None,
                )
                # Every normal _wait_for_done exit terminates the worker first.
                # Keep this check as defense in depth before credentials or
                # model-created artifacts are read from the shared workspace.
                _require_worker_stopped(stream, phase="post_process")
            except _WorkerStopUnconfirmed as e:
                sensitive_state_trusted = False
                sensitive_suppression_reason = "worker_stop_unconfirmed"
                if (
                    sensitive_quarantine_path is not None
                    and not workspace_guard_armed
                ):
                    self._quarantine_sensitive_workspace(
                        sandbox.root,
                        sensitive_quarantine_path,
                    )
                    sensitive_state_ever_failed = True
                    worker_stop_workspace_quarantined = True
                try:
                    await self.events.emit(
                        "cli.worker_stop_unconfirmed",
                        {
                            "state": e.state.value,
                            "phase": e.phase,
                            "response_suppressed": True,
                            "handoff_suppressed": True,
                            "workspace_quarantine_on_resume": (
                                workspace_guard_armed
                                and workspace_guard_phase
                                == _WORKSPACE_GUARD_LAUNCH_SETTLED
                            ),
                            "workspace_quarantined": (
                                worker_stop_workspace_quarantined
                            ),
                        },
                        call_id=spawn_call_id,
                    )
                except Exception:
                    pass
                else:
                    worker_stop_failure_emitted = True
                raise
            stdout_dropped = int(getattr(stream, "stdout_dropped_chars", 0) or 0)
            stderr_dropped = int(getattr(stream, "stderr_dropped_chars", 0) or 0)
            if stdout_dropped or stderr_dropped:
                output_truncation_emitted = True
                await self.events.emit(
                    "cli.output_truncated",
                    {
                        "stdout_dropped_chars": stdout_dropped,
                        "stderr_dropped_chars": stderr_dropped,
                        "retention": "bounded_tail",
                        "usage_events_captured": int(
                            getattr(stream, "usage_events_captured", 0) or 0
                        ),
                        "usage_capture_dropped_chars": int(
                            getattr(stream, "usage_capture_dropped_chars", 0) or 0
                        ),
                        "usage_capture_oversized_lines": int(
                            getattr(stream, "usage_capture_oversized_lines", 0) or 0
                        ),
                    },
                    call_id=spawn_call_id,
                )
            try:
                await self.refresh_sensitive_state(sandbox, inp)
            except Exception as e:
                sensitive_state_ever_failed = True
                if sensitive_quarantine_path is not None:
                    self._quarantine_sensitive_workspace(
                        sandbox.root,
                        sensitive_quarantine_path,
                    )
                await self.events.emit(
                    "cli.sensitive_state_refresh_failed",
                    {"type": type(e).__name__, "phase": "post_process"},
                    call_id=spawn_call_id,
                )
            else:
                sensitive_state_trusted = True
            # Meter as a retained, shielded task. A cancellation landing while
            # record_cli_usage runs must not skip it (that loses a detected
            # rate limit); the finally awaits this exact task instead of
            # re-running metering, so the ledger is never double-counted (A7).
            meter_task = asyncio.ensure_future(
                self.record_cli_usage(
                    getattr(stream, "metering_stdout", stream.stdout),
                    stream.stderr,
                    done,
                )
            )
            usage_recorded = True
            try:
                await asyncio.shield(meter_task)
            except Exception as e:  # a real failure; cancellation propagates
                if self.cli_usage_must_succeed():
                    required_usage_error = e
                await self.events.emit(
                    "cli.usage_record_failed",
                    {"type": type(e).__name__},
                    call_id=spawn_call_id,
                )
                if self.cli_usage_must_succeed():
                    raise RuntimeError(
                        "required CLI usage accounting failed; downstream "
                        "execution was stopped"
                    ) from e
            if not sensitive_state_trusted:
                await self.events.emit(
                    "cli.sensitive_artifacts_suppressed",
                    {
                        "reason": "credential_refresh_failed",
                        "response_suppressed": True,
                        "handoff_suppressed": True,
                    },
                    call_id=spawn_call_id,
                )
                raise RuntimeError(
                    "CLI credential state could not be refreshed; response and "
                    "handoff artifacts were suppressed"
                )
            out = await self.collect(sandbox, inp, done)
            if workspace_recovery_notice is not None:
                out = self.attach_workspace_recovery_notice(
                    out,
                    workspace_recovery_notice,
                )
            try:
                await self._emit_budget_warnings(self.tracker.check())
            except BudgetExhausted as e:
                await self.events.emit(
                    "budget.exhausted_post_call",
                    {
                        "scope": e.scope,
                        "kind": e.limit_kind,
                        "used": e.used,
                        "limit": e.limit,
                        "note": "CLI round complete; downstream pre-call checks will abort",
                    },
                    call_id=spawn_call_id,
                )
            return out
        finally:
            if workspace_recovery_claim is not None:
                try:
                    self._abort_workspace_recovery_claim(
                        workspace_recovery_claim
                    )
                except Exception as e:
                    try:
                        await self.events.emit(
                            "cli.workspace_recovery_claim_release_failed",
                            {
                                "type": type(e).__name__,
                                "msg": str(e),
                            },
                        )
                    except Exception:
                        pass
                finally:
                    workspace_recovery_claim = None
            # Terminate the streaming child unconditionally. If we're
            # being cancelled mid-``_wait_for_done`` the underlying
            # process is still alive; without this it can keep running
            # past the parent's cleanup and into container shutdown.
            # ``asyncio.shield`` keeps the terminate sequence from
            # being interrupted by the same cancellation that brought
            # us here. Done before metering so the transcript is complete
            # when we meter partial usage.
            if stream is not None:
                try:
                    await asyncio.shield(stream.terminate())
                except (asyncio.CancelledError, Exception):
                    sensitive_state_trusted = False
                    sensitive_suppression_reason = "worker_stop_unconfirmed"
            if stream is not None:
                stop_state = _worker_stop_state(stream)
            elif sandbox is not None and (
                _reported_worker_stop_state(sandbox) is not None
            ):
                stop_state = _sandbox_worker_stop_state(sandbox)
            elif stream_spawn_attempted:
                # Legacy/custom sandboxes cannot report whether an interrupted
                # spawn reached the operating system or container runtime.
                stop_state = WorkerStopState.UNKNOWN
            else:
                stop_state = WorkerStopState.STOPPED
            process_stopped = stop_state is WorkerStopState.STOPPED
            if not process_stopped:
                sensitive_state_trusted = False
                sensitive_suppression_reason = "worker_stop_unconfirmed"
                if (
                    sensitive_quarantine_path is not None
                    and not workspace_guard_armed
                    and not worker_stop_workspace_quarantined
                ):
                    # Persistent agents that do not opt into automatic workspace
                    # recovery still need the original fail-closed behavior.
                    # Without either mechanism, a later invocation could reuse
                    # files written by a worker whose terminal state is unknown.
                    self._quarantine_sensitive_workspace(
                        sandbox.root,
                        sensitive_quarantine_path,
                    )
                    sensitive_state_ever_failed = True
                    worker_stop_workspace_quarantined = True
                if not worker_stop_failure_emitted:
                    try:
                        await asyncio.shield(
                            self.events.emit(
                                "cli.worker_stop_unconfirmed",
                                {
                                    "state": stop_state.value,
                                    "phase": worker_stop_phase,
                                    "response_suppressed": True,
                                    "handoff_suppressed": True,
                                    "workspace_quarantine_on_resume": (
                                        workspace_guard_armed
                                        and workspace_guard_phase
                                        == _WORKSPACE_GUARD_LAUNCH_SETTLED
                                    ),
                                    "workspace_quarantined": (
                                        worker_stop_workspace_quarantined
                                    ),
                                },
                                call_id=spawn_call_id,
                            )
                        )
                    except (asyncio.CancelledError, Exception):
                        pass
                    else:
                        worker_stop_failure_emitted = True
            if (
                sandbox is not None
                and process_stopped
                and not sensitive_state_trusted
            ):
                try:
                    await asyncio.shield(self.refresh_sensitive_state(sandbox, inp))
                except asyncio.CancelledError:
                    sensitive_state_trusted = False
                    sensitive_state_ever_failed = True
                    if sensitive_quarantine_path is not None:
                        self._quarantine_sensitive_workspace(
                            sandbox.root,
                            sensitive_quarantine_path,
                        )
                except Exception as e:
                    sensitive_state_trusted = False
                    sensitive_state_ever_failed = True
                    if sensitive_quarantine_path is not None:
                        self._quarantine_sensitive_workspace(
                            sandbox.root,
                            sensitive_quarantine_path,
                        )
                    try:
                        await self.events.emit(
                            "cli.sensitive_state_refresh_failed",
                            {"type": type(e).__name__, "phase": "finalize"},
                        )
                    except Exception:
                        pass
                else:
                    sensitive_state_trusted = not sensitive_state_ever_failed
            if stream is not None and not output_truncation_emitted:
                stdout_dropped = int(
                    getattr(stream, "stdout_dropped_chars", 0) or 0
                )
                stderr_dropped = int(
                    getattr(stream, "stderr_dropped_chars", 0) or 0
                )
                if stdout_dropped or stderr_dropped:
                    try:
                        await asyncio.shield(
                            self.events.emit(
                                "cli.output_truncated",
                                {
                                    "stdout_dropped_chars": stdout_dropped,
                                    "stderr_dropped_chars": stderr_dropped,
                                    "retention": "bounded_tail",
                                    "usage_events_captured": int(
                                        getattr(stream, "usage_events_captured", 0)
                                        or 0
                                    ),
                                    "usage_capture_dropped_chars": int(
                                        getattr(
                                            stream,
                                            "usage_capture_dropped_chars",
                                            0,
                                        )
                                        or 0
                                    ),
                                    "usage_capture_oversized_lines": int(
                                        getattr(
                                            stream,
                                            "usage_capture_oversized_lines",
                                            0,
                                        )
                                        or 0
                                    ),
                                },
                                call_id=spawn_call_id,
                            )
                        )
                    except (asyncio.CancelledError, Exception):
                        pass
            # Meter exactly once, even under cancellation — else the run loses
            # real token/cost accounting. If the run reached a done record,
            # metering was already dispatched as meter_task; await that same task
            # (no re-run, no double count). Otherwise we were cancelled/errored
            # before the run produced a record, so meter the partial usage the
            # CLI may have spent. Shielded so the same cancellation can't skip it.
            if meter_task is not None:
                # Drain the retained metering to completion even under REPEATED
                # cancellation: a second cancel while awaiting would otherwise
                # abandon the still-running task, and event-loop shutdown then
                # cancels it and the metering is lost (B6). The task is shielded,
                # so re-awaiting resumes it; the loop exits once it is done.
                while not meter_task.done():
                    try:
                        await asyncio.shield(meter_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if self.cli_usage_must_succeed() and meter_task.done():
                    try:
                        meter_error = meter_task.exception()
                    except asyncio.CancelledError as e:
                        meter_error = e
                    if meter_error is not None and required_usage_error is None:
                        required_usage_error = meter_error
            elif stream is not None and not usage_recorded:
                partial_meter_task = asyncio.ensure_future(
                    self.record_cli_usage(
                        getattr(stream, "metering_stdout", stream.stdout),
                        stream.stderr,
                        CLIDoneRecord(
                            status="partial",
                            summary="(cancelled mid-run)",
                        ),
                    )
                )
                while not partial_meter_task.done():
                    try:
                        await asyncio.shield(partial_meter_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if self.cli_usage_must_succeed() and partial_meter_task.done():
                    try:
                        partial_meter_error = partial_meter_task.exception()
                    except asyncio.CancelledError as e:
                        partial_meter_error = e
                    if partial_meter_error is not None:
                        required_usage_error = partial_meter_error
            if sensitive_state_trusted:
                try:
                    stdout_text = self.sanitize_cli_output(
                        stream.stdout if stream is not None else ""
                    )
                    stderr_text = self.sanitize_cli_output(
                        stream.stderr if stream is not None else ""
                    )
                    if stdout_text:
                        (self.workdir / "cli_stdout.log").write_text(stdout_text, encoding="utf-8")
                    if stderr_text:
                        (self.workdir / "cli_stderr.log").write_text(stderr_text, encoding="utf-8")
                except (NameError, OSError):
                    pass
            elif stream is not None:
                try:
                    await self.events.emit(
                        "cli.sensitive_output_suppressed",
                        {
                            "reason": sensitive_suppression_reason,
                            "stdout_chars": int(
                                getattr(stream, "stdout_chars", len(stream.stdout))
                            ),
                            "stderr_chars": int(
                                getattr(stream, "stderr_chars", len(stream.stderr))
                            ),
                        },
                    )
                except Exception:
                    pass
            # Keep the sandbox dir on disk so the workdir captures artifacts.
            # Scrub per-invocation state only after the worker is confirmed
            # stopped; otherwise its active guard preserves the old workspace in
            # quarantine before a fresh invocation can begin.
            teardown_succeeded = False
            try:
                if sandbox is not None:
                    if process_stopped:
                        try:
                            await self.teardown(sandbox, inp)
                        except Exception as e:
                            try:
                                await self.events.emit(
                                    "cli.teardown_error",
                                    {"type": type(e).__name__, "msg": str(e)},
                                )
                            except Exception:
                                pass
                        else:
                            teardown_succeeded = True
                    if (
                        workspace_guard_armed
                        and workspace_active_guard is not None
                        and process_stopped
                        and teardown_succeeded
                        and sensitive_state_trusted
                    ):
                        try:
                            await _run_in_thread_uninterruptibly(
                                _clear_workspace_active_guard,
                                workspace_active_guard,
                            )
                        except Exception as e:
                            workspace_guard_error = e
                            try:
                                await self.events.emit(
                                    "cli.workspace_active_guard_clear_failed",
                                    {
                                        "workspace": str(sandbox.root),
                                        "type": type(e).__name__,
                                        "msg": str(e),
                                    },
                                )
                            except Exception:
                                pass
            finally:
                if storage_lease is not None and storage_lease.status is not None:
                    reservation_status = storage_lease.status
                    try:
                        if process_stopped:
                            await _release_storage_lease(storage_lease)
                        else:
                            storage_retention_guard_state = (
                                await _retain_storage_lease(storage_lease)
                            )
                    except Exception as e:
                        storage_lease_error = e
                        try:
                            await self.events.emit(
                                (
                                    "cli.storage_release_failed"
                                    if process_stopped
                                    else "cli.storage_retain_failed"
                                ),
                                {
                                    "workspace": str(sandbox.root) if sandbox else None,
                                    "type": type(e).__name__,
                                    "msg": str(e),
                                },
                            )
                        except Exception:
                            pass
                    else:
                        try:
                            await self.events.emit(
                                (
                                    "cli.storage_released"
                                    if process_stopped
                                    else "cli.storage_retained"
                                ),
                                {
                                    "workspace": str(sandbox.root) if sandbox else None,
                                    "reserved_bytes": reservation_status.requested_bytes,
                                    "retention_guard_state": (
                                        storage_retention_guard_state
                                        if not process_stopped
                                        else None
                                    ),
                                },
                            )
                        except Exception:
                            pass
            if required_usage_error is not None:
                raise RuntimeError(
                    "required CLI usage accounting failed; downstream "
                    "execution was stopped"
                ) from required_usage_error
            if storage_lease_error is not None:
                raise RuntimeError(
                    "CLI storage reservation could not be finalized safely"
                ) from storage_lease_error
            if workspace_guard_error is not None:
                raise RuntimeError(
                    "CLI workspace safety guard could not be cleared; the "
                    "workspace remains unavailable until guard recovery succeeds"
                ) from workspace_guard_error

    async def _wait_for_done(
        self,
        stream,
        done_path: Path,
        *,
        spawn_call_id: str,
        wrap_up_path: Path | None = None,
        soft_timeout_s: int = 0,
        workspace_root: Path | None = None,
    ) -> CLIDoneRecord:
        spawn_t = time.monotonic()
        last_heartbeat = spawn_t
        next_workspace_check = spawn_t + max(0.0, self.WORKSPACE_CHECK_INTERVAL_S)
        last_workspace_event = spawn_t
        last_emitted_workspace_usage: dict[str, Any] | None = None
        last_workspace_pressure = self._workspace_recovery_mode
        cleanup_warned = False
        wrap_up_signaled = False
        workspace_soft_warned = False
        while True:
            now = time.monotonic()
            if (
                workspace_root is not None
                and self._workspace_monitor_enabled()
                and now >= next_workspace_check
            ):
                scan_started = time.monotonic()
                usage = await asyncio.to_thread(
                    self._measure_workspace,
                    workspace_root,
                )
                scan_finished = time.monotonic()
                scan_duration_s = scan_finished - scan_started
                next_workspace_check = scan_finished + max(
                    max(0.0, self.WORKSPACE_CHECK_INTERVAL_S),
                    scan_duration_s * 4.0,
                )
                failure = self._workspace_limit_failure(usage)
                pressure = self._workspace_soft_pressure(usage)
                should_emit_usage = bool(
                    failure is not None
                    or pressure != last_workspace_pressure
                    or self._workspace_usage_materially_changed(
                        last_emitted_workspace_usage,
                        usage,
                    )
                    or scan_finished - last_workspace_event
                    >= self.WORKSPACE_USAGE_EVENT_INTERVAL_S
                )
                if should_emit_usage:
                    await self.events.emit(
                        "cli.workspace_usage",
                        {
                            **usage,
                            "phase": "running",
                            "scan_duration_s": scan_duration_s,
                            "next_check_in_s": max(
                                0.0, next_workspace_check - scan_finished
                            ),
                            "soft_limit_bytes": self.WORKSPACE_SOFT_LIMIT_BYTES or None,
                            "hard_limit_bytes": self.WORKSPACE_HARD_LIMIT_BYTES,
                            "soft_limit_entries": self.WORKSPACE_SOFT_LIMIT_ENTRIES or None,
                            "hard_limit_entries": self.WORKSPACE_HARD_LIMIT_ENTRIES or None,
                            "min_free_bytes": self.WORKSPACE_MIN_FREE_BYTES or None,
                            "min_free_inodes": self.WORKSPACE_MIN_FREE_INODES or None,
                            "recovery_mode": self._workspace_recovery_mode,
                        },
                        call_id=spawn_call_id,
                    )
                    last_workspace_event = scan_finished
                    last_emitted_workspace_usage = dict(usage)
                if failure is not None:
                    reason, detail = failure
                    if reason in {
                        "workspace_hard_limit",
                        "workspace_entry_limit",
                    }:
                        self._write_workspace_pressure(
                            workspace_root,
                            usage,
                            reason=reason,
                        )
                    await stream.terminate()
                    await self.events.emit(
                        "cli.workspace_limit_exceeded",
                        {
                            **usage,
                            "phase": "running",
                            "hard_limit_bytes": self.WORKSPACE_HARD_LIMIT_BYTES,
                            "hard_limit_entries": self.WORKSPACE_HARD_LIMIT_ENTRIES or None,
                            "min_free_bytes": self.WORKSPACE_MIN_FREE_BYTES or None,
                            "min_free_inodes": self.WORKSPACE_MIN_FREE_INODES or None,
                            "reason": reason,
                            "recovery_mode": self._workspace_recovery_mode,
                        },
                        call_id=spawn_call_id,
                    )
                    _require_worker_stopped(stream, phase="workspace_limit")
                    return CLIDoneRecord(
                        status="error",
                        summary=f"workspace safety limit reached; stopped CLI: {detail}",
                    )
                if pressure:
                    self._write_workspace_pressure(
                        workspace_root,
                        usage,
                        reason="soft_limit",
                    )
                else:
                    self._clear_workspace_pressure(workspace_root)
                if pressure and not workspace_soft_warned:
                    workspace_soft_warned = True
                    await self.events.emit(
                        "cli.workspace_limit_warning",
                        {
                            **usage,
                            "soft_limit_bytes": self.WORKSPACE_SOFT_LIMIT_BYTES,
                            "hard_limit_bytes": self.WORKSPACE_HARD_LIMIT_BYTES,
                            "soft_limit_entries": self.WORKSPACE_SOFT_LIMIT_ENTRIES or None,
                            "hard_limit_entries": self.WORKSPACE_HARD_LIMIT_ENTRIES or None,
                        },
                        call_id=spawn_call_id,
                    )
                elif not pressure and workspace_soft_warned:
                    workspace_soft_warned = False
                    await self.events.emit(
                        "cli.workspace_limit_cleared",
                        usage,
                        call_id=spawn_call_id,
                    )
                last_workspace_pressure = pressure
            if done_path.exists():
                grace_deadline = time.monotonic() + float(self.DONE_DRAIN_GRACE_S)
                while (
                    not stream.done
                    and time.monotonic() < grace_deadline
                    and stream.remaining_s > 0
                ):
                    await asyncio.sleep(self.POLL_INTERVAL_S)
                await stream.terminate()
                _require_worker_stopped(stream, phase="completion_signal")
                done = self._read_done(
                    done_path,
                    fallback_status=(
                        "error" if self.COMPLETION_SIGNAL != "exit" else "done"
                    ),
                    fallback_summary=(
                        "required completion record is invalid"
                        if self.COMPLETION_SIGNAL != "exit"
                        else None
                    ),
                )
                return await self._finalize_workspace_recovery(
                    done,
                    workspace_root,
                    spawn_call_id=spawn_call_id,
                )
            if stream.done:
                rc = stream.proc.returncode
                completion_required = self.COMPLETION_SIGNAL != "exit"
                fallback = (
                    (
                        self.MISSING_COMPLETION_STATUS
                        if rc == 0
                        else "error"
                    )
                    if completion_required
                    else ("done" if rc == 0 else "error")
                )
                fallback_summary = (
                    (
                        f"required {self.COMPLETION_SIGNAL} completion record was not "
                        "written; salvaged artifacts from a clean CLI exit"
                        if fallback == "partial"
                        else f"required {self.COMPLETION_SIGNAL} completion record was not written"
                    )
                    if completion_required
                    else None
                )
                try:
                    await stream.terminate()
                except Exception:
                    pass
                _require_worker_stopped(stream, phase="process_exit")
                await self.events.emit(
                    "cli.exit",
                    {
                        "sandbox_id": str(self.workdir),
                        "exit_code": rc,
                        "status": fallback,
                        "via_finish": False,
                        # Raw output is persisted only after subclasses have a
                        # chance to refresh rotated credential values.  Do not
                        # put pre-refresh output text into the event stream.
                        "stdout_chars": (
                            int(stream.stdout_chars)
                            if hasattr(stream, "stdout_chars")
                            else len(stream.stdout or "")
                        ),
                        "stderr_chars": (
                            int(stream.stderr_chars)
                            if hasattr(stream, "stderr_chars")
                            else len(stream.stderr or "")
                        ),
                    },
                    call_id=spawn_call_id,
                )
                done = self._read_done(
                    done_path,
                    fallback_status=fallback,
                    fallback_summary=fallback_summary,
                )
                return await self._finalize_workspace_recovery(
                    done,
                    workspace_root,
                    spawn_call_id=spawn_call_id,
                )
            if (
                not cleanup_warned
                and self.CLEANUP_GRACE_S > 0
                and stream.remaining_s <= self.CLEANUP_GRACE_S
            ):
                cleanup_warned = True
                await self.events.emit(
                    "cli.cleanup_grace",
                    {
                        "remaining_s": stream.remaining_s,
                        "message": "budget/timeout nearly exhausted; current sandbox files will be salvaged if finish is not called",
                    },
                    call_id=spawn_call_id,
                )
            if (
                not wrap_up_signaled
                and soft_timeout_s > 0
                and wrap_up_path is not None
                and (time.monotonic() - spawn_t) >= soft_timeout_s
            ):
                wrap_up_signaled = True
                try:
                    wrap_up_path.parent.mkdir(parents=True, exist_ok=True)
                    wrap_up_path.write_text(
                        "wrap up: soft timeout reached; finalize and call $FINISH_BIN\n",
                        encoding="utf-8",
                    )
                except OSError:
                    pass
                await self.events.emit(
                    "cli.wrap_up_signal",
                    {
                        "soft_timeout_s": soft_timeout_s,
                        "elapsed_s": time.monotonic() - spawn_t,
                    },
                    call_id=spawn_call_id,
                )
            if stream.remaining_s <= 0:
                await stream.terminate()
                await self.events.emit(
                    "cli.exit",
                    {"sandbox_id": str(self.workdir), "status": "partial", "reason": "timeout"},
                    call_id=spawn_call_id,
                )
                _require_worker_stopped(stream, phase="timeout")
                done = self._read_done(
                    done_path,
                    fallback_status="partial",
                    fallback_summary="budget/timeout reached; salvaged current sandbox state",
                )
                return await self._finalize_workspace_recovery(
                    done,
                    workspace_root,
                    spawn_call_id=spawn_call_id,
                )

            now = time.monotonic()
            if now - last_heartbeat >= self.HEARTBEAT_INTERVAL_S:
                last_heartbeat = now
                await self.events.emit(
                    "cli.heartbeat",
                    {
                        "remaining_s": stream.remaining_s,
                        "stdout_chars": (
                            int(stream.stdout_chars)
                            if hasattr(stream, "stdout_chars")
                            else len(stream.stdout)
                        ),
                        "stderr_chars": (
                            int(stream.stderr_chars)
                            if hasattr(stream, "stderr_chars")
                            else len(stream.stderr)
                        ),
                        "stdout_dropped_chars": int(
                            getattr(stream, "stdout_dropped_chars", 0) or 0
                        ),
                        "stderr_dropped_chars": int(
                            getattr(stream, "stderr_dropped_chars", 0) or 0
                        ),
                    },
                    call_id=spawn_call_id,
                )
            await asyncio.sleep(self.POLL_INTERVAL_S)

    def _read_done(
        self,
        done_path: Path,
        *,
        fallback_status: DoneStatus,
        fallback_summary: str | None = None,
    ) -> CLIDoneRecord:
        if done_path.exists():
            try:
                data = json.loads(done_path.read_text(encoding="utf-8"))
                return CLIDoneRecord.model_validate(data)
            except (json.JSONDecodeError, Exception):
                return CLIDoneRecord(
                    status=fallback_status,
                    summary=fallback_summary or "(invalid done.json)",
                )
        return CLIDoneRecord(
            status=fallback_status,
            summary=fallback_summary or "(no done.json written)",
        )

    async def _emit_budget_warnings(
        self,
        warnings: list[tuple[str, str, float, float]],
    ) -> None:
        for scope, kind, used, limit in warnings:
            await self.events.emit(
                "budget.warn",
                {"scope": scope, "kind": kind, "used": used, "limit": limit},
            )

    def _effective_timeout_s(self) -> int:
        timeout_s = int(self.SANDBOX.timeout_s)
        remaining_s = self.tracker.remaining_wallclock_s()
        if remaining_s is not None:
            timeout_s = min(timeout_s, max(1, int(remaining_s)))
        if timeout_s <= 0:
            raise BudgetExhausted("run", "wallclock_s", 0.0, 0.0)
        return timeout_s

    def _effective_soft_timeout_s(self, hard_timeout_s: int) -> int:
        configured_soft = int(self.SOFT_TIMEOUT_S) if self.SOFT_TIMEOUT_S else 0
        if configured_soft <= 0 or hard_timeout_s <= 1:
            return 0
        configured_hard = max(1, int(self.SANDBOX.timeout_s))
        configured_grace = max(1, configured_hard - configured_soft)
        effective_grace = min(configured_grace, max(1, hard_timeout_s // 2))
        return min(configured_soft, max(1, hard_timeout_s - effective_grace))

    def _install_shell_startup(
        self,
        sandbox: Sandbox,
        *,
        bin_dir: Path,
        shim: Path,
        done_path: Path,
    ) -> None:
        visible_bin = self._shell_visible_path(sandbox, bin_dir)
        visible_shim = self._shell_visible_path(sandbox, shim)
        visible_done = self._shell_visible_path(sandbox, done_path)
        block = (
            f"{_SHELL_START_BLOCK_BEGIN}\n"
            f"export FINISH_DONE_PATH={shlex.quote(visible_done)}\n"
            f"export FINISH_BIN={shlex.quote(visible_shim)}\n"
            f"export PATH={shlex.quote(visible_bin)}:\"$PATH\"\n"
            f"{_SHELL_START_BLOCK_END}\n"
        )
        for name in (".bash_profile", ".profile", ".bashrc"):
            path = sandbox.root / name
            try:
                existing = path.read_text(encoding="utf-8") if path.exists() else ""
                updated = self._replace_shell_start_block(existing, block)
                path.write_text(updated, encoding="utf-8")
            except OSError:
                continue

    def _replace_shell_start_block(self, text: str, block: str) -> str:
        begin = text.find(_SHELL_START_BLOCK_BEGIN)
        end = text.find(_SHELL_START_BLOCK_END)
        if begin >= 0 and end >= begin:
            end += len(_SHELL_START_BLOCK_END)
            suffix = text[end:]
            if suffix.startswith("\n"):
                suffix = suffix[1:]
            return block + suffix
        if text:
            return block + "\n" + text
        return block

    def _shell_visible_path(self, sandbox: Sandbox, path: Path) -> str:
        try:
            rel = path.resolve().relative_to(sandbox.root.resolve())
        except (OSError, RuntimeError, ValueError):
            return str(path)
        if resolve_backend(self.SANDBOX) == "docker":
            rel_text = rel.as_posix()
            return "/work" if not rel_text else f"/work/{rel_text}"
        return str(path)


__all__ = [
    "CLIAgent",
    "CLIDoneRecord",
    "FINISH_SCRIPT",
    "measure_workspace_usage",
]
