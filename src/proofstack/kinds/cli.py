"""CLIAgent — drive an external coding CLI with the ``finish`` stop signal."""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import time
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from proofstack.agent import Agent
from proofstack.budget import BudgetExhausted
from proofstack.context import RunContext
from proofstack.events import new_call_id
from proofstack.sandbox import make_sandbox, resolve_backend
from proofstack.sandbox.base import Sandbox, SandboxSpec
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


DoneStatus = Literal["done", "partial", "blocked", "timeout", "error"]


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
    total_bytes = 0
    allocated_bytes = 0
    files = 0
    directories = 0
    errors = 0
    broken_symlinks = 0
    files_over_100_mib = 0
    files_over_1_gib = 0
    largest: list[tuple[int, str]] = []
    scan_truncated = False
    root = Path(root)
    stack = [Path(root)]
    while stack and not scan_truncated:
        directory = stack.pop()
        try:
            entries = os.scandir(directory)
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
                except OSError:
                    errors += 1
                    break
                try:
                    if entry.is_dir(follow_symlinks=False):
                        directories += 1
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
                except OSError:
                    errors += 1
                    continue
                files += 1
                size = max(0, stat_result.st_size)
                total_bytes += size
                allocated_bytes += max(0, getattr(stat_result, "st_blocks", 0)) * 512
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
                    (stop_after_bytes > 0 and total_bytes >= stop_after_bytes)
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
        "files": files,
        "directories": directories,
        "entries": files + directories,
        "scan_truncated": scan_truncated,
        "errors": errors,
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
    WORKSPACE_RESERVATION_BYTES: ClassVar[int] = 0
    WORKSPACE_RESERVATION_DIR: ClassVar[Path | None] = None
    WORKSPACE_CHECK_INTERVAL_S: ClassVar[float] = 5.0
    # 'finish'/'file': the completion record (done.json) is REQUIRED — a CLI
    # exit without it (or with an unparseable one) is an error even on rc 0.
    # 'exit': the exit code is the completion signal (codex-style CLIs that
    # never call finish). Plain subclasses keep the legacy 'exit' semantics;
    # ConfigurableCLIAgent sets this from its completion_signal config.
    COMPLETION_SIGNAL: ClassVar[str] = "exit"

    def __init__(
        self,
        ctx: RunContext,
        *,
        sandbox_root: Path | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(ctx, **kw)
        self.sandbox_root = Path(sandbox_root) if sandbox_root is not None else None

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
        return measure_workspace_usage(
            root,
            stop_after_bytes=self.WORKSPACE_HARD_LIMIT_BYTES,
            stop_after_entries=self.WORKSPACE_HARD_LIMIT_ENTRIES,
        )

    def _workspace_limit_failure(
        self,
        usage: dict[str, Any],
    ) -> tuple[str, str] | None:
        entries = int(usage.get("entries", usage.get("files", 0)) or 0)
        free_bytes = usage.get("filesystem_free_bytes")
        free_inodes = usage.get("filesystem_free_inodes")
        if (
            self.WORKSPACE_HARD_LIMIT_BYTES > 0
            and int(usage.get("bytes", 0) or 0) >= self.WORKSPACE_HARD_LIMIT_BYTES
        ):
            return (
                "workspace_hard_limit",
                f"workspace size {usage['bytes']} is at or above hard limit "
                f"{self.WORKSPACE_HARD_LIMIT_BYTES} bytes",
            )
        if (
            self.WORKSPACE_HARD_LIMIT_ENTRIES > 0
            and entries >= self.WORKSPACE_HARD_LIMIT_ENTRIES
        ):
            return (
                "workspace_entry_limit",
                f"workspace entry count {entries} is at or above hard limit "
                f"{self.WORKSPACE_HARD_LIMIT_ENTRIES}",
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
            and isinstance(free_inodes, int)
            and free_inodes < self.WORKSPACE_MIN_FREE_INODES
        ):
            return (
                "filesystem_min_free_inodes",
                f"filesystem free inodes {free_inodes} are below required reserve "
                f"{self.WORKSPACE_MIN_FREE_INODES}",
            )
        return None

    # --- framework-managed -----------------------------------------------------

    async def run(self, inp: BaseModel) -> BaseModel:  # type: ignore[override]
        if not self.CLI_CMD:
            raise RuntimeError(f"{type(self).__name__}.CLI_CMD is empty")

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
        sandbox: Sandbox | None = None
        storage_lease: StorageReservationLease | None = None
        try:
            root = self.sandbox_root_for(inp)
            if root is not None:
                root.mkdir(parents=True, exist_ok=True)
                sandbox = make_sandbox(self.SANDBOX, root=root)
                persistent = True
            else:
                sandbox = make_sandbox(self.SANDBOX, root=self.workdir / "sandbox")
                persistent = False
            if persistent:
                runtime_dir = sandbox.root / ".pwc" / "runtime"
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
                reservation_dir = self.workspace_reservation_dir_for(sandbox.root)
                storage_lease = StorageReservationLease(
                    registry_dir=reservation_dir,
                    workspace_root=sandbox.root,
                    requested_bytes=self.WORKSPACE_RESERVATION_BYTES,
                    minimum_free_bytes=self.WORKSPACE_MIN_FREE_BYTES,
                    owner=f"{self.ctx.run_id}:{self.name}",
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

            await self.setup(sandbox, inp)

            if persistent and self._workspace_monitor_enabled():
                usage = await asyncio.to_thread(
                    self._measure_workspace,
                    sandbox.root,
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
                    },
                )
                failure = self._workspace_limit_failure(usage)
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

            spawn_call_id = new_call_id()
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
                },
                call_id=spawn_call_id,
            )

            stream = await sandbox.stream_command(
                self.CLI_CMD,
                env_extra=extra_env,
                extra_path=[bin_dir],
                timeout_s=timeout_s,
            )
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

            done = await self._wait_for_done(
                stream,
                done_path,
                spawn_call_id=spawn_call_id,
                wrap_up_path=wrap_up_path,
                soft_timeout_s=soft_timeout_s,
                workspace_root=sandbox.root if persistent else None,
            )
            # Meter as a retained, shielded task. A cancellation landing while
            # record_cli_usage runs must not skip it (that loses a detected
            # rate limit); the finally awaits this exact task instead of
            # re-running metering, so the ledger is never double-counted (A7).
            meter_task = asyncio.ensure_future(
                self.record_cli_usage(stream.stdout, stream.stderr, done)
            )
            usage_recorded = True
            try:
                await asyncio.shield(meter_task)
            except Exception as e:  # a real failure; cancellation propagates
                await self.events.emit(
                    "cli.usage_record_failed",
                    {"type": type(e).__name__, "msg": str(e)},
                    call_id=spawn_call_id,
                )
            out = await self.collect(sandbox, inp, done)
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
            elif stream is not None and not usage_recorded:
                try:
                    await asyncio.shield(
                        self.record_cli_usage(
                            stream.stdout,
                            stream.stderr,
                            CLIDoneRecord(status="partial", summary="(cancelled mid-run)"),
                        )
                    )
                except (asyncio.CancelledError, Exception):
                    pass
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
            # Keep the sandbox dir on disk so the workdir captures artifacts.
            # teardown() still runs so subclasses can scrub per-invocation
            # secrets such as copied CLI credentials.
            try:
                if sandbox is not None:
                    try:
                        await self.teardown(sandbox, inp)
                    except Exception as e:
                        await self.events.emit(
                            "cli.teardown_error",
                            {"type": type(e).__name__, "msg": str(e)},
                        )
            finally:
                if storage_lease is not None and storage_lease.status is not None:
                    reservation_status = storage_lease.status
                    try:
                        storage_lease.release()
                    except Exception as e:
                        try:
                            await self.events.emit(
                                "cli.storage_release_failed",
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
                                "cli.storage_released",
                                {
                                    "workspace": str(sandbox.root) if sandbox else None,
                                    "reserved_bytes": reservation_status.requested_bytes,
                                },
                            )
                        except Exception:
                            pass

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
        last_workspace_check = 0.0
        cleanup_warned = False
        wrap_up_signaled = False
        workspace_soft_warned = False
        while True:
            now = time.monotonic()
            if (
                workspace_root is not None
                and self._workspace_monitor_enabled()
                and now - last_workspace_check >= self.WORKSPACE_CHECK_INTERVAL_S
            ):
                last_workspace_check = now
                usage = await asyncio.to_thread(
                    self._measure_workspace,
                    workspace_root,
                )
                await self.events.emit(
                    "cli.workspace_usage",
                    {
                        **usage,
                        "phase": "running",
                        "soft_limit_bytes": self.WORKSPACE_SOFT_LIMIT_BYTES or None,
                        "hard_limit_bytes": self.WORKSPACE_HARD_LIMIT_BYTES,
                        "soft_limit_entries": self.WORKSPACE_SOFT_LIMIT_ENTRIES or None,
                        "hard_limit_entries": self.WORKSPACE_HARD_LIMIT_ENTRIES or None,
                        "min_free_bytes": self.WORKSPACE_MIN_FREE_BYTES or None,
                        "min_free_inodes": self.WORKSPACE_MIN_FREE_INODES or None,
                    },
                    call_id=spawn_call_id,
                )
                failure = self._workspace_limit_failure(usage)
                if failure is not None:
                    await stream.terminate()
                    reason, detail = failure
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
                        },
                        call_id=spawn_call_id,
                    )
                    return CLIDoneRecord(
                        status="error",
                        summary=f"workspace safety limit reached; stopped CLI: {detail}",
                    )
                if (
                    not workspace_soft_warned
                    and (
                        (
                            self.WORKSPACE_SOFT_LIMIT_BYTES > 0
                            and usage["bytes"] >= self.WORKSPACE_SOFT_LIMIT_BYTES
                        )
                        or (
                            self.WORKSPACE_SOFT_LIMIT_ENTRIES > 0
                            and int(
                                usage.get("entries", usage.get("files", 0)) or 0
                            )
                            >= self.WORKSPACE_SOFT_LIMIT_ENTRIES
                        )
                    )
                ):
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
            if done_path.exists():
                grace_deadline = time.monotonic() + float(self.DONE_DRAIN_GRACE_S)
                while (
                    not stream.done
                    and time.monotonic() < grace_deadline
                    and stream.remaining_s > 0
                ):
                    await asyncio.sleep(self.POLL_INTERVAL_S)
                await stream.terminate()
                return self._read_done(done_path, fallback_status="done")
            if stream.done:
                # CLI exited without calling finish. Use the exit
                # code as the done signal: 0 == clean termination == done,
                # non-zero == failure == error. This is a pragmatic
                # default for agents that don't (yet) wire finish
                # reliably. TODO(SPEC §13): harden the explicit
                # finish handshake and make exit-as-done opt-in.
                rc = stream.proc.returncode
                fallback = "done" if rc == 0 else "error"
                try:
                    await stream.terminate()
                except Exception:
                    pass
                stderr_tail = self.sanitize_cli_output(stream.stderr or "")[-2000:]
                stdout_tail = self.sanitize_cli_output(stream.stdout or "")[-1000:]
                await self.events.emit(
                    "cli.exit",
                    {
                        "sandbox_id": str(self.workdir),
                        "exit_code": rc,
                        "status": fallback,
                        "via_finish": False,
                        "stderr_tail": stderr_tail,
                        "stdout_tail": stdout_tail,
                    },
                    call_id=spawn_call_id,
                )
                return self._read_done(done_path, fallback_status=fallback)
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
                return self._read_done(
                    done_path,
                    fallback_status="partial",
                    fallback_summary="budget/timeout reached; salvaged current sandbox state",
                )

            now = time.monotonic()
            if now - last_heartbeat >= self.HEARTBEAT_INTERVAL_S:
                last_heartbeat = now
                await self.events.emit(
                    "cli.heartbeat",
                    {
                        "remaining_s": stream.remaining_s,
                        "stdout_chars": len(stream.stdout),
                        "stderr_chars": len(stream.stderr),
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
