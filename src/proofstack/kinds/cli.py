"""CLIAgent — drive an external coding CLI with the ``finish`` stop signal."""
from __future__ import annotations

import asyncio
import json
import shlex
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

    def sandbox_root_for(self, inp: BaseModel) -> Path | None:
        """Return a persistent sandbox root for this invocation, if any."""
        return self.sandbox_root

    # --- framework-managed -----------------------------------------------------

    async def run(self, inp: BaseModel) -> BaseModel:  # type: ignore[override]
        if not self.CLI_CMD:
            raise RuntimeError(f"{type(self).__name__}.CLI_CMD is empty")

        await self._emit_budget_warnings(self.tracker.check())
        self.tracker.add_tool_call()
        await self._emit_budget_warnings(self.tracker.check())

        root = self.sandbox_root_for(inp)
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            sandbox = make_sandbox(self.SANDBOX, root=root)
            persistent = True
        else:
            sandbox = make_sandbox(self.SANDBOX, root=self.workdir / "sandbox")
            persistent = False
        # Track the streaming process so the finally block can terminate
        # it unconditionally on cancellation. Without this, if the
        # surrounding task is cancelled while ``_wait_for_done`` is
        # awaiting, the codex (or other CLI) child keeps running until
        # its own timeout or until the container exits.
        stream = None
        try:
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

            await self.setup(sandbox, inp)

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
            )
            try:
                await self.record_cli_usage(stream.stdout, stream.stderr, done)
            except Exception as e:
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
            # us here.
            if stream is not None:
                try:
                    await asyncio.shield(stream.terminate())
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                stdout_text = stream.stdout if stream is not None else ""
                stderr_text = stream.stderr if stream is not None else ""
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
                await self.teardown(sandbox, inp)
            except Exception as e:
                await self.events.emit(
                    "cli.teardown_error",
                    {"type": type(e).__name__, "msg": str(e)},
                )

    async def _wait_for_done(
        self,
        stream,
        done_path: Path,
        *,
        spawn_call_id: str,
        wrap_up_path: Path | None = None,
        soft_timeout_s: int = 0,
    ) -> CLIDoneRecord:
        spawn_t = time.monotonic()
        last_heartbeat = spawn_t
        cleanup_warned = False
        wrap_up_signaled = False
        while True:
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
                stderr_tail = (stream.stderr or "")[-2000:]
                stdout_tail = (stream.stdout or "")[-1000:]
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


__all__ = ["CLIAgent", "CLIDoneRecord", "FINISH_SCRIPT"]
