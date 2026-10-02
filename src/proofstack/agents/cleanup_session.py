"""Persistent Claude Code editor with supervised, API-billed review tools."""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from proofstack.agent import Agent
from proofstack.agents.cleanup_tools import cleanup_tools
from proofstack.agents.cleanup_reviews import CleanupReviews
from proofstack.agents.configurable_cli import ConfigurableCLIAgent
from proofstack.agents.linear_read import linear_read as _linear_read, make_client as _linear_client
from proofstack.agents.writeup_loop import _compile_raw, _GateCanceller
from proofstack.atomic import write_text_atomic
from proofstack.budget import BudgetExhausted, BudgetRegistry, BudgetSpec
from proofstack.cli_usage import (
    _CLAUDE_MODEL_TOKEN_FIELDS, _usage_from_result_object, cost_for_claude_usage,
    cost_for_codex_usage, load_cost_rates, parse_claude_json, parse_codex_jsonl,
)
from proofstack.cleanup_runtime import check_cleanup_runtime
from proofstack.context import component_config_for_class
from proofstack.kinds.cli import CLIDoneRecord
from proofstack.latex_contract import render_firstproof_latex_contract
from proofstack.sandbox.base import SandboxSpawnError, WorkerStopState
from proofstack.sandbox.memory import GiB, MemoryPolicy, check_memory_registry


PROMPTS = Path(__file__).with_name("writeup_prompts")
MAX_FILE_BYTES = 2 * 1024 * 1024


class CleanupAccountingUncertain(RuntimeError):
    requires_reconciliation = True


class CleanupIncomplete(RuntimeError):
    """The editor stopped before finishing; ``output`` holds its last answer.tex."""

    def __init__(self, message, output):
        super().__init__(message)
        self.output = output


LEAD_MONITOR_MARGIN = 1.2


class CleanupUnavailable(RuntimeError):
    """The configured CLI backend cannot run; the API editor may be used."""


def _remaining_usd(tracker):
    amounts = [t.spec.max_usd - t.counters.usd for t in tracker.chain()
               if t.spec and t.spec.max_usd is not None and t.spec.max_usd >= 0]
    if not amounts or not math.isfinite(min(amounts)) or min(amounts) <= 0:
        raise ValueError("cleanup requires a positive finite remaining USD budget")
    return min(amounts)


def _top_up_review_budget(helper, invocation, parent, *, episode_remaining, max_topup):
    parent.check()
    # Preserve every dollar already promised to the still-running editor and
    # linear reader, including usage that has not arrived in the ledger yet.
    committed = max(invocation.spec.max_usd, invocation.counters.usd)
    unspent = committed - invocation.counters.usd
    grant = max(0.0, min(max_topup, episode_remaining - committed,
                         _remaining_usd(parent) - unspent))
    if grant:
        helper.spec = helper.spec.model_copy(update={"max_usd": helper.spec.max_usd + grant})
        invocation.spec = invocation.spec.model_copy(update={"max_usd": invocation.spec.max_usd + grant})
    return grant


def _read_file(root: Path, name: str, *, optional=False) -> str:
    path = root / name
    if path.parent.resolve() != root.resolve() and not path.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError("cleanup file escapes workspace")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        if optional:
            return ""
        raise
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_FILE_BYTES:
            raise ValueError(f"unsafe or oversized cleanup file: {name}")
        data = source.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"oversized cleanup file: {name}")
    return data.decode("utf-8")


def _save(path: Path, value):
    write_text_atomic(path, json.dumps(value, indent=2))


def _tokens_cumulative(usage):
    # parse_claude_json takes result tokens from complete modelUsage, which,
    # like total_cost_usd, covers the whole resumed session.
    return bool(usage.model_usage) and all(
        all(key in raw for key in _CLAUDE_MODEL_TOKEN_FIELDS.values()) for raw in usage.model_usage.values())


def _claude_transcript(workspace: Path, session_id: str) -> Path | None:
    return next(iter((workspace / ".claude" / "projects").glob(f"*/{session_id}.jsonl")), None)


def _saved_claude_total(transcript: Path | None, offset: int, cost_config: str):
    """Cumulative (usd, tokens) that Claude saved after ``offset``, which --resume restores."""
    if transcript is None:
        return None
    last = None
    try:
        with os.fdopen(os.open(transcript, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                return None
            source.seek(offset)
            for line in source:
                if b"cost-state" in line:
                    last = line
        record = json.loads(last) if last else None
        if not isinstance(record, dict) or record.get("type") != "cost-state":
            return None
        usage = _usage_from_result_object({"total_cost_usd": record["totalCostUSD"],
                                           "modelUsage": record.get("modelUsage") or {}})
        return cost_for_claude_usage(usage, cost_config=cost_config), usage.metered_tokens
    except (OSError, KeyError, ValueError):
        return None


def cleanup_accounting_unresolved(run_root: Path) -> bool:
    if (run_root / "cleanup-accounting-uncertain.json").exists():
        return True
    # SIGKILL cannot run an exception handler. Check the durable pre-launch
    # markers as well before admitting any new paid phase after a restart.
    for path in (run_root / "cleanup_sessions").glob("*/session.json"):
        try:
            state = json.loads(_read_file(path.parent, path.name))
            if not isinstance(state, dict) or state.get("in_flight", True):
                return True
        except (OSError, ValueError):
            return True
    return False


class CleanupSettings(BaseModel):
    model: str = "claude-fable-5-1"
    effort: str = "xhigh"
    cost_config: str = "models/anthropic/fable_51"
    codex_model: str = "gpt-6-astra"
    codex_effort: str = "xhigh"
    codex_cost_config: str = "models/openai/gpt-6-astra"
    # Reserved separately because Claude's cap does not cover external Codex.
    codex_budget_fraction: float = Field(default=0.3, ge=0, lt=1)
    # Prefix-only define-before-use reader; empty model disables the tool.
    linear_read_model: str = "models/anthropic/sonnet_5"
    linear_read_budget_fraction: float = Field(default=0.1, ge=0, lt=1)
    max_invocations: int = Field(default=3, ge=1, le=10)
    max_invocation_usd: float = Field(default=150, gt=0, allow_inf_nan=False)
    max_episode_usd: float = Field(default=500, gt=0, allow_inf_nan=False)
    # One failed mandatory review may borrow unallocated enclosing funds;
    # the active editor's promised allowance is never taken away.
    max_review_topup_usd: float = Field(default=75, ge=0, allow_inf_nan=False)
    review_retry_reserve_fraction: float = Field(default=0.2, ge=0, lt=1)
    finishing_seconds: float = Field(default=900, ge=0, allow_inf_nan=False)
    # Measured peak RSS: editor 0.33 GB, Codex 0.14 GB. One slot per problem
    # (up to 10 on the 64 GB host), so the shared cutoff never queues.
    memory_gb: float = Field(default=1, gt=0, allow_inf_nan=False)
    # Below Compute's 16 GB reserve: Compute's watchdog frees memory first, and
    # an editor is killed only when the host is genuinely short.
    memory_reserve_gb: float = Field(default=4, ge=0, allow_inf_nan=False)
    max_parallel_editors: int = Field(default=10, ge=1)
    max_parallel_codex_reviews: int = Field(default=10, ge=1)
    workspace_bytes: int = Field(default=2 * 1024**3, ge=MAX_FILE_BYTES)


async def check_cleanup_available(settings: CleanupSettings):
    from mathagents.config_loader import load_solver_config
    for model, ref in ((settings.model, settings.cost_config),
                       (settings.codex_model, settings.codex_cost_config)):
        cfg = load_solver_config(ref)
        if cfg.get("model") != model:
            raise ValueError(f"cleanup billing config does not match model {model}")
    load_cost_rates(settings.cost_config, require_cache_rates=True)
    load_cost_rates(settings.codex_cost_config)
    if settings.linear_read_model:
        load_cost_rates(settings.linear_read_model)
        if settings.codex_budget_fraction + settings.linear_read_budget_fraction >= 1:
            raise ValueError("cleanup helper budget fractions leave nothing for the editor")
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        raise CleanupUnavailable("cleanup requires ANTHROPIC_API_KEY; subscription fallback is disabled")
    if settings.codex_budget_fraction and not os.environ.get("OPENAI_API_KEY", "").strip():
        raise CleanupUnavailable("cleanup Codex reviews require OPENAI_API_KEY")
    try:
        return await asyncio.to_thread(check_cleanup_runtime, require_codex=bool(settings.codex_budget_fraction))
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        raise CleanupUnavailable(str(exc)) from exc


def cleanup_memory_policy(settings: CleanupSettings, registry_root: Path, role: Literal["editors", "codex"]):
    # Separate role pools avoid all editor slots waiting for Codex slots
    # they themselves occupy. Both retain the host-memory emergency floor.
    return MemoryPolicy(
        registry=registry_root / ".proofcouncil-cleanup-memory" / role,
        max_workers=settings.max_parallel_editors if role == "editors" else settings.max_parallel_codex_reviews,
        worker_bytes=int(settings.memory_gb * GiB),
        reserve_bytes=int(settings.memory_reserve_gb * GiB),
    )


async def check_cleanup_launch(inputs: dict, components: dict, registry_root: Path):
    if inputs.get("cleanup_backend", "claude_code") != "claude_code":
        return []
    config = component_config_for_class(components, CleanupSession, "CleanupSession")
    settings = CleanupSettings.model_validate(config.get("cleanup", {}))
    probes = []
    for role in ("editors", "codex") if settings.codex_budget_fraction else ("editors",):
        policy = cleanup_memory_policy(settings, registry_root, role)
        await asyncio.to_thread(check_memory_registry, policy)
        probes.append({"role": role, "registry": str(policy.registry), "ok": True,
                       "max_workers": policy.max_workers, "worker_bytes": policy.worker_bytes,
                       "reserve_bytes": policy.reserve_bytes})
    await check_cleanup_available(settings)
    return probes


class _MeteredTurn(ConfigurableCLIAgent):
    """One-use instance; bill once after exit, inspect live usage for admission."""
    # Cumulative Claude session (usd, tokens) already billed before this process.
    claude_baseline = (0.0, 0)
    claude_total = None

    def _claude_baseline_for(self, cost):
        # A lower total is not a continuation of the billed session.
        return self.claude_baseline if cost >= self.claude_baseline[0] - 1e-9 else (0.0, 0)

    def _claude_process_share(self, usage, cost, tokens):
        base_usd, base_tokens = self._claude_baseline_for(cost)
        own_tokens = max(0, tokens - base_tokens) if _tokens_cumulative(usage) else tokens
        self.claude_total = (cost, base_tokens + own_tokens)
        return max(0.0, cost - base_usd), own_tokens
    def extra_env(self, sandbox, inp):
        env = super().extra_env(sandbox, inp)
        if self.component_config["usage"]["type"] == "codex_jsonl":
            # Noninteractive Codex reads CODEX_API_KEY, not OPENAI_API_KEY.
            # Bind only in the child environment; never persist the key in config.
            key = sandbox.spec.build_env(sandbox_root=sandbox.root).get("OPENAI_API_KEY", "")
            if not key.strip():
                raise ValueError("cleanup Codex review requires a sandbox OPENAI_API_KEY")
            env["CODEX_API_KEY"] = key
        return env

    async def run(self, inp):
        try:
            return await super().run(inp)
        except BaseException as exc:
            if isinstance(exc, SandboxSpawnError) or not getattr(self, "launch_attempted", False):
                self.proven_not_started = True
            raise

    async def setup(self, sandbox, inp):
        try:
            await super().setup(sandbox, inp)
            (sandbox.root / "completion.json").unlink(missing_ok=True)
            spawn = sandbox.stream_command

            async def stream_command(*args, **kwargs):
                self.launch_attempted = True
                try:
                    return await spawn(*args, **kwargs)
                except BaseException:
                    # Admission errors and cancellation while queued leave the
                    # sandbox settled and stopped: no worker ever existed.
                    if (getattr(sandbox, "worker_launch_settled", False)
                            and getattr(sandbox, "worker_stop_state", None) is WorkerStopState.STOPPED):
                        self.launch_attempted = False
                    raise
            sandbox.stream_command = stream_command
            if self.component_config["usage"]["type"] == "codex_jsonl":
                # Codex rejects an explicit CODEX_HOME that does not yet exist.
                home = sandbox.root / ".codex"
                home.mkdir(mode=0o700, exist_ok=True)
                if home.is_symlink():
                    raise ValueError("unsafe cleanup Codex home")
        except BaseException:
            self.proven_not_started = True
            raise

    async def record_cli_usage(self, stdout_text, stderr_text, done):
        self.final_usage = (parse_claude_json(stdout_text).has_result
                            if self.component_config["usage"]["type"] == "claude_json" else True)
        stream = getattr(self, "metering_stream", None)
        if stream is not None and (getattr(stream, "usage_capture_dropped_chars", 0)
                                   or getattr(stream, "usage_capture_oversized_lines", 0)):
            # A valid Claude terminal total covers the whole invocation. Codex
            # totals cover only one turn, so lost earlier records cannot be ignored.
            if self.component_config["usage"]["type"] != "claude_json" or not self.final_usage:
                raise RuntimeError("cleanup usage capture is incomplete")
        await super().record_cli_usage(stdout_text, stderr_text, done)
        self.metered = True

    async def _wait_for_done(self, stream, done_path, **kwargs):
        self.metering_stream = stream
        async def watch():
            while True:
                await asyncio.sleep(1)
                if stream.done:
                    return None
                raw = getattr(stream, "metering_stdout", None) or stream.stdout or ""
                cfg = self.component_config["usage"]
                try:
                    if cfg["type"] == "claude_json":
                        usage = parse_claude_json(raw)
                        cost = cost_for_claude_usage(usage, cost_config=cfg["cost_config"]) if usage.found else 0
                        if usage.has_result:
                            cost = max(0.0, cost - self._claude_baseline_for(cost)[0])
                    else:
                        usage = parse_codex_jsonl(raw)
                        cost = cost_for_codex_usage(usage, **load_cost_rates(cfg["cost_config"]))
                    try:
                        # The turn's own spec is its cap; live_usd_limit may exceed it.
                        pool = _remaining_usd(self.tracker.parent)
                    except ValueError:
                        pool = 0
                    reason = "recorded cleanup allowance reached" if cost >= min(self.live_usd_limit, pool) else None
                    self.tracker.check()
                except (BudgetExhausted, ValueError) as exc:
                    # Raising would discard the draft; the run deadline trips
                    # here while the waiter is still stopping the worker.
                    reason = f"cleanup live check stopped the worker: {exc}"
                if reason is not None:
                    await stream.terminate()
                    return reason

        waiter = asyncio.create_task(super()._wait_for_done(stream, done_path, **kwargs))
        monitor = asyncio.create_task(watch())
        try:
            completed, _ = await asyncio.wait((waiter, monitor), return_when=asyncio.FIRST_COMPLETED)
            if waiter in completed:
                return await waiter
            if monitor in completed:
                reason = await monitor
                if reason is not None:
                    return CLIDoneRecord(status="error", summary=reason)
            return await waiter
        finally:
            for task in (waiter, monitor):
                if not task.done():
                    task.cancel()
            await asyncio.gather(waiter, monitor, return_exceptions=True)


class CleanupSession(Agent):
    """An editorial episode can span multiple invocations and critic repairs."""
    cache_enabled: ClassVar[bool] = False

    class Inputs(BaseModel):
        problem: str
        document: str
        baseline: str = ""
        research_notes: str = ""
        findings: str = ""
        constraints: str = ""
        partial: bool = False
        finishing_only: bool = False
        page_limit: int = Field(default=16, ge=1, le=200)
        session_key: str = Field(default="standalone", pattern=r"^[A-Za-z0-9_-]{1,100}$")

    class Outputs(BaseModel):
        answer_tex: str
        feedback_md: str
        status: Literal["ready", "unable", "incomplete"]
        summary: str
        workspace: Path
        session_id: str

    class _Completion(BaseModel):
        status: Literal["ready", "unable"]
        summary: str = ""

    async def _preflight(self, settings):
        await check_cleanup_available(settings)

    def _turn(self, cfg, name, usd, seconds, *, ctx=None):
        # Do not inherit generic '*' CLI configuration or subscription auth.
        parent = ctx.budgets.root() if ctx is not None else self.tracker
        ctx = replace(ctx or self.ctx, component_configs={name: cfg})
        turn = _MeteredTurn(ctx, name=name, budget=BudgetSpec(max_usd=usd, max_wallclock_s=seconds))
        turn.tracker.parent = parent
        turn.live_usd_limit = usd
        turn.metered = False
        turn.final_usage = False
        turn.proven_not_started = False
        return turn

    def _sandbox(self, settings, seconds, provider):
        role = "editors" if provider == "ANTHROPIC_API_KEY" else "codex"
        policy = cleanup_memory_policy(settings, self.ctx.root_workdir.parent, role)
        return {"backend": "subprocess", "memory_gb": settings.memory_gb,
                "limit_address_space": False, "timeout_s": max(1, int(seconds)),
                "memory_policy": policy,
                "provider_keys": [provider], "env_allowlist": ["PATH", "LANG", "LC_ALL"]}

    async def run(self, inp):
        accounting_guard = self.ctx.root_workdir / "cleanup-accounting-uncertain.json"
        if cleanup_accounting_unresolved(self.ctx.root_workdir):
            raise CleanupAccountingUncertain("cleanup usage is unresolved; reconcile it before continuing this run")
        settings = CleanupSettings.model_validate(self.component_config.get("cleanup", {}))
        await self._preflight(settings)
        self.tracker.check()
        _remaining_usd(self.tracker)
        seconds = self.tracker.remaining_wallclock_s()
        if seconds is None or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("cleanup requires a positive finite wallclock budget")
        root = self.ctx.root_workdir / "cleanup_sessions" / inp.session_key
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or root.parent.is_symlink():
            raise ValueError("unsafe cleanup session directory")
        lock = os.open(root / "supervisor.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                return await self._locked_run(inp, settings, root, time.monotonic() + seconds)
            except BaseException as exc:
                state_file = root / "session.json"
                if state_file.exists() and json.loads(_read_file(root, "session.json")).get("in_flight"):
                    _save(accounting_guard, {"session_key": inp.session_key, "reason": str(exc)})
                    if isinstance(exc, Exception):
                        raise CleanupAccountingUncertain(
                            "cleanup invocation/usage is unresolved; reconcile before continuing: " + str(exc)
                        ) from exc
                raise
        finally:
            os.close(lock)

    async def _locked_run(self, inp, settings, root, deadline):
        baseline = inp.baseline or inp.document
        identity = hashlib.sha256(json.dumps({
            "problem": inp.problem, "baseline": baseline, "partial": inp.partial,
            "page_limit": inp.page_limit, "settings": settings.model_dump(),
        }, sort_keys=True).encode()).hexdigest()
        state_path = root / "session.json"
        state = json.loads(_read_file(root, "session.json")) if state_path.exists() else {
            "identity": identity, "session_id": str(uuid.uuid4()), "resumable": False, "in_flight": False,
            "recorded_usd": 0.0, "claude_session_usd": 0.0, "claude_session_tokens": 0,
        }
        if state["identity"] != identity:
            raise ValueError("cleanup session belongs to different inputs/settings; use a new session_key")
        if state["in_flight"]:
            raise RuntimeError("cleanup session has an interrupted/unmetered invocation; reconcile it before reuse")
        spent = state.get("recorded_usd")
        if type(spent) not in (int, float) or not math.isfinite(spent) or spent < 0:
            raise CleanupAccountingUncertain("cleanup episode cost ledger is missing or invalid")
        claude_usd = state.setdefault("claude_session_usd", 0.0)
        claude_tokens = state.setdefault("claude_session_tokens", 0)
        if (type(claude_usd) not in (int, float) or not math.isfinite(claude_usd) or claude_usd < 0
                or type(claude_tokens) is not int or claude_tokens < 0):
            raise CleanupAccountingUncertain("cleanup Claude session total is invalid")
        _save(state_path, state)
        write_text_atomic(root / "baseline.tex", baseline)
        workspace = root / "workspace"
        files = {"baseline.tex": baseline, "problem.md": inp.problem,
                 "research-notes.md": inp.research_notes or _read_file(workspace, "research-notes.md", optional=True),
                 "findings.md": inp.findings,
                 "constraints.md": render_firstproof_latex_contract(inp.page_limit, can_compile=True)
                    + "\n" + inp.constraints + ("\nUNSOLVED: preserve all gaps. No post-cleanup mathematical review."
                                               if inp.partial else "\nSolved candidate: external critic acceptance is still required."),
                 "WRITING_GUIDANCE.md": (PROMPTS / "WRITING_GUIDANCE.md").read_text()}
        document = inp.document
        malformed_completion = False
        initial_remaining = max(0.0, deadline - time.monotonic())
        state.setdefault("finishing_at_unix_s", time.time() + initial_remaining
                         - min(settings.finishing_seconds, initial_remaining * 0.2))
        finishing_deadline = min(deadline, time.monotonic() + state["finishing_at_unix_s"] - time.time())
        _save(state_path, state)
        for attempt in range(settings.max_invocations):
            self.tracker.check()
            allowance = min(_remaining_usd(self.tracker), settings.max_invocation_usd,
                            settings.max_episode_usd - state["recorded_usd"])
            if allowance <= 0:
                raise RuntimeError("cleanup episode spending ceiling reached")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("cleanup deadline reached")
            retry_reserve = (min(settings.max_review_topup_usd, allowance * settings.review_retry_reserve_fraction)
                             if settings.codex_budget_fraction else 0.0)
            allocated = allowance - retry_reserve
            helper_allowance = allocated * settings.codex_budget_fraction
            linear_allowance = allocated * settings.linear_read_budget_fraction if settings.linear_read_model else 0
            lead_allowance = allocated - helper_allowance - linear_allowance
            invocation_registry = BudgetRegistry()
            invocation_tracker = invocation_registry.register_root(
                "run", BudgetSpec(max_usd=allocated, max_wallclock_s=remaining),
            )
            invocation_tracker.parent = self.tracker
            invocation_ctx = replace(self.ctx, budgets=invocation_registry)
            registry = BudgetRegistry()
            helper_tracker = registry.register_root("run", BudgetSpec(max_usd=helper_allowance,
                                                                       max_wallclock_s=remaining))
            helper_tracker.parent = invocation_tracker
            helper_ctx = replace(self.ctx, budgets=registry)
            helper_lock = asyncio.Lock()
            unsettled_helpers = set()
            linear_tracker = BudgetRegistry().register_root("run", BudgetSpec(max_usd=linear_allowance,
                                                                               max_wallclock_s=remaining))
            linear_tracker.parent = invocation_tracker
            linear_lock = asyncio.Lock()
            compile_lock = asyncio.Lock()
            invocation_error = None
            review_topup_used = False
            finishing_window = max(0.0, deadline - finishing_deadline)

            def progress():
                seconds_left = max(0, deadline - time.monotonic())
                finishing = inp.finishing_only or state.get("finishing", False) or time.monotonic() >= finishing_deadline
                if finishing and not state.get("finishing"):
                    state["finishing"] = True
                    _save(state_path, state)
                return {
                    "stage": "finishing" if finishing else "editing", "seconds_remaining": seconds_left,
                    "instruction": (
                        "Finish now: no optional polishing, broad reviews, or new native reviewer tasks. "
                        "Retrieve existing reports; complete any missing mandatory attribution check, "
                        "resolve outstanding findings, perform the final linear read, compile the exact final "
                        "answer.tex and write completion.json. Preserve gaps. Do not restart the rewrite."
                        if finishing else "Keep the manuscript within the page target throughout editing."
                    ),
                }

            def admit_review(purpose):
                nonlocal review_topup_used
                if progress()["stage"] == "finishing" and (purpose != "attribution" or reviews.attribution_retrieved()):
                    raise RuntimeError("Finishing stage: no new optional reviews; retrieve existing reports and complete the manuscript.")
                if unsettled_helpers:
                    raise CleanupAccountingUncertain("cleanup helper shutdown or usage is unresolved")
                if (purpose == "attribution" and not reviews.attribution_retrieved() and not review_topup_used
                        and helper_allowance > 0
                        and helper_tracker.spec.max_usd - helper_tracker.counters.usd <= helper_allowance * 0.25
                        and any(r["purpose"] == "attribution" and r["status"] == "failed"
                                for r in reviews.records.values())):
                    grant = _top_up_review_budget(
                        helper_tracker, invocation_tracker, self.tracker,
                        episode_remaining=settings.max_episode_usd - state["recorded_usd"],
                        max_topup=settings.max_review_topup_usd,
                    )
                    if grant:
                        review_topup_used = True
                        state.setdefault("review_topups", []).append({"usd": grant, "reason": "failed mandatory attribution review"})
                        _save(state_path, state)
                helper_tracker.check()

            async def list_files():
                reviews = workspace / "reviews"
                return [*files, "answer.tex", "feedback.md", "completion.json",
                        *[f"reviews/{p.name}" for p in sorted(reviews.glob("*.md"))]]

            async def read_file(path):
                if path not in await list_files():
                    raise ValueError("file is not part of this editorial workspace")
                content = _read_file(workspace, path, optional=path in {"feedback.md", "completion.json"})
                return {"content": content, "sha256": hashlib.sha256(content.encode()).hexdigest()}

            async def write_file(path, content):
                if path not in {"answer.tex", "feedback.md", "completion.json"}:
                    raise ValueError("only answer.tex, feedback.md and completion.json are editable")
                if len(content.encode()) > MAX_FILE_BYTES:
                    raise ValueError("editorial file exceeds size limit")
                write_text_atomic(workspace / path, content)
                return {"path": path, "sha256": hashlib.sha256(content.encode()).hexdigest()}

            async def compile_snapshot():
                tex = _read_file(workspace, "answer.tex")
                canceller = _GateCanceller()
                job = asyncio.create_task(asyncio.to_thread(_compile_raw, tex, None,
                                           deadline=deadline, canceller=canceller, secure=True))
                try:
                    ok, pages, detail = await asyncio.shield(job)
                except asyncio.CancelledError:
                    canceller.cancel()
                    await asyncio.shield(job)
                    raise
                result = {"compiled": ok, "pages": pages, "page_limit": inp.page_limit,
                          "within_limit": ok and 0 < pages <= inp.page_limit, "diagnostics": detail}
                await self.events.emit("cleanup.compile", result)
                return result

            async def compile_document():
                async with compile_lock:
                    self.tracker.check()
                    return await compile_snapshot()

            def review_snapshot():
                return {"baseline.tex": baseline, **{name: _read_file(workspace, name, optional=True) for name in
                        ("answer.tex", "problem.md", "research-notes.md", "findings.md", "constraints.md", "feedback.md")}}

            async def codex_review(task, *, review_id=None, snapshot=None):
                if not settings.codex_budget_fraction:
                    raise ValueError("Codex review allowance is disabled")
                if len(task) > 20000:
                    raise ValueError("review task is too long")
                async with helper_lock:
                    if unsettled_helpers:
                        raise CleanupAccountingUncertain("cleanup helper shutdown or usage is unresolved")
                    helper_tracker.check()
                    usd = _remaining_usd(helper_tracker)
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        raise TimeoutError("cleanup deadline reached")
                    review_id = review_id or uuid.uuid4().hex
                    review_root = root / "codex" / review_id
                    snapshot = review_snapshot() if snapshot is None else snapshot
                    prompt = ("You are an independent mathematical manuscript reviewer. Do not edit files or launch other models. "
                              "Use web search when the task asks you to verify citations or to find prior work. "
                              "Distinguish existing baseline gaps from errors introduced by editing. Return a substantive report.\n"
                              f"Recorded spending allowance: ${usd:.2f}. Prioritize the requested checks and explicitly list anything unverified.\n"
                              f"Time allowance: {int(rem)} seconds; deadline Unix {time.time() + rem:.0f}.\nTask: {task}\n"
                              + "\n".join(f"\n--- {name} ---\n{text}" for name, text in snapshot.items()))
                    cfg = {
                        "cmd": ["codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                                "--json", "--sandbox", "read-only", "-c", 'model_provider="openai"',
                                "-c", 'forced_login_method="api"', "-c", "features.shell_tool=false",
                                "-c", "features.multi_agent=false", "-c", "features.apps=false",
                                "-c", "features.plugins=false", "-c", "tools.web_search=true",
                                "--output-last-message", "report.md", "-"],
                        "model": settings.codex_model, "model_reasoning_effort": settings.codex_effort,
                        "prompt": "{prompt}", "completion_signal": "exit", "copy_codex_auth": False,
                        "env": {"CODEX_HOME": "{workspace}/.codex"},
                        "output_files": {"report": "report.md"}, "done_outputs": {"status": "status"},
                        "usage": {"type": "codex_jsonl", "model": settings.codex_model,
                                  "cost_config": settings.codex_cost_config},
                        "sandbox": self._sandbox(settings, rem, "OPENAI_API_KEY"),
                        "WORKSPACE_RECOVERY_ENABLED": True,
                        "WORKSPACE_RECOVERY_MAX_ATTEMPTS": 0,
                        "WORKSPACE_HARD_LIMIT_BYTES": settings.workspace_bytes,
                        "WORKSPACE_HARD_LIMIT_ENTRIES": 10000,
                    }
                    worker = self._turn(cfg, f"cleanup-codex-{review_id}", usd, rem, ctx=helper_ctx)
                    unsettled_helpers.add(review_id)
                    try:
                        out = await worker(workspace=str(review_root), prompt=prompt)
                    finally:
                        if worker.proven_not_started:
                            unsettled_helpers.discard(review_id)
                        elif getattr(getattr(worker, "metering_stream", None), "worker_stopped", False):
                            estimate = None
                            if not worker.metered:
                                # Only a stopped review can be settled by an estimate.
                                # Use the normal cost ledger for resume and usage export.
                                recorded = worker.tracker.counters.usd
                                charge = max(0.0, usd - recorded)
                                worker.tracker.add_usd(charge)
                                await worker.events.emit("model.call", {
                                    "model": settings.codex_model, "cost_usd": charge,
                                    "cost_estimated": True, "usage_unavailable": True,
                                    "via": "cleanup_codex_allowance_estimate",
                                    "cost_config": settings.codex_cost_config, "review_id": review_id,
                                }, call_id=f"cleanup-estimate:{review_id}")
                                estimate = {"review_id": review_id, "charged_usd": charge, "recorded_usd": recorded}
                            unsettled_helpers.discard(review_id)
                            if estimate is not None:
                                await self.events.emit("cleanup.helper_usage_estimated", estimate)
                    report = _read_file(review_root, "report.md")
                    if not report.strip() or out.status != "done":
                        raise RuntimeError("Codex reviewer did not finish; inspect retained CLI logs")
                    # Spend is already recorded; the pool check before the next
                    # review stops further calls, so a paid report is kept.
                    reviews = workspace / "reviews"
                    if reviews.is_symlink():
                        raise ValueError("unsafe reviews directory")
                    reviews.mkdir(exist_ok=True)
                    path = reviews / f"{review_id}.md"
                    write_text_atomic(path, report)
                    return {"report": report, "path": f"reviews/{review_id}.md", "cost_usd": worker.tracker.counters.usd}

            async def linear_read():
                from mathagents.provider_trace import COUNTS, ProviderTrace, active_trace

                async with linear_lock:
                    if any(key.startswith("linear-") for key in unsettled_helpers):
                        raise CleanupAccountingUncertain("linear reader usage is unresolved")
                    linear_tracker.check()
                    _remaining_usd(linear_tracker)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("cleanup deadline reached")
                    tex = _read_file(workspace, "answer.tex")
                    client = await asyncio.to_thread(_linear_client, settings.linear_read_model)
                    review_id = f"linear-{uuid.uuid4().hex}"
                    trace = ProviderTrace(self.workdir / "provider-attempts.jsonl",
                                          run_id=self.ctx.root_workdir.name, agent=self.workdir.name,
                                          call_id=review_id)
                    delivered = dict.fromkeys(("cost", *COUNTS), 0)
                    charged = dict(delivered)
                    unavailable = False
                    loop = asyncio.get_running_loop()

                    async def settle(status, detail=None):
                        nonlocal unavailable
                        if detail is not None:
                            for key in delivered:
                                delivered[key] += detail.get(key, 0) or 0
                            unavailable |= bool(detail.get("usage_unavailable", False))
                        totals = trace.totals()
                        current = {key: max(delivered[key], totals[key]) for key in delivered}
                        delta = {key: max(0, current[key] - charged[key]) for key in delivered}
                        missing_usage = unavailable or (detail is None and totals["usage_unavailable"])
                        if detail is None and not any(delta.values()) and not missing_usage:
                            return
                        from proofstack.provider_accounting import record_provider_usage
                        await record_provider_usage(self.ctx, linear_tracker, {
                            "model": client.model, "cost_usd": current["cost"],
                            "in_tokens": current["input_tokens"], "out_tokens": current["output_tokens"],
                            "reasoning_tokens": current["reasoning_tokens"],
                            "status": status, "via": "cleanup_linear_read",
                            "usage_unavailable": missing_usage,
                        }, call_id=review_id, emitter=self.events)
                        charged.update(current)

                    async def admit():
                        linear_tracker.check()
                        _remaining_usd(linear_tracker)
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("cleanup deadline reached")
                        for key in ("timeout", "max_wallclock_per_call_s"):
                            configured = getattr(client, key, None)
                            setattr(client, key, min(float(configured), remaining)
                                    if configured is not None else remaining)

                    async def received(detail):
                        await settle("completed", detail)
                        try:
                            linear_tracker.check()
                        except BudgetExhausted:
                            client.terminate()

                    def before_batch():
                        asyncio.run_coroutine_threadsafe(admit(), loop).result()

                    def on_result(detail):
                        asyncio.run_coroutine_threadsafe(received(detail), loop).result()

                    async def run_reader():
                        token = active_trace.set(trace)
                        status = "failed"
                        try:
                            result = await asyncio.to_thread(
                                _linear_read, tex, settings.linear_read_model, client=client,
                                before_batch=before_batch, on_result=on_result)
                            status = "completed"
                            return result
                        finally:
                            active_trace.reset(token)
                            await settle("cancelled" if client.terminated else status)
                            if not (unavailable or trace.totals()["usage_unavailable"]):
                                unsettled_helpers.discard(review_id)

                    unsettled_helpers.add(review_id)
                    job = asyncio.create_task(run_reader())
                    interrupted = False
                    # Cancelling to_thread only cancels its waiter. Keep the API
                    # worker and settlement alive, and join them before releasing MCP.
                    while not job.done():
                        try:
                            await asyncio.shield(job)
                        except asyncio.CancelledError:
                            interrupted = True
                            client.terminate()
                        except Exception:
                            break
                    if interrupted:
                        if not job.cancelled():
                            job.exception()
                        raise asyncio.CancelledError
                    result = job.result()
                    if review_id in unsettled_helpers:
                        raise CleanupAccountingUncertain("linear reader usage is unresolved")
                    await self.events.emit("cleanup.linear_read", {
                        "model": result["model"], "passages": result["passages"],
                        "findings": len(result["findings"]), "usd": result["cost_usd"]})
                    reviews = workspace / "reviews"
                    if reviews.is_symlink():
                        raise ValueError("unsafe reviews directory")
                    reviews.mkdir(exist_ok=True)
                    path = f"reviews/linear-{uuid.uuid4().hex}.md"
                    write_text_atomic(workspace / path, result["report"])
                    return {"report": result["report"], "path": path,
                            "findings": len(result["findings"]), "cost_usd": round(result["cost_usd"], 3)}

            turn = None
            try:
                with tempfile.TemporaryDirectory(prefix="cleanup-control-") as private:
                    config_path = Path(private) / "mcp.json"
                    reviews = CleanupReviews(root, workspace, snapshot=review_snapshot, execute=codex_review,
                                             read_text=_read_file, admit=admit_review)
                    finishing_only = progress()["stage"] == "finishing"
                    saved_draft_matches = (state["resumable"] and (workspace / "answer.tex").exists()
                                           and document == _read_file(workspace, "answer.tex"))
                    if finishing_only and reviews.attribution_retrieved():
                        # Finishing forbids new optional reviews. Do not reserve
                        # money again for a required report already paid for and read.
                        lead_allowance += helper_allowance + retry_reserve
                        invocation_tracker.spec = invocation_tracker.spec.model_copy(update={"max_usd": allowance})
                        helper_tracker.spec = helper_tracker.spec.model_copy(update={"max_usd": 0})
                        helper_allowance = 0.0
                    async with reviews, cleanup_tools(compile_document=compile_document, codex_review=codex_review,
                                             read_file=read_file, write_file=write_file, list_files=list_files,
                                             linear_read=linear_read if linear_allowance > 0 else None,
                                             review_start=reviews.start, review_status=reviews.status, review_read=reviews.note_read,
                                             cancel_reviews=reviews.cancel,
                                             progress=progress,
                                             call_timeout_s=remaining) as mcp:
                        _save(config_path, mcp)
                        config_path.chmod(0o600)
                        native_tools = "" if finishing_only else "Task"
                        tools = "mcp__cleanup__compile,mcp__cleanup__review,mcp__cleanup__review_status,mcp__cleanup__read,mcp__cleanup__write,mcp__cleanup__edit,mcp__cleanup__files"
                        if native_tools:
                            tools = native_tools + "," + tools
                        if linear_allowance > 0:
                            tools += ",mcp__cleanup__linear_read"
                        reviewers = {name: {"description": description,
                                           "prompt": "Review only. Read the files named by the lead in full, following next_offset until it is null. Return all substantive findings in your response. Do not edit files or delegate.",
                                           "tools": ["mcp__cleanup__read", "mcp__cleanup__files"], "model": settings.model}
                                     for name, description in (("correctness-reviewer", "Compare current proof to baseline for semantic changes"),
                                                               ("exposition-reviewer", "Review readability for a strong non-specialist mathematician in the area"))}
                        cfg = {
                            # --bare removes Task even when --agents is explicit.
                            "cmd": ["claude", "-p", "--setting-sources", "", "--disable-slash-commands",
                                    "--settings", '{"disableAllHooks":true}',
                                    "--output-format", "stream-json", "--verbose",
                                    "--effort", settings.effort, "--max-budget-usd", str(lead_allowance),
                                    "--strict-mcp-config", "--mcp-config", str(config_path),
                                    "--tools", native_tools, "--allowedTools", tools,
                                    "--disallowedTools", "Bash,Skill,Read,Glob,Grep,Edit,Write,WebFetch,WebSearch,"
                                    "Agent(general-purpose),Agent(Explore),Agent(Plan),Agent(claude),Agent(statusline-setup)",
                                    "--agents", json.dumps(reviewers),
                                    "--resume" if state["resumable"] else "--session-id", state["session_id"]],
                            "model": settings.model, "prompt": "{prompt}", "completion_signal": "exit",
                            "usage": {"type": "claude_json", "auth_mode": "api", "model": settings.model,
                                      "cost_config": settings.cost_config},
                            "sandbox": self._sandbox(settings, remaining, "ANTHROPIC_API_KEY"),
                            # Built-in agent types (general-purpose, Explore, ...) inherit the
                            # lead's write/review tools and can nest; --allowedTools cannot limit
                            # Agent to named types. This leaves only the --agents reviewers.
                            "env": {"CLAUDE_CONFIG_DIR": "{workspace}/.claude", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                                    "DISABLE_AUTOUPDATER": "1", "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
                                    "CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH": "1"},
                            "input_files": {name: {"from_input": f"file_{i}"} for i, name in enumerate([*files, "answer.tex"])},
                            "done_outputs": {"status": "status"},
                            "WORKSPACE_RECOVERY_ENABLED": True,
                            "WORKSPACE_RECOVERY_MAX_ATTEMPTS": 0,
                            "WORKSPACE_HARD_LIMIT_BYTES": settings.workspace_bytes,
                            "WORKSPACE_HARD_LIMIT_ENTRIES": 10000,
                        }
                        turn = self._turn(cfg, f"cleanup-editor-{uuid.uuid4().hex}", lead_allowance, remaining,
                                          ctx=invocation_ctx)
                        # Leave the cap to Claude's own clean stop, which writes a result record.
                        turn.live_usd_limit = lead_allowance * LEAD_MONITOR_MARGIN
                        if state["resumable"]:
                            turn.claude_baseline = (state["claude_session_usd"], state["claude_session_tokens"])
                        transcript = _claude_transcript(workspace, state["session_id"])
                        transcript_offset = transcript.stat().st_size if transcript is not None else 0
                        prompt = (PROMPTS / "cleanup-session.txt").read_text()
                        prompt += (f"\nThis invocation has up to {int(remaining)} seconds (deadline Unix {time.time() + remaining:.0f}) "
                                   f"and ${lead_allowance:.2f} for you and native reviewers, plus ${helper_allowance:.2f} reserved for Codex.\n")
                        prompt += ("Finishing stage is active now. " if finishing_only else
                                   f"Finishing stage starts with {int(min(remaining, finishing_window))} seconds left. ")
                        prompt += (
                                   "Managed read, write, compile and review results include cleanup_control with the current stage and instructions. "
                                   "On entering finishing, stop optional polishing and do not start native reviewer tasks. "
                                   "Only an outstanding mandatory attribution review may start; complete the final checks and signal completion.\n")
                        if finishing_only:
                            if inp.finishing_only and saved_draft_matches:
                                prompt += ("FINISHING-ONLY CONTINUATION: answer.tex is your own saved draft, possibly unfinished, "
                                           "not a newly reviewed manuscript. Initial findings may concern the earlier baseline; "
                                           "recheck them against the saved edits. Finish the saved edits; do not restart the rewrite.\n")
                            else:
                                prompt += ("FINISHING STAGE: answer.tex is the manuscript supplied for this invocation; "
                                           "the harness may have replaced your earlier draft. Re-read it with the current "
                                           "findings and constraints before making the requested repairs.\n")
                            prompt += ("Native reviewer tasks are disabled. Preserve the scope in constraints.md; "
                                       "finishing does not authorize substantive edits during a mechanical repair.\n")
                            if not helper_allowance and settings.codex_budget_fraction:
                                prompt += ("The required attribution report was already completed and retrieved. "
                                           "Reuse it; its former budget share is now available for finishing. "
                                           "No new Codex reviews are permitted in this invocation.\n")
                        if helper_allowance > 0:
                            prompt += ("A failed mandatory attribution review may receive one bounded budget top-up from unused "
                                       "enclosing funds. Use review_status to confirm failure, then request a focused retry; "
                                       "do not assume extra funds are available or bypass a denied retry.\n")
                        if attempt:
                            prompt += "Continue the current revision. Do not restart or repeat the full rewrite.\n"
                        if inp.findings.strip():
                            prompt += (
                                "\nCURRENT FINDINGS\nThe harness refreshed findings.md, constraints.md and "
                                "answer.tex for this invocation; re-read them even if they were read earlier "
                                "in this session. The latest findings are also reproduced below. "
                                "Reuse existing helper reports where still applicable.\n"
                            )
                            if inp.partial:
                                prompt += ("These findings describe an unsolved attempt. Preserve unresolved "
                                           "objections in the partial write-up; they are not instructions to "
                                           "solve missing mathematics. Follow constraints.md for any requested "
                                           "mechanical repairs.\n")
                            else:
                                prompt += (
                                    "CURRENT CRITIC REPAIR REQUEST: This is a targeted repair, not a new rewrite. "
                                    "Address each finding in answer.tex and explain changes or disagreements "
                                    "in feedback.md. Do not mark an unchanged manuscript ready: the harness "
                                    "will not pay to review it again. If you cannot make the requested repairs, "
                                    "preserve the draft and return status=unable with the reason.\n"
                                )
                            prompt += "\n" + inp.findings + "\nEND CURRENT FINDINGS\n"
                        if linear_allowance > 0:
                            prompt += (f"Linear-reader allowance for this invocation: ${linear_allowance:.2f}. "
                                       "If exhausted, do not retry; retain its existing findings and disclose "
                                       "any unfinished checks in feedback.md.\n")
                        else:
                            prompt += "The linear reader is disabled for this invocation. Do not request it; disclose this limitation in feedback.md.\n"
                        if not settings.codex_budget_fraction:
                            prompt += "Codex reviews are disabled for this run. Do not request attribution review; disclose this limitation in feedback.md.\n"
                        if malformed_completion:
                            prompt += ('Your completion.json was invalid; write a JSON object with "status" '
                                       '"ready" or "unable" and a "summary".\n')
                        state["in_flight"] = True
                        _save(state_path, state)
                        try:
                            out = await turn(workspace=str(workspace), prompt=prompt,
                                             **{f"file_{i}": text for i, text in enumerate([*files.values(), document])})
                        except BaseException as exc:
                            invocation_error = exc
            except BaseException as exc:
                invocation_error = exc
            if turn is None:
                raise invocation_error or RuntimeError("cleanup editor was not initialized")

            async def settle():
                # MCP shutdown has now joined every helper, including background
                # requests that were still running when Claude exited.
                if unsettled_helpers:
                    raise CleanupAccountingUncertain("cleanup helper shutdown or usage is unresolved; reconcile before reuse")
                stopped = getattr(getattr(turn, "metering_stream", None), "worker_stopped", False)
                resumable = state["resumable"] or not turn.proven_not_started
                estimated = 0.0
                claude_total = turn.claude_baseline
                if not turn.proven_not_started and stopped and not (turn.metered and turn.final_usage):
                    # A harness stop (deadline, cancel, monitor, memory kill) leaves
                    # no result record, but --max-budget-usd bounded the lead. On
                    # SIGTERM Claude still saves its cumulative total, which the
                    # next --resume starts from; after SIGKILL the old total stands.
                    saved = _saved_claude_total(_claude_transcript(workspace, state["session_id"]),
                                                transcript_offset, settings.cost_config)
                    known = 0.0
                    if saved is not None:
                        claude_total = saved
                        known = max(0.0, saved[0] - turn._claude_baseline_for(saved[0])[0])
                    recorded = turn.tracker.counters.usd
                    estimated = max(0.0, max(lead_allowance, known) - recorded)
                    observation = {
                        "session_id": state["session_id"], "charged_usd": estimated, "recorded_usd": recorded,
                        "saved_usage_usd": known if saved is not None else None,
                        "unreported_allowance_usd": max(0.0, lead_allowance - max(recorded, known)),
                    }
                    turn.tracker.add_usd(estimated)
                    await turn.events.emit("model.call", {
                        "model": settings.model, "cost_usd": estimated,
                        "cost_estimated": True, "usage_unavailable": True,
                        "via": "cleanup_editor_allowance_estimate",
                        "cost_config": settings.cost_config, "session_id": state["session_id"],
                    }, call_id=f"cleanup-estimate:{state['session_id']}:{uuid.uuid4().hex}")
                    await self.events.emit("cleanup.editor_usage_estimated", observation)
                    state.setdefault("editor_usage_estimates", []).append(observation)
                    # A worker killed before its first save has no transcript to resume.
                    resumable = state["resumable"] or any(
                        (workspace / ".claude" / "projects").glob(f"*/{state['session_id']}.jsonl"))
                elif not (turn.proven_not_started or (turn.metered and turn.final_usage and stopped)):
                    if invocation_error is not None:
                        raise invocation_error
                    raise RuntimeError("cleanup usage is unresolved; reconcile before reuse")
                elif turn.claude_total is not None:
                    claude_total = turn.claude_total
                state.update(in_flight=False, resumable=resumable,
                             claude_session_usd=claude_total[0], claude_session_tokens=claude_total[1],
                             recorded_usd=state["recorded_usd"] + invocation_tracker.counters.usd,
                             estimated_usd=state.get("estimated_usd", 0.0) + estimated)
                _save(state_path, state)

            settlement = asyncio.create_task(settle())
            while not settlement.done():
                try:
                    await asyncio.shield(settlement)
                except asyncio.CancelledError:
                    invocation_error = asyncio.CancelledError()
            settlement.result()
            if asyncio.current_task().cancelling():
                raise asyncio.CancelledError
            if invocation_error is not None and (turn.proven_not_started
                                                 or not isinstance(invocation_error, RuntimeError)):
                raise invocation_error
            document = _read_file(workspace, "answer.tex")
            feedback = _read_file(workspace, "feedback.md", optional=True)
            if invocation_error is not None or out.status != "done":
                reason = str(invocation_error) if invocation_error is not None else f"status {out.status}"
                raise CleanupIncomplete(
                    f"cleanup editor failed ({reason}); existing publication fallback remains available",
                    self.Outputs(answer_tex=document, feedback_md=feedback, status="incomplete", summary=reason,
                                 workspace=workspace, session_id=state["session_id"]),
                ) from invocation_error
            completion = _read_file(workspace, "completion.json", optional=True)
            try:
                done = self._Completion.model_validate_json(completion) if completion else None
            except ValueError:
                done = None
            malformed_completion = bool(completion) and done is None
            if done is not None:
                if done.status == "ready" and settings.codex_budget_fraction and not reviews.attribution_retrieved():
                    reason = "Required attribution review was not completed and retrieved; cleanup is incomplete."
                    raise CleanupIncomplete(reason, self.Outputs(
                        answer_tex=document, feedback_md=feedback, status="incomplete", summary=reason,
                        workspace=workspace, session_id=state["session_id"]))
                result = self.Outputs(answer_tex=document, feedback_md=feedback,
                                      status=done.status, summary=done.summary,
                                      workspace=workspace, session_id=state["session_id"])
                await self.events.emit("cleanup.revision", {"session_id": result.session_id, "status": result.status})
                return result
        raise RuntimeError("cleanup editor did not signal completion within its invocation limit")
