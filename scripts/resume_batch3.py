#!/usr/bin/env python3
"""Plan/replay a stopped Batch 3 batch; paid continuation requires --execute.

Use an extracted working copy, not the archive. Dry runs never edit it.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_batch3_recovery import inspect_run
from proofstack import BudgetSpec, RunContext
from proofstack.agent import Agent
from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow, FirstProofBatch3RehearsalWorkflow, _Schedule
from proofstack.atomic import write_text_atomic
from proofstack.context import ResumeCache
from proofstack.healthcheck import check_compute
from proofstack.agents.cleanup_session import check_cleanup_launch
from proofstack.registry import load_preset
from run_workflow import _argparser, _parse_kv_list, _parse_component_overrides, _deep_merge, _run_with_stop_signals


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def code_digest():
    paths = sorted(p for directory in ("src", "configs", "scripts") for p in (ROOT / directory).rglob("*")
                   if p.is_file() and p.suffix in {".py", ".yaml", ".txt", ".md"})
    return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_digest(root):
    """Bind approval to artifacts as well as clocks, without loading large files."""
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint contains symlinks; restore a self-contained copy")
        if path.is_file():
            with path.open("rb") as stream:
                hashes[str(path.relative_to(root))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest(hashes)


def save_json(path, value):
    write_text_atomic(path, json.dumps(value, indent=2, default=str) + "\n")


def finite(value, name):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def replace_paths(value, old: Path, new: Path):
    if isinstance(value, dict):
        return {k: replace_paths(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_paths(v, old, new) for v in value]
    if isinstance(value, (str, Path)) and (str(value) == str(old) or str(value).startswith(str(old) + "/")):
        return str(new) + str(value)[len(str(old)):]
    return value


def plan_run(root: Path, approval: dict) -> dict:
    checkpoint_sha256 = checkpoint_digest(root)
    saved = sorted((root / "agents").glob("FirstProofBatch3*Workflow-*/input.json"), key=lambda p: p.stat().st_mtime_ns)
    if not saved:
        raise ValueError(f"{root.name}: missing saved inputs")
    inputs = read_json(saved[-1])
    inventory = inspect_run(root, page_limit=inputs["page_limit"])
    args = _argparser().parse_args(read_json(root / "resume.json")["argv"][1:])
    original_root = args.output / root.name
    if not original_root.is_absolute():
        raise ValueError("Saved output path must be absolute; restore the original container layout")
    preset = load_preset(args.workflow)
    if not issubclass(preset.workflow_cls, FirstProofBatch3Workflow):
        raise ValueError("Not a Batch 3 preset")
    schedule = read_json(root / "batch3-schedule.json")
    extension = finite(approval.get("extend_seconds", 0), "extend_seconds")
    new_schedule = {**schedule, "research_deadline_unix_s": schedule["research_deadline_unix_s"] + extension,
                    "run_deadline_unix_s": schedule["run_deadline_unix_s"] + extension}
    _Schedule.model_validate(new_schedule)
    debits = read_json(root / "recovery-debits.json") if (root / "recovery-debits.json").exists() else {}
    for key, debit in approval.get("debits", {}).items():
        if not debit.get("reason") or not key:
            raise ValueError("Every debit needs a stable ID and reason")
        debit = {"cost_usd": finite(debit["cost_usd"], "cost_usd"), "reason": debit["reason"]}
        if key in debits and debits[key] != debit:
            raise ValueError(f"Cannot change previously posted debit {key}")
        debits[key] = debit
    liability_path = root / "recovery-liability.json"
    old_reserve = read_json(liability_path)["reserve_usd"] if liability_path.exists() else 0
    reserve = finite(approval.get("liability_reserve_usd", old_reserve), "liability_reserve_usd")
    old_debits = read_json(root / "recovery-debits.json") if (root / "recovery-debits.json").exists() else {}
    added_cost = sum(d["cost_usd"] for k, d in debits.items() if k not in old_debits)
    budget = schedule["initial_usd"] - reserve
    if budget <= 0:
        raise ValueError("Liability reserve consumes the entire budget")
    inputs.update(resume_run=True, research_deadline_unix_s=new_schedule["research_deadline_unix_s"],
                  run_deadline_unix_s=new_schedule["run_deadline_unix_s"])
    # Absolute deadlines remain authoritative; relative launch caps must not
    # truncate an explicitly approved extension.
    inputs["max_wallclock_s"] = max(1, new_schedule["run_deadline_unix_s"] - time.time())
    inputs["research_seconds"] = max(0, new_schedule["research_deadline_unix_s"] - time.time())
    if "compute_max_parallel_workers" in approval:
        inputs["compute_max_parallel_workers"] = approval["compute_max_parallel_workers"]
    inputs = FirstProofBatch3RehearsalWorkflow.Inputs(**inputs).model_dump(mode="json")
    recipe = {
        "workflow": "proofstack.agents.firstproof_batch3.FirstProofBatch3RehearsalWorkflow",
        "inputs": inputs,
        "components": _deep_merge(preset.component_configs, _parse_component_overrides(args.component)),
        "model_overrides": {**preset.model_overrides, **_parse_kv_list(args.model, label="--model")},
        "budget": {"max_usd": budget, "max_wallclock_s": inputs["max_wallclock_s"]},
    }
    return {**inventory, "root": str(root), "original_root": str(original_root),
            "paths_restored": root == original_root, "schedule_before": schedule, "schedule_after": new_schedule,
            "schedule_sha256": digest(schedule), "checkpoint_sha256": checkpoint_sha256,
            "recipe": recipe, "debits": debits,
            "effective_budget_usd": budget, "cumulative_known_cost_usd": inventory["cumulative_known_cost_usd"] + added_cost,
            "billing_reconciled": approval.get("billing_reconciled") is True,
            "reason": approval.get("reason", ""), "liability_reserve_usd": reserve}


class ReplayBoundary(BaseException):
    """Must not be swallowed by workflow fallback/recovery handlers."""


async def replay_run(plan):
    """Drive a disposable checkpoint copy up to its first uncached paid wave."""
    source = Path(plan["root"])
    original = Path(plan["original_root"])
    calls, hits = [], []
    with tempfile.TemporaryDirectory(prefix="batch3-replay-") as directory:
        copy = Path(directory) / source.name
        # Refuse symlinks rather than accidentally write through into the archive.
        if any(p.is_symlink() for p in source.rglob("*")):
            raise ValueError("Replay source contains symlinks; use a self-contained extracted checkpoint copy")
        await asyncio.to_thread(shutil.copytree, source, copy)
        if checkpoint_digest(copy) != plan["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed while preparing replay")
        for p in copy.rglob("*.json"):
            if "resume_cache" not in p.parts:
                save_json(p, replace_paths(read_json(p), original, copy))
        save_json(copy / "batch3-schedule.json", plan["schedule_after"])
        save_json(copy / "recovery-debits.json", plan["debits"])
        recipe = replace_paths(plan["recipe"], original, copy)
        def blocked(*args, **kwargs):
            raise ReplayBoundary("Network/model calls are forbidden during replay")
        ctx = RunContext.create(run_id=source.name, root_workdir=copy, flat=True,
                                run_budget=BudgetSpec(**recipe["budget"]), api_client_factory=blocked,
                                component_configs=recipe["components"], model_overrides=recipe["model_overrides"])
        original_call = Agent.__call__
        original_key = Agent._cache_key
        original_get = ResumeCache.get

        def key(agent, inp):
            return original_key(agent, type(inp).model_validate(replace_paths(inp.model_dump(mode="json"), copy, original)))

        def cache_get(cache, cache_key):
            return replace_paths(original_get(cache, cache_key), original, copy)

        async def guarded(agent, **kwargs):
            inp = agent.Inputs(**kwargs)
            cache_key = key(agent, inp)
            cached = agent.ctx.resume_cache.get(cache_key) if agent.cache_enabled else None
            reusable = cached is not None and agent.cache_output_is_reusable(agent._coerce_output(cached))
            if agent.execution_mode == "agent":
                record = {"agent": agent.name, "cache_key": cache_key}
                if not reusable:
                    calls.append(record)
                    raise ReplayBoundary(agent.name)
                def check_paths(value):
                    if isinstance(value, Path) and (not value.resolve().is_relative_to(copy) or not value.exists()):
                        raise ValueError(f"Cached artifact missing or outside restored run: {value}")
                    if isinstance(value, dict):
                        for item in value.values():
                            check_paths(item)
                    if isinstance(value, list):
                        for item in value:
                            check_paths(item)
                check_paths(agent._coerce_output(cached).model_dump())
                hits.append(record)
            return await original_call(agent, **kwargs)

        loop = asyncio.get_running_loop()
        previous_factory = loop.get_task_factory()
        spawned = set()

        def track_task(loop, coro, **kwargs):
            task = (previous_factory(loop, coro, **kwargs) if previous_factory else
                    asyncio.Task(coro, loop=loop, **kwargs))
            spawned.add(task)
            return task

        async def exercise():
            try:
                out = await FirstProofBatch3RehearsalWorkflow(ctx)(**recipe["inputs"])
                return {"submission_approved": out.submission_approved, "partial_ready": out.partial_ready}
            except ReplayBoundary:
                return {}
            finally:
                # Keep completed siblings too: simultaneous blocked leaves can
                # finish before the first boundary propagates out of a DAG.
                seen = set()
                failures = []
                while pending := spawned - seen:
                    seen.update(pending)
                    for task in pending:
                        if not task.done():
                            task.cancel()
                    results = await asyncio.gather(*pending, return_exceptions=True)
                    for error in results:
                        if isinstance(error, BaseException) and not isinstance(error, (ReplayBoundary, asyncio.CancelledError)):
                            failures.append(error)
                if failures:
                    raise RuntimeError("Unexpected failure in offline replay") from failures[0]

        with patch.object(Agent, "__call__", guarded), patch.object(Agent, "_cache_key", key), \
             patch.object(ResumeCache, "get", cache_get), patch.object(socket.socket, "connect", blocked), \
             patch.object(socket.socket, "connect_ex", blocked):
            loop.set_task_factory(track_task)
            try:
                result = await exercise()
            finally:
                loop.set_task_factory(previous_factory)
        return {"next_calls": sorted(calls, key=lambda x: (x["agent"], x["cache_key"])),
                "cache_hits": sorted(hits, key=lambda x: (x["agent"], x["cache_key"])), **result}


def check_authorization(plan, approval, code):
    if plan["proposed_boundary"] == "preserve_approved_no_calls":
        return
    if not plan["paths_restored"]:
        raise ValueError(f"Restore {plan['run_id']} at {plan['original_root']} before execution")
    if not plan["billing_reconciled"] or not plan["reason"].strip():
        raise ValueError("Explicit billing reconciliation and a recovery reason are required")
    if approval.get("code_sha256") != code or approval.get("schedule_sha256") != plan["schedule_sha256"]:
        raise ValueError("Code or schedule changed since approval; regenerate and review the dry-run plan")
    if approval.get("checkpoint_sha256") != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since approval; regenerate and review the dry-run plan")
    if plan["schedule_after"]["run_deadline_unix_s"] <= time.time():
        raise ValueError("Run clock expired; an explicitly approved rehearsal extension is required")
    if plan["cumulative_known_cost_usd"] >= plan["effective_budget_usd"]:
        raise ValueError("No research budget remains after reconciliation/reservation")
    if approval.get("replay") != plan.get("replay") or not plan.get("replay"):
        raise ValueError("Approve the exact offline replay next_calls/cache_hits before execution")


def apply_plan(plan):
    root = Path(plan["root"])
    if checkpoint_digest(root) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during recovery validation")
    # Journal the old/new values before publication, so a crash never leaves
    # an unexplained extension. Stable IDs make retries of debits idempotent.
    recovery = root / "recovery" / digest({k: plan[k] for k in (
        "schedule_after", "debits", "reason", "liability_reserve_usd", "checkpoint_sha256",
    )})
    recovery.mkdir(parents=True, exist_ok=True)
    save_json(recovery / "plan.json", plan)
    save_json(root / "recovery-debits.json", plan["debits"])
    save_json(root / "recovery-liability.json", {"reserve_usd": plan["liability_reserve_usd"], "reason": plan["reason"]})
    save_json(root / "batch3-schedule.json", plan["schedule_after"])
    save_json(recovery / "workflow.yaml", plan["recipe"])  # JSON is valid YAML.
    return recovery / "workflow.yaml"


@contextmanager
def batch_lock(root):
    with (root / ".batch3-resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


async def launch(plans, parallel):
    semaphore = asyncio.Semaphore(parallel)
    # All infrastructure gates pass before changing any checkpoint or spending.
    for plan in plans:
        if plan["proposed_boundary"] != "preserve_approved_no_calls":
            await check_compute(plan["recipe"]["inputs"], Path(plan["root"]).parent.parent)
            await check_cleanup_launch(plan["recipe"]["inputs"], plan["recipe"]["components"],
                                       Path(plan["root"]).parent)
    async def one(plan):
        if plan["proposed_boundary"] == "preserve_approved_no_calls":
            return {"run_id": plan["run_id"], "status": "preserved_approved", "paid_calls": 0}
        async with semaphore:
            recipe = apply_plan(plan)
            command = [sys.executable, str(ROOT / "scripts/run_workflow.py"), "--workflow", str(recipe),
                       "--restart-from", plan["root"], "--problem-id", plan["problem_id"]]
            log = recipe.parent / "resume.log"
            with log.open("ab") as stream:
                proc = await asyncio.create_subprocess_exec(*command, cwd=ROOT, stdout=stream, stderr=stream,
                                                            start_new_session=True)
                try:
                    code = await proc.wait()
                except asyncio.CancelledError:
                    if proc.returncode is None:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            await asyncio.wait_for(proc.wait(), 60)
                        except asyncio.TimeoutError:
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            await proc.wait()
                    raise
            return {"run_id": plan["run_id"], "returncode": code, "log": str(log)}
    tasks = [asyncio.create_task(one(plan)) for plan in plans]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow_runs", type=Path)
    parser.add_argument("--manifest", type=Path, help="Per-run explicit approvals, extensions, liability reserves and debits")
    parser.add_argument("--replay", action="store_true", help="Offline copied-checkpoint replay; no model or CLI worker calls")
    parser.add_argument("--execute", action="store_true", help="Permit paid continuation after validating manifest and replay")
    parser.add_argument("--parallel", type=int, default=10)
    parser.add_argument("--run", action="append", help="Limit planning/replay/execution to these run IDs")
    parser.add_argument("--report", type=Path, help="Write full JSON plan/report here, never inside a checkpoint")
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    approvals = read_json(args.manifest).get("runs", {}) if args.manifest else {}
    root = args.workflow_runs.resolve()
    if args.report and args.report.resolve().is_relative_to(root):
        parser.error("--report must be outside the checkpoint tree")
    code = code_digest()

    def prepare():
        plans = []
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            if args.run and directory.name not in args.run:
                continue
            plan = plan_run(directory, approvals.get(directory.name, {}))
            plan["code_sha256"] = code
            if args.replay or args.execute:
                plan["replay"] = asyncio.run(replay_run(plan))
            plans.append(plan)
        if not plans:
            raise ValueError("No Batch 3 runs found")
        if args.run and set(args.run) != {plan["run_id"] for plan in plans}:
            raise ValueError("One or more requested run IDs were not found")
        return plans

    try:
        if not args.execute:
            plans = prepare()
            report = {"read_only": True, "code_sha256": code, "runs": plans}
            if args.report:
                save_json(args.report, report)
            print(json.dumps({**report, "runs": [{k: v for k, v in plan.items() if k != "recipe"} for plan in plans]}, indent=2))
            return 0
        if not args.manifest:
            raise ValueError("--execute requires a reviewed manifest")
        with batch_lock(root):
            plans = prepare()
            for plan in plans:
                check_authorization(plan, approvals.get(plan["run_id"], {}), code)
                pid_path = Path(plan["root"]) / "run.pid"
                if pid_path.exists():
                    pid = int(pid_path.read_text())
                    if pid <= 0:
                        raise ValueError("Invalid recorded PID; verify the batch is stopped")
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise ValueError(f"{plan['run_id']}: recorded process still exists; verify the batch is stopped")
            results = asyncio.run(_run_with_stop_signals(lambda: launch(plans, args.parallel)))
            if isinstance(results, int):
                return results
            if args.report:
                save_json(args.report, {"results": results})
            print(json.dumps({"results": results}, indent=2))
            return int(any(row.get("returncode", 0) != 0 for row in results))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Batch resume refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
