"""Offline safety checks for the explicit, rehearsal-only batch recovery tool."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import resume_batch3 as recovery
from mathagents.provider_trace import usage_adjustments
from proofstack.agents.ac.ac_workflow import _problem_hash, _sum_logged_model_cost
from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow, FirstProofBatch3RehearsalWorkflow
from proofstack.latex_contract import normalize_submission_latex


@pytest.fixture(autouse=True)
def offline_cleanup_probe(monkeypatch):
    probe = AsyncMock()
    monkeypatch.setattr(recovery, "check_cleanup_launch", probe)
    return probe


def test_cleanup_gate_runs_before_any_resume_checkpoint_changes(checkpoint, monkeypatch, offline_cleanup_probe):
    plan = recovery.plan_run(checkpoint, {})
    before = recovery.checkpoint_digest(checkpoint)
    monkeypatch.setattr(recovery, "check_compute", AsyncMock())
    offline_cleanup_probe.side_effect = RuntimeError("cleanup host unavailable")
    monkeypatch.setattr(recovery, "apply_plan", lambda *_: pytest.fail("must not mutate checkpoints"))
    with pytest.raises(RuntimeError, match="cleanup host unavailable"):
        asyncio.run(recovery.launch([plan], 1))
    assert recovery.checkpoint_digest(checkpoint) == before
    offline_cleanup_probe.assert_awaited_once_with(
        plan["recipe"]["inputs"], plan["recipe"]["components"], checkpoint.parent)


@pytest.mark.parametrize("role", ["editors", "codex"])
def test_stale_cleanup_registry_blocks_resume_without_checkpoint_changes(checkpoint, monkeypatch, role):
    from proofstack.agents import cleanup_session
    from proofstack.sandbox.memory import GiB, MemoryRegistryError
    # Resume roots need not use the adapter's workflow_runs directory name.
    relocated = checkpoint.parent.parent / "recovery-copy"
    checkpoint.parent.rename(relocated)
    checkpoint = relocated / checkpoint.name
    registry = checkpoint.parent / ".proofcouncil-cleanup-memory" / role
    registry.mkdir(parents=True)
    control = registry / "control.json"
    saved = json.dumps({"max_workers": 10, "worker_bytes": GiB, "reserve_bytes": 4 * GiB})
    control.write_text(saved)
    plan = recovery.plan_run(checkpoint, {})
    before = recovery.checkpoint_digest(checkpoint)
    monkeypatch.setattr(recovery, "check_compute", AsyncMock())
    monkeypatch.setattr(recovery, "check_cleanup_launch", cleanup_session.check_cleanup_launch)
    runtime = AsyncMock()
    monkeypatch.setattr(cleanup_session, "check_cleanup_available", runtime)
    monkeypatch.setattr(recovery, "apply_plan", lambda *_: pytest.fail("must not mutate checkpoints"))
    with pytest.raises(MemoryRegistryError, match=role):
        asyncio.run(recovery.launch([plan], 1))
    runtime.assert_not_awaited()
    assert recovery.checkpoint_digest(checkpoint) == before
    assert control.read_text() == saved


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "workflow_runs/run-p"
    inputs = FirstProofBatch3RehearsalWorkflow.Inputs(
        problem="P", problem_id="p", page_limit=40, compute_max_parallel_workers=6,
    ).model_dump(mode="json")
    recovery.save_json(root / "agents/FirstProofBatch3Workflow-c0/input.json", inputs)
    now = time.time()
    recovery.save_json(root / "batch3-schedule.json", {
        "problem_hash": _problem_hash("P"), "initial_usd": 600, "partial_reserve_usd": 90,
        "research_deadline_unix_s": now + 1000, "run_deadline_unix_s": now + 2000,
    })
    recovery.save_json(root / "ac_workspaces/p/.ac/resume-state.json", {
        "problem_hash": _problem_hash("P"), "problem_id": "p", "last_round_run": 2,
    })
    recovery.save_json(root / "resume.json", {"argv": [
        "scripts/run_workflow.py", "--workflow", "firstproof_batch3",
        "--output", str(root.parent), "--problem-id", "p", "--problem-text", "P",
    ]})
    (root / "events.jsonl").write_text(json.dumps({"kind": "model.call", "payload": {"cost_usd": 12}}) + "\n")
    return root


def approve_export(root):
    text = normalize_submission_latex(r"\documentclass{article}\begin{document}Proof.\end{document}")
    (root / "submissions").mkdir()
    (root / "submissions/p.tex").write_text(text)
    recovery.save_json(root / "batch3-output.json", {
        "problem_id": "p", "submission_approved": True, "compiled": True, "pages": 1,
        "submission_sha256": hashlib.sha256(text.encode()).hexdigest(),
    })


def test_plan_preserves_original_clock_budget_page_and_slots(checkpoint):
    before = recovery.checkpoint_digest(checkpoint)
    plan = recovery.plan_run(checkpoint, {})
    assert plan["schedule_before"] == plan["schedule_after"]
    assert plan["recipe"]["inputs"]["page_limit"] == 40
    assert plan["recipe"]["inputs"]["compute_max_parallel_workers"] == 6
    assert plan["effective_budget_usd"] == 600
    assert plan["cumulative_known_cost_usd"] == 12
    assert plan["paths_restored"] and not plan["launch_authorized"]
    assert recovery.checkpoint_digest(checkpoint) == before
    with pytest.raises(ValueError):
        FirstProofBatch3Workflow.Inputs(problem="P", page_limit=40)


def test_explicit_extension_debits_and_reserve_survive_second_restart(checkpoint):
    approval = {"extend_seconds": 3600, "reason": "Approved interrupted rehearsal",
                "liability_reserve_usd": 20, "compute_max_parallel_workers": 4,
                "debits": {"legacy-claude": {"cost_usd": 7.0698525, "reason": "Saved usage record"}}}
    plan = recovery.plan_run(checkpoint, approval)
    assert plan["schedule_after"]["initial_usd"] == 600
    assert plan["effective_budget_usd"] == 580
    assert plan["recipe"]["inputs"]["compute_max_parallel_workers"] == 4
    assert plan["schedule_after"]["run_deadline_unix_s"] - plan["schedule_before"]["run_deadline_unix_s"] == 3600
    recipe = recovery.apply_plan(plan)
    journal = recovery.read_json(recipe.parent / "plan.json")
    assert journal["schedule_before"] == plan["schedule_before"]
    assert journal["schedule_after"] == plan["schedule_after"]
    for _ in range(2):
        assert _sum_logged_model_cost(checkpoint / "events.jsonl") == pytest.approx(19.0698525)
        assert len(usage_adjustments(checkpoint, [])) == 1
    # Re-planning cannot silently free the reserve or post the same debit twice.
    again = recovery.plan_run(checkpoint, {"debits": approval["debits"]})
    assert again["effective_budget_usd"] == 580
    assert again["cumulative_known_cost_usd"] == pytest.approx(19.0698525)
    assert again["schedule_before"] == again["schedule_after"] == plan["schedule_after"]
    release = recovery.plan_run(checkpoint, {"reason": approval["reason"], "liability_reserve_usd": 10})
    new_recipe = recovery.apply_plan(release)
    assert new_recipe != recipe
    assert recovery.read_json(recipe.parent / "plan.json") == journal
    assert recovery.plan_run(checkpoint, {})["effective_budget_usd"] == 590
    approval["debits"]["legacy-claude"]["cost_usd"] = 1
    with pytest.raises(ValueError, match="Cannot change"):
        recovery.plan_run(checkpoint, approval)


def authorization(checkpoint):
    approval = {"billing_reconciled": True, "reason": "Invoice checked", "code_sha256": "code"}
    plan = recovery.plan_run(checkpoint, approval)
    plan["replay"] = {"next_calls": [{"agent": "Author", "cache_key": "abc"}], "cache_hits": []}
    approval.update({key: plan[key] for key in ("schedule_sha256", "checkpoint_sha256", "replay")})
    return plan, approval


@pytest.mark.parametrize("change,match", [
    ("billing", "billing reconciliation"), ("path", "Restore"), ("code", "Code or schedule"),
    ("clock", "clock expired"), ("cost", "No research budget"),
    ("replay", "exact offline replay"), ("checkpoint", "Checkpoint changed"),
])
def test_execution_requires_reviewed_plan(checkpoint, change, match):
    plan, approval = authorization(checkpoint)
    recovery.check_authorization(plan, approval, "code")
    if change == "billing":
        plan["billing_reconciled"] = False
    elif change == "path":
        plan["paths_restored"] = False
    elif change == "code":
        approval["code_sha256"] = "old"
    elif change == "clock":
        plan["schedule_after"]["run_deadline_unix_s"] = 0
    elif change == "cost":
        plan["cumulative_known_cost_usd"] = 600
    elif change == "replay":
        approval["replay"] = {}
    else:
        approval["checkpoint_sha256"] = "old"
    with pytest.raises(ValueError, match=match):
        recovery.check_authorization(plan, approval, "code")


def test_checkpoint_mutation_after_approval_is_refused(checkpoint):
    plan, _ = authorization(checkpoint)
    (checkpoint / "events.jsonl").write_text("\n")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        recovery.apply_plan(plan)
    with pytest.raises(ValueError, match="Checkpoint changed"):
        asyncio.run(recovery.replay_run(plan))


def test_approved_replay_and_launch_make_zero_calls_and_no_checkpoint_edits(checkpoint, monkeypatch):
    approve_export(checkpoint)
    before = recovery.checkpoint_digest(checkpoint)
    plan = recovery.plan_run(checkpoint, {})
    probe = AsyncMock(side_effect=AssertionError("approved run must not start Compute"))
    spawn = AsyncMock(side_effect=AssertionError("approved run must not start a workflow"))
    monkeypatch.setattr(recovery, "check_compute", probe)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    replay = asyncio.run(recovery.replay_run(plan))
    assert replay == {"next_calls": [], "cache_hits": [], "submission_approved": True, "partial_ready": False}
    recovery.check_authorization(plan, {}, "any-code")
    result = asyncio.run(recovery.launch([plan], 10))
    assert result[0]["status"] == "preserved_approved"
    assert result[0]["paid_calls"] == 0
    assert recovery.checkpoint_digest(checkpoint) == before


def test_replay_reuses_completed_leaf_and_blocks_failed_leaf(checkpoint, monkeypatch):
    from proofstack.agent import Agent
    from pydantic import BaseModel

    class Seat(Agent):
        class Inputs(BaseModel):
            value: str
        class Outputs(BaseModel):
            ok: bool
        def cache_output_is_reusable(self, out):
            return out.ok
        async def run(self, inp):
            pytest.fail("Replay must never run a paid leaf")

    plan = recovery.plan_run(checkpoint, {})
    ctx = recovery.RunContext.create(run_id=checkpoint.name, root_workdir=checkpoint, flat=True)
    seat = Seat(ctx)
    for value, ok in (("done", True), ("failed", False)):
        key = seat._cache_key(Seat.Inputs(value=value))
        ctx.resume_cache.put(key, {"ok": ok})
    plan["checkpoint_sha256"] = recovery.checkpoint_digest(checkpoint)

    async def run(self, inp):
        seat = Seat(self.ctx)
        await seat(value="done")
        await seat(value="failed")
        pytest.fail("Replay must stop before uncached call")

    monkeypatch.setattr(FirstProofBatch3RehearsalWorkflow, "run", run)
    result = asyncio.run(recovery.replay_run(plan))
    assert len(result["cache_hits"]) == len(result["next_calls"]) == 1
    assert result["cache_hits"][0]["cache_key"] != result["next_calls"][0]["cache_key"]
    assert recovery.checkpoint_digest(checkpoint) == plan["checkpoint_sha256"]


def test_all_launch_gates_pass_before_mutation_or_spend(checkpoint, monkeypatch):
    plan = recovery.plan_run(checkpoint, {})
    probe = AsyncMock(side_effect=RuntimeError("native host fails"))
    monkeypatch.setattr(recovery, "check_compute", probe)
    monkeypatch.setattr(recovery, "apply_plan", lambda *_: pytest.fail("must not edit checkpoint"))
    with pytest.raises(RuntimeError, match="native host fails"):
        asyncio.run(recovery.launch([plan], 2))


def test_launch_enforces_shared_batch_semaphore(checkpoint, monkeypatch):
    plans = [dict(recovery.plan_run(checkpoint, {}), run_id=f"p-{i}") for i in range(5)]
    monkeypatch.setattr(recovery, "check_compute", AsyncMock())
    recipe = checkpoint / "recipe.yaml"
    monkeypatch.setattr(recovery, "apply_plan", lambda _: recipe)
    active, maximum = 0, 0
    class Process:
        async def wait(self):
            nonlocal active
            await asyncio.sleep(0.01)
            active -= 1
            return 0
    async def spawn(*args, **kwargs):
        nonlocal active, maximum
        assert kwargs["start_new_session"]
        active += 1
        maximum = max(maximum, active)
        return Process()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    results = asyncio.run(recovery.launch(plans, 2))
    assert len(results) == 5 and maximum == 2 and active == 0


def test_batch_lock_prevents_concurrent_recovery(tmp_path):
    with recovery.batch_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with recovery.batch_lock(tmp_path):
                pytest.fail("second launcher obtained lock")


def test_cancelled_launcher_terminates_and_waits_for_child(checkpoint, monkeypatch):
    plan = recovery.plan_run(checkpoint, {})
    monkeypatch.setattr(recovery, "check_compute", AsyncMock())
    monkeypatch.setattr(recovery, "apply_plan", lambda _: checkpoint / "recipe.yaml")
    kills = []
    async def exercise():
        started, exited = asyncio.Event(), asyncio.Event()
        class Process:
            pid = 12345
            returncode = None
            async def wait(self):
                await exited.wait()
                self.returncode = -15
                return self.returncode
        async def spawn(*args, **kwargs):
            started.set()
            return Process()
        def kill(pid, sig):
            kills.append((pid, sig))
            exited.set()
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr(recovery.os, "killpg", kill)
        task = asyncio.create_task(recovery.launch([plan], 1))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set()
    asyncio.run(exercise())
    assert kills == [(12345, recovery.signal.SIGTERM)]
