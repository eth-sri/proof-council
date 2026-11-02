"""Offline regressions for final-review recovery and terminal estimates."""
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mathagents.provider_trace import PRE_INFERENCE_REJECTIONS, ProviderTrace, latest_attempts
from proofstack.agents.final_stateless_review import FinalReviewBlocked, FinalReviewState, FinalStatelessCritic
from proofstack.budget import BudgetSpec
from proofstack.context import RunContext
from proofstack.provider_accounting import provider_attempts_settled
from test_firstproof_batch3 import ORIGINAL, TEX, pipeline
from test_review_six import pricing_client


@pytest.mark.parametrize("code", sorted(PRE_INFERENCE_REJECTIONS))
@pytest.mark.parametrize("evidence", ["none", "in_progress", "output", "reported_usage", "previous_charge"])
def test_pre_inference_rejection_requires_no_execution_evidence(
        tmp_path, monkeypatch, pricing_client, code, evidence):
    clock = [100.0]
    monkeypatch.setattr("mathagents.provider_trace.time.time", lambda: clock[0])
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl")
    trace.request(pricing_client, "call", 0, {"input": "x" * 3300000, "max_output_tokens": 128000})
    trace.response(pricing_client, "call", 0, {
        "id": "saved", "status": "in_progress" if evidence == "in_progress" else "queued"})
    if evidence == "previous_charge":
        trace.update(("call", 0), cost=2.0)
    clock[0] += 60
    response = {"id": "saved", "status": "failed", "error": {"code": code}, "output": []}
    if evidence == "output":
        response["output"] = [{"type": "message", "content": [{"type": "output_text", "text": "Paid work"}]}]
    if evidence == "reported_usage":
        response["usage"] = {"input_tokens": 10, "output_tokens": 20}
    trace.response(pricing_client, "call", 0, response)
    for _ in range(2):
        clock[0] += 3600
        ProviderTrace.restore(trace.path, trace.id).estimate_terminal_usage()
    row, = latest_attempts(trace.path)
    assert provider_attempts_settled([row])
    if evidence == "none":
        assert row["cost"] == 0 and row["cost_estimated"] and row["usage_unavailable"]
    elif evidence == "reported_usage":
        assert row["cost"] == pytest.approx(.0011) and not row["usage_unavailable"]
    elif evidence == "previous_charge":
        assert row["cost"] == 2, "Previously recorded charges must not be refunded"
    else:
        assert not row["pre_inference_rejection"] and row["cost"] > 22


@pytest.mark.parametrize("code", ["server_error", "context_length_exceeded"])
def test_backward_clock_step_does_not_leave_terminal_usage_unresolved(tmp_path, monkeypatch, pricing_client, code):
    clock = [100.0]
    monkeypatch.setattr("mathagents.provider_trace.time.time", lambda: clock[0])
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl")
    trace.request(pricing_client, "call", 0, {"input": "x" * 300000, "max_output_tokens": 128000})
    trace.response(pricing_client, "call", 0, {"id": "saved", "status": "queued"})
    clock[0] = 90.0
    trace.response(pricing_client, "call", 0, {"id": "saved", "status": "failed", "error": {"code": code}})
    for _ in range(2):
        clock[0] += 3600
        ProviderTrace.restore(trace.path, trace.id).estimate_terminal_usage()
    row, = latest_attempts(trace.path)
    assert provider_attempts_settled([row])
    assert row["estimate_elapsed_s"] == 0 and row["terminal_observed_at"] == 90
    assert row["cost"] == (1 if code == "server_error" else 0)


def saved_review(ctx, client, inp, *, attempts=1, separate_invocations=False):
    critic = FinalStatelessCritic(ctx)
    path = ctx.root_workdir / "agents/saved/provider-attempts.jsonl"
    trace = None
    for index in range(attempts):
        if trace is None or separate_invocations:
            trace = ProviderTrace(path, call_id=f"paid-{index}")
        trace.request(client, str(index), 0, {"input": "Synthetic proof", "max_output_tokens": 128000})
        trace.response(client, str(index), 0, {"id": f"saved-{index}", "status": "in_progress"})
    checkpoint = critic._recovery_checkpoint(inp)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({"status": "recovery_started", "kind": "ordinary",
        "attempt_workdir": "agents/saved", "review_input": inp.model_dump(mode="json")}))
    return critic, path


def completed(response_id, accepted=True):
    return {"id": response_id, "status": "completed", "usage": {"input_tokens": 10, "output_tokens": 20},
        "output": [{"type": "message", "content": [{"type": "output_text",
            "text": f"Saved verdict\n<answer_ready>{str(accepted).lower()}</answer_ready>"}]}]}


@pytest.mark.parametrize("run_expired", [False, True])
@pytest.mark.parametrize("accepted", [False, True])
def test_workflow_retrieves_completed_review_after_saved_window(
        pipeline, monkeypatch, pricing_client, run_expired, accepted):
    _, _, _, _, ctx, agent = pipeline
    ctx.api_client_factory = lambda _: pricing_client
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p", final_stateless_review=True, final_review_export_seconds=5)
    key, review_input = agent._final_review_identity(inp, TEX)
    state = FinalReviewState(key=key, review_input=review_input, max_usd=10, initial_spend=0,
                            deadline_unix_s=time.time() - 3600)
    path = agent._final_review_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json())
    _, trace_path = saved_review(ctx, pricing_client, review_input)
    calls = []
    def retrieve(response_id, *, timeout, on_response, cancel_pending):
        assert not cancel_pending and 0 < timeout <= 15
        calls.append(response_id)
        on_response(completed(response_id, accepted))
    monkeypatch.setattr(pricing_client, "reconcile_background_response", retrieve)
    monkeypatch.setattr(agent, "_phase", AsyncMock(side_effect=AssertionError("No new inference")))
    # The tracker has ample time; only the explicit export cutoff has passed.
    deadline = time.monotonic() + (4 if run_expired else 100)
    if run_expired:
        with pytest.raises(FinalReviewBlocked, match="unresolved"):
            asyncio.run(agent._review_final(inp, TEX, deadline=deadline))
        assert calls == [] and not provider_attempts_settled(latest_attempts(trace_path))
        assert ctx.budgets.root().counters.usd == 0
    else:
        for _ in range(2):
            result = asyncio.run(agent._review_final(inp, TEX, deadline=deadline))
            assert result.answer_ready is accepted
        assert calls == ["saved-0"] and provider_attempts_settled(latest_attempts(trace_path))
        assert ctx.budgets.root().counters.usd == pytest.approx(.0011)
        after = FinalReviewState.model_validate_json(path.read_text())
        assert after.status == "completed" and after.deadline_unix_s == state.deadline_unix_s


@pytest.mark.parametrize("separate_invocations", [False, True])
@pytest.mark.parametrize("first_errors", [False, True])
def test_recovery_polls_later_rejection_before_waiting_on_abandoned_attempt(
        tmp_path, monkeypatch, pricing_client, separate_invocations, first_errors):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, run_budget=BudgetSpec(max_wallclock_s=60),
        api_client_factory=lambda _: pricing_client,
        component_configs={"ACCritic": {"research_notes_transport": "inline"}})
    inp = FinalStatelessCritic.Inputs(problem="Synthetic", answer_tex="Synthetic proof", mode="fresh")
    critic, path = saved_review(ctx, pricing_client, inp, attempts=2, separate_invocations=separate_invocations)
    clock = [time.monotonic()]
    monkeypatch.setattr("proofstack.agents.ac.critic.time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time))
    sleep = asyncio.sleep
    async def advance(seconds):
        clock[0] += seconds
        await sleep(0)
    monkeypatch.setattr("proofstack.agents.ac.critic.asyncio.sleep", advance)
    calls, resolved = [], [False]
    def retrieve(response_id, *, timeout, on_response, cancel_pending):
        assert not cancel_pending
        calls.append(response_id)
        if response_id == "saved-0" and not resolved[0]:
            if first_errors:
                raise TimeoutError("Synthetic abandoned response")
            on_response({"id": response_id, "status": "in_progress"})
        else:
            on_response(completed(response_id, accepted=response_id == "saved-0"))
    monkeypatch.setattr(pricing_client, "reconcile_background_response", retrieve)
    with pytest.raises(FinalReviewBlocked, match="unresolved"):
        asyncio.run(critic.recover_completed(inp, deadline=clock[0] + 12, cutoff=clock[0] + 30))
    assert calls[:2] == ["saved-0", "saved-1"]
    assert calls.count("saved-1") == 1 and calls.count("saved-0") == 6
    assert [r["status"] for r in latest_attempts(path)] == ["in_progress", "completed"]
    assert ctx.budgets.root().counters.usd == pytest.approx(.0011)
    assert any("false" in json.loads(p.read_text())["report"] for p in path.parent.glob("provider-completed-*.json"))
    resolved[0] = True
    # Settling the first attempt must reuse, not erase, the saved rejection.
    result = asyncio.run(critic.recover_completed(inp, deadline=clock[0] - 1, cutoff=clock[0] + 30))
    assert not result.answer_ready
    assert ctx.budgets.root().counters.usd == pytest.approx(.0022)


def test_expired_tracker_forbids_even_one_retrieve(tmp_path, monkeypatch, pricing_client):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, api_client_factory=lambda _: pricing_client)
    inp = FinalStatelessCritic.Inputs(problem="Synthetic", answer_tex="Synthetic proof", mode="fresh")
    critic, _ = saved_review(ctx, pricing_client, inp)
    monkeypatch.setattr(critic.tracker, "remaining_wallclock_s", lambda: 0)
    monkeypatch.setattr(pricing_client, "reconcile_background_response", lambda *a, **kw: pytest.fail("Run expired"))
    with pytest.raises(FinalReviewBlocked, match="unresolved"):
        asyncio.run(critic.recover_completed(inp, deadline=time.monotonic() - 1, cutoff=time.monotonic() + 100))


def test_slow_first_retrieval_leaves_time_for_later_attempt(tmp_path, monkeypatch, pricing_client):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, run_budget=BudgetSpec(max_wallclock_s=60),
        api_client_factory=lambda _: pricing_client,
        component_configs={"ACCritic": {"research_notes_transport": "inline"}})
    inp = FinalStatelessCritic.Inputs(problem="Synthetic", answer_tex="Synthetic proof", mode="fresh")
    critic, path = saved_review(ctx, pricing_client, inp, attempts=2)
    clock = [100.0]
    monkeypatch.setattr("proofstack.agents.ac.critic.time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time))
    async def advance(seconds):
        clock[0] += seconds
    async def inline(func, *args, **kwargs):
        return func(*args, **kwargs)
    monkeypatch.setattr("proofstack.agents.ac.critic.asyncio.sleep", advance)
    monkeypatch.setattr("proofstack.agents.ac.critic.asyncio.to_thread", inline)
    calls = []
    def retrieve(response_id, *, timeout, on_response, cancel_pending):
        calls.append((response_id, timeout, cancel_pending))
        if response_id == "saved-0":
            clock[0] += timeout
        else:
            on_response(completed(response_id, accepted=False))
    monkeypatch.setattr(pricing_client, "reconcile_background_response", retrieve)
    with pytest.raises(FinalReviewBlocked, match="unresolved"):
        asyncio.run(critic.recover_completed(inp, deadline=clock[0] - 1, cutoff=clock[0] + 1))
    assert calls == [("saved-0", .5, False), ("saved-1", .5, False)]
    assert clock[0] == 101.0
    assert latest_attempts(path)[1]["status"] == "completed"
    assert ctx.budgets.root().counters.usd == pytest.approx(.0011)


def test_default_tail_io_error_propagates_without_marking_run_nonretryable(pipeline, monkeypatch):
    execute, calls, _, _, ctx, agent = pipeline
    out = asyncio.run(execute())
    assert out.submission_approved
    # Model a crash before publication of the manifest; research acceptance is durable.
    manifest = ctx.root_workdir / "batch3-output.json"
    manifest.unlink()
    schedule_path = ctx.root_workdir / "batch3-schedule.json"
    schedule = json.loads(schedule_path.read_text())
    schedule["research_deadline_unix_s"] = time.time() - 1
    schedule_path.write_text(json.dumps(schedule))
    before = dict(calls)
    publish = agent._publish_retained
    monkeypatch.setattr(agent, "_publish_retained", AsyncMock(side_effect=OSError("Temporary storage error")))
    with pytest.raises(OSError, match="Temporary storage error"):
        asyncio.run(execute(resume_run=True))
    assert not manifest.exists() and calls == before
    monkeypatch.setattr(agent, "_publish_retained", publish)
    assert asyncio.run(execute(resume_run=True)).submission_approved
    assert calls["research"] == before["research"]
