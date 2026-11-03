"""Offline coverage for supplementary writes and bounded recovery retries."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from proofstack import atomic
from proofstack.agents.batch3_critic import Batch3CleanupCritic
from proofstack.agents.final_stateless_review import FinalReviewBlocked, FinalStatelessCritic
from proofstack.budget import BudgetExhausted, BudgetSpec
from proofstack.context import RunContext
from test_review_seven import completed, saved_review
from test_review_six import pricing_client


@pytest.fixture
def failed_checkpoint(monkeypatch):
    write = atomic.write_text_atomic
    attempts = []
    def fail_receipt(path, text, **kwargs):
        if path.name == "completed_review.json":
            attempts.append(path)
            raise OSError("Synthetic checkpoint write failure")
        return write(path, text, **kwargs)
    monkeypatch.setattr(atomic, "write_text_atomic", fail_receipt)
    return attempts


@pytest.mark.parametrize("verdict", ["accept", "research"])
@pytest.mark.parametrize("over_budget", [False, True])
def test_failed_cleanup_checkpoint_does_not_skip_metering(tmp_path, failed_checkpoint, verdict, over_budget):
    calls = []
    class Client:
        model = "synthetic"
        def run_queries(self, conversations, **kwargs):
            calls.append(conversations)
            yield 0, [*conversations[0], {"role": "assistant", "content":
                f"Saved findings\n<cleanup_verdict>{verdict}</cleanup_verdict>"}], {
                    "cost": 2, "input_tokens": 10, "output_tokens": 20, "reasoning_tokens": 5}
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, api_client_factory=lambda _: Client(),
                            run_budget=BudgetSpec(max_usd=1 if over_budget else 10))
    critic = Batch3CleanupCritic(ctx)
    if over_budget:
        with pytest.raises(BudgetExhausted) as raised:
            asyncio.run(critic(problem="Synthetic", answer_tex="Proof", baseline_tex="Proof"))
        result = raised.value.completed_output
    else:
        result = asyncio.run(critic(problem="Synthetic", answer_tex="Proof", baseline_tex="Proof"))
    assert result.disposition == critic.completed_review.disposition == verdict
    assert len(calls) == len(failed_checkpoint) == 1
    assert not (critic.workdir / "completed_review.json").exists()
    assert ctx.budgets.root().counters.usd == 2
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    usage, = [event["payload"] for event in events if event["kind"] == "model.call"]
    assert usage["cost_usd"] == 2 and usage["in_tokens"] == 10 and usage["out_tokens"] == 20


@pytest.mark.parametrize("error_type", [TimeoutError, asyncio.CancelledError])
def test_checkpoint_write_does_not_replace_original_interruption(tmp_path, monkeypatch, failed_checkpoint, error_type):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
                            api_client_factory=lambda _: SimpleNamespace(model="synthetic"))
    critic = Batch3CleanupCritic(ctx)
    error = error_type("Synthetic interruption")
    error.completed_report = "Saved findings\n<cleanup_verdict>research</cleanup_verdict>"
    monkeypatch.setattr(critic, "_query", AsyncMock(side_effect=error))
    with pytest.raises(error_type) as raised:
        asyncio.run(critic(problem="Synthetic", answer_tex="Proof", baseline_tex="Proof"))
    assert raised.value is error
    assert critic.completed_review.disposition == "research" and len(failed_checkpoint) == 1


@pytest.fixture
def recovery(tmp_path, monkeypatch, pricing_client):
    clock = [100.0]
    monkeypatch.setattr("proofstack.agents.ac.critic.time", SimpleNamespace(monotonic=lambda: clock[0]))
    async def advance(seconds):
        clock[0] += seconds
    async def inline(func, *args, **kwargs):
        return func(*args, **kwargs)
    monkeypatch.setattr("proofstack.agents.ac.critic.asyncio.sleep", advance)
    monkeypatch.setattr("proofstack.agents.ac.critic.asyncio.to_thread", inline)
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, run_budget=BudgetSpec(max_usd=0),
        api_client_factory=lambda _: pricing_client,
        component_configs={"ACCritic": {"research_notes_transport": "inline"}})
    inp = FinalStatelessCritic.Inputs(problem="Synthetic", answer_tex="Proof", mode="fresh")
    critic, path = saved_review(ctx, pricing_client, inp)
    monkeypatch.setattr(critic.tracker, "remaining_wallclock_s", lambda: 3600)
    return ctx, critic, inp, path, clock


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("failure", ["raised", "empty"])
def test_expired_review_retries_failed_get_and_charges_once(recovery, monkeypatch, pricing_client, accepted, failure):
    ctx, critic, inp, _, clock = recovery
    calls = []
    def retrieve(response_id, *, timeout, on_response, cancel_pending):
        calls.append((clock[0], response_id, cancel_pending))
        if len(calls) == 1:
            if failure == "raised":
                raise TimeoutError("Synthetic failed GET")
            return {}
        on_response(completed(response_id, accepted))
    monkeypatch.setattr(pricing_client, "reconcile_background_response", retrieve)
    for _ in range(2):
        result = asyncio.run(critic.recover_completed(inp, deadline=90, cutoff=200))
        assert result.answer_ready is accepted
    assert calls == [(100, "saved-0", False), (105, "saved-0", False)]
    assert ctx.budgets.root().counters.usd == pytest.approx(.0011)


@pytest.mark.parametrize("cutoff,expected_times", [(200, [100, 105, 110, 115, 120, 125]), (107, [100, 105]), (100, [])])
@pytest.mark.parametrize("failure", ["raised", "empty"])
def test_failed_get_retries_share_one_window(recovery, monkeypatch, pricing_client, cutoff, expected_times, failure):
    ctx, critic, inp, _, clock = recovery
    calls = []
    def retrieve(response_id, *, timeout, on_response, cancel_pending):
        calls.append((clock[0], response_id, cancel_pending))
        if failure == "raised":
            raise TimeoutError("Synthetic persistent GET failure")
        return {}
    monkeypatch.setattr(pricing_client, "reconcile_background_response", retrieve)
    with pytest.raises(FinalReviewBlocked, match="unresolved"):
        asyncio.run(critic.recover_completed(inp, deadline=90, cutoff=cutoff))
    assert calls == [(when, "saved-0", False) for when in expected_times]
    assert clock[0] == min(cutoff, 130)
    assert ctx.budgets.root().counters.usd == 0
