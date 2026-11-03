"""Offline regressions for missing verdicts and provider-neutral settlement."""
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mathagents.api_client import APIClient
from mathagents.provider_trace import ProviderTrace
from proofstack.agents.batch3_critic import Batch3CleanupCritic
from proofstack.agents.final_stateless_review import FinalStatelessCritic
from proofstack.agents.proof_simplification import SimplificationCritic
from proofstack.budget import BudgetExhausted, BudgetSpec
from proofstack.context import RunContext
from proofstack.provider_accounting import provider_attempts_settled
from test_firstproof_batch3 import TEX, pipeline
from test_proof_simplification import simplification, states


@pytest.mark.parametrize("verdict", ["accept", "research"])
@pytest.mark.parametrize("over_budget", [False, True])
def test_unserializable_checkpoint_does_not_skip_metering(tmp_path, monkeypatch, verdict, over_budget):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
        api_client_factory=lambda _: SimpleNamespace(model="synthetic"),
        run_budget=BudgetSpec(max_usd=1 if over_budget else 10))
    critic = Batch3CleanupCritic(ctx)
    inp = critic.Inputs(problem="Synthetic", answer_tex="Proof", baseline_tex="Proof")
    raw = f"Synthetic\ud800 findings\n<cleanup_verdict>{verdict}</cleanup_verdict>"
    with pytest.raises(ValueError):
        critic.parse_output(raw, inp).model_dump_json()
    query = AsyncMock(return_value=(0, [{"role": "assistant", "content": raw}],
        {"cost": 2, "input_tokens": 10, "output_tokens": 20, "reasoning_tokens": 5}))
    monkeypatch.setattr(critic, "_query", query)
    if over_budget:
        with pytest.raises(BudgetExhausted) as raised:
            asyncio.run(critic.run(inp))
        result = raised.value.completed_output
    else:
        result = asyncio.run(critic.run(inp))
    assert result.disposition == critic.completed_review.disposition == verdict
    assert critic.completed_review.review_md == raw and query.await_count == 1
    assert not (critic.workdir / "completed_review.json").exists()
    assert ctx.budgets.root().counters.usd == 2
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    usage, = [event["payload"] for event in events if event["kind"] == "model.call"]
    assert usage["cost_usd"] == 2 and usage["in_tokens"] == 10 and usage["out_tokens"] == 20


@pytest.mark.parametrize("error_type", [TimeoutError, asyncio.CancelledError])
def test_serialization_failure_preserves_original_interruption(tmp_path, monkeypatch, error_type):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
        api_client_factory=lambda _: SimpleNamespace(model="synthetic"))
    critic = Batch3CleanupCritic(ctx)
    error = error_type("Synthetic interruption")
    error.completed_report = "Synthetic\ud800 findings\n<cleanup_verdict>research</cleanup_verdict>"
    monkeypatch.setattr(critic, "_query", AsyncMock(side_effect=error))
    with pytest.raises(error_type) as raised:
        asyncio.run(critic.run(critic.Inputs(problem="Synthetic", answer_tex="Proof")))
    assert raised.value is error
    assert critic.completed_review.disposition == "research"


@pytest.mark.parametrize("provider", ["openai", "openrouter", "together"])
@pytest.mark.parametrize("artifact", ["missing", "completed_review.json", "output.json", "provider"])
@pytest.mark.parametrize("verdict", ["accept", "research"])
def test_completed_simplification_requires_a_recoverable_verdict(
        simplification, monkeypatch, provider, artifact, verdict):
    execute, calls, _, _, ctx, _ = simplification

    async def interrupted(self, inp):
        calls["interrupted_critic"] += 1
        trace = ProviderTrace(self.workdir / "provider-attempts.jsonl", call_id="synthetic-paid-review")
        terminal = {"status": "completed"} if provider == "openai" else {"chat_completion_terminal": True}
        trace.update(("review", 0), provider=provider, response_id="saved-review",
                     cost=2, usage_unavailable=False, **terminal)
        report = f"Synthetic findings\n<cleanup_verdict>{verdict}</cleanup_verdict>"
        if artifact == "provider":
            (self.workdir / f"provider-completed-{trace.id}.json").write_text(json.dumps({"report": report}))
        elif artifact != "missing":
            (self.workdir / artifact).write_text(self.parse_output(report, inp).model_dump_json())
        raise asyncio.CancelledError

    monkeypatch.setattr(SimplificationCritic, "run", interrupted)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(simplification_rounds=1))
    path, = states(ctx)
    before_state = path.read_bytes()
    before_calls = dict(calls)
    before_spend = ctx.budgets.root().counters.usd
    schedule_path = ctx.root_workdir / "batch3-schedule.json"
    schedule = json.loads(schedule_path.read_text())
    schedule["research_deadline_unix_s"] = time.time() - 1
    schedule_path.write_text(json.dumps(schedule))
    out = asyncio.run(execute(resume_run=True, simplification_rounds=0))
    if artifact == "missing":
        again = asyncio.run(execute(resume_run=True, simplification_rounds=0))
        for result in (out, again):
            assert result.accepted_baseline_pending and not result.submission_approved
            assert "no recoverable verdict" in result.error and not result.error_retryable
            assert result.accepted_baseline_path.read_text() == TEX
        assert path.read_bytes() == before_state
        assert dict(calls) == before_calls
        assert ctx.budgets.root().counters.usd == before_spend + 2
    elif verdict == "accept":
        assert out.submission_approved and "Simpler proof 1" in out.answer_tex.read_text()
        assert calls["final_critic"] == 1
    else:
        assert not out.submission_approved and calls["final_critic"] == 0
        assert json.loads((ctx.root_workdir / "batch3-accepted-baseline.json").read_text())["revoked"]
    assert calls["research"] == calls["simplify_author"] == calls["interrupted_critic"] == 1


@pytest.mark.parametrize("provider", ["openai", "openrouter", "together"])
@pytest.mark.parametrize("unresolved", [
    {"reconciliation_pending": True}, {"usage_unavailable": True},
    {"chat_completion_terminal": False}, {"chat_completion_terminal": "true"},
])
def test_chat_completion_evidence_does_not_bypass_settlement_guards(provider, unresolved):
    row = {"provider": provider, "response_id": "saved-chat", "chat_completion_terminal": True,
           "usage_unavailable": False, "cost": .0011, **unresolved}
    assert not provider_attempts_settled([row])


@pytest.mark.parametrize("provider", ["openai", "openrouter", "together"])
@pytest.mark.parametrize("accepted", [False, True])
def test_saved_chat_review_recovers_and_is_charged_once(tmp_path, monkeypatch, provider, accepted):
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "synthetic")
    client = APIClient(api=provider, model="synthetic", use_openai_responses_api=False,
                       read_cost=10, write_cost=50)
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
        component_configs={"ACCritic": {"research_notes_transport": "inline"}},
        api_client_factory=lambda _: pytest.fail("Recovery must not start another model call"))
    critic = FinalStatelessCritic(ctx)
    inp = critic.Inputs(problem="Synthetic", answer_tex="Proof", mode="fresh")
    raw = f"Synthetic review\n<answer_ready>{str(accepted).lower()}</answer_ready>"
    folder = tmp_path / "agents/saved"
    trace = ProviderTrace(folder / "provider-attempts.jsonl", call_id="paid-chat-review")
    trace.request(client, "chat", 0, {})
    trace.response(client, "chat", 0, {"id": "saved-chat", "choices": [
        {"finish_reason": "stop", "message": {"role": "assistant", "content": raw}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20}})
    checkpoint = critic._recovery_checkpoint(inp)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({"status": "completed", "kind": "ordinary",
        "attempt_workdir": "agents/saved", "review_input": inp.model_dump(mode="json"),
        "output": critic.parse_output(raw, inp).model_dump(mode="json")}))
    for _ in range(2):
        assert asyncio.run(critic.recover_completed(inp)).answer_ready is accepted
    assert ctx.budgets.root().counters.usd == pytest.approx(.0011)
