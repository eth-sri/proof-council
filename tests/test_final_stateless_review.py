from __future__ import annotations

import asyncio
import hashlib
import json
import time

import pytest

from proofstack.agents.ac.critic import ACCritic
from proofstack.agents.final_stateless_review import FinalReviewBlocked, FinalStatelessCritic
from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow
from proofstack.budget import BudgetExhausted
from proofstack.kinds.api_call import APICallAgent
from test_firstproof_batch3 import ORIGINAL, TEX, pipeline
from test_proof_simplification import simplification


def receipts(ctx):
    return list((ctx.root_workdir / "final_stateless_review").glob("*.json"))


def test_final_review_follows_cleanup_and_is_blind(simplification):
    execute, calls, seen, _, ctx, _ = simplification
    out = asyncio.run(execute(simplification_rounds=3))
    assert out.submission_approved and calls["final_critic"] == 1
    inp = seen["final_inputs"][0]
    assert inp.problem == ORIGINAL and inp.answer_tex == out.answer_tex.read_text()
    assert "Revision rewrite" in inp.answer_tex
    assert inp.mode == "fresh" and inp.prior_messages == [] and inp.omit_author_thinking
    assert inp.research_notes_tex == inp.author_thinking == inp.references_bib == ""
    critic = FinalStatelessCritic(ctx)
    assert critic.render_messages(inp) == ACCritic(ctx).render_messages(inp)
    text = critic.render_messages(inp)[0]["content"]
    assert "Research findings" not in text and "Independent final review" not in text
    assert "accepted baseline" not in text.lower()
    assert out.final_critic_review_md == "Independent final review"
    state = json.loads(receipts(ctx)[0].read_text())
    assert state["status"] == "completed" and state["review"]["answer_ready"]
    assert out.submission_sha256 == hashlib.sha256(inp.answer_tex.encode()).hexdigest()


def test_final_gate_can_be_enabled_without_simplification(simplification):
    execute, calls, _, _, _, _ = simplification
    assert asyncio.run(execute(final_stateless_review=True)).submission_approved
    assert calls["simplify_author"] == 0 and calls["final_critic"] == 1


def test_disabling_switch_does_not_disable_gate_with_simplification(simplification):
    execute, calls, _, _, _, _ = simplification
    assert asyncio.run(execute(simplification_rounds=1, final_stateless_review=False)).submission_approved
    assert calls["final_critic"] == 1


def test_reserves_are_withheld_from_earlier_phases(simplification):
    execute, _, seen, _, ctx, _ = simplification
    asyncio.run(execute(simplification_rounds=3, partial_cleanup_seconds=100))
    assert seen["research_caps"] == [75]
    assert seen["simplify_caps"] == [27.5] * 3
    assert seen["final_caps"] == [10]
    schedule = json.loads((ctx.root_workdir / "batch3-schedule.json").read_text())
    assert schedule["final_stateless_review"] and schedule["final_review_reserve_usd"] == 10
    assert schedule["run_deadline_unix_s"] - schedule["research_deadline_unix_s"] == pytest.approx(225, abs=0.1)
    events = [json.loads(line) for line in (ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    phases = [e["payload"] for e in events if e["kind"] == "batch3.phase_start"]
    cleanup = next(e for e in phases if e["phase"] == "candidate_cleanup")
    assert cleanup["max_usd"] == 49
    assert cleanup["max_wallclock_s"] <= 875
    final = next(e for e in phases if e["phase"] == "final_stateless_review")
    assert 0 < final["max_wallclock_s"] <= 120


@pytest.mark.parametrize("overrides", [
    {"final_review_reserve_usd": 90},
    {"final_review_seconds": 999},
])
def test_insufficient_initial_reserve_fails_before_paid_work(simplification, overrides):
    execute, calls, _, _, _, _ = simplification
    with pytest.raises(ValueError, match="reserves"):
        asyncio.run(execute(final_stateless_review=True, **overrides))
    assert calls["research"] == calls["final_critic"] == 0


def test_final_rejection_returns_to_research_then_rechecks_new_bytes(simplification):
    execute, calls, seen, options, _, _ = simplification
    options["final_verdicts"] = [False, True]
    out = asyncio.run(execute(simplification_rounds=1))
    assert out.submission_approved and calls["research"] == calls["final_critic"] == 2
    assert "Final stateless critic" in seen["resumed_states"][0]["pending_critique"]
    assert seen["final_inputs"][0].answer_tex != seen["final_inputs"][1].answer_tex
    assert seen["final_inputs"][1].prior_messages == []


def test_final_rejection_never_exports_accepted_fallback(simplification):
    execute, calls, _, options, ctx, _ = simplification
    options["final_verdicts"] = [False]
    out = asyncio.run(execute(simplification_rounds=1))
    assert not out.submission_approved and out.partial_ready
    assert calls["final_critic"] > 0
    assert not list((ctx.root_workdir / "submissions").rglob("*.tex"))


@pytest.mark.parametrize("verdict", [None, "missing"])
def test_invalid_final_verdict_blocks_approval_and_paid_retries(simplification, verdict):
    execute, calls, _, options, _, _ = simplification
    options["final_verdicts"] = [verdict]
    out = asyncio.run(execute(simplification_rounds=1))
    assert not out.submission_approved and out.partial_ready and not out.error_retryable
    assert calls["research"] == calls["rewrite"] == calls["final_critic"] == 1
    assert calls["partial"] == 0


def test_completed_final_acceptance_survives_charged_budget_exception(simplification):
    execute, calls, _, options, _, _ = simplification
    options["final_cost"] = 11
    out = asyncio.run(execute(simplification_rounds=1))
    assert out.submission_approved and calls["final_critic"] == 1


@pytest.mark.parametrize("error", [TimeoutError("deadline"), RuntimeError("provider poll failed")])
def test_final_error_cannot_publish_baseline_or_start_partial_editor(simplification, error):
    execute, calls, _, options, ctx, _ = simplification
    options["final_error"] = error
    out = asyncio.run(execute(simplification_rounds=1))
    assert not out.submission_approved and out.partial_ready
    assert not out.error_retryable and "Final stateless review" in out.error
    assert calls["partial"] == 0 and calls["final_critic"] == 1
    assert json.loads(receipts(ctx)[0].read_text())["status"] == "pending"


def test_cleanup_failure_fallback_still_needs_final_review(simplification):
    execute, calls, seen, options, _, _ = simplification
    options["rewrite_error"] = BudgetExhausted("run", "usd", 1, 2)
    out = asyncio.run(execute(simplification_rounds=1))
    assert out.submission_approved and calls["final_critic"] == 1
    assert "Simpler proof" in seen["final_inputs"][0].answer_tex
    assert "Revision rewrite" not in seen["final_inputs"][0].answer_tex


def test_direct_publication_and_changed_bytes_cannot_bypass_gate(simplification):
    execute, _, _, _, _, agent = simplification
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p", simplification_rounds=1)
    with pytest.raises(FinalReviewBlocked):
        agent._publish(inp, agent.Outputs(problem_id="p"), TEX, 1, partial=False)
    out = asyncio.run(execute(simplification_rounds=1))
    with pytest.raises(FinalReviewBlocked):
        agent._publish(inp, out, out.answer_tex.read_text() + "\n% altered", 1, partial=False)
    changed_problem = inp.model_copy(update={"problem": "Another problem"})
    assert agent._final_approval(changed_problem, out.answer_tex.read_text()) is None
    assert agent._final_approval(inp.model_copy(update={"page_limit": 15}), out.answer_tex.read_text()) is None


def test_legacy_accepted_output_does_not_bypass_new_gate(simplification):
    execute, _, _, _, ctx, agent = simplification
    out = asyncio.run(execute())
    assert out.submission_approved
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p", final_stateless_review=True)
    assert not agent._restore_output(inp).submission_approved
    assert not json.loads((ctx.root_workdir / "batch3-output.json").read_text())["submission_approved"]


def test_resume_does_not_disable_gate_or_repeat_completed_review(simplification):
    execute, calls, _, _, ctx, _ = simplification
    out = asyncio.run(execute(simplification_rounds=1))
    before = dict(calls)
    resumed = asyncio.run(execute(resume_run=True, simplification_rounds=0,
                                 final_stateless_review=False, final_review_reserve_usd=1))
    assert resumed.submission_approved and resumed.submission_sha256 == out.submission_sha256
    assert calls == before
    assert json.loads((ctx.root_workdir / "batch3-schedule.json").read_text())["final_review_reserve_usd"] == 10


def test_resume_pending_review_precedes_any_research_or_cleanup(simplification, monkeypatch):
    execute, calls, _, _, ctx, _ = simplification
    original = FinalStatelessCritic.run

    async def cancel(self, inp):
        calls["cancelled_final"] += 1
        raise asyncio.CancelledError

    monkeypatch.setattr(FinalStatelessCritic, "run", cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(simplification_rounds=1))
    state_before = json.loads(receipts(ctx)[0].read_text())
    before = dict(calls)
    monkeypatch.setattr(FinalStatelessCritic, "run", original)
    out = asyncio.run(execute(resume_run=True, simplification_rounds=0))
    assert out.submission_approved
    assert calls["research"] == before["research"] and calls["rewrite"] == before["rewrite"]
    state_after = json.loads(receipts(ctx)[0].read_text())
    assert state_after["deadline_unix_s"] == state_before["deadline_unix_s"]
    assert state_after["max_usd"] == state_before["max_usd"]


def test_resume_completed_verdict_before_publication_does_not_repay(simplification, monkeypatch):
    execute, calls, _, _, ctx, agent = simplification
    publish = agent._publish

    def cancel(inp, out, document, pages, *, partial):
        if not partial:
            raise asyncio.CancelledError
        return publish(inp, out, document, pages, partial=partial)

    monkeypatch.setattr(agent, "_publish", cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(simplification_rounds=1))
    assert calls["final_critic"] == 1
    monkeypatch.setattr(agent, "_publish", publish)
    out = asyncio.run(execute(resume_run=True, simplification_rounds=0))
    assert out.submission_approved and calls["final_critic"] == calls["rewrite"] == 1


def test_final_critic_uses_separate_provider_receipts_and_no_agent_cache(pipeline):
    _, _, _, _, ctx, _ = pipeline
    ctx.component_configs["ACCritic"]["max_hosted_tool_calls"] = 19
    regular, final = ACCritic(ctx), FinalStatelessCritic(ctx)
    inp = regular.Inputs(problem=ORIGINAL, answer_tex=TEX, omit_author_thinking=True)
    assert final.component_config["max_hosted_tool_calls"] == 19
    assert final.render_messages(inp) == regular.render_messages(inp)
    assert final._recovery_checkpoint(inp) != regular._recovery_checkpoint(inp)
    assert not final.cache_enabled


def test_real_critic_recovers_completed_report_after_cancellation_without_repaying(simplification, monkeypatch):
    execute, calls, _, _, _, _ = simplification
    monkeypatch.setattr(FinalStatelessCritic, "run", ACCritic.run)

    async def completed_then_cancelled(self, inp):
        calls["paid_final"] += 1
        self.tracker.add_usd(2)
        self._on_response("Cold review\n<answer_ready>true</answer_ready>", inp)
        raise asyncio.CancelledError

    monkeypatch.setattr(APICallAgent, "run", completed_then_cancelled)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(simplification_rounds=1))
    assert calls["paid_final"] == 1
    out = asyncio.run(execute(resume_run=True, simplification_rounds=0))
    assert out.submission_approved and out.final_critic_review_md == "Cold review"
    assert calls["paid_final"] == calls["rewrite"] == calls["research"] == 1


def test_real_final_timeout_is_fail_closed_and_resume_cannot_reset_clock(simplification, monkeypatch):
    execute, calls, _, _, ctx, _ = simplification

    async def wait_forever(self, inp):
        calls["timed_final"] += 1
        await asyncio.Event().wait()

    monkeypatch.setattr(FinalStatelessCritic, "run", wait_forever)
    out = asyncio.run(execute(simplification_rounds=1, final_review_seconds=0.05))
    assert not out.submission_approved and out.partial_ready
    state = json.loads(receipts(ctx)[0].read_text())
    assert state["deadline_unix_s"] < time.time()
    before = dict(calls)
    resumed = asyncio.run(execute(resume_run=True, simplification_rounds=0, final_review_seconds=100))
    assert not resumed.submission_approved and calls["timed_final"] == 1
    assert calls["research"] == before["research"] and calls["rewrite"] == before["rewrite"]
    assert json.loads(receipts(ctx)[0].read_text())["deadline_unix_s"] == state["deadline_unix_s"]


def test_final_review_resume_preserves_charged_allowance(simplification, monkeypatch):
    execute, calls, seen, _, ctx, _ = simplification
    original = FinalStatelessCritic.run

    async def charged_then_cancelled(self, inp):
        self.tracker.add_usd(3)
        raise asyncio.CancelledError

    monkeypatch.setattr(FinalStatelessCritic, "run", charged_then_cancelled)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(simplification_rounds=1))
    state = json.loads(receipts(ctx)[0].read_text())
    monkeypatch.setattr(FinalStatelessCritic, "run", original)
    assert asyncio.run(execute(resume_run=True, final_review_reserve_usd=100)).submission_approved
    assert seen["final_caps"] == [7]
    assert json.loads(receipts(ctx)[0].read_text())["initial_spend"] == state["initial_spend"]


def test_rejection_at_research_cutoff_does_not_restart_research(simplification, monkeypatch):
    execute, calls, _, options, _, agent = simplification
    options["final_verdicts"] = [False]
    original = FinalStatelessCritic.run
    real_time = time.time
    real_monotonic = time.monotonic
    offset = [0]
    from types import SimpleNamespace
    monkeypatch.setattr("proofstack.agents.firstproof_batch3.time", SimpleNamespace(
        time=lambda: real_time() + offset[0], monotonic=lambda: real_monotonic() + offset[0]))

    async def finish_at_cutoff(self, inp):
        result = await original(self, inp)
        offset[0] = 860  # Past research cutoff (775s), before final phase deadline/run deadline.
        return result

    # Start final review late enough that its own 120-second window is still open.
    run_final = agent._review_final

    async def start_late(*args, **kwargs):
        offset[0] = 770
        return await run_final(*args, **kwargs)

    monkeypatch.setattr(agent, "_review_final", start_late)
    monkeypatch.setattr(FinalStatelessCritic, "run", finish_at_cutoff)
    out = asyncio.run(execute(simplification_rounds=1, partial_cleanup_seconds=100))
    assert not out.submission_approved and calls["research"] == calls["final_critic"] == 1


def test_pending_review_configuration_change_blocks_resume(simplification, monkeypatch):
    execute, calls, _, _, ctx, _ = simplification

    async def cancel(self, inp):
        raise asyncio.CancelledError

    monkeypatch.setattr(FinalStatelessCritic, "run", cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(simplification_rounds=1))
    before = dict(calls)
    ctx.component_configs["ACCritic"]["max_hosted_tool_calls"] = 31
    with pytest.raises(FinalReviewBlocked, match="configuration changed"):
        asyncio.run(execute(resume_run=True))
    assert calls == before
