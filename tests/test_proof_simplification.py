from __future__ import annotations

import asyncio
import json
import time

import pytest
from pydantic import ValidationError

from proofstack.agents.ac.author import Author
from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow
from proofstack.agents.final_stateless_review import FinalStatelessCritic
from proofstack.agents.proof_simplification import SimplificationCritic, SimplificationInterrupted
from proofstack.budget import BudgetExhausted
from proofstack.registry import load_preset
from test_firstproof_batch3 import ORIGINAL, TEX, pipeline


@pytest.fixture
def simplification(pipeline, monkeypatch):
    execute, calls, seen, options, ctx, agent = pipeline

    async def author(self, inp):
        calls["simplify_author"] += 1
        seen.setdefault("simplify_inputs", []).append(inp)
        seen.setdefault("simplify_caps", []).append(self.ctx.budgets.root().spec.max_usd)
        self.tracker.add_usd(options.get("simplify_author_cost", 1))
        if options.get("author_error"):
            raise options["author_error"]
        number = calls["simplify_author"]
        document = inp.answer_tex if options.get("unchanged") else TEX.replace("Proof.", f"Simpler proof {number}.")
        out = self.Outputs(answer_tex=document, ready=True, research_notes_tex=f"Notes {number}",
                           thinking_summary=f"Removed case split {number}", artifact_status="ok")
        return out.model_copy(update=options.get("simplify_output", {}))

    async def critic(self, inp):
        calls["simplify_critic"] += 1
        seen.setdefault("simplify_reviews", []).append(inp)
        self.tracker.add_usd(1)
        verdicts = options.get("simplify_verdicts", ["accept"])
        verdict = verdicts[min(calls["simplify_critic"] - 1, len(verdicts) - 1)]
        out = self.parse_output(f"Review {calls['simplify_critic']}\n<cleanup_verdict>{verdict}</cleanup_verdict>", inp)
        if verdict == "research" and options.get("stop_accepting_after_baseline_flaw"):
            options.update(agreed=False, research_accepted=False)
        if options.get("simplify_review_error"):
            self.completed_review = out
            raise options["simplify_review_error"]
        return out

    async def final_critic(self, inp):
        calls["final_critic"] += 1
        seen.setdefault("final_inputs", []).append(inp)
        seen.setdefault("final_caps", []).append(self.ctx.budgets.root().spec.max_usd)
        verdicts = options.get("final_verdicts", [True])
        verdict = verdicts[min(calls["final_critic"] - 1, len(verdicts) - 1)]
        text = f"Independent final review\n<answer_ready>{str(verdict).lower()}</answer_ready>"
        out = self.parse_output(text, inp)
        self._completed_review = out
        self.tracker.add_usd(options.get("final_cost", 2))
        if options.get("final_error"):
            raise options["final_error"]
        return out

    monkeypatch.setattr(Author, "run", author)
    monkeypatch.setattr(SimplificationCritic, "run", critic)
    monkeypatch.setattr(FinalStatelessCritic, "run", final_critic)

    async def execute_with_reserves(**overrides):
        return await execute(**{"final_review_reserve_usd": 10, "final_review_seconds": 120,
                                "final_review_export_seconds": 5, **overrides})

    return execute_with_reserves, calls, seen, options, ctx, agent


def states(ctx):
    return list((ctx.root_workdir / "proof_simplification").glob("*/state.json"))


def test_disabled_by_default(simplification):
    execute, calls, _, _, ctx, _ = simplification
    out = asyncio.run(execute())
    assert out.submission_approved
    assert calls["simplify_author"] == calls["simplify_critic"] == 0
    assert not states(ctx)


@pytest.mark.parametrize("value", [-1, 11, True, 1.5])
def test_invalid_round_limit(value):
    with pytest.raises(ValidationError):
        FirstProofBatch3Workflow.Inputs(problem="p", problem_id="p", simplification_rounds=value)


def test_three_rounds_then_cleanup(simplification):
    execute, calls, seen, _, ctx, _ = simplification
    out = asyncio.run(execute(simplification_rounds=3))
    assert out.submission_approved
    assert calls["research"] == calls["rewrite"] == calls["review"] == 1
    assert calls["simplify_author"] == calls["simplify_critic"] == 3
    inputs = seen["simplify_inputs"]
    assert [i.round for i in inputs] == [1, 2, 3]
    assert all(i.n_rounds == 3 and i.problem == ORIGINAL for i in inputs)
    assert inputs[0].answer_tex == TEX
    assert "Simpler proof 1" in inputs[1].answer_tex
    assert "Simpler proof 2" in inputs[2].answer_tex
    assert all("not new problem-solving" in i.workflow_feedback for i in inputs)
    assert all(i.mode == "fresh" and i.prior_messages == [] for i in seen["simplify_reviews"])
    assert "Simpler proof 3" in seen["review_inputs"][0].baseline_tex
    state = json.loads(states(ctx)[0].read_text())
    assert state["completed"] == 3 and state["stage"] == "done"
    # Research spent 20; 15 for partial cleanup plus 10 for final review remain reserved.
    assert seen["simplify_caps"] == [27.5] * 3
    assert calls["final_critic"] == 1
    assert len(list(states(ctx)[0].parent.glob("round-*.json"))) == 3


def test_rejected_attempt_never_becomes_baseline(simplification):
    execute, calls, seen, options, _, _ = simplification
    options["simplify_verdicts"] = ["accept", "repair", "accept"]
    out = asyncio.run(execute(simplification_rounds=3))
    assert out.submission_approved and calls["simplify_critic"] == 3
    assert "Simpler proof 1" in seen["simplify_inputs"][2].answer_tex
    assert "Simpler proof 2" not in seen["simplify_inputs"][2].answer_tex
    assert "Review 2" in seen["simplify_inputs"][2].prev_critique
    assert "Simpler proof 1" in seen["simplify_reviews"][2].baseline_tex


@pytest.mark.parametrize("verdict", ["repair", "invalid", "restore"])
def test_all_rejected_keeps_original(simplification, verdict):
    execute, calls, seen, options, _, _ = simplification
    options["simplify_verdicts"] = [verdict]
    assert asyncio.run(execute(simplification_rounds=3)).submission_approved
    assert calls["simplify_author"] == 3
    assert all(i.answer_tex == TEX for i in seen["simplify_inputs"])
    assert seen["review_inputs"][0].baseline_tex == TEX


@pytest.mark.parametrize("output", [
    {"ready": False}, {"artifact_status": "failed"}, {"compute_instructions": "check"},
    {"council_question": "check"}, {"answer_tex": TEX + "\\input{missing}"},
])
def test_ineligible_candidate_cannot_replace_baseline(simplification, output):
    execute, calls, seen, options, _, _ = simplification
    options["simplify_output"] = output
    assert asyncio.run(execute(simplification_rounds=3)).submission_approved
    assert calls["simplify_author"] == 3 and calls["simplify_critic"] == 0
    assert seen["review_inputs"][0].baseline_tex == TEX


def test_unchanged_candidate_does_not_pay_for_redundant_review(simplification):
    execute, calls, seen, options, ctx, _ = simplification
    options["unchanged"] = True
    assert asyncio.run(execute(simplification_rounds=3)).submission_approved
    assert calls["simplify_author"] == 3 and calls["simplify_critic"] == 0
    assert json.loads(states(ctx)[0].read_text())["completed"] == 3
    assert seen["review_inputs"][0].baseline_tex == TEX


def test_unsolved_draft_does_not_enter_simplification(simplification):
    execute, calls, _, options, ctx, _ = simplification
    options["agreed"] = False
    out = asyncio.run(execute(simplification_rounds=3))
    assert out.partial_ready and not out.submission_approved
    assert calls["simplify_author"] == 0 and not states(ctx)


def test_baseline_flaw_revokes_acceptance(simplification):
    execute, calls, seen, options, ctx, _ = simplification
    options["simplify_verdicts"] = ["research"]
    options["stop_accepting_after_baseline_flaw"] = True
    out = asyncio.run(execute(simplification_rounds=3))
    assert not out.submission_approved
    assert calls["research"] > 1 and calls["rewrite"] == 0
    assert "Prior acceptance is revoked" in seen["resumed_states"][0]["pending_critique"]
    assert seen["research_inputs"][1].n_rounds > seen["research_inputs"][0].n_rounds
    assert json.loads((ctx.root_workdir / "batch3-accepted-baseline.json").read_text())["revoked"]


def test_cleanup_failure_retains_latest_simplified_acceptance(simplification):
    execute, _, _, options, _, _ = simplification
    options["rewrite_error"] = BudgetExhausted("run", "usd", 1, 2)
    out = asyncio.run(execute(simplification_rounds=3))
    assert out.submission_approved and "Simpler proof 3" in out.answer_tex.read_text()


@pytest.mark.parametrize("verdict", ["accept", "research"])
def test_completed_critic_verdict_survives_budget_exception(simplification, verdict):
    execute, _, _, options, ctx, _ = simplification
    options["simplify_verdicts"] = [verdict]
    options["stop_accepting_after_baseline_flaw"] = True
    options["simplify_review_error"] = BudgetExhausted("run", "usd", 1, 2)
    out = asyncio.run(execute(simplification_rounds=3))
    if verdict == "accept":
        assert out.submission_approved
        assert "Simpler proof 1" in json.loads(states(ctx)[0].read_text())["document"]
    else:
        assert not out.submission_approved
        assert json.loads((ctx.root_workdir / "batch3-accepted-baseline.json").read_text())["revoked"]


def test_exhausted_allowance_does_not_start_next_round(simplification):
    execute, calls, seen, options, _, _ = simplification
    options["simplify_author_cost"] = 32
    assert asyncio.run(execute(simplification_rounds=3)).submission_approved
    assert calls["simplify_author"] == calls["simplify_critic"] == 1
    assert "Simpler proof 1" in seen["review_inputs"][0].baseline_tex


def test_interrupting_call_keeps_baseline_and_blocks_further_paid_work(simplification):
    execute, calls, _, options, ctx, agent = simplification
    options["author_error"] = RuntimeError("provider poll failed")
    out = asyncio.run(execute(simplification_rounds=3))
    assert not out.submission_approved and not out.partial_ready
    assert out.accepted_baseline_pending and out.answer_tex.read_text() == TEX
    assert out.output_kind == "accepted_baseline_needs_final_review"
    assert not out.error_retryable and "reconcile provider work" in out.error
    assert calls["rewrite"] == calls["final_critic"] == 0 and calls["simplify_author"] == 1
    inp = FirstProofBatch3Workflow.Inputs(problem=ORIGINAL, problem_id="p", simplification_rounds=3)
    with pytest.raises(SimplificationInterrupted, match="reconcile provider work"):
        asyncio.run(agent._simplify_accepted(inp, TEX, "", "", out, deadline=time.monotonic() + 30, reserve=0))
    assert calls["simplify_author"] == 1
    assert json.loads(states(ctx)[0].read_text())["stage"] == "author"


def test_completed_phase_is_not_repeated_on_resume(simplification):
    execute, calls, _, _, ctx, agent = simplification
    out = asyncio.run(execute(simplification_rounds=3))
    state = json.loads(states(ctx)[0].read_text())
    before = dict(calls)
    inp = FirstProofBatch3Workflow.Inputs(problem=ORIGINAL, problem_id="p", simplification_rounds=3)
    document, _, review = asyncio.run(agent._simplify_accepted(
        inp, TEX, "", "", out, deadline=time.monotonic() + 999, reserve=0,
    ))
    assert document == state["document"] and review is None and calls == before
    assert json.loads(states(ctx)[0].read_text())["deadline_unix_s"] == state["deadline_unix_s"]


def test_review_prompt_requires_actual_simplification_and_preservation(pipeline):
    _, _, _, _, ctx, _ = pipeline
    critic = SimplificationCritic(ctx)
    inp = critic.Inputs(problem=ORIGINAL, baseline_tex="BASELINE", answer_tex="CANDIDATE",
                        prior_messages=[{"role": "assistant", "content": "OLD POSITIVE REVIEW"}])
    messages = critic.render_messages(inp)
    assert len(messages) == 1
    text = messages[0]["content"]
    for phrase in ("BASELINE", "CANDIDATE", "hypotheses, quantifiers", "genuinely simplifies",
                   "Do not choose research merely because the new candidate fails"):
        assert phrase in text
    assert "OLD POSITIVE REVIEW" not in text
    assert critic.MODEL == load_preset("firstproof_batch3").component_configs["Batch3CleanupCritic"]["model"]


def test_cancel_after_author_resumes_saved_candidate_without_repaying(simplification, monkeypatch):
    execute, calls, seen, _, ctx, agent = simplification
    check = agent._check
    interrupted = False

    async def interrupt(document, inp, deadline):
        nonlocal interrupted
        if "Simpler proof 1" in document and not interrupted:
            interrupted = True
            raise asyncio.CancelledError
        return await check(document, inp, deadline)

    monkeypatch.setattr(agent, "_check", interrupt)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(simplification_rounds=3))
    state_before = json.loads(states(ctx)[0].read_text())
    assert state_before["stage"] == "candidate"
    assert calls["simplify_author"] == 1 and calls["simplify_critic"] == 0
    inp = FirstProofBatch3Workflow.Inputs(problem=ORIGINAL, problem_id="p", simplification_rounds=3)
    document, _, review = asyncio.run(agent._simplify_accepted(
        inp, TEX, "", "", agent.Outputs(problem_id="p"), deadline=time.monotonic() + 1000, reserve=15,
    ))
    assert review is None and "Simpler proof 3" in document
    assert calls["simplify_author"] == calls["simplify_critic"] == 3
    state_after = json.loads(states(ctx)[0].read_text())
    assert state_after["deadline_unix_s"] == state_before["deadline_unix_s"]
    assert state_after["max_usd"] == state_before["max_usd"]
    assert seen["simplify_caps"][1] == 26.5  # first Author's dollar is not refunded


def test_cancellation_inside_review_does_not_replay_model_call(simplification, monkeypatch):
    execute, calls, _, _, ctx, agent = simplification

    async def cancel_review(self, inp):
        calls["cancelled_critic"] += 1
        raise asyncio.CancelledError

    monkeypatch.setattr(SimplificationCritic, "run", cancel_review)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(simplification_rounds=3))
    assert json.loads(states(ctx)[0].read_text())["stage"] == "critic"
    assert agent._accepted_baseline.document == TEX
    inp = FirstProofBatch3Workflow.Inputs(problem=ORIGINAL, problem_id="p", simplification_rounds=3)
    with pytest.raises(SimplificationInterrupted):
        asyncio.run(agent._simplify_accepted(
            inp, TEX, "", "", agent.Outputs(problem_id="p"), deadline=time.monotonic() + 30, reserve=0,
        ))
    assert calls["cancelled_critic"] == 1


def test_timeout_between_rounds_still_allows_cleanup(simplification, monkeypatch):
    execute, calls, seen, _, ctx, agent = simplification
    emit = agent.events.emit

    async def emit_then_timeout(kind, *args, **kwargs):
        result = await emit(kind, *args, **kwargs)
        if kind == "batch3.simplification_review":
            raise TimeoutError("simplification phase deadline")
        return result

    monkeypatch.setattr(agent.events, "emit", emit_then_timeout)
    assert asyncio.run(execute(simplification_rounds=3)).submission_approved
    assert calls["simplify_author"] == calls["rewrite"] == 1
    assert "Simpler proof 1" in seen["review_inputs"][0].baseline_tex
    state = json.loads(states(ctx)[0].read_text())
    assert state["stage"] == "done" and state["completed"] == 1


def test_saved_baseline_rejection_is_not_lost_when_resume_allowance_expired(simplification, monkeypatch):
    execute, _, _, options, ctx, agent = simplification
    options["simplify_verdicts"] = ["research"]
    handoff = agent._handoff

    def interrupt_handoff(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(agent, "_handoff", interrupt_handoff)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(simplification_rounds=3))
    assert json.loads(states(ctx)[0].read_text())["stage"] == "rejected"
    monkeypatch.setattr(agent, "_handoff", handoff)
    inp = FirstProofBatch3Workflow.Inputs(problem=ORIGINAL, problem_id="p", simplification_rounds=3)
    _, _, review = asyncio.run(agent._simplify_accepted(
        inp, TEX, "", "", agent.Outputs(problem_id="p"), deadline=time.monotonic() - 1, reserve=1000,
    ))
    assert review.disposition == "research" and agent._accepted_baseline is None


def test_critic_configuration_inherits_cleanup_limits(pipeline):
    _, _, _, _, ctx, _ = pipeline
    ctx.component_configs["Batch3CleanupCritic"].update(context_window_tokens=500000, max_hosted_tool_calls=17)
    ctx.component_configs["SimplificationCritic"] = {"max_hosted_tool_calls": 19}
    critic = SimplificationCritic(ctx)
    assert critic.component_config["context_window_tokens"] == 500000
    assert critic.component_config["max_hosted_tool_calls"] == 19


@pytest.mark.parametrize("stage", ["author", "critic"])
@pytest.mark.parametrize("settled", [False, True])
def test_real_simplification_timeout_preserves_baseline(simplification, monkeypatch, stage, settled):
    execute, calls, _, _, ctx, agent = simplification
    phase = agent._phase

    async def short_phase(name, *, usd, deadline, call):
        if name == "proof_simplification":
            deadline = time.monotonic() + 0.1
        return await phase(name, usd=usd, deadline=deadline, call=call)

    async def slow(self, inp):
        calls["slow"] += 1
        try:
            await asyncio.Event().wait()
        finally:
            (self.workdir / "provider-settlement.json").write_text(json.dumps({"settled": settled}))

    monkeypatch.setattr(agent, "_phase", short_phase)
    monkeypatch.setattr(Author if stage == "author" else SimplificationCritic, "run", slow)
    out = asyncio.run(execute(simplification_rounds=3))
    assert calls["slow"] == 1
    if settled:
        assert out.submission_approved
        assert calls["rewrite"] == calls["final_critic"] == 1
        assert json.loads(states(ctx)[0].read_text())["document"] == TEX
    else:
        assert out.accepted_baseline_pending and out.answer_tex.read_text() == TEX
        assert not out.partial_ready and not out.submission_approved
        assert calls["rewrite"] == calls["final_critic"] == calls["partial"] == 0


@pytest.mark.parametrize("after_cutoff", [False, True])
def test_resume_checks_unresolved_simplification_even_when_disabled(simplification, after_cutoff):
    execute, calls, _, options, ctx, _ = simplification
    options["author_error"] = RuntimeError("transport lost")
    asyncio.run(execute(simplification_rounds=1))
    if after_cutoff:
        path = ctx.root_workdir / "batch3-schedule.json"
        saved = json.loads(path.read_text())
        saved["research_deadline_unix_s"] = time.time() - 1
        path.write_text(json.dumps(saved))
    before = dict(calls)
    out = asyncio.run(execute(resume_run=True, simplification_rounds=0))
    assert out.accepted_baseline_pending and not out.error_retryable
    assert dict(calls) == before


def test_operator_settlement_unblocks_parent_simplification_checkpoint(simplification):
    from proofstack.subscription import reconcile_stopped_call

    execute, calls, _, options, ctx, _ = simplification
    options["author_error"] = RuntimeError("worker interrupted")
    assert asyncio.run(execute(simplification_rounds=1)).accepted_baseline_pending
    state = json.loads(states(ctx)[0].read_text())
    [folder] = (ctx.root_workdir / "agents").glob(f"{state['call_agent']}-c*")
    receipt_id = "b" * 64
    receipts = ctx.root_workdir / "subscription_calls"
    receipts.mkdir()
    (receipts / f"{receipt_id}.json").write_text('{"status":"started","attempts":1}')
    (folder / "subscription-settlement.json").write_text(json.dumps({"settled": False, "receipt_id": receipt_id}))
    reconcile_stopped_call(ctx.root_workdir, receipt_id, unreported_allowance_usd=5, workers_stopped=True)
    out = asyncio.run(execute(resume_run=True, simplification_rounds=0))
    assert out.submission_approved
    assert calls["simplify_author"] == 1
    assert calls["rewrite"] == calls["final_critic"] == 1
    assert json.loads(states(ctx)[0].read_text())["stage"] == "done"


@pytest.mark.parametrize("verdict", ["accept", "research"])
def test_resume_processes_saved_review_before_cutoff_shortcut(simplification, monkeypatch, verdict):
    execute, calls, _, _, ctx, _ = simplification

    async def interrupted(self, inp):
        self._on_response(f"Found issue\n<cleanup_verdict>{verdict}</cleanup_verdict>", inp)
        (self.workdir / "provider-settlement.json").write_text('{"settled":true}')
        raise asyncio.CancelledError

    monkeypatch.setattr(SimplificationCritic, "run", interrupted)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(simplification_rounds=1))
    path = ctx.root_workdir / "batch3-schedule.json"
    saved = json.loads(path.read_text())
    saved["research_deadline_unix_s"] = time.time() - 1
    path.write_text(json.dumps(saved))
    out = asyncio.run(execute(resume_run=True, simplification_rounds=0))
    assert calls["research"] == 1
    if verdict == "accept":
        assert out.submission_approved and "Simpler proof 1" in out.answer_tex.read_text()
    else:
        assert not out.submission_approved and calls["final_critic"] == 0
        assert json.loads((ctx.root_workdir / "batch3-accepted-baseline.json").read_text())["revoked"]


@pytest.mark.parametrize("unchanged", [True, False])
def test_cleanup_history_uses_explicit_baseline_provenance(simplification, unchanged):
    execute, _, seen, options, ctx, _ = simplification
    options["unchanged"] = unchanged
    asyncio.run(execute(simplification_rounds=1))
    inp = seen["review_inputs"][0]
    from test_firstproof_batch3 import HISTORY
    from proofstack.agents.batch3_critic import Batch3CleanupCritic
    assert inp.prior_messages == HISTORY
    assert bool(inp.baseline_provenance) is not unchanged
    rendered = Batch3CleanupCritic(ctx).render_messages(inp)
    if not unchanged:
        assert "Baseline provenance" in rendered[-1]["content"]
