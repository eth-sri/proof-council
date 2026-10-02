"""Offline context recovery; fixtures contain no private research material."""
import asyncio
import json

import pytest

from proofstack.agents.ac.critic import ACCritic, CriticContextTooLarge
from proofstack.budget import BudgetExhausted
from proofstack.context import RunContext
from proofstack.kinds.api_call import APICallAgent


@pytest.fixture
def runner():
    with asyncio.Runner() as value:
        yield value


@pytest.fixture
def critic(tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("test attempted a provider call")

    ctx = RunContext.create(root_workdir=tmp_path, flat=True, api_client_factory=no_network,
                            component_configs={"ACCritic": {"research_notes_transport": "inline"}})
    return ACCritic(ctx)


def packet(**updates):
    return ACCritic.Inputs(**{
        "problem": "Original question", "answer_tex": "Complete proof",
        "research_notes_tex": "Current notes", "references_bib": "References",
        "author_thinking": "Author summary", "mode": "stateful", "round": 2,
        "prior_messages": [{"role": "user", "content": "Old draft"},
                           {"role": "assistant", "content": "Unresolved objection"}],
        **updates,
    })


def events(critic):
    return [json.loads(line) for line in (critic.ctx.root_workdir / "events.jsonl").read_text().splitlines()]


def install_reply(monkeypatch, seen):
    async def reply(self, inp):
        seen.append(inp)
        return self.parse_output("Full review\n<answer_ready>false</answer_ready>", inp)

    monkeypatch.setattr(APICallAgent, "run", reply)


def test_fitting_conversation_is_unchanged(critic, monkeypatch):
    seen = []
    install_reply(monkeypatch, seen)
    inp = packet()
    out = asyncio.run(critic(**inp.model_dump()))
    assert seen == [inp]
    assert out.messages_after[:-2] == inp.prior_messages
    assert out.mode == "stateful"
    saved = json.loads(critic._recovery_checkpoint(inp).read_text())
    assert saved["kind"] == "ordinary" and saved["status"] == "completed"
    assert saved["review_input"] == inp.model_dump(mode="json")
    assert not any(e["kind"] == "ac.critic.context.reduced" for e in events(critic))


def test_compacts_only_when_oversized_and_preserves_every_finding(critic, monkeypatch):
    seen = []
    install_reply(monkeypatch, seen)
    history = []
    for n in range(4):
        history += [{"role": "user", "content": "OLD_DRAFT" + "x" * 700_000},
                    {"role": "assistant", "content": f"Finding {n}"}]
    inp = packet(prior_messages=history)
    out = asyncio.run(critic(**inp.model_dump()))
    assert len(seen) == 1 and seen[0].context_compacted
    assert out.mode == "stateful"
    assert out.messages_after[:-2] == history[1::2]
    current = out.messages_after[-2]["content"]
    for text in (inp.problem, inp.answer_tex, inp.research_notes_tex, inp.references_bib):
        assert text in current
    assert "OLD_DRAFT" not in json.dumps(out.messages_after)
    assert inp.prior_messages == history
    stages = [e["payload"]["stage"] for e in events(critic) if e["kind"] == "ac.critic.context.preflight"]
    assert stages == ["original", "compacted"]


def test_oversized_reports_reset_without_truncating_current_files(critic, monkeypatch):
    seen = []
    install_reply(monkeypatch, seen)
    inp = packet(prior_messages=[{"role": "assistant", "content": "x" * 3_000_000}])
    out = asyncio.run(critic(**inp.model_dump()))
    assert len(seen) == 1 and seen[0].mode == "fresh"
    assert seen[0].omit_author_thinking and not seen[0].prior_messages
    assert out.mode == "fresh" and len(out.messages_after) == 2
    assert seen[0].answer_tex == inp.answer_tex
    assert seen[0].research_notes_tex == inp.research_notes_tex


@pytest.mark.parametrize("field", ["problem", "answer_tex", "research_notes_tex", "references_bib"])
def test_estimate_cannot_falsely_reject_fresh_packet(critic, field, monkeypatch):
    seen = []
    install_reply(monkeypatch, seen)
    inp = packet(mode="fresh", prior_messages=[], **{field: "x" * 3_000_000})
    asyncio.run(critic(**inp.model_dump()))
    assert seen == [inp]


def test_large_ascii_packet_not_compacted_by_byte_bound(critic, monkeypatch):
    seen = []
    install_reply(monkeypatch, seen)
    inp = packet(answer_tex="Complete mathematical argument. " * 35_000)
    check = critic.context_preflight(inp)
    assert check["input_token_upper_bound"] > check["input_token_budget"]
    assert check["fits"]
    asyncio.run(critic(**inp.model_dump()))
    assert seen == [inp]


def test_provider_overflow_retries_locally_once_and_persists_actual_conversation(critic, monkeypatch):
    seen = []

    async def reply(self, inp):
        seen.append(inp)
        self.tracker.add_usd(2)
        if len(seen) == 1:
            raise ValueError("OpenAI response.status=failed: context_length_exceeded")
        return self.parse_output("Fresh finding\n<answer_ready>false</answer_ready>", inp)

    monkeypatch.setattr(APICallAgent, "run", reply)
    inp = packet()
    out = asyncio.run(critic(**inp.model_dump()))
    assert [x.mode for x in seen] == ["stateful", "fresh"]
    assert seen[1].prior_messages == [] and seen[1].omit_author_thinking
    assert out.mode == "fresh" and len(out.messages_after) == 2
    assert "Old draft" not in json.dumps(out.messages_after)
    # A new agent at the same checkpoint must not replay either paid call.
    restored = asyncio.run(ACCritic(critic.ctx)(**inp.model_dump()))
    assert restored == out and len(seen) == 2
    assert critic.ctx.budgets.root().counters.usd == 4
    ledger = json.loads(next((critic.ctx.root_workdir / "critic_context").glob("*.json")).read_text())
    assert ledger["status"] == "completed"


def test_exhausted_recovery_is_terminal_across_resume(critic, monkeypatch):
    seen = []

    async def fail(self, inp):
        seen.append(inp.mode)
        raise ValueError("context_length_exceeded")

    monkeypatch.setattr(APICallAgent, "run", fail)
    for agent in (critic, ACCritic(critic.ctx)):
        with pytest.raises(CriticContextTooLarge, match="do not retry"):
            asyncio.run(agent(**packet().model_dump()))
    assert seen == ["stateful", "fresh"]


def test_fresh_provider_rejection_is_not_retried_unchanged(critic, monkeypatch):
    seen = []

    async def fail(self, inp):
        seen.append(inp.mode)
        raise ValueError("context_length_exceeded")

    monkeypatch.setattr(APICallAgent, "run", fail)
    with pytest.raises(CriticContextTooLarge):
        asyncio.run(critic(**packet(mode="fresh", prior_messages=[]).model_dump()))
    assert seen == ["fresh"]


def test_omitted_thinking_does_not_change_receipt_on_resume(critic, monkeypatch):
    seen = []

    async def fail(self, inp):
        seen.append(inp)
        raise ValueError("context_length_exceeded")

    monkeypatch.setattr(APICallAgent, "run", fail)
    original = packet(mode="fresh", prior_messages=[], omit_author_thinking=True, author_thinking="")
    restored = original.model_copy(update={"author_thinking": "Restored author summary", "n_rounds": 20})
    for inp in (original, restored):
        with pytest.raises(CriticContextTooLarge):
            asyncio.run(critic(**inp.model_dump()))
    assert len(seen) == 1
    assert critic._recovery_checkpoint(original) == critic._recovery_checkpoint(restored)


def test_unrelated_recovery_receipt_does_not_disable_later_review_cache(critic, monkeypatch):
    seen = []
    install_reply(monkeypatch, seen)
    unrelated = critic._recovery_checkpoint(packet(round=1))
    unrelated.parent.mkdir()
    unrelated.write_text(json.dumps({"status": "blocked", "reason": "Previous round"}))
    inp = packet(round=2)
    first = asyncio.run(critic(**inp.model_dump()))
    assert asyncio.run(critic(**inp.model_dump())) == first
    assert len(seen) == 1


def test_cancelled_recovery_cannot_reset_retry_allowance(critic, monkeypatch):
    seen = []

    async def fail(self, inp):
        seen.append(inp.mode)
        if inp.mode == "stateful":
            raise ValueError("context_length_exceeded")
        raise asyncio.CancelledError()

    monkeypatch.setattr(APICallAgent, "run", fail)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(critic(**packet().model_dump()))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ACCritic(critic.ctx)(**packet().model_dump()))
    with pytest.raises(CriticContextTooLarge, match="two attempts"):
        asyncio.run(ACCritic(critic.ctx)(**packet().model_dump()))
    assert seen == ["stateful", "fresh", "fresh"]


@pytest.mark.parametrize("mode", ["fresh", "stateful"])
@pytest.mark.parametrize("error", [asyncio.CancelledError, RuntimeError])
def test_ordinary_review_can_resume_after_two_interruptions(critic, monkeypatch, runner, mode, error):
    seen = []

    async def reply(self, inp):
        seen.append(inp)
        if len(seen) <= 2:
            raise error("synthetic interruption")
        return self.parse_output("<answer_ready>false</answer_ready>", inp)

    monkeypatch.setattr(APICallAgent, "run", reply)
    inp = packet(mode=mode)
    for _ in range(2):
        with pytest.raises(error):
            runner.run(ACCritic(critic.ctx)(**inp.model_dump()))
    out = runner.run(ACCritic(critic.ctx)(**inp.model_dump()))
    assert seen == [inp] * 3 and not out.answer_ready
    saved = json.loads(critic._recovery_checkpoint(inp).read_text())
    assert saved["attempts"] == 3 and saved["kind"] == "ordinary" and saved["status"] == "completed"
    assert runner.run(ACCritic(critic.ctx)(**inp.model_dump())) == out
    assert len(seen) == 3


@pytest.mark.parametrize("mode", ["fresh", "stateful"])
def test_resumed_ordinary_review_keeps_context_error_handling(critic, monkeypatch, runner, mode):
    seen = []

    async def reply(self, inp):
        seen.append(inp)
        if len(seen) == 1:
            raise asyncio.CancelledError()
        if len(seen) == 2:
            raise ValueError("context_length_exceeded")
        return self.parse_output("<answer_ready>false</answer_ready>", inp)

    monkeypatch.setattr(APICallAgent, "run", reply)
    inp = packet(mode=mode)
    with pytest.raises(asyncio.CancelledError):
        runner.run(critic(**inp.model_dump()))
    if mode == "fresh":
        with pytest.raises(CriticContextTooLarge, match="complete fresh"):
            runner.run(ACCritic(critic.ctx)(**inp.model_dump()))
        with pytest.raises(CriticContextTooLarge):
            runner.run(ACCritic(critic.ctx)(**inp.model_dump()))
        assert len(seen) == 2
    else:
        out = runner.run(ACCritic(critic.ctx)(**inp.model_dump()))
        assert [x.mode for x in seen] == ["stateful", "stateful", "fresh"]
        assert not seen[-1].prior_messages and seen[-1].omit_author_thinking
        assert seen[-1].answer_tex == inp.answer_tex and out.mode == "fresh"
        assert runner.run(ACCritic(critic.ctx)(**inp.model_dump())) == out
        assert len(seen) == 3


def test_local_recovery_does_not_cancel_parallel_work(critic, monkeypatch):
    seen = []

    async def exercise():
        computing = asyncio.Event()
        recovered = asyncio.Event()

        async def reply(self, inp):
            if inp.mode == "stateful":
                await computing.wait()
                raise ValueError("context_length_exceeded")
            recovered.set()
            return self.parse_output("<answer_ready>false</answer_ready>", inp)

        async def compute():
            computing.set()
            await recovered.wait()
            seen.append("compute finished")

        monkeypatch.setattr(APICallAgent, "run", reply)
        await asyncio.gather(critic(**packet().model_dump()), compute())

    asyncio.run(exercise())
    assert seen == ["compute finished"]


@pytest.mark.parametrize("error", [RuntimeError("provider unavailable"), asyncio.CancelledError(),
                                   BudgetExhausted("run", "usd", 2, 1)])
def test_unrelated_failures_and_cancellation_propagate(critic, monkeypatch, error):
    seen = []

    async def fail(self, inp):
        seen.append(inp)
        raise error

    monkeypatch.setattr(APICallAgent, "run", fail)
    with pytest.raises(type(error)):
        asyncio.run(critic(**packet().model_dump()))
    assert len(seen) == 1


def test_output_and_tool_reserves_honor_configuration(critic):
    critic.ctx.component_configs["ACCritic"] = {
        "context_window_tokens": 300_000, "context_tool_reserve_tokens": 32_000,
    }
    critic.ctx.model_overrides["ACCritic"] = {"base": "models/openai/gpt-6-astra-pro", "max_tokens": 64_000}
    assert critic.context_preflight(packet())["input_token_budget"] == 204_000


@pytest.mark.parametrize("window", [None, "invalid", 0, -1, 128_000])
def test_invalid_context_configuration_fails_closed(critic, window):
    critic.ctx.component_configs["ACCritic"] = {"context_window_tokens": window}
    with pytest.raises(CriticContextTooLarge, match="Invalid"):
        asyncio.run(critic(**packet().model_dump()))


@pytest.mark.parametrize("visual", [False, True])
@pytest.mark.parametrize("recover", [False, True])
def test_workflow_context_recovery_checkpoint_and_nonretryable_failure(tmp_path, monkeypatch, visual, recover, runner):
    from proofstack.agents.ac.ac_workflow import ACWorkflow, _CompileResult
    from proofstack.agents.ac.author import Author
    from proofstack.registry import load_preset

    preset = load_preset("author_critic")
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
                            component_configs=preset.component_configs,
                            api_client_factory=lambda *a, **k: pytest.fail("provider call"))
    ctx.component_configs.setdefault("ACCritic", {})["research_notes_transport"] = "inline"
    workflow = preset.workflow_cls(ctx) if visual else ACWorkflow(ctx)
    authors, reviews = [], []

    async def author(self, inp):
        authors.append(inp.round)
        return self.Outputs(answer_tex=f"Manuscript round {inp.round}", research_notes_tex="notes",
                            thinking_summary="summary", ready=False)

    async def review(self, inp):
        reviews.append((inp.round, inp.mode))
        self.tracker.add_usd(1)
        if inp.round == 1 and (inp.mode == "stateful" or not recover):
            raise ValueError("context_length_exceeded")
        return self.parse_output("Still a gap\n<answer_ready>false</answer_ready>", inp)

    def compile_document(tex, **kwargs):
        return _CompileResult(tex=tex + "\n% normalized", tex_path=None, pdf_path=None, compiled=True, pages=1)

    monkeypatch.setattr(Author, "run", author)
    monkeypatch.setattr(APICallAgent, "run", review)
    monkeypatch.setattr("proofstack.agents.ac.ac_workflow._simple_compile_latex", compile_document)
    monkeypatch.setattr("proofstack.agents.ac.visual_blocks._simple_compile_latex", compile_document)
    inputs = dict(problem="Synthetic question", problem_id="p", n_rounds=1, stop_after_review_round=True,
                  enable_council=False, enable_compute=False, enable_final_critic=False)
    out = runner.run(workflow(**inputs))
    assert authors == [0, 1]
    assert reviews == [(0, "fresh"), (1, "stateful"), (1, "fresh")]
    checkpoint = next(tmp_path.glob("ac_workspaces/*/.ac/resume-state.json"))
    saved = json.loads(checkpoint.read_text())
    if not recover:
        assert not out.error_retryable and out.last_gasp
        assert not out.last_critic_accepted
        assert saved["awaiting_review_round"] == 1
        assert "Manuscript round 1" in (checkpoint.parent.parent / "answer.tex").read_text()
        resumed = preset.workflow_cls(ctx) if visual else ACWorkflow(ctx)
        resumed_out = runner.run(resumed(**{**inputs, "n_rounds": 2, "resume_run": True}))
        assert not resumed_out.error_retryable
        assert authors == [0, 1]
        assert reviews == [(0, "fresh"), (1, "stateful"), (1, "fresh")]
        return
    assert not out.error and not out.last_gasp
    assert saved["critic_instance_turn"] == 1
    assert saved["next_round"] == 2 and saved["awaiting_review_round"] is None
    assert len(saved["critic_conversation"]) == 2
    assert "Manuscript round 0" not in json.dumps(saved["critic_conversation"])
    # Actual workflow resume must continue at the next Author, with the new
    # critic conversation, without replaying any old Author or review call.
    resumed = preset.workflow_cls(ctx) if visual else ACWorkflow(ctx)
    resumed_out = runner.run(resumed(**{**inputs, "n_rounds": 2, "resume_run": True}))
    assert not resumed_out.error
    assert authors == [0, 1, 2]
    assert reviews[-1] == (2, "stateful")


def test_recovery_respects_failed_attempt_spend_and_remaining_deadline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from proofstack.budget import BudgetSpec

    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
                            run_budget=BudgetSpec(max_usd=1, max_wallclock_s=60),
                            component_configs={"ACCritic": {"research_notes_transport": "inline"}})
    critic = ACCritic(ctx)
    critic._client = SimpleNamespace(model="fake")
    calls = []

    async def query(self, client, messages, query, **kwargs):
        calls.append(messages)
        self.tracker.add_usd(1)
        raise ValueError("context_length_exceeded")

    monkeypatch.setattr(APICallAgent, "_query", query)
    with pytest.raises(BudgetExhausted):
        asyncio.run(critic(**packet().model_dump()))
    assert len(calls) == 1
    assert ctx.budgets.root().counters.usd == 1
    assert ctx.budgets.root().spec.max_wallclock_s == 60


def test_interrupted_recovery_resumes_fresh_without_replaying_history(critic, monkeypatch, runner):
    seen = []

    async def reply(self, inp):
        seen.append(inp.mode)
        if len(seen) == 1:
            raise ValueError("context_length_exceeded")
        if len(seen) == 2:
            raise RuntimeError("temporary provider outage")
        return self.parse_output("<answer_ready>false</answer_ready>", inp)

    monkeypatch.setattr(APICallAgent, "run", reply)
    with pytest.raises(RuntimeError, match="outage"):
        runner.run(critic(**packet().model_dump()))
    out = runner.run(ACCritic(critic.ctx)(**packet().model_dump()))
    assert out.mode == "fresh"
    assert seen == ["stateful", "fresh", "fresh"]
    assert runner.run(ACCritic(critic.ctx)(**packet().model_dump())) == out
    assert len(seen) == 3


@pytest.mark.parametrize("durable", [False, True])
def test_reconciles_completed_recovery_after_cancellation(critic, monkeypatch, runner, durable):
    seen = []

    async def reply(self, inp):
        seen.append(inp.mode)
        if inp.mode == "stateful":
            raise ValueError("context_length_exceeded")
        raw = "Independent rejection\n<answer_ready>false</answer_ready>"
        if durable:
            row = {"attempt_id": "fresh:1:0", "invocation_id": "fresh", "status": "completed"}
            (self.workdir / "provider-attempts.jsonl").write_text(json.dumps(row) + "\n")
            (self.workdir / "provider-completed-fresh.json").write_text(json.dumps({"report": raw}))
        else:
            self._on_response(raw, inp)
        raise asyncio.CancelledError()

    monkeypatch.setattr(APICallAgent, "run", reply)
    with pytest.raises(asyncio.CancelledError):
        runner.run(critic(**packet().model_dump()))
    out = runner.run(ACCritic(critic.ctx)(**packet().model_dump()))
    assert not out.answer_ready and out.review_md == "Independent rejection"
    assert seen == ["stateful", "fresh"]


def test_unresolved_recovery_does_not_start_duplicate_call(critic, monkeypatch, runner):
    seen = []

    async def reply(self, inp):
        seen.append(inp.mode)
        if inp.mode == "stateful":
            raise ValueError("context_length_exceeded")
        row = {"attempt_id": "fresh:1:0", "invocation_id": "fresh", "status": "in_progress",
               "outcome": "response", "reconciliation_pending": True}
        (self.workdir / "provider-attempts.jsonl").write_text(json.dumps(row) + "\n")
        raise asyncio.CancelledError()

    monkeypatch.setattr(APICallAgent, "run", reply)
    with pytest.raises(asyncio.CancelledError):
        runner.run(critic(**packet().model_dump()))
    with pytest.raises(RuntimeError, match="unresolved"):
        runner.run(ACCritic(critic.ctx)(**packet().model_dump()))
    assert seen == ["stateful", "fresh"]


@pytest.mark.parametrize("status", [None, "queued", "in_progress"])
@pytest.mark.parametrize("outcome", ["error", "failed", "cancelled"])
def test_poll_error_with_response_id_cannot_authorize_fresh_retry(critic, monkeypatch, runner, status, outcome):
    seen = []

    async def reply(self, inp):
        seen.append(inp.mode)
        if len(seen) == 1:
            raise ValueError("context_length_exceeded")
        if len(seen) == 2:
            row = {"attempt_id": "fresh:1:0", "invocation_id": "fresh", "response_id": "synthetic-response",
                   "status": status, "outcome": outcome}
            (self.workdir / "provider-attempts.jsonl").write_text(json.dumps(row) + "\n")
            raise RuntimeError("poll connection failed before supervisor restart")
        return self.parse_output("<answer_ready>false</answer_ready>", inp)

    monkeypatch.setattr(APICallAgent, "run", reply)
    with pytest.raises(RuntimeError, match="poll connection"):
        runner.run(critic(**packet().model_dump()))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="unresolved provider work"):
            runner.run(ACCritic(critic.ctx)(**packet().model_dump()))
    assert seen == ["stateful", "fresh"]


@pytest.mark.parametrize("status,response_id", [
    ("failed", "synthetic-response"), ("cancelled", "synthetic-response"),
    ("incomplete", "synthetic-response"), ("completed", "synthetic-response"),
    (None, None),
])
def test_terminal_response_or_failed_create_allows_bounded_recovery(critic, monkeypatch, runner, status, response_id):
    seen = []

    async def reply(self, inp):
        seen.append(inp.mode)
        if len(seen) == 1:
            raise ValueError("context_length_exceeded")
        if len(seen) == 2:
            row = {"attempt_id": "fresh:1:0", "invocation_id": "fresh", "response_id": response_id,
                   "status": status, "outcome": "error"}
            (self.workdir / "provider-attempts.jsonl").write_text(json.dumps(row) + "\n")
            raise RuntimeError("settled provider failure")
        return self.parse_output("<answer_ready>false</answer_ready>", inp)

    monkeypatch.setattr(APICallAgent, "run", reply)
    with pytest.raises(RuntimeError, match="settled provider"):
        runner.run(critic(**packet().model_dump()))
    out = runner.run(ACCritic(critic.ctx)(**packet().model_dump()))
    assert out.mode == "fresh" and not out.answer_ready
    assert runner.run(ACCritic(critic.ctx)(**packet().model_dump())) == out
    assert seen == ["stateful", "fresh", "fresh"]


@pytest.mark.parametrize("visual", [False, True])
@pytest.mark.parametrize("preflight", [False, True])
def test_ready_recovered_fresh_review_needs_no_extra_critic(tmp_path, monkeypatch, runner, visual, preflight):
    from proofstack.agents.ac.ac_workflow import ACWorkflow, _CompileResult
    from proofstack.agents.ac.author import Author
    from proofstack.registry import load_preset

    preset = load_preset("author_critic")
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, component_configs=preset.component_configs)
    seen = []

    async def author(self, inp):
        return self.Outputs(answer_tex="Complete synthetic proof", ready=inp.round > 0)

    async def review(self, inp):
        seen.append((inp.round, inp.mode))
        if inp.round == 1 and inp.mode == "stateful":
            raise ValueError("context_length_exceeded")
        ready = "true" if inp.round == 1 else "false"
        return self.parse_output(f"<answer_ready>{ready}</answer_ready>", inp)

    def check(self, inp):
        return {"fits": inp.round == 0 or inp.mode == "fresh"}

    def compile_document(tex, **kwargs):
        return _CompileResult(tex=tex, tex_path=None, pdf_path=None, compiled=True, pages=1)

    monkeypatch.setattr(Author, "run", author)
    monkeypatch.setattr(APICallAgent, "run", review)
    monkeypatch.setattr("proofstack.agents.ac.ac_workflow._simple_compile_latex", compile_document)
    monkeypatch.setattr("proofstack.agents.ac.visual_blocks._simple_compile_latex", compile_document)
    if preflight:
        monkeypatch.setattr(ACCritic, "context_preflight", check)
    workflow = preset.workflow_cls(ctx) if visual else ACWorkflow(ctx)
    out = runner.run(workflow(problem="Synthetic question", problem_id="p", n_rounds=3,
                              enable_council=False, enable_compute=False, enable_final_critic=False))
    assert out.early_stopped and out.last_critic_accepted
    assert seen == ([(0, "fresh"), (1, "fresh")] if preflight else
                    [(0, "fresh"), (1, "stateful"), (1, "fresh")])


def test_join_never_ignores_forced_fresh_rejection(tmp_path, runner):
    from proofstack.agents.ac.author import Author
    from proofstack.agents.ac.visual_blocks import ACReviewJoinBlock

    ctx = RunContext.create(root_workdir=tmp_path, flat=True)
    accepted = ACCritic.Outputs(mode="fresh", answer_ready=True)
    rejection = ACCritic.Outputs(mode="fresh", answer_ready=False, review_md="Fatal gap")
    state = {"workspace": str(tmp_path), "current_author": Author.Outputs(ready=True).model_dump(),
             "round_review": accepted.model_dump(), "forced_fresh_review": rejection.model_dump()}
    out = runner.run(ACReviewJoinBlock(ctx)(base_state=state))
    assert not out.ready_for_gate
    assert "Fatal gap" in out.state["pending_critique"]


@pytest.mark.parametrize("visual", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_pending_last_round_mode_survives_extended_bound(tmp_path, monkeypatch, runner, visual, legacy):
    from proofstack.agents.ac.ac_workflow import ACWorkflow, _CompileResult
    from proofstack.agents.ac.author import Author
    from proofstack.registry import load_preset

    preset = load_preset("author_critic")
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, component_configs=preset.component_configs,
                            api_client_factory=lambda *a, **k: pytest.fail("provider call"))
    ctx.component_configs.setdefault("ACCritic", {})["research_notes_transport"] = "inline"
    calls, authors = [], []

    async def author(self, inp):
        authors.append(inp.round)
        return self.Outputs(answer_tex=f"Manuscript {inp.round}", research_notes_tex="notes", ready=False)

    async def review(self, inp):
        calls.append((inp.round, inp.mode))
        if inp.round == 1:
            raise ValueError("context_length_exceeded")
        return self.parse_output("Gap\n<answer_ready>false</answer_ready>", inp)

    def compile_document(tex, **kw):
        return _CompileResult(tex=tex, tex_path=None, pdf_path=None, compiled=True, pages=1)

    monkeypatch.setattr(Author, "run", author)
    monkeypatch.setattr(APICallAgent, "run", review)
    monkeypatch.setattr("proofstack.agents.ac.ac_workflow._simple_compile_latex", compile_document)
    monkeypatch.setattr("proofstack.agents.ac.visual_blocks._simple_compile_latex", compile_document)
    cls = preset.workflow_cls if visual else ACWorkflow
    inp = dict(problem="Synthetic question", problem_id="p", n_rounds=1,
               enable_council=False, enable_compute=False, enable_final_critic=False)
    first = runner.run(cls(ctx)(**inp))
    assert first.error and calls == [(0, "fresh"), (1, "fresh")]
    checkpoint = next(tmp_path.glob("ac_workspaces/*/.ac/resume-state.json"))
    state = json.loads(checkpoint.read_text())
    assert state["awaiting_review_context"]["mode"] == "fresh"
    if legacy:
        state.pop("awaiting_review_context")
        checkpoint.write_text(json.dumps(state))
    second = runner.run(cls(ctx)(**{**inp, "resume_run": True, "n_rounds": 2}))
    assert second.error and not second.error_retryable
    assert authors == [0, 1] and calls == [(0, "fresh"), (1, "fresh")]
