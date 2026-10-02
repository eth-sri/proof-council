from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import time
from collections import Counter
from types import SimpleNamespace

import pytest

from app.dev_data import validate_preset_yaml
from proofstack.agents.ac.ac_workflow import ACWorkflow, ACDAGWorkflow, _problem_hash
from proofstack.agents.batch3_critic import Batch3CleanupCritic
from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow
from proofstack.agents.writeup_loop import RewriteSeat, RepairSeat, _assemble
from proofstack.budget import BudgetExhausted, BudgetSpec, SubscriptionParked
from proofstack.context import RunContext
from proofstack.latex_contract import normalize_firstproof_latex
from proofstack.registry import load_preset


TEX = normalize_firstproof_latex("\\documentclass[12pt]{article}\n\\begin{document}\n\nProof.\n\\end{document}")
ORIGINAL = "Prove the ORIGINAL problem"
HISTORY = [{"role": "user", "content": ORIGINAL}, {"role": "assistant", "content": "Research review"}]


def _round_snapshot(workspace, round, document, *, accepted=True, forced=None, parse_failed=False, bibliography=""):
    snapshot = workspace / ".ac" / f"round-{round}"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "answer.tex").write_text(document)
    (snapshot / "references.bib").write_text(bibliography)
    (snapshot / "review_outputs.json").write_text(json.dumps({
        "answer_ready": accepted, "parse_failed": parse_failed, "review_md": f"Round {round} stateful review",
    }))
    if forced is not None:
        (snapshot / "forced_fresh_review_outputs.json").write_text(json.dumps({
            "answer_ready": forced, "parse_failed": parse_failed, "review_md": f"Round {round} fresh review",
        }))
    return snapshot


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    calls, seen, options = Counter(), {}, {}
    preset = load_preset("firstproof_batch3")

    def no_network(*args, **kwargs):
        raise AssertionError("offline test attempted a provider call")

    ctx = RunContext.create(
        run_id="batch3-test", root_workdir=tmp_path / "workflow_runs" / "batch3-test", flat=True,
        run_budget=BudgetSpec(max_usd=100, max_wallclock_s=1000),
        component_configs=preset.component_configs, api_client_factory=no_network,
    )
    workspace = ctx.root_workdir / "ac_workspaces" / f"p-{_problem_hash(ORIGINAL)}"
    checkpoint = workspace / ".ac/resume-state.json"

    async def research(self, inp):
        calls["research"] += 1
        seen.setdefault("research_inputs", []).append(inp)
        seen.setdefault("research_caps", []).append(self.ctx.budgets.root().spec.max_usd)
        if inp.resume_run:
            seen.setdefault("resumed_states", []).append(json.loads(checkpoint.read_text()))
        self.tracker.add_usd(20)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        round_no = calls["research"] * 3
        checkpoint.write_text(json.dumps({
            "problem_hash": _problem_hash(ORIGINAL),
            "critic_conversation": HISTORY, "review_history": [{"review_md": "Research findings"}],
            "early_stopped": options.get("agreed", True), "next_round": round_no + 1,
            "last_round_run": round_no, "terminal_outputs": {"early_stopped": True},
            **options.get("checkpoint", {}),
        }))
        answer = workspace / "answer.tex"
        answer.write_text(options.get("research_document", TEX))
        _round_snapshot(workspace, round_no, options.get("reviewed_document", answer.read_text()),
                        accepted=options.get("research_accepted", True), forced=options.get("forced_accepted"),
                        bibliography=options.get("bibliography", ""))
        bib = workspace / "references.bib"
        bib.write_text(options.get("bibliography", ""))
        notes = workspace / "research_notes.tex"
        notes.write_text("Private scratch work")
        if options.get("research_sleep"):
            try:
                options["research_started"].set()
                await asyncio.Event().wait()
            finally:
                calls["research_cancelled"] += 1
        if "research_error" in options:
            raise options["research_error"]
        return self.Outputs(
            problem_id=inp.problem_id, answer_tex=answer, references_bib=bib,
            research_notes_tex=notes, compiled=True, pages=1, rounds_completed=round_no,
            early_stopped=options.get("agreed", True), last_critic_accepted=options.get("research_accepted", True),
        )

    async def edit(self, inp):
        kind = "repair" if isinstance(self, RepairSeat) else "rewrite"
        if "UNSOLVED attempt" in inp.prompt:
            kind = "partial"
        calls[kind] += 1
        seen.setdefault("prompts", []).append(inp.prompt)
        self.tracker.add_usd(5)
        if options.get(f"{kind}_error"):
            raise options[f"{kind}_error"]
        revised = TEX.replace("Proof.", f"Proof. Revision {kind} {calls[kind]}.")
        return self.Outputs(text=options.get("edit_document", revised))

    async def compile_document(self, document, *, deadline):
        calls["compile"] += 1
        seen["compiled_document"] = document
        return options.get("compile_ok", True), options.get("pages", 1), ""

    async def critic(self, inp):
        calls["review"] += 1
        seen.setdefault("review_inputs", []).append(inp)
        self.tracker.add_usd(2)
        verdicts = options.get("verdicts", ["accept"])
        verdict = verdicts[min(calls["review"] - 1, len(verdicts) - 1)]
        out = self.parse_output(f"Review {calls['review']}\n<cleanup_verdict>{verdict}</cleanup_verdict>", inp)
        if options.get("review_error"):
            self.completed_review = out
            raise options["review_error"]
        return out

    monkeypatch.setattr(ACDAGWorkflow, "run", research)
    monkeypatch.setattr(RewriteSeat, "run", edit)
    monkeypatch.setattr(RepairSeat, "run", edit)
    monkeypatch.setattr(FirstProofBatch3Workflow, "_compile", compile_document)
    monkeypatch.setattr(Batch3CleanupCritic, "run", critic)
    agent = preset.workflow_cls(ctx)

    async def execute(**overrides):
        # These legacy editor-path tests mock RewriteSeat/RepairSeat explicitly.
        # Their short clocks cannot fit the competition's partial window.
        overrides = {"cleanup_backend": "api", "partial_cleanup_seconds": 0, **overrides}
        return await agent(**preset.build_inputs(problem=ORIGINAL, problem_id="p", cli_overrides=overrides))

    return execute, calls, seen, options, ctx, agent


@pytest.fixture
def workflow_clock(monkeypatch):
    clock = SimpleNamespace(now=time.monotonic())
    monkeypatch.setattr("proofstack.agents.firstproof_batch3.time",
                        SimpleNamespace(monotonic=lambda: clock.now, time=time.time))
    return clock


@pytest.mark.parametrize("pending_round", [None, 0, 4])
def test_partial_rewrite_identifies_reviews_of_an_older_draft(pipeline, pending_round):
    execute, calls, seen, options, _, _ = pipeline
    options.update(research_error=BudgetExhausted("run", "usd", 85, 86), checkpoint={
        "awaiting_review_round": pending_round,
        "awaiting_author": {"answer_tex": TEX} if pending_round is not None else None,
    })
    out = asyncio.run(execute())
    assert out.partial_ready and not out.submission_approved
    assert calls["partial"] == 1 and calls["review"] == 0
    prompt = seen["prompts"][-1]
    assert "Research findings" in prompt
    assert ("refer to an earlier revision, not this draft" in prompt) == (pending_round is not None)
    if pending_round is not None:
        assert "do not reintroduce obsolete caveats" in prompt
        assert "assume claimed repairs are verified" in prompt


def test_preset_and_pro_models():
    preset = load_preset("firstproof_batch3")
    report = validate_preset_yaml(preset.source_path.read_text())
    assert report["ok"], report
    assert preset.inputs["page_limit"] == 16
    for seat in ("Author", "ACCritic", "RewriteSeat", "Batch3CleanupCritic", "RepairSeat"):
        assert preset.component_configs[seat]["model"] == "models/openai/gpt-6-astra-pro"
    assert preset.inputs["research_seconds"] == 23 * 3600
    assert preset.inputs["partial_cleanup_seconds"] == 2 * 60 * 60
    assert FirstProofBatch3Workflow.Inputs(problem=ORIGINAL, problem_id="p").partial_cleanup_seconds == 7200
    assert preset.inputs["compute_workspace_hard_limit_bytes"] == 8 * 1024**3


def test_persistent_cleanup_is_default_and_api_remains_an_explicit_override():
    assert FirstProofBatch3Workflow.Inputs(problem=ORIGINAL, problem_id="p").cleanup_backend == "claude_code"
    for name in ("firstproof_batch3", "firstproof_batch3_multiauthor"):
        preset = load_preset(name)
        assert preset.inputs["cleanup_backend"] == "claude_code"
        default = preset.workflow_cls.Inputs(**preset.build_inputs(problem=ORIGINAL, problem_id="p"))
        assert default.cleanup_backend == "claude_code"
        overridden = preset.workflow_cls.Inputs(**preset.build_inputs(
            problem=ORIGINAL, problem_id="p", cli_overrides={"cleanup_backend": "api"}))
        assert overridden.cleanup_backend == "api"


def test_batch3_presets_keep_the_same_64gb_compute_policy():
    baseline = load_preset("firstproof_batch3")
    multi = load_preset("firstproof_batch3_multiauthor")
    expected = {
        "compute_sandbox_backend": "subprocess",
        "compute_memory_gb": 8,
        "compute_max_parallel_workers": 4,
        "compute_memory_reserve_gb": 16,
    }
    for preset in (baseline, multi):
        actual = preset.workflow_cls.Inputs(**preset.build_inputs(problem=ORIGINAL, problem_id="p"))
        assert {key: getattr(actual, key) for key in expected} == expected
    assert {k: v for k, v in baseline.inputs.items() if k.startswith("compute_")} == {
        k: v for k, v in multi.inputs.items() if k.startswith("compute_")
    }


def test_batch3_presets_differ_only_in_author_parallelism_and_description():
    baseline = load_preset("firstproof_batch3")
    multi = load_preset("firstproof_batch3_multiauthor")
    assert baseline.inputs["author_parallelism"] == 1
    assert multi.inputs["author_parallelism"] == 4
    baseline_raw, multi_raw = baseline.raw.copy(), multi.raw.copy()
    baseline_raw.pop("description")
    multi_raw.pop("description")
    baseline_raw["inputs"] = dict(baseline_raw["inputs"], author_parallelism=4)
    assert baseline_raw == multi_raw
    delegation = baseline.component_configs["Author"]["delegation"]
    assert not {"enabled", "max_threads", "max_tasks_per_wave"} & delegation.keys()
    assert delegation["asynchronous"] is True
    assert delegation["max_tasks_per_turn"] is None
    assert delegation["helper_timeout_s"] is None


def test_ten_cleanup_editors_fit_beside_compute_and_survive_its_pressure(pipeline, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from proofstack.agents.cleanup_session import CleanupSession, CleanupSettings
    from proofstack.sandbox import memory
    from proofstack.sandbox.memory import GiB, MemoryLease, MemoryPolicy

    settings = CleanupSettings()
    assert settings.max_parallel_editors >= 10 and settings.max_parallel_codex_reviews >= 10
    editors = CleanupSession(pipeline[4])._sandbox(settings, 60, "ANTHROPIC_API_KEY")["memory_policy"]
    compute = MemoryPolicy(tmp_path / "compute", 4, 8 * GiB, 16 * GiB)
    available = [16 * GiB]
    monkeypatch.setattr(memory, "available_memory", lambda: available[0])
    monkeypatch.setattr(memory, "markers_rss", lambda markers: {
        m["token"]: (7 * GiB if m["token"] == "compute" else int(.36 * GiB), True) for m in markers})
    leases = [MemoryLease(editors, SimpleNamespace(token=f"editor-{i}", created_at=time.time())) for i in range(10)]
    worker = MemoryLease(compute, SimpleNamespace(token="compute", created_at=time.time()))
    try:
        # Compute keeps about its 16 GB reserve free; ten editors are still admitted.
        assert all(lease.try_acquire() for lease in leases)
        available[0] = 30 * GiB
        assert worker.try_acquire()
        available[0] = 10 * GiB
        assert worker.sample()["reason"] == "shared_memory_pressure"
        assert all(lease.sample()["reason"] is None for lease in leases)
        available[0] = 3 * GiB
        assert any(lease.sample()["reason"] == "shared_memory_pressure" for lease in leases)
    finally:
        for lease in (*leases, worker):
            lease.close()


def test_research_and_candidate_cleanup_leave_the_partial_window(pipeline, monkeypatch):
    execute, _, _, _, _, agent = pipeline
    phases = {}
    phase = agent._phase

    async def observed(name, **kwargs):
        phases.setdefault(name, kwargs["deadline"])
        return await phase(name, **kwargs)

    monkeypatch.setattr(agent, "_phase", observed)
    start = time.monotonic()
    out = asyncio.run(execute(max_wallclock_s=600, partial_cleanup_seconds=400))
    assert out.submission_approved
    assert 195 < phases["research"] - start < 201
    assert 195 < phases["candidate_cleanup"] - start < 201


def test_editorial_failures_do_not_use_research_recoveries(pipeline):
    execute, calls, _, options, _, _ = pipeline
    options["verdicts"] = ["repair", "accept"]
    out = asyncio.run(execute(max_cleanup_repairs=0, max_recovery_attempts=0))
    assert out.submission_approved and calls["research"] == 2


def test_editorial_handoffs_are_bounded(pipeline):
    execute, calls, _, options, _, _ = pipeline
    options["verdicts"] = ["repair"]
    out = asyncio.run(execute(max_cleanup_repairs=0, max_cleanup_handoffs=2, max_recovery_attempts=0))
    assert out.submission_approved and not out.partial_ready
    assert out.answer_tex.read_text() == TEX
    assert calls["research"] == calls["rewrite"] == 3 and calls["partial"] == 0


@pytest.mark.parametrize("verdict", ["repair", "restore", "research"])
def test_accepted_fallback_survives_editorial_handoff_not_mathematical_rejection(pipeline, monkeypatch, verdict):
    execute, calls, _, options, _, _ = pipeline
    options["verdicts"] = [verdict]
    research = ACDAGWorkflow.run

    async def later_draft(self, inp):
        if calls["research"]:
            options.update(agreed=False, research_document=TEX.replace("Proof.", "Unaccepted new argument."))
        return await research(self, inp)

    monkeypatch.setattr(ACDAGWorkflow, "run", later_draft)
    out = asyncio.run(execute(max_cleanup_repairs=0))
    assert calls["research"] > 1 and calls["review"] == 1
    if verdict == "research":
        assert not out.submission_approved and out.partial_ready
    else:
        assert out.submission_approved and not out.partial_ready
        assert out.answer_tex.read_text() == TEX
        assert out.final_critic_review_md == "Research findings"
        assert calls["partial"] == 0


@pytest.mark.parametrize("verdict", ["repair", "restore", "research"])
def test_accepted_fallback_survives_a_new_workflow_instance(pipeline, monkeypatch, verdict):
    execute, calls, _, options, ctx, agent = pipeline
    options["verdicts"] = [verdict]
    research = ACDAGWorkflow.run

    async def interrupted_research(self, inp):
        if calls["research"]:
            raise asyncio.CancelledError
        return await research(self, inp)

    monkeypatch.setattr(ACDAGWorkflow, "run", interrupted_research)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute(max_cleanup_repairs=0))
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    state = agent._research_state(inp)
    assert not state["early_stopped"]
    assert (agent._restore_accepted(inp) is not None) == (verdict != "research")

    before = calls.copy()
    resumed = type(agent)(ctx)
    out = asyncio.run(resumed(
        problem=ORIGINAL, problem_id="p", resume_run=True, research_seconds=0,
        partial_cleanup_seconds=0, cleanup_backend="api",
    ))
    assert calls["research"] == before["research"] and calls["review"] == before["review"]
    if verdict == "research":
        assert out.partial_ready and not out.submission_approved
    else:
        assert out.submission_approved and out.answer_tex.read_text() == TEX
        assert calls["partial"] == 0


@pytest.mark.parametrize("mutation", ["document", "problem", "id", "json", "normalization"])
def test_retained_acceptance_validates_content_and_problem(pipeline, mutation):
    _, _, _, _, _, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    _round_snapshot(agent._workspace(inp), 0, TEX)
    agent._remember_accepted(inp, TEX, "Accepted proof")
    path = agent._accepted_checkpoint()
    record = json.loads(path.read_text())
    if mutation == "document":
        record["document"] = TEX.replace("Proof.", "Unreviewed edits.")
    elif mutation == "problem":
        record["problem_hash"] = _problem_hash("Another problem")
    elif mutation == "id":
        record["problem_id"] = "another-id"
    elif mutation == "normalization":
        record["document"] = r"\documentclass{book}\begin{document}Proof.\end{document}"
        record["sha256"] = hashlib.sha256(record["document"].encode()).hexdigest()
    path.write_text("not json" if mutation == "json" else json.dumps(record))
    assert agent._restore_accepted(inp) is None


@pytest.mark.parametrize("changed_export", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_revoked_terminal_acceptance_cannot_be_reused_after_interrupted_handoff(pipeline, monkeypatch, changed_export, legacy):
    execute, calls, _, options, ctx, agent = pipeline
    options["verdicts"] = ["research"]
    revoke = agent._revoke_accepted

    def revoke_then_interrupt(*args):
        revoke(*args)
        raise asyncio.CancelledError

    monkeypatch.setattr(agent, "_revoke_accepted", revoke_then_interrupt)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute())
    resumed = type(agent)(ctx)
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    assert resumed._research_state(inp)["early_stopped"]
    assert resumed._restore_accepted(inp) is None
    if legacy:
        record = json.loads(agent._accepted_checkpoint().read_text())
        record.pop("source_bib_sha256")
        agent._accepted_checkpoint().write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="revoked by cleanup"):
        export = TEX.replace("Proof.", "Proof. New bibliography formatting.") if changed_export else TEX
        resumed._remember_accepted(inp, export, "Stale cached review")
    assert calls["review"] == 1


def test_new_acceptance_replaces_retained_baseline(pipeline):
    _, _, _, _, ctx, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    _round_snapshot(agent._workspace(inp), 0, TEX)
    agent._remember_accepted(inp, TEX, "First acceptance")
    revised = TEX.replace("Proof.", "A better proof.")
    _round_snapshot(agent._workspace(inp), 0, revised)
    agent._remember_accepted(inp, revised, "Second acceptance")
    resumed = type(agent)(ctx)
    baseline = resumed._restore_accepted(inp)
    assert baseline.document == revised and baseline.review_md == "Second acceptance"


def test_fresh_run_does_not_restore_retained_acceptance(pipeline):
    execute, calls, _, _, _, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    _round_snapshot(agent._workspace(inp), 0, TEX)
    agent._remember_accepted(inp, TEX, "Earlier run")
    out = asyncio.run(execute(research_seconds=0))
    assert not out.submission_approved and not calls
    assert agent._restore_accepted(inp) is None


@pytest.mark.parametrize("changed", [False, True])
def test_later_research_rejection_only_revokes_the_exact_baseline(pipeline, monkeypatch, changed):
    execute, calls, _, options, _, agent = pipeline
    options["verdicts"] = ["repair"]
    research = ACDAGWorkflow.run

    async def rejecting_research(self, inp):
        later = calls["research"] > 0
        if later:
            options["agreed"] = False
            options["research_accepted"] = False
            if changed:
                options["research_document"] = TEX.replace("Proof.", "Different rejected argument.")
        out = await research(self, inp)
        if later:
            out.last_critic_accepted = False
        return out

    monkeypatch.setattr(ACDAGWorkflow, "run", rejecting_research)
    out = asyncio.run(execute(max_cleanup_repairs=0))
    assert out.submission_approved == changed
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    assert (agent._restore_accepted(inp) is not None) == changed
    if changed:
        assert out.answer_tex.read_text() == TEX


@pytest.mark.parametrize("critic", ["stateful", "fresh"])
@pytest.mark.parametrize("ending", ["normal", "last_gasp"])
@pytest.mark.parametrize("same_export", [False, True])
def test_export_equivalent_source_rejection(pipeline, monkeypatch, critic, ending, same_export):
    execute, calls, _, options, _, agent = pipeline
    raw = TEX.replace("Proof.", r"Proof.\bibliography{references}")
    bib = "@book{reference, title={Verified reference}, year={2020}}"
    bbl = r"\begin{thebibliography}{1}\bibitem{reference} Verified reference.\end{thebibliography}"
    options.update(verdicts=["repair"], research_document=raw, bibliography=bib)
    research = ACDAGWorkflow.run

    def compile_bibliography(document, bibliography, *, bbl_output, **kwargs):
        assert bibliography == bib
        bbl_output.write_text(bbl)

    monkeypatch.setattr("proofstack.agents.firstproof_batch3._compile_raw", compile_bibliography)

    async def rejecting_research(self, inp):
        later = calls["research"] > 0
        if later:
            changed = raw.replace(r"\bibliography{", r"\bibliography {")
            if not same_export:
                changed = changed.replace("Proof.", "A different argument.")
            options.update(agreed=False, research_document=changed, research_accepted=critic == "fresh",
                           forced_accepted=False if critic == "fresh" else None)
        result = await research(self, inp)
        result.last_gasp = later and ending == "last_gasp"
        return result

    monkeypatch.setattr(ACDAGWorkflow, "run", rejecting_research)
    out = asyncio.run(execute(max_cleanup_repairs=0))
    revoked = same_export and ending == "normal"
    assert out.submission_approved == (not revoked)
    if revoked:
        record = json.loads(agent._accepted_checkpoint().read_text())
        assert record["revoked"] and record["round"] == 6
        assert record["review_md"] == f"Round 6 {'fresh' if critic == 'fresh' else 'stateful'} review"
    else:
        assert out.answer_tex.read_text() == raw.replace(r"\bibliography{references}", bbl)


@pytest.mark.parametrize("critic", ["stateful", "fresh"])
@pytest.mark.parametrize("legacy", ["none", "tex_only", "no_hashes"])
def test_changed_bibliography_does_not_revoke_old_source(pipeline, critic, legacy):
    _, _, _, _, ctx, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    workspace = agent._workspace(inp)
    raw = TEX.replace("Proof.", r"Proof.\bibliography{references}")
    bib = "@book{reference, title={Correct reference}}"
    _round_snapshot(workspace, 0, raw, bibliography=bib)
    agent._remember_accepted(inp, TEX, "Accepted proof with correct references")
    if legacy != "none":
        record = json.loads(agent._accepted_checkpoint().read_text())
        record.pop("source_bib_sha256")
        if legacy == "no_hashes":
            record.pop("source_sha256")
        agent._accepted_checkpoint().write_text(json.dumps(record))
    (workspace / "references.bib").write_text("Live references have changed too")
    resumed = type(agent)(ctx)
    resumed._accepted_baseline = resumed._restore_accepted(inp)
    assert resumed._accepted_baseline.source_bib_sha256 == hashlib.sha256(bib.encode()).hexdigest()
    _round_snapshot(workspace, 1, raw, bibliography="@book{reference, title={Wrong reference}}",
                    accepted=critic == "fresh", forced=False if critic == "fresh" else None)
    resumed._reconcile_accepted(inp)
    assert resumed._accepted_baseline is not None
    # A subsequent rejection of the original source *and* references does revoke.
    _round_snapshot(workspace, 2, raw, bibliography=bib,
                    accepted=critic == "fresh", forced=False if critic == "fresh" else None)
    resumed._reconcile_accepted(inp)
    record = json.loads(agent._accepted_checkpoint().read_text())
    assert record["revoked"] and record["round"] == 2
    assert record["source_bib_sha256"] == hashlib.sha256(bib.encode()).hexdigest()


def test_legacy_bibliography_migration_rejects_mismatched_source(pipeline):
    _, _, _, _, _, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    snapshot = _round_snapshot(agent._workspace(inp), 0, TEX)
    agent._remember_accepted(inp, TEX, "Accepted proof")
    record = json.loads(agent._accepted_checkpoint().read_text())
    record.pop("source_bib_sha256")
    agent._accepted_checkpoint().write_text(json.dumps(record))
    (snapshot / "answer.tex").write_text("Not the accepted round's source")
    assert agent._restore_accepted(inp) is None


def test_missing_bibliography_is_empty_but_unreadable_bibliography_is_not(pipeline, monkeypatch):
    _, _, _, _, _, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    snapshot = _round_snapshot(agent._workspace(inp), 0, TEX)
    expected = agent._snapshot_hashes(snapshot)
    (snapshot / "references.bib").unlink()
    assert agent._snapshot_hashes(snapshot) == expected
    read_bytes = type(snapshot).read_bytes

    def denied_bibliography(path):
        if path.name == "references.bib":
            raise PermissionError("cannot read reviewed bibliography")
        return read_bytes(path)

    monkeypatch.setattr(type(snapshot), "read_bytes", denied_bibliography)
    agent._remember_accepted(inp, TEX, "Accepted proof")
    assert agent._snapshot_hashes(snapshot) is None
    assert not agent._accepted_checkpoint().exists()


@pytest.mark.parametrize("critic", ["stateful", "fresh"])
@pytest.mark.parametrize("ending", ["different_draft", "timeout", "error_output", "fatal", "cancel"])
def test_round_rejection_revokes_before_later_draft_or_interruption(pipeline, monkeypatch, critic, ending):
    execute, calls, _, options, ctx, agent = pipeline
    # AC can compile/inline the source before returning its exported answer.
    # The fingerprint must come from the actual reviewed round, not that export.
    raw = TEX.replace("Proof.", r"Proof.\bibliography{references}")
    options.update(verdicts=["repair"], reviewed_document=raw)
    research = ACDAGWorkflow.run
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    workspace = agent._workspace(inp)

    async def reject_then_continue(self, research_input):
        if not calls["research"]:
            return await research(self, research_input)
        retained = agent._restore_accepted(inp)
        if retained is not None:
            assert retained.source_sha256 == hashlib.sha256(raw.encode()).hexdigest()
            assert retained.sha256 == hashlib.sha256(TEX.encode()).hexdigest()
        _round_snapshot(workspace, 4, raw, accepted=critic == "fresh",
                        forced=False if critic == "fresh" else None)
        if ending == "timeout":
            raise TimeoutError("later research timed out")
        if ending == "fatal":
            raise TypeError("later research failed")
        if ending == "cancel":
            raise asyncio.CancelledError
        options.update(agreed=False, research_document=TEX.replace("Proof.", "Different unaccepted draft."),
                       reviewed_document=TEX.replace("Proof.", "Different unaccepted draft."))
        out = await research(self, research_input)
        # The final stateful verdict says nothing about the earlier rejection.
        assert out.last_critic_accepted is True
        if ending == "error_output":
            out.error = "Research failed after the review"
        return out

    monkeypatch.setattr(ACDAGWorkflow, "run", reject_then_continue)
    if ending == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(execute(max_cleanup_repairs=0, max_recovery_attempts=0))
        resumed = type(agent)(ctx)
        out = asyncio.run(resumed(problem=ORIGINAL, problem_id="p", resume_run=True,
                                 research_seconds=0, partial_cleanup_seconds=0, cleanup_backend="api"))
    else:
        out = asyncio.run(execute(max_cleanup_repairs=0, max_recovery_attempts=0))
    assert not out.submission_approved
    assert agent._restore_accepted(inp) is None
    record = json.loads(agent._accepted_checkpoint().read_text())
    assert record["revoked"] and record["round"] == 4
    assert record["sha256"] == hashlib.sha256(TEX.encode()).hexdigest()
    assert record["review_md"] == f"Round 4 {'fresh' if critic == 'fresh' else 'stateful'} review"


def test_fresh_acceptance_after_an_earlier_rejection_can_replace_the_baseline(pipeline, monkeypatch):
    execute, calls, _, options, _, agent = pipeline
    options["verdicts"] = ["repair", "accept"]
    research = ACDAGWorkflow.run
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")

    async def reject_then_accept(self, research_input):
        if calls["research"]:
            _round_snapshot(agent._workspace(inp), 4, TEX, accepted=False)
        return await research(self, research_input)

    monkeypatch.setattr(ACDAGWorkflow, "run", reject_then_accept)
    out = asyncio.run(execute(max_cleanup_repairs=0))
    assert out.submission_approved and calls["research"] == 2
    assert agent._restore_accepted(inp).round == 6


@pytest.mark.parametrize("review", [None, {"review_md": "No verdict"},
                                   {"answer_ready": False, "parse_failed": True},
                                   {"answer_ready": True}, {"answer_ready": False, "review_md": {}}])
@pytest.mark.parametrize("exported", [False, True])
def test_missing_or_invalid_snapshot_verdict_does_not_revoke(pipeline, review, exported):
    _, _, _, _, _, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    _round_snapshot(agent._workspace(inp), 0, TEX)
    agent._remember_accepted(inp, TEX, "Accepted proof")
    source = TEX.replace("Proof.", r"Proof.\bibliography{references}") if exported else TEX
    snapshot = _round_snapshot(agent._workspace(inp), 1, source)
    (snapshot / "review_outputs.json").write_text("{" if review is None else json.dumps(review))
    (snapshot.parent / "resume-state.json").write_text(json.dumps({"last_round_run": 1}))
    agent._reconcile_accepted(inp, exported_document=TEX if exported else None)
    assert agent._restore_accepted(inp) is not None


def test_export_comparison_does_not_reuse_an_older_rounds_rejection(pipeline):
    _, _, _, _, _, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    workspace = agent._workspace(inp)
    _round_snapshot(workspace, 0, TEX)
    agent._remember_accepted(inp, TEX, "Accepted proof")
    _round_snapshot(workspace, 1, TEX.replace("Proof.", "Different rejected draft."), accepted=False)
    snapshot = _round_snapshot(workspace, 2, TEX.replace("Proof.", r"Proof.\bibliography{references}"))
    (snapshot.parent / "resume-state.json").write_text(json.dumps({"last_round_run": 2}))
    agent._reconcile_accepted(inp, exported_document=TEX)
    assert agent._accepted_baseline is not None
    assert agent._restore_accepted(inp) is not None


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("submitted", [False, True])
def test_resume_reconciles_snapshots_before_publishing_or_returning_approval(pipeline, legacy, submitted):
    execute, calls, _, options, ctx, agent = pipeline
    options.update(verdicts=["repair"])
    out = asyncio.run(execute(max_cleanup_repairs=0, max_cleanup_handoffs=0))
    assert out.submission_approved and out.answer_tex.read_text() == TEX
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    if legacy:
        record = json.loads(agent._accepted_checkpoint().read_text())
        record.pop("source_sha256")
        record.pop("source_bib_sha256")
        agent._accepted_checkpoint().write_text(json.dumps(record))
    if not submitted:
        # Model a crash before fallback publication, with only a partial manifest.
        agent._publish(inp, out, agent._label_partial(TEX), 1, partial=True)
    _round_snapshot(agent._workspace(inp), 4, TEX, accepted=True, forced=False)
    before = calls.copy()
    resumed = type(agent)(ctx)
    result = asyncio.run(resumed(problem=ORIGINAL, problem_id="p", resume_run=True,
                                research_seconds=0, partial_cleanup_seconds=0, cleanup_backend="api"))
    assert result.partial_ready and not result.submission_approved
    assert calls["research"] == before["research"] and calls["review"] == before["review"]
    manifest = json.loads((ctx.root_workdir / "batch3-output.json").read_text())
    assert not manifest["submission_approved"]
    assert resumed._restore_accepted(inp) is None


def test_accounting_block_cannot_restore_a_revoked_approved_fallback(pipeline):
    execute, calls, _, options, ctx, agent = pipeline
    options["verdicts"] = ["repair"]
    assert asyncio.run(execute(max_cleanup_repairs=0, max_cleanup_handoffs=0)).submission_approved
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    _round_snapshot(agent._workspace(inp), 4, TEX, accepted=False)
    (ctx.root_workdir / "cleanup-accounting-uncertain.json").write_text("{}")
    before = calls.copy()
    result = asyncio.run(execute(resume_run=True))
    assert not result.submission_approved and not result.error_retryable
    assert calls == before
    assert not json.loads((ctx.root_workdir / "batch3-output.json").read_text())["submission_approved"]


@pytest.mark.parametrize("legacy", [False, True])
def test_missing_reviewed_source_does_not_create_an_unsafe_retained_fallback(pipeline, legacy):
    _, _, _, _, _, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    if legacy:
        snapshot = _round_snapshot(agent._workspace(inp), 0, TEX)
        agent._remember_accepted(inp, TEX, "Accepted proof")
        record = json.loads(agent._accepted_checkpoint().read_text())
        record.pop("source_sha256")
        record.pop("source_bib_sha256")
        agent._accepted_checkpoint().write_text(json.dumps(record))
        (snapshot / "answer.tex").unlink()
    else:
        agent._remember_accepted(inp, TEX, "Accepted proof")
    assert agent._restore_accepted(inp) is None


def test_fallback_publication_rechecks_snapshots(pipeline):
    _, _, _, _, _, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    raw = TEX.replace("Proof.", r"Proof.\bibliography{references}")
    _round_snapshot(agent._workspace(inp), 0, raw)
    agent._remember_accepted(inp, TEX, "Accepted proof")
    # A resubmission of the exported version must match too, not just raw TeX.
    _round_snapshot(agent._workspace(inp), 1, TEX, accepted=False)
    out = agent.Outputs(problem_id="p")
    assert asyncio.run(agent._publish_retained(inp, out, deadline=time.monotonic() + 60)) is None
    assert not out.submission_approved and out.answer_tex is None


@pytest.mark.parametrize("options", [{"compile_ok": False}, {"pages": 17}, {"pages": 0}])
def test_retained_baseline_still_has_to_pass_export_checks(pipeline, options):
    _, _, _, config, _, agent = pipeline
    config.update(options)
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    _round_snapshot(agent._workspace(inp), 0, TEX)
    agent._remember_accepted(inp, TEX, "Accepted mathematics")
    out = agent.Outputs(problem_id="p")
    assert asyncio.run(agent._publish_retained(inp, out, deadline=time.monotonic() + 60)) is None
    assert not out.submission_approved and out.cleanup_errors


@pytest.mark.parametrize("workflow", ["firstproof_batch3", "firstproof_batch3_multiauthor"])
@pytest.mark.parametrize("parallelism", [1, 2, 7])
def test_batch3_author_parallelism_is_a_direct_input_override(workflow, parallelism):
    preset = load_preset(workflow)
    actual = preset.workflow_cls.Inputs(**preset.build_inputs(
        problem=ORIGINAL, problem_id="p", cli_overrides={"author_parallelism": parallelism},
    ))
    assert actual.author_parallelism == parallelism
    assert actual.compute_max_parallel_workers == 4


@pytest.mark.parametrize("workers", [0, -1])
@pytest.mark.parametrize("workflow", ["firstproof_batch3", "firstproof_batch3_multiauthor"])
def test_batch3_rejects_overrides_that_disable_the_memory_guard(workflow, workers):
    preset = load_preset(workflow)
    with pytest.raises(ValueError, match="compute_max_parallel_workers"):
        preset.workflow_cls.Inputs(**preset.build_inputs(
            problem=ORIGINAL, problem_id="p", cli_overrides={"compute_max_parallel_workers": workers},
        ))


def test_batch3_compute_cap_is_configurable_without_changing_generic_defaults():
    inp = FirstProofBatch3Workflow.Inputs(problem=ORIGINAL, problem_id="p", compute_max_parallel_workers=2)
    assert inp.compute_max_parallel_workers == 2 and inp.compute_sandbox_backend == "subprocess"
    assert ACWorkflow.Inputs(problem=ORIGINAL, problem_id="p").compute_max_parallel_workers == 0


def test_one_rewrite_reviews_and_ships_exact_document(pipeline):
    execute, calls, seen, _, ctx, _ = pipeline
    out = asyncio.run(execute())
    assert out.submission_approved and not out.abstained, out.error
    assert calls == {"research": 1, "rewrite": 1, "compile": 3, "review": 1}
    assert seen["research_inputs"][0].stop_after_review_round
    assert not seen["research_inputs"][0].enable_final_critic
    review = seen["review_inputs"][0]
    assert review.problem == ORIGINAL and review.baseline_tex == TEX
    assert review.mode == "stateful" and review.prior_messages == HISTORY
    assert out.answer_tex.read_text() == review.answer_tex == seen["compiled_document"]
    assert out.submission_sha256 == hashlib.sha256(out.answer_tex.read_bytes()).hexdigest()
    assert not out.partial_ready and out.partial_sha256 is None
    assert ctx.budgets.root().counters.usd == 27
    assert seen["research_caps"] == [85]


def test_targeted_repairs_preserve_critic_conversation(pipeline):
    execute, calls, seen, options, _, _ = pipeline
    options["verdicts"] = ["repair", "repair", "accept"]
    out = asyncio.run(execute())
    assert out.submission_approved
    assert calls["rewrite"] == 1 and calls["repair"] == 2 and calls["review"] == 3
    first, second, third = seen["review_inputs"]
    assert second.baseline_tex == third.baseline_tex == TEX
    assert len(second.prior_messages) == len(first.prior_messages) + 1
    assert len(third.prior_messages) == len(second.prior_messages) + 1
    assert first.answer_tex in second.prior_messages[-2]["content"]
    assert "Research review" in third.prior_messages[0]["content"]
    assert "Review 1" in third.prior_messages[1]["content"]
    assert "Review 1" in seen["prompts"][1] and "targeted editorial repairs" in seen["prompts"][1]
    assert out.answer_tex.read_text() == third.answer_tex


def test_rewrite_regression_restores_baseline_then_reviews_without_new_research(pipeline):
    execute, calls, seen, options, _, _ = pipeline
    options["verdicts"] = ["restore", "accept"]
    out = asyncio.run(execute())
    assert out.submission_approved
    assert calls["research"] == 1 and calls["rewrite"] == 1
    assert calls["repair"] == 1 and calls["review"] == 2
    assert "Restore only the specific baseline proof passages" in seen["prompts"][1]
    assert "Pre-rewrite baseline" in seen["prompts"][1]
    assert "Review 1" in seen["prompts"][1]
    assert seen["review_inputs"][1].prior_messages


def test_baseline_flaw_after_restoration_goes_back_to_research(pipeline):
    execute, calls, seen, options, _, _ = pipeline
    options["verdicts"] = ["restore", "research", "accept"]
    out = asyncio.run(execute())
    assert out.submission_approved and calls["research"] == 2
    assert calls["repair"] == 1
    assert "substantive mathematical flaw" in seen["resumed_states"][0]["pending_critique"]


def test_mathematical_flaw_returns_to_checkpoint_without_resetting_budget(pipeline):
    execute, calls, seen, options, ctx, _ = pipeline
    options["verdicts"] = ["research", "accept"]
    out = asyncio.run(execute())
    assert out.submission_approved and calls["research"] == 2
    assert not seen["research_inputs"][0].resume_run
    assert seen["research_inputs"][1].resume_run
    assert seen["research_inputs"][1].n_rounds > seen["research_inputs"][0].n_rounds
    resumed = seen["resumed_states"][0]
    assert not resumed["early_stopped"] and resumed["terminal_outputs"] is None
    assert resumed["next_round"] == 4 and "Review 1" in resumed["pending_critique"]
    assert TEX in resumed["pending_critique"]
    assert resumed["review_history"][-1]["answer_ready"] is False
    assert len(resumed["critic_conversation"]) == len(HISTORY) + 1
    assert resumed["critic_conversation"][0] == HISTORY[1]
    assert seen["research_caps"] == [85, 58]
    assert ctx.budgets.root().counters.usd == 54


def test_flaw_latched_before_bookkeeping_failure_is_not_lost(pipeline):
    execute, calls, _, options, _, agent = pipeline
    options.update(verdicts=["research"], review_error=RuntimeError("log failure"))
    out = asyncio.run(execute(max_recovery_attempts=0, max_cleanup_handoffs=0))
    assert out.partial_ready and not out.submission_approved
    state = agent._research_state(agent.Inputs(problem=ORIGINAL, problem_id="p"))
    assert not state["early_stopped"] and "Review 1" in state["pending_critique"]
    assert list(agent.ctx.root_workdir.rglob("research-handoff.json"))
    assert calls["review"] == 1 and calls["partial"] == 1


def test_unsolved_research_continues_in_chunks_until_cleanup_reserve(pipeline):
    execute, calls, seen, options, _, _ = pipeline
    options["agreed"] = False
    out = asyncio.run(execute())
    assert calls["research"] == 5
    assert all(i.resume_run for i in seen["research_inputs"][1:])
    assert calls["review"] == 0
    assert out.partial_ready and not out.submission_approved
    assert "Partial result: not a complete solution" in out.answer_tex.read_text()


def test_cutoff_cancels_research_then_rewrites_once_without_critic(pipeline, workflow_clock, monkeypatch):
    execute, calls, seen, options, ctx, _ = pipeline
    options["research_sleep"] = True

    async def exercise():
        started = options["research_started"] = asyncio.Event()

        async def controlled_wait_for(awaitable, timeout):
            if calls["research_cancelled"]:
                return await awaitable
            task = asyncio.create_task(awaitable)
            try:
                # Start the cutoff only after the research checkpoint is saved.
                await asyncio.wait_for(started.wait(), timeout=10)
                assert timeout == 300
                workflow_clock.now += timeout
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            raise TimeoutError("controlled research cutoff")

        monkeypatch.setattr("proofstack.agents.firstproof_batch3.asyncio",
                            SimpleNamespace(**{**vars(asyncio), "wait_for": controlled_wait_for}))
        # The real budget tracker has ample headroom; only the controlled
        # workflow clock decides when this test reaches its research cutoff.
        return await execute(research_seconds=300)

    out = asyncio.run(exercise())
    assert calls == {"research": 1, "research_cancelled": 1, "partial": 1, "compile": 2}
    assert out.partial_ready and not out.submission_approved and not out.early_stopped
    assert out.output_kind == "partial_unreviewed"
    assert out.partial_sha256 == hashlib.sha256(out.answer_tex.read_bytes()).hexdigest()
    assert "Partial result: not a complete solution" in out.answer_tex.read_text()
    assert ctx.budgets.root().counters.usd == 25
    assert "Research findings" in seen["prompts"][-1]


def test_phase_real_timeout_cancels_started_work_and_records_end(pipeline, workflow_clock, monkeypatch):
    _, _, _, _, _, agent = pipeline
    events = []

    async def emit(name, data):
        events.append((name, data))

    monkeypatch.setattr(agent.events, "emit", emit)

    async def exercise():
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def work():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(work())
        await asyncio.wait_for(started.wait(), timeout=10)
        try:
            # Freeze deadline calculation, not asyncio's real timer. No disk
            # writes or task-start scheduling can consume the timeout first.
            with pytest.raises(TimeoutError):
                await agent._phase("research", usd=10, deadline=workflow_clock.now + .02,
                                   call=lambda ctx: task)
            assert task.cancelled() and cancelled.is_set()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())
    assert [name for name, _ in events] == ["batch3.phase_start", "batch3.phase_end"]
    assert events[-1][1]["phase"] == "research"


def test_absolute_research_deadline_is_not_reset_on_late_start(pipeline):
    execute, calls, _, _, _, agent = pipeline
    workspace = agent._workspace(agent.Inputs(problem=ORIGINAL, problem_id="p"))
    workspace.mkdir(parents=True)
    (workspace / "answer.tex").write_text(TEX)
    out = asyncio.run(execute(research_deadline_unix_s=time.time() - 1))
    assert calls == {"partial": 1, "compile": 2}
    assert out.partial_ready


def test_mathematical_rejection_after_cutoff_cannot_restart_research(pipeline, workflow_clock, monkeypatch):
    execute, calls, seen, options, _, _ = pipeline
    options["verdicts"] = ["research"]
    review = Batch3CleanupCritic.run

    async def delayed_review(self, inp):
        workflow_clock.now += 301
        return await review(self, inp)

    monkeypatch.setattr(Batch3CleanupCritic, "run", delayed_review)
    out = asyncio.run(execute(research_seconds=300))
    assert out.partial_ready and not out.submission_approved
    assert calls["research"] == calls["review"] == calls["partial"] == 1
    assert "Review 1" in seen["prompts"][-1]


def test_partial_rewrite_reserves_compilation_time(pipeline, monkeypatch):
    execute, _, _, options, _, agent = pipeline
    options["research_error"] = TimeoutError("cutoff")
    phases = []
    phase = agent._phase

    async def observed(name, **kwargs):
        phases.append((name, kwargs["deadline"]))
        return await phase(name, **kwargs)

    monkeypatch.setattr(agent, "_phase", observed)
    start = time.monotonic()
    out = asyncio.run(execute(max_wallclock_s=30))
    assert out.partial_ready
    cleanup_deadline = dict(phases)["partial_rewrite"]
    # The 90/115 split scales down; three seconds remain for exact export.
    assert 20 < cleanup_deadline - start < 22


def test_failed_partial_rewrite_retains_labelled_compiling_fallback(pipeline):
    execute, calls, _, options, _, _ = pipeline
    options.update(research_error=TimeoutError("cutoff"), partial_error=RuntimeError("provider unavailable"))
    out = asyncio.run(execute())
    assert out.partial_ready and out.cleanup_errors and not out.submission_approved
    assert "Proof." in out.answer_tex.read_text()
    assert calls["review"] == 0 and calls["partial"] == 1


@pytest.mark.parametrize("reply", ["UNABLE: the proof is incomplete", "", TEX + "\nUNABLE: missing argument"])
def test_incomplete_editor_reply_cannot_replace_partial_fallback(pipeline, reply):
    execute, calls, _, options, _, _ = pipeline
    options.update(research_error=TimeoutError("cutoff"), edit_document=reply)
    out = asyncio.run(execute())
    assert out.partial_ready and out.cleanup_errors and not out.submission_approved
    assert "Proof." in out.answer_tex.read_text()
    assert "UNABLE:" not in out.answer_tex.read_text()
    assert calls["review"] == 0 and calls["partial"] == 1


@pytest.mark.parametrize("options", [{"compile_ok": False}, {"pages": 17}, {"pages": 0}])
def test_partial_compile_and_page_gate(pipeline, options):
    execute, _, _, config, _, _ = pipeline
    config.update(research_error=TimeoutError("cutoff"), **options)
    out = asyncio.run(execute())
    assert out.abstained and not out.partial_ready and not out.submission_approved


def test_missing_attempt_abstains_without_calls(pipeline):
    execute, calls, _, _, _, _ = pipeline
    out = asyncio.run(execute(research_seconds=0))
    assert out.abstained and not calls and out.error


def test_invalid_critic_verdict_cannot_approve(pipeline):
    execute, calls, _, options, _, _ = pipeline
    options["verdicts"] = ["nonsense"]
    out = asyncio.run(execute(max_recovery_attempts=0, max_cleanup_handoffs=0))
    # The unreviewed rewrite is not approved; the research-accepted baseline is.
    assert out.submission_approved and out.answer_tex.read_text() == TEX
    assert any("invalid routing verdict" in error for error in out.cleanup_errors)
    assert calls["review"] == 1 and calls["partial"] == 0


def test_failed_restoration_publishes_only_the_accepted_baseline(pipeline):
    execute, calls, _, options, _, _ = pipeline
    options.update(verdicts=["restore"], repair_error=RuntimeError("repair failed"))
    out = asyncio.run(execute(max_recovery_attempts=0, max_cleanup_handoffs=0))
    assert out.submission_approved and not out.partial_ready
    assert out.answer_tex.read_text() == TEX and calls["partial"] == 0


def test_accepted_baseline_is_not_published_after_the_deadline(pipeline):
    _, calls, _, _, _, agent = pipeline
    out = agent.Outputs(problem_id="p")
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    assert asyncio.run(agent._publish_accepted(inp, out, TEX, deadline=time.monotonic() - 1)) is None
    assert out.answer_tex is None and calls["compile"] == 0


def test_repair_cap_stops_editorial_loop(pipeline):
    execute, calls, _, options, _, _ = pipeline
    options["verdicts"] = ["repair"]
    out = asyncio.run(execute(max_cleanup_repairs=1, max_recovery_attempts=0, max_cleanup_handoffs=0))
    assert out.submission_approved and out.answer_tex.read_text() == TEX
    assert calls["review"] == 2 and calls["repair"] == 1


@pytest.mark.parametrize("verdict", ["repair", "restore"])
def test_unchanged_repair_is_not_submitted_for_another_paid_review(pipeline, verdict):
    execute, calls, _, options, ctx, _ = pipeline
    options.update(verdicts=[verdict, "accept"], edit_document=TEX)
    out = asyncio.run(execute(max_recovery_attempts=0, max_cleanup_handoffs=0))
    assert calls["review"] == calls["repair"] == calls["rewrite"] == 1
    events = [json.loads(line) for line in (ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    event = next(e["payload"] for e in events if e["kind"] == "cleanup.repair_unchanged")
    assert event["disposition"] == verdict
    assert event["candidate_sha256"] == hashlib.sha256(TEX.encode()).hexdigest()
    # Neither failed editorial repairs nor rewrite-only regressions revoke
    # the baseline. The uncorrected candidate is never submitted instead.
    assert out.submission_approved and out.answer_tex.read_text() == TEX



def test_phase_root_enforces_local_and_shared_budget(pipeline):
    _, _, _, _, ctx, agent = pipeline
    phase = agent._phase_context("test", usd=10, seconds=60)
    child = phase.budgets.child("agent:test", parent=phase.budgets.root())
    child.add_usd(10)
    with pytest.raises(BudgetExhausted):
        child.check()
    assert ctx.budgets.root().counters.usd == 10
    assert agent._remaining_usd() == 90


@pytest.mark.parametrize("already_accounted", [0, 12])
def test_resume_cost_offset_does_not_charge_twice(pipeline, already_accounted):
    _, _, _, _, ctx, agent = pipeline
    (ctx.root_workdir / "events.jsonl").write_text(json.dumps({"kind": "model.call", "payload": {"cost_usd": 12}}) + "\n")
    ctx.budgets.root().add_usd(already_accounted)
    phase = agent._phase_context("resume", usd=100, seconds=60)
    helper = ACWorkflow(phase)
    asyncio.run(helper._apply_resume_budget_offset())
    asyncio.run(helper._apply_resume_budget_offset())
    assert ctx.budgets.root().counters.usd == 12


def test_external_cancellation_does_not_start_partial_cleanup(pipeline, monkeypatch):
    execute, calls, _, _, _, _ = pipeline
    entered = asyncio.Event()

    async def swallowed_cancel(self, inp):
        calls["rewrite"] += 1
        entered.set()
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            return self.Outputs(text=TEX)

    monkeypatch.setattr(RewriteSeat, "run", swallowed_cancel)

    async def run():
        task = asyncio.create_task(execute())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert calls == {"research": 1, "rewrite": 1, "compile": 2}
    assert json.loads((pipeline[4].root_workdir / "batch3-output.json").read_text())["partial_ready"]


def test_subscription_park_propagates(pipeline):
    execute, _, _, options, _, _ = pipeline
    options["research_error"] = SubscriptionParked("week", 1000, 10, 10)
    with pytest.raises(SubscriptionParked):
        asyncio.run(execute())


@pytest.mark.parametrize("command", [r"\bibliography{refs}", r"\addbibresource[location=local]{refs.bib}", r"\input other.tex"])
def test_external_files_cannot_be_approved(pipeline, command):
    execute, _, _, options, _, _ = pipeline
    options["edit_document"] = TEX.replace(r"\end{document}", command + "\n" + r"\end{document}")
    out = asyncio.run(execute(max_cleanup_repairs=0))
    # Only the accepted baseline, which has no external file, is published.
    assert out.submission_approved and out.answer_tex.read_text() == TEX
    assert command not in out.answer_tex.read_text()


@pytest.mark.parametrize("verdict", ["accept", "repair", "research"])
def test_critic_routing_and_regression_prompt(tmp_path, verdict):
    critic = Batch3CleanupCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    inp = critic.Inputs(problem=ORIGINAL, answer_tex=TEX, baseline_tex="baseline", prior_messages=HISTORY)
    raw = f"Findings\n<cleanup_verdict>{verdict}</cleanup_verdict>"
    out = critic.parse_output(raw, inp)
    assert out.disposition == verdict and out.answer_ready == (verdict == "accept")
    assert not out.parse_failed and out.messages_after[0] == HISTORY[1]
    prompt = out.messages_after[-2]["content"]
    assert ORIGINAL in prompt and "baseline" in prompt and "entire argument" in prompt
    assert "Prior acceptance is not evidence" in prompt


@pytest.mark.parametrize("raw", ["", "accept", "<cleanup_verdict>accept</cleanup_verdict>\nBut invalid",
    "<cleanup_verdict>repair</cleanup_verdict>\n<cleanup_verdict>accept</cleanup_verdict>"])
def test_critic_rejects_ambiguous_verdicts(tmp_path, raw):
    critic = Batch3CleanupCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    out = critic.parse_output(raw, critic.Inputs(problem=ORIGINAL))
    assert out.parse_failed and not out.answer_ready


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_exact_document_compile_smoke(tmp_path):
    agent = FirstProofBatch3Workflow(RunContext.create(root_workdir=tmp_path, flat=True))
    compiled, pages, _ = asyncio.run(agent._compile(TEX, deadline=time.monotonic() + 60))
    assert compiled and pages == 1


@pytest.mark.parametrize("overrides", [{"page_limit": 17},
    {"research_seconds": -1}, {"partial_cleanup_reserve_fraction": 1}, {"max_cleanup_repairs": -1}])
def test_invalid_competition_inputs_are_rejected(pipeline, overrides):
    execute, calls, _, _, _, _ = pipeline
    with pytest.raises(ValueError):
        asyncio.run(execute(**overrides))
    assert not calls


def test_submission_constraints_are_opt_in_and_not_spliced_into_documents():
    plain = _assemble("rewrite-wrapper.txt", {"document": TEX})
    assert plain == _assemble("rewrite-wrapper.txt", {"document": TEX}, writing_constraints=None)
    constrained = _assemble("rewrite-wrapper.txt", {"document": TEX}, writing_constraints="16 pages; keep {{document}} literal")
    assert constrained == plain + "\n\n# Submission constraints\n\n16 pages; keep {{document}} literal"
    assert "# Submission constraints" not in _assemble("cold-referee.txt", {"document": TEX})


def test_real_research_dag_resumes_after_editorial_rejection(tmp_path, monkeypatch):
    from proofstack.agents.ac import ac_workflow
    from proofstack.agents.ac.author import Author
    from proofstack.agents.ac.critic import ACCritic

    preset = load_preset("firstproof_batch3")
    authors, editorial_reviews = [], []

    def no_network(*args, **kwargs):
        raise AssertionError("offline integration test attempted a paid call")

    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
        run_budget=BudgetSpec(max_usd=100, max_wallclock_s=60),
        component_configs=preset.component_configs, api_client_factory=no_network)

    async def author(self, inp):
        authors.append(inp)
        self.tracker.add_usd(.2)
        return self.Outputs(answer_tex=TEX, research_notes_tex="Research notes",
                            references_bib="", ready=inp.round >= 1)

    async def research_critic(self, inp):
        self.tracker.add_usd(.1)
        metadata = self._notes_metadata(inp)
        self._provider_tool_evidence = [{
            "type": "code_interpreter_call", "status": "completed", "container_id": "synthetic",
            "code": self._notes_verification_code(inp),
            "outputs": [{"type": "logs", "logs": json.dumps({
                k: metadata[k] for k in ("filename", "bytes", "sha256")})}],
        }]
        return self.parse_output("Research findings\n<research_notes_status>verified</research_notes_status>\n<answer_ready>true</answer_ready>" if inp.round >= 1
                                 else "Continue\n<answer_ready>false</answer_ready>", inp)

    async def edit(self, inp):
        return self.Outputs(text=TEX)

    async def review(self, inp):
        editorial_reviews.append(inp)
        verdict = "research" if len(editorial_reviews) == 1 else "accept"
        return self.parse_output(f"A missing mathematical argument\n<cleanup_verdict>{verdict}</cleanup_verdict>", inp)

    async def ready(self, workspace, *, page_limit):
        return True, []

    def compile_research(tex, **kwargs):
        return ac_workflow._CompileResult(tex=tex, tex_path=None, pdf_path=None, compiled=True, pages=1)

    async def compile_editorial(self, *args, **kwargs):
        return True, 1, "OK"

    monkeypatch.setattr(Author, "run", author)
    monkeypatch.setattr(ACCritic, "run", research_critic)
    monkeypatch.setattr(Batch3CleanupCritic, "run", review)
    monkeypatch.setattr(RewriteSeat, "run", edit)
    monkeypatch.setattr(RepairSeat, "run", edit)
    monkeypatch.setattr(ACWorkflow, "_deterministic_ready", ready)
    monkeypatch.setattr(ac_workflow, "_simple_compile_latex", compile_research)
    monkeypatch.setattr(FirstProofBatch3Workflow, "_compile", compile_editorial)

    async def run():
        return await preset.workflow_cls(ctx)(**preset.build_inputs(problem=ORIGINAL, problem_id="p",
            cli_overrides={"n_rounds": 2, "enable_compute": False, "enable_council": False,
                           "cleanup_backend": "api", "partial_cleanup_seconds": 0}))

    out = asyncio.run(run())
    assert out.submission_approved, out.model_dump()
    assert [i.round for i in authors] == [0, 1, 2]
    assert "missing mathematical argument" in authors[-1].prev_critique
    assert "Prior acceptance is revoked" in authors[-1].prev_critique
    assert editorial_reviews[0].prior_messages
    assert out.rounds_completed == 2
    assert "Final export check" in authors[1].workflow_feedback
    assert "Aim for at most 15 pages" in authors[1].workflow_feedback
    workspace = ctx.root_workdir / "ac_workspaces" / f"p-{_problem_hash(ORIGINAL)}"
    assert len(list((workspace / ".ac").glob("partial-export-round-*.log"))) == 3
    assert list((ctx.root_workdir / "partials" / "p").glob("*.tex"))


def _exported_path(ctx):
    from dataclasses import replace
    from test_firstproof_profiles import fp

    settings = replace(fp._settings(), workflow="firstproof_batch3", batch3=True,
                       page_limit=16, output_dir=ctx.root_workdir.parent.parent)
    problem = fp.Problem(ordinal=1, original_id="p", safe_id="p", text=ORIGINAL, input_error=None,
        problem_path=ctx.root_workdir / "problem.tex", log_path=ctx.root_workdir / "run.log",
        output_tex_path=ctx.root_workdir / "out.tex", run_id=ctx.root_workdir.name)
    return fp._find_solution_tex(problem, settings)


@pytest.mark.parametrize("body", [
    "\n\nProof.\n\n",
    "A.\n \n \n \n \nB.",
    "A.\n" + " \t\n" * 31 + "B.",
    r"\textcolor{red}{Proof.}",
    r"\providecommand{\url}[1]{#1}\url{https://example.org}",
])
@pytest.mark.parametrize("partial", [False, True])
def test_workflow_publication_survives_actual_adapter_normalization(pipeline, body, partial):
    execute, _, seen, options, ctx, _ = pipeline
    options["edit_document"] = TEX.replace("Proof.", body)
    if partial:
        options["research_error"] = TimeoutError("research unavailable")
    out = asyncio.run(execute(max_recovery_attempts=0, max_cleanup_handoffs=0))
    assert out.partial_ready == partial
    assert out.submission_approved != partial
    exported = _exported_path(ctx)
    assert exported is not None and exported.read_bytes() == out.answer_tex.read_bytes()
    if not partial:
        assert exported.read_text() == seen["review_inputs"][-1].answer_tex


def test_cited_fallback_is_inlined_before_a_failed_rewrite(pipeline, monkeypatch):
    from proofstack.agents import firstproof_batch3 as module

    execute, _, _, options, ctx, _ = pipeline
    bib = "@book{ref, author={A. Writer}, title={Result}, year={2020}}"
    document = TEX.replace("Proof.", r"Proof \cite{ref}.\bibliographystyle{plain}\bibliography{references}")
    options.update(research_document=document, bibliography=bib,
                   research_error=TimeoutError("cutoff"), partial_error=RuntimeError("provider down"))

    def compile_bib(tex, bib_text, **kwargs):
        assert bib_text == bib
        assert "\\bibliography{references}" in tex
        kwargs["bbl_output"].write_text(r"\begin{thebibliography}{1}\bibitem{ref}A. Writer. Result. 2020.\end{thebibliography}")
        return True, 1, ""

    monkeypatch.setattr(module, "_compile_raw", compile_bib)
    out = asyncio.run(execute(max_recovery_attempts=0, max_cleanup_handoffs=0))
    assert out.partial_ready and _exported_path(ctx) is not None
    assert r"\bibitem{ref}" in out.answer_tex.read_text()
    assert r"\bibliography{" not in out.answer_tex.read_text()
    assert not list(ctx.root_workdir.rglob("fallback.bbl"))


def test_cleanup_timeout_leaves_an_exportable_fallback(pipeline, monkeypatch):
    execute, calls, _, _, ctx, agent = pipeline

    async def stuck(*args, **kwargs):
        assert _exported_path(ctx) is not None
        await asyncio.sleep(10)

    monkeypatch.setattr(agent, "_candidate_loop", stuck)
    out = asyncio.run(execute(max_wallclock_s=.15, max_recovery_attempts=0, max_cleanup_handoffs=0))
    assert _exported_path(ctx) is not None
    assert out.submission_approved and out.answer_tex.read_text() == TEX


@pytest.mark.parametrize("partial,disposition", [(False, "repair"), (False, "restore"), (True, None)])
@pytest.mark.parametrize("workflow", ["firstproof_batch3", "firstproof_batch3_multiauthor"])
def test_default_claude_cleanup_keeps_publication_gates(pipeline, monkeypatch, partial, disposition, workflow):
    from proofstack.agents.cleanup_session import CleanupSession
    _, calls, seen, options, ctx, agent = pipeline
    edits = []

    async def session(self, inp):
        edits.append(inp)
        self.tracker.add_usd(1)
        return self.Outputs(answer_tex=inp.document + "\n% cli revision\n", feedback_md="Editorial feedback",
                            status="ready", summary="Edited", workspace=ctx.root_workdir,
                            session_id="session")

    monkeypatch.setattr(CleanupSession, "run", session)
    if partial:
        options["research_error"] = TimeoutError("research cutoff")
    else:
        options["verdicts"] = [disposition, "accept"]
    out = asyncio.run(agent(**load_preset(workflow).build_inputs(
        problem=ORIGINAL, problem_id="p", cli_overrides={"max_recovery_attempts": 0, "max_cleanup_handoffs": 0,
                                                   "partial_cleanup_seconds": 0})))
    assert len(edits) == (1 if partial else 2)
    assert all(edit.partial == partial for edit in edits)
    assert all(edit.page_limit == 16 and "page" in edit.constraints for edit in edits)
    assert calls["rewrite"] == calls["repair"] == 0
    assert calls["review"] == (0 if partial else 2)
    assert out.partial_ready == partial
    assert out.submission_approved == (not partial)
    if not partial:
        assert all(review.editor_response == "Editorial feedback" for review in seen["review_inputs"])
        assert edits[0].session_key == edits[1].session_key
        assert edits[0].baseline == edits[1].baseline
        assert "Review 1" in edits[1].findings
        if disposition == "restore":
            assert "Restore only" in edits[1].constraints
            assert "status=unable in completion.json" in edits[1].constraints
            assert "return UNABLE:" not in edits[1].constraints
    assert "% cli revision" in out.answer_tex.read_text()


def test_context_failure_does_not_restart_research_or_rewrite(pipeline, monkeypatch):
    from proofstack.agents.batch3_critic import CleanupContextTooLarge
    execute, calls, _, _, ctx, _ = pipeline

    async def too_large(self, inp):
        calls["review"] += 1
        raise CleanupContextTooLarge("context_length_exceeded")

    monkeypatch.setattr(Batch3CleanupCritic, "run", too_large)
    out = asyncio.run(execute(max_recovery_attempts=2))
    assert not out.error_retryable and out.submission_approved
    assert out.answer_tex.read_text() == TEX
    assert calls["research"] == calls["rewrite"] == calls["review"] == 1
    assert calls["repair"] == calls["partial"] == 0
    assert "context_length_exceeded" in out.error
    assert list(ctx.root_workdir.rglob("candidate-1-0.tex"))


def test_context_failure_during_restoration_retains_baseline_acceptance(pipeline):
    from proofstack.agents.batch3_critic import CleanupContextTooLarge
    execute, calls, _, options, _, _ = pipeline
    options.update(verdicts=["restore"], repair_error=CleanupContextTooLarge("context_length_exceeded"))
    out = asyncio.run(execute(max_recovery_attempts=2))
    assert not out.error_retryable and out.submission_approved
    assert out.answer_tex.read_text() == TEX and not out.partial_ready
    assert calls["research"] == calls["rewrite"] == calls["repair"] == 1
    assert calls["review"] == 1 and calls["partial"] == 0


def test_research_context_failure_is_not_retried_or_paid_rewritten(pipeline):
    from proofstack.agents.ac.critic import CriticContextTooLarge

    execute, calls, _, options, _, _ = pipeline
    options.update(research_error=CriticContextTooLarge("fresh review is too large"),
                   research_accepted=False, agreed=False)
    out = asyncio.run(execute(max_recovery_attempts=2))
    assert not out.error_retryable and not out.submission_approved
    assert calls["research"] == 1
    assert calls["rewrite"] == calls["repair"] == calls["partial"] == calls["review"] == 0
    assert out.answer_tex.is_file()


def test_cleanup_session_keys_survive_restarts_without_baseline_collisions(pipeline, monkeypatch):
    from proofstack.agents.cleanup_session import CleanupSession
    _, _, _, _, ctx, agent = pipeline
    keys = []

    async def session(self, inp):
        keys.append(inp.session_key)
        return self.Outputs(answer_tex=inp.document, feedback_md="", status="ready", summary="Edited",
                            workspace=ctx.root_workdir, session_id="session")

    monkeypatch.setattr(CleanupSession, "run", session)
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p", cleanup_backend="claude_code")

    async def run():
        # Recreated workflow instances reuse the local episode number after a restart.
        for baseline, episode in ((TEX, 1), (TEX + "\n% revised argument", 1), (TEX, 4)):
            resumed = type(agent)(ctx)
            await resumed._candidate_loop(ctx, inp, baseline, "", episode=episode,
                                          deadline=time.monotonic() + 60)

    asyncio.run(run())
    assert keys[0] != keys[1] and keys[0] == keys[2]


def test_failed_claude_cleanup_preserves_partial_fallback(pipeline, monkeypatch):
    from proofstack.agents.cleanup_session import CleanupSession
    execute, calls, _, options, _, _ = pipeline

    async def broken(self, inp):
        raise RuntimeError("CLI usage unknown")

    monkeypatch.setattr(CleanupSession, "run", broken)
    options["research_error"] = TimeoutError("research cutoff")
    out = asyncio.run(execute(cleanup_backend="claude_code", max_recovery_attempts=0, max_cleanup_handoffs=0))
    assert out.partial_ready and not out.submission_approved
    assert calls["review"] == 0
    assert "Proof." in out.answer_tex.read_text()
    assert any("CLI usage unknown" in error for error in out.cleanup_errors)
    assert calls["research"] == 1


@pytest.mark.parametrize("crash_marker_only", [False, True])
def test_unresolved_cleanup_usage_never_retries_paid_work(pipeline, monkeypatch, crash_marker_only):
    from proofstack.agents.cleanup_session import CleanupSession, CleanupAccountingUncertain

    execute, calls, _, _, ctx, _ = pipeline
    edits = []

    async def broken(self, inp):
        edits.append(inp)
        (ctx.root_workdir / "cleanup-accounting-uncertain.json").write_text("{}")
        raise CleanupAccountingUncertain("CLI usage unknown")

    monkeypatch.setattr(CleanupSession, "run", broken)
    first = asyncio.run(execute(cleanup_backend="claude_code"))
    # Publishing the accepted baseline needs no model call; the block stays.
    assert first.submission_approved and first.answer_tex.read_text() == TEX
    assert not first.error_retryable and "CLI usage unknown" in first.error
    assert calls["research"] == 1 and calls["review"] == 0
    assert len(edits) == 1
    counts = calls.copy()
    if crash_marker_only:
        (ctx.root_workdir / "cleanup-accounting-uncertain.json").unlink()
        session = ctx.root_workdir / "cleanup_sessions/solved-1"
        session.mkdir(parents=True)
        (session / "session.json").write_text(json.dumps({"in_flight": True}))
    second = asyncio.run(execute(cleanup_backend="claude_code", resume_run=True))
    assert not second.error_retryable and "unresolved" in second.error
    assert second.answer_tex == first.answer_tex
    assert calls == counts and len(edits) == 1


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="TeX tools not installed")
def test_publication_check_cannot_read_host_files(tmp_path):
    ctx = RunContext.create(root_workdir=tmp_path / "run", flat=True)
    agent = FirstProofBatch3Workflow(ctx)
    secret = tmp_path / "host-secret.tex"
    secret.write_text("PRIVATE-HOST-CONTENT\n")
    tex = TEX.replace("Proof.", r"\newread\hostfile\openin\hostfile=" + str(secret)
                      + r"\relax\read\hostfile to \stolen\typeout{\stolen}")
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    ok, _, detail = asyncio.run(agent._check(tex, inp, time.monotonic() + 30))
    assert not ok and "PRIVATE-HOST-CONTENT" not in detail


def test_completed_acceptance_survives_cost_exhaustion(pipeline):
    execute, calls, seen, options, ctx, _ = pipeline
    options["review_error"] = BudgetExhausted("candidate", "usd", 2, 3)
    out = asyncio.run(execute())
    assert out.submission_approved and not out.partial_ready
    assert out.answer_tex.read_text() == seen["review_inputs"][-1].answer_tex
    assert _exported_path(ctx) is not None
    assert calls["partial"] == 0


def test_actual_api_accounting_preserves_over_budget_acceptance(pipeline, monkeypatch):
    from proofstack.kinds.api_call import APICallAgent

    execute, calls, _, _, ctx, _ = pipeline

    class Client:
        model = "offline-critic"
        timeout = max_wallclock_per_call_s = 100

        def run_queries(self, conversations, **kwargs):
            reply = "Verified\n<cleanup_verdict>accept</cleanup_verdict>"
            yield 0, [*conversations[0], {"role": "assistant", "content": reply}], {"cost": 80}

    ctx.api_client_factory = lambda _: Client()
    monkeypatch.setattr(Batch3CleanupCritic, "run", APICallAgent.run)
    out = asyncio.run(execute())
    assert out.submission_approved and calls["partial"] == 0
    assert ctx.budgets.root().counters.usd == 105
    assert _exported_path(ctx) is not None
    assert '"cost_usd": 80.0' in (ctx.root_workdir / "events.jsonl").read_text()


@pytest.mark.parametrize("verdict,legacy,expected", [
    ("accept", "true", "accept"), ("repair", "false", "repair"),
    ("research", "false", "research"), ("accept", "false", "invalid"),
    ("research", "true", "invalid"),
])
def test_cleanup_verdict_with_legacy_research_tag(tmp_path, verdict, legacy, expected):
    critic = Batch3CleanupCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    raw = f"Findings\n<cleanup_verdict>{verdict}</cleanup_verdict>\n<answer_ready>{legacy}</answer_ready>"
    out = critic.parse_output(raw, critic.Inputs(problem=ORIGINAL))
    assert out.disposition == expected


def test_transient_research_failure_resumes_checkpoint(pipeline, monkeypatch):
    execute, calls, seen, _, ctx, _ = pipeline
    research = ACDAGWorkflow.run

    async def fail_once(self, inp):
        out = await research(self, inp)
        if calls["research"] == 1:
            raise ConnectionError("temporary provider outage")
        return out

    monkeypatch.setattr(ACDAGWorkflow, "run", fail_once)
    out = asyncio.run(execute())
    assert out.submission_approved and calls["research"] == 2
    assert seen["research_inputs"][1].resume_run
    assert "batch3.recovery" in (ctx.root_workdir / "events.jsonl").read_text()


def test_persistent_failure_has_a_bounded_recovery_count(pipeline):
    execute, calls, _, options, _, _ = pipeline
    options["research_error"] = ConnectionError("provider unavailable")
    out = asyncio.run(execute(max_recovery_attempts=2))
    assert out.partial_ready and calls["research"] == 3 and calls["partial"] == 1


@pytest.mark.parametrize("error", [TypeError("unexpected keyword argument"), AttributeError("broken adapter")])
@pytest.mark.parametrize("serialized", [False, True])
def test_programming_error_stops_without_further_paid_calls(pipeline, monkeypatch, error, serialized):
    execute, calls, _, _, ctx, _ = pipeline
    research = ACDAGWorkflow.run

    async def broken(self, inp):
        await research(self, inp)
        if not serialized:
            raise error
        # Exercise the actual DAG salvage boundary, including JSON serialization.
        out = await self._last_gasp(inp, {}, error)
        return self.Outputs.model_validate_json(out.model_dump_json())

    monkeypatch.setattr(ACDAGWorkflow, "run", broken)
    out = asyncio.run(execute())
    assert calls["research"] == 1
    assert calls["rewrite"] == calls["repair"] == calls["review"] == calls["partial"] == 0
    assert out.error and type(error).__name__ in out.error
    assert not out.error_retryable
    assert out.partial_ready and not out.submission_approved
    assert "Proof." in out.answer_tex.read_text()
    events = [json.loads(line) for line in (ctx.root_workdir / "events.jsonl").read_text().splitlines()]
    assert not any(e["kind"] == "batch3.recovery" for e in events)
    assert any(e["kind"] == "batch3.recovery_blocked" for e in events)
    assert json.loads((ctx.root_workdir / "batch3-output.json").read_text())["error"] == out.error


@pytest.mark.parametrize("serialized", [False, True])
def test_fatal_retry_decision_survives_interrupted_fallback(pipeline, monkeypatch, serialized):
    execute, calls, _, _, ctx, agent = pipeline
    research = ACDAGWorkflow.run

    async def broken(self, inp):
        out = await research(self, inp)
        if serialized:
            return out.model_copy(update={"error": "TypeError: broken adapter", "error_retryable": False})
        raise TypeError("broken adapter")

    async def interrupted_fallback(*args, **kwargs):
        saved = json.loads((ctx.root_workdir / "batch3-output.json").read_text())
        assert saved["error_retryable"] is False and saved["error"]
        raise asyncio.CancelledError

    monkeypatch.setattr(ACDAGWorkflow, "run", broken)
    monkeypatch.setattr(agent, "_fallback", interrupted_fallback)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(execute())
    assert calls["research"] == 1 and calls["partial"] == 0


def test_explicit_resume_after_programming_fix_is_allowed(pipeline):
    execute, calls, _, options, _, _ = pipeline
    options["research_error"] = TypeError("broken adapter")
    first = asyncio.run(execute())
    assert not first.error_retryable
    del options["research_error"]
    second = asyncio.run(execute(resume_run=True))
    assert calls["research"] == 2
    assert second.submission_approved and second.error is None and second.error_retryable


def test_serialized_transient_research_error_can_still_recover(pipeline, monkeypatch):
    execute, calls, _, _, _, _ = pipeline
    research = ACDAGWorkflow.run

    async def flaky(self, inp):
        out = await research(self, inp)
        if calls["research"] == 1:
            return await self._last_gasp(inp, {}, ConnectionError("provider temporarily unavailable"))
        return out

    monkeypatch.setattr(ACDAGWorkflow, "run", flaky)
    out = asyncio.run(execute())
    assert out.submission_approved and calls["research"] == 2


def test_programming_error_in_cleanup_does_not_restart_research(pipeline):
    execute, calls, _, options, _, _ = pipeline
    options["rewrite_error"] = TypeError("broken rewrite adapter")
    out = asyncio.run(execute())
    assert calls["research"] == calls["rewrite"] == 1
    assert calls["review"] == calls["partial"] == 0
    assert out.error and not out.error_retryable and "broken rewrite adapter" in out.error
    assert out.submission_approved and out.answer_tex.read_text() == TEX


def test_repair_limit_returns_to_research_before_giving_up(pipeline):
    execute, calls, seen, options, _, _ = pipeline
    options["verdicts"] = ["repair", "accept"]
    out = asyncio.run(execute(max_cleanup_repairs=0))
    assert out.submission_approved and calls["research"] == 2
    assert "Editorial cleanup could not finish" in seen["resumed_states"][0]["pending_critique"]


def test_round_chunks_cannot_exceed_ac_input_limit(pipeline):
    execute, calls, seen, options, _, _ = pipeline
    options["agreed"] = False
    out = asyncio.run(execute(n_rounds=200))
    assert out.partial_ready and calls["research"] >= 3
    assert [inp.n_rounds for inp in seen["research_inputs"]][:3] == [200, 400, 500]


def test_resume_without_saved_schedule_is_rejected_before_calls(pipeline):
    execute, calls, _, _, _, _ = pipeline
    with pytest.raises(ValueError):
        asyncio.run(execute(resume_run=True))
    assert not calls


@pytest.mark.parametrize("legacy_debit", [0, 7.0698525])
def test_resume_keeps_absolute_deadlines_and_cumulative_budget(pipeline, legacy_debit):
    from proofstack.agents.firstproof_batch3 import _save_json

    execute, calls, _, _, ctx, _ = pipeline
    now = time.time()
    _save_json(ctx.root_workdir / "batch3-schedule.json", {
        "problem_hash": _problem_hash(ORIGINAL), "research_deadline_unix_s": now - 1,
        "run_deadline_unix_s": now + 500, "initial_usd": 100, "partial_reserve_usd": 15,
    })
    (ctx.root_workdir / "events.jsonl").write_text(json.dumps({"kind": "model.call", "payload": {"cost_usd": 40}}) + "\n")
    if legacy_debit:
        _save_json(ctx.root_workdir / "recovery-debits.json", {
            "legacy": {"cost_usd": legacy_debit, "reason": "Known missing provider usage"},
        })
    out = asyncio.run(execute(resume_run=True))
    assert out.abstained and not calls
    assert ctx.budgets.root().counters.usd == pytest.approx(40 + legacy_debit)
    saved = json.loads((ctx.root_workdir / "batch3-schedule.json").read_text())
    assert saved["run_deadline_unix_s"] == now + 500


def test_resume_receipt_update_during_settlement_is_charged_only_once(pipeline, monkeypatch):
    from mathagents.provider_trace import ProviderTrace, latest_attempts
    from proofstack.agents.ac.critic import ACCritic
    from proofstack.agents.firstproof_batch3 import _save_json

    execute, calls, _, _, ctx, _ = pipeline
    now = time.time()
    _save_json(ctx.root_workdir / "batch3-schedule.json", {
        "problem_hash": _problem_hash(ORIGINAL), "research_deadline_unix_s": now - 1,
        "run_deadline_unix_s": now + 500, "initial_usd": 100, "partial_reserve_usd": 15,
    })
    # An older process left a crash fragment; agent.start will follow it on resume.
    (ctx.root_workdir / "events.jsonl").write_text('{"kind":')
    trace = ProviderTrace(ctx.root_workdir / "agents/critic/provider-attempts.jsonl", call_id="saved-call")
    trace.update(("request", 0), response_id="saved", status="completed", cost=2, usage_unavailable=False)
    emit = ctx.events.emit

    async def late_receipt(kind, *args, **kwargs):
        await emit(kind, *args, **kwargs)
        if kind == "model.call":
            trace.update(("request", 0), cost=3)

    monkeypatch.setattr(ctx.events, "emit", late_receipt)

    async def exercise():
        out = await execute(resume_run=True)
        assert out.abstained and not calls
        assert ctx.budgets.root().counters.usd == 2
        critic = ACCritic(ctx)
        await critic._reconcile_recovery(trace.path, latest_attempts(trace.path))
        await critic._reconcile_recovery(trace.path, latest_attempts(trace.path))
        assert ctx.budgets.root().counters.usd == 3

    asyncio.run(exercise())


@pytest.mark.parametrize("cross_cutoff", [False, True])
def test_adapter_crash_retry_restores_spend_and_original_schedule(pipeline, monkeypatch, cross_cutoff):
    from dataclasses import replace
    from test_firstproof_profiles import fp

    _, calls, seen, _, ctx, _ = pipeline
    preset = load_preset("firstproof_batch3")
    unix_now = [time.time()]
    monkeypatch.setattr(time, "time", lambda: unix_now[0])
    settings = replace(fp._settings(), workflow="firstproof_batch3", batch3=True,
                       page_limit=16, n_rounds=50, budget_usd_per_question=100,
                       output_dir=ctx.root_workdir.parent.parent,
                       deadline_at=time.monotonic() + 600, research_deadline_at=time.monotonic() + 200)
    problem = fp.Problem(ordinal=1, original_id="p", safe_id="p", text=ORIGINAL, input_error=None,
        problem_path=ctx.root_workdir / "problem.tex", log_path=ctx.root_workdir / "run.log",
        output_tex_path=ctx.root_workdir / "out.tex", run_id=ctx.root_workdir.name)
    research = ACDAGWorkflow.run
    contexts, schedules = [], []

    async def interrupted_research(self, inp):
        result = await research(self, inp)
        if len(contexts) == 1:
            await self.events.emit("model.call", {"cost_usd": 20})
            raise asyncio.CancelledError("simulated process death before research returns")
        return result

    async def run_subprocess(p, s, *, restart_from, n_rounds, stage_index):
        fresh = RunContext.create(
            run_id=p.run_id, root_workdir=ctx.root_workdir, flat=True,
            run_budget=BudgetSpec(max_usd=s.budget_usd_per_question, max_wallclock_s=1000),
            component_configs=preset.component_configs, api_client_factory=ctx.api_client_factory,
        )
        contexts.append(fresh)
        assert restart_from == (None if len(contexts) == 1 else p.run_id)
        try:
            await preset.workflow_cls(fresh)(**preset.build_inputs(
                problem=p.text, problem_id=p.safe_id, cli_overrides={
                    "resume_run": restart_from is not None, "n_rounds": n_rounds,
                    "max_wallclock_s": 1000, "research_seconds": 200, "partial_cleanup_seconds": 0,
                    "cleanup_backend": "api",
                },
            ))
        except asyncio.CancelledError:
            assert len(contexts) == 1
            assert not (ctx.root_workdir / "batch3-output.json").exists()
            if cross_cutoff:
                unix_now[0] += 250
            returncode = -9
        else:
            returncode = 0
        schedules.append((ctx.root_workdir / "batch3-schedule.json").read_bytes())
        return returncode

    async def compile(*args, **kwargs):
        return True, "OK"

    monkeypatch.setattr(ACDAGWorkflow, "run", interrupted_research)
    monkeypatch.setattr(fp, "_run_subprocess", run_subprocess)
    monkeypatch.setattr(fp, "_verify_exact_latex_for_submission", compile)
    result = asyncio.run(fp._run_problem(problem, settings, asyncio.Semaphore(1)))
    assert len(contexts) == 2 and schedules[0] == schedules[1]
    assert contexts[0].budgets.root().counters.usd == 20
    assert contexts[1].budgets.root().counters.usd == (25 if cross_cutoff else 47)
    assert seen["research_caps"] == ([85] if cross_cutoff else [85, 65])
    assert calls["research"] == (1 if cross_cutoff else 2)
    assert result.solved != cross_cutoff
    if cross_cutoff:
        assert calls["partial"] == 1 and not calls["review"]
        assert result.status == "partial_unreviewed"
    else:
        assert seen["research_inputs"][-1].resume_run
        assert seen["resumed_states"][0]["last_round_run"] == 3
    assert fp._find_solution_tex(problem, settings) is not None


def test_resumed_cleanup_preserves_prior_episode_artifacts(pipeline, monkeypatch):
    execute, calls, _, options, ctx, agent = pipeline
    options["verdicts"] = ["research"]
    remaining = agent._remaining_usd
    # Stop after the rejection without spending the real budget needed to resume.
    monkeypatch.setattr(agent, "_remaining_usd", lambda: 0 if calls["review"] else remaining())
    first = asyncio.run(execute(max_recovery_attempts=0, max_cleanup_handoffs=0))
    assert first.partial_ready and not first.submission_approved
    monkeypatch.setattr(agent, "_remaining_usd", remaining)
    prior = {p: p.read_bytes() for pattern in ("candidate-*.tex", "review-*.json")
             for p in ctx.root_workdir.rglob(pattern)}
    assert prior and any(p.name.startswith("review-") for p in prior)
    options["verdicts"] = ["accept"]
    second = asyncio.run(execute(resume_run=True))
    assert second.submission_approved
    current = {p for pattern in ("candidate-*.tex", "review-*.json") for p in ctx.root_workdir.rglob(pattern)}
    new_paths = current - prior.keys()
    assert {p.name for p in prior} <= {p.name for p in new_paths}
    assert all(p.read_bytes() == content for p, content in prior.items())


def test_resume_returns_already_accepted_exact_output_without_calls(pipeline):
    execute, calls, _, _, ctx, _ = pipeline
    first = asyncio.run(execute())
    before = calls.copy()
    resumed = asyncio.run(execute(resume_run=True))
    assert resumed.submission_approved and calls == before
    assert resumed.answer_tex.read_bytes() == first.answer_tex.read_bytes()
    assert _exported_path(ctx) is not None


def test_resumed_checkpoint_overrides_old_terminal_metadata(pipeline, monkeypatch):
    execute, _, _, options, ctx, agent = pipeline
    options["research_error"] = ConnectionError("first run interrupted")
    first = asyncio.run(execute(max_recovery_attempts=0, max_cleanup_handoffs=0))
    assert first.partial_ready
    ctx.write_metadata({"status": "ok", "outputs": first.model_dump(mode="json")})
    options.pop("research_error")
    options["research_document"] = TEX.replace("Proof.", "A later research draft.")

    async def resume_then_interrupt():
        entered = asyncio.Event()

        async def stuck(*args, **kwargs):
            entered.set()
            await asyncio.sleep(100)

        monkeypatch.setattr(agent, "_candidate_loop", stuck)
        task = asyncio.create_task(execute(resume_run=True))
        try:
            await asyncio.wait_for(entered.wait(), 3)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(resume_then_interrupt())
    restored = agent._restore_output(agent.Inputs(problem=ORIGINAL, problem_id="p"))
    assert restored.partial_ready and restored.partial_sha256 != first.partial_sha256
    assert _exported_path(ctx) == restored.answer_tex
    assert "A later research draft." in restored.answer_tex.read_text()
    assert first.answer_tex.exists()


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("failure", ["document_replace", "manifest_write", "manifest_replace", "after_commit"])
def test_interrupted_publication_keeps_a_valid_committed_revision(pipeline, monkeypatch, partial, failure):
    from pathlib import Path
    from proofstack.agents import firstproof_batch3 as module

    _, _, _, _, ctx, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    out = agent.Outputs(problem_id="p")
    agent._publish(inp, out, agent._label_partial(TEX), 1, partial=True)
    original = out.model_copy(deep=True)
    original_text = original.answer_tex.read_text()
    assert _exported_path(ctx) == original.answer_tex

    class InterruptedPublication(BaseException):
        pass

    save_json, replace_file = module._save_json, Path.replace

    def interrupted_replace(path, target):
        if ((failure == "manifest_replace" and path.name == "batch3-output.tmp")
                or (failure == "document_replace" and path.name != "batch3-output.tmp")):
            raise InterruptedPublication
        return replace_file(path, target)

    def interrupted_save(path, value):
        if failure == "manifest_write":
            path.with_suffix(".tmp").write_text('{"unfinished":')
            raise InterruptedPublication
        save_json(path, value)
        if failure == "after_commit":
            raise InterruptedPublication

    monkeypatch.setattr(Path, "replace", interrupted_replace)
    monkeypatch.setattr(module, "_save_json", interrupted_save)
    revised = TEX.replace("Proof.", "Later proof.")
    if partial:
        revised = agent._label_partial(revised)
    with pytest.raises(InterruptedPublication):
        agent._publish(inp, out, revised, 1, partial=partial)
    assert out == original
    assert original.answer_tex.read_text() == original_text
    restored = agent._restore_output(inp)
    expected_text = revised if failure == "after_commit" else original_text
    assert restored.answer_tex.read_text() == expected_text
    assert _exported_path(ctx) == restored.answer_tex
    assert restored.submission_approved == (failure == "after_commit" and not partial)


def test_publication_io_failure_does_not_mutate_fallback_output(pipeline, monkeypatch):
    from proofstack.agents import firstproof_batch3 as module

    _, _, _, _, ctx, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    out = agent.Outputs(problem_id="p")
    agent._publish(inp, out, agent._label_partial(TEX), 1, partial=True)
    original = out.model_copy(deep=True)

    def fail(*args):
        raise OSError("checkpoint write failed")

    monkeypatch.setattr(module, "_save_json", fail)
    asyncio.run(agent._fallback(inp, out, TEX.replace("Proof.", "Later proof."), deadline=time.monotonic() + 60))
    assert out.answer_tex == original.answer_tex and out.partial_sha256 == original.partial_sha256
    assert "checkpoint write failed" in out.cleanup_errors[-1]
    assert _exported_path(ctx) == original.answer_tex


@pytest.mark.parametrize("partial", [False, True])
def test_versioned_publication_restores_in_a_copied_run_and_rejects_tampering(pipeline, tmp_path, partial):
    _, _, _, _, ctx, agent = pipeline
    inp = agent.Inputs(problem=ORIGINAL, problem_id="p")
    out = agent.Outputs(problem_id="p")
    agent._publish(inp, out, TEX, 1, partial=partial)
    copied = tmp_path / "copied-run"
    shutil.copytree(ctx.root_workdir, copied)
    resumed = FirstProofBatch3Workflow(RunContext.create(root_workdir=copied, flat=True))
    restored = resumed._restore_output(inp)
    assert restored.answer_tex.is_relative_to(copied)
    assert restored.answer_tex.read_text() == TEX
    out.answer_tex.write_text(TEX.replace("Proof.", "Changed since review."))
    assert _exported_path(ctx) is None
    assert agent._restore_output(inp).abstained


@pytest.mark.parametrize("cost", [100, 101])
def test_completed_partial_rewrite_survives_final_usd_charge(pipeline, monkeypatch, cost):
    from proofstack.kinds.api_call import APICallAgent

    execute, calls, _, _, ctx, agent = pipeline
    workspace = agent._workspace(agent.Inputs(problem=ORIGINAL, problem_id="p"))
    workspace.mkdir(parents=True)
    (workspace / "answer.tex").write_text(TEX)
    revised = TEX.replace("Proof.", "A clearer partial argument, with its remaining gap explicitly stated.")
    queries = []

    class Client:
        model = "offline-rewrite"
        timeout = max_wallclock_per_call_s = 100

        def run_queries(self, conversations, **kwargs):
            queries.append(conversations)
            yield 0, [*conversations[0], {"role": "assistant", "content": revised}], {"cost": cost}

    ctx.api_client_factory = lambda _: Client()
    monkeypatch.setattr(RewriteSeat, "run", APICallAgent.run)
    out = asyncio.run(execute(research_seconds=0))
    assert out.partial_ready and not out.submission_approved
    assert "A clearer partial argument" in out.answer_tex.read_text()
    assert ctx.budgets.root().counters.usd == cost
    assert len(queries) == 1 and calls == {"compile": 2}
    assert _exported_path(ctx) == out.answer_tex


@pytest.mark.parametrize("reply,pages", [("UNABLE: missing argument", 1), ("incomplete", 1), (TEX, 17), (TEX, 0)])
def test_over_budget_partial_reply_still_passes_normal_gates(pipeline, monkeypatch, reply, pages):
    execute, calls, _, options, _, agent = pipeline
    workspace = agent._workspace(agent.Inputs(problem=ORIGINAL, problem_id="p"))
    workspace.mkdir(parents=True)
    (workspace / "answer.tex").write_text(TEX)

    async def over_budget(self, inp):
        self.tracker.add_usd(101)
        options["pages"] = pages
        exc = BudgetExhausted("run", "usd", 100, 101)
        exc.completed_output = self.Outputs(text=reply)
        raise exc

    monkeypatch.setattr(RewriteSeat, "run", over_budget)
    out = asyncio.run(execute(research_seconds=0))
    assert out.partial_ready and not out.submission_approved and out.cleanup_errors
    assert out.answer_tex.read_text() == agent._label_partial(TEX)
    assert calls["review"] == 0


@pytest.mark.parametrize("partial,limit_kind,completed", [(False, "usd", True), (True, "wallclock_s", True), (True, "usd", False)])
def test_reply_recovery_is_limited_to_completed_usd_exhausted_partials(pipeline, monkeypatch, partial, limit_kind, completed):
    _, _, _, _, ctx, agent = pipeline
    exc = BudgetExhausted("run", limit_kind, 100, 101)
    if completed:
        exc.completed_output = RewriteSeat.Outputs(text=TEX)

    async def fail(self, inp):
        raise exc

    monkeypatch.setattr(RewriteSeat, "run", fail)
    with pytest.raises(BudgetExhausted) as caught:
        inp = agent.Inputs(problem=ORIGINAL, problem_id="p", cleanup_backend="api")
        asyncio.run(agent._edit(ctx, inp, TEX, name="rewrite", partial=partial))
    assert caught.value is exc


def test_resume_restores_paid_cost_before_sizing_research_phase(pipeline):
    from proofstack.agents.firstproof_batch3 import _save_json

    execute, _, seen, _, ctx, _ = pipeline
    now = time.time()
    _save_json(ctx.root_workdir / "batch3-schedule.json", {
        "problem_hash": _problem_hash(ORIGINAL), "research_deadline_unix_s": now + 400,
        "run_deadline_unix_s": now + 500, "initial_usd": 100, "partial_reserve_usd": 15,
    })
    (ctx.root_workdir / "events.jsonl").write_text(json.dumps({"kind": "model.call", "payload": {"cost_usd": 40}}) + "\n")
    out = asyncio.run(execute(resume_run=True))
    assert out.submission_approved
    assert seen["research_caps"] == [45]
    assert ctx.budgets.root().counters.usd == 67


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_failed_compile_includes_actionable_diagnostics(tmp_path):
    from proofstack.agents.writeup_loop import _compile_raw

    bad = TEX.replace("Proof.", r"\DefinitelyUndefinedCommand")
    compiled, pages, detail = _compile_raw(bad, None, deadline=time.monotonic() + 30)
    assert not compiled and pages == 0
    assert "Undefined control sequence" in detail and "DefinitelyUndefinedCommand" in detail


def test_targeted_repair_receives_compile_diagnostics(pipeline, monkeypatch):
    execute, _, seen, options, _, agent = pipeline
    options["verdicts"] = ["repair", "accept"]
    compile_document = agent._compile
    checked = 0

    async def fail_candidate_once(document, *, deadline):
        nonlocal checked
        checked += 1
        if checked == 3:
            return False, 0, "Undefined control sequence: \\MyBrokenMacro"
        return await compile_document(document, deadline=deadline)

    monkeypatch.setattr(agent, "_compile", fail_candidate_once)
    out = asyncio.run(execute())
    assert out.submission_approved
    assert "Undefined control sequence: \\MyBrokenMacro" in seen["prompts"][1]


@pytest.mark.skipif(not shutil.which("pdflatex") or not shutil.which("bibtex"), reason="TeX tools not installed")
def test_real_bibliography_fallback_compiles_and_exports(pipeline, monkeypatch):
    execute, _, _, options, ctx, _ = pipeline
    # Restore the real deterministic gate for this end-to-end bibliography test.
    async def compile_real(self, document, *, deadline):
        from proofstack.agents.writeup_loop import _compile_raw
        return await asyncio.to_thread(_compile_raw, document, None, deadline=deadline)

    monkeypatch.setattr(FirstProofBatch3Workflow, "_compile", compile_real)
    options.update(
        research_document=TEX.replace("Proof.", r"Proof \cite{ref}.\bibliographystyle{plain}\bibliography{references}"),
        bibliography="@book{ref, author={A. Writer}, title={Result}, year={2020}, publisher={Publisher}}",
        research_error=TimeoutError("cutoff"), partial_error=RuntimeError("provider down"),
    )
    out = asyncio.run(execute(max_recovery_attempts=0, max_cleanup_handoffs=0))
    assert out.partial_ready and out.compiled and out.pages == 1, out.model_dump()
    assert r"\bibitem{ref}" in out.answer_tex.read_text()
    assert _exported_path(ctx) is not None
