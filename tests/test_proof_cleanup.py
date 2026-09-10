from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.dev_data import validate_preset_yaml
from proofstack.agents.ac.ac_workflow import ACDAGWorkflow, ACWorkflow
from proofstack.agents.ac.critic import ACCritic
from proofstack.agents.ac.visual_blocks import ACInitBlock
from proofstack.agents.configurable_prompt import ConfigurablePromptAgent
from proofstack.agents.proof_cleanup import ProofCleanupWorkflow
from proofstack.budget import BudgetExhausted, SubscriptionParked
from proofstack.context import RunContext
from proofstack.registry import load_preset


@pytest.fixture
def offline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "proofstack.sandbox.subprocess._terminate_marked_processes",
        AsyncMock(return_value=True),
    )
    calls = Counter()
    failures = {}
    accepted = {"value": True}
    original = tmp_path / "solutions" / "p.tex"
    original.parent.mkdir()
    original.write_text("ORIGINAL PROOF", encoding="utf-8")
    bib = tmp_path / "references.bib"
    bib.write_text("@book{a,title={Original}}", encoding="utf-8")

    async def no_network(self):
        raise AssertionError("Tests must not create an API client")

    async def editor(self, inp):
        calls[self.name] += 1
        self.render_messages(inp)
        if self.name in failures:
            raise failures[self.name]
        return self.parse_output(
            f"<answer_tex>{inp.answer_tex} EDITED</answer_tex>"
            f"<references_bib>{inp.references_bib}</references_bib>", inp,
        )

    async def critic(self, inp):
        calls["fresh_critic"] += 1
        self.render_messages(inp)
        if "fresh_critic" in failures:
            raise failures["fresh_critic"]
        return self.Outputs(answer_ready=accepted["value"], review_md="Reviewed")

    async def author(self, inp):
        calls["author_critic"] += 1
        return self.Outputs(
            problem_id=inp.problem_id, answer_tex=original,
            references_bib=bib, research_notes_tex=bib,
            compiled=True, pages=1, rounds_completed=2,
            last_critic_accepted=True, final_critic_mode_run="not_run",
            last_gasp="author_critic" in failures,
            error=str(failures["author_critic"]) if "author_critic" in failures else None,
        )

    monkeypatch.setattr(ConfigurablePromptAgent, "_get_client", no_network)
    monkeypatch.setattr(ConfigurablePromptAgent, "run", editor)
    monkeypatch.setattr(ACCritic, "run", critic)
    monkeypatch.setattr(ACDAGWorkflow, "run", author)

    def execute(name="proof_cleanup", **overrides):
        preset = load_preset(name)
        ctx = RunContext.create(
            run_id="test", root_workdir=tmp_path, flat=True,
            component_configs=preset.component_configs,
        )
        inputs = preset.build_inputs(problem="Prove P", problem_id="p", cli_overrides={
            "answer_tex_path": original, "references_bib_path": bib, **overrides,
        })
        return asyncio.run(preset.workflow_cls(ctx)(**inputs)).model_dump(mode="json")

    return execute, calls, failures, original, bib, accepted


@pytest.mark.parametrize("name", ["proof_cleanup", "author_critic_cleanup"])
def test_presets_validate(name):
    report = validate_preset_yaml(load_preset(name).source_path.read_text())
    assert report["ok"], report["errors"]


def test_file_changes_invalidate_editors_but_unchanged_content_is_reused(offline):
    execute, calls, _, original, bib, _ = offline
    first = execute()
    assert first["error"] is None
    assert first["status"] == "done"
    assert first["answer_tex"] == "ORIGINAL PROOF EDITED EDITED"
    assert original.read_text() == "ORIGINAL PROOF"
    Path(first["answer_tex_path"]).unlink()
    repeated = execute()
    assert Path(repeated["answer_tex_path"]).read_text() == first["answer_tex"]
    assert calls == {"scholarship_audit_editor": 1, "writing_editor": 1}

    original.write_text("UPDATED PROOF")
    updated = execute()
    assert updated["answer_tex"] == "UPDATED PROOF EDITED EDITED"
    assert calls == {"scholarship_audit_editor": 2, "writing_editor": 2}

    bib.write_text("@book{b,title={Updated}}")
    assert execute()["references_bib"] == bib.read_text()
    assert calls == {"scholarship_audit_editor": 3, "writing_editor": 3}


@pytest.mark.parametrize("field", ["answer_tex_path", "references_bib_path"])
def test_missing_file_can_be_repaired_and_retried(offline, tmp_path, field):
    execute, calls, _, original, _, _ = offline
    missing = tmp_path / "missing"
    result = execute(**{field: missing})
    assert result["status"] == "error"
    assert "load" in result["error"]
    assert not calls
    assert original.read_text() == "ORIGINAL PROOF"
    missing.write_text("REPAIRED INPUT")
    result = execute(**{field: missing})
    assert result["error"] is None
    assert result["status"] == "done"
    assert calls == {"scholarship_audit_editor": 1, "writing_editor": 1}


def test_inline_proof_and_empty_bibliography(offline):
    execute, _, _, _, _, _ = offline
    result = execute(answer_tex_path="", references_bib_path="", answer_tex="INLINE", references_bib="")
    assert result["answer_tex"] == "INLINE EDITED EDITED"
    assert result["references_bib"] == ""
    assert result["status"] == "done"


@pytest.mark.parametrize("stage", ["scholarship_audit_editor", "writing_editor", "fresh_critic"])
def test_editor_and_critic_errors_are_resumable_without_repeating_successful_edits(offline, stage):
    execute, calls, failures, original, _, _ = offline
    failures[stage] = RuntimeError("temporary provider outage")
    failed = execute("author_critic_cleanup")
    assert "temporary provider outage" in failed["error"]
    assert failed["final_critic_answer_ready"] is False
    assert original.read_text() == "ORIGINAL PROOF"
    assert failed["original_proof"]["answer_tex"] == str(original)
    if stage == "fresh_critic":
        assert Path(failed["cleaned_proof"]["answer_tex"]).exists()

    del failures[stage]
    result = execute("author_critic_cleanup", resume_run=True)
    assert result["error"] is None
    assert result["final_critic_answer_ready"] is True
    assert calls[stage] == 2
    for earlier in ["scholarship_audit_editor", "writing_editor"]:
        if earlier != stage:
            assert calls[earlier] == 1
    assert original.read_text() == "ORIGINAL PROOF"


@pytest.mark.parametrize("stage", ["scholarship_audit_editor", "fresh_critic"])
def test_budget_exhaustion_keeps_existing_proof_and_error(offline, stage):
    execute, _, failures, original, _, _ = offline
    failures[stage] = BudgetExhausted("run", "usd", 10, 11)
    result = execute("author_critic_cleanup")
    assert "BudgetExhausted" in result["error"]
    assert result["final_critic_answer_ready"] is False
    assert original.read_text() == "ORIGINAL PROOF"


def test_park_is_not_converted_to_a_completed_error(offline):
    execute, _, failures, _, _, _ = offline
    failures["scholarship_audit_editor"] = SubscriptionParked("daily", 60, 10, 10)
    with pytest.raises(SubscriptionParked):
        execute("author_critic_cleanup")


def test_critic_rejection_is_not_a_provider_error(offline):
    execute, _, _, original, _, accepted = offline
    accepted["value"] = False
    result = execute("author_critic_cleanup")
    assert result["final_critic_answer_ready"] is False
    assert result["error"] is None
    assert original.read_text() == "ORIGINAL PROOF"


def _runner():
    spec = importlib.util.spec_from_file_location("cleanup_run_workflow", ROOT / "scripts/run_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_marks_upstream_failure_as_error(offline, tmp_path):
    _, _, failures, _, _, _ = offline
    failures["author_critic"] = RuntimeError("Author failed")
    argv = ["run_workflow.py", "--workflow", "author_critic_cleanup",
            "--restart-from", str(tmp_path), "--problem-text", "Prove P", "--problem-id", "p"]
    with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(_runner().amain())
    metadata = json.loads((tmp_path / "run-metadata.json").read_text())
    assert metadata["status"] == "error"
    assert metadata["outputs"]["error"] == "Author failed"


def test_cli_restart_restores_round_and_spend_without_overriding_ac_defaults(tmp_path):
    ctx = RunContext.create(run_id="old", root_workdir=tmp_path, flat=True)
    helper = ACWorkflow(ctx)
    workspace = helper._workspace_path("p", "Prove P")
    asyncio.run(helper._init_workspace(workspace, problem_text="Prove P"))
    for filename, text in [("answer.tex", "CHECKPOINT PROOF"), ("references.bib", ""), ("research_notes.tex", "NOTES")]:
        (workspace / filename).write_text(text)
    helper._save_resume_state(
        workspace, inp=ACWorkflow.Inputs(problem="Prove P", problem_id="p", n_rounds=10),
        last_round_run=2, next_round=3, review_history=[], critic_conversation=[],
        critic_instance_turn=0, pending_council_text="", pending_compute_text="",
        pending_compute_zip_path=None, pending_critique="", early_stopped=False,
    )
    asyncio.run(ctx.events.emit("model.call", {"cost_usd": 7.5}))
    observed = {}

    async def author(self, inp):
        observed["resume_run"] = inp.resume_run
        observed["n_rounds"] = inp.n_rounds
        state = (await ACInitBlock(self.ctx)(**inp.model_dump())).state
        observed["next_round"] = state["next_round"]
        observed["spend"] = self.ctx.budgets.root("run").counters.usd
        observed["workspace"] = state["workspace"]
        return self.Outputs(
            problem_id="p", answer_tex=workspace / "answer.tex",
            references_bib=workspace / "references.bib", research_notes_tex=workspace / "research_notes.tex",
            last_critic_accepted=False,
        )

    argv = ["run_workflow.py", "--workflow", "author_critic_cleanup",
            "--restart-from", str(tmp_path), "--problem-text", "Prove P", "--problem-id", "p"]
    with patch.object(sys, "argv", argv), patch.object(ACDAGWorkflow, "run", author), contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(_runner().amain())
    assert observed == {"resume_run": True, "n_rounds": 10, "next_round": 3,
                        "spend": 7.5, "workspace": str(workspace)}


def test_cleanup_drops_restart_flags_from_editor_inputs():
    inputs = ProofCleanupWorkflow.Inputs(problem="P", problem_id="p", resume_run=True, n_rounds=50)
    assert "resume_run" not in inputs.model_dump()
    assert "n_rounds" not in inputs.model_dump()
