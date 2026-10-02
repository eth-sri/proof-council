import asyncio
import json
import os
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from proofstack.agents.ac.author import Author
from proofstack.agents.ac.multi_author import MultiAuthor, SubAuthorSeat
from proofstack.agents.ac.delegation_recovery import has_research_checkpoint, problem_key
from proofstack.budget import BudgetSpec
from proofstack.context import RunContext


def lead_at(root, **inputs):
    ctx = RunContext.create(root_workdir=root, flat=True, run_budget=BudgetSpec(max_wallclock_s=80000),
                            component_configs={"Author": {"delegation": {"enabled": True}}})
    lead = MultiAuthor(ctx, name="Author")
    lead._fallback_workdir = root / "agents" / ("Author-" + uuid.uuid4().hex)
    lead._fallback_workdir.mkdir(parents=True)
    inp = Author.Inputs(problem="P", recovery_problem="P", round=0, n_rounds=3, **inputs)
    lead._render_container_messages(inp, "")
    return lead


def test_new_invocation_recovers_reports_and_consumed_waves(tmp_path):
    async def fake(self, inp):
        self.context_view.publish("verifier.cpp", "int main() { return 0; }")
        return self.parse_output("Lemma L proved, subject to checking.", inp)

    async def scenario():
        first = lead_at(tmp_path)
        with patch.object(SubAuthorSeat, "run", fake):
            await asyncio.to_thread(first._delegate, [{"role": "prover", "task": "Prove L"}])
        assert has_research_checkpoint(tmp_path, "P")
        second = lead_at(tmp_path)
        assert second.workdir != first.workdir
        assert second._recovered and second._waves_done == 1
        assert second._agent_counter == 1
        assert second._seats["prover1"]["messages_after"] == []
        assert "helpers/wave1-prover1/verifier.cpp" in second._context.files
        assert "Lemma L proved" in second._context.files[second._seats["prover1"]["report_path"]]
        assert "RECOVERED RESEARCH" in second._with_guide([{"role": "system", "content": "x"}])[-1]["content"]
        with patch.object(SubAuthorSeat, "run", fake):
            await asyncio.to_thread(second._delegate, [{"role": "checker", "task": "Check L", "depends_on": ["prover1"]}])
        third = lead_at(tmp_path)
        assert third._waves_done == 2
        assert "No delegation waves left" in third._delegate([{"role": "prover", "task": "Again"}])

    asyncio.run(scenario())


def test_corrupt_artifact_blocks_recovery_before_new_work(tmp_path):
    async def scenario():
        first = lead_at(tmp_path)
        async def fake(self, inp):
            return self.parse_output("proof", inp)
        with patch.object(SubAuthorSeat, "run", fake):
            await asyncio.to_thread(first._delegate, [{"role": "prover", "task": "L"}])
        report = first._context.root / first._seats["prover1"]["report_path"]
        report.write_text("changed")
        assert not has_research_checkpoint(tmp_path, "P")
        with pytest.raises(ValueError, match="digest mismatch"):
            lead_at(tmp_path)
    asyncio.run(scenario())


def test_continuation_notes_do_not_hide_canonical_problem_recovery(tmp_path):
    async def scenario():
        first = lead_at(tmp_path)
        async def fake(self, inp):
            return self.parse_output("earlier work", inp)
        with patch.object(SubAuthorSeat, "run", fake):
            await asyncio.to_thread(first._delegate, [{"role": "prover", "task": "L"}])
        second = MultiAuthor(first.ctx, name="Author")
        second._render_container_messages(Author.Inputs(problem="P\nAdditional instructions: continue",
                                            recovery_problem="P", round=0, n_rounds=3), "")
        assert second._recovered
        assert "Additional instructions" in second._context.files["round/problem.txt"]
        checkpoint = tmp_path / "helper-recovery" / problem_key("P") / "round-0.json"
        old_checkpoint = checkpoint.read_bytes()
        changed = lead_at(tmp_path, answer_tex="different baseline")
        assert not changed._recovered
        assert changed._waves_done == 0 and changed._agent_counter == 0
        assert not changed._seats and not changed._delegation_log
        assert not any(p.startswith("helpers/") for p in changed._context.files)
        assert changed._context.files["round/answer.tex"] == "different baseline"
        assert checkpoint.read_bytes() == old_checkpoint
        with patch.object(SubAuthorSeat, "run", fake):
            await asyncio.to_thread(changed._delegate, [{"role": "prover", "task": "New baseline"}])
        resumed = lead_at(tmp_path, answer_tex="different baseline")
        assert resumed._recovered and resumed._waves_done == 1
    asyncio.run(scenario())


def test_interrupted_wave_keeps_partial_reports_and_does_not_reset_allowance(tmp_path):
    async def fake(self, inp):
        self.context_view.publish("progress.md", "Checked finite cases; full theorem remains open.")
        await asyncio.sleep(60)

    async def scenario():
        first = lead_at(tmp_path)
        first.delegation = {"enabled": True, "job_timeout_s": .05}
        with patch.object(SubAuthorSeat, "run", fake):
            text = await asyncio.to_thread(first._delegate, [{"role": "prover", "task": "L"}])
        assert "Checked finite cases" in text and "deadline" in text
        second = lead_at(tmp_path)
        assert second._waves_done == 1
        assert any("progress.md" in p for p in second._context.files)
        saved = json.loads((tmp_path / "helper-recovery" / problem_key("P") / "round-0.json").read_text())
        assert saved["waves_done"] == 1
    asyncio.run(scenario())


def test_recovery_rejects_symlinked_artifacts(tmp_path):
    async def scenario():
        first = lead_at(tmp_path)
        async def fake(self, inp):
            return self.parse_output("proof", inp)
        with patch.object(SubAuthorSeat, "run", fake):
            await asyncio.to_thread(first._delegate, [{"role": "prover", "task": "L"}])
        report = first._context.root / first._seats["prover1"]["report_path"]
        report.unlink()
        report.symlink_to(tmp_path / "outside.txt")
        (tmp_path / "outside.txt").write_text("proof")
        with pytest.raises(ValueError, match="symlink"):
            lead_at(tmp_path)
    asyncio.run(scenario())


@pytest.mark.parametrize("corruption", ["manifest_symlink", "fifo", "records", "input_hash", "other_problem"])
def test_recovery_rejects_unsafe_manifest_and_bad_records(tmp_path, corruption):
    async def scenario():
        first = lead_at(tmp_path)
        async def fake(self, inp):
            return self.parse_output("proof", inp)
        with patch.object(SubAuthorSeat, "run", fake):
            await asyncio.to_thread(first._delegate, [{"role": "prover", "task": "L"}])
        manifest = tmp_path / "helper-recovery" / problem_key("P") / "round-0.json"
        if corruption == "manifest_symlink":
            target = tmp_path / "copied.json"
            manifest.rename(target)
            manifest.symlink_to(target)
        elif corruption == "fifo":
            artifact = first._context.root / first._seats["prover1"]["report_path"]
            artifact.unlink()
            os.mkfifo(artifact)
        else:
            row = json.loads(manifest.read_text())
            if corruption == "input_hash":
                row.pop("input_hash")
            elif corruption == "other_problem":
                row["problem_hash"] = problem_key("Q")
            else:
                row["log"] = [None]
            manifest.write_text(json.dumps(row))
        assert not has_research_checkpoint(tmp_path, "P")
        with pytest.raises(ValueError):
            lead_at(tmp_path)
    asyncio.run(scenario())


def test_late_completed_report_survives_helper_cancellation(tmp_path):
    from proofstack.kinds.api_call import APICallAgent
    async def interrupted(self, inp):
        exc = asyncio.CancelledError()
        exc.completed_report = "Completed provider proof, not yet critic-reviewed."
        raise exc
    async def scenario():
        lead = lead_at(tmp_path)
        with patch.object(APICallAgent, "run", interrupted):
            with pytest.raises(asyncio.CancelledError):
                await lead._run_wave(1, [{"agent_id": None, "role": "prover", "task": "L", "include_workspace": True}],
                                     "", 60, None)
        assert any("Completed provider proof" in body for body in lead._context.files.values())
    asyncio.run(scenario())


def test_completed_helper_report_survives_cancellation_during_attachment_cleanup(tmp_path):
    from proofstack.kinds.api_call import APICallAgent
    entered, release = threading.Event(), threading.Event()

    def delete(ids):
        entered.set()
        assert release.wait(3)

    async def completed(self, inp):
        self._helper_sandbox = SimpleNamespace(close=lambda: ["file-1"], _delete=delete, failures=[])
        return SubAuthorSeat.Outputs(report="Finished lemma before attachment cleanup.")

    async def scenario():
        lead = lead_at(tmp_path)
        lead.delegation = {"enabled": True, "job_timeout_s": .1}
        try:
            with patch.object(APICallAgent, "run", completed):
                result = await asyncio.to_thread(lead._delegate, [{"role": "prover", "task": "L"}])
            assert entered.is_set()
            assert "Finished lemma before attachment cleanup" in result
            assert "deadline" in result
            resumed = lead_at(tmp_path)
            assert resumed._recovered and resumed._waves_done == 1
            assert any("Finished lemma before attachment cleanup" in text for text in resumed._context.files.values())
        finally:
            release.set()
    asyncio.run(scenario())


def test_first_turn_helper_checkpoint_is_retryable_without_manuscript(tmp_path):
    import sys
    import time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import firstproof_entrypoint as fp
    from proofstack.agents.firstproof_batch3 import _problem_hash
    settings = fp.Settings(input_path=tmp_path / "input.json", output_dir=tmp_path / "output",
        workflow="firstproof_batch3", max_parallel=1, page_limit=16, budget_usd_per_question=10,
        n_rounds=1, round_batch_size=1, compute_codex_sandbox="docker-bypass", runner_script="unused",
        warnings=[], deadline_seconds=300, batch3=True)
    problem = fp.Problem(0, "p", "p", "P", None, tmp_path / "p", tmp_path / "log", tmp_path / "p.tex", "r")
    run = settings.output_dir / "workflow_runs/r"
    async def scenario():
        lead = lead_at(run)
        async def fake(self, inp):
            return self.parse_output("Useful first-turn research", inp)
        with patch.object(SubAuthorSeat, "run", fake):
            await asyncio.to_thread(lead._delegate, [{"role": "prover", "task": "L"}])
        (run / "batch3-schedule.json").write_text(json.dumps({"problem_hash": _problem_hash("P"),
            "research_deadline_unix_s": time.time() + 200, "run_deadline_unix_s": time.time() + 300,
            "initial_usd": 10, "partial_reserve_usd": 1}))
        assert fp._batch3_can_retry(problem, settings, 1)
        settings.stop_requested.set()
        assert not fp._batch3_can_retry(problem, settings, 1)
    asyncio.run(scenario())
