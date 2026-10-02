"""Exercise the CLI result boundary without starting any model calls."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow
from proofstack.budget import SubscriptionParked


@pytest.mark.parametrize("outcome,expected_code,expected_status", [
    ("accepted", 0, "ok"),
    ("partial", 0, "ok"),
    ("fatal", 1, "error"),
    ("retryable_error", 1, "error"),
    ("raised_error", 1, "error"),
    ("parked", 2, "parked"),
])
def test_cli_exit_code_matches_persisted_status(
    tmp_path, monkeypatch, capsys, outcome, expected_code, expected_status,
):
    script = Path(__file__).resolve().parents[1] / "scripts/run_workflow.py"
    spec = importlib.util.spec_from_file_location("_run_workflow_exit_status_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_id = "offline-exit-status"
    run_dir = tmp_path / run_id

    async def offline_run(self, inp):
        if outcome == "parked":
            raise SubscriptionParked("weekly", 3600, 100, 100)
        if outcome == "raised_error":
            raise RuntimeError("offline failure")
        answer = self.ctx.root_workdir / "saved-draft.tex"
        answer.write_text("Saved draft", encoding="utf-8")
        return self.Outputs(
            problem_id=inp.problem_id, answer_tex=answer,
            compiled=True, pages=1, abstained=False,
            output_kind="accepted_solution" if outcome == "accepted" else "partial_unreviewed",
            submission_approved=outcome == "accepted",
            partial_ready=outcome != "accepted",
            error="offline failure" if outcome in {"fatal", "retryable_error"} else None,
            error_retryable=outcome != "fatal",
        )

    monkeypatch.setattr(FirstProofBatch3Workflow, "run", offline_run)
    monkeypatch.setattr(sys, "argv", [
        str(script), "--workflow", "firstproof_batch3", "--problem-text", "Offline test",
        "--problem-id", "offline", "--run-id", run_id, "--output", str(tmp_path),
    ])

    code = asyncio.run(module.amain())

    assert code == expected_code
    metadata = json.loads((run_dir / "run-metadata.json").read_text())
    assert metadata["status"] == expected_status
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [event["payload"]["status"] for event in events if event["kind"] == "run.end"] == [expected_status]
    assert not any(event["kind"].startswith("model.call") for event in events)
    assert not (run_dir / "run.pid").exists()
    if outcome not in {"parked", "raised_error"}:
        assert (run_dir / "saved-draft.tex").read_text() == "Saved draft"
        assert metadata["outputs"]["error_retryable"] == (outcome != "fatal")
        assert '"answer_tex":' in capsys.readouterr().out
