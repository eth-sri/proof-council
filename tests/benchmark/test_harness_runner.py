from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from proofstack.evaluation import harness_runner
from proofstack.evaluation.harness_runner import (
    HarnessSettings, _read_proof, run_harness,
)


CHILD_SCRIPT = '''
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
for name in ["workflow", "problem", "problem-id", "run-id", "output", "budget-usd", "input"]:
    p.add_argument("--" + name)
args = p.parse_args()
directory = Path(args.output) / args.run_id
proof = directory / "answer.tex"
proof.write_text("A synthetic proof.")
(directory / "received.json").write_text(json.dumps(vars(args)))
(directory / "events.jsonl").write_text(json.dumps({
    "kind": "model.call", "call_id": "synthetic", "payload": {
        "cost_usd": 0.25, "in_tokens": 5, "out_tokens": 7}}) + "\\n")
(directory / "run-metadata.json").write_text(json.dumps({
    "status": "ok", "outputs": {"answer_tex": str(proof)}}))
'''


@pytest.fixture
def settings(tmp_path):
    root = tmp_path / "repository"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/run_workflow.py").write_text(CHILD_SCRIPT)
    workflow = root / "workflow.yaml"
    workflow.write_text("Synthetic workflow fixture")
    return HarnessSettings(root, workflow, tmp_path / "benchmark", 1.5, 5)


def record():
    return {"problem_id": "../../unsafe/id", "problem": "A synthetic question.",
            "reference_solutions": ["Must never reach the solver."]}


def test_real_subprocess_passes_only_statement_and_collects_proof(settings):
    result = run_harness(settings, record())
    assert result["status"] == "solved"
    assert result["solution"] == "A synthetic proof."
    assert result["problem_id"] == record()["problem_id"]
    assert result["cost_usd"] == 0.25
    assert result["tokens"] == 12
    assert result["accounting_complete"] is True
    directory = Path(result["run_directory"])
    assert directory.is_relative_to(settings.output)
    received = json.loads((directory / "received.json").read_text())
    assert received["problem_id"] == result["execution_id"]
    assert "/" not in received["problem_id"]
    assert received["budget_usd"] == "1.5"
    assert received["input"] == "max_wallclock_s=5"
    assert Path(received["problem"]).read_text() == record()["problem"]
    assert "reference_solutions" not in json.dumps(received)
    with pytest.raises(FileExistsError):
        run_harness(settings, record())


@pytest.mark.parametrize("mutation", [
    '\nraise SystemExit(1)\n',
    '\n(directory / "run-metadata.json").write_text("bad JSON")\n',
    '\n(directory / "run-metadata.json").unlink()\n',
    '\n(directory / "run-metadata.json").write_text(json.dumps({"status": "parked"}))\n',
])
def test_failure_never_submits_saved_proof(settings, mutation):
    script = settings.repository_root / "scripts/run_workflow.py"
    script.write_text(CHILD_SCRIPT + mutation)
    result = run_harness(settings, record())
    assert result["status"] == "solver_error"
    assert result["solution"] is None
    assert result["accounting_complete"] is False
    assert result["cost_usd"] == 0.25


def test_missing_declared_file_is_not_graded_as_text(settings):
    script = settings.repository_root / "scripts/run_workflow.py"
    script.write_text(CHILD_SCRIPT + '\nproof.unlink()\n')
    result = run_harness(settings, record())
    assert result["status"] == "missing_proof"
    assert result["solution"] is None
    assert result["accounting_complete"] is True


def test_inline_and_custom_field_answers(settings):
    script = settings.repository_root / "scripts/run_workflow.py"
    script.write_text(CHILD_SCRIPT + '''
(directory / "run-metadata.json").write_text(json.dumps({
    "status": "ok", "outputs": {"final": {"proof": "A complete synthetic argument."}}}))
''')
    result = run_harness(replace(settings, answer_field="final.proof"), record())
    assert result["solution"] == "A complete synthetic argument."
    assert result["answer_field"] == "final.proof"


def test_proof_reader_rejects_paths_and_symlinks_outside_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    outside = tmp_path / "secret.tex"
    outside.write_text("Outside the run")
    (run / "link.tex").symlink_to(outside)
    for value in (str(outside), "../secret.tex", "link.tex"):
        with pytest.raises(ValueError, match="contained"):
            _read_proof(value, run)
    assert _read_proof("Proof of x/y = z.", run) == "Proof of x/y = z."


def test_proof_reader_detects_files_and_rejects_missing_paths(tmp_path):
    (tmp_path / "answer").write_text("A complete synthetic proof.")
    assert _read_proof("answer", tmp_path) == "A complete synthetic proof."
    with pytest.raises(FileNotFoundError):
        _read_proof("missing.tex", tmp_path)


def test_timeout_terminates_child_and_records_partial_usage(settings):
    script = settings.repository_root / "scripts/run_workflow.py"
    script.write_text(CHILD_SCRIPT + '\nimport time\ntime.sleep(30)\n')
    result = run_harness(replace(settings, timeout_seconds=0.3), record())
    assert result["status"] == "timeout"
    assert result["timed_out"] is True
    assert result["solution"] is None
    assert result["accounting_complete"] is False
    assert result["runtime_seconds"] < 10


def test_interrupted_attempt_keeps_usage_and_runtime(settings, monkeypatch):
    def interrupt(command, log_path, **kwargs):
        directory = log_path.parent
        (directory / "events.jsonl").write_text(json.dumps({
            "kind": "model.call", "payload": {"cost_usd": 0.5, "in_tokens": 10}}))
        (directory / "run-metadata.json").write_text(json.dumps({
            "status": "ok", "outputs": {"solution": "A saved synthetic proof."}}))
        raise KeyboardInterrupt

    monkeypatch.setattr(harness_runner, "run_process", interrupt)
    result = run_harness(settings, record())
    assert result["status"] == "interrupted"
    assert result["error"] == "Interrupted"
    assert result["solution"] is None
    assert result["returncode"] is None
    assert result["runtime_seconds"] >= 0
    assert result["cost_usd"] == 0.5
    assert result["tokens"] == 10
    assert result["accounting_complete"] is False


def test_sigterm_stops_solver_and_checkpoints_attempt(settings, tmp_path):
    repository = Path(__file__).resolve().parents[2]
    child_script = settings.repository_root / "scripts/run_workflow.py"
    child_script.write_text(CHILD_SCRIPT + '''
import os
import time
(directory / "child.pid").write_text(str(os.getpid()))
time.sleep(30)
''')
    parent_script = tmp_path / "benchmark_parent.py"
    parent_script.write_text(f'''
import sys
sys.path[:0] = [{str(repository)!r}, {str(repository / 'src')!r}]
from pathlib import Path
from scripts import benchmark_harness
benchmark_harness.REPOSITORY_ROOT = Path({str(settings.repository_root)!r})
def prepare(args):
    args.output = Path({str(settings.output)!r})
    args.solver_timeout_seconds = 30
    args.output.mkdir()
    return [{{"problem_id": "synthetic", "problem": "Synthetic statement"}}], {{}}
benchmark_harness.prepare_benchmark = prepare
raise SystemExit(benchmark_harness.main([
    "--workflow", {str(settings.workflow)!r},
    "--grader-model", "fake", "--grader-budget-usd", "1",
    "--solver-budget-usd-per-problem", "1",
]))
''')
    parent = subprocess.Popen([sys.executable, str(parent_script)], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    child_pid = None
    try:
        deadline = time.monotonic() + 15
        while not list(settings.output.glob("solver-runs/*/child.pid")):
            assert parent.poll() is None, "Benchmark parent exited before launching its solver"
            assert time.monotonic() < deadline, "Solver did not start"
            time.sleep(0.05)
        child_pid = int(next(settings.output.glob("solver-runs/*/child.pid")).read_text())
        parent.terminate()
        assert parent.wait(timeout=10) == 130
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        summary = json.loads((settings.output / "summary.json").read_text())
        assert summary["solver_statuses"] == {"interrupted": 1}
        assert summary["usage"]["recorded_total_cost_usd"] == 0.25
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan"), True])
def test_invalid_budget_is_rejected_before_launch(settings, budget):
    with pytest.raises(ValueError, match="budget_usd"):
        run_harness(replace(settings, budget_usd=budget), record())
    assert not settings.output.exists()
