import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts import benchmark_harness
from proofstack.evaluation import proof_evaluation


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    repository = benchmark_harness.REPOSITORY_ROOT
    for name in ("scripts", "src"):
        (tmp_path / name).symlink_to(repository / name, target_is_directory=True)
    configs = tmp_path / "configs/evaluators"
    configs.mkdir(parents=True)
    (configs / "proofgrader.yaml").write_bytes((repository / "configs/evaluators/proofgrader.yaml").read_bytes())
    (configs / "benchmark.yaml").write_text("solver_timeout_seconds: 30\n")
    monkeypatch.setattr(benchmark_harness, "REPOSITORY_ROOT", tmp_path)
    dataset = tmp_path / "opc_problems"
    dataset.mkdir()
    pd.DataFrame([{"problem_id": "synthetic", "problem": "Prove that 1 + 1 = 2."}]).to_parquet(
        dataset / "problems.parquet", index=False)
    pd.DataFrame([{"problem_id": "synthetic", "reference_solutions": ["PRIVATE REFERENCE SENTINEL"],
                   "problem_issues": [], "reference_issues": [], "reference_notes": [],
                   "reference_status": "available"}]).to_parquet(dataset / "reference_solutions.parquet", index=False)
    workflow = tmp_path / "deterministic.yaml"
    workflow.write_text(yaml.safe_dump({
        "workflow": "proofstack.agents.dag_workflow.DAGWorkflow",
        "dag": {"nodes": [{"id": "proof", "kind": "if_else", "inputs": {},
                           "condition": {"python": "True"},
                           "then": {"solution": "Synthetic candidate proof."}}],
                "outputs": {"solution": "$node.proof.solution"}},
    }))
    return dataset, workflow, tmp_path / "outputs/benchmarks"


def arguments(inputs, *extra):
    return ["--workflow", str(inputs[1]), "--limit", "1", *extra]


def run_output(inputs):
    runs = list(inputs[2].iterdir())
    assert len(runs) == 1
    return runs[0]


def forbid_calls(*args, **kwargs):
    pytest.fail("Unexpected model resolution or execution")


def test_dry_run_saves_reproducible_solver_safe_inputs(inputs, monkeypatch):
    monkeypatch.setattr(benchmark_harness, "run_harness", forbid_calls)
    monkeypatch.setattr(benchmark_harness, "run_process", forbid_calls)
    monkeypatch.setattr(proof_evaluation, "resolve_model", forbid_calls)
    monkeypatch.setattr(benchmark_harness, "resolve_model", forbid_calls)
    assert benchmark_harness.main(arguments(inputs, "--dry-run")) == 0
    workflow, output = inputs[1], run_output(inputs)
    assert (output / "workflow.yaml").read_bytes() == workflow.read_bytes()
    plan = json.loads((output / "plan.json").read_text())
    assert plan["selection"]["selected_problem_ids"] == ["synthetic"]
    assert plan["maximum_planned_grading_calls"] == 5
    assert plan["solver_timeout_seconds"] == 30
    assert plan["answer_format"] == "auto"
    assert not (output / "solver-runs").exists()
    assert "PRIVATE REFERENCE" not in (output / "problems.json").read_text()


@pytest.mark.parametrize("flag,value", [("--limit", "0"), ("--samples", "2"),
                                       ("--solver-budget-usd-per-problem", "nan"),
                                       ("--grader-timeout-seconds", "0")])
def test_invalid_parameters_fail_before_output(inputs, flag, value):
    with pytest.raises(SystemExit):
        benchmark_harness.main(arguments(inputs, "--dry-run", flag, value))
    assert not inputs[2].exists()


def test_live_requires_both_budgets_and_grader(inputs):
    with pytest.raises(SystemExit):
        benchmark_harness.main(arguments(inputs))


@pytest.mark.parametrize("removed", [
    ["--dataset-dir", "elsewhere"], ["--output", "elsewhere"],
    ["--include-flagged-references"], ["--solver-timeout-seconds", "30"],
    ["--answer-format", "auto"],
])
def test_removed_options_are_rejected(inputs, removed):
    with pytest.raises(SystemExit) as error:
        benchmark_harness.main(arguments(inputs, "--dry-run", *removed))
    assert error.value.code == 2
    assert not inputs[2].exists()


def test_fixed_paths_do_not_depend_on_working_directory(inputs, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert benchmark_harness.main(arguments(inputs, "--dry-run")) == 0
    assert (run_output(inputs) / "plan.json").is_file()
    assert not (elsewhere / "outputs").exists()


@pytest.mark.parametrize("value", [None, True, 0, -1, "600", float("inf"), float("nan")])
def test_invalid_config_timeout_fails_before_output(inputs, value):
    config = inputs[0].parent / "configs/evaluators/benchmark.yaml"
    config.write_text(yaml.safe_dump({"solver_timeout_seconds": value}))
    assert benchmark_harness.main(arguments(inputs, "--dry-run")) == 2
    assert not inputs[2].exists()


def test_missing_timeout_config_fails_before_output(inputs):
    (inputs[0].parent / "configs/evaluators/benchmark.yaml").unlink()
    assert benchmark_harness.main(arguments(inputs, "--dry-run")) == 2
    assert not inputs[2].exists()


def test_existing_output_is_preserved(inputs):
    assert benchmark_harness.main(arguments(inputs, "--dry-run")) == 0
    output = run_output(inputs)
    sentinel = output / "previous-results.json"
    sentinel.write_text("previous")
    assert benchmark_harness.main(arguments(inputs, "--dry-run")) == 0
    assert len(list(inputs[2].iterdir())) == 2
    assert sentinel.read_text() == "previous"


def test_missing_workflow_is_a_clean_preflight_error(inputs, capsys):
    inputs[1].unlink()
    assert benchmark_harness.main(arguments(inputs, "--dry-run")) == 2
    assert "preset not found" in capsys.readouterr().err
    assert not inputs[2].exists()


def test_invalid_yaml_and_frozen_schemes_fail_preflight(inputs):
    dataset, workflow, output = inputs
    schemes = dataset / "schemes.json"
    schemes.write_text("{}")
    assert benchmark_harness.main(arguments(inputs, "--dry-run", "--marking-schemes", str(schemes))) == 2
    assert not output.exists()
    workflow.write_text("workflow: proofstack.agents.dag_workflow.DAGWorkflow\ndag: {nodes: invalid}")
    assert benchmark_harness.main(arguments(inputs, "--dry-run")) == 2
    assert not output.exists()


class FakeClient:
    model = "offline-test"

    def __init__(self):
        self.messages = []

    def run_queries(self, queries, no_tqdm=False):
        self.messages.extend(queries)
        if len(self.messages) == 1:
            reply = {"usable": True, "marking_scheme": "Synthetic seven-point scheme."}
        else:
            reply = {"score": 7, "rationale": "Synthetic grade.", "errors": []}
        yield 0, [{"role": "assistant", "content": json.dumps(reply)}], {
            "cost": 0.01, "input_tokens": 10, "output_tokens": 10, "reasoning_tokens": 0,
        }


def live_arguments(inputs):
    return arguments(inputs, "--grader-model", "fake", "--solver-budget-usd-per-problem", "0.1",
                     "--grader-budget-usd", "0.1")


def test_yaml_to_real_solver_to_fake_judge_end_to_end(inputs, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(proof_evaluation, "resolve_model", lambda reference: {"model": "fake"})
    monkeypatch.setattr(benchmark_harness, "resolve_model", lambda reference: {"model": "fake"})
    create_context = proof_evaluation.RunContext.create
    monkeypatch.setattr(proof_evaluation.RunContext, "create", staticmethod(
        lambda **kwargs: create_context(**kwargs, api_client_factory=lambda config: client)))

    def run_fake_grading(command, log_path, **kwargs):
        assert command[1:3] == ["-m", "proofstack.evaluation.proof_evaluation"]
        code = proof_evaluation.grade_benchmark(Path(command[3]), benchmark_harness.REPOSITORY_ROOT)
        return {"returncode": code, "timed_out": False, "runtime_seconds": 1.0}

    monkeypatch.setattr(benchmark_harness, "run_process", run_fake_grading)
    assert benchmark_harness.main(live_arguments(inputs)) == 0
    output = run_output(inputs)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["graded_count"] == 1 and summary["full_credit_rate"] == 1
    assert summary["usage"]["recorded_total_cost_usd"] == pytest.approx(0.06)
    assert summary["usage"]["accounting_complete"]
    assert len(client.messages) == 6
    assert "PRIVATE REFERENCE SENTINEL" in client.messages[0][-1]["content"]
    for path in (output / "solver-runs").rglob("*.json*"):
        assert "PRIVATE REFERENCE SENTINEL" not in path.read_text()
    candidate = json.loads((output / "candidates.json").read_text())[0]
    assert candidate["problem_id"] == "synthetic"
    assert candidate["solution"] == "Synthetic candidate proof."


@pytest.mark.parametrize("solver_status,expected_code", [("solver_error", 1), ("interrupted", 130)])
def test_failed_or_interrupted_solver_is_reported_without_grading(inputs, monkeypatch, solver_status, expected_code):
    monkeypatch.setattr(benchmark_harness, "resolve_model", lambda reference: {"model": "fake"})
    monkeypatch.setattr(benchmark_harness, "run_process", forbid_calls)
    def run_solver(settings, record):
        assert settings.timeout_seconds == 30
        return {"problem_id": record["problem_id"], "status": solver_status, "solution": None,
                "cost_usd": 0.01, "tokens": 10, "accounting_complete": False, "runtime_seconds": 2}

    monkeypatch.setattr(benchmark_harness, "run_harness", run_solver)
    assert benchmark_harness.main(live_arguments(inputs)) == expected_code
    summary = json.loads((run_output(inputs) / "summary.json").read_text())
    assert summary["selected_count"] == 1 and summary["full_credit_rate"] == 0
    assert summary["mean_score_of_graded"] is None
    assert summary["solver_statuses"] == {solver_status: 1}


def test_grading_interrupt_recovers_completed_checkpoints(inputs, monkeypatch):
    monkeypatch.setattr(benchmark_harness, "resolve_model", lambda reference: {"model": "fake"})
    args = benchmark_harness.parse_arguments(live_arguments(inputs))
    _, plan = benchmark_harness.prepare_benchmark(args)
    (args.output / "candidates.json").write_text(json.dumps([{"problem_id": "synthetic"}]))
    grading = args.output / "grading"
    grading.mkdir()
    (grading / "results.json").write_text(json.dumps([
        {"problem_id": "synthetic", "candidate_id": "default", "status": "graded", "score": 7}]))

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(benchmark_harness, "run_process", interrupt)
    grades, process = benchmark_harness.run_grading(args, plan)
    assert grades[0]["score"] == 7
    assert process["status"] == "interrupted"
    assert not process["usage"]["accounting_complete"]


def test_worker_does_not_regenerate_missing_frozen_schemes(inputs, monkeypatch):
    monkeypatch.setattr(benchmark_harness, "resolve_model", lambda reference: {"model": "fake"})
    monkeypatch.setattr(proof_evaluation, "resolve_model", forbid_calls)
    args = benchmark_harness.parse_arguments(live_arguments(inputs))
    _, plan = benchmark_harness.prepare_benchmark(args)
    plan["uses_frozen_marking_schemes"] = True
    (args.output / "plan.json").write_text(json.dumps(plan))
    (args.output / "candidates.json").write_text(json.dumps([
        {"problem_id": "synthetic", "solution": "Synthetic proof"}]))
    assert proof_evaluation.grade_benchmark(args.output / "plan.json", inputs[0].parent) == 2
    assert not (args.output / "grading").exists()


@pytest.mark.parametrize("results", [
    {"bad": "format"},
    [{"problem_id": "other"}],
    [{"problem_id": "synthetic", "candidate_id": "default"}],
    [{"problem_id": "synthetic", "candidate_id": "default", "status": "graded", "score": 8}],
])
def test_corrupt_grader_checkpoint_is_reported_as_failure(inputs, monkeypatch, results):
    monkeypatch.setattr(benchmark_harness, "resolve_model", lambda reference: {"model": "fake"})
    args = benchmark_harness.parse_arguments(live_arguments(inputs))
    _, plan = benchmark_harness.prepare_benchmark(args)
    (args.output / "candidates.json").write_text(json.dumps([{"problem_id": "synthetic"}]))
    (args.output / "grading").mkdir()
    (args.output / "grading/results.json").write_text(json.dumps(results))
    monkeypatch.setattr(benchmark_harness, "run_process", lambda *args, **kwargs: {
        "returncode": 0, "timed_out": False, "runtime_seconds": 1})
    grades, process = benchmark_harness.run_grading(args, plan)
    assert grades == [] and process["status"] == "error"
    assert "Invalid evaluator checkpoint" in process["error"]
