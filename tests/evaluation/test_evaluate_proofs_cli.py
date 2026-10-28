import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd
import pytest

from scripts import evaluate_proofs
from proofstack.context import RunContext
from proofstack.evaluation import proof_evaluation
from proofstack.evaluation.proof_dataset import load_cases
from proofstack.evaluation.proof_grader import MarkingSchemeAgent, ProofGrader


REPOSITORY_ROOT = evaluate_proofs.REPOSITORY_ROOT


def write_inputs(repository, *, include_flagged=False):
    directory = repository / "opc_problems"
    directory.mkdir(parents=True)
    prompts = repository / "configs/evaluators/proofgrader.yaml"
    prompts.parent.mkdir(parents=True)
    shutil.copyfile(REPOSITORY_ROOT / "configs/evaluators/proofgrader.yaml", prompts)
    problems = [{"problem_id": "p", "problem": "Synthetic problem"}]
    references = [{
        "problem_id": "p", "reference_solutions": ["Synthetic reference"],
        "problem_issues": [], "reference_issues": [], "reference_notes": [],
        "reference_status": "available",
    }]
    if include_flagged:
        problems.append({"problem_id": "flagged", "problem": "Synthetic flagged problem"})
        references.append({**references[0], "problem_id": "flagged", "reference_status": "needs_review",
                           "reference_notes": ["Synthetic reference needs review."]})
    pd.DataFrame(problems).to_parquet(directory / "problems.parquet", index=False)
    pd.DataFrame(references).to_parquet(directory / "reference_solutions.parquet", index=False)
    candidates = repository / "candidates.json"
    candidates.write_text(json.dumps([
        {"problem_id": problem["problem_id"], "solution": "Synthetic candidate"} for problem in problems
    ]))
    return candidates


def test_dry_run_never_constructs_model_client(tmp_path, monkeypatch):
    candidates = write_inputs(tmp_path, include_flagged=True)
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not resolve or call a model")
    monkeypatch.setattr(proof_evaluation, "resolve_model", forbidden)
    monkeypatch.setattr(evaluate_proofs, "REPOSITORY_ROOT", tmp_path)
    status = evaluate_proofs.main([
        "--solutions", str(candidates), "--dry-run",
    ])
    assert status == 0
    output, = (tmp_path / "outputs/evaluations").iterdir()
    plan = json.loads((output / "plan.json").read_text())
    assert plan["candidates"] == 2
    assert plan["eligible_candidates"] == 1
    assert plan["skipped_flagged_candidates"] == 1
    assert plan["planned_scheme_calls"] == 1
    assert plan["planned_grading_calls"] == 5
    assert not (output / "events.jsonl").exists()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_budget_is_rejected(value):
    with pytest.raises(SystemExit):
        evaluate_proofs.parse_arguments([
            "--solutions", "unused.json", "--budget-usd", value,
        ])


def test_live_mode_requires_model_and_budget():
    with pytest.raises(SystemExit):
        evaluate_proofs.parse_arguments(["--solutions", "unused.json"])


@pytest.mark.parametrize("removed_option", [
    ["--dataset-dir", "somewhere"],
    ["--output", "somewhere"],
    ["--include-flagged-references"],
])
def test_removed_options_are_rejected(removed_option):
    with pytest.raises(SystemExit) as error:
        evaluate_proofs.parse_arguments([
            "--solutions", "unused.json", "--dry-run", *removed_option,
        ])
    assert error.value.code == 2


def test_repeated_runs_preserve_previous_outputs(tmp_path, monkeypatch):
    candidates = write_inputs(tmp_path)
    monkeypatch.setattr(evaluate_proofs, "REPOSITORY_ROOT", tmp_path)
    arguments = ["--solutions", str(candidates), "--dry-run"]
    assert evaluate_proofs.main(arguments) == 0
    output, = (tmp_path / "outputs/evaluations").iterdir()
    sentinel = output / "result.txt"
    sentinel.write_text("Existing result")
    plan = (output / "plan.json").read_bytes()
    assert evaluate_proofs.main(arguments) == 0
    runs = list((tmp_path / "outputs/evaluations").iterdir())
    assert len(runs) == 2
    assert all((run / "plan.json").is_file() for run in runs)
    assert sentinel.read_text() == "Existing result"
    assert (output / "plan.json").read_bytes() == plan


def test_fixed_dataset_and_output_paths_do_not_depend_on_cwd(tmp_path, monkeypatch, capsys):
    repository = tmp_path / "repository"
    candidates = write_inputs(repository)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(evaluate_proofs, "REPOSITORY_ROOT", repository)
    monkeypatch.chdir(elsewhere)
    assert evaluate_proofs.main(["--solutions", str(candidates), "--dry-run"]) == 0
    output, = (repository / "outputs/evaluations").iterdir()
    assert (output / "plan.json").is_file()
    assert not (elsewhere / "outputs").exists()
    assert str(output) in capsys.readouterr().out


def test_real_model_configuration_uses_existing_client_factory(tmp_path):
    prompts = proof_evaluation.load_prompts(
        evaluate_proofs.REPOSITORY_ROOT / "configs/evaluators/proofgrader.yaml")
    model = proof_evaluation.resolve_model("models/openai/gpt-6-astra")
    context = RunContext.create(root_workdir=tmp_path, flat=True, api_client_factory=lambda config: config,
                                component_configs={
        "ProofGrader": {"model": model, "system_prompt": prompts["grading_prompt"]},
        "MarkingSchemeAgent": {"model": model, "system_prompt": prompts["marking_scheme_prompt"]},
    })
    for agent in (ProofGrader(context), MarkingSchemeAgent(context)):
        config = agent._build_client(agent.MODEL)
        assert config["model"] == model["model"]
        assert config["tools"] == []
        assert config["max_tool_calls"] == 0
        assert agent.SYSTEM_PROMPT.strip()


def test_frozen_schemes_from_prior_run_allow_identically_skipped_cases(tmp_path, make_runner,
                                                                   monkeypatch):
    candidates = write_inputs(tmp_path, include_flagged=True)
    cases = load_cases(tmp_path / "opc_problems", candidates)
    reply = json.dumps({"score": 7, "rationale": "Synthetic grade.", "errors": []})
    runner, _, _ = make_runner([reply], samples=1)
    results = asyncio.run(runner.run(cases))
    assert results[1]["status"] == "skipped_reference_issue"
    monkeypatch.setattr(evaluate_proofs, "REPOSITORY_ROOT", tmp_path)
    assert evaluate_proofs.main([
        "--dry-run", "--solutions", str(candidates),
        "--marking-schemes", str(runner.output_dir / "marking_schemes.json"),
    ]) == 0


def test_sigterm_cancels_provider_and_saves_evaluation(tmp_path):
    candidates = write_inputs(tmp_path)
    output = tmp_path / "interrupted"
    fixture = Path(__file__).parent / "fixtures/slow_provider.py"
    process = subprocess.Popen([sys.executable, str(fixture), str(tmp_path), str(candidates), str(output)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 15
        while not (output / "provider_started").exists():
            assert process.poll() is None, "Evaluator exited before calling the provider"
            assert time.monotonic() < deadline, "Provider did not start"
            time.sleep(0.05)
        process.terminate()
        assert process.wait(timeout=10) == 130
        assert (output / "provider_terminated").exists()
        result = json.loads((output / "results.json").read_text())[0]
        assert result["status"] == "interrupted" and result["score"] is None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
