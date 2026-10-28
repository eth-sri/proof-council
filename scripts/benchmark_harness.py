"""Run one workflow YAML on OPC problems and grade its proofs with ProofGrader-style voting."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import yaml

from scripts._env import load_dotenv_file

load_dotenv_file(REPOSITORY_ROOT / ".env")

from proofstack.context import component_config_for_class
from proofstack.evaluation.benchmark_dataset import select_problems
from proofstack.evaluation.benchmark_report import summarize_benchmark
from proofstack.evaluation.configuration import load_prompts, load_solver_timeout, positive_number, resolve_model
from proofstack.evaluation.harness_runner import HarnessSettings, run_harness
from proofstack.evaluation.proof_dataset import (
    content_hash, dataset_provenance, load_marking_schemes, load_reference_cases,
)
from proofstack.evaluation.run_support import (
    create_run_directory, interrupt_on_sigterm, run_process, subprocess_environment, usage_report, write_json,
)
from proofstack.registry import PresetError, load_preset


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, type=Path, help="One workflow YAML file")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=10)
    selection.add_argument("--all", action="store_true", help="Use every eligible unique statement")
    selection.add_argument("--problem-id", action="append", help="Select specific IDs; repeat as needed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--answer-field", help="Top-level workflow output containing the final proof")
    add_grading_arguments(parser)
    parser.add_argument("--solver-budget-usd-per-problem", type=positive_number)
    parser.add_argument("--dry-run", action="store_true", help="Validate and save the plan; no solver or judge calls")
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    return args


def add_grading_arguments(parser):
    parser.add_argument("--grader-model", help="Model config used by the evaluator")
    parser.add_argument("--scheme-model", help="Defaults to --grader-model")
    parser.add_argument("--marking-schemes", type=Path, help="Reuse frozen schemes for fair comparisons")
    parser.add_argument("--samples", type=int, default=5, help="Positive odd number of grading votes")
    parser.add_argument("--grader-budget-usd", type=positive_number)
    parser.add_argument("--grader-timeout-seconds", type=positive_number, default=3600)


def validate_arguments(parser, args):
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.samples < 1 or args.samples % 2 == 0:
        parser.error("--samples must be a positive odd integer")
    if not args.dry_run and (not args.grader_model or args.grader_budget_usd is None
                            or args.solver_budget_usd_per_problem is None):
        parser.error("Live benchmarks require --grader-model, --grader-budget-usd, "
                     "and --solver-budget-usd-per-problem")
    for field in ("workflow", "marking_schemes"):
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).resolve())


def validate_workflow(path, records):
    preset = load_preset(str(path))
    inputs = preset.workflow_cls.Inputs
    if "problem" not in inputs.model_fields and inputs.model_config.get("extra") != "allow":
        raise ValueError("Benchmark workflow must accept a 'problem' input")
    for record in records:
        inputs.model_validate(preset.build_inputs(problem=record["problem"], problem_id="benchmark"))
    from proofstack.agents.dag_workflow import DAGWorkflow
    from proofstack.dag_schema import build_dag_report

    if issubclass(preset.workflow_cls, DAGWorkflow):
        config = component_config_for_class(preset.component_configs, preset.workflow_cls,
                                            preset.workflow_cls.__name__)
        report = build_dag_report(config.get("dag"), component_configs=preset.component_configs,
                                  workflow_inputs=preset.inputs,
                                  workflow_input_schema=inputs.model_json_schema().get("properties", {}))
        if not report.ok:
            raise ValueError("Invalid workflow: " + "; ".join(report.errors))
    return path.read_bytes()


def build_plan(args, selection, workflow_bytes, schemes):
    count = len(selection.records)
    solver_budget = args.solver_budget_usd_per_problem
    return {
        "workflow": str(args.workflow), "workflow_sha256": hashlib.sha256(workflow_bytes).hexdigest(),
        "dataset": selection.provenance, "selection": selection.report,
        "selected_count": count, "dry_run": args.dry_run,
        "answer_field": args.answer_field, "answer_format": "auto",
        "solver_budget_usd_per_problem": solver_budget,
        "solver_timeout_seconds": args.solver_timeout_seconds,
        "grader_model": args.grader_model, "scheme_model": args.scheme_model or args.grader_model,
        "grader_budget_usd": args.grader_budget_usd, "grader_timeout_seconds": args.grader_timeout_seconds,
        "samples": args.samples, "grading_prompts_sha256": content_hash(load_prompts(REPOSITORY_ROOT / "configs/evaluators/proofgrader.yaml")),
        "uses_frozen_marking_schemes": args.marking_schemes is not None,
        "maximum_planned_scheme_calls": count - len(set(selection.report["selected_problem_ids"]) & schemes.keys()),
        "maximum_planned_grading_calls": count * args.samples,
        "nominal_total_budget_usd": (count * solver_budget + args.grader_budget_usd
                                     if solver_budget is not None and args.grader_budget_usd is not None else None),
        "budget_note": "USD limits are cooperative; in-flight calls may exceed them. No retries or automatic resume.",
    }


def prepare_benchmark(args):
    args.dataset_dir = REPOSITORY_ROOT / "opc_problems"
    args.solver_timeout_seconds = load_solver_timeout(REPOSITORY_ROOT)
    selection = select_problems(args.dataset_dir, limit=None if args.all else args.limit, seed=args.seed,
                                problem_ids=args.problem_id)
    workflow_bytes = validate_workflow(args.workflow, selection.records)
    cases = load_reference_cases(args.dataset_dir, selection.report["selected_problem_ids"])
    schemes = load_marking_schemes(args.marking_schemes, cases)
    if not args.dry_run:
        resolve_model(args.grader_model)
        resolve_model(args.scheme_model or args.grader_model)
    plan = build_plan(args, selection, workflow_bytes, schemes)
    args.output = create_run_directory(REPOSITORY_ROOT, "benchmarks")
    print(f"Output directory: {args.output}", flush=True)
    (args.output / "workflow.yaml").write_bytes(workflow_bytes)
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "problems.json", selection.records)
    if args.marking_schemes:
        write_json(args.output / "frozen_marking_schemes.json", schemes)
    return selection.records, plan


def grader_command(args):
    return [sys.executable, "-m", "proofstack.evaluation.proof_evaluation", str(args.output / "plan.json")]


def run_grading(args, plan):
    started = time.monotonic()
    process = {"returncode": None, "timed_out": False}
    try:
        if dataset_provenance(args.dataset_dir) != plan["dataset"]:
            raise ValueError("Dataset changed after benchmark selection")
        process = run_process(grader_command(args), args.output / "grading.log", cwd=REPOSITORY_ROOT,
                              timeout_seconds=args.grader_timeout_seconds,
                              env=subprocess_environment(REPOSITORY_ROOT))
        status = "timeout" if process["timed_out"] else "complete" if process["returncode"] == 0 else "error"
    except KeyboardInterrupt:
        status, process["error"] = "interrupted", "Interrupted during grading"
    except (OSError, ValueError) as error:
        status, process["error"] = "error", str(error)
    process.setdefault("runtime_seconds", time.monotonic() - started)
    try:
        results = read_grade_results(args.output)
    except (OSError, ValueError, TypeError, KeyError) as error:
        results, status, process["error"] = [], "error", f"Invalid evaluator checkpoint: {error}"
    usage = usage_report(args.output / "grading", completed=status == "complete")
    return results, {**process, "status": status, "usage": usage}


def read_grade_results(output):
    candidates = json.loads((output / "candidates.json").read_text())
    expected_ids = {candidate["problem_id"] for candidate in candidates}
    results_path = output / "grading/results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    if not isinstance(results, list):
        raise ValueError("Evaluator results must be a JSON list")
    seen = set()
    for result in results:
        if not isinstance(result, dict) or result.get("problem_id") not in expected_ids:
            raise ValueError("Evaluator returned an unknown problem ID")
        if result["problem_id"] in seen or result.get("candidate_id") != "default":
            raise ValueError("Evaluator returned duplicate or unexpected candidate IDs")
        if not isinstance(result.get("status"), str):
            raise ValueError("Evaluator result is missing a status")
        if result["status"] == "graded":
            score = result.get("score")
            if type(score) not in (int, float) or not 0 <= score <= 7:
                raise ValueError("Completed grade must have a score between 0 and 7")
        seen.add(result["problem_id"])
    return results


def save_progress(output, records, solver_results, grade_results, grading):
    by_id = {result["problem_id"]: result for result in solver_results}
    grades = {result["problem_id"]: result for result in grade_results}
    combined = []
    for record in records:
        problem_id = record["problem_id"]
        solver_result = dict(by_id.get(problem_id, {"status": "not_run"}))
        solver_result.pop("solution", None)
        grade_result = grades.get(problem_id, {"status": "not_run", "score": None})
        combined.append({"problem_id": problem_id, "solver": solver_result, "grade": grade_result})
    write_json(output / "results.json", combined)
    summary = summarize_benchmark(solver_results, grade_results, selected_count=len(records),
                                  grading_usage=grading.get("usage"), grading_status=grading["status"],
                                  grading_runtime_seconds=grading.get("runtime_seconds", 0))
    write_json(output / "summary.json", summary)
    write_json(output / "grading_process.json", grading)
    return summary


def run_benchmark(args, records, plan):
    settings = HarnessSettings(repository_root=REPOSITORY_ROOT, workflow=args.output / "workflow.yaml",
                               output=args.output, budget_usd=args.solver_budget_usd_per_problem,
                               timeout_seconds=args.solver_timeout_seconds,
                               answer_field=args.answer_field)
    solver_results, grades, candidates = [], [], []
    grading = {"status": "not_run"}
    save_progress(args.output, records, solver_results, grades, grading)
    try:
        for record in records:
            result = run_harness(settings, record)
            solver_results.append(result)
            if result["status"] == "solved":
                candidates.append({"problem_id": record["problem_id"], "candidate_id": "default",
                                   "solution": result["solution"]})
            write_json(args.output / "candidates.json", candidates)
            save_progress(args.output, records, solver_results, grades, grading)
            print(f"[{len(solver_results)}/{len(records)}] {record['problem_id']}: {result['status']}", flush=True)
            if result["status"] == "interrupted":
                return 130
        if candidates:
            grading = {"status": "running"}
            save_progress(args.output, records, solver_results, grades, grading)
            grades, grading = run_grading(args, plan)
    finally:
        summary = save_progress(args.output, records, solver_results, grades, grading)
    print(json.dumps(summary, indent=2))
    if grading["status"] == "interrupted":
        return 130
    return 0 if summary["graded_count"] == len(records) and grading["status"] == "complete" else 1


@interrupt_on_sigterm()
def main(argv=None):
    args = parse_arguments(argv)
    try:
        records, plan = prepare_benchmark(args)
        if args.dry_run:
            print(f"Validated {len(records)} problems and workflow; plan saved to {args.output / 'plan.json'}")
            return 0
        return run_benchmark(args, records, plan)
    except KeyboardInterrupt:
        print("Benchmark interrupted; completed artifacts have been retained.", file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, TypeError, PresetError, yaml.YAMLError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
