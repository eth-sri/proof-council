"""Evaluate saved candidates, including the benchmark's isolated grading phase."""

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

from proofstack.budget import BudgetSpec
from proofstack.context import RunContext
from proofstack.evaluation.proof_dataset import (
    content_hash, dataset_provenance, load_cases, load_marking_schemes,
)
from proofstack.evaluation.proof_grader_runner import ProofGraderRunner
from proofstack.evaluation.configuration import load_prompts, resolve_model
from proofstack.evaluation.run_support import create_run_directory, interrupt_on_sigterm, write_json


def evaluation_plan(args, dataset_dir, cases, schemes, prompts):
    eligible = eligible_cases(cases)
    scheme_calls = len({case.problem_id for case in eligible} - schemes.keys())
    return {
        "method": "ProofGrader-style; independently written prompts and integration",
        "dataset": dataset_provenance(dataset_dir),
        "candidates_sha256": content_hash([vars(case) for case in cases]),
        "prompts_sha256": content_hash(prompts),
        "model": args.model,
        "scheme_model": args.scheme_model or args.model,
        "samples": args.samples,
        "candidates": len(cases),
        "eligible_candidates": len(eligible),
        "skipped_flagged_candidates": len(cases) - len(eligible),
        "planned_scheme_calls": scheme_calls,
        "planned_grading_calls": len(eligible) * args.samples,
        "budget_usd": args.budget_usd,
        "timeout_seconds": args.timeout_seconds,
        "dry_run": args.dry_run,
    }


def eligible_cases(cases):
    return [case for case in cases if case.reference_status == "available"]


async def run_evaluation(args, output, cases, schemes, prompts, plan):
    model = resolve_model(args.model)
    scheme_model = resolve_model(args.scheme_model or args.model)
    components = {
        "ProofGrader": {"model": model, "system_prompt": prompts["grading_prompt"]},
        "MarkingSchemeAgent": {"model": scheme_model, "system_prompt": prompts["marking_scheme_prompt"]},
    }
    context = RunContext.create(
        root_workdir=output, flat=True,
        run_budget=BudgetSpec(max_usd=args.budget_usd, max_wallclock_s=args.timeout_seconds),
        component_configs=components, config_snapshot={**plan, "components": components},
    )
    context.write_metadata({"evaluator": "ProofGrader-style"})
    runner = ProofGraderRunner(context, samples=args.samples,
                              marking_schemes=schemes)
    results = await runner.run(cases)
    return 0 if all(result["status"] in {"graded", "skipped_reference_issue"} for result in results) else 1


@interrupt_on_sigterm()
def evaluate_saved_proofs(args, repository_root, *, output=None):
    try:
        dataset_dir = repository_root / "opc_problems"
        cases = load_cases(dataset_dir, args.solutions)
        eligible = eligible_cases(cases)
        schemes = load_marking_schemes(args.marking_schemes, eligible)
        prompts = load_prompts(repository_root / "configs/evaluators/proofgrader.yaml")
        plan = evaluation_plan(args, dataset_dir, cases, schemes, prompts)
        if output is None:
            output = create_run_directory(repository_root, "evaluations")
        else:
            output.mkdir(parents=True, exist_ok=False)
        print(f"Output directory: {output}", flush=True)
        write_json(output / "plan.json", plan)
        write_json(output / "marking_schemes.json", schemes)
        if args.dry_run:
            print(f"Validated {len(cases)} candidates; planned model calls: "
                  f"{plan['planned_scheme_calls']} marking schemes + {plan['planned_grading_calls']} grades.")
            return 0
        return asyncio.run(run_evaluation(args, output, cases, schemes, prompts, plan))
    except KeyboardInterrupt:
        print("Evaluation interrupted; completed artifacts have been retained.", file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2


def grade_benchmark(plan_path, repository_root):
    """Internal subprocess entry point using the saved benchmark plan."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if dataset_provenance(repository_root / "opc_problems") != plan["dataset"]:
        raise ValueError("Dataset changed after benchmark selection")
    output = plan_path.parent
    frozen_schemes = output / "frozen_marking_schemes.json"
    args = SimpleNamespace(
        solutions=output / "candidates.json", model=plan["grader_model"],
        scheme_model=plan["scheme_model"], samples=plan["samples"],
        budget_usd=plan["grader_budget_usd"], timeout_seconds=plan["grader_timeout_seconds"],
        marking_schemes=frozen_schemes if plan["uses_frozen_marking_schemes"] else None, dry_run=False,
    )
    return evaluate_saved_proofs(args, repository_root, output=output / "grading")


if __name__ == "__main__":
    raise SystemExit(grade_benchmark(Path(sys.argv[1]).resolve(), Path(__file__).resolve().parents[3]))
