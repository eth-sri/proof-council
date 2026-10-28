"""Grade saved proofs against OPC references using a ProofGrader-style ensemble."""

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts._env import load_dotenv_file

load_dotenv_file(REPOSITORY_ROOT / ".env")

from proofstack.evaluation.configuration import positive_number
from proofstack.evaluation.proof_evaluation import evaluate_saved_proofs


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solutions", required=True, type=Path, help="Candidate JSON or JSONL file")
    parser.add_argument("--model", help="ProofCouncil model config for grading")
    parser.add_argument("--scheme-model", help="Model config for marking schemes (defaults to --model)")
    parser.add_argument("--marking-schemes", type=Path, help="Reuse frozen marking_schemes.json")
    parser.add_argument("--samples", type=int, default=5, help="Positive odd ensemble size (default: 5)")
    parser.add_argument("--budget-usd", type=positive_number, help="Required for paid evaluation")
    parser.add_argument("--timeout-seconds", type=positive_number, default=3600)
    parser.add_argument("--dry-run", action="store_true", help="Validate and write a plan without model calls")
    args = parser.parse_args(argv)
    if args.samples < 1 or args.samples % 2 == 0:
        parser.error("--samples must be a positive odd integer")
    if not args.dry_run and (not args.model or args.budget_usd is None):
        parser.error("Live evaluation requires --model and --budget-usd")
    return args


def main(argv=None):
    return evaluate_saved_proofs(parse_arguments(argv), REPOSITORY_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
