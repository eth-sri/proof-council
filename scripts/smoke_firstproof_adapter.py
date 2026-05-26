"""Smoke-test the First Proof adapter with a mocked workflow runner.

This avoids provider API calls while verifying the harness-facing files.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


MOCK_RUNNER = r'''
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--workflow", required=True)
parser.add_argument("--problem", required=True)
parser.add_argument("--problem-id", required=True)
parser.add_argument("--run-id", required=True)
parser.add_argument("--run-name")
parser.add_argument("--output", required=True)
parser.add_argument("--budget-usd")
parser.add_argument("--input", action="append", default=[])
parser.add_argument("--restart-from")
parser.add_argument("--additional-instructions")
args = parser.parse_args()
if args.workflow != "author_critic_long":
    raise SystemExit(f"unexpected workflow: {args.workflow}")
if args.budget_usd != "1000":
    raise SystemExit(f"unexpected budget: {args.budget_usd}")
inputs = dict(item.split("=", 1) for item in args.input)
if sorted(args.input) not in (
    [
        "compute_codex_sandbox=docker-bypass",
        "n_rounds=5",
        "page_limit=12",
        "stop_after_review_round=true",
    ],
    [
        "compute_codex_sandbox=docker-bypass",
        "n_rounds=10",
        "page_limit=12",
        "stop_after_review_round=true",
    ],
):
    raise SystemExit(f"unexpected workflow inputs: {args.input}")
if args.restart_from and inputs["n_rounds"] != "10":
    raise SystemExit(f"restart stage should request 10 rounds: {args.input}")
if not args.restart_from and inputs["n_rounds"] != "5":
    raise SystemExit(f"first stage should request 5 rounds: {args.input}")
if os.environ.get("PROOFSTACK_SANDBOX_BACKEND") != "subprocess":
    raise SystemExit("PROOFSTACK_SANDBOX_BACKEND was not forced to subprocess")

run_dir = Path(args.output) / args.run_id
solutions = run_dir / "solutions"
solutions.mkdir(parents=True, exist_ok=True)
n_rounds = int(inputs["n_rounds"])
solved = args.problem_id == "prob-001" or n_rounds >= 10
tex = (
    "\\documentclass{article}\n"
    "\\begin{document}\n"
    f"Mock solution for {args.problem_id} after {n_rounds} rounds using {args.workflow.replace('_', ' ')}.\n"
    "\\end{document}\n"
)
(solutions / f"{args.problem_id}.tex").write_text(tex, encoding="utf-8")
(run_dir / "run-metadata.json").write_text(
    json.dumps(
        {
            "status": "ok",
            "outputs": {
                "answer_tex": str(solutions / f"{args.problem_id}.tex"),
                "compiled": True,
                "pages": 1,
                "rounds_completed": n_rounds,
                "early_stopped": solved,
            },
        }
    ),
    encoding="utf-8",
)
event = {
    "kind": "model.call",
    "payload": {
        "model": "models/openai/mock",
        "in_tokens": 3,
        "out_tokens": 5,
        "cost_usd": 0.001,
    },
}
with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(event) + "\n")
print(f"mock workflow completed for {args.problem_id}")
'''


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="firstproof-smoke-") as tmp:
        root = Path(tmp)
        input_path = root / "input.json"
        output_dir = root / "output"
        mock_runner = root / "mock_run_workflow.py"
        mock_runner.write_text(MOCK_RUNNER, encoding="utf-8")
        _write_json(
            input_path,
            {
                "problems": [
                    {"id": "prob-001", "latex": "Prove that 1+1=2."},
                    {"id": "prob-002", "latex": "Prove that 2+2=4."},
                ]
            },
        )
        env = os.environ.copy()
        env.update(
            {
                "FIRSTPROOF_INPUT_PATH": str(input_path),
                "FIRSTPROOF_OUTPUT_DIR": str(output_dir),
                "FIRSTPROOF_MAX_PARALLEL": "1",
                "FIRSTPROOF_RUN_WORKFLOW_SCRIPT": str(mock_runner),
                "FIRSTPROOF_RUN_NAMESPACE": "smoke",
                "FIRSTPROOF_HEALTHCHECK": "off",
            }
        )
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "firstproof_entrypoint.py")],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            return proc.returncode
        required = [
            output_dir / "solutions.json",
            output_dir / "run_summary.json",
            output_dir / "token_usage.jsonl",
            output_dir / "prob-001.tex",
            output_dir / "prob-002.tex",
            output_dir / "staged_solutions" / "rounds-005" / "prob-001.tex",
            output_dir / "staged_solutions" / "rounds-005" / "prob-002.tex",
            output_dir / "staged_solutions" / "rounds-010" / "prob-002.tex",
            output_dir / "workflow_runs" / "firstproof-prob-001-smoke" / "solutions" / "prob-001.tex",
            output_dir / "workflow_runs" / "firstproof-prob-002-smoke" / "solutions" / "prob-002.tex",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            print("missing expected files:")
            for path in missing:
                print(f"  {path}")
            print(proc.stdout)
            return 1
        summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
        statuses = [item["status"] for item in summary["per_problem"]]
        if statuses != ["ok", "ok"]:
            print(f"unexpected statuses: {statuses}")
            print(proc.stdout)
            return 1
        stage_counts = [len(item.get("stages", [])) for item in summary["per_problem"]]
        if stage_counts != [1, 2]:
            print(f"unexpected stage counts: {stage_counts}")
            print(proc.stdout)
            return 1
        for tex_path in (output_dir / "prob-001.tex", output_dir / "prob-002.tex"):
            tex = tex_path.read_text(encoding="utf-8")
            if "\\documentclass[12pt]{article}" not in tex:
                print(f"{tex_path} was not normalized to 12pt article")
                print(proc.stdout)
                return 1
        totals = summary.get("totals") or {}
        if totals.get("input_tokens") != 9 or totals.get("output_tokens") != 15:
            print(f"unexpected token totals: {totals}")
            print(proc.stdout)
            return 1
        token_lines = (output_dir / "token_usage.jsonl").read_text(encoding="utf-8").strip().splitlines()
        if len(token_lines) != 3:
            print(f"expected three token usage lines, got {len(token_lines)}")
            print(proc.stdout)
            return 1
        print("First Proof adapter smoke test passed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
