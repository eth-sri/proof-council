"""End-to-end smoke test: build a RunContext, run a configurable prompt node.

Usage:
    OPENAI_API_KEY=... uv run python scripts/proofstack_smoke.py
    OPENAI_API_KEY=... uv run python scripts/proofstack_smoke.py \\
        --model models/openai/gpt-54-mini --problem-file my_problem.txt

Writes ``outputs/<run-id>/`` with ``events.jsonl``, ``run-metadata.json``,
and the agent's ``input.json`` / ``output.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="proofstack smoke test")
    p.add_argument(
        "--model",
        default="models/openai/gpt-54-mini",
        help="Model config reference (e.g. models/openai/gpt-54-mini).",
    )
    p.add_argument(
        "--problem-file",
        type=Path,
        default=None,
        help="Path to a text/LaTeX file with the problem statement.",
    )
    p.add_argument(
        "--problem",
        default=(
            "Prove that for every positive integer $n$, "
            "$1 + 2 + \\cdots + n = \\frac{n(n+1)}{2}$."
        ),
        help="Inline problem statement (used if --problem-file not given).",
    )
    p.add_argument(
        "--max-usd",
        type=float,
        default=2.0,
        help="Total run budget in USD (raises BudgetExhausted past this).",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Optional run id (defaults to YYYYMMDD-HHMMSS).",
    )
    return p


async def amain() -> int:
    args = _argparser().parse_args()
    # Local import so --help works without the heavy provider SDKs loaded.
    from proofstack import BudgetSpec, RunContext
    from proofstack.agents.configurable_prompt import ConfigurablePromptAgent

    problem = (
        args.problem_file.read_text(encoding="utf-8")
        if args.problem_file is not None
        else args.problem
    )

    ctx = RunContext.create(
        run_id=args.run_id,
        run_budget=BudgetSpec(max_usd=args.max_usd),
        component_configs={
            "smoke_solver": {
                "model": args.model,
                "system_prompt": (
                    "You are an expert research mathematician. Return only the proof "
                    "inside <solution>...</solution>."
                ),
                "user_prompt": "Problem:\n\n{problem}\n\nWrite a complete proof.",
                "input_schema": {"problem": "string"},
                "output": {"xml_tags": ["solution"], "default_field": "solution"},
            }
        },
        config_snapshot={"model": args.model, "problem_preview": problem[:200]},
    )
    await ctx.events.emit("run.start", {"model": args.model})

    solver = ConfigurablePromptAgent(ctx, name="smoke_solver")
    try:
        out = await solver(problem=problem)
    except Exception as e:
        await ctx.events.emit("run.end", {"status": "error", "msg": str(e)})
        ctx.write_metadata({"status": "error", "error": str(e)})
        print(f"smoke test failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    run_root = ctx.root_workdir
    tracker = ctx.budgets.root("run")
    summary = {
        "run_id": ctx.run_id,
        "run_dir": str(run_root),
        "cost_usd": tracker.counters.usd,
        "tokens": tracker.counters.tokens,
        "wallclock_s": tracker.counters.wallclock_s(),
        "solution_preview": (out.solution if hasattr(out, "solution") else str(out))[:400],
    }
    await ctx.events.emit("run.end", {"status": "ok", **summary})
    ctx.write_metadata({"status": "ok", **summary})

    print(json.dumps(summary, indent=2, default=str))
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
