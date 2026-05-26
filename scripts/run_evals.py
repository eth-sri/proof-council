"""Run agent regression eval cases (SPEC §13).

Each case is a YAML under ``evals/agents/<agent>/<case_id>.yaml``.

Usage::

    # Run a single case (path or short id)
    uv run python scripts/run_evals.py --case prompt/example_case

    # Run every case under one agent dir
    uv run python scripts/run_evals.py --agent prompt

    # Run all cases (default)
    uv run python scripts/run_evals.py

    # CI mode: machine-readable lines, exits non-zero on any FAIL
    uv run python scripts/run_evals.py --ci

Two evaluation modes share the schema:

  - **fixture**: ``fixture_raw_response`` is set in the YAML — the
    runner injects the file's contents into the agent's
    ``parse_output`` directly, skipping the model call entirely. Used
    for parser-only regression tests.
  - **live**: no ``fixture_raw_response`` — the runner constructs the
    agent and invokes ``__call__`` normally, which makes a real model
    call (billed against ``budget_usd``).

Hard checks run before the judge is invoked. If any hard check fails,
the case fails and the judge is skipped (faster + cheaper + clearer
failure signal). Otherwise the EvalJudge is invoked with the
success_criteria and the case passes/fails per its verdict.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from proofstack import BudgetSpec, RunContext  # noqa: E402
from proofstack.agents.eval_judge import EvalJudge  # noqa: E402

EVALS_ROOT = REPO_ROOT / "evals" / "agents"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class HardCheck:
    field_name: str
    min_length: int | None = None
    min_count: int | None = None
    where: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalCase:
    path: Path
    agent_qualname: str
    case_id: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    fixture_raw_response: str | None = None
    hard_checks: list[HardCheck] = field(default_factory=list)
    success_criteria: str = ""
    budget_usd: float = 0.05
    judge_model: str | None = None

    @property
    def short_name(self) -> str:
        return f"{self.path.parent.name}/{self.case_id}"


def load_case(path: Path) -> EvalCase:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: top-level must be a mapping")

    inputs = _resolve_inputs(raw.get("inputs") or {}, base=path.parent)
    fixture = raw.get("fixture_raw_response")
    if fixture and not Path(fixture).is_absolute():
        fixture = str((path.parent / fixture).resolve())

    hard = []
    for hc in raw.get("hard_checks") or []:
        if not isinstance(hc, dict):
            continue
        hard.append(
            HardCheck(
                field_name=hc.get("field", ""),
                min_length=hc.get("min_length"),
                min_count=hc.get("min_count"),
                where=hc.get("where") or {},
            )
        )

    return EvalCase(
        path=path,
        agent_qualname=raw["agent"],
        case_id=raw.get("case_id") or path.stem,
        description=raw.get("description", "") or "",
        tags=list(raw.get("tags") or []),
        metadata=dict(raw.get("metadata") or {}),
        inputs=inputs,
        fixture_raw_response=fixture,
        hard_checks=hard,
        success_criteria=raw.get("success_criteria", "") or "",
        budget_usd=float(raw.get("budget_usd", 0.05)),
        judge_model=raw.get("judge_model"),
    )


def _resolve_inputs(raw_inputs: dict[str, Any], *, base: Path) -> dict[str, Any]:
    """Expand ``foo_path: relative/file`` into ``foo: <file contents>``."""
    out: dict[str, Any] = {}
    for key, value in raw_inputs.items():
        if key.endswith("_path") and isinstance(value, str):
            target = Path(value)
            if not target.is_absolute():
                target = base / target
            out[key[: -len("_path")]] = target.read_text(encoding="utf-8")
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_cases(
    *,
    case_filter: str | None = None,
    agent_filter: str | None = None,
    tag_filter: list[str] | None = None,
) -> list[EvalCase]:
    paths = sorted(EVALS_ROOT.rglob("*.yaml"))
    cases: list[EvalCase] = []
    for p in paths:
        try:
            c = load_case(p)
        except Exception as e:
            print(f"[skip] {p}: load failed: {e}", file=sys.stderr)
            continue
        if case_filter and c.short_name != case_filter and c.case_id != case_filter:
            continue
        if agent_filter and p.parent.name != agent_filter:
            continue
        if tag_filter and not any(t in c.tags for t in tag_filter):
            continue
        cases.append(c)
    return cases


# ---------------------------------------------------------------------------
# Hard checks
# ---------------------------------------------------------------------------


def run_hard_checks(check_specs: list[HardCheck], output: dict[str, Any]) -> list[str]:
    """Return a list of failure messages; empty list means all checks passed."""
    failures: list[str] = []
    for hc in check_specs:
        value = output.get(hc.field_name)
        if hc.min_length is not None:
            n = len(value) if hasattr(value, "__len__") else 0
            if n < hc.min_length:
                failures.append(
                    f"hard_check failed: len({hc.field_name}) = {n} < {hc.min_length}"
                )
        if hc.min_count is not None:
            if not isinstance(value, list):
                failures.append(
                    f"hard_check failed: {hc.field_name} is not a list (got {type(value).__name__})"
                )
                continue
            matched = [
                item
                for item in value
                if isinstance(item, dict)
                and all(item.get(k) == v for k, v in hc.where.items())
            ]
            if len(matched) < hc.min_count:
                failures.append(
                    f"hard_check failed: count({hc.field_name} where {hc.where}) "
                    f"= {len(matched)} < {hc.min_count}"
                )
    return failures


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case: EvalCase
    status: str                 # pass | fail | inconclusive | error
    duration_s: float
    cost_usd: float
    rationale: str = ""
    hard_check_failures: list[str] = field(default_factory=list)
    judge_verdict: str | None = None
    judge_confidence: float | None = None
    flagged_aspects: list[str] = field(default_factory=list)
    error_msg: str | None = None
    output_preview: dict[str, Any] | None = None


def _import_agent_class(qualname: str) -> type:
    if "." not in qualname:
        raise SystemExit(f"agent qualname must be dotted: {qualname!r}")
    module_path, _, name = qualname.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, name, None)
    if cls is None:
        raise SystemExit(f"{module_path} has no attribute {name}")
    return cls


async def run_case(case: EvalCase, *, output_root: Path) -> CaseResult:
    started = datetime.now(timezone.utc).timestamp()
    cls = _import_agent_class(case.agent_qualname)

    run_id = f"eval-{case.path.parent.name}-{case.case_id}-{started:.0f}"
    ctx = RunContext.create(
        run_id=run_id,
        root_workdir=output_root,
        flat=False,
        run_budget=BudgetSpec(max_usd=case.budget_usd),
        mode="eval",
    )

    # Emit run.start so the run dir is discoverable by the dev UI even
    # on cases that fail hard checks before any agent runs (fixture mode).
    await ctx.events.emit(
        "run.start",
        {
            "case": case.short_name,
            "agent": case.agent_qualname,
            "fixture_mode": bool(case.fixture_raw_response),
            "tags": case.tags,
        },
    )

    result = await _run_case_body(case, cls, ctx, started)

    await ctx.events.emit(
        "run.end",
        {
            "status": result.status,
            "duration_s": round(result.duration_s, 3),
            "cost_usd": round(result.cost_usd, 6),
            "judge_verdict": result.judge_verdict,
            "hard_check_failures": result.hard_check_failures,
            "error": result.error_msg,
        },
    )
    ctx.write_metadata(
        {
            "case": case.short_name,
            "case_path": str(case.path),
            "agent": case.agent_qualname,
            "status": result.status,
            "duration_s": round(result.duration_s, 3),
            "cost_usd": round(result.cost_usd, 6),
            "judge_verdict": result.judge_verdict,
            "judge_confidence": result.judge_confidence,
            "rationale": result.rationale,
            "flagged_aspects": result.flagged_aspects,
            "hard_check_failures": result.hard_check_failures,
            "error": result.error_msg,
        }
    )
    return result


async def _run_case_body(
    case: EvalCase,
    cls: type,
    ctx: RunContext,
    started: float,
) -> CaseResult:
    """Run one case and return the result. Caller owns run.start/run.end."""

    def _now_dt() -> float:
        return datetime.now(timezone.utc).timestamp() - started

    inp_model = cls.Inputs.model_validate(case.inputs)

    # --- run the agent ---------------------------------------------------
    try:
        if case.fixture_raw_response:
            agent = cls(ctx)
            raw_text = Path(case.fixture_raw_response).read_text(encoding="utf-8")
            out = agent.parse_output(raw_text, inp_model)
        else:
            agent = cls(ctx)
            out = await agent(**case.inputs)
    except Exception as e:
        return CaseResult(
            case=case,
            status="error",
            duration_s=_now_dt(),
            cost_usd=ctx.budgets.root().counters.usd,
            error_msg=f"{type(e).__name__}: {e}",
        )

    out_json = out.model_dump(mode="json") if hasattr(out, "model_dump") else dict(out)

    # --- hard checks first ----------------------------------------------
    hc_failures = run_hard_checks(case.hard_checks, out_json)
    if hc_failures:
        return CaseResult(
            case=case,
            status="fail",
            duration_s=_now_dt(),
            cost_usd=ctx.budgets.root().counters.usd,
            hard_check_failures=hc_failures,
            rationale="failed hard structural check; judge skipped",
            output_preview=out_json,
        )

    # --- judge -----------------------------------------------------------
    judge = EvalJudge(ctx)
    if case.judge_model:
        # One-off override; touch the class through model_overrides so the
        # judge invocation picks it up via ctx.model_for().
        ctx.model_overrides[judge.name] = case.judge_model
    try:
        verdict_out = await judge(
            agent_under_test=case.agent_qualname,
            case_inputs=case.inputs,
            success_criteria=case.success_criteria,
            agent_output=out_json,
        )
    except Exception as e:
        return CaseResult(
            case=case,
            status="error",
            duration_s=_now_dt(),
            cost_usd=ctx.budgets.root().counters.usd,
            error_msg=f"judge raised {type(e).__name__}: {e}",
            output_preview=out_json,
        )

    return CaseResult(
        case=case,
        status=verdict_out.verdict,
        duration_s=_now_dt(),
        cost_usd=ctx.budgets.root().counters.usd,
        rationale=verdict_out.rationale,
        judge_verdict=verdict_out.verdict,
        judge_confidence=verdict_out.confidence,
        flagged_aspects=list(verdict_out.flagged_aspects),
        output_preview=out_json,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


_STATUS_GLYPH = {
    "pass": "OK ",
    "fail": "FAIL",
    "inconclusive": "??  ",
    "error": "ERR ",
}


def print_human(result: CaseResult) -> None:
    g = _STATUS_GLYPH.get(result.status, "?")
    print(
        f"[{g}] {result.case.short_name}  "
        f"({result.duration_s:.2f}s, ${result.cost_usd:.4f})"
    )
    if result.hard_check_failures:
        for f in result.hard_check_failures:
            print(f"        {f}")
    if result.error_msg:
        print(f"        error: {result.error_msg}")
    if result.rationale:
        rationale = result.rationale.strip().splitlines()[0][:200]
        print(f"        {rationale}")
    if result.flagged_aspects:
        for a in result.flagged_aspects[:5]:
            print(f"        - {a}")


def print_ci_line(result: CaseResult) -> None:
    payload = {
        "case": result.case.short_name,
        "status": result.status,
        "duration_s": round(result.duration_s, 3),
        "cost_usd": round(result.cost_usd, 4),
        "judge_verdict": result.judge_verdict,
        "judge_confidence": result.judge_confidence,
        "hard_check_failures": result.hard_check_failures,
        "error": result.error_msg,
    }
    print(json.dumps(payload, default=str))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run agent regression eval cases.")
    p.add_argument("--case", help="Run a single case by short_name (agent/case_id) or id.")
    p.add_argument("--agent", help="Run all cases under one agent dir.")
    p.add_argument("--tag", action="append", default=[], help="Filter by tag. Repeatable.")
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "evals",
        help="Where eval run dirs are written (default: outputs/evals/).",
    )
    p.add_argument("--ci", action="store_true", help="Machine-readable JSONL output; non-zero exit on any fail.")
    p.add_argument("--concurrency", type=int, default=4, help="Parallel cases (default 4).")
    return p


async def amain() -> int:
    args = _argparser().parse_args()
    cases = discover_cases(
        case_filter=args.case,
        agent_filter=args.agent,
        tag_filter=args.tag or None,
    )
    explicit_filter = bool(args.case or args.agent or args.tag)
    if not cases:
        print("no cases matched", file=sys.stderr)
        # Bare invocation against an empty suite is a state, not a
        # failure. But: an explicit filter or CI mode with zero matches
        # almost always means a typo / renamed case / accidentally
        # excluded suite, and silently exiting 0 lets automation skip
        # the regressions it was supposed to gate on.
        if explicit_filter or args.ci:
            return 1
        return 0

    args.output.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)

    async def _bounded(case: EvalCase) -> CaseResult:
        async with sem:
            return await run_case(case, output_root=args.output)

    results = await asyncio.gather(
        *[_bounded(c) for c in cases], return_exceptions=False
    )

    for r in results:
        (print_ci_line if args.ci else print_human)(r)

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    total_cost = sum(r.cost_usd for r in results)
    summary = (
        f"\n{len(results)} case(s): "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        + f" — total cost ${total_cost:.4f}"
    )
    if args.ci:
        print(json.dumps({"summary": counts, "total_cost_usd": round(total_cost, 4)}))
    else:
        print(summary)

    return 1 if (counts.get("fail", 0) + counts.get("error", 0)) > 0 else 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
