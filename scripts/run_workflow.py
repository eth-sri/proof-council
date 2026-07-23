"""Run one named workflow preset on one problem.

Usage::

    uv run python scripts/run_workflow.py --workflow author_critic \
        --problem problems/example.txt

    uv run python scripts/run_workflow.py --workflow firstproof_smoke_fast \
        --problem-text "Prove that sqrt(2) is irrational." \
        --problem-id sqrt2_ad_hoc

Presets live under ``configs/workflows/`` and are loaded by
``proofstack.registry``. Run output goes to ``outputs/<run-id>/`` by
default.

This script is the canonical runner for config-first workflow presets.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Load `.env` BEFORE importing project modules — see scripts/_env.py.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from _env import load_dotenv_file  # noqa: E402

load_dotenv_file(REPO_ROOT / ".env")

from proofstack import BudgetSpec, RunContext  # noqa: E402
from proofstack.monitor import DEFAULT_MONITOR_MODEL, RunMonitor  # noqa: E402
from proofstack.registry import load_preset  # noqa: E402
from proofstack.subscription import RESUME_ENV_ALLOWLIST  # noqa: E402


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a ProofCouncil workflow preset on one problem.")
    p.add_argument(
        "--workflow",
        required=True,
        help="DAG preset name under configs/workflows/ (e.g. 'author_critic') or an explicit .yaml path.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--problem", type=Path, help="Path to a text file containing the problem statement.")
    src.add_argument("--problem-text", help="Inline problem statement (LaTeX or plain text).")
    p.add_argument("--problem-id", help="Stable id for the problem (defaults to file stem).")
    p.add_argument("--output", type=Path, default=Path("outputs"))
    p.add_argument(
        "--input",
        action="append",
        default=[],
        help="Override a preset input field. Repeatable. Format: KEY=VALUE or KEY=@json-literal.",
    )
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help="Override a model for a named agent. Repeatable. Format: AGENT=models/path or '*'=models/path.",
    )
    p.add_argument(
        "--component",
        action="append",
        default=[],
        help=(
            "Override component config. Repeatable. Format: AGENT.KEY=VALUE "
            "or AGENT.KEY=@json-literal, e.g. Solver.model=models/openai/gpt-54-mini."
        ),
    )
    p.add_argument("--run-id", help="Explicit run id (default: YYYYMMDD-HHMMSS).")
    p.add_argument("--run-name", help="Human-readable name shown in the dashboard.")
    p.add_argument("--resume-from", help="Run id to resume from (uses its resume_cache).")
    p.add_argument(
        "--restart-from",
        help=(
            "Existing run id or run directory to continue. Defaults to "
            "continuing in that same directory."
        ),
    )
    p.add_argument(
        "--restart-copy",
        action="store_true",
        help=(
            "With --restart-from, copy the old run directory to --run-id "
            "(or a fresh timestamp) and continue in the copy."
        ),
    )
    p.add_argument(
        "--additional-instructions",
        help=(
            "Append extra guidance to the problem for this run. Workflows "
            "with an additional_instructions input receive it there."
        ),
    )
    p.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="Override the preset's max_usd budget.",
    )
    p.add_argument("--monitor", action="store_true", help="Enable live LLM monitor summaries.")
    p.add_argument(
        "--monitor-model",
        default=str(DEFAULT_MONITOR_MODEL),
        help="Model config used by --monitor.",
    )
    return p


def _parse_kv_list(items: list[str], *, label: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw in items:
        if "=" not in raw:
            raise SystemExit(f"invalid {label} override (expected KEY=VALUE): {raw!r}")
        key, _, val = raw.partition("=")
        key = key.strip()
        val = val.strip()
        if val.startswith("@"):
            try:
                parsed: Any = json.loads(val[1:])
            except json.JSONDecodeError as e:
                raise SystemExit(f"invalid JSON in {label} {key}: {e}") from e
        else:
            parsed = _autocast(val)
        out[key] = parsed
    return out


def _parse_component_overrides(items: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in items:
        if "." not in raw or "=" not in raw:
            raise SystemExit(
                f"invalid --component override (expected AGENT.KEY=VALUE): {raw!r}"
            )
        agent_key, _, rest = raw.partition(".")
        key, _, val = rest.partition("=")
        agent_key = agent_key.strip()
        key = key.strip()
        val = val.strip()
        if not agent_key or not key:
            raise SystemExit(
                f"invalid --component override (expected AGENT.KEY=VALUE): {raw!r}"
            )
        if val.startswith("@"):
            try:
                parsed: Any = json.loads(val[1:])
            except json.JSONDecodeError as e:
                raise SystemExit(f"invalid JSON in --component {agent_key}.{key}: {e}") from e
        else:
            parsed = _autocast(val)
        out.setdefault(agent_key, {})[key] = parsed
    return out


def _autocast(val: str) -> Any:
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    if val.lower() in ("none", "null"):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _read_problem(
    args: argparse.Namespace,
    *,
    restart_dir: Path | None = None,
) -> tuple[str, str]:
    if args.problem is not None:
        path = args.problem
        if not path.exists():
            raise SystemExit(f"problem file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        pid = args.problem_id or path.stem
        return text, pid
    if args.problem_text is not None:
        pid = args.problem_id or "inline"
        return args.problem_text, pid
    if restart_dir is not None:
        recovered = _read_problem_from_run(restart_dir, problem_id=args.problem_id)
        if recovered is not None:
            return recovered
    raise SystemExit("must pass --problem or --problem-text")


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_restart_dir(output_root: Path, restart_from: str) -> Path:
    raw = Path(restart_from)
    if raw.exists():
        return raw.resolve()
    return (output_root / restart_from).resolve()


def _read_problem_from_run(
    run_dir: Path,
    *,
    problem_id: str | None = None,
) -> tuple[str, str] | None:
    meta = _read_json(run_dir / "run-metadata.json")
    meta_problem_id = None
    if isinstance(meta, dict):
        snapshot = meta.get("config_snapshot")
        if isinstance(snapshot, dict):
            meta_problem_id = snapshot.get("problem_id")
        outputs = meta.get("outputs")
        if meta_problem_id is None and isinstance(outputs, dict):
            meta_problem_id = outputs.get("problem_id")
    chosen_problem_id = problem_id or (str(meta_problem_id) if meta_problem_id else None)

    workspaces_root = run_dir / "ac_workspaces"
    candidates = sorted(p for p in workspaces_root.glob("*") if p.is_dir())
    if chosen_problem_id:
        safe = _safe_id(chosen_problem_id)
        preferred = workspaces_root / safe
        if preferred.exists():
            candidates = [preferred]
    if candidates:
        workspace = candidates[0]
        problem_path = workspace / "problem.txt"
        if problem_path.exists():
            pid = chosen_problem_id or workspace.name
            return problem_path.read_text(encoding="utf-8").strip(), pid

    event_problem = _read_problem_from_events(run_dir)
    if event_problem is not None:
        text, event_pid = event_problem
        return text, chosen_problem_id or event_pid
    return None


def _read_problem_from_events(run_dir: Path) -> tuple[str, str] | None:
    events_path = run_dir / "events.jsonl"
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("kind") != "run.start":
            continue
        payload = _inflate_event_refs(run_dir, event.get("payload"))
        if not isinstance(payload, dict):
            continue
        inputs = payload.get("inputs")
        if not isinstance(inputs, dict):
            continue
        problem = inputs.get("problem")
        problem_id = inputs.get("problem_id") or payload.get("problem_id") or "restarted"
        if isinstance(problem, str) and problem.strip():
            return problem.strip(), str(problem_id)
    return None


def _inflate_event_refs(run_dir: Path, value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$ref"} and isinstance(value.get("$ref"), str):
            ref_path = run_dir / value["$ref"]
            try:
                return ref_path.read_text(encoding="utf-8")
            except OSError:
                return value
        return {k: _inflate_event_refs(run_dir, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_inflate_event_refs(run_dir, v) for v in value]
    return value


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _safe_id(value: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "problem"


def _write_resume_spec(
    run_dir: Path,
    args: argparse.Namespace,
    problem_text: str,
    problem_id: str,
    run_id: str,
    output_root: Path,
) -> None:
    """Write resume.json: the argv (sans python exe, sans --resume-from) that
    relaunches this exact run. The dashboard appends --resume-from <run_id>."""
    argv: list[str] = [
        "scripts/run_workflow.py",
        "--workflow", args.workflow,
        "--problem-text", problem_text,
        "--problem-id", problem_id,
        "--run-id", run_id,
        "--output", str(output_root),
    ]
    if args.run_name:
        argv += ["--run-name", args.run_name]
    for kv in args.input or []:
        argv += ["--input", kv]
    for kv in args.model or []:
        argv += ["--model", kv]
    for kv in args.component or []:
        argv += ["--component", kv]
    if args.additional_instructions:
        argv += ["--additional-instructions", args.additional_instructions]
    if args.budget_usd is not None:
        argv += ["--budget-usd", str(args.budget_usd)]
    if args.monitor:
        argv += ["--monitor", "--monitor-model", args.monitor_model]
    # Pacing is env-only, so persist the run's override here — resume rebuilds a
    # fresh env and would otherwise fall back to the global setting.
    env = {
        k: v
        for k in RESUME_ENV_ALLOWLIST
        if (v := os.environ.get(k)) is not None
    }
    try:
        (run_dir / "resume.json").write_text(
            json.dumps(
                {"run_id": run_id, "argv": argv, "env": env}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def _write_run_pid(run_dir: Path) -> None:
    """Record this process's PID so the dashboard's Stop button can terminate
    the run's process group. The dashboard launches this script as a process-
    group leader (start_new_session / batch start_new_session), so this PID is
    the run's PGID; the Stop route verifies that before signalling the group."""
    try:
        (run_dir / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass


def _clear_run_pid(run_dir: Path) -> None:
    try:
        (run_dir / "run.pid").unlink()
    except OSError:
        pass


def _append_additional_instructions(problem: str, extra: str) -> str:
    extra = extra.strip()
    if not extra:
        return problem
    return (
        problem.rstrip()
        + "\n\nAdditional instructions for this run:\n"
        + extra
    ).strip()


async def amain() -> int:
    args = _argparser().parse_args()

    preset = load_preset(args.workflow)

    cli_inputs = _parse_kv_list(args.input, label="--input")
    cli_model_overrides = _parse_kv_list(args.model, label="--model")
    cli_component_overrides = _parse_component_overrides(args.component)
    workflow_fields = preset.workflow_cls.Inputs.model_fields  # type: ignore[attr-defined]

    restart_source_dir: Path | None = None
    output_root = args.output
    if args.restart_copy and not args.restart_from:
        raise SystemExit("--restart-copy requires --restart-from")
    if args.restart_from:
        restart_source_dir = _resolve_restart_dir(args.output, args.restart_from)
        if not restart_source_dir.exists():
            raise SystemExit(f"restart source not found: {restart_source_dir}")
        if args.restart_copy:
            run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
            output_root = args.output
            output_root.mkdir(parents=True, exist_ok=True)
            dest = (output_root / run_id).resolve()
            if dest.exists():
                raise SystemExit(f"restart copy destination already exists: {dest}")
            shutil.copytree(restart_source_dir, dest, symlinks=True)
            restart_dir_for_problem = dest
        else:
            if args.run_id and args.run_id != restart_source_dir.name:
                raise SystemExit(
                    "--run-id with --restart-from requires --restart-copy "
                    "unless it matches the existing run id"
                )
            run_id = restart_source_dir.name
            output_root = restart_source_dir.parent
            restart_dir_for_problem = restart_source_dir
        if "resume_run" in workflow_fields:
            cli_inputs.setdefault("resume_run", True)
    else:
        run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        restart_dir_for_problem = None

    problem_text, problem_id = _read_problem(
        args,
        restart_dir=restart_dir_for_problem,
    )
    if args.additional_instructions:
        if "additional_instructions" in workflow_fields:
            cli_inputs.setdefault(
                "additional_instructions", args.additional_instructions
            )
        else:
            problem_text = _append_additional_instructions(
                problem_text, args.additional_instructions
            )

    merged_model_overrides: dict[str, Any] = dict(preset.model_overrides)
    merged_model_overrides.update(cli_model_overrides)
    merged_component_configs = _deep_merge(
        dict(preset.component_configs),
        cli_component_overrides,
    )

    raw_inputs = dict(preset.inputs)
    raw_inputs.update(cli_inputs)
    budget = _budget_with_input_overrides(preset.budget, raw_inputs)
    if args.budget_usd is not None:
        base = budget.model_dump() if budget else {}
        base["max_usd"] = args.budget_usd
        budget = BudgetSpec(**base)

    output_root.mkdir(parents=True, exist_ok=True)

    ctx = RunContext.create(
        run_id=run_id,
        root_workdir=output_root,
        flat=False,
        run_budget=budget,
        model_overrides=merged_model_overrides,
        component_configs=merged_component_configs,
        resume_from=args.resume_from,
        config_snapshot={
            "preset": preset.name,
            "preset_path": str(preset.source_path),
            "problem_id": problem_id,
            "display_name": args.run_name,
            "restart_from": str(restart_source_dir) if restart_source_dir else None,
            "restart_copy": bool(args.restart_copy),
            "cli_inputs": cli_inputs,
            "cli_model_overrides": cli_model_overrides,
            "cli_component_overrides": cli_component_overrides,
            "component_configs": merged_component_configs,
            "monitor": {
                "enabled": bool(args.monitor),
                "model": args.monitor_model if args.monitor else None,
            },
        },
    )

    # Persist exactly how to relaunch this run, so the dashboard's "Resume"
    # button can replay it with --resume-from (in-place resume from the cache).
    _write_resume_spec(ctx.root_workdir, args, problem_text, problem_id, run_id, output_root)
    _write_run_pid(ctx.root_workdir)

    built_inputs = preset.build_inputs(
        problem=problem_text,
        problem_id=problem_id,
        cli_overrides=cli_inputs,
    )
    if args.monitor:
        ctx.monitor = RunMonitor(
            ctx,
            model=args.monitor_model,
            problem=str(built_inputs.get("problem") or problem_text or ""),
            problem_id=str(built_inputs.get("problem_id") or problem_id or ""),
            workflow_structure=_workflow_structure_for_monitor(preset.raw),
        )

    wf_cls = preset.workflow_cls
    # Keep the default name (== class name) so model_overrides keyed by
    # the class name (common in presets and --model AGENT=MODEL CLI
    # overrides) resolve. run_workflow.py runs exactly one problem per
    # invocation, so no event-log disambiguation is needed here.
    wf = wf_cls(ctx)

    await ctx.events.emit(
        "run.start",
        {
            "preset": preset.name,
            "workflow": f"{wf_cls.__module__}.{wf_cls.__name__}",
            "problem_id": problem_id,
            "display_name": args.run_name,
            "inputs": built_inputs,
        },
    )

    try:
        out = await wf(**built_inputs)
    except Exception as e:
        await ctx.events.emit(
            "run.end",
            {"status": "error", "type": type(e).__name__, "msg": str(e)},
        )
        await _drain_monitor(ctx)
        ctx.write_metadata(
            _run_metadata(
                ctx,
                {
                    "status": "error",
                    "display_name": args.run_name,
                    "error": f"{type(e).__name__}: {e}",
                },
            )
        )
        print(f"workflow failed: {type(e).__name__}: {e}", file=sys.stderr)
        _clear_run_pid(ctx.root_workdir)
        return 1

    out_json = out.model_dump(mode="json") if hasattr(out, "model_dump") else out

    status = "error" if isinstance(out_json, dict) and out_json.get("error") else "ok"
    await ctx.events.emit("run.end", {"status": status})
    await _drain_monitor(ctx)
    ctx.write_metadata(
        _run_metadata(
            ctx,
            {"status": status, "display_name": args.run_name, "outputs": out_json},
        )
    )

    _clear_run_pid(ctx.root_workdir)
    print(f"run_id: {run_id}")
    print(f"output: {ctx.root_workdir}")
    print("outputs:")
    print(json.dumps(out_json, ensure_ascii=False, indent=2, default=str))
    return 0


async def _drain_monitor(ctx: RunContext) -> None:
    monitor = getattr(ctx, "monitor", None)
    drain = getattr(monitor, "drain", None)
    if not callable(drain):
        return
    try:
        await drain()
    except Exception:
        pass


def _run_metadata(ctx: RunContext, extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(extra)
    monitor = ctx.config_snapshot.get("monitor")
    if isinstance(monitor, dict):
        out["monitor"] = monitor
    return out


def main() -> int:
    return asyncio.run(amain())


def _workflow_structure_for_monitor(raw: dict[str, Any]) -> dict[str, Any]:
    dag = raw.get("dag") if isinstance(raw.get("dag"), dict) else {}
    nodes = []
    for node in dag.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        ui = node.get("ui") if isinstance(node.get("ui"), dict) else {}
        nodes.append(
            {
                "id": node.get("id"),
                "label": ui.get("label") or node.get("id"),
                "kind": node.get("kind", "agent"),
                "agent": node.get("agent") or node.get("preset"),
                "component": node.get("name"),
                "needs": node.get("needs") or [],
            }
        )
    outputs = dag.get("outputs") if isinstance(dag.get("outputs"), dict) else {}
    return {
        "workflow": raw.get("workflow"),
        "description": raw.get("description", ""),
        "nodes": nodes,
        "outputs": list(outputs),
    }


def _budget_with_input_overrides(
    budget: BudgetSpec | None,
    inputs: dict[str, Any],
) -> BudgetSpec | None:
    base = budget.model_dump(exclude_none=True) if budget else {}
    for input_key, budget_key in (
        ("max_usd", "max_usd"),
        ("max_wallclock_s", "max_wallclock_s"),
    ):
        value = inputs.get(input_key)
        if value in (None, ""):
            continue
        base[budget_key] = float(value)
    return BudgetSpec(**base) if base else None


if __name__ == "__main__":
    raise SystemExit(main())
