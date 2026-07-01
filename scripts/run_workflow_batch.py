"""Run one workflow preset on several problems with bounded parallelism."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a ProofCouncil workflow preset on several problems.")
    p.add_argument("--workflow", required=True)
    p.add_argument("--problems-file", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-name", help="Human-readable name shown in the dashboard.")
    p.add_argument("--max-parallel", type=int, default=1)
    p.add_argument(
        "--input",
        action="append",
        default=[],
        help="Workflow input override forwarded to each run. Repeatable. KEY=VALUE.",
    )
    p.add_argument("--monitor", action="store_true", help="Enable live LLM monitor summaries for each problem run.")
    p.add_argument("--monitor-model", default="models/openai/gpt-54-mini")
    return p


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value).strip())
    return safe.strip("_") or "problem"


def _human_label(value: str) -> str:
    return _slug(value).replace("_", " ").replace("-", " ").title()


def _child_run_name(run_name: str | None, workflow: str, problem: dict[str, str]) -> str:
    base = str(run_name or _human_label(workflow)).strip() or _human_label(workflow)
    problem_name = str(problem.get("display_name") or _human_label(problem.get("id", ""))).strip()
    if not problem_name:
        return base
    if base == problem_name or base.endswith(f" · {problem_name}"):
        return base
    return f"{base} · {problem_name}"


def _load_problems(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    problems = raw.get("problems") if isinstance(raw, dict) else raw
    if not isinstance(problems, list):
        raise SystemExit("problems file must contain a list or {'problems': [...]}")
    out: list[dict[str, str]] = []
    for idx, item in enumerate(problems, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"problem {idx}: expected an object")
        problem_id = _slug(str(item.get("id") or f"problem_{idx}"))
        problem_text = str(
            item.get("latex")
            or item.get("text")
            or item.get("problem")
            or item.get("problem_text")
            or item.get("statement")
            or ""
        ).strip()
        if not problem_text:
            raise SystemExit(f"problem {problem_id}: empty problem text")
        display_name = str(item.get("display_name") or item.get("title") or _human_label(problem_id))
        out.append({"id": problem_id, "latex": problem_text, "display_name": display_name})
    return out


def _write_metadata(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


async def amain() -> int:
    args = _argparser().parse_args()
    problems = _load_problems(args.problems_file)
    max_parallel = max(1, int(args.max_parallel or 1))

    outputs_root = args.output.resolve()
    outputs_root.mkdir(parents=True, exist_ok=True)
    batch_dir = outputs_root / args.run_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = batch_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    metadata_path = batch_dir / "run-metadata.json"

    manifest = {
        "started_at": _now(),
        "preset": args.workflow,
        "max_parallel": max_parallel,
        "problems": {
            problem["id"]: {
                "status": "queued",
                "problem_id": problem["id"],
                "display_name": problem.get("display_name") or _human_label(problem["id"]),
                "run_id": f"{args.run_id}-{_slug(problem['id'])}",
            }
            for problem in problems
        },
    }
    meta: dict[str, Any] = {
        "status": "running",
        "display_name": args.run_name,
        "started_by": "dashboard",
        "preset": args.workflow,
        "monitor": {"enabled": bool(args.monitor), "model": args.monitor_model if args.monitor else None},
        "started_at": manifest["started_at"],
        "manifest": manifest,
    }
    _write_metadata(metadata_path, meta)

    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_parallel)

    async def update_problem(problem_id: str, **fields: Any) -> None:
        async with lock:
            manifest["problems"][problem_id].update(fields)
            statuses = [
                p.get("status")
                for p in manifest["problems"].values()
                if isinstance(p, dict) and p.get("status")
            ]
            _write_metadata(metadata_path, meta)

    async def run_one(problem: dict[str, str]) -> int:
        async with semaphore:
            problem_id = problem["id"]
            run_id = manifest["problems"][problem_id]["run_id"]
            log_path = logs_dir / f"{_slug(problem_id)}.log"
            await update_problem(problem_id, status="running", started_at=_now())
            cmd = [
                sys.executable,
                "scripts/run_workflow.py",
                "--workflow",
                args.workflow,
                "--problem-text",
                problem["latex"],
                "--problem-id",
                problem_id,
                "--run-id",
                run_id,
                "--run-name",
                _child_run_name(args.run_name, args.workflow, problem),
                "--output",
                str(outputs_root),
            ]
            for override in args.input or []:
                cmd.extend(["--input", override])
            if args.monitor:
                cmd.extend(["--monitor", "--monitor-model", args.monitor_model])
            with log_path.open("a", encoding="utf-8") as log:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    # Own session/process group per child run, so the dashboard's
                    # Stop button can terminate one run's group (the worker + its
                    # sandbox CLIs) without killing sibling problems or the batch.
                    start_new_session=True,
                )
                code = await proc.wait()
            await update_problem(
                problem_id,
                status="ok" if code == 0 else "error",
                finished_at=_now(),
                returncode=code,
                log=f"logs/{log_path.name}",
            )
            return code

    codes = await asyncio.gather(*(run_one(problem) for problem in problems))
    meta["status"] = "ok" if all(code == 0 for code in codes) else "error"
    meta["finished_at"] = _now()
    manifest["finished_at"] = meta["finished_at"]
    _write_metadata(metadata_path, meta)
    return 0 if meta["status"] == "ok" else 1


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
