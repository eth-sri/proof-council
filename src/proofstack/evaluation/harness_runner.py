"""Run one workflow subprocess and collect its declared proof and usage."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

from proofstack.evaluation.run_support import run_process, subprocess_environment, usage_report


@dataclass(frozen=True)
class HarnessSettings:
    repository_root: Path
    workflow: Path
    output: Path
    budget_usd: float
    timeout_seconds: float
    answer_field: str | None = None


def _metadata(run_directory):
    path = run_directory / "run-metadata.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Workflow metadata must be an object")
    return value


def _answer_value(outputs, field):
    fields = [field] if field else ["solution", "answer_tex", "solution_tex"]
    for name in fields:
        value = outputs
        for key in name.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, str) and value.strip():
            return name, value
    raise ValueError(f"No nonempty proof in workflow output fields: {', '.join(fields)}")


def _looks_like_file(value):
    if "\n" in value or "\r" in value:
        return False
    return (value.startswith(("/", "./", "../", "~/"))
            or bool(re.search(r"\.(tex|txt|md|latex)$", value, flags=re.IGNORECASE))
            or ("/" in value and not any(char.isspace() for char in value)))


def _read_proof(value, run_directory):
    path = Path(value)
    candidate = path if path.is_absolute() else run_directory / path
    try:
        exists = candidate.is_file() if "\n" not in value and len(value) < 4096 else False
    except OSError:
        exists = False
    if not exists and not _looks_like_file(value):
        return value
    resolved = candidate.resolve()
    if not resolved.is_relative_to(run_directory.resolve()):
        raise ValueError("Proof file must be contained in its workflow run directory")
    proof = resolved.read_text(encoding="utf-8")
    if not proof.strip():
        raise ValueError("Workflow proof file is empty")
    return proof


def _collect_result(result, settings, run_directory) -> bool:
    try:
        metadata = _metadata(run_directory)
    except (OSError, ValueError) as error:
        result.update(status="timeout" if result["timed_out"] else "solver_error",
                      error=result["error"] or f"Cannot read workflow metadata: {error}")
        return False
    result["workflow_status"] = metadata.get("status")
    outputs = metadata.get("outputs")
    succeeded = (result["returncode"] == 0 and not result["timed_out"]
                 and metadata.get("status") == "ok" and isinstance(outputs, dict)
                 and not outputs.get("error"))
    if not succeeded:
        result.update(status="timeout" if result["timed_out"] else "solver_error",
                      error=result["error"] or metadata.get("error")
                      or (outputs.get("error") if isinstance(outputs, dict) else None)
                      or f"Workflow exited with code {result['returncode']}")
        return False
    try:
        field, value = _answer_value(outputs, settings.answer_field)
        result.update(solution=_read_proof(value, run_directory),
                      answer_field=field, status="solved")
    except (OSError, ValueError) as error:
        result.update(status="missing_proof", error=str(error))
    return True


def _workflow_command(settings, problem_path, execution_id):
    return [sys.executable, str(settings.repository_root / "scripts/run_workflow.py"),
            "--workflow", str(settings.workflow.resolve()), "--problem", str(problem_path),
            "--problem-id", execution_id, "--run-id", execution_id,
            "--output", str(settings.output.resolve() / "solver-runs"),
            "--budget-usd", str(settings.budget_usd),
            "--input", f"max_wallclock_s={settings.timeout_seconds}"]


def _validate_settings(settings, record):
    for name in ("budget_usd", "timeout_seconds"):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    if settings.answer_field is not None and (
            not isinstance(settings.answer_field, str)
            or any(not part.strip() for part in settings.answer_field.split("."))):
        raise ValueError("answer_field must be a nonempty output field path")
    for key in ("problem_id", "problem"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"{key} must be nonempty text")


def run_harness(settings: HarnessSettings, record: dict) -> dict:
    _validate_settings(settings, record)
    execution_id = "problem-" + hashlib.sha256(record["problem_id"].encode()).hexdigest()
    run_directory = settings.output.resolve() / "solver-runs" / execution_id
    run_directory.mkdir(parents=True, exist_ok=False)
    problem_path, log_path = run_directory / "problem.txt", run_directory / "workflow.log"
    problem_path.write_text(record["problem"], encoding="utf-8")
    result = {"problem_id": record["problem_id"], "execution_id": execution_id,
              "solution": None, "run_directory": str(run_directory), "log_path": str(log_path)}
    started = time.monotonic()
    interrupted = False
    try:
        result.update(run_process(_workflow_command(settings, problem_path, execution_id), log_path,
                                 cwd=settings.repository_root, timeout_seconds=settings.timeout_seconds,
                                 env=subprocess_environment(settings.repository_root)))
    except KeyboardInterrupt:
        interrupted = True
        result.update(returncode=None, timed_out=False, error="Interrupted",
                      runtime_seconds=time.monotonic() - started)
    workflow_completed = _collect_result(result, settings, run_directory)
    if interrupted:
        result.update(status="interrupted", solution=None)
        workflow_completed = False
    result.update(usage_report(run_directory, completed=workflow_completed))
    return result
