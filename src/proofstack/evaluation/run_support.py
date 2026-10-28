"""Shared process lifetime, checkpoints, and usage accounting for evaluation runs."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from proofstack.atomic import write_text_atomic
from proofstack.provider_accounting import provider_usage_snapshot


def create_run_directory(repository_root: Path, category: str) -> Path:
    directory = repository_root / "outputs" / category
    directory.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-")
    return Path(tempfile.mkdtemp(prefix=prefix, dir=directory))


def write_json(path: Path, value):
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _interrupt(signum, frame):
    raise KeyboardInterrupt("Received SIGTERM")


@contextmanager
def interrupt_on_sigterm():
    previous = signal.signal(signal.SIGTERM, _interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def subprocess_environment(repository_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    paths = [str(repository_root / "src"), str(repository_root)]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _signal_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _stop_process(process):
    _signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The leader may exit before its tool subprocesses do.
        _signal_group(process, signal.SIGKILL)
    process.wait(timeout=5)


def run_process(command, log_path: Path, *, cwd: Path, timeout_seconds: float,
                env: dict[str, str] | None = None) -> dict:
    started = time.monotonic()
    result = {"returncode": None, "timed_out": False, "error": None}
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            result["error"] = f"Cannot launch subprocess: {error}"
        else:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                result.update(timed_out=True, error=f"Exceeded {timeout_seconds:g} seconds")
                _stop_process(process)
            except BaseException:
                _stop_process(process)
                raise
            result["returncode"] = process.returncode
    result["runtime_seconds"] = time.monotonic() - started
    return result


def _usage_flags(run_directory):
    unavailable = estimated = False
    for line in (run_directory / "events.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("kind") != "model.call":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            unavailable |= bool(payload.get("usage_unavailable"))
            estimated |= bool(payload.get("cost_estimated"))
    return unavailable, estimated


def usage_report(run_directory: Path, *, completed: bool) -> dict:
    try:
        snapshot = provider_usage_snapshot(run_directory)
        unavailable, estimated = _usage_flags(run_directory)
        unavailable |= any(item["payload"].get("usage_unavailable", False)
                           for item in snapshot.adjustments)
        estimated |= any(item["payload"].get("cost_estimated", False)
                         for item in snapshot.adjustments)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"cost_usd": None, "tokens": None, "accounting_complete": False,
                "accounting_error": f"{type(error).__name__}: {error}"}
    return {"cost_usd": snapshot.cost_usd, "tokens": snapshot.tokens,
            "accounting_complete": completed and not snapshot.damaged and not unavailable,
            "cost_estimated": estimated, "usage_unavailable": unavailable,
            "damaged_usage_lines": snapshot.damaged}
