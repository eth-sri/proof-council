"""First Proof AWS harness adapter.

The harness starts this container without CLI arguments. This adapter reads
First Proof input JSON, runs one configured ProofCouncil workflow per problem,
and writes the required aggregate files under /data/output.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from proofstack.agents.writeup_loop import _GateCanceller


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from proofstack.latex_contract import (  # noqa: E402
    DEFAULT_FIRSTPROOF_PAGE_LIMIT,
    normalize_submission_latex,
    repair_common_missing_packages as _repair_common_missing_packages,
    normalize_documentclass,
    strip_forbidden_formatting_commands,
    strip_forbidden_packages,
)
from _secret_paths import (  # noqa: E402
    is_secret_dir_name as _is_retrieval_secret_dir,
    is_secret_file_name as _is_retrieval_secret_file,
)

DEFAULT_INPUT_PATH = Path("/data/input/input.json")
DEFAULT_OUTPUT_DIR = Path("/data/output")
DEFAULT_TMP_PROBLEM_DIR = Path("/tmp/firstproof_problems")
LIVE_SNAPSHOT_INTERVAL_S = 60.0


def _batch3_cleanup_seconds(workflow):
    from proofstack.registry import load_preset
    from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow

    preset = load_preset(workflow)
    default = FirstProofBatch3Workflow.Inputs.model_fields["partial_cleanup_seconds"].default
    return float(preset.build_inputs().get("partial_cleanup_seconds", default))


@dataclass(frozen=True)
class Settings:
    input_path: Path
    output_dir: Path
    workflow: str
    max_parallel: int
    page_limit: int
    budget_usd_per_question: float
    n_rounds: int
    round_batch_size: int
    compute_codex_sandbox: str
    runner_script: str
    warnings: list[str]
    # Soft internal deadline. ``None`` disables the deadline path
    # (the harness's outer ``timeout`` SIGKILL is still the hard bound).
    deadline_seconds: float | None
    run_namespace: str = ""
    adaptive_continuation: bool = False
    adaptive_max_rounds: int = 200
    deadline_at: float | None = None
    research_deadline_at: float | None = None
    batch3: bool = False
    compute_max_parallel_workers: int | None = None
    author_parallelism: int | None = None
    compute_disabled_reason: str | None = None
    cleanup_disabled_reason: str | None = None
    stop_requested: asyncio.Event = field(default_factory=asyncio.Event, compare=False, repr=False)
    event_snapshots: dict[Path, _EventSnapshot] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class Problem:
    ordinal: int
    original_id: str
    safe_id: str
    text: str
    input_error: str | None
    problem_path: Path
    log_path: Path
    output_tex_path: Path
    run_id: str


@dataclass
class StageResult:
    stage_index: int
    n_rounds: int
    status: str
    returncode: int | None
    solved: bool
    started_at: str
    finished_at: str
    duration_seconds: float
    staged_solution_path: Path | None = None
    error: str | None = None


@dataclass
class ProblemResult:
    original_id: str
    safe_id: str
    status: str
    returncode: int | None
    run_id: str
    log_path: Path
    output_tex_path: Path
    latex: str
    started_at: str
    finished_at: str
    duration_seconds: float
    error: str | None = None
    rejected_solution_path: Path | None = None
    solved: bool = False
    stages: list[StageResult] = field(default_factory=list)
    in_progress: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_int_env(name: str, default: int, warnings: list[str], *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError:
        warnings.append(f"{name}={raw!r} is not an integer; using {default}.")
        return default
    if value < minimum:
        warnings.append(f"{name}={raw!r} is below {minimum}; using {default}.")
        return default
    return value


def _read_float_env(name: str, default: float, warnings: list[str], *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except ValueError:
        warnings.append(f"{name}={raw!r} is not a number; using {default}.")
        return default
    if value < minimum:
        warnings.append(f"{name}={raw!r} is below {minimum}; using {default}.")
        return default
    return value


def _read_bool_env(name: str, default: bool, warnings: list[str]) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    warnings.append(f"{name}={raw!r} is not a boolean; using {default}.")
    return default


def _resolve_deadline_seconds(warnings: list[str]) -> float | None:
    """Compute the soft internal deadline.

    Priority:
      1. ``FIRSTPROOF_DEADLINE_MINUTES`` env override (use as-is, in min).
      2. ``hardware.json:timeout_minutes`` from the repo root —
         deadline = ``max((timeout_minutes - 5) * 60, 60)`` so a 5 min
         drain window is reserved before the harness's outer
         ``timeout`` SIGKILL fires.
      3. ``None`` (no internal deadline; the harness SIGKILL is the only
         bound).
    """
    raw = os.environ.get("FIRSTPROOF_DEADLINE_MINUTES")
    if raw not in (None, ""):
        try:
            return max(float(raw) * 60.0, 60.0)
        except ValueError:
            warnings.append(f"FIRSTPROOF_DEADLINE_MINUTES={raw!r} is not a number; ignoring.")
    hw_path = REPO_ROOT / "hardware.json"
    if hw_path.exists():
        try:
            data = json.loads(hw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"hardware.json present but not parseable: {exc}")
            return None
        raw_timeout = data.get("timeout_minutes") if isinstance(data, dict) else None
        if isinstance(raw_timeout, (int, float)) and raw_timeout > 0:
            return max((float(raw_timeout) - 5.0) * 60.0, 60.0)
        if raw_timeout is not None:
            warnings.append(
                f"hardware.json:timeout_minutes={raw_timeout!r} is not a positive number; "
                "no internal deadline set."
            )
    return None


def _settings() -> Settings:
    warnings: list[str] = []
    deadline_seconds = _resolve_deadline_seconds(warnings)
    run_namespace = os.environ.get("FIRSTPROOF_RUN_NAMESPACE")
    if not run_namespace:
        run_namespace = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    workflow = os.environ.get("FIRSTPROOF_WORKFLOW") or "firstproof_submission"
    from proofstack.agents.firstproof_batch3 import FirstProofBatch3Workflow
    from proofstack.registry import PresetError, load_preset

    preset = None
    try:
        preset = load_preset(workflow)
        batch3 = issubclass(preset.workflow_cls, FirstProofBatch3Workflow)
    except (PresetError, OSError, ValueError, KeyError, ImportError) as exc:
        warnings.append(f"Could not inspect workflow {workflow!r}: {exc}")
        batch3 = False
    compute_max_parallel_workers = None
    if batch3:
        default_workers = int(preset.inputs.get(
            "compute_max_parallel_workers",
            preset.workflow_cls.Inputs.model_fields["compute_max_parallel_workers"].default,
        ))
        if default_workers < 1:
            raise ValueError("Batch 3 requires a positive compute_max_parallel_workers")
        compute_max_parallel_workers = _read_int_env(
            "FIRSTPROOF_COMPUTE_MAX_PARALLEL_WORKERS", default_workers, warnings,
        )
    author_parallelism = None
    if preset is not None and "author_parallelism" in preset.workflow_cls.Inputs.model_fields:
        default_parallelism = preset.inputs.get(
            "author_parallelism",
            preset.workflow_cls.Inputs.model_fields["author_parallelism"].default,
        )
        if (
            isinstance(default_parallelism, bool)
            or not isinstance(default_parallelism, int)
            or default_parallelism < 1
        ):
            raise ValueError("Workflow requires a positive integer author_parallelism")
        author_parallelism = _read_int_env(
            "FIRSTPROOF_AUTHOR_PARALLELISM", default_parallelism, warnings,
        )
    settings = Settings(
        input_path=Path(os.environ.get("FIRSTPROOF_INPUT_PATH") or DEFAULT_INPUT_PATH),
        output_dir=Path(os.environ.get("FIRSTPROOF_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR),
        workflow=workflow,
        max_parallel=_read_int_env(
            "FIRSTPROOF_MAX_PARALLEL",
            10 if batch3 else 6,
            warnings,
        ),
        page_limit=_read_int_env(
            "FIRSTPROOF_PAGE_LIMIT",
            16 if batch3 else DEFAULT_FIRSTPROOF_PAGE_LIMIT,
            warnings,
        ),
        budget_usd_per_question=_read_float_env(
            "FIRSTPROOF_BUDGET_USD_PER_QUESTION",
            1050.0 if batch3 else 1000.0,
            warnings,
        ),
        n_rounds=_read_int_env(
            "FIRSTPROOF_N_ROUNDS",
            50 if batch3 else 10,
            warnings,
        ),
        round_batch_size=_read_int_env(
            "FIRSTPROOF_ROUND_BATCH_SIZE",
            5,
            warnings,
        ),
        adaptive_continuation=_read_bool_env(
            "FIRSTPROOF_ADAPTIVE_CONTINUATION",
            not batch3,
            warnings,
        ),
        adaptive_max_rounds=_read_int_env(
            "FIRSTPROOF_ADAPTIVE_MAX_ROUNDS",
            200,
            warnings,
        ),
        compute_codex_sandbox=(
            os.environ.get("FIRSTPROOF_COMPUTE_CODEX_SANDBOX")
            or "docker-bypass"
        ),
        runner_script=os.environ.get("FIRSTPROOF_RUN_WORKFLOW_SCRIPT") or "scripts/run_workflow.py",
        warnings=warnings,
        deadline_seconds=deadline_seconds,
        run_namespace=run_namespace,
        batch3=batch3,
        compute_max_parallel_workers=compute_max_parallel_workers,
        author_parallelism=author_parallelism,
    )
    if batch3 and settings.adaptive_continuation:
        warnings.append("Batch 3 runs the complete pipeline once; disabling FIRSTPROOF_ADAPTIVE_CONTINUATION")
        settings = replace(settings, adaptive_continuation=False)
    if batch3 and not 1 <= settings.page_limit <= 16:
        warnings.append("Batch 3 requires a page limit between 1 and 16; clamping FIRSTPROOF_PAGE_LIMIT")
        settings = replace(settings, page_limit=max(1, min(16, settings.page_limit)))
    if batch3 and settings.n_rounds > 500:
        warnings.append("Batch 3 supports at most 500 rounds; clamping FIRSTPROOF_N_ROUNDS")
        settings = replace(settings, n_rounds=500)
    return settings


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".firstproof_write_test"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    # First Proof mounts a fresh /data/output per run, but a local
    # re-run can reuse the directory. Drop stale healthcheck artefacts
    # so usage rows from a previous run don't leak into
    # ``token_usage.jsonl``. Remove the obsolete proceed-sentinel too;
    # it no longer bypasses a failed startup gate.
    for stale in ("healthcheck.json", "healthcheck.proceed"):
        target = path / stale
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
    # Do NOT pre-create ``logs/`` and ``workflow_runs/`` here. The
    # bundled run.sh decides success by counting entries in /data/output;
    # eager subdirs would let a crashed adapter look successful before
    # any solution exists. Those subdirs are created lazily by
    # ``_run_problem`` (for ``logs/``) and by the workflow subprocess
    # (for ``workflow_runs/``) only once we actually commit to running.


def _chmod_for_removal(path: Path, warnings: list[str]) -> None:
    try:
        if path.is_symlink():
            return
        if path.is_dir():
            for root, dirnames, filenames in os.walk(path, topdown=False, followlinks=False):
                root_path = Path(root)
                for filename in filenames:
                    child = root_path / filename
                    try:
                        if not child.is_symlink():
                            os.chmod(child, 0o600)
                    except OSError as exc:
                        warnings.append(f"could not prepare unsafe output file for removal {child}: {exc}")
                for dirname in dirnames:
                    child = root_path / dirname
                    try:
                        if not child.is_symlink():
                            os.chmod(child, 0o700)
                    except OSError as exc:
                        warnings.append(f"could not prepare unsafe output directory for removal {child}: {exc}")
            os.chmod(path, 0o700)
        elif path.exists():
            os.chmod(path, 0o600)
    except OSError as exc:
        warnings.append(f"could not prepare unsafe output path for removal {path}: {exc}")


def _remove_output_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _remove_output_path_robust(path: Path, warnings: list[str], label: str) -> bool:
    try:
        _remove_output_path(path)
        return True
    except FileNotFoundError:
        return True
    except OSError as exc:
        warnings.append(f"could not remove {label} output path {path}: {exc}; retrying after chmod")

    _chmod_for_removal(path, warnings)
    try:
        _remove_output_path(path)
        return True
    except FileNotFoundError:
        return True
    except OSError as exc:
        warnings.append(f"RETRIEVAL_UNSAFE: could not remove {label} output path {path}: {exc}")
        return False


def _finalize_output_permissions(output_dir: Path) -> list[str]:
    """Make /data/output safe for the scaffold's recursive retrieval.

    The First Proof runner copies /home/ubuntu/output as the ``ubuntu``
    user after the container exits. Workflow scratch directories can
    contain root-owned or deliberately 0600 files, so scrub known secret
    homes first and then make the remaining tree world-readable.
    """

    warnings: list[str] = []
    unsafe_paths: set[Path] = set()
    if not output_dir.exists():
        return warnings

    def on_walk_error(exc: OSError) -> None:
        warnings.append(f"could not inspect output path {exc.filename}: {exc}")

    for root, dirnames, filenames in os.walk(output_dir, topdown=True, followlinks=False, onerror=on_walk_error):
        root_path = Path(root)
        kept_dirs: list[str] = []
        for dirname in dirnames:
            child = root_path / dirname
            try:
                is_symlink = child.is_symlink()
            except OSError as exc:
                warnings.append(f"could not inspect output directory {child}: {exc}")
                unsafe_paths.add(child)
                continue
            if is_symlink:
                if not _remove_output_path_robust(child, warnings, "symlink"):
                    unsafe_paths.add(child)
                continue
            if _is_retrieval_secret_dir(dirname):
                if not _remove_output_path_robust(child, warnings, "secret"):
                    unsafe_paths.add(child)
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            child = root_path / filename
            try:
                is_symlink = child.is_symlink()
                is_file = child.is_file()
            except OSError as exc:
                warnings.append(f"could not inspect output file {child}: {exc}")
                unsafe_paths.add(child)
                continue
            should_remove = (
                is_symlink
                or _is_retrieval_secret_file(filename)
                or not is_file
            )
            if not should_remove:
                continue
            if not _remove_output_path_robust(child, warnings, "unsafe"):
                unsafe_paths.add(child)

    try:
        os.chmod(output_dir, 0o755)
    except OSError as exc:
        warnings.append(f"could not chmod output directory {output_dir}: {exc}")

    for root, dirnames, filenames in os.walk(output_dir, topdown=True, followlinks=False, onerror=on_walk_error):
        root_path = Path(root)
        kept_dirs = []
        for dirname in dirnames:
            child = root_path / dirname
            if child in unsafe_paths:
                warnings.append(f"RETRIEVAL_UNSAFE: skipping chmod for unrecovered unsafe output directory {child}")
                continue
            try:
                is_symlink = child.is_symlink()
            except OSError as exc:
                warnings.append(f"could not inspect output directory {child}: {exc}")
                continue
            if is_symlink:
                if not _remove_output_path_robust(child, warnings, "late symlink"):
                    unsafe_paths.add(child)
                    warnings.append(f"RETRIEVAL_UNSAFE: skipping chmod for unrecovered unsafe output directory {child}")
                continue
            if _is_retrieval_secret_dir(dirname):
                if not _remove_output_path_robust(child, warnings, "late secret"):
                    unsafe_paths.add(child)
                    warnings.append(f"RETRIEVAL_UNSAFE: skipping chmod for unrecovered unsafe output directory {child}")
                continue
            try:
                os.chmod(child, 0o755)
            except OSError as exc:
                warnings.append(f"could not chmod output directory {child}: {exc}")
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs
        for filename in filenames:
            child = root_path / filename
            if child in unsafe_paths:
                warnings.append(f"RETRIEVAL_UNSAFE: skipping chmod for unrecovered unsafe output file {child}")
                continue
            try:
                is_symlink = child.is_symlink()
                is_file = child.is_file()
            except OSError as exc:
                warnings.append(f"could not inspect output file {child}: {exc}")
                continue
            should_remove = (
                is_symlink
                or _is_retrieval_secret_file(filename)
                or not is_file
            )
            if should_remove:
                if not _remove_output_path_robust(child, warnings, "late unsafe"):
                    unsafe_paths.add(child)
                    warnings.append(f"RETRIEVAL_UNSAFE: skipping chmod for unrecovered unsafe output file {child}")
                continue
            try:
                os.chmod(child, 0o644)
            except OSError as exc:
                warnings.append(f"could not chmod output file {child}: {exc}")

    return warnings


def _load_problem_items(path: Path) -> list[Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("problems")
    else:
        items = raw
    if not isinstance(items, list):
        raise ValueError("input JSON must be a list or an object with a 'problems' list")
    return items


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_id(value: str, fallback: str) -> str:
    cleaned = _SAFE_ID_RE.sub("_", value.strip()).strip("._-")
    if not cleaned:
        cleaned = fallback
    return cleaned[:120].strip("._-") or fallback


def _unique_safe_id(base: str, seen: set[str]) -> str:
    candidate = base
    count = 1
    while candidate in seen:
        count += 1
        suffix = f"-{count}"
        candidate = f"{base[: 120 - len(suffix)]}{suffix}"
    seen.add(candidate)
    return candidate


def _problem_text(item: Any) -> tuple[str, str | None]:
    if not isinstance(item, dict):
        return "", "problem entry is not an object"
    raw = item.get("latex")
    if raw is None:
        return "", "problem has no 'latex' field"
    text = str(raw).strip()
    if not text:
        return "", "problem latex is empty"
    return text, None


def _parse_problems(items: list[Any], settings: Settings) -> list[Problem]:
    tmp_dir = Path(os.environ.get("FIRSTPROOF_TMP_PROBLEM_DIR") or DEFAULT_TMP_PROBLEM_DIR)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    problems: list[Problem] = []
    for idx, item in enumerate(items, start=1):
        default_id = f"prob-{idx:03d}"
        if isinstance(item, dict) and "id" in item:
            original_id = str(item.get("id") or "").strip() or default_id
        else:
            original_id = default_id
        safe_base = _safe_id(original_id, fallback=default_id)
        safe = _unique_safe_id(safe_base, seen)
        text, input_error = _problem_text(item)
        run_suffix = _safe_id(settings.run_namespace, fallback="run")
        run_id = (
            f"firstproof-{safe}-{run_suffix}"
            if settings.run_namespace
            else f"firstproof-{safe}"
        )
        problem_path = tmp_dir / f"{safe}.tex"
        problem_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
        problems.append(
            Problem(
                ordinal=idx,
                original_id=original_id,
                safe_id=safe,
                text=text,
                input_error=input_error,
                problem_path=problem_path,
                log_path=settings.output_dir / "logs" / f"{safe}.log",
                output_tex_path=settings.output_dir / f"{safe}.tex",
                run_id=run_id,
            )
        )
    return problems


def _format_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def _strip_forbidden_packages(tex: str) -> tuple[str, list[str]]:
    return strip_forbidden_packages(tex)


def _strip_forbidden_formatting_commands(tex: str, removals: list[str] | None) -> str:
    cleaned, command_removals = strip_forbidden_formatting_commands(tex)
    if removals is not None:
        removals.extend(command_removals)
    return cleaned


def _repair_normalized_latex(tex: str, removals: list[str] | None) -> str:
    return _repair_common_missing_packages(
        _strip_forbidden_formatting_commands(tex, removals),
        removals,
    )


def _ensure_complete_latex(tex: str, *, removals: list[str] | None = None) -> str:
    return normalize_submission_latex(tex, removals=removals)


def _normalize_documentclass(tex: str, *, removals: list[str] | None = None) -> str:
    return normalize_documentclass(tex, removals=removals)


def _run_validation_command(
    cmd: list[str], *, timeout_s: float, deadline: float | None = None,
    canceller: _GateCanceller | None = None, cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    from proofstack.agents.writeup_loop import _kill_compiler

    if canceller is not None and canceller.stopped():
        raise InterruptedError("submission compilation cancelled")
    if deadline is not None:
        timeout_s = min(timeout_s, deadline - time.monotonic())
    if timeout_s <= 0:
        raise subprocess.TimeoutExpired(cmd, timeout_s)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    if canceller is not None:
        canceller.track(proc)
    try:
        if deadline is not None:
            timeout_s = min(timeout_s, max(0.0, deadline - time.monotonic()))
        stdout, stderr = proc.communicate(timeout=timeout_s)
        if canceller is not None and canceller.stopped():
            raise InterruptedError("submission compilation cancelled")
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except BaseException:
        _kill_compiler(proc)
        with contextlib.suppress(Exception):
            proc.communicate(timeout=5)
        raise
    finally:
        if canceller is not None:
            canceller.untrack()


def _count_pdf_pages(
    pdf_path: Path, *, deadline: float | None = None, canceller: _GateCanceller | None = None,
) -> int:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import fitz  # type: ignore[import-not-found]

        with fitz.open(pdf_path) as doc:
            return int(doc.page_count)
    except Exception:
        try:
            proc = _run_validation_command(
                ["pdfinfo", str(pdf_path)], timeout_s=30, deadline=deadline, canceller=canceller,
            )
            if proc.returncode == 0:
                match = re.search(rb"^Pages:\s*(\d+)\s*$", proc.stdout, re.MULTILINE)
                if match:
                    return int(match.group(1))
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        try:
            data = pdf_path.read_bytes()
        except OSError:
            return 0
        return len(re.findall(rb"/Type\s*/Page(?!s)", data))


def _compile_exact_latex(
    tex: str,
    *,
    page_limit: int,
    timeout_s: int = 120,
    deadline: float | None = None,
    canceller: _GateCanceller | None = None,
) -> tuple[bool, str]:
    """Compile the exact output twice, sharing one timeout across both passes."""
    with tempfile.TemporaryDirectory(prefix="firstproof_final_tex_") as work_str:
        work = Path(work_str)
        tex_path = work / "solution.tex"
        tex_path.write_text(tex, encoding="utf-8")
        timeout_deadline = time.monotonic() + timeout_s
        deadline = min(timeout_deadline, deadline) if deadline is not None else timeout_deadline
        try:
            # Unresolved references on the first pass can change pagination.
            for _ in range(2):
                proc = _run_validation_command(
                    [
                        "pdflatex",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "solution.tex",
                    ],
                    cwd=work,
                    timeout_s=timeout_s,
                    deadline=deadline,
                    canceller=canceller,
                )
                if proc.returncode != 0:
                    break
            pdf_path = work / "solution.pdf"
            if proc.returncode == 0 and pdf_path.exists():
                page_count = _count_pdf_pages(pdf_path, deadline=deadline, canceller=canceller)
                if canceller is not None and canceller.stopped():
                    raise InterruptedError("submission compilation cancelled")
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired("submission validation", timeout_s)
                if page_count <= 0:
                    return False, "pdflatex produced a PDF but page count could not be determined"
                if page_count > page_limit:
                    return (
                        False,
                        f"pdflatex produced {page_count} pages, above page_limit={page_limit}",
                    )
                return True, f"compiled with {page_count} pages"
        except FileNotFoundError:
            return False, "pdflatex binary not found"
        except subprocess.TimeoutExpired:
            return False, "submission compilation timed out or reached the batch deadline"
        except InterruptedError:
            return False, "submission compilation cancelled"

        log_path = work / "solution.log"
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
            log_tail = stderr[-4000:] or "(no pdflatex log produced)"
        return False, f"pdflatex exited with {proc.returncode}; log tail:\n{log_tail}"


async def _verify_exact_latex_for_submission(
    problem: Problem,
    settings: Settings,
    tex: str,
) -> tuple[bool, str]:
    from proofstack.agents.writeup_loop import _GateCanceller

    if settings.stop_requested.is_set():
        raise asyncio.CancelledError("operator stop")
    if settings.deadline_at is not None and time.monotonic() >= settings.deadline_at:
        raise asyncio.CancelledError("batch deadline")
    canceller = _GateCanceller()
    worker = asyncio.create_task(asyncio.to_thread(
        _compile_exact_latex,
        tex,
        page_limit=settings.page_limit,
        deadline=settings.deadline_at,
        canceller=canceller,
    ))
    try:
        ok, detail = await asyncio.shield(worker)
    except asyncio.CancelledError:
        canceller.cancel()
        # Reap the compiler and remove its temporary files before returning.
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(worker)
        raise
    if settings.stop_requested.is_set():
        raise asyncio.CancelledError("operator stop")
    if settings.deadline_at is not None and time.monotonic() >= settings.deadline_at:
        raise asyncio.CancelledError("batch deadline")
    if not ok:
        await _write_log_line(
            problem.log_path,
            f"[{_utc_now()}] final LaTeX verification failed: {detail}\n",
        )
    return ok, detail


def _fallback_tex(problem: Problem, reason: str) -> str:
    return (
        "\\documentclass[12pt]{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\begin{document}\n"
        "\\section*{First Proof fallback solution}\n"
        "The workflow did not produce a final solution for problem "
        f"\\texttt{{{_latex_escape(problem.original_id)}}}.\n\n"
        f"Reason: {_latex_escape(reason)}\n\n"
        "The detailed log is available at "
        f"\\texttt{{{_latex_escape(str(problem.log_path))}}}.\n"
        "\\end{document}\n"
    )


async def _write_log_line(path: Path, line: str) -> None:
    def write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(line.encode("utf-8", errors="replace"))
            handle.flush()

    await asyncio.to_thread(write)


async def _terminate_workflow_subprocess(proc: asyncio.subprocess.Process, *, grace_s: float = 5.0) -> None:
    """SIGTERM-then-SIGKILL the workflow process group (best-effort).

    ``_run_subprocess`` spawns the workflow child in a new session
    (``start_new_session=True``) so any descendants — CAS subprocesses,
    codex CLI, pdflatex invocations the workflow kicks off — share its
    pgid. On internal-deadline cancellation the parent task aborts but
    the child can keep burning model budget; this helper terminates the
    whole group.

    We attempt the group kill *even if the direct child has already
    exited*: a short-lived launcher (shell wrapper, npm shim) can exit
    while leaving long-running descendants (codex, a CAS subprocess)
    alive in the same pgid. Skipping the group SIGTERM/SIGKILL on
    ``proc.returncode is not None`` would leak those descendants until
    container teardown.
    """
    pid = proc.pid
    # Phase 1: group SIGTERM. Reaches descendants even when the direct
    # child has already exited.
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    if proc.returncode is None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_s)
        except asyncio.TimeoutError:
            pass
    # Phase 2: group SIGKILL. Always attempted, for the same reason.
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    if proc.returncode is None:
        try:
            await proc.wait()
        except ProcessLookupError:
            pass


def _max_scheduled_rounds(settings: Settings) -> int:
    initial_total = max(int(settings.n_rounds), 1)
    if not settings.adaptive_continuation:
        return initial_total
    return max(initial_total, int(settings.adaptive_max_rounds))


def _stage_has_followup_rounds(settings: Settings, n_rounds: int) -> bool:
    return int(n_rounds) < _max_scheduled_rounds(settings)


def _round_schedule(settings: Settings) -> list[int]:
    if settings.batch3:
        return [settings.n_rounds]
    total = _max_scheduled_rounds(settings)
    batch = max(int(settings.round_batch_size), 1)
    if batch >= total:
        return [total]
    schedule = list(range(batch, total, batch))
    if not schedule or schedule[-1] != total:
        schedule.append(total)
    return schedule


@asynccontextmanager
async def _problem_slot(settings: Settings, semaphore: asyncio.Semaphore):
    """Limit research concurrency without queueing deadline cleanup behind it."""
    cutoff = settings.research_deadline_at
    if not settings.batch3 or cutoff is None:
        async with semaphore:
            yield
        return
    held = False
    timer = None

    def release():
        nonlocal held
        if held:
            held = False
            semaphore.release()

    async def release_at_cutoff():
        await asyncio.sleep(max(0.0, cutoff - time.monotonic()))
        release()

    try:
        if time.monotonic() < cutoff:
            try:
                await asyncio.wait_for(semaphore.acquire(), timeout=cutoff - time.monotonic())
                held = True
            except TimeoutError:
                pass
        if held:
            timer = asyncio.create_task(release_at_cutoff())
        yield
    finally:
        if timer is not None:
            timer.cancel()
            try:
                await timer
            except asyncio.CancelledError:
                pass
        release()


async def _run_subprocess(
    problem: Problem,
    settings: Settings,
    *,
    n_rounds: int,
    restart_from: str | None = None,
    stage_index: int = 1,
) -> int:
    if settings.stop_requested.is_set():
        raise asyncio.CancelledError("operator stop")
    runner = settings.runner_script
    cmd = [
        sys.executable,
        runner,
        "--workflow",
        settings.workflow,
        "--problem",
        str(problem.problem_path),
        "--problem-id",
        problem.safe_id,
        "--run-id",
        problem.run_id,
        "--run-name",
        f"FirstProof {problem.original_id}",
        "--output",
        str(settings.output_dir / "workflow_runs"),
        "--budget-usd",
        _format_number(settings.budget_usd_per_question),
        "--input",
        f"n_rounds={n_rounds}",
        "--input",
        f"page_limit={settings.page_limit}",
        "--input",
        f"compute_codex_sandbox={settings.compute_codex_sandbox}",
    ]
    if settings.compute_max_parallel_workers is not None:
        cmd.extend(["--input", f"compute_max_parallel_workers={settings.compute_max_parallel_workers}"])
    if settings.compute_disabled_reason is not None:
        cmd.extend(["--input", "enable_compute=false"])
    if settings.cleanup_disabled_reason is not None:
        cmd.extend(["--input", "cleanup_backend=api"])
    if settings.author_parallelism is not None:
        cmd.extend(["--input", f"author_parallelism={settings.author_parallelism}"])
    if settings.batch3 and settings.deadline_at is not None:
        remaining = settings.deadline_at - time.monotonic() - 60
        if remaining <= 0:
            raise TimeoutError("no container time left to start Batch 3 workflow")
        now, unix_now = time.monotonic(), time.time()
        research_at = settings.research_deadline_at
        if research_at is None:
            research_at = settings.deadline_at - 60 - _batch3_cleanup_seconds(settings.workflow)
        cmd.extend([
            "--input", f"max_wallclock_s={remaining}",
            "--input", f"research_deadline_unix_s={unix_now + research_at - now}",
            "--input", f"run_deadline_unix_s={unix_now + settings.deadline_at - now - 60}",
        ])
    stop_after_review_round = stage_index > 0 and _stage_has_followup_rounds(
        settings,
        n_rounds,
    )
    if stop_after_review_round:
        cmd.extend(["--input", "stop_after_review_round=true"])
    if restart_from:
        cmd.extend(["--restart-from", restart_from])
        cmd.extend(
            [
                "--additional-instructions",
                (
                    "Resume the interrupted Batch 3 workflow from its saved checkpoint, "
                    "retaining the original deadlines, cumulative budget and review gates."
                    if settings.batch3 else
                    "The previous pass did not reach Author/Critic agreement. "
                    f"Continue the existing run up to round {n_rounds}; focus on "
                    "resolving the remaining Critic objections instead of restarting."
                ),
            ]
        )
    await _write_log_line(
        problem.log_path,
        f"[{_utc_now()}] starting workflow stage {stage_index} with n_rounds={n_rounds}\n",
    )
    await _write_log_line(problem.log_path, f"command: {' '.join(cmd)}\n\n")

    if settings.stop_requested.is_set():
        raise asyncio.CancelledError("operator stop")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=REPO_ROOT,
        env=_workflow_env(settings.output_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # New POSIX session so os.killpg can reach the child's whole
        # subprocess tree (codex CLI, pdflatex, etc.) on cancellation
        # — see _terminate_workflow_subprocess.
        start_new_session=True,
    )
    assert proc.stdout is not None
    try:
        if settings.stop_requested.is_set():
            raise asyncio.CancelledError("operator stop")
        while True:
            chunk = await proc.stdout.read(8192)
            if not chunk:
                break

            def append(data: bytes = chunk) -> None:
                with problem.log_path.open("ab") as handle:
                    handle.write(data)
                    handle.flush()

            await asyncio.to_thread(append)
        code = await proc.wait()
    except asyncio.CancelledError:
        # Internal deadline reached, or the run_and_record task was
        # cancelled. Tear down the workflow process group so it doesn't
        # keep burning API budget after we've already moved on.
        await _write_log_line(
            problem.log_path,
            f"\n[{_utc_now()}] workflow cancelled; terminating subprocess group\n",
        )
        await _terminate_workflow_subprocess(proc, grace_s=15.0)
        raise
    finally:
        # Belt-and-suspenders: if the proc somehow left a child alive
        # (e.g. proc.wait() returned but a CAS child kept running), make
        # sure we tear it down. Cheap when there's nothing to do.
        if proc.returncode is None:
            await _terminate_workflow_subprocess(proc, grace_s=2.0)
    await _write_log_line(
        problem.log_path,
        f"\n[{_utc_now()}] workflow stage {stage_index} exited with {code}\n",
    )
    return int(code)


def _workflow_env(output_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    # First Proof's docker run does not mount a Docker socket or grant
    # privileged mode. Treat the submission container as the isolation
    # boundary and run ProofCouncil's internal CLI sandboxes as subprocesses.
    env.setdefault("PROOFSTACK_SANDBOX_BACKEND", "subprocess")
    # Route the legacy debug request logger into ``/data/output/logs/requests/``
    # so its per-request JSON files are retrieved by run.sh with the rest
    # of the submission. The default lands at ``/app/logs/requests`` which
    # the harness ignores.
    if output_dir is not None:
        env.setdefault(
            "MATHAGENTS_REQUEST_LOG_DIR",
            str(output_dir / "logs" / "requests"),
        )
    return env


def _read_run_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "run-metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _metadata_tex_candidates(run_dir: Path) -> list[Path]:
    metadata = _read_run_metadata(run_dir)
    if not metadata:
        return []
    candidates: list[Path] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
            return
        if isinstance(value, list):
            for nested in value:
                visit(nested)
            return
        if not isinstance(value, str) or not value.endswith(".tex"):
            return
        path = Path(value)
        if not path.is_absolute():
            path = run_dir / path
        candidates.append(path)

    visit(metadata.get("outputs"))
    return candidates


def _workflow_output_rejection(
    problem: Problem,
    settings: Settings,
    *,
    min_rounds_completed: int | None = None,
) -> str | None:
    run_dir = settings.output_dir / "workflow_runs" / problem.run_id
    metadata = _read_run_metadata(run_dir)
    if metadata.get("status") == "error":
        return "workflow metadata reported status=error"
    outputs = _workflow_outputs(problem, settings)
    if not isinstance(outputs, dict):
        return None
    output_error = outputs.get("error")
    if output_error:
        first_line = str(output_error).splitlines()[0]
        return f"workflow output reported error: {first_line}"
    completed = _workflow_rounds_completed(problem, settings)
    if (
        min_rounds_completed is not None
        and completed is not None
        and completed < min_rounds_completed
        and outputs.get("early_stopped") is not True
        and not _batch3_partial(problem, settings)
    ):
        return (
            f"workflow reported only {completed} completed rounds, "
            f"below required {min_rounds_completed}"
        )
    compiled = outputs.get("compiled")
    if compiled is False:
        return "workflow reported that the final LaTeX did not compile"
    pages = outputs.get("pages")
    if pages is None:
        return None
    try:
        page_count = int(pages)
    except (TypeError, ValueError):
        return None
    if page_count > settings.page_limit:
        return f"workflow reported {page_count} pages, above page_limit={settings.page_limit}"
    return None


def _workflow_outputs(problem: Problem, settings: Settings) -> dict[str, Any]:
    run_dir = settings.output_dir / "workflow_runs" / problem.run_id
    if settings.batch3:
        try:
            checkpoint = json.loads((run_dir / "batch3-output.json").read_text(encoding="utf-8"))
            return checkpoint if isinstance(checkpoint, dict) else {}
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            return {}
    outputs = _read_run_metadata(run_dir).get("outputs")
    return outputs if isinstance(outputs, dict) else {}


def _batch3_partial(problem: Problem, settings: Settings) -> bool:
    if not settings.batch3:
        return False
    outputs = _workflow_outputs(problem, settings)
    return (outputs.get("partial_ready") is True
            and outputs.get("submission_approved") is False
            and outputs.get("output_kind") == "partial_unreviewed")


def _workflow_budget_exhausted(problem: Problem, settings: Settings) -> str | None:
    run_dir = settings.output_dir / "workflow_runs" / problem.run_id
    metadata = _read_run_metadata(run_dir)
    outputs = metadata.get("outputs")
    raw_parts: list[str] = []
    for value in (
        metadata.get("error"),
        outputs.get("error") if isinstance(outputs, dict) else None,
    ):
        if value:
            raw_parts.append(str(value))
    text = "\n".join(raw_parts)
    if "BudgetExhausted" not in text and "Budget exhausted" not in text:
        return None
    first_line = text.splitlines()[0] if text else "budget exhausted"
    return f"workflow budget exhausted: {first_line}"


def _workflow_rounds_completed(problem: Problem, settings: Settings) -> int | None:
    outputs = _workflow_outputs(problem, settings)
    raw = outputs.get("rounds_completed")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _author_critic_agreed(problem: Problem, settings: Settings) -> bool:
    outputs = _workflow_outputs(problem, settings)
    if settings.batch3:
        return (outputs.get("submission_approved") is True
                and outputs.get("partial_ready") is not True
                and outputs.get("early_stopped") is True)
    return bool(outputs.get("early_stopped") is True)


def _published_solution(problem: Problem, settings: Settings) -> tuple[Path, str] | None:
    """Read one committed publication and validate the exact bytes returned."""
    from proofstack.agents.firstproof_batch3 import published_document_path

    run_dir = settings.output_dir / "workflow_runs" / problem.run_id
    outputs = _workflow_outputs(problem, settings)
    partial = (outputs.get("partial_ready") is True
               and outputs.get("submission_approved") is False
               and outputs.get("output_kind") == "partial_unreviewed")
    if not partial and (outputs.get("submission_approved") is not True or outputs.get("partial_ready") is True):
        return None
    if partial or settings.stop_requested.is_set() or (
        settings.deadline_at is not None and time.monotonic() >= settings.deadline_at
    ):
        try:
            valid_pages = 0 < int(outputs.get("pages", 0)) <= settings.page_limit
        except (TypeError, ValueError):
            valid_pages = False
        if outputs.get("compiled") is not True or not valid_pages:
            return None
    digest = outputs.get("partial_sha256" if partial else "submission_sha256")
    candidate = published_document_path(
        run_dir, problem.safe_id, partial=partial, digest=digest,
        version=outputs.get("publication_version"),
    )
    if candidate is None:
        return None
    try:
        document = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if hashlib.sha256(document.encode("utf-8")).hexdigest() != digest:
        return None
    # Never normalize or repair away from the published bytes.
    return (candidate, document) if _ensure_complete_latex(document) == document else None


def _find_solution_tex(problem: Problem, settings: Settings) -> Path | None:
    run_dir = settings.output_dir / "workflow_runs" / problem.run_id
    if settings.batch3:
        published = _published_solution(problem, settings)
        return published[0] if published is not None else None
    preferred = run_dir / "solutions" / f"{problem.safe_id}.tex"
    candidates = [preferred, *_metadata_tex_candidates(run_dir)]
    solutions_dir = run_dir / "solutions"
    if solutions_dir.exists():
        candidates.extend(sorted(solutions_dir.glob("*.tex")))
    workspace_answer = run_dir / "ac_workspaces" / problem.safe_id / "answer.tex"
    candidates.append(workspace_answer)
    workspaces_dir = run_dir / "ac_workspaces"
    if workspaces_dir.exists():
        candidates.extend(sorted(workspaces_dir.glob("*/answer.tex")))
    seen: set[Path] = set()
    for raw_candidate in candidates:
        try:
            candidate = raw_candidate.resolve()
        except (OSError, RuntimeError):
            candidate = raw_candidate
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _stage_solution_candidate(
    problem: Problem,
    settings: Settings,
    *,
    n_rounds: int,
    returncode: int | None,
) -> Path | None:
    if returncode != 0:
        return None
    completed = _workflow_rounds_completed(problem, settings)
    if (
        completed is not None
        and completed < n_rounds
        and not _author_critic_agreed(problem, settings)
        and not _batch3_partial(problem, settings)
    ):
        return None
    return _find_solution_tex(problem, settings)


def _preserve_stage_solution(
    problem: Problem,
    settings: Settings,
    *,
    n_rounds: int,
    solution_path: Path | None,
) -> Path | None:
    """Preserve the raw workflow answer after a round batch.

    This intentionally copies the workflow's file as-is. The top-level
    First Proof submission is still normalized and compile-checked later;
    these stage snapshots are for restart/debugging/post-mortem review.
    """
    if solution_path is None:
        return None
    try:
        if not solution_path.is_file():
            return None
    except OSError:
        return None
    stage_dir = settings.output_dir / "staged_solutions" / f"rounds-{n_rounds:03d}"
    dst = stage_dir / f"{problem.safe_id}.tex"
    try:
        stage_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(solution_path, dst)
    except OSError:
        return None
    return dst


async def _read_normalized_solution(problem: Problem, solution_path: Path) -> tuple[str, list[str]]:
    removals: list[str] = []
    latex = _ensure_complete_latex(
        solution_path.read_text(encoding="utf-8", errors="replace"),
        removals=removals,
    )
    if removals:
        await _write_log_line(
            problem.log_path,
            f"[{_utc_now()}] LaTeX normalization actions: "
            + "; ".join(removals)
            + "\n",
    )
    return latex, removals


async def _verified_solution_or_fallback(
    problem: Problem,
    settings: Settings,
    *,
    solution_path: Path | None,
    reason: str,
    fallback_status: str,
    solution_status: str,
    log_context: str,
) -> tuple[str, str, str, Path | None]:
    if solution_path is None:
        return _fallback_tex(problem, reason), fallback_status, reason, None
    try:
        latex, _removals = await _read_normalized_solution(problem, solution_path)
    except OSError as exc:
        read_reason = f"{reason}; found {solution_path} but could not read it: {exc}"
        return _fallback_tex(problem, read_reason), fallback_status, read_reason, None
    compiled, compile_detail = await _verify_exact_latex_for_submission(
        problem, settings, latex
    )
    if not compiled:
        verified_reason = (
            f"{reason}; found {solution_path} but adapter-normalized LaTeX "
            f"did not compile: {compile_detail.splitlines()[0]}"
        )
        rejected = _preserve_rejected_solution(
            problem, settings, solution_path, reason=verified_reason
        )
        return _fallback_tex(problem, verified_reason), fallback_status, verified_reason, rejected
    await _write_log_line(
        problem.log_path,
        f"[{_utc_now()}] {log_context}: shipping {solution_path} despite {reason}\n",
    )
    return latex, solution_status, reason, None


async def _ship_solution_or_fallback(
    problem: Problem,
    settings: Settings,
    *,
    reason: str,
    fallback_status: str,
    solution_status: str,
) -> tuple[str, str, str, Path | None]:
    if settings.stop_requested.is_set() or (
        settings.deadline_at is not None and time.monotonic() >= settings.deadline_at
    ):
        # A committed publication already passed compilation. Export only those
        # exact bytes, without starting new model work or a compiler on shutdown.
        stopped = settings.stop_requested.is_set()
        status = "operator_stopped" if stopped else "deadline_cancelled"
        reason = "operator stopped workflow" if stopped else "internal deadline reached"
        published = _published_solution(problem, settings) if settings.batch3 else None
        if published is not None:
            return published[1], status + "_with_solution", reason + "; exported validated publication", None
        try:
            latex = problem.output_tex_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            latex = _fallback_tex(problem, reason)
        return latex, status, reason + "; raw research retained for recovery", None
    return await _verified_solution_or_fallback(
        problem,
        settings,
        solution_path=_find_solution_tex(problem, settings),
        reason=reason,
        fallback_status=fallback_status,
        solution_status=solution_status,
        log_context="fail-open finalization",
    )


async def _stage_snapshot_latex(
    problem: Problem,
    settings: Settings,
    *,
    solution_path: Path | None,
    stage_status: str,
    stage_error: str | None,
) -> tuple[str, str, str | None, Path | None]:
    if solution_path is None:
        reason = stage_error or "stage completed but no current solution .tex was found"
        return _fallback_tex(problem, reason), "missing_solution", reason, None
    if stage_error:
        if stage_status == "budget_exhausted":
            return await _verified_solution_or_fallback(
                problem,
                settings,
                solution_path=solution_path,
                reason=stage_error,
                fallback_status="budget_exhausted",
                solution_status="budget_exhausted_with_solution",
                log_context="fail-open stage snapshot",
            )
        rejected = _preserve_rejected_solution(
            problem, settings, solution_path, reason=stage_error
        )
        return _fallback_tex(problem, stage_error), stage_status, stage_error, rejected
    try:
        latex, _removals = await _read_normalized_solution(problem, solution_path)
    except OSError as exc:
        reason = f"could not read stage solution .tex: {exc}"
        return _fallback_tex(problem, reason), "solution_read_error", reason, None
    compiled, compile_detail = await _verify_exact_latex_for_submission(
        problem, settings, latex
    )
    if not compiled:
        reason = (
            "adapter-normalized stage LaTeX did not compile: "
            + compile_detail.splitlines()[0]
        )
        rejected = _preserve_rejected_solution(
            problem, settings, solution_path, reason=reason
        )
        return _fallback_tex(problem, reason), "solution_contract_error", reason, rejected
    return latex, stage_status, stage_error, None


_FINAL_FALLBACK_STATUSES = frozenset(
    {
        "deadline_cancelled",
        "adapter_error",
        "workflow_error",
        "budget_exhausted",
        "budget_exhausted_with_solution",
        "missing_solution",
        "solution_contract_error",
        "solution_read_error",
        "deadline_cancelled_with_solution",
        "adapter_error_with_solution",
        "workflow_error_with_solution",
    }
)


async def _prefer_best_stage_solution(
    problem: Problem,
    *,
    latex: str,
    status: str,
    reason: str | None,
    best_stage_result: ProblemResult | None,
) -> tuple[str, str, str | None]:
    if status not in _FINAL_FALLBACK_STATUSES or best_stage_result is None:
        return latex, status, reason
    best_stage = best_stage_result.stages[-1] if best_stage_result.stages else None
    stage_label = (
        f"{best_stage.n_rounds} rounds"
        if best_stage is not None
        else "an earlier stage"
    )
    previous_reason = reason or status
    await _write_log_line(
        problem.log_path,
        f"[{_utc_now()}] finalization fallback: shipping best compiling "
        f"stage solution from {stage_label} after final status {status}: "
        f"{previous_reason}\n",
    )
    return (
        best_stage_result.latex,
        "best_stage_solution",
        f"{previous_reason}; shipped best compiling stage solution from {stage_label}",
    )


async def _prefer_best_stage_latex_for_snapshot(
    problem: Problem,
    *,
    latex: str,
    status: str,
    reason: str | None,
    best_stage_result: ProblemResult | None,
) -> tuple[str, str | None]:
    if status not in _FINAL_FALLBACK_STATUSES or best_stage_result is None:
        return latex, reason
    best_stage = best_stage_result.stages[-1] if best_stage_result.stages else None
    stage_label = (
        f"{best_stage.n_rounds} rounds"
        if best_stage is not None
        else "an earlier stage"
    )
    previous_reason = reason or status
    await _write_log_line(
        problem.log_path,
        f"[{_utc_now()}] stage snapshot fallback: keeping best compiling "
        f"stage solution from {stage_label} after stage status {status}: "
        f"{previous_reason}\n",
    )
    return (
        best_stage_result.latex,
        f"{previous_reason}; top-level output kept best compiling stage solution from {stage_label}",
    )


def _preserve_rejected_solution(
    problem: Problem,
    settings: Settings,
    solution_path: Path | None,
    *,
    reason: str,
) -> Path | None:
    """Copy a rejected real solution to a stable top-level artifact.

    The required ``/data/output/<id>.tex`` may need to be a fallback so
    the harness gets a compiling file, but we still want the actual
    workflow output for debugging and post-mortem review.
    """
    if solution_path is None:
        return None
    try:
        if not solution_path.is_file():
            return None
    except OSError:
        return None
    failed_dir = settings.output_dir / "failed_solutions"
    failed_dir.mkdir(parents=True, exist_ok=True)
    dst = failed_dir / f"{problem.safe_id}.tex"
    shutil.copyfile(solution_path, dst)
    note = failed_dir / f"{problem.safe_id}.reason.txt"
    note.write_text(reason + "\n", encoding="utf-8")

    run_dir = settings.output_dir / "workflow_runs" / problem.run_id
    compile_log = run_dir / "ac_workspaces" / problem.safe_id / ".ac" / "final-compile.log"
    if not compile_log.is_file():
        matches = sorted((run_dir / "ac_workspaces").glob("*/.ac/final-compile.log"))
        compile_log = matches[0] if matches else compile_log
    try:
        if compile_log.is_file():
            shutil.copyfile(compile_log, failed_dir / f"{problem.safe_id}.compile.log")
    except OSError:
        pass
    return dst


def _batch3_can_retry(problem: Problem, settings: Settings, returncode: int) -> bool:
    if (settings.stop_requested.is_set() or not settings.batch3
            or returncode in (0, 2, 130, 143, -signal.SIGINT, -signal.SIGTERM)):
        return False
    if _workflow_outputs(problem, settings).get("error_retryable") is False:
        return False
    if _workflow_budget_exhausted(problem, settings):
        return False
    # An interim partial is a crash fallback, not a completed research run.
    if (_workflow_outputs(problem, settings).get("submission_approved") is True
            and _find_solution_tex(problem, settings) is not None):
        return False
    if settings.deadline_at is not None and settings.deadline_at - time.monotonic() <= 60:
        return False
    from proofstack.agents.firstproof_batch3 import _Schedule, _problem_hash

    run_dir = settings.output_dir / "workflow_runs" / problem.run_id
    try:
        schedule = _Schedule.model_validate_json((run_dir / "batch3-schedule.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    # --restart-from restores recorded spend and clamps the new invocation to
    # this schedule, even if the very first Author call never saved research.
    return (schedule.problem_hash == _problem_hash(problem.text)
            and schedule.run_deadline_unix_s > time.time())


async def _run_problem(
    problem: Problem,
    settings: Settings,
    semaphore: asyncio.Semaphore,
    on_stage_complete: Callable[[ProblemResult], Awaitable[None]] | None = None,
) -> ProblemResult:
    started = _utc_now()
    start_time = time.monotonic()
    problem.log_path.parent.mkdir(parents=True, exist_ok=True)

    # Stub the per-problem .tex with a fallback document up front. If the
    # adapter is SIGKILL'd (24h cap) or otherwise cancelled while this
    # problem is in flight, the bundled run.sh will still find a
    # syntactically valid 12pt article in /data/output/ for this problem
    # rather than nothing. Real success overwrites this at the end.
    try:
        _write_text_atomic(
            problem.output_tex_path,
            _fallback_tex(problem, "workflow did not finish before container exit"),
        )
    except OSError:
        # If we can't even write the stub, _run_problem will fail in a
        # more obvious way later; nothing useful to do here.
        pass

    if problem.input_error:
        await _write_log_line(
            problem.log_path,
            f"[{started}] not running workflow: {problem.input_error}\n",
        )
        latex = _fallback_tex(problem, problem.input_error)
        _write_text_atomic(problem.output_tex_path, latex)
        finished = _utc_now()
        return ProblemResult(
            original_id=problem.original_id,
            safe_id=problem.safe_id,
            status="input_error",
            returncode=None,
            run_id=problem.run_id,
            log_path=problem.log_path,
            output_tex_path=problem.output_tex_path,
            latex=latex,
            started_at=started,
            finished_at=finished,
            duration_seconds=round(time.monotonic() - start_time, 3),
            error=problem.input_error,
        )

    stages: list[StageResult] = []
    best_stage_result: ProblemResult | None = None
    returncode: int | None = None
    round_schedule = _round_schedule(settings)
    for stage_index, n_rounds in enumerate(round_schedule, start=1):
        stage_started = _utc_now()
        stage_start_time = time.monotonic()
        restart_from = problem.run_id if stage_index > 1 else None
        try:
            async with _problem_slot(settings, semaphore):
                if settings.stop_requested.is_set():
                    raise asyncio.CancelledError("operator stop")
                returncode = await _run_subprocess(
                    problem,
                    settings,
                    n_rounds=n_rounds,
                    restart_from=restart_from,
                    stage_index=stage_index,
                )
                if _batch3_can_retry(problem, settings, returncode):
                    await _write_log_line(
                        problem.log_path,
                        f"[{_utc_now()}] Batch 3 early workflow failure (exit {returncode}); "
                        "checkpoint retry 1/1 with the original deadlines and budget\n",
                    )
                    returncode = await _run_subprocess(
                        problem,
                        settings,
                        n_rounds=n_rounds,
                        restart_from=problem.run_id,
                        stage_index=stage_index,
                    )
        except asyncio.CancelledError:
            returncode = None
            reason = ("operator stopped workflow" if settings.stop_requested.is_set()
                      else "internal deadline cancelled workflow before it finished")
            await _write_log_line(problem.log_path, f"\n[{_utc_now()}] adapter cancelled: {reason}\n")
            latex, status, error, rejected = await _ship_solution_or_fallback(
                problem,
                settings,
                reason=reason,
                fallback_status="operator_stopped" if settings.stop_requested.is_set() else "deadline_cancelled",
                solution_status="operator_stopped_with_solution" if settings.stop_requested.is_set() else "deadline_cancelled_with_solution",
            )
            latex, status, error = await _prefer_best_stage_solution(
                problem,
                latex=latex,
                status=status,
                reason=error,
                best_stage_result=best_stage_result,
            )
            _write_text_atomic(problem.output_tex_path, latex)
            finished = _utc_now()
            return ProblemResult(
                original_id=problem.original_id,
                safe_id=problem.safe_id,
                status=status,
                returncode=returncode,
                run_id=problem.run_id,
                log_path=problem.log_path,
                output_tex_path=problem.output_tex_path,
                latex=latex,
                started_at=started,
                finished_at=finished,
                duration_seconds=round(time.monotonic() - start_time, 3),
                error=error,
                rejected_solution_path=rejected,
                solved=False,
                stages=stages,
            )
        except Exception as exc:
            returncode = None
            await _write_log_line(problem.log_path, f"\n[{_utc_now()}] adapter error: {type(exc).__name__}: {exc}\n")
            reason = f"adapter failed to run workflow: {type(exc).__name__}: {exc}"
            latex, status, error, rejected = await _ship_solution_or_fallback(
                problem,
                settings,
                reason=reason,
                fallback_status="adapter_error",
                solution_status="adapter_error_with_solution",
            )
            latex, status, error = await _prefer_best_stage_solution(
                problem,
                latex=latex,
                status=status,
                reason=error,
                best_stage_result=best_stage_result,
            )
            _write_text_atomic(problem.output_tex_path, latex)
            finished = _utc_now()
            return ProblemResult(
                original_id=problem.original_id,
                safe_id=problem.safe_id,
                status=status,
                returncode=returncode,
                run_id=problem.run_id,
                log_path=problem.log_path,
                output_tex_path=problem.output_tex_path,
                latex=latex,
                started_at=started,
                finished_at=finished,
                duration_seconds=round(time.monotonic() - start_time, 3),
                error=error,
                rejected_solution_path=rejected,
                solved=False,
                stages=stages,
            )

        metadata_reason = _workflow_output_rejection(
            problem,
            settings,
            min_rounds_completed=n_rounds,
        )
        budget_reason = _workflow_budget_exhausted(problem, settings)
        solution_path = (
            _find_solution_tex(problem, settings)
            if budget_reason
            else _stage_solution_candidate(
                problem,
                settings,
                n_rounds=n_rounds,
                returncode=returncode,
            )
        )
        stage_solution_path = _preserve_stage_solution(
            problem,
            settings,
            n_rounds=n_rounds,
            solution_path=(
                solution_path
                if metadata_reason is None or budget_reason is not None
                else None
            ),
        )
        solved = (
            returncode == 0
            and solution_path is not None
            and metadata_reason is None
            and _author_critic_agreed(problem, settings)
        )
        if returncode != 0:
            stage_status = "workflow_error"
            stage_error = f"workflow subprocess exited with return code {returncode}"
        elif budget_reason:
            stage_status = "budget_exhausted"
            stage_error = budget_reason
        elif metadata_reason:
            stage_status = "solution_contract_error"
            stage_error = metadata_reason
        elif solved:
            stage_status = "solved"
            stage_error = None
        elif solution_path is not None and _batch3_partial(problem, settings):
            stage_status = "partial_unreviewed"
            stage_error = None
        else:
            stage_status = "needs_more_rounds"
            stage_error = None
        snapshot_latex, snapshot_status, snapshot_error, snapshot_rejected = (
            await _stage_snapshot_latex(
                problem,
                settings,
                solution_path=solution_path,
                stage_status=stage_status,
                stage_error=stage_error,
            )
        )
        if snapshot_error is not None and stage_status in {"needs_more_rounds", "solved", "partial_unreviewed"}:
            stage_status = snapshot_status
            stage_error = snapshot_error
            solved = False
        elif snapshot_status == "budget_exhausted_with_solution":
            stage_status = snapshot_status
        snapshot_latex, stage_error = await _prefer_best_stage_latex_for_snapshot(
            problem,
            latex=snapshot_latex,
            status=stage_status,
            reason=stage_error,
            best_stage_result=best_stage_result,
        )
        should_stop = solved or returncode != 0 or budget_reason is not None
        has_next_stage = stage_index < len(round_schedule)
        continue_next = has_next_stage and not should_stop
        stages.append(
            StageResult(
                stage_index=stage_index,
                n_rounds=n_rounds,
                status=stage_status,
                returncode=returncode,
                solved=solved,
                started_at=stage_started,
                finished_at=_utc_now(),
                duration_seconds=round(time.monotonic() - stage_start_time, 3),
                staged_solution_path=stage_solution_path,
                error=stage_error,
            )
        )
        _write_text_atomic(problem.output_tex_path, snapshot_latex)
        snapshot_result = ProblemResult(
            original_id=problem.original_id,
            safe_id=problem.safe_id,
            status=stage_status,
            returncode=returncode,
            run_id=problem.run_id,
            log_path=problem.log_path,
            output_tex_path=problem.output_tex_path,
            latex=snapshot_latex,
            started_at=started,
            finished_at=_utc_now(),
            duration_seconds=round(time.monotonic() - start_time, 3),
            error=stage_error,
            rejected_solution_path=snapshot_rejected,
            solved=solved,
            stages=list(stages),
            in_progress=continue_next,
        )
        if stage_status in {"needs_more_rounds", "solved", "partial_unreviewed"} and stage_error is None:
            best_stage_result = snapshot_result
        if on_stage_complete is not None:
            await on_stage_complete(snapshot_result)
        if should_stop:
            break

    solution_path = _find_solution_tex(problem, settings)
    rejected_solution_path: Path | None = None
    final_budget_reason = _workflow_budget_exhausted(problem, settings)
    final_min_rounds_completed = stages[-1].n_rounds if stages else settings.n_rounds
    if returncode != 0:
        reason = f"workflow subprocess exited with return code {returncode}"
        latex, status, reason, rejected_solution_path = await _ship_solution_or_fallback(
            problem,
            settings,
            reason=reason,
            fallback_status="workflow_error",
            solution_status="workflow_error_with_solution",
        )
    elif final_budget_reason:
        reason = final_budget_reason
        latex, status, reason, rejected_solution_path = await _ship_solution_or_fallback(
            problem,
            settings,
            reason=reason,
            fallback_status="budget_exhausted",
            solution_status="budget_exhausted_with_solution",
        )
    elif solution_path is None:
        reason = "workflow completed but no solution .tex was found"
        status = "missing_solution"
        latex = _fallback_tex(problem, reason)
    elif metadata_reason := _workflow_output_rejection(
        problem,
        settings,
        min_rounds_completed=final_min_rounds_completed,
    ):
        reason = metadata_reason
        status = "solution_contract_error"
        latex = _fallback_tex(problem, reason)
        rejected_solution_path = _preserve_rejected_solution(
            problem, settings, solution_path, reason=reason
        )
    else:
        try:
            latex, removals = await _read_normalized_solution(problem, solution_path)
        except OSError as exc:
            reason = f"could not read solution .tex: {exc}"
            status = "solution_read_error"
            latex = _fallback_tex(problem, reason)
        else:
            reason = None
            status = "partial_unreviewed" if _batch3_partial(problem, settings) else "ok"
            compiled, compile_detail = await _verify_exact_latex_for_submission(
                problem, settings, latex
            )
            if not compiled:
                reason = (
                    "adapter-normalized final LaTeX did not compile: "
                    + compile_detail.splitlines()[0]
                )
                status = "solution_contract_error"
                latex = _fallback_tex(problem, reason)
                rejected_solution_path = _preserve_rejected_solution(
                    problem, settings, solution_path, reason=reason
                )
    latex, status, reason = await _prefer_best_stage_solution(
        problem,
        latex=latex,
        status=status,
        reason=reason,
        best_stage_result=best_stage_result,
    )
    _write_text_atomic(problem.output_tex_path, latex)
    finished = _utc_now()
    return ProblemResult(
        original_id=problem.original_id,
        safe_id=problem.safe_id,
        status=status,
        returncode=returncode,
        run_id=problem.run_id,
        log_path=problem.log_path,
        output_tex_path=problem.output_tex_path,
        latex=latex,
        started_at=started,
        finished_at=finished,
        duration_seconds=round(time.monotonic() - start_time, 3),
        error=reason,
        rejected_solution_path=rejected_solution_path,
        solved=status == "ok" and _author_critic_agreed(problem, settings),
        stages=stages,
    )


async def _exception_result(
    problem: Problem,
    exc: BaseException,
    settings: Settings,
    *,
    started_at: str | None = None,
    start_time: float | None = None,
) -> ProblemResult:
    started = started_at or _utc_now()
    reason = f"adapter error: {type(exc).__name__}: {exc}"
    try:
        await _write_log_line(problem.log_path, f"\n[{started}] {reason}\n")
    except OSError:
        pass
    latex, status, error, rejected = await _ship_solution_or_fallback(
        problem,
        settings,
        reason=reason,
        fallback_status="adapter_error",
        solution_status="adapter_error_with_solution",
    )
    try:
        _write_text_atomic(problem.output_tex_path, latex)
    except OSError as write_exc:
        error = f"{error}; failed to write final .tex: {write_exc}"
    finished = _utc_now()
    duration = round(time.monotonic() - start_time, 3) if start_time is not None else 0.0
    return ProblemResult(
        original_id=problem.original_id,
        safe_id=problem.safe_id,
        status=status,
        returncode=None,
        run_id=problem.run_id,
        log_path=problem.log_path,
        output_tex_path=problem.output_tex_path,
        latex=latex,
        started_at=started,
        finished_at=finished,
        duration_seconds=duration,
        error=error,
        rejected_solution_path=rejected,
    )


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _infer_provider(model: Any) -> str | None:
    if not model:
        return None
    text = str(model).lower()
    for provider in (
        "openai",
        "anthropic",
        "google",
        "gemini",
        "xai",
        "glm",
        "deepseek",
        "moonshot",
        "openrouter",
        "together",
        "stepfun",
    ):
        if provider in text:
            return "google" if provider == "gemini" else provider
    if text.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if "claude" in text:
        return "anthropic"
    return None


def _usage_record(problem: Problem, event: dict[str, Any]) -> dict[str, Any] | None:
    kind = str(event.get("kind") or "")
    # Phase costs summarize model.call events already included in the export.
    if kind in {"batch3.phase_end", "ac.author.delegate.wave_done", "ac.author.subagent_done"}:
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    merged = {**usage, **payload}
    model = _first_present(merged, ("model", "model_name", "model_id"))
    input_tokens = _first_present(merged, ("input_tokens", "in_tokens", "prompt_tokens"))
    output_tokens = _first_present(merged, ("output_tokens", "out_tokens", "completion_tokens"))
    reasoning_tokens = _first_present(
        merged,
        ("reasoning_tokens", "reasoning_output_tokens", "reasoning_out_tokens"),
    )
    total_tokens = _first_present(merged, ("total_tokens",))
    if total_tokens is None:
        try:
            if input_tokens is not None and output_tokens is not None:
                total_tokens = int(input_tokens) + int(output_tokens)
        except (TypeError, ValueError):
            total_tokens = None
    cost_usd = _first_present(merged, ("cost_usd", "cost"))
    has_usage = any(
        value is not None
        for value in (input_tokens, output_tokens, reasoning_tokens, total_tokens, cost_usd)
    )
    usage_like = "usage" in kind or kind in {"model.call", "multiturn.end"}
    if not has_usage and not usage_like:
        return None
    record: dict[str, Any] = {
        "problem_id": problem.original_id,
        "safe_id": problem.safe_id,
        "event_type": kind,
    }
    provider = _first_present(merged, ("provider", "api"))
    if model is not None:
        record["model"] = model
    record["provider"] = provider or _infer_provider(model)
    if input_tokens is not None:
        record["input_tokens"] = input_tokens
    if output_tokens is not None:
        record["output_tokens"] = output_tokens
    if reasoning_tokens is not None:
        record["reasoning_tokens"] = reasoning_tokens
    if total_tokens is not None:
        record["total_tokens"] = total_tokens
    if cost_usd is not None:
        record["cost_usd"] = cost_usd
    if merged.get("usage_unavailable"):
        record["usage_unavailable"] = True
    if not has_usage:
        record["raw_event"] = event
    return record


def _collect_healthcheck_usage(settings: Settings) -> tuple[list[dict[str, Any]], list[str]]:
    """Pull token/cost records from the healthcheck report, if it ran.

    First Proof's spec requires logging tokens for *every* API call. The
    preflight probes count even though they don't produce solutions.
    """
    report_path = settings.output_dir / "healthcheck.json"
    if not report_path.exists():
        return [], []
    warnings: list[str] = []
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"healthcheck.json could not be parsed: {exc}")
        return [], warnings
    records: list[dict[str, Any]] = []
    for probe in report.get("probes") or []:
        if not isinstance(probe, dict):
            continue
        record = {
            "problem_id": f"__healthcheck__{probe.get('role', 'unknown')}",
            "safe_id": "__healthcheck__",
            "event_type": "healthcheck.probe",
            "model": probe.get("model_ref"),
            "provider": _infer_provider(probe.get("model_ref")),
            "input_tokens": probe.get("input_tokens", 0),
            "output_tokens": probe.get("output_tokens", 0),
            "reasoning_tokens": probe.get("reasoning_tokens", 0),
            "total_tokens": int(probe.get("input_tokens", 0) or 0)
            + int(probe.get("output_tokens", 0) or 0),
            "cost_usd": probe.get("cost_usd", 0.0),
        }
        records.append(record)
    return records, warnings


def _empty_problem_health() -> dict:
    return {"helpers_finished": 0, "helpers_completed": 0, "helpers_timed_out": 0,
            "helpers_artifact_degraded": 0, "last_activity": None}


@dataclass
class _EventSnapshot:
    identity: tuple[int, int] | None = None
    offset: int = 0
    line_number: int = 0
    mtime_ns: int = 0
    anchor: bytes = b""
    records: list[dict] = field(default_factory=list)
    model_events: list[dict] = field(default_factory=list)
    health: dict = field(default_factory=_empty_problem_health)
    warnings: list[str] = field(default_factory=list)

    def consume(self, problem: Problem, line: bytes, *, provisional=False) -> None:
        if not line.strip():
            return
        try:
            event = json.loads(line)
        except (ValueError, UnicodeError):
            if not provisional:
                self.warnings.append(f"{problem.original_id}: invalid JSON event at line {self.line_number}")
            return
        if not isinstance(event, dict):
            return
        record = _usage_record(problem, event)
        if record is not None:
            self.records.append(record)
        kind = event.get("kind")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if kind == "model.call":
            # Reconciliation only needs billed calls, not tool transcripts.
            self.model_events.append({"kind": kind, "call_id": event.get("call_id"), "payload": {
                k: payload.get(k) for k in ("cost_usd", "in_tokens", "out_tokens", "reasoning_tokens")
            }})
        self.health["last_activity"] = {"kind": kind, "timestamp": event.get("timestamp", event.get("ts"))}
        if kind == "ac.author.subagent_done":
            self.health["helpers_finished"] += 1
            self.health["helpers_completed"] += not bool(payload.get("error"))
            self.health["helpers_timed_out"] += "deadline" in str(payload.get("error", ""))
            self.health["helpers_artifact_degraded"] += bool(payload.get("checkpoint_errors") or payload.get("artifact_error"))


def _problem_event_snapshot(problem: Problem, settings: Settings) -> _EventSnapshot:
    path = settings.output_dir / "workflow_runs" / problem.run_id / "events.jsonl"
    cached = settings.event_snapshots.setdefault(path, _EventSnapshot())
    tail = b""
    warning = None
    try:
        with path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            identity = (stat.st_dev, stat.st_ino)
            handle.seek(max(0, cached.offset - len(cached.anchor)))
            anchor = handle.read(len(cached.anchor))
            if (cached.identity != identity or stat.st_size < cached.offset or anchor != cached.anchor
                    or (stat.st_size == cached.offset and stat.st_mtime_ns != cached.mtime_ns)):
                cached = _EventSnapshot(identity=identity)
                settings.event_snapshots[path] = cached
            handle.seek(cached.offset)
            # Bound this snapshot to the size observed at open, even while writers append.
            while handle.tell() < stat.st_size:
                line = handle.readline(stat.st_size - handle.tell())
                if not line:
                    break
                if not line.endswith(b"\n"):
                    tail = line
                    break
                cached.line_number += 1
                cached.consume(problem, line)
                cached.offset = handle.tell()
            handle.seek(max(0, cached.offset - 256))
            cached.anchor = handle.read(min(cached.offset, 256))
            cached.mtime_ns = stat.st_mtime_ns
    except FileNotFoundError:
        cached = _EventSnapshot()
        settings.event_snapshots.pop(path, None)
        warning = f"{problem.original_id}: no events.jsonl found at {path}"
    except OSError as exc:
        warning = f"{problem.original_id}: could not read token events: {exc}"
    view = replace(cached, records=list(cached.records), model_events=list(cached.model_events),
                   health=dict(cached.health), warnings=list(cached.warnings))
    if tail:
        # A valid unterminated final event is visible, but not committed to the
        # cursor: the next poll re-reads it in case the writer was interrupted.
        view.consume(problem, tail, provisional=True)
    if warning:
        view.warnings.append(warning)
    return view


def _collect_token_usage(problems: list[Problem], settings: Settings, *, snapshots=None) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for problem in problems:
        events_path = settings.output_dir / "workflow_runs" / problem.run_id / "events.jsonl"
        snapshot = snapshots[problem.run_id] if snapshots is not None else _problem_event_snapshot(problem, settings)
        records.extend(snapshot.records)
        warnings.extend(snapshot.warnings)
        from mathagents.provider_trace import usage_adjustments
        try:
            for event in usage_adjustments(events_path.parent, snapshot.model_events):
                record = _usage_record(problem, event)
                if record is not None:
                    records.append(record)
                if event["payload"].get("usage_unavailable"):
                    warnings.append(f"{problem.original_id}: provider usage remains unresolved for call {event['call_id']}")
        except (OSError, ValueError) as exc:
            warnings.append(f"{problem.original_id}: provider accounting could not be reconciled: {exc}")
    if not records:
        warnings.append("no token or cost usage events were found")
    return records, warnings


def _token_totals(records: list[dict[str, Any]]) -> dict[str, float | int]:
    totals: dict[str, float | int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }
    for record in records:
        for key in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
            value = record.get(key)
            try:
                totals[key] = int(totals[key]) + int(value)
            except (TypeError, ValueError):
                continue
        value = record.get("cost_usd")
        try:
            totals["cost_usd"] = float(totals["cost_usd"]) + float(value)
        except (TypeError, ValueError):
            continue
    totals["cost_usd"] = round(float(totals["cost_usd"]), 8)
    return totals


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    """``_write_json`` plus rename-from-tmpfile so a SIGKILL mid-write
    cannot leave the file in a half-written state."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_text_atomic(path: Path, text: str) -> None:
    """Rename-from-tmpfile text writer for per-problem .tex artifacts.

    Without this, a SIGKILL mid-``write_text`` could leave a truncated
    .tex on disk; the harness would still see ``OUTPUT_COUNT > 0`` and
    report SUCCESS, and the grader would receive a half-document. The
    tmp filename includes the pid so concurrent problem tasks can't
    collide on it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _aggregate_payloads(
    problems: list[Problem],
    results: list[ProblemResult | None],
    settings: Settings,
    overall_started: str,
    overall_start_time: float,
    *,
    in_progress: bool,
    deadline_reached: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Compute the JSON shapes for solutions.json and run_summary.json
    plus the list of token_usage.jsonl rows, given the current set of
    completed problem results (some of which may still be ``None`` while
    the run is in flight).

    ``in_progress=True`` marks ``run_summary.json`` so a consumer can
    tell a partial-snapshot from the final summary.
    """
    def _latex_for_pending(problem: Problem) -> str:
        """Read the per-problem stub written at the start of _run_problem,
        falling back to a freshly-generated fallback document if the
        stub isn't on disk (shouldn't happen, but defensive)."""
        try:
            if problem.output_tex_path.exists():
                return problem.output_tex_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        return _fallback_tex(
            problem, "workflow did not finish before container exit"
        )

    solutions_payload = {
        "solutions": [
            {
                "id": result.original_id if result else problem.original_id,
                # When a result is present, prefer its latex field. When
                # a problem hasn't completed (None) or completed but
                # somehow produced empty latex, fall back to the stub
                # that _run_problem wrote up front. A partial run thus
                # ships valid fallback solutions for the unfinished
                # problems instead of empty strings.
                "latex": (
                    result.latex
                    if (result is not None and result.latex)
                    else _latex_for_pending(problem)
                ),
            }
            for problem, result in zip(problems, results)
        ]
    }

    finalized = [r for r in results if r is not None and not r.in_progress]
    snapshots = {p.run_id: _problem_event_snapshot(p, settings) for p in problems}
    token_records, token_warnings = _collect_token_usage(problems, settings, snapshots=snapshots)
    healthcheck_records, healthcheck_warnings = _collect_healthcheck_usage(settings)
    token_records_all = healthcheck_records + token_records
    token_warnings.extend(healthcheck_warnings)
    token_totals = _token_totals(token_records_all)

    finished = _utc_now()
    duration = round(time.monotonic() - overall_start_time, 3)
    warnings = [*settings.warnings, *token_warnings]
    summary_payload = {
        "started_at": overall_started,
        "finished_at": finished,
        "duration_seconds": duration,
        "in_progress": in_progress,
        "updated_at": finished,
        "operator_stopped": settings.stop_requested.is_set(),
        "deadline_reached": deadline_reached,
        "deadline_seconds": settings.deadline_seconds,
        "workflow": settings.workflow,
        "max_parallel": settings.max_parallel,
        "page_limit": settings.page_limit,
        "n_rounds": settings.n_rounds,
        "round_batch_size": settings.round_batch_size,
        "adaptive_continuation": settings.adaptive_continuation,
        "adaptive_max_rounds": settings.adaptive_max_rounds,
        "round_schedule": _round_schedule(settings),
        "compute_codex_sandbox": settings.compute_codex_sandbox,
        "compute_max_parallel_workers": settings.compute_max_parallel_workers,
        "compute_disabled_reason": settings.compute_disabled_reason,
        "cleanup_disabled_reason": settings.cleanup_disabled_reason,
        "author_parallelism": settings.author_parallelism,
        "budget_usd_per_question": settings.budget_usd_per_question,
        "total_budget_usd_requested": settings.budget_usd_per_question * len(problems),
        "problem_count": len(problems),
        "completed_count": len(finalized),
        "totals": token_totals,
        "token_counting_convention": {
            "input_tokens": "Provider input tokens; Google includes promptTokenCount plus toolUsePromptTokenCount.",
            "output_tokens": (
                "Billable output tokens including reasoning/thinking. Google candidatesTokenCount "
                "and thoughtsTokenCount are combined by the adapter. Do not add reasoning again."
            ),
            "reasoning_tokens": (
                "internal-reasoning tokens. Surfaced via output_tokens_details.reasoning_tokens (OpenAI), "
                "completion_tokens_details.reasoning_tokens (OpenAI Chat-Completions o-family), "
                "output_tokens_details.thinking_tokens (Anthropic), thoughtsTokenCount (Google), "
                "or codex JSONL reasoning_out_tokens (Compute Worker). Zero when not separately reported."
            ),
            "total_tokens": (
                "When the provider reports a total_tokens field, that is used verbatim. Otherwise we "
                "synthesize input_tokens + output_tokens. Reasoning is already included in output."
            ),
            "cost_usd": "USD cost per call, computed from each provider's rates declared in configs/models/.",
        },
        "per_problem": [
            {**(_summary_problem(result, settings.output_dir) if result else {
                "id": problem.original_id,
                "status": "running" if (settings.output_dir / "workflow_runs" / problem.run_id / "events.jsonl").exists() else "pending",
            }), "health": snapshots[problem.run_id].health,
                "totals": _token_totals([record for record in token_records if record["safe_id"] == problem.safe_id])}
            for problem, result in zip(problems, results)
        ],
        "warnings": warnings,
    }
    return solutions_payload, summary_payload, token_records_all


def _live_problem_health(problem: Problem, settings: Settings) -> dict:
    return _problem_event_snapshot(problem, settings).health


async def _write_aggregates(
    problems: list[Problem],
    results: list[ProblemResult | None],
    settings: Settings,
    overall_started: str,
    overall_start_time: float,
    *,
    in_progress: bool,
    deadline_reached: bool = False,
) -> None:
    """Build + write solutions.json, run_summary.json, token_usage.jsonl
    atomically. Safe to call repeatedly during the run; see
    ``_aggregate_payloads`` for the shape contract.
    """
    def _do_writes() -> None:
        solutions_payload, summary_payload, token_records_all = _aggregate_payloads(
            problems, results, settings, overall_started, overall_start_time,
            in_progress=in_progress, deadline_reached=deadline_reached,
        )
        _write_json_atomic(settings.output_dir / "solutions.json", solutions_payload)
        _write_json_atomic(settings.output_dir / "run_summary.json", summary_payload)
        token_path = settings.output_dir / "token_usage.jsonl"
        tmp = token_path.with_suffix(token_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for record in token_records_all:
                handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        os.replace(tmp, token_path)

    writer = asyncio.create_task(asyncio.to_thread(_do_writes))
    try:
        await asyncio.shield(writer)
    except asyncio.CancelledError:
        # Keep the caller's aggregate lock until the thread exits. Otherwise a
        # late live snapshot can overwrite the final summary after shutdown.
        await writer
        raise


def _summary_problem(result: ProblemResult, output_dir: Path) -> dict[str, Any]:
    try:
        log_path = str(result.log_path.relative_to(output_dir))
    except ValueError:
        log_path = str(result.log_path)
    try:
        tex_path = str(result.output_tex_path.relative_to(output_dir))
    except ValueError:
        tex_path = str(result.output_tex_path)
    out = {
        "id": result.original_id,
        "safe_id": result.safe_id,
        "status": result.status,
        "returncode": result.returncode,
        "solved": result.solved,
        "in_progress": result.in_progress,
        "run_id": result.run_id,
        "log_path": log_path,
        "output_tex_path": tex_path,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "duration_seconds": result.duration_seconds,
        "stages": [
            _summary_stage(stage, output_dir)
            for stage in result.stages
        ],
    }
    if result.error:
        out["error"] = result.error
    if result.rejected_solution_path is not None:
        try:
            out["rejected_solution_path"] = str(
                result.rejected_solution_path.relative_to(output_dir)
            )
        except ValueError:
            out["rejected_solution_path"] = str(result.rejected_solution_path)
    return out


def _summary_stage(stage: StageResult, output_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "stage_index": stage.stage_index,
        "n_rounds": stage.n_rounds,
        "status": stage.status,
        "returncode": stage.returncode,
        "solved": stage.solved,
        "started_at": stage.started_at,
        "finished_at": stage.finished_at,
        "duration_seconds": stage.duration_seconds,
    }
    if stage.error:
        out["error"] = stage.error
    if stage.staged_solution_path is not None:
        try:
            out["staged_solution_path"] = str(
                stage.staged_solution_path.relative_to(output_dir)
            )
        except ValueError:
            out["staged_solution_path"] = str(stage.staged_solution_path)
    return out


async def _bootstrap_codex_auth() -> tuple[bool, str | None]:
    """Best-effort one-time codex CLI login from ``OPENAI_API_KEY``.

    The Compute Worker (`src/proofstack/agents/ac/compute.py:Compute.setup`)
    copies ``~/.codex/auth.json`` into its per-invocation sandbox. In a
    fresh First Proof container that file does not exist — codex is
    installed globally via npm but has never been logged in. Without
    this bootstrap, every ``<compute_agent>`` call silently fails on
    auth.

    The bootstrap:
      * does nothing when ``OPENAI_API_KEY`` is missing;
      * does nothing when ``~/.codex/auth.json`` already exists (so
        local devs with a personal codex login are not overwritten);
      * does nothing when the ``codex`` binary is not on PATH;
      * runs ``codex login --with-api-key`` with the API key piped on
        stdin and a 30s timeout.

    Returns ``(did_anything, warning)``.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY not set; skipping codex auth bootstrap."
    auth_path = Path.home() / ".codex" / "auth.json"
    if auth_path.exists():
        return False, None
    codex_bin = shutil.which("codex")
    if codex_bin is None:
        return False, "codex CLI not on PATH; skipping codex auth bootstrap."
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            codex_bin, "login", "--with-api-key",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(api_key.encode()), timeout=30)
    except (TimeoutError, OSError) as exc:
        return False, f"codex login failed: {type(exc).__name__}: {exc}"
    finally:
        if proc is not None and proc.returncode is None:
            await _terminate_workflow_subprocess(proc, grace_s=2.0)
    if proc.returncode != 0:
        return False, (
            f"codex login --with-api-key exited {proc.returncode}: "
            f"{(stderr or stdout).decode(errors='replace').strip()[:400]}"
        )
    return True, None


async def _run_healthcheck(settings: Settings) -> Settings:
    """Batch 3 always checks the native execution host before paid research."""
    from proofstack.healthcheck import check_compute
    from proofstack.agents.cleanup_session import check_cleanup_launch, CleanupUnavailable
    from proofstack.registry import load_preset

    mode = (os.environ.get("FIRSTPROOF_HEALTHCHECK") or "off").lower()
    if mode not in {"off", "warn", "strict"}:
        raise ValueError("FIRSTPROOF_HEALTHCHECK must be off, warn, or strict")
    if not settings.batch3 and mode == "off":
        return settings
    preset = load_preset(settings.workflow)
    overrides = {"compute_codex_sandbox": settings.compute_codex_sandbox,
                 "compute_sandbox_backend": _workflow_env()["PROOFSTACK_SANDBOX_BACKEND"]}
    if settings.compute_max_parallel_workers is not None:
        overrides["compute_max_parallel_workers"] = settings.compute_max_parallel_workers
    inputs = preset.workflow_cls.Inputs(**preset.build_inputs(
        problem="offline preflight", problem_id="__healthcheck__", cli_overrides=overrides,
    )).model_dump()
    try:
        await check_compute(inputs, settings.output_dir)
    except RuntimeError as exc:
        if mode == "strict":
            raise
        warning = str(exc)
        if settings.batch3:
            settings = replace(settings, compute_disabled_reason=str(exc))
            warning += (
                "; disabling Compute for every problem in this launch, including retries. "
                "Continuing with the configured Author, critic, council and cleanup workflow."
            )
        settings.warnings.append(warning)
        print(f"FirstProof adapter warning: {warning}", file=sys.stderr)
    if settings.batch3:
        report = {"paid_calls": 0, "backend": inputs["cleanup_backend"], "ok": False}
        try:
            report["registry_probes"] = await check_cleanup_launch(
                inputs, preset.component_configs, settings.output_dir / "workflow_runs")
            report["ok"] = True
        except CleanupUnavailable as exc:
            report.update(ok=False, reason=str(exc), backend="api" if mode != "strict" else "claude_code")
            if mode == "strict":
                raise
            settings = replace(settings, cleanup_disabled_reason=str(exc))
            warning = (f"{exc}; selecting API cleanup for every problem in this launch, including retries. "
                       "Claude Code cleanup is unavailable.")
            settings.warnings.append(warning)
            print(f"FirstProof adapter warning: {warning}", file=sys.stderr)
        except Exception as exc:
            report.update(reason=str(exc), error_type=type(exc).__name__)
            raise
        finally:
            _write_json_atomic(settings.output_dir / "cleanup-healthcheck.json", report)
    return settings


async def _startup_checks(settings: Settings) -> Settings:
    if settings.stop_requested.is_set():
        raise asyncio.CancelledError("operator stop during startup")

    async def checks():
        bootstrap_done, bootstrap_warning = await _bootstrap_codex_auth()
        if bootstrap_done:
            print("FirstProof adapter: codex CLI authenticated via OPENAI_API_KEY.")
        if bootstrap_warning:
            settings.warnings.append(f"codex_auth_bootstrap: {bootstrap_warning}")
            print(f"FirstProof adapter: {bootstrap_warning}", file=sys.stderr)
        if settings.stop_requested.is_set():
            raise asyncio.CancelledError("operator stop during startup")
        return await _run_healthcheck(settings)

    probe = asyncio.create_task(checks())
    stop = asyncio.create_task(settings.stop_requested.wait())
    try:
        await asyncio.wait((probe, stop), return_when=asyncio.FIRST_COMPLETED)
        if settings.stop_requested.is_set():
            raise asyncio.CancelledError("operator stop during startup")
        return probe.result()
    finally:
        for task in (probe, stop):
            if not task.done():
                task.cancel()
        await asyncio.gather(probe, stop, return_exceptions=True)


async def _amain_run(settings: Settings, *, on_output_ready=None) -> int:
    if settings.batch3 and any((settings.output_dir / "workflow_runs").glob("*/batch3-schedule.json")):
        print("FirstProof adapter: existing Batch 3 checkpoints found; use scripts/resume_batch3.py "
              "instead of resetting their clocks with a fresh launch.", file=sys.stderr)
        return 2
    overall_started = _utc_now()
    overall_start_time = time.monotonic()
    try:
        _prepare_output_dir(settings.output_dir)
    except OSError as exc:
        print(f"FirstProof adapter fatal: output directory is not writable: {exc}", file=sys.stderr)
        return 2
    if on_output_ready is not None:
        on_output_ready()
    try:
        items = _load_problem_items(settings.input_path)
    except Exception as exc:
        print(f"FirstProof adapter fatal: could not read input JSON: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        problems = _parse_problems(items, settings)
    except OSError as exc:
        print(f"FirstProof adapter fatal: could not prepare problem files: {exc}", file=sys.stderr)
        return 2

    if settings.deadline_seconds is not None:
        settings = replace(
            settings,
            deadline_at=overall_start_time + settings.deadline_seconds,
            research_deadline_at=overall_start_time + min(82800, max(0, settings.deadline_seconds - (
                _batch3_cleanup_seconds(settings.workflow) + 60 if settings.batch3 else 3300))),
        )

    # Write the per-problem fallback .tex stubs *eagerly*, before any
    # startup probe or task and before the zero-th aggregate snapshot
    # below. Combined with _aggregate_payloads' _latex_for_pending, this
    # means even an early crash leaves /data/output populated with
    # valid (fallback) solutions for every problem.
    for problem in problems:
        try:
            _write_text_atomic(
                problem.output_tex_path,
                _fallback_tex(problem, "workflow did not start before container exit"),
            )
        except OSError as stub_exc:
            settings.warnings.append(
                f"initial stub write failed for {problem.original_id}: {stub_exc}"
            )

    # Maintain a running results list and rewrite the aggregates after
    # every problem completion. A SIGKILL at hour 23 then leaves a valid
    # solutions.json / run_summary.json / token_usage.jsonl describing
    # whatever finished, instead of nothing.
    results: list[ProblemResult | None] = [None] * len(problems)
    aggregate_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(settings.max_parallel)

    # Zero-th aggregate snapshot: every record is ``None``, but
    # solutions.json picks up the freshly-written stubs through
    # _latex_for_pending. An early crash before the first problem
    # finishes still leaves a structurally valid solutions.json +
    # run_summary.json + token_usage.jsonl on disk.
    try:
        await _write_aggregates(
            problems, results, settings, overall_started, overall_start_time, in_progress=True
        )
    except OSError as zeroth_exc:
        print(
            f"FirstProof adapter: zero-th aggregate write failed: {zeroth_exc}",
            file=sys.stderr,
        )

    # Prepare authentication for later workers. The native-host launch probe
    # itself is credential-free and never submits a model request.
    try:
        settings = await _startup_checks(settings)
    except asyncio.CancelledError:
        if not settings.stop_requested.is_set():
            raise
        stopped_results = [await _exception_result(problem, RuntimeError("operator stop during startup"), settings)
                           for problem in problems]
        await _write_aggregates(
            problems, stopped_results, settings, overall_started, overall_start_time, in_progress=False,
        )
        return 143
    except (RuntimeError, ValueError, OSError) as exc:
        settings.warnings.append(f"startup healthcheck failed: {exc}")
        print(f"FirstProof adapter fatal: {exc}", file=sys.stderr)
        failed_results = [await _exception_result(problem, exc, settings) for problem in problems]
        try:
            await _write_aggregates(
                problems, failed_results, settings, overall_started, overall_start_time, in_progress=False
            )
        except OSError as write_exc:
            print(f"FirstProof adapter: failed startup aggregate write failed: {write_exc}", file=sys.stderr)
        return 2

    # Persist any degraded-mode decision before the first paid workflow starts.
    try:
        await _write_aggregates(
            problems, results, settings, overall_started, overall_start_time, in_progress=True
        )
    except OSError as write_exc:
        print(f"FirstProof adapter: startup aggregate write failed: {write_exc}", file=sys.stderr)

    print(f"FirstProof adapter starting {len(problems)} problem(s) with max_parallel={settings.max_parallel}")
    if settings.deadline_seconds is not None:
        print(
            f"FirstProof adapter: internal deadline set at "
            f"{settings.deadline_seconds:.0f}s "
            f"({settings.deadline_seconds / 60:.1f} min) from adapter start."
        )

    async def _run_and_record(idx: int, problem: Problem) -> None:
        async def record_stage_snapshot(snapshot: ProblemResult) -> None:
            async with aggregate_lock:
                results[idx] = snapshot
                try:
                    await _write_aggregates(
                        problems, results, settings, overall_started, overall_start_time, in_progress=True
                    )
                except OSError as write_exc:
                    print(
                        f"FirstProof adapter: stage-aggregate write failed: {write_exc}",
                        file=sys.stderr,
                    )

        try:
            result = await _run_problem(
                problem,
                settings,
                semaphore,
                on_stage_complete=record_stage_snapshot,
            )
        except BaseException as exc:
            async with aggregate_lock:
                previous_snapshot = results[idx]
            if previous_snapshot is not None and not settings.stop_requested.is_set():
                try:
                    await _write_log_line(
                        problem.log_path,
                        f"\n[{_utc_now()}] adapter interrupted after a stage "
                        f"snapshot; preserving last promoted stage result: "
                        f"{type(exc).__name__}: {exc}\n",
                    )
                except OSError:
                    pass
                result = previous_snapshot
            else:
                result = await _exception_result(problem, exc, settings)
        async with aggregate_lock:
            results[idx] = result
            try:
                await _write_aggregates(
                    problems, results, settings, overall_started, overall_start_time, in_progress=True
                )
            except OSError as write_exc:
                print(
                    f"FirstProof adapter: partial-aggregate write failed: {write_exc}",
                    file=sys.stderr,
                )

    tasks = [
        asyncio.create_task(_run_and_record(i, problem))
        for i, problem in enumerate(problems)
    ]
    gather_fut = asyncio.gather(*tasks, return_exceptions=True)
    stop_task = asyncio.create_task(settings.stop_requested.wait())

    async def live_snapshots():
        while True:
            await asyncio.sleep(LIVE_SNAPSHOT_INTERVAL_S)
            async with aggregate_lock:
                try:
                    await _write_aggregates(
                        problems, results, settings, overall_started, overall_start_time, in_progress=True,
                    )
                except OSError as exc:
                    print(f"FirstProof adapter: live snapshot failed: {type(exc).__name__}", file=sys.stderr)

    snapshot_task = asyncio.create_task(live_snapshots())

    deadline_reached = False
    try:
        timeout = None if settings.deadline_at is None else max(0, settings.deadline_at - time.monotonic())
        done, _ = await asyncio.wait((gather_fut, stop_task), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if gather_fut not in done:
            deadline_reached = not settings.stop_requested.is_set()
            raise asyncio.TimeoutError
    except (asyncio.TimeoutError, asyncio.CancelledError):
        print(
            "FirstProof adapter: stop/deadline requested; cancelling in-flight "
            "problems and flushing partial aggregates.",
            file=sys.stderr,
        )
        # Cancel all children once, leaving their cleanup a bounded drain interval.
        for task in tasks:
            if not task.done():
                task.cancel()
        # Give them up to 20s to reconcile providers and write their aggregate row,
        # then move on. Anything still blocked will be SIGKILLed by the
        # harness's outer ``timeout`` shortly.
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=20.0,
            )
        except asyncio.TimeoutError:
            print(
                "FirstProof adapter: drain expired with tasks still pending; "
                "writing final aggregate with whatever lives in `results`.",
                file=sys.stderr,
            )
    finally:
        stop_task.cancel()
        snapshot_task.cancel()
        await asyncio.gather(stop_task, snapshot_task, return_exceptions=True)

    # Final aggregate write — same shape, but flagged ``in_progress=False``.
    final_results: list[ProblemResult] = []
    for problem, result in zip(problems, results):
        if result is None or (settings.stop_requested.is_set() and result.in_progress):
            # Defensive: a task that raised (or was cancelled at the
            # deadline) before _exception_result could have left a slot
            # empty. Synthesize a fallback so the final summary is
            # well-formed and the on-disk .tex stub remains the truth
            # for that problem. A second cancellation during shutdown can also
            # leave an earlier in-progress stage; re-read its publication now.
            reason = (
                "internal deadline reached before this problem finished"
                if deadline_reached
                else "problem task produced no result"
            )
            final_results.append(
                await _exception_result(problem, RuntimeError(reason), settings)
            )
        else:
            final_results.append(result)
    await _write_aggregates(
        problems,
        final_results,
        settings,
        overall_started,
        overall_start_time,
        in_progress=False,
        deadline_reached=deadline_reached,
    )
    duration = round(time.monotonic() - overall_start_time, 3)
    print(f"FirstProof adapter finished in {duration:.3f}s; outputs written to {settings.output_dir}")
    return 143 if settings.stop_requested.is_set() else 0


async def _amain() -> int:
    settings = _settings()
    loop = asyncio.get_running_loop()
    handlers = {}
    output_ready = False
    def mark_output_ready():
        nonlocal output_ready
        output_ready = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous = signal.getsignal(sig)
            loop.add_signal_handler(sig, settings.stop_requested.set)
            handlers[sig] = previous
        except (NotImplementedError, RuntimeError, ValueError):
            pass
    try:
        return await _amain_run(settings, on_output_ready=mark_output_ready)
    finally:
        try:
            warnings = _finalize_output_permissions(settings.output_dir) if output_ready else []
        except Exception as exc:
            warnings = [f"RETRIEVAL_UNSAFE: output finalization failed: {type(exc).__name__}"]
        for warning in warnings:
            print(f"FirstProof adapter: {warning}", file=sys.stderr)
        for sig, previous in handlers.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, previous)


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
