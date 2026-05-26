"""Local-subprocess Python / C++ runner.

Replaces the Modal-based ``mathagents.tools.code_execution`` per
SPEC §3.3.1. No external runtime; the orchestrator's container is the
sandbox.

Public callable: ``run_code(code, lang)``. Designed to be exposed as a
plain function-tool to a MultiTurnAgent.

Snippets are run with a deliberately scrubbed environment — no
provider API keys, no parent home directory. A model-driven snippet
must NOT be able to print secrets back to itself by reading
``os.environ``.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Lang = Literal["python", "cpp"]

DEFAULT_TIMEOUT_S = 30
DEFAULT_MEMORY_GB = 2

# Minimal env vars passed through to snippets. Excludes EVERY provider
# API key and the parent ``HOME``. ``LANG``/``LC_ALL`` so Unicode
# stdio works; ``PATH`` so ``g++`` / ``python3`` can be found.
_SAFE_ENV_KEYS: tuple[str, ...] = ("PATH", "LANG", "LC_ALL", "TERM")


@dataclass
class CodeExecResult:
    stdout: str
    stderr: str
    returncode: int
    duration_s: float
    timed_out: bool


def _resource_preexec(memory_gb: int) -> "callable | None":
    try:
        import resource

        def _apply() -> None:
            try:
                lim = memory_gb * 1024 * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            except (ValueError, OSError):
                pass

        return _apply
    except ImportError:
        return None


def run_python(code: str, *, timeout_s: int = DEFAULT_TIMEOUT_S, memory_gb: int = DEFAULT_MEMORY_GB) -> CodeExecResult:
    """Run a Python snippet in a fresh subprocess and return its output."""
    if shutil.which("python3") is None:
        return CodeExecResult("", "python3 not found on PATH", 127, 0.0, False)
    with tempfile.TemporaryDirectory(prefix="proofstack-py-") as td:
        path = Path(td) / "snippet.py"
        path.write_text(code, encoding="utf-8")
        return _exec(["python3", str(path)], timeout_s=timeout_s, cwd=td, memory_gb=memory_gb)


def run_cpp(code: str, *, timeout_s: int = DEFAULT_TIMEOUT_S, memory_gb: int = DEFAULT_MEMORY_GB) -> CodeExecResult:
    """Compile and run a C++ snippet (g++ -O2) in a sandbox."""
    if shutil.which("g++") is None:
        return CodeExecResult("", "g++ not found on PATH", 127, 0.0, False)
    with tempfile.TemporaryDirectory(prefix="proofstack-cpp-") as td:
        src = Path(td) / "snippet.cpp"
        bin_path = Path(td) / "snippet.bin"
        src.write_text(code, encoding="utf-8")
        compile_res = _exec(
            ["g++", "-O2", "-std=c++20", "-o", str(bin_path), str(src)],
            timeout_s=60,
            cwd=td,
            memory_gb=memory_gb,
        )
        if compile_res.returncode != 0:
            return compile_res
        return _exec([str(bin_path)], timeout_s=timeout_s, cwd=td, memory_gb=memory_gb)


def run_code(code: str, lang: Lang = "python", *, timeout_s: int = DEFAULT_TIMEOUT_S) -> CodeExecResult:
    """Convenience dispatch by language."""
    if lang == "python":
        return run_python(code, timeout_s=timeout_s)
    if lang == "cpp":
        return run_cpp(code, timeout_s=timeout_s)
    return CodeExecResult("", f"unsupported lang: {lang}", 64, 0.0, False)


async def arun_code(code: str, lang: Lang = "python", *, timeout_s: int = DEFAULT_TIMEOUT_S) -> CodeExecResult:
    """Async wrapper for use from MultiTurnAgent tool dispatch."""
    return await asyncio.to_thread(run_code, code, lang, timeout_s=timeout_s)


def _safe_env(cwd: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        if (val := os.environ.get(key)) is not None:
            env[key] = val
    # Pin HOME/TMPDIR to the snippet's own working dir so anything that
    # writes to ~/ lands inside the throwaway tempdir, not the host home.
    env["HOME"] = cwd
    env["TMPDIR"] = cwd
    return env


def _exec(cmd: list[str], *, timeout_s: int, cwd: str, memory_gb: int) -> CodeExecResult:
    import time as _time

    start = _time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            timeout=timeout_s,
            preexec_fn=_resource_preexec(memory_gb),
            text=False,
            env=_safe_env(cwd),
        )
    except subprocess.TimeoutExpired as e:
        return CodeExecResult(
            stdout=(e.stdout or b"").decode("utf-8", errors="replace"),
            stderr=(e.stderr or b"").decode("utf-8", errors="replace") + f"\n(timed out after {timeout_s}s)",
            returncode=-9,
            duration_s=_time.monotonic() - start,
            timed_out=True,
        )
    return CodeExecResult(
        stdout=proc.stdout.decode("utf-8", errors="replace"),
        stderr=proc.stderr.decode("utf-8", errors="replace"),
        returncode=proc.returncode,
        duration_s=_time.monotonic() - start,
        timed_out=False,
    )


__all__ = [
    "CodeExecResult",
    "DEFAULT_MEMORY_GB",
    "DEFAULT_TIMEOUT_S",
    "Lang",
    "arun_code",
    "run_code",
    "run_cpp",
    "run_python",
]
