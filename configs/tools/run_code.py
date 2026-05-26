from proofstack.tools.code_execution import run_code as _run_code


def run_code(code: str, lang: str = "python", timeout_s: int = 30) -> dict:
    """Run a short Python or C++ snippet in the local sandbox."""
    result = _run_code(code, lang, timeout_s=timeout_s)
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "duration_s": result.duration_s,
        "timed_out": result.timed_out,
    }
