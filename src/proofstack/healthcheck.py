"""Credential-free launch gate using the same sandbox policy as Compute."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from proofstack.agents.ac.compute import Compute, compute_sandbox_spec, _require_codex_cli_version
from proofstack.atomic import write_text_atomic
from proofstack.sandbox import make_sandbox, resolve_backend


async def check_compute(inputs: dict, output_dir: Path) -> dict:
    report = {"ok": False, "paid_calls": 0, "probes": []}
    try:
        if not inputs.get("enable_compute", True):
            report.update(ok=True, compute="disabled")
            return report
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".compute-preflight-", dir=output_dir) as tmp:
            values = {name: inputs[f"compute_{name}"] for name in Compute.Inputs.model_fields
                      if f"compute_{name}" in inputs}
            worker = Compute.Inputs(compute_workspace=Path(tmp), problem="offline preflight",
                                    problem_id="__healthcheck__", round=0, instructions="", **values)
            spec = compute_sandbox_spec(worker, registry_root=output_dir / "workflow_runs")
            report["compute"] = {
                "backend": resolve_backend(spec), "memory_gb": spec.memory_gb,
                "limit_address_space": spec.limit_address_space,
                "max_parallel_workers": worker.max_parallel_workers,
                "reserve_gb": worker.memory_reserve_gb,
            }
            async with make_sandbox(spec, root=Path(tmp)) as sandbox:
                await asyncio.wait_for(_require_codex_cli_version(sandbox), timeout=180)
                # run_command bypasses the shared memory registry. Exercise the
                # same streamed admission and cleanup path as a real worker.
                # Subprocess admission defaults to the same timeout as execution.
                handle = await sandbox.stream_command(
                    ["codex", "--version"], timeout_s=30,
                )
                try:
                    code = await handle.wait(timeout_s=30)
                    if code != 0 or getattr(handle, "memory_failure", None):
                        raise RuntimeError(f"Compute admission probe failed (exit {code})")
                finally:
                    await handle.terminate()
                if not handle.worker_stopped:
                    raise RuntimeError("Compute admission probe cleanup was not confirmed")
                report["probes"].append({"name": "streamed_worker_admission", "ok": True,
                                         "worker_stopped": True})
        report["ok"] = True
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(f"Launch healthcheck failed: {report['error']}") from exc
    finally:
        write_text_atomic(output_dir / "healthcheck.json", json.dumps(report, indent=2))
