"""R1-A6: a subscription park is a resumable pause, not a crash. run_workflow
records a distinct "parked" status; the dashboard surfaces it as resumable
(Resume button + friendly banner, non-error pill) rather than a red failure.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from proofstack.budget import SubscriptionParked  # noqa: E402
from app.dev_data import _normalize_run_status  # noqa: E402
from app.dev import _TERMINAL_RUN_STATUSES, create_app  # noqa: E402


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ParkInputs(BaseModel):
    model_config = ConfigDict(extra="allow")


class _ParkWorkflow:
    Inputs = _ParkInputs

    def __init__(self, ctx, *a, **k) -> None:
        self.ctx = ctx

    async def __call__(self, **kwargs):
        raise SubscriptionParked("five_hour", 7200.0, 100.0, 100.0)


class _ParkPreset:
    name = "park_demo"
    workflow_cls = _ParkWorkflow
    model_overrides: dict = {}
    component_configs: dict = {}
    inputs: dict = {}
    budget = None
    source_path = Path("/tmp/park_demo.yaml")
    raw: dict = {}

    def build_inputs(self, *, problem=None, problem_id=None, cli_overrides=None, **_k):
        return {}


class RunWorkflowParkedStatusTests(unittest.TestCase):
    def test_parked_workflow_writes_parked_status_and_exit_2(self) -> None:
        import asyncio

        module = _load_module("run_workflow_parked", "scripts/run_workflow.py")
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "outputs"
            argv = [
                "run_workflow.py",
                "--workflow", "park_demo",
                "--problem-text", "Prove P.",
                "--output", str(out_root),
                "--run-id", "parked1",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                module, "load_preset", return_value=_ParkPreset()
            ):
                rc = asyncio.run(module.amain())

            meta = json.loads(
                (out_root / "parked1" / "run-metadata.json").read_text(encoding="utf-8")
            )

        self.assertEqual(rc, 2)
        self.assertEqual(meta["status"], "parked")


class ParkedStatusNormalizationTests(unittest.TestCase):
    def test_parked_normalizes_to_itself_and_is_not_terminal(self) -> None:
        self.assertEqual(_normalize_run_status("parked"), "parked")
        self.assertEqual(_normalize_run_status("PARKED"), "parked")
        self.assertNotIn("parked", _TERMINAL_RUN_STATUSES)


class DashboardParkedRunTests(unittest.TestCase):
    def _write_parked_run(self, root: Path) -> None:
        run_dir = root / "parked1"
        run_dir.mkdir(parents=True)
        (run_dir / "run-metadata.json").write_text(
            json.dumps({"status": "parked", "display_name": "Parked Run"}),
            encoding="utf-8",
        )
        (run_dir / "events.jsonl").write_text(
            json.dumps({"kind": "run.start", "payload": {"preset": "park_demo"}}) + "\n"
            + json.dumps({"kind": "run.end", "payload": {"status": "parked"}}) + "\n",
            encoding="utf-8",
        )
        (run_dir / "resume.json").write_text(
            json.dumps({"run_id": "parked1", "argv": ["scripts/run_workflow.py"]}),
            encoding="utf-8",
        )

    def test_run_detail_offers_resume_and_parked_banner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_parked_run(root)
            app = create_app(runs_roots=(root,))
            with app.test_client() as client:
                resp = client.get("/run/parked1")
                body = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Resume run", body)
        self.assertIn("parked (paused in a resumable state)", body)

    def test_runs_list_uses_parked_pill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_parked_run(root)
            app = create_app(runs_roots=(root,))
            with app.test_client() as client:
                resp = client.get("/runs")
                body = resp.get_data(as_text=True)
        self.assertEqual(resp.status_code, 200)
        # The rendered pill element (space-separated classes), not the CSS rule
        # (dot-separated). The parked run must not paint as a red error pill.
        self.assertIn("pill status-parked", body)
        self.assertNotIn("pill status-error", body)


if __name__ == "__main__":
    unittest.main()
