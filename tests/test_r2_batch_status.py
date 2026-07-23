"""R2 Round-3: parked is a first-class batch outcome, distinct from error.

A batch child that parks on a usage window exits 2 (resumable). Two seams
dropped that distinction:

  B5 — the batch PARENT collapsed any non-ok child to error/exit-1, so a batch
       whose only non-ok children parked was reported as a crash and refused
       Resume. The parent now mirrors the child tri-state (ok / parked / error).
  B6 — the dashboard's per-problem status combiner ranked parked ABOVE error,
       so a mixed batch with a genuine crash displayed as a resumable parked
       run, masking the failure. Error now outranks parked.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.dev_data import _status_from_problem_statuses  # noqa: E402


class BatchStatusPrecedenceTests(unittest.TestCase):
    """B6: error outranks parked; running/stopped still win over both."""

    def test_error_outranks_parked(self) -> None:
        self.assertEqual(_status_from_problem_statuses(["parked", "error"]), "error")

    def test_all_parked_is_parked(self) -> None:
        self.assertEqual(_status_from_problem_statuses(["parked", "parked"]), "parked")

    def test_running_still_wins_over_error(self) -> None:
        self.assertEqual(_status_from_problem_statuses(["running", "error"]), "running")

    def test_stopped_still_wins_over_error(self) -> None:
        self.assertEqual(_status_from_problem_statuses(["stopped", "error"]), "stopped")

    def test_all_finished(self) -> None:
        self.assertEqual(_status_from_problem_statuses(["finished", "finished"]), "finished")


class BatchParentExitTests(unittest.TestCase):
    """B5: the batch parent status/exit-code mirrors the child tri-state."""

    def _run_with_child_code(self, code: int) -> tuple[str, str, int]:
        spec = importlib.util.spec_from_file_location(
            "rwb_under_test", ROOT / "scripts" / "run_workflow_batch.py"
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class FakeProcess:
            async def wait(self) -> int:
                return code

        async def fake_create_subprocess_exec(*args, **kwargs):
            return FakeProcess()

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            problems = tmp / "problems.json"
            problems.write_text(json.dumps([{"id": "p", "latex": "P"}]), encoding="utf-8")
            argv = [
                "run_workflow_batch.py",
                "--workflow", "demo",
                "--problems-file", str(problems),
                "--output", str(tmp / "out"),
                "--run-id", "batch",
            ]
            with mock.patch.object(asyncio, "create_subprocess_exec", fake_create_subprocess_exec), \
                 mock.patch.object(sys, "argv", argv):
                rc = asyncio.run(mod.amain())
            meta = json.loads(
                (tmp / "out" / "batch" / "run-metadata.json").read_text(encoding="utf-8")
            )
        return meta["status"], meta["manifest"]["problems"]["p"]["status"], rc

    def test_parked_child_parks_parent(self) -> None:
        self.assertEqual(self._run_with_child_code(2), ("parked", "parked", 2))

    def test_error_child_errors_parent(self) -> None:
        self.assertEqual(self._run_with_child_code(1), ("error", "error", 1))

    def test_ok_child_ok_parent(self) -> None:
        self.assertEqual(self._run_with_child_code(0), ("ok", "ok", 0))


if __name__ == "__main__":
    unittest.main()
