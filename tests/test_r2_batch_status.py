"""R2 Round-3/4: parked is a first-class batch outcome, distinct from error.

A batch child that parks on a usage window exits 2 (resumable). Several seams
dropped or over-trusted that distinction:

  B5 — the batch PARENT collapsed any non-ok child to error/exit-1.
  B6 — the dashboard status combiner ranked parked ABOVE error, masking a crash.
  A3 — exit 2 is ALSO argparse's parse-error code, so a real launch failure was
       reported as parked. Trust it as a park only when the child recorded one.
  B4 — a legacy parent persisted as "error" whose children all finished (after a
       resume) stayed "error"; a derived "finished" now retires that stale error.
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

from app.dev_data import _status_from_problem_statuses, discover_runs  # noqa: E402


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
    """B5/A3: the parent status/exit mirrors the child tri-state, and exit 2 is
    a park only when the child actually recorded one."""

    def _run_batch(self, code: int, child_meta: str | None = None) -> tuple[str, str, int]:
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
            if child_meta is not None:
                a = [str(x) for x in args]
                out = Path(a[a.index("--output") + 1])
                rid = a[a.index("--run-id") + 1]
                child = out / rid
                child.mkdir(parents=True, exist_ok=True)
                (child / "run-metadata.json").write_text(
                    json.dumps({"status": child_meta}), encoding="utf-8"
                )
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
        # exit 2 AND the child recorded a park
        self.assertEqual(self._run_batch(2, child_meta="parked"), ("parked", "parked", 2))

    def test_argparse_failure_exit2_is_error_not_parked(self) -> None:
        # exit 2 with NO recorded park (argparse/launch failure) -> error
        self.assertEqual(self._run_batch(2, child_meta=None), ("error", "error", 1))

    def test_error_child_errors_parent(self) -> None:
        self.assertEqual(self._run_batch(1), ("error", "error", 1))

    def test_ok_child_ok_parent(self) -> None:
        self.assertEqual(self._run_batch(0), ("ok", "ok", 0))


class LegacyBatchFinishedOverrideTests(unittest.TestCase):
    """B4: a stale 'error' parent whose children all finished becomes finished."""

    def _write(self, path: Path, meta: dict) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "run-metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    def test_stale_error_parent_becomes_finished_after_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root / "batch",
                {
                    "status": "error",  # legacy: persisted before the resume
                    "manifest": {"problems": {"p": {"run_id": "child", "status": "finished"}}},
                },
            )
            self._write(root / "child", {"status": "finished"})
            run = next(r for r in discover_runs((root,)) if r.run_id == "batch")
            self.assertEqual(run.status, "finished")

    def test_stale_error_parent_stays_error_while_a_child_still_running(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write(
                root / "batch",
                {
                    "status": "error",
                    "manifest": {
                        "problems": {
                            "p1": {"run_id": "c1", "status": "finished"},
                            "p2": {"run_id": "c2", "status": "queued"},
                        }
                    },
                },
            )
            self._write(root / "c1", {"status": "finished"})
            run = next(r for r in discover_runs((root,)) if r.run_id == "batch")
            # not all finished -> must NOT fabricate a finish
            self.assertNotEqual(run.status, "finished")


if __name__ == "__main__":
    unittest.main()
