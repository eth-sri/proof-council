"""Nav "waiting on compute worker" indicator: a non-terminal run counts as
compute-busy iff a live CLI worker process has its cwd inside the run dir.
Worker discovery is /proc-based and injected here via monkeypatching."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app import dev_data  # noqa: E402


def _write_run(root: Path, name: str, status: str = "running") -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "run-metadata.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    return run_dir


class ComputeBusyTests(unittest.TestCase):
    def test_worker_inside_run_dir_marks_run_busy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _write_run(Path(tmp), "r1")
            workspace = run_dir / "ac_workspaces" / "p-r5" / "compute"
            workspace.mkdir(parents=True)
            with mock.patch.object(
                dev_data, "_cli_worker_cwds", return_value=[workspace.resolve()]
            ):
                busy = dev_data.find_runs_with_busy_compute([Path(tmp)])
        self.assertEqual([b["run_id"] for b in busy], ["r1"])
        self.assertEqual(busy[0]["count"], 1)

    def test_worker_elsewhere_or_terminal_run_not_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(Path(tmp), "r1")
            done_dir = _write_run(Path(tmp), "r2", status="finished")
            inside_done = done_dir / "ac_workspaces" / "x"
            inside_done.mkdir(parents=True)
            elsewhere = Path(tmp) / "unrelated"
            elsewhere.mkdir()
            with mock.patch.object(
                dev_data,
                "_cli_worker_cwds",
                return_value=[elsewhere.resolve(), inside_done.resolve()],
            ):
                busy = dev_data.find_runs_with_busy_compute([Path(tmp)])
        self.assertEqual(busy, [])

    def test_no_workers_short_circuits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(Path(tmp), "r1")
            with mock.patch.object(dev_data, "_cli_worker_cwds", return_value=[]):
                self.assertEqual(
                    dev_data.find_runs_with_busy_compute([Path(tmp)]), []
                )


if __name__ == "__main__":
    unittest.main()
