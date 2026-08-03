"""Session-detached CLI children must not survive a dashboard Stop.

The registry records each spawned child's pid/pgid; stop_run_process
kills registered groups on every exit path — including 'worker already
dead', which is exactly when orphans exist."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.dev_data import stop_run_process  # noqa: E402
from proofstack.child_registry import (  # noqa: E402
    kill_registered_children,
    register_child,
    unregister_child,
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class ChildRegistryTests(unittest.TestCase):
    def test_register_unregister_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            register_child(run_dir, pid=os.getpid(), cmd0="python3", label="x")
            entries = json.loads(
                (run_dir / "run-children.json").read_text(encoding="utf-8")
            )
            self.assertEqual(entries[0]["pid"], os.getpid())
            unregister_child(run_dir, os.getpid())
            self.assertFalse((run_dir / "run-children.json").exists())

    def test_kill_registered_children_kills_detached_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                start_new_session=True,
            )
            try:
                register_child(run_dir, pid=proc.pid, cmd0=sys.executable)
                result = kill_registered_children(run_dir, grace_s=2.0)
                self.assertIn(proc.pid, result["children_signalled"])
                deadline = time.time() + 3
                while time.time() < deadline and proc.poll() is None:
                    time.sleep(0.05)
                self.assertIsNotNone(proc.poll())
                self.assertFalse((run_dir / "run-children.json").exists())
            finally:
                if proc.poll() is None:
                    proc.kill()

    def test_stale_entry_with_dead_pid_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            dead = subprocess.Popen([sys.executable, "-c", "pass"])
            dead.wait()
            register_child(run_dir, pid=dead.pid, cmd0=sys.executable)
            result = kill_registered_children(run_dir)
            self.assertEqual(result["children_signalled"], [])

    def test_stop_run_process_cleans_children_even_when_worker_dead(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            # No run.pid at all: the "worker already dead" path.
            proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                start_new_session=True,
            )
            try:
                register_child(run_dir, pid=proc.pid, cmd0=sys.executable)
                result = stop_run_process(run_dir, grace_s=2.0)
                self.assertFalse(result["signalled"])  # no worker to stop
                self.assertIn(proc.pid, result["children_signalled"])
                deadline = time.time() + 3
                while time.time() < deadline and proc.poll() is None:
                    time.sleep(0.05)
                self.assertIsNotNone(proc.poll())
            finally:
                if proc.poll() is None:
                    proc.kill()


if __name__ == "__main__":
    unittest.main()
