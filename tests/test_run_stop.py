import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.dev import create_app  # noqa: E402
from app.dev_data import (  # noqa: E402
    _read_run_info,
    read_run_pid,
    run_process_alive,
    stop_run_process,
    write_stopped_marker,
)


def _spawn_group_leader(seconds: int = 30):
    import subprocess

    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
        start_new_session=True,  # own process group, like the dashboard launches
    )


class StopRunProcessTests(unittest.TestCase):
    def test_stop_terminates_process_group_and_marks_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            proc = _spawn_group_leader()
            pid = proc.pid
            (run_dir / "run.pid").write_text(str(pid), encoding="utf-8")
            self.addCleanup(proc.wait)
            self.addCleanup(lambda: self._reap(pid))

            # Sanity: launched as its own group leader, so it is killable as a group.
            self.assertTrue(run_process_alive(run_dir))
            self.assertEqual(os.getpgid(pid), pid)

            result = stop_run_process(run_dir, grace_s=2.0)
            self.assertTrue(result["signalled"])
            self._wait_dead(pid)
            self.assertFalse(run_process_alive(run_dir))
            # run.pid is cleared once the process is signalled.
            self.assertIsNone(read_run_pid(run_dir))

    def test_stop_when_no_live_process_is_not_signalled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            # A PID that is (almost certainly) not a live process.
            (run_dir / "run.pid").write_text("2147480000", encoding="utf-8")
            result = stop_run_process(run_dir)
            self.assertFalse(result["signalled"])

    def test_stopped_marker_makes_status_nonterminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            # has_events -> would otherwise read as "running" forever.
            (run_dir / "events.jsonl").write_text(
                '{"kind": "run.start", "payload": {}}\n', encoding="utf-8"
            )
            write_stopped_marker(run_dir, signalled=True)
            info = _read_run_info(run_dir)
            self.assertEqual(info.status, "stopped")
            self.assertNotIn(info.status, {"finished", "error"})

    def test_terminal_status_wins_over_stopped_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            # A resumed run that actually finished: run.end ok must win.
            (run_dir / "events.jsonl").write_text(
                '{"kind": "run.start", "payload": {}}\n'
                '{"kind": "run.end", "payload": {"status": "ok"}}\n',
                encoding="utf-8",
            )
            write_stopped_marker(run_dir, signalled=True)
            info = _read_run_info(run_dir)
            self.assertEqual(info.status, "finished")

    def test_stop_route_kills_group_and_shows_resume(self) -> None:
        # Faithful end-to-end of the Stop button: a real process group (spawned
        # the way the dashboard launches a worker) is terminated by the HTTP
        # route, the run flips to non-terminal "stopped", and the run page then
        # offers Resume instead of Stop.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "stoptest-run"
            run_dir.mkdir()
            (run_dir / "events.jsonl").write_text(
                '{"kind": "run.start", "payload": {"preset": "demo"}}\n',
                encoding="utf-8",
            )
            (run_dir / "resume.json").write_text(
                json.dumps({"run_id": "stoptest-run", "argv": ["scripts/run_workflow.py"]}),
                encoding="utf-8",
            )
            proc = _spawn_group_leader()
            pid = proc.pid
            self.addCleanup(proc.wait)
            self.addCleanup(lambda: self._reap(pid))
            (run_dir / "run.pid").write_text(str(pid), encoding="utf-8")

            app = create_app(runs_roots=(root,))
            with app.test_client() as client:
                # While alive, the detail page offers Stop, not Resume.
                page = client.get("/run/stoptest-run").get_data(as_text=True)
                self.assertIn("Stop run", page)

                resp = client.post("/run/stoptest-run/stop")
                self.assertEqual(resp.status_code, 302)

                self._wait_dead(pid)
                self.assertFalse(run_process_alive(run_dir))
                self.assertEqual(_read_run_info(run_dir).status, "stopped")

                page = client.get("/run/stoptest-run").get_data(as_text=True)
                self.assertIn("Resume run", page)
                self.assertNotIn("Stop run", page)

    def test_phantom_run_flagged_only_when_pid_present_and_dead(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def make_run(name: str) -> Path:
                d = root / name
                d.mkdir()
                (d / "events.jsonl").write_text(
                    '{"kind": "run.start", "payload": {}}\n', encoding="utf-8"
                )
                return d

            dead = make_run("dead")
            (dead / "run.pid").write_text("2147480000", encoding="utf-8")
            self.assertTrue(_read_run_info(dead).process_dead)

            no_pid = make_run("no-pid")  # e.g. batch parent / legacy run
            self.assertFalse(_read_run_info(no_pid).process_dead)

            alive = make_run("alive")  # e.g. a run still working / waiting on a human
            proc = _spawn_group_leader()
            self.addCleanup(proc.wait)
            self.addCleanup(lambda: self._reap(proc.pid))
            (alive / "run.pid").write_text(str(proc.pid), encoding="utf-8")
            self.assertFalse(_read_run_info(alive).process_dead)

    def _wait_dead(self, pid: int, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)

    def _reap(self, pid: int) -> None:
        try:
            os.killpg(pid, 9)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            os.waitpid(pid, 0)
        except (ChildProcessError, OSError):
            pass


if __name__ == "__main__":
    unittest.main()
