"""R1-A12 (load_pending half): a human.waiting that lands AFTER a
human.submitted/timeout on the same stable response path must REOPEN the task.

The old logic put a path into a permanent ``resolved`` set, so a genuine re-ask
(a loop revisiting the node, or a resumed run re-emitting the ask on a cache
miss) stayed suppressed forever. It now tracks the latest lifecycle event per
path; a dropped valid response file still short-circuits listing (answer in
flight).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.dev_data import load_pending_human_tasks  # noqa: E402


def _write_run(root: Path, events: list[dict], task_output_fields: dict | None = None) -> Path:
    run_dir = root / "r"
    inbox = run_dir / "human_inbox"
    inbox.mkdir(parents=True)
    (run_dir / "run-metadata.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )
    # A task.json so the loader can read prompt/output fields.
    task_path = run_dir / "t.task.json"
    task_path.write_text(
        json.dumps({"prompt": "review", "output_fields": task_output_fields or {"response": "string"}}),
        encoding="utf-8",
    )
    resp_path = inbox / "t.response.json"
    lines = []
    for ev in events:
        payload = {"response_path": str(resp_path), "task_path": str(task_path)}
        payload.update(ev.get("payload", {}))
        lines.append(json.dumps({"kind": ev["kind"], "agent": "t", "payload": payload}))
    (run_dir / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return resp_path


class HumanRelistTests(unittest.TestCase):
    def test_reask_after_submit_reopens_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(
                Path(tmp),
                [
                    {"kind": "human.waiting"},
                    {"kind": "human.submitted"},
                    {"kind": "human.waiting"},  # genuine re-ask on the same path
                ],
            )
            tasks = load_pending_human_tasks(Path(tmp) / "r")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["response_filename"], "t.response.json")

    def test_submitted_last_stays_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(
                Path(tmp),
                [
                    {"kind": "human.waiting"},
                    {"kind": "human.submitted"},
                ],
            )
            tasks = load_pending_human_tasks(Path(tmp) / "r")
        self.assertEqual(tasks, [])

    def test_reask_with_valid_response_file_is_not_listed(self) -> None:
        # A re-ask whose answer is already dropped (in flight) must not re-list.
        with tempfile.TemporaryDirectory() as tmp:
            resp_path = _write_run(
                Path(tmp),
                [
                    {"kind": "human.waiting"},
                    {"kind": "human.submitted"},
                    {"kind": "human.waiting"},
                ],
            )
            resp_path.write_text(json.dumps({"response": "answer"}), encoding="utf-8")
            tasks = load_pending_human_tasks(Path(tmp) / "r")
        self.assertEqual(tasks, [])

    def test_timeout_then_reask_reopens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(
                Path(tmp),
                [
                    {"kind": "human.waiting"},
                    {"kind": "human.timeout"},
                    {"kind": "human.waiting"},
                ],
            )
            tasks = load_pending_human_tasks(Path(tmp) / "r")
        self.assertEqual(len(tasks), 1)


if __name__ == "__main__":
    unittest.main()
