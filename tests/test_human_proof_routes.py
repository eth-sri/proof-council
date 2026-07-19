from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.dev import create_app  # noqa: E402


def _write_human_run(root: Path, run_id: str) -> None:
    run_dir = root / run_id
    inbox = run_dir / "human_inbox"
    inbox.mkdir(parents=True)
    (run_dir / "run-metadata.json").write_text(
        json.dumps({"status": "running", "display_name": run_id}), encoding="utf-8"
    )
    events = []
    for name, proof in (("taskA", "PROOF-ALPHA"), ("taskB", "PROOF-BETA")):
        task_path = run_dir / f"{name}.task.json"
        task_path.write_text(
            json.dumps({"prompt": "review", "inputs": {"proof": proof}}),
            encoding="utf-8",
        )
        events.append(
            {
                "kind": "human.waiting",
                "agent": name,
                "payload": {
                    "response_path": str(inbox / f"{name}.response.json"),
                    "task_path": str(task_path),
                },
            }
        )
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )


class HumanProofRouteTests(unittest.TestCase):
    def test_stale_task_id_is_404_not_another_tasks_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_human_run(root, "hp-run")
            app = create_app(runs_roots=(root,))
            with app.test_client() as client:
                # valid task id -> that task's proof
                ok = client.get("/run/hp-run/human-proof.tex?task=taskB.response.json")
                self.assertEqual(ok.status_code, 200)
                self.assertIn("PROOF-BETA", ok.get_data(as_text=True))

                # stale/unknown id must 404, never another pending task's proof
                # under the requested filename
                stale = client.get("/run/hp-run/human-proof.tex?task=gone.response.json")
                self.assertEqual(stale.status_code, 404)

                # no id at all keeps the convenience fallback to the first task
                first = client.get("/run/hp-run/human-proof.tex")
                self.assertEqual(first.status_code, 200)
                self.assertIn("PROOF-ALPHA", first.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
