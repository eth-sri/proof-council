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


class HumanSubmitCoercionTests(unittest.TestCase):
    """A10: the submit route coerces a field to its declared output type, so an
    array/object field is not handed to the resumed node as a raw string."""

    def _setup(self, root: Path, output_fields: dict) -> Path:
        run_dir = root / "cr"
        inbox = run_dir / "human_inbox"
        inbox.mkdir(parents=True)
        (run_dir / "run-metadata.json").write_text(
            json.dumps({"status": "running", "display_name": "cr"}), encoding="utf-8"
        )
        task_path = run_dir / "t.task.json"
        task_path.write_text(
            json.dumps({"prompt": "p", "output_fields": output_fields}), encoding="utf-8"
        )
        (run_dir / "events.jsonl").write_text(
            json.dumps(
                {
                    "kind": "human.waiting",
                    "agent": "t",
                    "payload": {
                        "response_path": str(inbox / "t.response.json"),
                        "task_path": str(task_path),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return inbox / "t.response.json"

    def test_declared_types_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            response_file = self._setup(
                root,
                {"items": "array", "meta": "object", "note": "string"},
            )
            app = create_app(runs_roots=(root,))
            with app.test_client() as client:
                resp = client.post(
                    "/run/cr/human",
                    data={
                        "response_filename": "t.response.json",
                        "f_items": "[1, 2, 3]",
                        "f_meta": '{"a": 1}',
                        "f_note": "just text",
                    },
                )
            self.assertEqual(resp.status_code, 302)
            written = json.loads(response_file.read_text(encoding="utf-8"))
            self.assertEqual(written["items"], [1, 2, 3])
            self.assertEqual(written["meta"], {"a": 1})
            self.assertEqual(written["note"], "just text")

    def test_malformed_structured_field_is_rejected(self) -> None:
        # a declared array that isn't valid JSON is rejected (B5): the Outputs
        # model would silently accept a raw string, so the route refuses it and
        # leaves the task pending rather than writing a malformed response.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            response_file = self._setup(root, {"bad": "array"})
            app = create_app(runs_roots=(root,))
            with app.test_client() as client:
                resp = client.post(
                    "/run/cr/human",
                    data={"response_filename": "t.response.json", "f_bad": "not json"},
                )
            self.assertEqual(resp.status_code, 400)
            self.assertFalse(response_file.exists())


if __name__ == "__main__":
    unittest.main()
