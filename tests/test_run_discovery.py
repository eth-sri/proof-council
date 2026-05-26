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
from app.dev_data import discover_runs, find_run  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_event(path: Path, **event) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


class RunDiscoveryTests(unittest.TestCase):
    def test_discover_runs_uses_display_name_and_problem_summary_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "internal-run-id"
            run_dir.mkdir()
            _write_json(
                run_dir / "run-metadata.json",
                {
                    "display_name": "Readable Batch",
                    "manifest": {
                        "started_at": "2026-05-08T09:00:00",
                        "problems": {
                            "sqrt2": {
                                "status": "ok",
                                "problem_id": "sqrt2",
                                "display_name": "Square Root 2",
                            },
                            "primes": {
                                "status": "running",
                                "problem_id": "primes",
                                "display_name": "Infinitely Many Primes",
                            },
                        },
                    },
                },
            )

            runs = discover_runs([root])

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].display_name, "Readable Batch")
        self.assertEqual(runs[0].problem_summary, "2 problems")
        self.assertEqual(runs[0].status, "running")

    def test_discover_runs_fills_single_problem_from_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "timestamp-id"
            run_dir.mkdir()
            events_path = run_dir / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="run.start",
                payload={"preset": "missing_agent", "problem_id": "sqrt2_problem"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:01:00.000Z",
                kind="run.end",
                payload={"status": "ok"},
            )

            runs = discover_runs([root])

        self.assertEqual(runs[0].problem_summary, "Sqrt2 Problem")
        self.assertEqual(runs[0].n_problems, 1)
        self.assertEqual(runs[0].display_name, "Missing Agent · Sqrt2 Problem")
        self.assertEqual(runs[0].status, "finished")

    def test_discover_runs_normalizes_error_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "failed-run"
            run_dir.mkdir()
            _write_json(
                run_dir / "run-metadata.json",
                {"status": "failed", "display_name": "Failed Run"},
            )

            runs = discover_runs([root])

        self.assertEqual(runs[0].status, "error")

    def test_discover_runs_treats_last_gasp_as_error_even_with_ok_run_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "last-gasp-run"
            run_dir.mkdir()
            events_path = run_dir / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="run.start",
                payload={"preset": "demo", "problem_id": "sqrt2"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:01.000Z",
                kind="workflow.last_gasp",
                payload={"type": "KeyError", "msg": "'problem'"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:02.000Z",
                kind="run.end",
                payload={"status": "ok"},
            )

            runs = discover_runs([root])

        self.assertEqual(runs[0].status, "error")

    def test_discover_runs_hides_batch_children_and_sums_child_costs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "batch-run"
            batch_dir.mkdir()
            _write_json(
                batch_dir / "run-metadata.json",
                {
                    "display_name": "Jaunty Proof · 2 problems",
                    "manifest": {
                        "started_at": "2026-05-09T09:58:00.000Z",
                        "finished_at": "2026-05-09T10:01:00.000Z",
                        "problems": {
                            "example": {
                                "status": "ok",
                                "problem_id": "example",
                                "display_name": "Example",
                                "run_id": "batch-run-example",
                            },
                            "hard": {
                                "status": "ok",
                                "problem_id": "hard",
                                "display_name": "Hard Problem",
                                "run_id": "batch-run-hard",
                            },
                        },
                    },
                },
            )
            for run_id, cost in (("batch-run-example", 0.0033), ("batch-run-hard", 0.0164)):
                child_dir = root / run_id
                child_dir.mkdir()
                events_path = child_dir / "events.jsonl"
                _write_event(
                    events_path,
                    ts="2026-05-09T09:58:00.000Z",
                    kind="run.start",
                    payload={"preset": "jaunty_proof", "problem_id": run_id},
                )
                _write_event(
                    events_path,
                    ts="2026-05-09T10:01:00.000Z",
                    kind="model.call",
                    payload={"cost_usd": cost},
                )
                _write_event(
                    events_path,
                    ts="2026-05-09T10:01:00.000Z",
                    kind="run.end",
                    payload={"status": "ok"},
                )

            runs = discover_runs([root])
            child = find_run([root], "batch-run-hard")

        self.assertEqual([run.run_id for run in runs], ["batch-run"])
        self.assertAlmostEqual(runs[0].cost_usd or 0.0, 0.0197)
        self.assertEqual(runs[0].problem_summary, "2 problems")
        self.assertEqual(runs[0].wallclock_s, 180.0)
        self.assertIsNotNone(child)
        self.assertEqual(child.run_id, "batch-run-hard")

    def test_runs_page_renders_batch_row_without_child_rows_or_utc_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "batch-run"
            batch_dir.mkdir()
            _write_json(
                batch_dir / "run-metadata.json",
                {
                    "display_name": "Jaunty Proof · 2 problems",
                    "manifest": {
                        "started_at": "2026-05-09T09:58:00.000Z",
                        "finished_at": "2026-05-09T10:01:00.000Z",
                        "problems": {
                            "example": {
                                "status": "ok",
                                "display_name": "Example",
                                "run_id": "batch-run-example",
                            },
                            "hard": {
                                "status": "ok",
                                "display_name": "Hard Problem",
                                "run_id": "batch-run-hard",
                            },
                        },
                    },
                },
            )
            for run_id, display_name, cost in (
                ("batch-run-example", "Jaunty Proof · 2 problems · Example", 0.0033),
                ("batch-run-hard", "Jaunty Proof · 2 problems · Hard Problem", 0.0164),
            ):
                child_dir = root / run_id
                child_dir.mkdir()
                _write_json(child_dir / "run-metadata.json", {"display_name": display_name})
                _write_event(
                    child_dir / "events.jsonl",
                    ts="2026-05-09T09:58:00.000Z",
                    kind="model.call",
                    payload={"cost_usd": cost},
                )
            app = create_app(runs_roots=(root,))

            with app.test_client() as client:
                response = client.get("/runs")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Jaunty Proof · 2 problems", html)
        self.assertIn("$0.0197", html)
        self.assertIn(">batch<", html)
        self.assertIn("2026-05-09 11:58", html)
        self.assertNotIn("Jaunty Proof · 2 problems · Example", html)
        self.assertNotIn("Jaunty Proof · 2 problems · Hard Problem", html)
        self.assertNotIn("no events", html)
        self.assertNotIn("UTC", html)

    def test_runs_page_hides_mode_column_and_uses_display_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "internal-run-id"
            run_dir.mkdir()
            _write_json(
                run_dir / "run-metadata.json",
                {
                    "display_name": "Readable Run",
                    "manifest": {"problems": {"sqrt2": {"status": "queued", "problem_id": "sqrt2"}}},
                },
            )
            app = create_app(runs_roots=(root,))

            with app.test_client() as client:
                response = client.get("/runs")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Readable Run", html)
        self.assertIn("<th>Status</th>", html)
        self.assertIn("running", html)
        self.assertIn("Sqrt2", html)
        self.assertNotIn("<th>Mode</th>", html)

    def test_call_detail_hides_problem_field_when_problem_box_is_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            events_path = run_dir / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="agent.start",
                call_id="solver-call",
                agent="Solver",
                agent_path="Solver",
                execution_mode="agent",
                payload={
                    "input": {
                        "problem": "Prove that the square root of 2 is irrational.",
                        "solution": None,
                        "approach": None,
                        "attempt": 2,
                    }
                },
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:01:00.000Z",
                kind="agent.end",
                call_id="solver-call",
                agent="Solver",
                agent_path="Solver",
                payload={"output": {"result": "done"}},
            )
            app = create_app(runs_roots=(root,))

            with app.test_client() as client:
                response = client.get("/run/run/call/1")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Prove that the square root of 2 is irrational.", html)
        self.assertNotIn("<th>problem</th>", html)
        self.assertNotIn("<th>solution</th>", html)
        self.assertNotIn("<th>approach</th>", html)
        self.assertNotIn(">none<", html.lower())
        self.assertIn("<th>attempt</th>", html)
        self.assertIn("<details>\n        <summary>Input</summary>", html)
        self.assertNotIn("<details open>\n        <summary>Input</summary>", html)

    def test_call_detail_renders_non_solution_text_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            agent_dir = run_dir / "agents" / "cfg_prompt-c0-verifier"
            agent_dir.mkdir(parents=True)
            events_path = run_dir / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="agent.start",
                call_id="verifier",
                agent="cfg_prompt",
                agent_path="DAGWorkflow.cfg_prompt",
                execution_mode="agent",
                payload={"input": {"solution": "draft"}},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:01:00.000Z",
                kind="agent.end",
                call_id="verifier",
                agent="cfg_prompt",
                agent_path="DAGWorkflow.cfg_prompt",
                payload={"output": {"verification": "The proof has a gap."}},
            )
            _write_json(
                agent_dir / "output.json",
                {"verification": "The proof has a gap.", "raw_text": "internal raw text"},
            )
            app = create_app(runs_roots=(root,))

            with app.test_client() as client:
                response = client.get("/run/run/call/1")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("<summary>verification</summary>", html)
        self.assertIn("The proof has a gap.", html)
        self.assertNotIn("internal raw text", html)

    def test_internal_events_are_not_exposed_in_run_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            events_path = run_dir / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="run.start",
                payload={"preset": "demo", "internal": "run-visible-only-to-machines"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:01.000Z",
                kind="agent.start",
                call_id="solver-call",
                agent="Solver",
                agent_path="Solver",
                execution_mode="agent",
                payload={"input": {"attempt": 1}},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:02.000Z",
                kind="workflow.debug",
                parent_call_id="solver-call",
                payload={"internal": "framework-visible-only-to-machines"},
            )
            app = create_app(runs_roots=(root,))

            with app.test_client() as client:
                run_response = client.get("/run/run")
                call_response = client.get("/run/run/call/1")

        run_html = run_response.get_data(as_text=True)
        call_html = call_response.get_data(as_text=True)
        self.assertEqual(run_response.status_code, 200)
        self.assertEqual(call_response.status_code, 200)
        self.assertNotIn("Run-level events", run_html)
        self.assertNotIn("run-visible-only-to-machines", run_html)
        self.assertNotIn("framework events", call_html)
        self.assertNotIn("framework-visible-only-to-machines", call_html)

    def test_child_call_list_does_not_show_call_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            events_path = run_dir / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="agent.start",
                call_id="parent-call",
                agent="Parent",
                agent_path="Parent",
                execution_mode="agent",
                payload={"input": {}},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:01.000Z",
                kind="agent.start",
                call_id="child-call",
                parent_call_id="parent-call",
                agent="Child",
                agent_path="Parent.Child",
                execution_mode="agent",
                payload={"input": {}},
            )
            app = create_app(runs_roots=(root,))

            with app.test_client() as client:
                response = client.get("/run/run/call/1")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Child", html)
        self.assertNotIn("[child-", html)
        self.assertNotIn("child-call", html)

    def test_raw_call_id_urls_are_not_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            _write_event(
                run_dir / "events.jsonl",
                ts="2026-05-08T09:00:00.000Z",
                kind="agent.start",
                call_id="raw-call-id",
                agent="Solver",
                agent_path="Solver",
                execution_mode="agent",
                payload={"input": {}},
            )
            app = create_app(runs_roots=(root,))

            with app.test_client() as client:
                raw_response = client.get("/run/run/call/raw-call-id")
                ref_response = client.get("/run/run/call/1")

        self.assertEqual(raw_response.status_code, 404)
        self.assertEqual(ref_response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
