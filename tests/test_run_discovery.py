from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import app.dev as dev  # noqa: E402
from app.dev import create_app  # noqa: E402
from app.dev_data import discover_runs, find_run, resolve_output_refs  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_event(path: Path, **event) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _batch_module():
    spec = importlib.util.spec_from_file_location("run_workflow_batch", ROOT / "scripts" / "run_workflow_batch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunDiscoveryTests(unittest.TestCase):
    def test_resolve_output_refs_inflates_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            blobs = run / "events_blobs"
            blobs.mkdir()
            (blobs / "p.txt").write_text("PROOF BODY", encoding="utf-8")
            out = resolve_output_refs(
                run,
                {
                    "best_tex": {"$ref": "events_blobs/p.txt"},
                    "verdict": "ready",
                    "missing": {"$ref": "events_blobs/nope.txt"},
                },
            )
        self.assertEqual(out["best_tex"], "PROOF BODY")  # inflated
        self.assertEqual(out["verdict"], "ready")  # plain value untouched
        self.assertEqual(out["missing"], {"$ref": "events_blobs/nope.txt"})  # unreadable left as-is

    def test_run_agent_problem_picker_discovers_non_txt_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "example.txt").write_text("Plain text problem.", encoding="utf-8")
            (root / "hilbert.tex").write_text(r"\begin{problem}Hilbert\end{problem}", encoding="utf-8")
            (root / "notes.md").write_text("# Markdown problem", encoding="utf-8")
            (root / ".hidden.tex").write_text("hidden", encoding="utf-8")

            with patch.object(dev, "PROBLEMS_ROOT", root):
                problems = dev._discover_problem_files()

        self.assertEqual([p["id"] for p in problems], ["example", "hilbert.tex", "notes.md"])
        self.assertEqual([p["title"] for p in problems], ["example", "hilbert", "notes"])

    def test_run_agent_problem_selection_accepts_exact_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "example.txt").write_text("Plain text problem.", encoding="utf-8")
            (root / "hilbert.tex").write_text("TeX problem.", encoding="utf-8")

            with patch.object(dev, "PROBLEMS_ROOT", root):
                selected = dev._selected_run_problems({"problems": ["example", "hilbert.tex"]})

        self.assertEqual(
            selected,
            [
                {"id": "example", "text": "Plain text problem.", "latex": "Plain text problem.", "display_name": "Example"},
                {"id": "hilbert", "text": "TeX problem.", "latex": "TeX problem.", "display_name": "Hilbert"},
            ],
        )

    def test_batch_problem_loader_accepts_run_agent_text_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            problems_path = Path(tmp) / "problems.json"
            _write_json(
                problems_path,
                {"problems": [{"id": "sqrt2", "text": "Prove sqrt(2) is irrational.", "display_name": "Sqrt 2"}]},
            )

            problems = _batch_module()._load_problems(problems_path)

        self.assertEqual(
            problems,
            [{"id": "sqrt2", "latex": "Prove sqrt(2) is irrational.", "display_name": "Sqrt 2"}],
        )

    def test_batch_child_run_name_does_not_repeat_single_problem_label(self) -> None:
        batch = _batch_module()

        self.assertEqual(
            batch._child_run_name(
                "Author Critic Smoke Mini · Brokenarxiv Sample",
                "author_critic_smoke_mini",
                {"id": "brokenarxiv_sample", "display_name": "Brokenarxiv Sample"},
            ),
            "Author Critic Smoke Mini · Brokenarxiv Sample",
        )
        self.assertEqual(
            batch._child_run_name(
                "Jaunty Proof · 2 problems",
                "jaunty_proof",
                {"id": "hard", "display_name": "Hard Problem"},
            ),
            "Jaunty Proof · 2 problems · Hard Problem",
        )

    def test_batch_run_detail_waits_for_missing_child_run_instead_of_linking_404(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "batch-run"
            batch_dir.mkdir()
            _write_json(
                batch_dir / "run-metadata.json",
                {
                    "display_name": "Author Critic · Example",
                    "manifest": {
                        "started_at": "2026-06-04T18:04:00",
                        "problems": {
                            "example": {
                                "status": "queued",
                                "problem_id": "example",
                                "display_name": "Example",
                                "run_id": "batch-run-example",
                            }
                        },
                    },
                },
            )
            app = create_app(runs_roots=(root,))

            with app.test_client() as client:
                response = client.get("/run/batch-run")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Waiting for run...", html)
        self.assertNotIn('href="/run/batch-run-example"', html)

    def test_batch_run_detail_shows_early_launcher_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "batch-run"
            batch_dir.mkdir()
            _write_json(
                batch_dir / "run-metadata.json",
                {
                    "status": "starting",
                    "display_name": "Author Critic · Example",
                    "manifest": {
                        "started_at": "2026-06-04T18:04:00",
                        "problems": {
                            "example": {
                                "status": "queued",
                                "problem_id": "example",
                                "display_name": "Example",
                                "run_id": "batch-run-example",
                            }
                        },
                    },
                },
            )
            (batch_dir / "dashboard-subprocess.log").write_text("problem example: empty problem text\n", encoding="utf-8")
            app = create_app(runs_roots=(root,))

            with app.test_client() as client:
                response = client.get("/run/batch-run")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("error", html)
        self.assertIn("problem example: empty problem text", html)
        self.assertNotIn('href="/run/batch-run-example"', html)

    def test_starting_unmonitored_run_polls_graph_without_monitor_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "sample-run"
            run_dir.mkdir()
            _write_json(
                run_dir / "run-metadata.json",
                {
                    "status": "starting",
                    "display_name": "Sample Run",
                    "preset": "demo",
                    "monitor": {"enabled": False, "model": None},
                },
            )
            app = create_app(runs_roots=(root,))

            with app.test_client() as client:
                response = client.get("/run/sample-run")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Execution graph", html)
        self.assertIn("startRunGraphPolling", html)
        self.assertIn("No workflow node events recorded.", html)
        self.assertNotIn("Monitor summaries", html)
        self.assertNotIn('id="run-monitor-panel"', html)
        self.assertNotIn('startRunMonitorPolling("/run/sample-run/monitor-fragment")', html)

    def test_starting_monitored_run_shows_monitor_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "sample-run"
            run_dir.mkdir()
            _write_json(
                run_dir / "run-metadata.json",
                {
                    "status": "starting",
                    "display_name": "Sample Run",
                    "preset": "demo",
                    "monitor": {"enabled": True, "model": "models/openai/gpt-54-mini"},
                },
            )
            app = create_app(runs_roots=(root,))

            with app.test_client() as client:
                response = client.get("/run/sample-run")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Monitor summaries", html)
        self.assertIn("startRunMonitorPolling", html)
        self.assertIn("No monitor summaries yet.", html)

    def test_finished_monitored_run_keeps_monitor_section_from_config_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "sample-run"
            run_dir.mkdir()
            _write_json(
                run_dir / "run-metadata.json",
                {
                    "status": "ok",
                    "display_name": "Sample Run",
                    "config_snapshot": {
                        "monitor": {"enabled": True, "model": "models/openai/gpt-54-mini"},
                    },
                },
            )
            app = create_app(runs_roots=(root,))

            with app.test_client() as client:
                response = client.get("/run/sample-run")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Monitor summaries", html)
        self.assertIn("No monitor summaries yet.", html)

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

    def test_discover_runs_collapses_repeated_display_name_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "duplicated-title"
            run_dir.mkdir()
            events_path = run_dir / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="run.start",
                payload={
                    "display_name": "Author Critic Smoke Mini · Brokenarxiv Sample · Brokenarxiv Sample",
                },
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:01:00.000Z",
                kind="run.end",
                payload={"status": "ok"},
            )

            runs = discover_runs([root])

        self.assertEqual(
            runs[0].display_name,
            "Author Critic Smoke Mini · Brokenarxiv Sample",
        )

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

    def test_batch_parent_status_follows_finished_child_when_manifest_stale(self) -> None:
        # The batch process died after the child finished but before recording
        # the result, so the manifest is stuck "running". The child wrote
        # run.end: ok — its real status must win, so the parent reads "finished"
        # (consistent with what the child/detail page shows), not "running".
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "batch-run"
            batch_dir.mkdir()
            _write_json(
                batch_dir / "run-metadata.json",
                {
                    "status": "running",
                    "manifest": {
                        "started_at": "2026-06-24T15:27:00.000Z",
                        "problems": {
                            "example": {
                                "status": "running",
                                "problem_id": "example",
                                "display_name": "Example",
                                "run_id": "batch-run-example",
                            },
                        },
                    },
                },
            )
            child_dir = root / "batch-run-example"
            child_dir.mkdir()
            events_path = child_dir / "events.jsonl"
            _write_event(events_path, ts="2026-06-24T15:27:00.000Z", kind="run.start", payload={})
            _write_event(events_path, ts="2026-06-24T15:28:00.000Z", kind="run.end", payload={"status": "ok"})

            runs = discover_runs([root])
            parent = find_run([root], "batch-run")

        self.assertEqual([run.run_id for run in runs], ["batch-run"])
        self.assertEqual(runs[0].status, "finished")
        self.assertEqual(parent.status, "finished")
        self.assertFalse(parent.process_dead)

    def test_batch_parent_stays_running_while_a_child_still_runs(self) -> None:
        # A genuinely in-progress batch (one child finished, one still running)
        # must remain "running" — the recompute must not declare it finished early.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "batch-run"
            batch_dir.mkdir()
            _write_json(
                batch_dir / "run-metadata.json",
                {
                    "status": "running",
                    "manifest": {
                        "started_at": "2026-06-24T15:27:00.000Z",
                        "problems": {
                            "a": {"status": "running", "problem_id": "a", "run_id": "batch-run-a"},
                            "b": {"status": "running", "problem_id": "b", "run_id": "batch-run-b"},
                        },
                    },
                },
            )
            done = root / "batch-run-a"
            done.mkdir()
            _write_event(done / "events.jsonl", ts="2026-06-24T15:27:00.000Z", kind="run.start", payload={})
            _write_event(done / "events.jsonl", ts="2026-06-24T15:28:00.000Z", kind="run.end", payload={"status": "ok"})
            going = root / "batch-run-b"
            going.mkdir()
            _write_event(going / "events.jsonl", ts="2026-06-24T15:27:00.000Z", kind="run.start", payload={})

            runs = discover_runs([root])

        self.assertEqual(runs[0].status, "running")

    def test_batch_parent_shows_stopped_over_error_when_child_was_stopped(self) -> None:
        # A stopped child exits non-zero, so the batch records the problem (and
        # itself) as "error". But the child has a stopped marker — the user
        # paused it — so the parent must read "stopped" (resumable), not "error".
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "batch-run"
            batch_dir.mkdir()
            _write_json(
                batch_dir / "run-metadata.json",
                {
                    "status": "error",
                    "manifest": {
                        "started_at": "2026-06-25T06:30:00.000Z",
                        "problems": {
                            "g": {"status": "error", "problem_id": "g", "run_id": "batch-run-g"},
                        },
                    },
                },
            )
            child = root / "batch-run-g"
            child.mkdir()
            _write_event(child / "events.jsonl", ts="2026-06-25T06:30:00.000Z", kind="run.start", payload={})
            _write_json(child / "run-control.json", {"status": "stopped", "signalled": True})

            runs = discover_runs([root])
            child_info = find_run([root], "batch-run-g")

        self.assertEqual(child_info.status, "stopped")
        self.assertEqual(runs[0].status, "stopped")

    def test_discover_runs_sums_and_aggregates_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "batch-run"
            batch_dir.mkdir()
            _write_json(
                batch_dir / "run-metadata.json",
                {
                    "manifest": {
                        "started_at": "2026-06-25T09:00:00.000Z",
                        "finished_at": "2026-06-25T09:02:00.000Z",
                        "problems": {
                            "a": {"status": "ok", "problem_id": "a", "run_id": "batch-run-a"},
                            "b": {"status": "ok", "problem_id": "b", "run_id": "batch-run-b"},
                        },
                    },
                },
            )
            # Child a: two claude model.calls reporting metered_tokens.
            a = root / "batch-run-a"
            a.mkdir()
            _write_event(a / "events.jsonl", ts="2026-06-25T09:00:00.000Z", kind="run.start", payload={})
            _write_event(a / "events.jsonl", ts="2026-06-25T09:01:00.000Z", kind="model.call", payload={"metered_tokens": 1500, "out_tokens": 5})
            _write_event(a / "events.jsonl", ts="2026-06-25T09:01:30.000Z", kind="model.call", payload={"metered_tokens": 500})
            _write_event(a / "events.jsonl", ts="2026-06-25T09:02:00.000Z", kind="run.end", payload={"status": "ok"})
            # Child b: an API-style model.call with only in/out tokens (no metered).
            b = root / "batch-run-b"
            b.mkdir()
            _write_event(b / "events.jsonl", ts="2026-06-25T09:00:00.000Z", kind="run.start", payload={})
            _write_event(b / "events.jsonl", ts="2026-06-25T09:01:00.000Z", kind="model.call", payload={"in_tokens": 500, "out_tokens": 250})
            _write_event(b / "events.jsonl", ts="2026-06-25T09:02:00.000Z", kind="run.end", payload={"status": "ok"})

            find_run([root], "batch-run-a")
            child_a = find_run([root], "batch-run-a")
            child_b = find_run([root], "batch-run-b")
            runs = discover_runs([root])

        self.assertEqual(child_a.tokens, 2000)
        self.assertEqual(child_b.tokens, 750)
        self.assertEqual(runs[0].tokens, 2750)

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

    def test_call_detail_renders_recorded_messages_as_ordered_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            agent_dir = run_dir / "agents" / "ACCritic-c1-critic-call"
            agent_dir.mkdir(parents=True)
            events_path = run_dir / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="agent.start",
                call_id="critic-call",
                agent="ACCritic",
                agent_path="DAGWorkflow.ACCritic",
                execution_mode="agent",
                payload={"input": {"mode": "stateful", "prior_messages": [{"role": "user", "content": "old"}]}},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:01:00.000Z",
                kind="agent.end",
                call_id="critic-call",
                agent="ACCritic",
                agent_path="DAGWorkflow.ACCritic",
                payload={"output": {"review_md": "done"}},
            )
            (agent_dir / "messages.json").write_text(
                json.dumps(
                    [
                        {"role": "user", "content": "initial fresh critic prompt"},
                        {"role": "assistant", "content": "prior referee report"},
                        {"role": "user", "content": "stateful revised-draft prompt"},
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            app = create_app(runs_roots=(root,))

            with app.test_client() as client:
                response = client.get("/run/run/call/1")

        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("user #1", html)
        self.assertIn("assistant #1", html)
        self.assertIn("user #2", html)
        self.assertIn("initial fresh critic prompt", html)
        self.assertIn("prior referee report", html)
        self.assertIn("stateful revised-draft prompt", html)


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
