from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agent import Agent  # noqa: E402
from proofstack.context import RunContext  # noqa: E402
from proofstack.monitor import RunMonitor, normalize_monitor_model_spec  # noqa: E402


class _FakeClient:
    model = "fake-monitor"
    queries = []

    def run_queries(self, queries, no_tqdm=False):
        self.queries.append(queries[0])
        yield (
            0,
            [*queries[0], {"role": "assistant", "content": "The solver finished and produced a short proof summary."}],
            {"cost": 0.01, "input_tokens": 12, "output_tokens": 8},
        )


class _BlockingClient:
    model = "blocking-monitor"
    queries = []
    release = threading.Event()

    def run_queries(self, queries, no_tqdm=False):
        self.release.wait(timeout=2)
        self.queries.append(queries[0])
        yield (
            0,
            [*queries[0], {"role": "assistant", "content": "The background monitor finished."}],
            {"cost": 0.0, "input_tokens": 1, "output_tokens": 1},
        )


class _TinyAgent(Agent):
    class Inputs(BaseModel):
        problem: str

    class Outputs(BaseModel):
        solution: str

    async def run(self, inp: Inputs) -> Outputs:
        return self.Outputs(solution=f"solved: {inp.problem}")


class RunMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeClient.queries.clear()
        _BlockingClient.queries.clear()
        _BlockingClient.release.clear()

    def test_agent_end_writes_monitor_summary_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                api_client_factory=lambda _model: _FakeClient(),
            )
            ctx.monitor = RunMonitor(ctx, model="fake", problem="Prove P.", problem_id="p")

            async def run_agent() -> BaseModel:
                out = await _TinyAgent(ctx, name="solver")(problem="P")
                await ctx.monitor.drain()
                return out

            out = asyncio.run(run_agent())

            events = [
                json.loads(line)
                for line in (Path(temp_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            summaries = [event for event in events if event.get("kind") == "monitor.summary"]
            model_calls = [event for event in events if event.get("kind") == "model.call"]

            self.assertEqual(out.solution, "solved: P")
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["payload"]["display_label"], "Solver")
            self.assertNotIn("agent_path", summaries[0]["payload"])
            self.assertIn("short proof summary", summaries[0]["payload"]["summary"])
            self.assertEqual(model_calls[-1]["payload"]["via"], "run_monitor")

    def test_agent_end_does_not_wait_for_monitor_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                api_client_factory=lambda _model: _BlockingClient(),
            )
            ctx.monitor = RunMonitor(ctx, model="blocking", problem="Prove P.", problem_id="p")

            async def run_agent() -> BaseModel:
                try:
                    out = await asyncio.wait_for(_TinyAgent(ctx, name="solver")(problem="P"), timeout=0.25)
                finally:
                    _BlockingClient.release.set()
                await ctx.monitor.drain()
                return out

            out = asyncio.run(run_agent())

            self.assertEqual(out.solution, "solved: P")

            events = [
                json.loads(line)
                for line in (Path(temp_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            summaries = [event for event in events if event.get("kind") == "monitor.summary"]
            self.assertEqual(len(summaries), 1)
            self.assertIn("background monitor", summaries[0]["payload"]["summary"].lower())

    def test_monitor_prompt_uses_node_label_instead_of_internal_dag_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                api_client_factory=lambda _model: _FakeClient(),
            )
            monitor = RunMonitor(
                ctx,
                model="fake",
                problem="Prove P.",
                problem_id="p",
                workflow_structure={
                    "workflow": "test",
                    "nodes": [
                        {
                            "id": "cfg_draft_solution",
                            "label": "Draft solution",
                            "kind": "agent",
                            "needs": [],
                        }
                    ],
                },
            )

            asyncio.run(
                monitor.record_agent_end(
                    call_id="call-1",
                    agent="cfg_draft_solution",
                    agent_path="DAGWorkflow.cfg_draft_solution",
                    execution_mode="api",
                    input_json={"problem": "P"},
                    output_json={"solution": "S"},
                )
            )

            prompt = _FakeClient.queries[-1][-1]["content"]
            event = json.loads((Path(temp_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])

            self.assertIn("Draft solution", prompt)
            self.assertNotIn("DAGWorkflow.cfg_draft_solution", prompt)
            self.assertNotIn("cfg_draft_solution", prompt)
            self.assertEqual(event["payload"]["display_label"], "Draft solution")
            self.assertNotIn("agent_path", event["payload"])

    def test_monitor_maps_stale_gpt55_mini_alias_to_default(self) -> None:
        self.assertEqual(normalize_monitor_model_spec("gpt-5.5"), "models/openai/gpt-54-mini")
        self.assertEqual(normalize_monitor_model_spec("openai/gpt-55-mini"), "models/openai/gpt-54-mini")

    def test_monitor_keeps_standard_sol_aliases_out_of_pro_mode(self) -> None:
        self.assertEqual(
            normalize_monitor_model_spec("gpt-5.6"),
            "models/openai/gpt-56-sol",
        )
        self.assertEqual(
            normalize_monitor_model_spec("gpt-5.6-sol"),
            "models/openai/gpt-56-sol",
        )
        self.assertEqual(
            normalize_monitor_model_spec("gpt-5.6-sol--max"),
            "models/openai/gpt-56-sol-max",
        )
        self.assertEqual(
            normalize_monitor_model_spec("gpt-5.6-sol-pro"),
            "models/openai/gpt-56-sol-pro",
        )

    def test_monitor_accepts_relative_config_ref(self) -> None:
        self.assertEqual(normalize_monitor_model_spec("openai/gpt-54-mini"), "models/openai/gpt-54-mini")


if __name__ == "__main__":
    unittest.main()
