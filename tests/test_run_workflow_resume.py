from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _run_workflow_module():
    spec = importlib.util.spec_from_file_location(
        "run_workflow",
        ROOT / "scripts" / "run_workflow.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunWorkflowResumeTests(unittest.TestCase):
    def test_resume_spec_preserves_instruction_and_budget_overrides(self) -> None:
        module = _run_workflow_module()
        args = SimpleNamespace(
            workflow="human_loop_demo",
            run_name="Human Council",
            input=["claude_model=haiku"],
            model=[],
            component=[],
            additional_instructions="Use the short proof.",
            budget_usd=1.25,
            monitor=True,
            monitor_model="models/openai/gpt-54-mini",
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            module._write_resume_spec(
                run_dir,
                args,
                "Problem text",
                "problem-id",
                "run-id",
                Path("outputs"),
            )

            spec = json.loads((run_dir / "resume.json").read_text(encoding="utf-8"))

        argv = spec["argv"]
        self.assertIn("--additional-instructions", argv)
        self.assertEqual(
            argv[argv.index("--additional-instructions") + 1],
            "Use the short proof.",
        )
        self.assertIn("--budget-usd", argv)
        self.assertEqual(argv[argv.index("--budget-usd") + 1], "1.25")


class BudgetSeedTests(unittest.TestCase):
    def test_resume_seeds_root_tracker_from_prior_events(self) -> None:
        from proofstack.context import RunContext

        module = _run_workflow_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            events = [
                {"kind": "model.call", "payload": {"cost_usd": 1.5, "in_tokens": 100, "out_tokens": 20}},
                {"kind": "model.call", "payload": {"cost_usd": 0.0, "metered_tokens": 5000}},
                {"kind": "tool.call", "payload": {}},
                {"kind": "tool.call", "payload": {}},
                {"kind": "run.start", "payload": {}},
            ]
            (run_dir / "events.jsonl").write_text(
                "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
            )
            ctx = RunContext.create(run_id="r", root_workdir=run_dir, flat=True)
            module._seed_budget_from_prior_events(ctx)
            root = ctx.budgets.root("run")
        self.assertAlmostEqual(root.counters.usd, 1.5)
        self.assertEqual(root.counters.tokens, 5120)
        self.assertEqual(root.counters.tool_calls, 2)

    def test_fresh_run_seeds_nothing(self) -> None:
        from proofstack.context import RunContext

        module = _run_workflow_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            ctx = RunContext.create(run_id="r", root_workdir=run_dir, flat=True)
            module._seed_budget_from_prior_events(ctx)
            root = ctx.budgets.root("run")
        self.assertEqual(root.counters.usd, 0.0)
        self.assertEqual(root.counters.tokens, 0)


if __name__ == "__main__":
    unittest.main()
