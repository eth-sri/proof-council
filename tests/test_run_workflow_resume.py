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
            workflow="human_smoke",
            run_name="Human Smoke",
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


if __name__ == "__main__":
    unittest.main()
