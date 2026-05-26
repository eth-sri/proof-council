from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.registry import WorkflowPreset  # noqa: E402


class _DeclaredInputWorkflow:
    class Inputs(BaseModel):
        problem: str = ""
        problem_id: str = ""
        n_attempts: int = 1


class _ExtraInputWorkflow:
    class Inputs(BaseModel):
        model_config = ConfigDict(extra="allow")

        n_attempts: int = 1


class _NoProblemWorkflow:
    class Inputs(BaseModel):
        n_attempts: int = 1


def _preset(workflow_cls: type, inputs: dict) -> WorkflowPreset:
    return WorkflowPreset(
        name="test",
        source_path=Path("test.yaml"),
        workflow_cls=workflow_cls,  # type: ignore[arg-type]
        inputs=inputs,
    )


class WorkflowPresetInputTests(unittest.TestCase):
    def test_problem_text_replaces_empty_preset_default(self) -> None:
        preset = _preset(
            _DeclaredInputWorkflow,
            {"problem": "", "problem_id": "", "n_attempts": 3},
        )

        inputs = preset.build_inputs(
            problem="Prove that there are infinitely many primes.",
            problem_id="euclid",
        )

        self.assertEqual(
            inputs["problem"], "Prove that there are infinitely many primes."
        )
        self.assertEqual(inputs["problem_id"], "euclid")
        self.assertEqual(inputs["n_attempts"], 3)

    def test_problem_text_is_injected_for_extra_allowed_workflows(self) -> None:
        preset = _preset(_ExtraInputWorkflow, {"problem": "", "n_attempts": 2})

        inputs = preset.build_inputs(problem="Show that sqrt(2) is irrational.")

        self.assertEqual(inputs["problem"], "Show that sqrt(2) is irrational.")
        self.assertEqual(inputs["n_attempts"], 2)

    def test_cli_overrides_still_win_over_launched_problem_text(self) -> None:
        preset = _preset(_DeclaredInputWorkflow, {"problem": ""})

        inputs = preset.build_inputs(
            problem="Problem selected in the run UI.",
            cli_overrides={"problem": "Explicit override."},
        )

        self.assertEqual(inputs["problem"], "Explicit override.")

    def test_problem_is_not_injected_into_strict_workflow_without_problem_field(self) -> None:
        preset = _preset(_NoProblemWorkflow, {"problem": "", "n_attempts": 4})

        inputs = preset.build_inputs(problem="This should not be accepted.")

        self.assertNotIn("problem", inputs)
        self.assertEqual(inputs["n_attempts"], 4)


if __name__ == "__main__":
    unittest.main()
