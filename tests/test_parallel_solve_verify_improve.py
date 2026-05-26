from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.dev_data import discover_agent_palette_items, discover_agents, validate_preset_yaml  # noqa: E402
from proofstack.agents.parallel_solve_verify_improve import ParallelSolveVerifyImprove  # noqa: E402
from proofstack.context import RunContext  # noqa: E402


class _FakePromptAgent:
    calls: list[tuple[str, dict]] = []

    def __init__(self, ctx, *, name=None, parent_budget_scope="run") -> None:
        self.ctx = ctx
        self.name = name or "fake"
        self.parent_budget_scope = parent_budget_scope

    async def __call__(self, **kwargs):
        type(self).calls.append((self.name, kwargs))
        if self.name.endswith(".solver"):
            return type("Out", (), {"solution": f"draft-{kwargs['branch_index']}"})()
        if self.name.endswith(".verifier"):
            verdict = "incorrect" if kwargs["iteration"] == 1 else "correct"
            return type("Out", (), {"verification": f"check-{kwargs['branch_index']}-{kwargs['iteration']}", "verdict": verdict})()
        if self.name.endswith(".improver"):
            return type("Out", (), {"solution": f"{kwargs['solution']}-improved"})()
        if self.name.endswith(".merger"):
            return type("Out", (), {"solution": f"merged\n{kwargs['candidate_proofs']}"})()
        raise AssertionError(self.name)


class ParallelSolveVerifyImproveTests(unittest.TestCase):
    def test_runs_n_branches_for_m_passes_and_merges(self) -> None:
        _FakePromptAgent.calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={"parallel": {"n": 2, "m": 2}},
            )
            agent = ParallelSolveVerifyImprove(ctx, name="parallel")
            with patch(
                "proofstack.agents.parallel_solve_verify_improve.ConfigurablePromptAgent",
                _FakePromptAgent,
            ):
                out = asyncio.run(agent(problem="P", literature_search="L", n=99, m=99))

            self.assertIn("draft-1-improved", out.solution)
            self.assertIn("draft-2-improved", out.solution)
            verifier_calls = [call for call in _FakePromptAgent.calls if call[0].endswith(".verifier")]
            improver_calls = [call for call in _FakePromptAgent.calls if call[0].endswith(".improver")]
            self.assertEqual(len(verifier_calls), 4)
            self.assertEqual(len(improver_calls), 2)

            branch_logs = list((Path(temp_dir) / "agents").glob("parallel-*/branches.json"))
            self.assertEqual(len(branch_logs), 1)
            branches = json.loads(branch_logs[0].read_text(encoding="utf-8"))
            self.assertEqual([branch["branch"] for branch in branches], [1, 2])

    def test_python_node_is_discoverable_and_small_on_the_canvas(self) -> None:
        agents = {(agent.module, agent.qualname) for agent in discover_agents()}
        self.assertIn(
            ("proofstack.agents.parallel_solve_verify_improve", "ParallelSolveVerifyImprove"),
            agents,
        )
        palette = {item["agent"]: item for item in discover_agent_palette_items()}
        self.assertEqual(
            palette["proofstack.agents.parallel_solve_verify_improve.ParallelSolveVerifyImprove"]["template"],
            "python_agent",
        )

        report = validate_preset_yaml(
            textwrap.dedent(
                """
                workflow: proofstack.agents.dag_workflow.DAGWorkflow
                components:
                  cfg_parallel:
                    model: models/openai/gpt-54-mini
                    n: 4
                    m: 2
                    solver_system_prompt: custom solver
                    verifier_system_prompt: custom verifier
                    improver_system_prompt: custom improver
                    merger_system_prompt: custom merger
                dag:
                  nodes:
                    - id: parallel
                      kind: agent
                      agent: proofstack.agents.parallel_solve_verify_improve.ParallelSolveVerifyImprove
                      name: cfg_parallel
                      inputs:
                        problem: $input.problem
                        n: 99
                        m: 99
                  outputs:
                    solution: $node.parallel.solution
                """
            )
        )
        nodes = {node["id"]: node for node in report["nodes"]}

        self.assertTrue(report["ok"], report.get("errors"))
        self.assertEqual(nodes["parallel"]["output_fields"], ["solution"])
        self.assertEqual(nodes["parallel"]["input_fields"], ["literature_search"])
        self.assertEqual(nodes["parallel"]["component_config"]["n"], 4)
        self.assertEqual(nodes["parallel"]["component_config"]["m"], 2)
        self.assertIn("__editor__", nodes["parallel"]["component_config"])
        self.assertEqual(nodes["parallel"]["component_config"]["__hidden_inputs__"], ["m", "n"])


if __name__ == "__main__":
    unittest.main()
