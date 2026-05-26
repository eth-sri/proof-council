from __future__ import annotations

import sys
import textwrap
import unittest
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.dev_data import mutate_preset_yaml, validate_preset_yaml  # noqa: E402


def _edges_to(raw_yaml: str, target: str) -> list[dict[str, Any]]:
    report = validate_preset_yaml(raw_yaml)
    assert report["ok"], report["errors"]
    return [edge for edge in report["edges"] if edge.get("target") == target]


def _edge_paths_to(raw_yaml: str, target: str) -> list[tuple[str, str]]:
    return [(edge["source_path"], edge["target_path"]) for edge in _edges_to(raw_yaml, target)]


def _node(raw_yaml: str, node_id: str) -> dict[str, Any]:
    raw = yaml.safe_load(raw_yaml)
    for node in raw["dag"]["nodes"]:
        if node["id"] == node_id:
            return node
    raise AssertionError(f"missing node {node_id!r}")


RUNTIME_EDGE_FIXTURE = textwrap.dedent(
    """
    workflow: proofstack.agents.dag_workflow.DAGWorkflow
    components:
      cfg_solver:
        output:
          xml_tags: [solution]
          default_field: solution
      cfg_merge:
        output:
          xml_tags: [solution]
          default_field: solution
    dag:
      nodes:
        - id: branches
          kind: map_chain
          foreach: $input.items
          foreach_default: [null]
          collect:
            final: $step.solver.solution
          steps:
            - id: solver
              agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
              name: cfg_solver
              inputs:
                problem: $item
        - id: merged
          kind: agent
          needs: [branches]
          agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
          name: cfg_merge
          when:
            ref: $node.branches.finals
            min_len: 2
          default:
            solution: $node.branches.finals.0
          inputs:
            solutions_text:
              join: $node.branches.finals
              sep: "\\n\\n"
      outputs:
        solution: $node.merged.solution
    """
)

FIRST_PROOF_MERGE_FIXTURES = {
    "minimal_first_proof": textwrap.dedent(
        """
        workflow: proofstack.agents.dag_workflow.DAGWorkflow
        components:
          cfg_solver:
            output:
              xml_tags: [solution]
              default_field: solution
          cfg_merger:
            input_schema:
              problem: string
              solutions_text: string
            output:
              xml_tags: [solution]
              default_field: solution
        dag:
          nodes:
            - id: ideator
              kind: agent
              agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
              name: cfg_solver
              inputs:
                problem: $input.problem
              default:
                approaches: []
            - id: branches
              kind: map_chain
              needs: [ideator]
              foreach: $node.ideator.approaches
              foreach_default: [null]
              collect:
                final:
                  coalesce:
                    - $step.improver.solution
                    - $step.solver.solution
              steps:
                - id: solver
                  agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                  name: cfg_solver
                  inputs:
                    problem: $input.problem
                    approach: $item
                - id: improver
                  agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                  name: cfg_solver
                  default: {}
                  inputs:
                    problem: $input.problem
            - id: merged
              kind: agent
              needs: [branches]
              agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
              name: cfg_merger
              inputs:
                problem: $input.problem
                solutions_text:
                  join: $node.branches.finals
                  sep: "\\n\\n"
          outputs:
            solution: $node.merged.solution
        """
    ),
    "configured_first_proof": textwrap.dedent(
        """
        workflow: proofstack.agents.dag_workflow.DAGWorkflow
        inputs:
          n_approaches: 3
          skip_literature_search: false
        components:
          cfg_ideator:
            output:
              xml_lists:
                approaches: approach
              default_field: text
          cfg_solver:
            output:
              xml_tags: [solution]
              default_field: solution
          cfg_merger:
            input_schema:
              problem: string
              solutions_text: string
            output:
              xml_tags: [solution]
              default_field: solution
        dag:
          nodes:
            - id: ideator
              kind: agent
              agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
              name: cfg_ideator
              inputs:
                problem: $input.problem
                n:
                  coalesce:
                    - $input.n_approaches
                    - 3
              default:
                approaches: []
            - id: literature
              kind: agent
              agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
              name: cfg_solver
              when:
                not:
                  ref: $input.skip_literature_search
                  equals: true
              inputs:
                problem: $input.problem
              default:
                citations: []
            - id: branches
              kind: map_chain
              needs: [ideator]
              foreach: $node.ideator.approaches
              foreach_default: [null]
              collect:
                draft: $step.solver.solution
                final: $step.solver.solution
              steps:
                - id: solver
                  agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                  name: cfg_solver
                  inputs:
                    problem: $input.problem
                    approach: $item
            - id: merged
              kind: agent
              needs: [branches]
              agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
              name: cfg_merger
              inputs:
                problem: $input.problem
                solutions_text:
                  join: $node.branches.finals
                  sep: "\\n\\n---\\n\\n"
          outputs:
            solution: $node.merged.solution
        """
    ),
}


class WorkflowGraphEdgeTests(unittest.TestCase):
    def test_first_proof_merge_has_only_the_real_prompt_input_edge(self) -> None:
        for workflow, raw_yaml in FIRST_PROOF_MERGE_FIXTURES.items():
            with self.subTest(workflow=workflow):
                self.assertEqual(
                    _edge_paths_to(raw_yaml, "merged"),
                    [("finals", "inputs.solutions_text.join")],
                )

    def test_explicit_condition_and_fallback_edges_are_reported(self) -> None:
        self.assertEqual(
            _edge_paths_to(RUNTIME_EDGE_FIXTURE, "merged"),
            [
                ("finals", "when.ref"),
                ("finals.0", "default.solution"),
                ("finals", "inputs.solutions_text.join"),
            ],
        )

    def test_disconnect_condition_edge_removes_when_only(self) -> None:
        result = mutate_preset_yaml(
            RUNTIME_EDGE_FIXTURE,
            {
                "op": "disconnect_edge",
                "target_node": "merged",
                "target_field": "__condition",
            },
        )

        self.assertTrue(result["ok"], result["errors"])
        merged = _node(result["raw_yaml"], "merged")
        self.assertNotIn("when", merged)
        self.assertEqual(merged["default"], {"solution": "$node.branches.finals.0"})
        self.assertEqual(
            _edge_paths_to(result["raw_yaml"], "merged"),
            [
                ("finals.0", "default.solution"),
                ("finals", "inputs.solutions_text.join"),
            ],
        )

    def test_disconnect_fallback_edge_removes_default_only(self) -> None:
        result = mutate_preset_yaml(
            RUNTIME_EDGE_FIXTURE,
            {
                "op": "disconnect_edge",
                "target_node": "merged",
                "target_field": "__fallback.solution",
            },
        )

        self.assertTrue(result["ok"], result["errors"])
        merged = _node(result["raw_yaml"], "merged")
        self.assertEqual(merged["when"], {"ref": "$node.branches.finals", "min_len": 2})
        self.assertNotIn("default", merged)
        self.assertEqual(
            _edge_paths_to(result["raw_yaml"], "merged"),
            [
                ("finals", "when.ref"),
                ("finals", "inputs.solutions_text.join"),
            ],
        )

    def test_connect_condition_preserves_existing_condition_options(self) -> None:
        without_when = mutate_preset_yaml(
            RUNTIME_EDGE_FIXTURE,
            {
                "op": "disconnect_edge",
                "target_node": "merged",
                "target_field": "__condition",
            },
        )["raw_yaml"]

        result = mutate_preset_yaml(
            without_when,
            {
                "op": "connect_edge",
                "source_node": "branches",
                "source_field": "finals",
                "target_node": "merged",
                "target_field": "__condition",
            },
        )

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(_node(result["raw_yaml"], "merged")["when"], {"ref": "$node.branches.finals"})

    def test_connect_if_branch_output_creates_inputs_get_run_condition(self) -> None:
        raw_yaml = textwrap.dedent(
            """
            workflow: proofstack.agents.dag_workflow.DAGWorkflow
            components:
              cfg_target:
                output:
                  xml_tags: [solution]
                  default_field: solution
            dag:
              nodes:
                - id: router
                  kind: if_else
                  condition: true
                  then:
                    'True': true
                  else:
                    'False': true
                - id: target
                  kind: agent
                  agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                  name: cfg_target
              outputs:
                solution: $node.target.solution
            """
        )

        result = mutate_preset_yaml(
            raw_yaml,
            {
                "op": "connect_edge",
                "source_node": "router",
                "source_field": "False",
                "target_node": "target",
                "target_field": "__condition",
            },
        )

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(
            _node(result["raw_yaml"], "target")["when"],
            {"inputs": {"False": "$node.router.False"}, "python": 'inputs.get("False")'},
        )
        self.assertEqual(_edge_paths_to(result["raw_yaml"], "target"), [("False", "when.inputs.False")])

    def test_connect_fallback_can_target_source_subpaths(self) -> None:
        without_default = mutate_preset_yaml(
            RUNTIME_EDGE_FIXTURE,
            {
                "op": "disconnect_edge",
                "target_node": "merged",
                "target_field": "__fallback.solution",
            },
        )["raw_yaml"]

        result = mutate_preset_yaml(
            without_default,
            {
                "op": "connect_edge",
                "source_node": "branches",
                "source_field": "finals.0",
                "target_node": "merged",
                "target_field": "__fallback",
            },
        )

        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(_node(result["raw_yaml"], "merged")["default"], {"solution": "$node.branches.finals.0"})


if __name__ == "__main__":
    unittest.main()
