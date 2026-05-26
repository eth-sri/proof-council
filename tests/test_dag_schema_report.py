from __future__ import annotations

import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.dev_data import validate_preset_yaml  # noqa: E402


def _report(raw_yaml: str) -> dict:
    report = validate_preset_yaml(raw_yaml)
    assert report["ok"], report["errors"]
    return report


SCHEMA_FIXTURE = textwrap.dedent(
    """
    workflow: proofstack.agents.dag_workflow.DAGWorkflow
    components:
      cfg_source:
        output:
          xml_lists:
            proofs: proof
          default_field: text
      cfg_text_target:
        input_schema:
          proof_text: string
        output:
          xml_tags: [solution]
          default_field: solution
    dag:
      nodes:
        - id: source
          kind: agent
          agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
          name: cfg_source
        - id: consumer
          kind: agent
          needs: [source]
          agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
          name: cfg_text_target
          inputs:
            proof_text: $node.source.proofs
      outputs:
        solution: $node.consumer.solution
    """
)


class DAGSchemaReportTests(unittest.TestCase):
    def test_type_mismatch_warning_uses_user_facing_type_names(self) -> None:
        report = _report(SCHEMA_FIXTURE)

        edge = next(edge for edge in report["edges"] if edge["target"] == "consumer")
        self.assertEqual(edge["status"], "warning")
        self.assertEqual(edge["source_schema"]["type"], "array")
        self.assertEqual(edge["target_schema"]["type"], "string")
        self.assertEqual(
            report["warnings"],
            [
                "Type mismatch on edge: source.proofs (list) -> consumer.proof_text (str)."
            ],
        )

    def test_join_transform_satisfies_string_input_schema(self) -> None:
        raw_yaml = SCHEMA_FIXTURE.replace(
            "proof_text: $node.source.proofs",
            "proof_text:\n              join: $node.source.proofs\n              sep: '\\n\\n'",
        )
        report = _report(raw_yaml)

        edge = next(edge for edge in report["edges"] if edge["target"] == "consumer")
        self.assertEqual(edge["target_path"], "inputs.proof_text.join")
        self.assertEqual(edge["status"], "ok")
        self.assertFalse(report["warnings"])

    def test_unused_node_warning_tracks_only_nodes_that_can_affect_outputs(self) -> None:
        raw_yaml = SCHEMA_FIXTURE.replace(
            "  outputs:\n    solution: $node.consumer.solution",
            "    - id: dead_end\n"
            "      kind: agent\n"
            "      agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent\n"
            "      name: cfg_text_target\n"
            "  outputs:\n"
            "    solution: $node.consumer.solution",
        )
        report = _report(raw_yaml)

        warning = next(item for item in report["warnings"] if "not connected" in item)
        self.assertIn("dead end", warning)
        self.assertNotIn("source", warning)
        self.assertNotIn("consumer", warning)

    def test_if_else_output_fields_hide_runtime_condition_branch_and_inputs(self) -> None:
        raw_yaml = textwrap.dedent(
            """
            workflow: proofstack.agents.dag_workflow.DAGWorkflow
            dag:
              nodes:
                - id: router
                  kind: if_else
                  inputs:
                    verdict: $input.verdict
                  condition:
                    ref: $inputs.verdict
                    equals: ok
                  outputs:
                    passthrough: $inputs.verdict
                  then:
                    solution: accepted
                  else:
                    solution: retry
              outputs:
                solution: $node.router.solution
            """
        )

        router = next(node for node in _report(raw_yaml)["nodes"] if node["id"] == "router")
        self.assertEqual(router["output_fields"], ["condition", "passthrough", "solution"])
        self.assertEqual(router["outputs_schema"]["condition"]["type"], "boolean")
        self.assertNotIn("branch", router["outputs_schema"])
        self.assertNotIn("verdict", router["outputs_schema"])

    def test_if_else_defaults_to_true_false_branch_outputs(self) -> None:
        raw_yaml = textwrap.dedent(
            """
            workflow: proofstack.agents.dag_workflow.DAGWorkflow
            dag:
              nodes:
                - id: router
                  kind: if_else
                  inputs:
                    verdict: $input.verdict
                  condition:
                    ref: $inputs.verdict
                    equals: ok
              outputs:
                accepted: $node.router.True
            """
        )

        router = next(node for node in _report(raw_yaml)["nodes"] if node["id"] == "router")
        self.assertEqual(router["output_fields"], ["False", "True", "condition"])
        self.assertEqual(set(router["outputs_schema"]), {"False", "True", "condition"})

    def test_repeat_node_output_fields_are_only_explicit_outputs(self) -> None:
        raw_yaml = textwrap.dedent(
            """
            workflow: proofstack.agents.dag_workflow.DAGWorkflow
            dag:
              nodes:
                - id: loop
                  kind: repeat
                  max_iterations: 2
                  condition: true
                  initial_state:
                    solution: $input.seed
                  body:
                    nodes:
                      - id: marker
                        kind: if_else
                        condition: false
                        else:
                          solution: done
                    state_updates:
                      solution: $node.marker.solution
                  outputs:
                    solution: $state.solution
              outputs:
                solution: $node.loop.solution
            """
        )

        loop = next(node for node in _report(raw_yaml)["nodes"] if node["id"] == "loop")
        self.assertEqual(loop["output_fields"], ["solution"])
        self.assertEqual(loop["outputs_schema"], {"solution": {"type": "string"}})

    def test_regular_workflow_outputs_count_as_returned_outputs(self) -> None:
        raw_yaml = textwrap.dedent(
            """
            workflow: proofstack.agents.dag_workflow.DAGWorkflow
            components:
              cfg:
                output:
                  xml_tags: [solution]
                  default_field: solution
            dag:
              nodes:
                - id: solver
                  kind: agent
                  agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                  name: cfg
              outputs:
                solution: $node.solver.solution
            """
        )
        report = _report(raw_yaml)

        edge = next(edge for edge in report["edges"] if edge["edge_kind"] == "output")
        self.assertEqual(edge["source"], "solver")
        self.assertEqual(edge["target_path"], "outputs.solution")
        self.assertFalse(report["warnings"])

    def test_coalesced_workflow_outputs_render_each_possible_source_edge(self) -> None:
        raw_yaml = textwrap.dedent(
            """
            workflow: proofstack.agents.dag_workflow.DAGWorkflow
            components:
              cfg:
                output:
                  xml_tags: [solution]
                  default_field: solution
            dag:
              nodes:
                - id: correct_branch
                  kind: agent
                  agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                  name: cfg
                - id: incorrect_branch
                  kind: agent
                  agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                  name: cfg
              outputs:
                solution:
                  coalesce:
                    - $node.correct_branch.solution
                    - $node.incorrect_branch.solution
            """
        )
        report = _report(raw_yaml)

        output_edges = [edge for edge in report["edges"] if edge["edge_kind"] == "output"]
        self.assertEqual(
            [(edge["source"], edge["target_path"]) for edge in output_edges],
            [
                ("correct_branch", "outputs.solution.coalesce.0"),
                ("incorrect_branch", "outputs.solution.coalesce.1"),
            ],
        )

    def test_configured_prompt_outputs_do_not_add_synthetic_text_field(self) -> None:
        raw_yaml = textwrap.dedent(
            """
            workflow: proofstack.agents.dag_workflow.DAGWorkflow
            components:
              cfg_verifier:
                output:
                  xml_tags: [verification]
                  default_field: verification
            dag:
              nodes:
                - id: verifier
                  kind: agent
                  agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                  name: cfg_verifier
              outputs:
                verification: $node.verifier.verification
            """
        )

        verifier = next(node for node in _report(raw_yaml)["nodes"] if node["id"] == "verifier")

        self.assertEqual(verifier["output_fields"], ["verification"])
        self.assertEqual(verifier["outputs_schema"], {"verification": {"type": "string"}})

    def test_global_workflow_inputs_are_not_reported_as_node_sockets(self) -> None:
        raw_yaml = textwrap.dedent(
            """
            workflow: proofstack.agents.dag_workflow.DAGWorkflow
            inputs:
              problem: ""
            components:
              cfg_solver:
                user_prompt: "{problem}"
                output:
                  xml_tags: [solution]
                  default_field: solution
              cfg_verifier:
                user_prompt: "{problem}\\n{solution}"
                output:
                  xml_tags: [verification]
                  default_field: verification
            dag:
              nodes:
                - id: solver
                  kind: agent
                  agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                  name: cfg_solver
                - id: verifier
                  kind: agent
                  needs: [solver]
                  agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                  name: cfg_verifier
                  inputs:
                    solution: $node.solver.solution
              outputs:
                verification: $node.verifier.verification
            """
        )
        report = _report(raw_yaml)

        solver = next(node for node in report["nodes"] if node["id"] == "solver")
        verifier = next(node for node in report["nodes"] if node["id"] == "verifier")
        self.assertEqual(solver["input_fields"], [])
        self.assertEqual(verifier["input_fields"], ["solution"])

    def test_workflow_named_field_stays_visible_when_wired_from_a_node(self) -> None:
        raw_yaml = textwrap.dedent(
            """
            workflow: proofstack.agents.dag_workflow.DAGWorkflow
            inputs:
              problem: ""
            components:
              cfg:
                user_prompt: "{problem}"
                output:
                  xml_tags: [problem]
                  default_field: problem
            dag:
              nodes:
                - id: source
                  kind: agent
                  agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                  name: cfg
                - id: consumer
                  kind: agent
                  needs: [source]
                  agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                  name: cfg
                  inputs:
                    problem: $node.source.problem
              outputs:
                problem: $node.consumer.problem
            """
        )

        consumer = next(node for node in _report(raw_yaml)["nodes"] if node["id"] == "consumer")
        self.assertEqual(consumer["input_fields"], ["problem"])

    def test_workspace_stays_visible_when_wired_from_workflow_input(self) -> None:
        raw_yaml = textwrap.dedent(
            """
            workflow: proofstack.agents.dag_workflow.DAGWorkflow
            inputs:
              workspace: ""
            components:
              cfg_cli:
                cmd: [sh, -c, "finish '{\\"status\\":\\"done\\"}'"]
                input_schema:
                  workspace: string
            dag:
              nodes:
                - id: worker
                  kind: agent
                  agent: proofstack.agents.configurable_cli.ConfigurableCLIAgent
                  name: cfg_cli
                  inputs:
                    workspace: $input.workspace
              outputs:
                workspace: $node.worker.workspace
            """
        )

        worker = next(node for node in _report(raw_yaml)["nodes"] if node["id"] == "worker")
        self.assertEqual(worker["input_fields"], ["workspace"])


if __name__ == "__main__":
    unittest.main()
