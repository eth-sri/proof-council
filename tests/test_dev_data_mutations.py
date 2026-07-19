from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.dev_data import (  # noqa: E402
    CONFIGURABLE_CLI_AGENT,
    _op_set_executor,
    discover_exported_presets,
    discover_presets,
    mutate_preset_yaml,
    preset_dag_report,
    save_tool_definition,
    validate_preset_yaml,
)
from proofstack.registry import PresetError  # noqa: E402


def _raw(raw_yaml: str) -> dict[str, Any]:
    return yaml.safe_load(raw_yaml)


def _node(raw_yaml: str, node_id: str) -> dict[str, Any]:
    for node in _raw(raw_yaml)["dag"]["nodes"]:
        if node["id"] == node_id:
            return node
    raise AssertionError(f"missing node {node_id!r}")


def _mutate(raw_yaml: str, operation: dict[str, Any]) -> str:
    result = mutate_preset_yaml(raw_yaml, operation)
    assert result["ok"], result["errors"]
    return result["raw_yaml"]


EDITOR_FIXTURE = textwrap.dedent(
    """
    workflow: proofstack.agents.dag_workflow.DAGWorkflow
    inputs:
      problem: Prove something.
      max_usd: 12
      custom: old
    budget:
      max_usd: 12
      max_wallclock_s: 300
    components:
      cfg_source:
        user_prompt: source
        output:
          xml_tags: [text]
          default_field: text
      cfg_consumer:
        user_prompt: consumer
        input_schema:
          value: string
        output:
          xml_tags: [solution]
          default_field: solution
      cfg_unused:
        user_prompt: unused
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
          ui:
            managed_needs: [source]
          agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
          name: cfg_consumer
          inputs:
            value: $node.source.text
      outputs:
        solution: $node.consumer.solution
      ui:
        workflow_output:
          x: 10
          y: 20
    """
)

EDITOR_FIXTURE_WITH_EXTRA_SOURCE = EDITOR_FIXTURE.replace(
    "    - id: consumer\n",
    """    - id: extra
      kind: agent
      agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
      name: cfg_unused
    - id: consumer
""",
)

REPEAT_FIXTURE = textwrap.dedent(
    """
    workflow: proofstack.agents.dag_workflow.DAGWorkflow
    inputs:
      problem: Prove something.
    components:
      cfg_source:
        user_prompt: source
        output:
          xml_tags: [solution]
          default_field: solution
      cfg_consumer:
        user_prompt: consumer
        input_schema:
          value: string
        output:
          xml_tags: [solution]
          default_field: solution
    dag:
      nodes:
        - id: seed
          kind: agent
          agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
          name: cfg_source
        - id: loop
          kind: repeat
          needs: [seed]
          max_iterations: 3
          condition:
            python: iteration < max_iterations
          initial_state:
            solution: $node.seed.solution
            verdict: gap
          body:
            nodes:
              - id: verifier
                kind: if_else
                inputs:
                  verdict: $state.verdict
                condition:
                  ref: $inputs.verdict
                  equals: ok
                then:
                  ok: true
                else:
                  gap: true
              - id: improver
                kind: agent
                needs: [verifier]
                agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                name: cfg_consumer
                when:
                  ref: $node.verifier.gap
                  equals: true
                default:
                  solution: $state.solution
                inputs:
                  value: $node.verifier.gap
            state_updates:
              solution: $node.improver.solution
          outputs:
            solution: $state.solution
      outputs:
        solution: $node.loop.solution
    """
)

BUDGET_FALLBACK_FIXTURE = textwrap.dedent(
    """
    workflow: proofstack.agents.dag_workflow.DAGWorkflow
    components:
      cfg_source:
        user_prompt: source
        output:
          xml_tags: [solution]
          default_field: solution
    dag:
      nodes:
        - id: source
          kind: agent
          agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
          name: cfg_source
        - id: extra
          kind: agent
          agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
          name: cfg_source
        - id: budget_fallback
          kind: agent
          run_on: budget_exhausted
          agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
          name: cfg_source
          needs: [source]
          inputs:
            available_outputs: {}
      outputs:
        solution: $node.source.solution
    """
)

ALETHEIA_REPEAT_FIXTURE = textwrap.dedent(
    """
    workflow: proofstack.agents.dag_workflow.DAGWorkflow
    inputs:
      problem: Prove something.
      max_iterations: 5
    components:
      cfg_solver:
        user_prompt: Solve the problem.
        input_schema:
          problem: string
          restart_reason: string
          attempt: integer
        output:
          xml_tags: [solution]
          default_field: solution
      cfg_verifier:
        user_prompt: Verify the proof.
        input_schema:
          problem: string
          solution: string
        output:
          xml_tags: [verdict, feedback]
          default_field: feedback
      cfg_improver:
        user_prompt: Improve the proof.
        input_schema:
          problem: string
          previous_solution: string
          gap_report: string
        output:
          xml_tags: [solution]
          default_field: solution
    dag:
      nodes:
        - id: initial_proof
          kind: agent
          agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
          name: cfg_solver
          inputs:
            problem: $input.problem
            restart_reason: ''
            attempt: 1
        - id: aletheia_repeat
          kind: repeat
          needs: [initial_proof]
          max_iterations:
            coalesce:
              - $input.max_iterations
              - 5
          condition:
            python: iteration < max_iterations
          initial_state:
            solution: $node.initial_proof.solution
            verdict: gap
            feedback: Verify the initial proof before accepting it.
          body:
            nodes:
              - id: verifier
                kind: agent
                agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                name: cfg_verifier
                soft_fail: true
                default:
                  verdict: ok
                  feedback: Verification failed; returning current proof.
                inputs:
                  problem: $input.problem
                  solution: $state.solution
              - id: verifier_error
                kind: if_else
                needs: [verifier]
                inputs:
                  verdict: $node.verifier.verdict
                condition:
                  any:
                    - ref: $inputs.verdict
                      equals: critical
                    - ref: $inputs.verdict
                      equals: wrong
                    - ref: $inputs.verdict
                      equals: error
                then_label: error
                else_label: not error
                then:
                  error: true
                else:
                  not_error: true
              - id: verifier_complete
                kind: if_else
                needs: [verifier_error]
                when:
                  ref: $node.verifier_error.not_error
                  equals: true
                inputs:
                  verdict: $node.verifier.verdict
                condition:
                  ref: $inputs.verdict
                  equals: ok
                then_label: complete
                else_label: gap
                then: {}
                else:
                  needs_improvement: true
              - id: improve_gap
                kind: agent
                needs: [verifier_complete]
                agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                name: cfg_improver
                soft_fail: true
                when:
                  ref: $node.verifier_complete.needs_improvement
                  equals: true
                default: {}
                inputs:
                  problem: $input.problem
                  previous_solution: $state.solution
                  gap_report: $node.verifier.feedback
              - id: restart_proof
                kind: agent
                needs: [verifier_error]
                agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                name: cfg_solver
                soft_fail: true
                when:
                  ref: $node.verifier_error.error
                  equals: true
                default: {}
                inputs:
                  problem: $input.problem
                  restart_reason: $node.verifier.feedback
                  attempt:
                    format: "{value}"
                    fields:
                      value: $iteration
            state_updates:
              verdict: $node.verifier.verdict
              feedback: $node.verifier.feedback
              solution:
                coalesce:
                  - $node.restart_proof.solution
                  - $node.improve_gap.solution
                  - $state.solution
          outputs:
            solution: $state.solution
      outputs:
        solution: $node.aletheia_repeat.solution
    """
)


class DevDataMutationTests(unittest.TestCase):
    def test_add_if_else_node_uses_string_true_false_output_names(self) -> None:
        raw_yaml = _mutate(
            textwrap.dedent(
                """
                workflow: proofstack.agents.dag_workflow.DAGWorkflow
                dag:
                  nodes: []
                  outputs: {}
                """
            ),
            {
                "op": "add_node",
                "template": "if_else",
                "node_id": "router",
                "x": 0,
                "y": 0,
            },
        )
        router = _node(raw_yaml, "router")

        self.assertEqual(router["then"], {"True": True})
        self.assertEqual(router["else"], {"False": True})
        self.assertNotIn(True, router["then"])
        self.assertNotIn(False, router["else"])

    def test_removed_proof_work_templates_cannot_be_added(self) -> None:
        raw_yaml = textwrap.dedent(
            """
            workflow: proofstack.agents.dag_workflow.DAGWorkflow
            dag:
              nodes: []
              outputs: {}
            """
        )
        for template in ("solver", "validator", "improver", "map_chain", "join"):
            with self.subTest(template=template):
                result = mutate_preset_yaml(raw_yaml, {"op": "add_node", "template": template})
                self.assertFalse(result["ok"])
                self.assertIn("has been removed", " ".join(result["errors"]))

    def test_add_parallel_svi_node_creates_python_agent_node(self) -> None:
        raw_yaml = _mutate(
            textwrap.dedent(
                """
                workflow: proofstack.agents.dag_workflow.DAGWorkflow
                dag:
                  nodes: []
                  outputs: {}
                """
            ),
            {
                "op": "add_node",
                "template": "python_agent",
                "agent": "proofstack.agents.parallel_solve_verify_improve.ParallelSolveVerifyImprove",
                "node_id": "parallel",
                "label": "Parallel Solve / Verify / Improve",
                "x": 0,
                "y": 0,
            },
        )
        raw = _raw(raw_yaml)
        node = _node(raw_yaml, "parallel")

        self.assertEqual(
            node["agent"],
            "proofstack.agents.parallel_solve_verify_improve.ParallelSolveVerifyImprove",
        )
        self.assertEqual(node["inputs"], {"problem": "$input.problem"})
        self.assertEqual(raw["components"]["cfg_parallel"]["model"], "models/openai/gpt-54-mini")
        self.assertEqual(raw["components"]["cfg_parallel"]["n"], 3)
        self.assertEqual(raw["components"]["cfg_parallel"]["m"], 3)
        self.assertIn("solver_system_prompt", raw["components"]["cfg_parallel"])
        self.assertTrue(validate_preset_yaml(raw_yaml)["ok"])

    def test_add_cli_agent_uses_codex_template(self) -> None:
        raw_yaml = _mutate(
            textwrap.dedent(
                """
                workflow: proofstack.agents.dag_workflow.DAGWorkflow
                dag:
                  nodes: []
                  outputs: {}
                """
            ),
            {
                "op": "add_node",
                "template": "cli_agent",
                "node_id": "cli",
                "x": 0,
                "y": 0,
            },
        )
        raw = _raw(raw_yaml)
        cfg = raw["components"]["cfg_cli"]

        self.assertEqual(cfg["cmd"][:2], ["codex", "exec"])
        self.assertEqual(cfg["model"], "gpt-5.4-mini")
        self.assertEqual(cfg["model_reasoning_effort"], "low")
        self.assertEqual(cfg["codex_sandbox"], "auto")
        self.assertIs(cfg["copy_codex_auth"], True)
        self.assertIs(cfg["sandbox"]["docker_no_new_privileges"], False)
        self.assertEqual(cfg["usage"]["type"], "codex_jsonl")
        # Built on the shared executor scaffold: delivery mechanics are
        # auto-appended, so the starter prompt must not hand-write them.
        self.assertEqual(cfg["contract"], "auto")
        self.assertNotIn("finish", cfg["prompt"])

    EXECUTOR_FIXTURE = textwrap.dedent(
        """
        workflow: proofstack.agents.dag_workflow.DAGWorkflow
        inputs:
          problem: ''
        components:
          cfg_hint:
            cmd: [claude, -p, --output-format, stream-json, --verbose, --model, sonnet,
                  --permission-mode, acceptEdits, --allowedTools, 'Bash(finish:*)']
            soft_timeout_s: 780
            sandbox: {backend: subprocess, timeout_s: 900}
            env: {HOME: '{env:HOME}'}
            usage: {type: claude_json}
            cache_enabled: true
            contract: auto
            prompt: |
              You are the specialist.

              Problem:
              {problem}
            input_schema:
              problem: string
              workspace: string
            output_schema:
              workspace: string
              hint: string
              status: string
            output_files:
              hint: hint.txt
            done_outputs:
              status: status
        dag:
          nodes:
            - id: rounds
              kind: repeat
              max_iterations: 2
              condition: {python: 'iteration < max_iterations'}
              initial_state: {memory: ''}
              body:
                nodes:
                  - id: hint
                    kind: agent
                    agent: proofstack.agents.configurable_cli.ConfigurableCLIAgent
                    name: cfg_hint
                    inputs:
                      problem: $input.problem
                state_updates:
                  memory: $node.hint.hint
          outputs:
            memory: $state.memory
        """
    )

    def test_set_executor_swaps_cli_component_to_codex(self) -> None:
        raw_yaml = _mutate(
            self.EXECUTOR_FIXTURE,
            {"op": "set_executor", "name": "cfg_hint", "executor": "codex_cli"},
        )
        raw = _raw(raw_yaml)
        cfg = raw["components"]["cfg_hint"]

        self.assertEqual(cfg["cmd"][:2], ["codex", "exec"])
        self.assertIs(cfg["copy_codex_auth"], True)
        self.assertEqual(cfg["contract"], "auto")
        self.assertNotIn("env", cfg)   # codex auth travels via CODEX_HOME, not HOME
        self.assertNotIn("usage", cfg)
        # Task identity survives the swap.
        self.assertIn("You are the specialist.", cfg["prompt"])
        self.assertEqual(cfg["output_files"], {"hint": "hint.txt"})
        node = raw["dag"]["nodes"][0]["body"]["nodes"][0]
        self.assertEqual(
            node["agent"], "proofstack.agents.configurable_cli.ConfigurableCLIAgent"
        )

    def test_set_executor_swaps_cli_component_to_human(self) -> None:
        raw_yaml = _mutate(
            self.EXECUTOR_FIXTURE,
            {"op": "set_executor", "name": "cfg_hint", "executor": "human"},
        )
        raw = _raw(raw_yaml)
        cfg = raw["components"]["cfg_hint"]

        for key in ("cmd", "env", "usage", "sandbox", "contract", "cache_enabled"):
            self.assertNotIn(key, cfg)
        self.assertIn("You are the specialist.", cfg["prompt"])
        # workspace is CLI mechanics; the human answers via the web form.
        self.assertNotIn("workspace", cfg["input_schema"])
        self.assertNotIn("workspace", cfg["output_schema"])
        self.assertIn("hint", cfg["output_schema"])
        node = raw["dag"]["nodes"][0]["body"]["nodes"][0]
        self.assertEqual(node["agent"], "proofstack.agents.human_agent.HumanAgent")

    def test_set_executor_swaps_cli_component_to_api_and_back(self) -> None:
        raw_yaml = _mutate(
            self.EXECUTOR_FIXTURE,
            {"op": "set_executor", "name": "cfg_hint", "executor": "api"},
        )
        raw = _raw(raw_yaml)
        cfg = raw["components"]["cfg_hint"]

        self.assertEqual(cfg["model"], "models/anthropic/sonnet_46")
        # CLI prompt becomes the API user prompt; single text output maps to
        # default_field so the whole response is the hint.
        self.assertIn("You are the specialist.", cfg["user_prompt"])
        self.assertNotIn("prompt", cfg)
        self.assertEqual(cfg["output"], {"default_field": "hint"})
        node = _raw(raw_yaml)["dag"]["nodes"][0]["body"]["nodes"][0]
        self.assertEqual(
            node["agent"],
            "proofstack.agents.configurable_prompt.ConfigurablePromptAgent",
        )

        back = _raw(
            _mutate(
                raw_yaml,
                {"op": "set_executor", "name": "cfg_hint", "executor": "claude_cli"},
            )
        )
        cfg2 = back["components"]["cfg_hint"]
        self.assertEqual(cfg2["cmd"][0], "claude")
        self.assertEqual(cfg2["contract"], "auto")
        self.assertIn("You are the specialist.", cfg2["prompt"])
        self.assertNotIn("user_prompt", cfg2)
        self.assertEqual(cfg2["output_files"], {"hint": "hint.txt"})
        self.assertEqual(cfg2["output_schema"].get("workspace"), "string")

    def test_set_executor_binds_model_to_declared_workflow_knob(self) -> None:
        # A tiered preset declares model knobs; scaffolds must bind to them so a
        # dropdown round-trip does not silently pin a node to a literal model.
        fixture = self.EXECUTOR_FIXTURE.replace(
            "inputs:\n  problem: ''",
            "inputs:\n  problem: ''\n  base_model: sonnet\n  gpt_model: gpt-5.4-mini",
        )

        as_codex = _mutate(
            fixture, {"op": "set_executor", "name": "cfg_hint", "executor": "codex_cli"}
        )
        codex_cfg = _raw(as_codex)["components"]["cfg_hint"]
        self.assertIn("-m", codex_cfg["cmd"])
        self.assertEqual(codex_cfg["cmd"][codex_cfg["cmd"].index("-m") + 1], "{gpt_model}")
        self.assertNotIn("model", codex_cfg)

        back = _mutate(
            as_codex, {"op": "set_executor", "name": "cfg_hint", "executor": "claude_cli"}
        )
        claude_cfg = _raw(back)["components"]["cfg_hint"]
        idx = claude_cfg["cmd"].index("--model")
        self.assertEqual(claude_cfg["cmd"][idx + 1], "{base_model}")

    def test_set_executor_falls_back_to_literal_model_without_knobs(self) -> None:
        as_claude = _mutate(
            _mutate(
                self.EXECUTOR_FIXTURE,
                {"op": "set_executor", "name": "cfg_hint", "executor": "codex_cli"},
            ),
            {"op": "set_executor", "name": "cfg_hint", "executor": "claude_cli"},
        )
        cfg = _raw(as_claude)["components"]["cfg_hint"]
        self.assertEqual(cfg["cmd"][cfg["cmd"].index("--model") + 1], "sonnet")

    def test_rename_cli_file_output_updates_existing_refs(self) -> None:
        raw_yaml = _mutate(
            textwrap.dedent(
                """
                workflow: proofstack.agents.dag_workflow.DAGWorkflow
                components:
                  cfg_cli:
                    cmd: [sh, -c, "finish '{\\"status\\":\\"done\\"}'"]
                    output_schema:
                      workspace: string
                      answer_tex: string
                    output_files:
                      answer_tex: answer.tex
                  cfg_consumer:
                    user_prompt: "{solution}"
                    input_schema:
                      solution: string
                    output:
                      xml_tags: [result]
                      default_field: result
                dag:
                  nodes:
                    - id: cli
                      kind: agent
                      agent: proofstack.agents.configurable_cli.ConfigurableCLIAgent
                      name: cfg_cli
                    - id: consumer
                      kind: agent
                      needs: [cli]
                      agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                      name: cfg_consumer
                      inputs:
                        solution: $node.cli.answer_tex
                  outputs:
                    solution: $node.cli.answer_tex
                """
            ),
            {
                "op": "update_component",
                "name": "cfg_cli",
                "fields": {
                    "__rename_output_refs": {"answer_tex": "solution_tex"},
                    "output_schema": {"workspace": "string", "solution_tex": "string"},
                    "output_files": {"solution_tex": "solution.tex"},
                },
            },
        )
        raw = _raw(raw_yaml)
        consumer = _node(raw_yaml, "consumer")

        self.assertEqual(raw["dag"]["outputs"]["solution"], "$node.cli.solution_tex")
        self.assertEqual(consumer["inputs"]["solution"], "$node.cli.solution_tex")
        self.assertEqual(raw["components"]["cfg_cli"]["output_files"], {"solution_tex": "solution.tex"})
        self.assertNotIn("answer_tex", raw["components"]["cfg_cli"]["output_schema"])
        self.assertTrue(validate_preset_yaml(raw_yaml)["ok"])

    def test_add_latex_cli_agent_disables_docker_no_new_privileges(self) -> None:
        raw_yaml = _mutate(
            textwrap.dedent(
                """
                workflow: proofstack.agents.dag_workflow.DAGWorkflow
                dag:
                  nodes: []
                  outputs: {}
                """
            ),
            {
                "op": "add_node",
                "template": "latex",
                "node_id": "compile",
                "x": 0,
                "y": 0,
            },
        )
        raw = _raw(raw_yaml)

        self.assertIs(raw["components"]["cfg_compile"]["sandbox"]["docker_no_new_privileges"], False)
        self.assertEqual(raw["components"]["cfg_compile"]["cmd"][1], "-c")

    def test_validate_report_keeps_budget_out_of_workflow_inputs(self) -> None:
        report = validate_preset_yaml(EDITOR_FIXTURE)

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["workflow_budget"], {"max_usd": 12, "max_wallclock_s": 300})
        self.assertIn("custom", report["workflow_inputs"])
        self.assertNotIn("max_usd", report["workflow_inputs"])
        self.assertNotIn("max_wallclock_s", report["workflow_input_schema"])

    def test_update_workflow_inputs_filters_budget_names(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE,
            {
                "op": "update_workflow_inputs",
                "inputs": [
                    {"field": "problem", "value": "New problem"},
                    {"field": "max_usd", "value": "999"},
                    {"field": "custom", "value": "kept"},
                ],
            },
        )

        self.assertEqual(_raw(raw_yaml)["inputs"], {"problem": "New problem", "custom": "kept"})

    def test_update_workflow_budget_can_remove_budget_mapping(self) -> None:
        raw_yaml = _mutate(EDITOR_FIXTURE, {"op": "update_workflow_budget", "budget": []})

        self.assertNotIn("budget", _raw(raw_yaml))
        report = validate_preset_yaml(raw_yaml)
        self.assertIn("max_usd", report["workflow_inputs"])

    def test_update_node_inputs_adds_and_prunes_managed_needs(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE,
            {
                "op": "update_node_inputs",
                "node_id": "consumer",
                "inputs": [{"field": "value", "value": "literal"}],
            },
        )
        consumer = _node(raw_yaml, "consumer")
        self.assertNotIn("needs", consumer)
        self.assertNotIn("managed_needs", consumer.get("ui", {}))

        raw_yaml = _mutate(
            raw_yaml,
            {
                "op": "update_node_inputs",
                "node_id": "consumer",
                "inputs": [{"field": "value", "value": "$node.source.text"}],
            },
        )
        consumer = _node(raw_yaml, "consumer")
        self.assertEqual(consumer["needs"], ["source"])
        self.assertEqual(consumer["ui"]["managed_needs"], ["source"])

    def test_copy_nodes_rewrites_internal_edges_to_the_copied_nodes(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE,
            {
                "op": "copy_nodes",
                "nodes": [
                    {"node_id": "consumer", "x": 640, "y": 100, "label": "Consumer copy"},
                    {"node_id": "source", "x": 300, "y": 100, "label": "Source copy"},
                ],
            },
        )

        source_copy = _node(raw_yaml, "source_copy")
        consumer_copy = _node(raw_yaml, "consumer_copy")

        self.assertEqual(source_copy["ui"]["x"], 300)
        self.assertEqual(source_copy["ui"]["label"], "source")
        self.assertEqual(consumer_copy["ui"]["label"], "consumer")
        self.assertEqual(consumer_copy["inputs"]["value"], "$node.source_copy.text")
        self.assertEqual(consumer_copy["needs"], ["source_copy"])
        self.assertEqual(consumer_copy["ui"]["managed_needs"], ["source_copy"])
        self.assertTrue(validate_preset_yaml(raw_yaml)["ok"])

    def test_copy_node_preserves_visible_title_without_copy_suffix(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE,
            {
                "op": "update_node",
                "node_id": "consumer",
                "fields": {"label": "Verifier"},
            },
        )
        raw_yaml = _mutate(
            raw_yaml,
            {
                "op": "copy_node",
                "node_id": "consumer",
                "x": 640,
                "y": 100,
                "label": "Verifier copy",
            },
        )
        copied = _node(raw_yaml, "consumer_copy")

        self.assertEqual(copied["ui"]["label"], "Verifier")
        self.assertNotIn("copy", copied["ui"]["label"].lower())

    def test_untie_component_copies_shared_prompt_for_current_node(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE,
            {
                "op": "copy_node",
                "node_id": "consumer",
                "x": 640,
                "y": 100,
            },
        )
        self.assertEqual(_node(raw_yaml, "consumer")["name"], "cfg_consumer")
        self.assertEqual(_node(raw_yaml, "consumer_copy")["name"], "cfg_consumer")

        raw_yaml = _mutate(
            raw_yaml,
            {
                "op": "untie_component",
                "node_id": "consumer_copy",
            },
        )
        raw = _raw(raw_yaml)
        consumer = _node(raw_yaml, "consumer")
        copied = _node(raw_yaml, "consumer_copy")

        self.assertEqual(consumer["name"], "cfg_consumer")
        self.assertEqual(copied["name"], "cfg_consumer_copy")
        self.assertEqual(raw["components"]["cfg_consumer_copy"], raw["components"]["cfg_consumer"])
        self.assertTrue(validate_preset_yaml(raw_yaml)["ok"])

    def test_copy_nodes_keeps_repeat_body_node_refs_internal(self) -> None:
        raw_yaml = _mutate(
            REPEAT_FIXTURE,
            {
                "op": "copy_nodes",
                "nodes": [
                    {"node_id": "seed", "x": 300, "y": 100},
                    {"node_id": "loop", "x": 660, "y": 100},
                ],
            },
        )

        loop_copy = _node(raw_yaml, "loop_copy")
        improver = loop_copy["body"]["nodes"][1]

        self.assertEqual(loop_copy["initial_state"]["solution"], "$node.seed_copy.solution")
        self.assertEqual(loop_copy["needs"], ["seed_copy"])
        self.assertEqual(improver["when"]["ref"], "$node.verifier.gap")
        self.assertEqual(improver["inputs"]["value"], "$node.verifier.gap")
        self.assertTrue(validate_preset_yaml(raw_yaml)["ok"])

    def test_renaming_node_rewrites_all_node_references(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE,
            {
                "op": "update_node",
                "node_id": "source",
                "fields": {"id": "source renamed"},
            },
        )

        self.assertEqual(_node(raw_yaml, "source_renamed")["id"], "source_renamed")
        self.assertEqual(_node(raw_yaml, "consumer")["inputs"]["value"], "$node.source_renamed.text")

    def test_tie_component_removes_unused_old_prompt_but_not_shared_prompt(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE,
            {
                "op": "tie_component",
                "node_id": "consumer",
                "target_name": "cfg_source",
            },
        )
        raw = _raw(raw_yaml)

        self.assertEqual(_node(raw_yaml, "consumer")["name"], "cfg_source")
        self.assertIn("cfg_source", raw["components"])
        self.assertNotIn("cfg_consumer", raw["components"])
        self.assertIn("cfg_unused", raw["components"])

    def test_workflow_output_rename_rejects_collisions_and_preserves_order(self) -> None:
        with_extra = _mutate(
            EDITOR_FIXTURE,
            {"op": "add_workflow_output", "field": "summary", "value": "$node.consumer.text"},
        )
        collision = mutate_preset_yaml(
            with_extra,
            {
                "op": "update_workflow_output",
                "field": "summary",
                "new_field": "solution",
                "value": "$node.consumer.text",
            },
        )
        self.assertFalse(collision["ok"])
        self.assertIn("workflow output already exists", collision["errors"][0])

        renamed = _mutate(
            with_extra,
            {
                "op": "update_workflow_output",
                "field": "summary",
                "new_field": "final_summary",
                "value": "$node.consumer.text",
            },
        )
        self.assertEqual(list(_raw(renamed)["dag"]["outputs"]), ["solution", "final_summary"])

    def test_disconnect_workflow_output_clears_source_without_deleting_socket(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE,
            {
                "op": "disconnect_edge",
                "target_node": "__workflow_outputs",
                "target_field": "solution",
            },
        )
        outputs = _raw(raw_yaml)["dag"]["outputs"]

        self.assertIn("solution", outputs)
        self.assertEqual(outputs["solution"], "")
        self.assertTrue(validate_preset_yaml(raw_yaml)["ok"])

        raw_yaml = _mutate(
            raw_yaml,
            {
                "op": "connect_edge",
                "source_node": "consumer",
                "source_field": "solution",
                "target_node": "__workflow_outputs",
                "target_field": "solution",
            },
        )
        self.assertEqual(_raw(raw_yaml)["dag"]["outputs"]["solution"], "$node.consumer.solution")

        raw_yaml = _mutate(
            EDITOR_FIXTURE,
            {
                "op": "connect_edge",
                "source_node": "source",
                "source_field": "text",
                "target_node": "__workflow_outputs",
                "target_field": "solution",
            },
        )
        raw_yaml = _mutate(raw_yaml, {"op": "delete_node", "node_id": "consumer"})
        self.assertEqual(_raw(raw_yaml)["dag"]["outputs"]["solution"], "$node.source.text")

    def test_workflow_output_accepts_multiple_mutually_exclusive_sources(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE,
            {
                "op": "connect_edge",
                "source_node": "source",
                "source_field": "text",
                "target_node": "__workflow_outputs",
                "target_field": "solution",
            },
        )

        self.assertEqual(
            _raw(raw_yaml)["dag"]["outputs"]["solution"],
            {"coalesce": ["$node.consumer.solution", "$node.source.text"]},
        )
        self.assertTrue(validate_preset_yaml(raw_yaml)["ok"])

        raw_yaml = _mutate(
            raw_yaml,
            {
                "op": "disconnect_edge",
                "source_node": "source",
                "source_field": "text",
                "target_node": "__workflow_outputs",
                "target_field": "solution",
            },
        )
        self.assertEqual(_raw(raw_yaml)["dag"]["outputs"]["solution"], "$node.consumer.solution")

    def test_regular_node_input_accepts_multiple_mutually_exclusive_sources(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE_WITH_EXTRA_SOURCE,
            {
                "op": "connect_edge",
                "source_node": "extra",
                "source_field": "solution",
                "target_node": "consumer",
                "target_field": "value",
            },
        )
        consumer = _node(raw_yaml, "consumer")

        self.assertEqual(
            consumer["inputs"]["value"],
            {"coalesce": ["$node.source.text", "$node.extra.solution"]},
        )
        self.assertEqual(consumer["needs"], ["source", "extra"])
        report = validate_preset_yaml(raw_yaml)
        self.assertTrue(report["ok"], report["errors"])
        target_paths = sorted(
            edge["target_path"]
            for edge in report["edges"]
            if edge["target"] == "consumer" and edge["target_path"].startswith("inputs.value")
        )
        self.assertEqual(target_paths, ["inputs.value.coalesce.0", "inputs.value.coalesce.1"])

    def test_disconnect_one_regular_node_input_source_keeps_other_source(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE_WITH_EXTRA_SOURCE,
            {
                "op": "connect_edge",
                "source_node": "extra",
                "source_field": "solution",
                "target_node": "consumer",
                "target_field": "value",
            },
        )
        raw_yaml = _mutate(
            raw_yaml,
            {
                "op": "disconnect_edge",
                "source_node": "extra",
                "source_field": "solution",
                "target_node": "consumer",
                "target_field": "value",
            },
        )
        consumer = _node(raw_yaml, "consumer")

        self.assertEqual(consumer["inputs"]["value"], "$node.source.text")
        self.assertEqual(consumer["needs"], ["source"])
        self.assertTrue(validate_preset_yaml(raw_yaml)["ok"])

    def test_delete_node_clears_workflow_output_source_without_deleting_output(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE,
            {
                "op": "delete_node",
                "node_id": "consumer",
            },
        )
        outputs = _raw(raw_yaml)["dag"]["outputs"]

        self.assertIn("solution", outputs)
        self.assertEqual(outputs["solution"], "")
        self.assertTrue(validate_preset_yaml(raw_yaml)["ok"])

    def test_disconnect_node_input_keeps_input_as_matching_workflow_input(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE,
            {
                "op": "disconnect_edge",
                "target_node": "consumer",
                "target_field": "value",
            },
        )
        consumer = _node(raw_yaml, "consumer")

        self.assertEqual(consumer["inputs"], {"value": "$input.value"})
        self.assertNotIn("needs", consumer)
        self.assertTrue(validate_preset_yaml(raw_yaml)["ok"])

    def test_budget_fallback_available_outputs_accepts_multiple_sources(self) -> None:
        raw_yaml = _mutate(
            BUDGET_FALLBACK_FIXTURE,
            {
                "op": "connect_edge",
                "source_node": "source",
                "source_field": "solution",
                "target_node": "budget_fallback",
                "target_field": "available_outputs",
            },
        )
        raw_yaml = _mutate(
            raw_yaml,
            {
                "op": "connect_edge",
                "source_node": "extra",
                "source_field": "solution",
                "target_node": "budget_fallback",
                "target_field": "available_outputs",
            },
        )
        inputs = _node(raw_yaml, "budget_fallback")["inputs"]

        self.assertEqual(
            inputs["available_outputs"],
            {
                "source.solution": "$node.source.solution",
                "extra.solution": "$node.extra.solution",
            },
        )
        report = validate_preset_yaml(raw_yaml)
        self.assertTrue(report["ok"], report["errors"])
        target_paths = sorted(
            edge["target_path"]
            for edge in report["edges"]
            if edge["target"] == "budget_fallback" and edge["target_path"].startswith("inputs.available_outputs")
        )
        self.assertEqual(
            target_paths,
            [
                "inputs.available_outputs.extra.solution",
                "inputs.available_outputs.source.solution",
            ],
        )

    def test_disconnect_budget_fallback_available_outputs_keeps_socket(self) -> None:
        raw_yaml = _mutate(
            BUDGET_FALLBACK_FIXTURE,
            {
                "op": "connect_edge",
                "source_node": "source",
                "source_field": "solution",
                "target_node": "budget_fallback",
                "target_field": "available_outputs",
            },
        )
        raw_yaml = _mutate(
            raw_yaml,
            {
                "op": "disconnect_edge",
                "source_node": "source",
                "source_field": "solution",
                "target_node": "budget_fallback",
                "target_field": "available_outputs",
            },
        )
        inputs = _node(raw_yaml, "budget_fallback")["inputs"]
        report = validate_preset_yaml(raw_yaml)
        budget_node = next(node for node in report["nodes"] if node["id"] == "budget_fallback")

        self.assertEqual(inputs, {"available_outputs": {}})
        self.assertIn("available_outputs", budget_node["input_fields"])
        self.assertTrue(report["ok"], report["errors"])

    def test_create_repeat_zone_allows_same_start_and_end_node(self) -> None:
        raw_yaml = _mutate(
            textwrap.dedent(
                """
                workflow: proofstack.agents.dag_workflow.DAGWorkflow
                components:
                  cfg_single:
                    user_prompt: Verify and improve once.
                    output:
                      xml_tags: [solution]
                      default_field: solution
                dag:
                  nodes:
                    - id: verifier
                      kind: agent
                      agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                      name: cfg_single
                      ui:
                        x: 120
                        y: 220
                  outputs:
                    solution: $node.verifier.solution
                """
            ),
            {
                "op": "create_repeat_zone_from_path",
                "start_node_id": "verifier",
                "end_node_id": "verifier",
            },
        )
        dag = _raw(raw_yaml)["dag"]
        loop = dag["nodes"][0]

        self.assertEqual(len(dag["nodes"]), 1)
        self.assertEqual(loop["kind"], "repeat")
        self.assertEqual(loop["body"]["nodes"][0]["id"], "verifier")
        self.assertEqual(loop["outputs"], {"solution": "$state.solution"})
        self.assertEqual(loop["body"]["state_updates"], {"solution": "$node.verifier.solution"})
        self.assertNotIn("verifier", loop["body"]["state_updates"])
        self.assertNotIn("verifier_solution", loop["body"]["state_updates"])
        self.assertEqual(dag["outputs"], {"solution": "$node.repeat_verifier_to_verifier.solution"})
        report = validate_preset_yaml(raw_yaml)
        self.assertTrue(report["ok"], report["errors"])

    def test_create_repeat_zone_preserves_state_inputs_without_internal_aliases(self) -> None:
        raw_yaml = _mutate(
            textwrap.dedent(
                """
                workflow: proofstack.agents.dag_workflow.DAGWorkflow
                components:
                  cfg_step:
                    user_prompt: Improve {solution}.
                    input_schema:
                      solution: string
                    output:
                      xml_tags: [solution]
                      default_field: solution
                dag:
                  nodes:
                    - id: improve
                      kind: agent
                      agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                      name: cfg_step
                      inputs:
                        solution: $state.solution
                  outputs:
                    solution: $node.improve.solution
                """
            ),
            {
                "op": "create_repeat_zone_from_path",
                "start_node_id": "improve",
                "end_node_id": "improve",
            },
        )
        loop = _raw(raw_yaml)["dag"]["nodes"][0]

        self.assertEqual(loop["initial_state"], {"solution": "$state.solution"})
        self.assertEqual(loop["body"]["nodes"][0]["inputs"], {"solution": "$state.solution"})
        self.assertEqual(loop["body"]["state_updates"], {"solution": "$node.improve.solution"})
        self.assertEqual(loop["outputs"], {"solution": "$state.solution"})

    def test_create_repeat_zone_does_not_rewrite_existing_repeat_body_local_refs(self) -> None:
        raw_yaml = _mutate(
            textwrap.dedent(
                """
                workflow: proofstack.agents.dag_workflow.DAGWorkflow
                components:
                  cfg_step:
                    user_prompt: Step.
                    output:
                      xml_tags: [solution]
                      default_field: solution
                dag:
                  nodes:
                    - id: existing_loop
                      kind: repeat
                      max_iterations: 2
                      condition:
                        python: iteration < max_iterations
                      initial_state:
                        solution: seed
                      body:
                        nodes:
                          - id: verify_improve
                            kind: agent
                            agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                            name: cfg_step
                        state_updates:
                          solution: $node.verify_improve.solution
                      outputs:
                        solution: $state.solution
                    - id: verify_improve
                      kind: agent
                      agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
                      name: cfg_step
                  outputs:
                    solution: $node.verify_improve.solution
                """
            ),
            {
                "op": "create_repeat_zone_from_path",
                "start_node_id": "verify_improve",
                "end_node_id": "verify_improve",
            },
        )
        raw = _raw(raw_yaml)
        existing_loop = raw["dag"]["nodes"][0]
        new_loop = raw["dag"]["nodes"][1]

        self.assertEqual(existing_loop["body"]["state_updates"], {"solution": "$node.verify_improve.solution"})
        self.assertEqual(new_loop["id"], "repeat_verify_improve_to_verify_improve")
        self.assertEqual(raw["dag"]["outputs"], {"solution": "$node.repeat_verify_improve_to_verify_improve.solution"})
        self.assertTrue(validate_preset_yaml(raw_yaml)["ok"])

    def test_if_else_branch_outputs_are_editable_and_reported(self) -> None:
        raw_yaml = _mutate(
            textwrap.dedent(
                """
                workflow: proofstack.agents.dag_workflow.DAGWorkflow
                dag:
                  nodes:
                    - id: router
                      kind: if_else
                      condition: true
                      then:
                        accepted: true
                      else:
                        retry: true
                  outputs:
                    accepted: $node.router.accepted
                """
            ),
            {
                "op": "update_node_outputs",
                "node_id": "router",
                "then_field": "accepted",
                "else_field": "retry",
            },
        )
        router = _node(raw_yaml, "router")
        report = validate_preset_yaml(raw_yaml)
        reported_router = next(node for node in report["nodes"] if node["id"] == "router")

        self.assertEqual(router["then"], {"accepted": True})
        self.assertEqual(router["else"], {"retry": True})
        self.assertNotIn("condition", router.get("output_schema", {}))
        self.assertEqual(reported_router["output_fields"], ["accepted", "condition", "retry"])
        self.assertEqual(reported_router["outputs_schema"]["condition"]["type"], "boolean")
        self.assertNotIn("branch", reported_router["outputs_schema"])

    def test_repeat_body_if_else_branch_outputs_are_editable(self) -> None:
        raw_yaml = _mutate(
            REPEAT_FIXTURE,
            {
                "op": "update_node_outputs",
                "node_id": "loop::body::verifier",
                "then_field": "ok",
                "else_field": "gap",
            },
        )
        verifier = _raw(raw_yaml)["dag"]["nodes"][1]["body"]["nodes"][0]

        self.assertEqual(verifier["then"], {"ok": True})
        self.assertEqual(verifier["else"], {"gap": True})

    def test_update_repeat_body_node_text_uses_visual_editor_id(self) -> None:
        raw_yaml = _mutate(
            REPEAT_FIXTURE,
            {
                "op": "update_node",
                "node_id": "loop::body::verifier",
                "fields": {
                    "label": "Verify draft",
                    "subtitle": "Checks the current draft.",
                },
            },
        )
        verifier = _raw(raw_yaml)["dag"]["nodes"][1]["body"]["nodes"][0]

        self.assertEqual(verifier["ui"]["label"], "Verify draft")
        self.assertEqual(verifier["ui"]["subtitle"], "Checks the current draft.")

    def test_update_when_supports_equals_and_invalid_length_limits_fail(self) -> None:
        raw_yaml = _mutate(
            EDITOR_FIXTURE,
            {
                "op": "update_node_when",
                "node_id": "consumer",
                "mode": "equals",
                "body": "$input.mode",
                "compare_value": "fast",
            },
        )
        self.assertEqual(_node(raw_yaml, "consumer")["when"], {"ref": "$input.mode", "equals": "fast"})

        result = mutate_preset_yaml(
            EDITOR_FIXTURE,
            {
                "op": "update_node_when",
                "node_id": "consumer",
                "mode": "ref",
                "body": "$node.source.text",
                "min_len": "not a number",
            },
        )
        self.assertFalse(result["ok"])
        self.assertIn("Minimum items must be a number", result["errors"][0])

    def test_tool_save_rejects_spaces_and_invalid_descriptor_function_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid_yaml = textwrap.dedent(
                """
                name: valid_tool
                python_function: valid_tool
                description: ok
                parameters:
                  type: object
                """
            )
            tool = save_tool_definition("old", "valid_tool", valid_yaml, "def valid_tool():\n    return {}\n", root=root)
            self.assertEqual(tool.name, "valid_tool")
            self.assertTrue((root / "valid_tool.yaml").exists())

            with self.assertRaises(PresetError):
                save_tool_definition("valid_tool", "Bad Tool", valid_yaml, "", root=root)

            invalid_yaml = valid_yaml.replace("python_function: valid_tool", "python_function: bad tool")
            with self.assertRaises(PresetError):
                save_tool_definition("valid_tool", "valid_tool_2", invalid_yaml, "", root=root)

    def test_exported_presets_include_workflow_output_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "subagent.yaml").write_text(
                textwrap.dedent(
                    """
                    workflow: proofstack.agents.dag_workflow.DAGWorkflow
                    export:
                      visible_as_node: true
                      label: Test Subagent
                    inputs:
                      problem: ''
                      solution: ''
                    dag:
                      nodes: []
                      outputs:
                        solution: $input.solution
                        verdict: $input.verdict
                        verification: $input.verification
                    """
                ),
                encoding="utf-8",
            )

            presets = discover_exported_presets(root)

        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0]["name"], "subagent")
        self.assertEqual(presets[0]["inputs"], ["problem", "solution"])
        self.assertEqual(presets[0]["outputs"], ["solution", "verdict", "verification"])

    def test_author_critic_preset_validates_as_visual_dag_wrapper(self) -> None:
        raw_yaml = (ROOT / "configs" / "workflows" / "author_critic.yaml").read_text(encoding="utf-8")

        validation = validate_preset_yaml(raw_yaml)
        route_report = preset_dag_report("author_critic").to_dict()

        expected_nodes = ["problem", "author_critic_loop", "return"]
        expected_body = [
            "author",
            "stateful_critic",
            "llm_council",
            "compute_node",
            "fresh_critic",
            "review_join",
            "ready_gate",
            "compile_gate",
        ]
        for report in (validation, route_report):
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["errors"], [])
            self.assertEqual([node["id"] for node in report["nodes"]], expected_nodes)
            self.assertEqual(report["nodes"][0]["agent"], "proofstack.agents.ac.ACInitBlock")
            self.assertEqual(report["nodes"][1]["kind"], "repeat")
            body = report["nodes"][1]["config"]["body"]["nodes"]
            self.assertEqual([node["id"] for node in body], expected_body)
            body_by_id = {node["id"]: node for node in body}
            self.assertIn("answer_tex", body_by_id["author"]["output_fields"])
            self.assertIn("answer_ready", body_by_id["stateful_critic"]["output_fields"])
            self.assertIn("feedback", body_by_id["llm_council"]["output_fields"])
            self.assertIn("response_md", body_by_id["compute_node"]["output_fields"])
            self.assertIn("ready_for_gate", body_by_id["review_join"]["output_fields"])
            self.assertEqual(body_by_id["compile_gate"]["output_fields"], ["state"])
            self.assertEqual(report["nodes"][-1]["agent"], "proofstack.agents.ac.ACReturnBlock")
            self.assertIn("answer_tex", report["workflow_outputs"])
            self.assertTrue(report["edges"])
            self.assertTrue(all(edge["status"] == "ok" for edge in report["edges"]))
            self.assertIn("n_rounds", report["workflow_inputs"])
            self.assertIn("problem", report["workflow_input_schema"])

    def test_firstproof_submission_preset_is_app_runnable(self) -> None:
        raw_yaml = (ROOT / "configs" / "workflows" / "firstproof_submission.yaml").read_text(encoding="utf-8")

        validation = validate_preset_yaml(raw_yaml)
        route_report = preset_dag_report("firstproof_submission").to_dict()
        preset_names = {preset.name: preset for preset in discover_presets()}
        raw = yaml.safe_load(raw_yaml)

        self.assertIn("firstproof_submission", preset_names)
        self.assertEqual(preset_names["firstproof_submission"].label, "FirstProof Submission")
        self.assertEqual(raw["inputs"]["n_rounds"], 200)
        self.assertEqual(raw["budget"]["max_usd"], 1000.0)
        self.assertEqual(raw["inputs"]["page_limit"], 12)
        expected_nodes = ["problem", "author_critic_loop", "return"]
        expected_body = [
            "author",
            "stateful_critic",
            "llm_council",
            "compute_node",
            "fresh_critic",
            "review_join",
            "ready_gate",
            "compile_gate",
        ]
        for report in (validation, route_report):
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["errors"], [])
            self.assertEqual([node["id"] for node in report["nodes"]], expected_nodes)
            body = report["nodes"][1]["config"]["body"]["nodes"]
            self.assertEqual([node["id"] for node in body], expected_body)
            self.assertEqual(report["nodes"][-1]["agent"], "proofstack.agents.ac.ACReturnBlock")
            self.assertIn("answer_tex", report["workflow_outputs"])
            self.assertTrue(report["edges"])
            self.assertTrue(all(edge["status"] == "ok" for edge in report["edges"]))

    def test_author_critic_smoke_mini_preset_is_all_mini_and_app_runnable(self) -> None:
        raw_yaml = (ROOT / "configs" / "workflows" / "author_critic_smoke_mini.yaml").read_text(encoding="utf-8")

        validation = validate_preset_yaml(raw_yaml)
        route_report = preset_dag_report("author_critic_smoke_mini").to_dict()
        preset_names = {preset.name: preset for preset in discover_presets()}
        raw = yaml.safe_load(raw_yaml)

        self.assertIn("author_critic_smoke_mini", preset_names)
        self.assertEqual(preset_names["author_critic_smoke_mini"].label, "Author Critic Smoke Mini")
        self.assertEqual(raw["components"]["Author"]["model"], "models/openai/gpt-54-mini")
        self.assertNotIn("USE_CONTAINER_FILES", raw["components"]["Author"])
        self.assertEqual(raw["components"]["ACCritic"]["model"], "models/openai/gpt-54-mini")
        self.assertEqual(raw["inputs"]["full_critic_interval"], 3)
        self.assertEqual(raw["inputs"]["council_models"], ["models/openai/gpt-54-mini"] * 3)
        self.assertEqual(raw["inputs"]["compute_model"], "gpt-5.4-mini")
        self.assertEqual(raw["inputs"]["compute_cost_config"], "models/openai/gpt-54-mini")
        for report in (validation, route_report):
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["errors"], [])
            self.assertEqual([node["id"] for node in report["nodes"]], ["problem", "author_critic_loop", "return"])
            body = report["nodes"][1]["config"]["body"]["nodes"]
            self.assertEqual(
                [node["id"] for node in body],
                [
                    "author",
                    "stateful_critic",
                    "llm_council",
                    "compute_node",
                    "fresh_critic",
                    "review_join",
                    "ready_gate",
                    "compile_gate",
                ],
            )
            self.assertTrue(report["edges"])
            self.assertTrue(all(edge["status"] == "ok" for edge in report["edges"]))

    def test_repeat_internal_edges_are_mutable_from_visual_node_ids(self) -> None:
        raw_yaml = _mutate(
            REPEAT_FIXTURE,
            {
                "op": "disconnect_edge",
                "target_node": "loop::body::improver",
                "target_field": "value",
            },
        )
        improver = _raw(raw_yaml)["dag"]["nodes"][1]["body"]["nodes"][1]
        self.assertEqual(improver["inputs"], {"value": "$input.value"})
        self.assertEqual(improver["needs"], ["verifier"])

        raw_yaml = _mutate(
            raw_yaml,
            {
                "op": "disconnect_edge",
                "target_node": "loop::body::improver",
                "target_field": "__condition",
            },
        )
        improver = _raw(raw_yaml)["dag"]["nodes"][1]["body"]["nodes"][1]
        self.assertNotIn("when", improver)
        self.assertNotIn("needs", improver)

        raw_yaml = _mutate(
            raw_yaml,
            {
                "op": "connect_edge",
                "source_node": "loop::repeat_input",
                "source_field": "solution",
                "target_node": "loop::body::improver",
                "target_field": "value",
            },
        )
        improver = _raw(raw_yaml)["dag"]["nodes"][1]["body"]["nodes"][1]
        self.assertEqual(improver["inputs"]["value"], "$state.solution")

        raw_yaml = _mutate(
            raw_yaml,
            {
                "op": "reconnect_edge",
                "source_node": "loop::body::verifier",
                "source_field": "gap",
                "old_target_node": "loop::repeat_output",
                "old_target_field": "solution",
                "target_node": "loop::repeat_output",
                "target_field": "solution",
            },
        )
        updates = _raw(raw_yaml)["dag"]["nodes"][1]["body"]["state_updates"]
        self.assertEqual(updates["solution"], "$node.verifier.gap")

        raw_yaml = _mutate(
            raw_yaml,
            {
                "op": "connect_edge",
                "source_node": "loop::repeat_output",
                "source_field": "solution",
                "target_node": "__workflow_outputs",
                "target_field": "summary",
            },
        )
        self.assertEqual(_raw(raw_yaml)["dag"]["outputs"]["summary"], "$node.loop.solution")

    def test_repeat_virtual_boundary_nodes_are_movable(self) -> None:
        raw_yaml = _mutate(
            REPEAT_FIXTURE,
            {
                "op": "move_repeat_virtual_node",
                "loop_id": "loop",
                "visual_type": "repeat_input",
                "x": 123,
                "y": 456,
            },
        )
        raw_yaml = _mutate(
            raw_yaml,
            {
                "op": "move_repeat_virtual_node",
                "loop_id": "loop",
                "visual_type": "repeat_output",
                "x": 789,
                "y": 321,
            },
        )
        loop = _raw(raw_yaml)["dag"]["nodes"][1]

        self.assertEqual(loop["ui"]["repeat_input"], {"x": 123, "y": 456})
        self.assertEqual(loop["ui"]["repeat_output"], {"x": 789, "y": 321})
        report = validate_preset_yaml(raw_yaml)
        self.assertTrue(report["ok"], report["errors"])

    def test_repeat_zone_visuals_move_as_group(self) -> None:
        raw_yaml = _mutate(
            REPEAT_FIXTURE,
            {
                "op": "move_repeat_zone_visuals",
                "loop_id": "loop",
                "repeat_input": {"x": 100, "y": 200},
                "repeat_output": {"x": 900, "y": 220},
                "body_nodes": [
                    {"body_node_id": "verifier", "x": 360, "y": 210},
                    {"body_node_id": "improver", "x": 620, "y": 240},
                ],
            },
        )
        loop = _raw(raw_yaml)["dag"]["nodes"][1]
        body_nodes = {node["id"]: node for node in loop["body"]["nodes"]}

        self.assertEqual(loop["ui"]["repeat_input"], {"x": 100, "y": 200})
        self.assertEqual(loop["ui"]["repeat_output"], {"x": 900, "y": 220})
        self.assertEqual(body_nodes["verifier"]["ui"], {"x": 360, "y": 210})
        self.assertEqual(body_nodes["improver"]["ui"], {"x": 620, "y": 240})
        report = validate_preset_yaml(raw_yaml)
        self.assertTrue(report["ok"], report["errors"])

    def test_deleting_budget_fallback_node_uses_regular_workflow_outputs(self) -> None:
        raw_yaml = _mutate(
            BUDGET_FALLBACK_FIXTURE,
            {"op": "delete_node", "node_id": "budget_fallback"},
        )
        raw = _raw(raw_yaml)

        self.assertEqual(raw["dag"]["outputs"], {"solution": "$node.source.solution"})
        self.assertTrue(validate_preset_yaml(raw_yaml)["ok"])

    def test_aletheia_repeat_uses_two_branch_conditions(self) -> None:
        raw_yaml = ALETHEIA_REPEAT_FIXTURE
        raw = _raw(raw_yaml)
        repeat = next(node for node in raw["dag"]["nodes"] if node["id"] == "aletheia_repeat")
        body_nodes = {node["id"]: node for node in repeat["body"]["nodes"]}

        self.assertIn("verifier_error", body_nodes)
        self.assertIn("verifier_complete", body_nodes)
        self.assertEqual(body_nodes["verifier_complete"]["when"], {"ref": "$node.verifier_error.not_error", "equals": True})
        self.assertEqual(body_nodes["verifier_complete"]["then"], {})
        self.assertEqual(body_nodes["restart_proof"]["when"], {"ref": "$node.verifier_error.error", "equals": True})
        self.assertEqual(body_nodes["improve_gap"]["when"], {"ref": "$node.verifier_complete.needs_improvement", "equals": True})
        self.assertNotIn("done", repeat["initial_state"])
        self.assertNotIn("done", repeat["body"]["state_updates"])
        self.assertEqual(raw["dag"]["outputs"], {"solution": "$node.aletheia_repeat.solution"})
        self.assertTrue(validate_preset_yaml(raw_yaml)["ok"])


API_OUTPUTS_FIXTURE = textwrap.dedent(
    """
    workflow: proofstack.agents.dag_workflow.DAGWorkflow
    inputs:
      problem: ''
    components:
      cfg_review:
        system_prompt: Review rigorously.
        user_prompt: 'Assess: {problem}'
        model: models/anthropic/sonnet_46
        output:
          xml_tags: [report, verdict, answer_tex]
          default_field: report
        input_schema:
          problem: string
        output_schema:
          report: string
          verdict: string
          answer_tex: string
    dag:
      nodes:
        - id: review
          kind: agent
          agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
          name: cfg_review
          inputs:
            problem: $input.problem
      outputs:
        verdict: $node.review.verdict
    """
)


class SetExecutorCoverageTests(unittest.TestCase):
    """Regressions from PR review: executor swaps must keep outputs producible
    and retarget every node shape that invokes the component."""

    def test_switch_api_to_cli_creates_output_files_for_business_outputs(self) -> None:
        raw_yaml = _mutate(
            API_OUTPUTS_FIXTURE,
            {"op": "set_executor", "name": "cfg_review", "executor": "claude_cli"},
        )
        cfg = _raw(raw_yaml)["components"]["cfg_review"]

        # every declared output the CLI can't report via done.json now has a file
        self.assertEqual(
            cfg["output_files"],
            {"report": "report.txt", "verdict": "verdict.txt", "answer_tex": "answer.tex"},
        )
        self.assertEqual(cfg["output_schema"]["workspace"], "string")
        self.assertEqual(cfg["output_schema"]["status"], "string")

    def test_switch_to_cli_regenerates_projections_from_the_contract(self) -> None:
        raw = _raw(API_OUTPUTS_FIXTURE)
        cfg = raw["components"]["cfg_review"]
        # pre-existing projections are absorbed into the contract, then
        # regenerated — stale or custom mappings are not merged back (that is
        # how renamed outputs used to resurrect)
        cfg["output_files"] = {"report": "report.md"}
        cfg["done_outputs"] = {"verdict": "summary"}
        _op_set_executor(raw, {"executor": "codex_cli", "name": "cfg_review"})

        self.assertEqual(
            cfg["output_files"],
            {"report": "report.txt", "verdict": "verdict.txt", "answer_tex": "answer.tex"},
        )
        self.assertEqual(cfg["done_outputs"], {"status": "status"})

    def test_switch_basic_io_component_keeps_its_only_output(self) -> None:
        # Basic I/O shape: output.default_field is the ONLY declaration of the
        # business output — no output_schema at all.
        raw = {
            "components": {
                "cfg_basic": {
                    "system_prompt": "Answer.",
                    "user_prompt": "Q: {question}",
                    "model": "models/anthropic/sonnet_46",
                    "output": {"default_field": "output"},
                    "input_schema": {"question": "string"},
                }
            },
            "dag": {"nodes": []},
        }
        cfg = raw["components"]["cfg_basic"]
        _op_set_executor(raw, {"executor": "codex_cli", "name": "cfg_basic"})

        self.assertEqual(cfg["output_schema"]["output"], "string")
        self.assertEqual(cfg["output_files"], {"output": "output.txt"})

        _op_set_executor(raw, {"executor": "human", "name": "cfg_basic"})
        self.assertIn("output", cfg["output_schema"])

        _op_set_executor(raw, {"executor": "api", "name": "cfg_basic"})
        self.assertEqual(cfg["output"], {"default_field": "output"})

    def test_switch_mixed_cli_outputs_to_api_requests_all_fields(self) -> None:
        # CLI component sourcing outputs from BOTH output_files and
        # done_outputs: the API extraction spec must request every field, not
        # just the file-backed ones.
        raw = {
            "components": {
                "cfg_mixed": {
                    "prompt": "Review and report.",
                    "input_schema": {"solution": "string", "workspace": "string"},
                    "output_schema": {
                        "workspace": "string",
                        "status": "string",
                        "notes": "string",
                        "verdict": "string",
                    },
                    "output_files": {"notes": "notes.md"},
                    "done_outputs": {"verdict": "summary", "status": "status"},
                }
            },
            "dag": {"nodes": []},
        }
        cfg = raw["components"]["cfg_mixed"]
        _op_set_executor(raw, {"executor": "api", "name": "cfg_mixed"})

        self.assertEqual(
            cfg["output"], {"xml_tags": ["notes", "verdict"], "default_field": "notes"}
        )
        # CLI mechanics must not be advertised as API outputs/inputs: the API
        # parser never emits workspace or status
        self.assertNotIn("workspace", cfg["output_schema"])
        self.assertNotIn("status", cfg["output_schema"])
        self.assertNotIn("workspace", cfg["input_schema"])
        self.assertEqual(cfg["output_schema"], {"notes": "string", "verdict": "string"})

    def test_set_executor_retargets_join_or_agent_and_map_chain_steps(self) -> None:
        raw = {
            "components": {
                "cfg_judge": {
                    "prompt": "judge",
                    "input_schema": {"source": "list"},
                    "output_schema": {"verdict": "string"},
                }
            },
            "dag": {
                "nodes": [
                    {
                        "id": "pick",
                        "kind": "join_or_agent",
                        "agent": "proofstack.agents.configurable_prompt.ConfigurablePromptAgent",
                        "name": "cfg_judge",
                        "source": "$node.fan.finals",
                    },
                    {
                        "id": "fan",
                        "kind": "map_chain",
                        "foreach": "$input.problems",
                        "steps": [
                            {
                                "id": "s1",
                                "agent": "proofstack.agents.configurable_prompt.ConfigurablePromptAgent",
                                "name": "cfg_judge",
                                "inputs": {},
                            }
                        ],
                    },
                ]
            },
        }
        _op_set_executor(raw, {"executor": "codex_cli", "name": "cfg_judge"})

        nodes = raw["dag"]["nodes"]
        self.assertEqual(nodes[0]["agent"], CONFIGURABLE_CLI_AGENT)
        self.assertEqual(nodes[1]["steps"][0]["agent"], CONFIGURABLE_CLI_AGENT)


if __name__ == "__main__":
    unittest.main()
