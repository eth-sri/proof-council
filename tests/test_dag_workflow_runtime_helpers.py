from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.dag_workflow import (  # noqa: E402
    DAGWorkflow,
    _agent_inputs_with_workflow_defaults,
    _condition,
    _eval_value,
    _node_condition_scope,
    _resolve_path,
)
from proofstack.context import RunContext  # noqa: E402


class _Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.items.append((name, payload))


class _FlexibleAgent:
    name = "cfg_prompt"
    component_config = {}

    class Inputs(BaseModel):
        model_config = ConfigDict(extra="allow")

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True}


class _EchoAgent:
    name = "echo"
    component_config = {}

    class Inputs(BaseModel):
        model_config = ConfigDict(extra="allow")

    def __init__(self, ctx=None, *, name=None) -> None:
        self.name = name or "echo"

    async def __call__(self, **kwargs):
        return dict(kwargs)


class _EchoWorkflow:
    class Inputs(BaseModel):
        model_config = ConfigDict(extra="allow")

    def __init__(self, ctx=None, *, name=None) -> None:
        self.ctx = ctx
        self.name = name or "echo_workflow"

    async def __call__(self, **kwargs):
        return dict(kwargs)


class _EchoPreset:
    name = "echo_child"
    workflow_cls = _EchoWorkflow
    component_configs = {_EchoWorkflow.__name__: {}}
    model_overrides: dict = {}

    def build_inputs(self, *, cli_overrides=None, **_kwargs):
        return dict(cli_overrides or {})


class DAGWorkflowRuntimeHelperTests(unittest.TestCase):
    def test_agent_input_schema_defaults_turn_missing_strings_into_empty_strings(self) -> None:
        agent = _EchoAgent()
        agent.component_config = {
            "input_schema": {
                "plan_md": "string",
                "prev_review_md": {"type": "string"},
                "round": "integer",
            }
        }

        inputs = _agent_inputs_with_workflow_defaults(
            agent,
            {"plan_md": None, "round": None},
            {"input": {}},
        )

        self.assertEqual(inputs["plan_md"], "")
        self.assertEqual(inputs["prev_review_md"], "")
        self.assertIsNone(inputs["round"])

    def test_eval_value_handles_nested_transforms_without_mutating_scope(self) -> None:
        scope = {
            "node": {
                "solver": {"drafts": ["a", "b"], "solution": ""},
                "backup": {"solution": "fallback"},
            },
            "input": {"problem": "P"},
        }

        self.assertEqual(_eval_value({"len": "$node.solver.drafts"}, scope), 2)
        self.assertEqual(
            _eval_value({"join": "$node.solver.drafts", "sep": " | "}, scope),
            "a | b",
        )
        self.assertEqual(
            _eval_value(
                {"format": "{problem}: {solution}", "fields": {"problem": "$input.problem", "solution": "$node.backup.solution"}},
                scope,
            ),
            "P: fallback",
        )
        self.assertEqual(
            _eval_value({"coalesce": ["$node.solver.solution", "$node.backup.solution"]}, scope),
            "fallback",
        )
        self.assertEqual(
            _eval_value({"add": ["$node.solver.drafts.0", 2]}, {"node": {"solver": {"drafts": [3]}}}),
            5,
        )
        self.assertTrue(
            _eval_value(
                {
                    "inputs": {"status": "$node.worker.status"},
                    "python": 'inputs.get("status") in ("done", "partial")',
                },
                {"node": {"worker": {"status": "done"}}},
            )
        )

    def test_build_outputs_can_coalesce_mutually_exclusive_branch_results(self) -> None:
        workflow = DAGWorkflow.__new__(DAGWorkflow)
        state = {
            "node": {
                "correct_branch": {},
                "incorrect_branch": {"solution": "fixed proof"},
            },
            "input": {},
        }

        out = workflow._build_outputs(
            {
                "solution": {
                    "coalesce": [
                        "$node.correct_branch.solution",
                        "$node.incorrect_branch.solution",
                    ]
                }
            },
            state,
        )

        self.assertEqual(out, {"solution": "fixed proof"})

    def test_bug_report_transform_ignores_ok_findings_and_formats_failures(self) -> None:
        report = _eval_value(
            {
                "bug_report_from_findings": [
                    {"verdict": "ok", "text": "fine", "comment": ""},
                    {"verdict": "gap", "text": "missing lemma", "comment": "prove it"},
                ]
            },
            {},
        )

        self.assertIn("[GAP] missing lemma", report)
        self.assertIn("prove it", report)
        self.assertNotIn("fine", report)

    def test_condition_modes_cover_comparisons_lists_verdicts_and_logic(self) -> None:
        scope = {
            "input": {"mode": "fast"},
            "node": {
                "validator": {
                    "findings": [
                        {"verdict": "ok"},
                        {"verdict": "gap"},
                    ]
                }
            },
            "items": ["a", "b"],
        }

        self.assertTrue(_condition({"ref": "$input.mode", "equals": "fast"}, scope))
        self.assertTrue(_condition({"ref": "$input.mode", "not_equals": "slow"}, scope))
        self.assertTrue(_condition({"ref": "$items", "min_len": 2, "max_len": 2}, scope))
        self.assertTrue(_condition({"ref": "$items", "contains": "b"}, scope))
        self.assertTrue(_condition({"ref": "$node.validator.findings", "any_verdict": ["gap"]}, scope))
        self.assertTrue(_condition({"all": [{"ref": "$items"}, {"not": {"ref": "$input.missing"}}]}, scope))
        self.assertFalse(_condition({"any": [{"ref": "$input.missing"}, False]}, scope))
        self.assertTrue(
            _condition(
                {
                    "inputs": {"False": "$node.validator.findings.1.verdict"},
                    "python": 'inputs.get("False") == "gap"',
                },
                scope,
            )
        )

    def test_python_condition_rejects_imports_and_dunder_access(self) -> None:
        self.assertTrue(_condition({"python": "len(items) == 2"}, {"items": [1, 2]}))

        with self.assertRaises(ValueError):
            _condition({"python_code": "import os\nresult = True"}, {})
        with self.assertRaises(ValueError):
            _condition({"python": "().__class__"}, {})

    def test_resolve_path_supports_lists_objects_and_callables(self) -> None:
        class Box:
            value = "object-value"

        scope = {
            "items": [{"name": "first"}, {"name": "second"}],
            "box": Box(),
            "run": {"elapsed_s": lambda: 3.5},
        }

        self.assertEqual(_resolve_path("items.1.name", scope), "second")
        self.assertEqual(_resolve_path("box.value", scope), "object-value")
        self.assertEqual(_resolve_path("run.elapsed_s", scope), 3.5)
        self.assertIsNone(_resolve_path("items.10.name", scope))

    def test_agent_nodes_receive_workflow_inputs_without_explicit_wires(self) -> None:
        workflow = DAGWorkflow.__new__(DAGWorkflow)
        workflow.events = _Events()
        agent = _FlexibleAgent()
        workflow._agent_for = lambda node: agent  # type: ignore[method-assign]

        class Inputs(BaseModel):
            problem: str
            problem_id: str

        state = {
            "input": Inputs(problem="Prove P.", problem_id="p"),
            "node": {"solver": {"solution": "draft"}},
        }
        out = asyncio.run(
            workflow._run_agent_node(
                {
                    "id": "verifier",
                    "kind": "agent",
                    "inputs": {"solution": "$node.solver.solution"},
                },
                state,
            )
        )

        self.assertEqual(out, {"ok": True})
        self.assertEqual(agent.calls[0]["problem"], "Prove P.")
        self.assertEqual(agent.calls[0]["problem_id"], "p")
        self.assertEqual(agent.calls[0]["solution"], "draft")

    def test_run_node_emits_dag_error_when_agent_node_fails(self) -> None:
        workflow = DAGWorkflow.__new__(DAGWorkflow)
        workflow.events = _Events()

        async def fail(node, state):
            raise KeyError("problem")

        workflow._run_agent_node = fail  # type: ignore[method-assign]
        state = {"input": {}, "node": {}, "path": {}}

        with self.assertRaises(KeyError):
            asyncio.run(workflow._run_node({"id": "verifier", "kind": "agent"}, state))

        self.assertIn(("dag.node_started", {"node": "verifier", "kind": "agent"}), workflow.events.items)
        self.assertIn(
            (
                "dag.node_error",
                {"node": "verifier", "kind": "agent", "type": "KeyError", "msg": "'problem'"},
            ),
            workflow.events.items,
        )

    def test_loop_node_stops_when_condition_turns_false_and_reports_history(self) -> None:
        workflow = DAGWorkflow.__new__(DAGWorkflow)
        workflow.events = _Events()
        state = {"node": {}, "input": {}, "best_tex": None}
        node = {
            "id": "loop",
            "kind": "repeat",
            "max_iterations": 5,
            "initial_state": {"done": False, "solution": "start"},
            "condition": {"python": "not inputs.get('done')"},
            "body": {
                "nodes": [],
                "state_updates": {"done": True, "solution": "finished"},
            },
            "outputs": {
                "solution": "$state.solution",
                "done": "$state.done",
                "iterations": "$loop.iterations",
                "reason": "$loop.reason",
            },
        }

        out = asyncio.run(workflow._run_loop_node(node, state))

        self.assertEqual(out["solution"], "finished")
        self.assertTrue(out["done"])
        self.assertEqual(out["iterations"], 1)
        self.assertEqual(out["reason"], "condition_false")
        self.assertEqual(len(out["history"]), 1)
        self.assertEqual(workflow.events.items[0][0], "dag.loop_iteration_started")


    def test_loop_history_serializes_pydantic_node_outputs_for_conditions(self) -> None:
        class VerifierOutput(BaseModel):
            verdict: str
            verification: str = ""

        workflow = DAGWorkflow.__new__(DAGWorkflow)
        workflow.events = _Events()

        async def run_nodes(_nodes, scope, **_kwargs):
            scope["node"]["verifier"] = VerifierOutput(verdict="correct", verification="ok")
            scope["node"]["improver"] = {"solution": "improved"}
            return False

        workflow._run_nodes = run_nodes
        state = {"node": {}, "input": {}, "best_tex": None}
        node = {
            "id": "loop",
            "kind": "repeat",
            "max_iterations": 3,
            "initial_state": {"solution": "start"},
            "condition": {
                "python": (
                    'iteration == 0 or '
                    'str(history[-1]["node"].get("verifier", {}).get("verdict", "")).strip().lower() != "correct"'
                )
            },
            "body": {
                "nodes": [{"id": "verifier", "kind": "agent"}],
                "state_updates": {"solution": "$node.improver.solution"},
            },
            "outputs": {"history": "$history", "iterations": "$loop.iterations", "reason": "$loop.reason"},
        }

        out = asyncio.run(workflow._run_loop_node(node, state))

        self.assertEqual(out["iterations"], 1)
        self.assertEqual(out["reason"], "condition_false")
        self.assertEqual(out["history"][0]["node"]["verifier"]["verdict"], "correct")

    def test_workflow_ref_inherits_parent_workflow_inputs(self) -> None:
        workflow = DAGWorkflow.__new__(DAGWorkflow)
        with tempfile.TemporaryDirectory() as temp_dir:
            workflow.ctx = RunContext.create(run_id="test", root_workdir=temp_dir, flat=True)
            state = {
                "input": {"problem": "parent problem", "problem_id": "p1"},
                "node": {},
            }
            node = {
                "id": "child",
                "kind": "workflow_ref",
                "preset": "echo_child",
                "inputs": {"solution": "draft proof"},
            }
            with patch("proofstack.registry.load_preset", return_value=_EchoPreset()):
                out = asyncio.run(workflow._run_workflow_ref_node(node, state))

        self.assertEqual(out["problem"], "parent problem")
        self.assertEqual(out["problem_id"], "p1")
        self.assertEqual(out["solution"], "draft proof")

    def test_loop_node_rejects_invalid_state_update_shapes(self) -> None:
        workflow = DAGWorkflow.__new__(DAGWorkflow)
        workflow.events = _Events()
        node = {
            "id": "loop",
            "kind": "repeat",
            "max_iterations": 1,
            "initial_state": {},
            "body": {"nodes": [], "state_updates": ["not", "a", "mapping"]},
        }

        with self.assertRaisesRegex(TypeError, "state_updates must evaluate to a mapping"):
            asyncio.run(workflow._run_loop_node(node, {"node": {}, "input": {}}))

    def test_if_else_branch_with_no_active_edge_terminates_that_path(self) -> None:
        workflow = DAGWorkflow.__new__(DAGWorkflow)
        workflow.events = _Events()
        state = {"node": {}, "input": {"mode": "stop"}}
        nodes = [
            {
                "id": "router",
                "kind": "if_else",
                "inputs": {"mode": "$input.mode"},
                "condition": {"ref": "$inputs.mode", "equals": "go"},
                "then": {"go": True},
                "else": {},
            },
            {
                "id": "next_step",
                "kind": "if_else",
                "needs": ["router"],
                "inputs": {"go": "$node.router.go"},
                "condition": {"ref": "$inputs.go", "equals": True},
                "then": {"ran": True},
            },
        ]

        terminal = asyncio.run(workflow._run_nodes(nodes, state))

        self.assertTrue(terminal)
        self.assertIn("router", state["node"])
        self.assertNotIn("next_step", state["node"])
        self.assertIn(("dag.branch_terminal", {"node": "router", "branch": "else"}), workflow.events.items)
        self.assertIn(("dag.node_pruned", {"node": "next_step", "reason": "router.else branch ended"}), workflow.events.items)

    def test_if_else_active_branch_edge_still_runs_downstream_node(self) -> None:
        workflow = DAGWorkflow.__new__(DAGWorkflow)
        workflow.events = _Events()
        state = {"node": {}, "input": {"mode": "go"}}
        nodes = [
            {
                "id": "router",
                "kind": "if_else",
                "inputs": {"mode": "$input.mode"},
                "condition": {"ref": "$inputs.mode", "equals": "go"},
                "then": {"go": True},
                "else": {},
            },
            {
                "id": "next_step",
                "kind": "if_else",
                "needs": ["router"],
                "inputs": {"go": "$node.router.go"},
                "condition": {"ref": "$inputs.go", "equals": True},
                "then": {"ran": True},
            },
        ]

        asyncio.run(workflow._run_nodes(nodes, state))

        self.assertTrue(state["node"]["next_step"]["ran"])

    def test_repeat_stops_when_if_else_terminal_branch_is_taken(self) -> None:
        workflow = DAGWorkflow.__new__(DAGWorkflow)
        workflow.events = _Events()
        state = {"node": {}, "input": {}, "best_tex": None}
        node = {
            "id": "loop",
            "kind": "repeat",
            "max_iterations": 5,
            "initial_state": {"solution": "current"},
            "condition": True,
            "body": {
                "nodes": [
                    {
                        "id": "router",
                        "kind": "if_else",
                        "condition": False,
                        "then": {"continue": True},
                        "else": {},
                    },
                    {
                        "id": "repair",
                        "kind": "if_else",
                        "needs": ["router"],
                        "inputs": {"continue": "$node.router.continue"},
                        "condition": {"ref": "$inputs.continue", "equals": True},
                        "then": {"solution": "changed"},
                    },
                ],
                "state_updates": {
                    "solution": {"coalesce": ["$node.repair.solution", "$state.solution"]}
                },
            },
            "outputs": {
                "solution": "$state.solution",
                "iterations": "$loop.iterations",
                "reason": "$loop.reason",
            },
        }

        out = asyncio.run(workflow._run_loop_node(node, state))

        self.assertEqual(out["solution"], "current")
        self.assertEqual(out["iterations"], 1)
        self.assertEqual(out["reason"], "terminal_branch")

    def test_repeat_state_update_counts_as_drawn_if_else_edge(self) -> None:
        workflow = DAGWorkflow.__new__(DAGWorkflow)
        workflow.events = _Events()
        state = {"node": {}, "input": {}, "best_tex": None}
        node = {
            "id": "loop",
            "kind": "repeat",
            "max_iterations": 2,
            "initial_state": {"solution": "current"},
            "condition": True,
            "body": {
                "nodes": [
                    {
                        "id": "router",
                        "kind": "if_else",
                        "condition": True,
                        "then": {"solution": "updated"},
                        "else": {},
                    },
                ],
                "state_updates": {"solution": "$node.router.solution"},
            },
            "outputs": {
                "solution": "$state.solution",
                "iterations": "$loop.iterations",
                "reason": "$loop.reason",
            },
        }

        out = asyncio.run(workflow._run_loop_node(node, state))

        self.assertEqual(out["solution"], "updated")
        self.assertEqual(out["iterations"], 2)
        self.assertEqual(out["reason"], "max_iterations")

    def test_if_else_and_join_nodes_have_no_model_dependency_for_simple_cases(self) -> None:
        workflow = DAGWorkflow.__new__(DAGWorkflow)
        state = {
            "node": {"source": {"values": ["single"]}},
            "input": {"verdict": "ok"},
        }

        routed = asyncio.run(
            workflow._run_if_else_node(
                {
                    "id": "router",
                    "kind": "if_else",
                    "inputs": {"verdict": "$input.verdict"},
                    "condition": {"ref": "$inputs.verdict", "equals": "ok"},
                    "then": {"solution": "accepted"},
                    "else": {"solution": "retry"},
                },
                state,
            )
        )
        self.assertEqual(routed["solution"], "accepted")
        self.assertTrue(routed["condition"])
        self.assertNotIn("branch", routed)
        self.assertEqual(state["_if_branch"]["router"], "then")

        state["input"]["verdict"] = "retry"
        routed = asyncio.run(
            workflow._run_if_else_node(
                {
                    "id": "default_router",
                    "kind": "if_else",
                    "inputs": {"verdict": "$input.verdict"},
                    "condition": {"ref": "$inputs.verdict", "equals": "ok"},
                },
                state,
            )
        )
        self.assertEqual(routed, {"False": True, "condition": False})
        self.assertEqual(state["_if_branch"]["default_router"], "else")

        joined = asyncio.run(
            workflow._run_join_or_agent(
                {"id": "join", "kind": "join_or_agent", "source": "$node.source.values", "output_field": "solution"},
                state,
            )
        )
        self.assertEqual(joined, {"solution": "single"})

    def test_coalesced_input_runs_after_one_mutually_exclusive_branch_is_pruned(self) -> None:
        workflow = DAGWorkflow.__new__(DAGWorkflow)
        workflow.ctx = None
        workflow._agents = {}
        workflow.events = _Events()
        workflow._agent_for = lambda _node: _EchoAgent()
        nodes = [
            {
                "id": "router",
                "kind": "if_else",
                "inputs": {"verdict": "$input.verdict"},
                "condition": {"python": 'inputs.get("verdict") == "correct"'},
                "then": {"True": True},
                "else": {"False": True},
            },
            {
                "id": "correct",
                "kind": "agent",
                "needs": ["router"],
                "when": {"inputs": {"True": "$node.router.True"}, "python": 'inputs.get("True")'},
                "inputs": {"solution": "polished proof"},
            },
            {
                "id": "incorrect",
                "kind": "agent",
                "needs": ["router"],
                "when": {"inputs": {"False": "$node.router.False"}, "python": 'inputs.get("False")'},
                "inputs": {"solution": "repaired proof"},
            },
            {
                "id": "latex",
                "kind": "agent",
                "needs": ["correct", "incorrect"],
                "inputs": {
                    "tex_body": {
                        "coalesce": [
                            "$node.correct.solution",
                            "$node.incorrect.solution",
                        ]
                    }
                },
            },
        ]
        state = {
            "input": {"verdict": "correct"},
            "node": {},
            "path": {"node_counts": {}, "last_node": None, "current_node": None},
        }

        asyncio.run(workflow._run_nodes(nodes, state))

        self.assertEqual(state["node"]["correct"]["solution"], "polished proof")
        self.assertNotIn("incorrect", state["node"])
        self.assertEqual(state["node"]["latex"]["tex_body"], "polished proof")


if __name__ == "__main__":
    unittest.main()
