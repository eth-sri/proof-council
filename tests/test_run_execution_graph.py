from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.dev_data import (  # noqa: E402
    load_event_tree,
    load_execution_graph,
    workflow_input_from_tree,
    workflow_output_from_tree,
)


def _write_event(path: Path, **event) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


class RunExecutionGraphTests(unittest.TestCase):
    def test_run_graph_template_never_renders_node_subtitles(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_run_graph.html").read_text()

        self.assertNotIn("node.subtitle", template)
        self.assertNotIn("<p>{{ node.subtitle }}</p>", template)

    def test_run_detail_template_shows_workflow_input_and_output_below_execution_graph(self) -> None:
        template = (ROOT / "app" / "templates" / "dev_run_detail.html").read_text()

        self.assertLess(template.index("Execution graph"), template.index("Workflow input"))
        self.assertLess(template.index("Workflow input"), template.index("Workflow output"))
        self.assertIn("ui.render_input(workflow_input", template)
        self.assertIn("ui.render_output(workflow_output", template)

    def test_workflow_input_and_problem_preview_are_blue_latex_surfaces(self) -> None:
        macros = (ROOT / "app" / "templates" / "dev_macros.html").read_text()
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()
        run_agent = (ROOT / "app" / "templates" / "dev_run_agent.html").read_text()
        input_start = macros.index("{% macro render_input")
        input_body = macros[input_start:]

        self.assertIn('class="message-block message-input"', input_body)
        self.assertIn('class="rendered-text latex-text"', input_body)
        self.assertIn("details.message-input", base)
        self.assertIn("border-left-color: #2563eb", base)
        self.assertIn("details.message-block .rendered-text", base)
        self.assertIn("border: 0;", base)
        self.assertIn('class="problem-preview latex-text"', run_agent)
        self.assertIn("renderLatexElements(row)", run_agent)
        self.assertIn("processEnvironments: false", base)

    def test_io_message_blocks_have_no_orange_accents(self) -> None:
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()
        output_start = base.index("details.message-output")
        input_start = base.index("details.message-input")
        io_css = base[output_start:input_start] + base[input_start:base.index(".problem-preview", input_start)]
        forbidden = {
            "#f97316", "#ea580c", "#fff7ed", "#fffaf5", "#ffedd5",
            "#9a3412", "#92400e", "#854d0e", "#fef3c7", "#fef9c3",
            "#ca8a04", "#eab308", "#fed7aa", "#d97706",
        }
        present = sorted(item for item in forbidden if item in io_css)
        self.assertEqual(present, [])

    def test_execution_graph_keeps_status_colors(self) -> None:
        base = (ROOT / "app" / "templates" / "dev_base.html").read_text()

        self.assertIn(".graph-node.status-ok { border-left-color: #16a34a; }", base)
        self.assertIn(".graph-node.status-running { border-left-color: #f97316; }", base)
        self.assertIn(".graph-node.status-pending { border-left-color: #eab308; }", base)

    def test_workflow_input_uses_top_level_workflow_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_path = Path(tmp) / "run"
            run_path.mkdir()
            events_path = run_path / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="agent.start",
                call_id="workflow",
                parent_call_id=None,
                agent="DAGWorkflow",
                agent_path="DAGWorkflow",
                execution_mode="workflow",
                payload={
                    "input": {
                        "problem": "Prove that the square root of 2 is irrational.",
                        "problem_id": "editor_sample",
                        "unused": None,
                        "artifact": "/tmp/run/input.json",
                    }
                },
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:01.000Z",
                kind="agent.start",
                call_id="solver-call",
                parent_call_id="workflow",
                agent="cfg_solver",
                agent_path="DAGWorkflow.cfg_solver",
                execution_mode="agent",
                payload={"input": {"problem": "child input"}},
            )

            tree = load_event_tree(run_path)

        self.assertEqual(
            workflow_input_from_tree(tree),
            {
                "problem": "Prove that the square root of 2 is irrational.",
                "problem_id": "editor_sample",
            },
        )

    def test_workflow_output_uses_top_level_workflow_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_path = Path(tmp) / "run"
            run_path.mkdir()
            events_path = run_path / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="agent.start",
                call_id="workflow",
                parent_call_id=None,
                agent="DAGWorkflow",
                agent_path="DAGWorkflow",
                execution_mode="workflow",
                payload={},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:01.000Z",
                kind="agent.start",
                call_id="solver-call",
                parent_call_id="workflow",
                agent="cfg_solver",
                agent_path="DAGWorkflow.cfg_solver",
                execution_mode="agent",
                payload={},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:02.000Z",
                kind="agent.end",
                call_id="solver-call",
                parent_call_id="workflow",
                payload={"output": {"solution": "child output"}},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:03.000Z",
                kind="agent.end",
                call_id="workflow",
                parent_call_id=None,
                payload={
                    "output": {
                        "solution": "<solution>final proof</solution>",
                        "raw_text": "internal model transcript",
                        "empty": None,
                        "artifact": "/tmp/run/output.tex",
                        "tex_path": "/tmp/run/final.tex",
                        "pdf_path": "/tmp/run/final.pdf",
                        "solution_tex": "/tmp/run/solutions/final.tex",
                        "nested": {"keep": "visible", "missing": None},
                    }
                },
            )

            tree = load_event_tree(run_path)

        self.assertEqual(
            workflow_output_from_tree(tree),
            {
                "solution": "<solution>final proof</solution>",
                "tex_path": "/tmp/run/final.tex",
                "pdf_path": "/tmp/run/final.pdf",
                "solution_tex": "/tmp/run/solutions/final.tex",
                "nested": {"keep": "visible"},
            },
        )

    def test_agent_output_status_error_marks_call_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_path = Path(tmp) / "run"
            run_path.mkdir()
            events_path = run_path / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="agent.start",
                call_id="compile-call",
                parent_call_id="workflow",
                agent="cfg_compile_latex",
                agent_path="DAGWorkflow.cfg_compile_latex",
                execution_mode="agent",
                payload={"input": {}},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:01.000Z",
                kind="agent.end",
                call_id="compile-call",
                parent_call_id="workflow",
                agent="cfg_compile_latex",
                agent_path="DAGWorkflow.cfg_compile_latex",
                execution_mode="agent",
                payload={"output": {"status": "error", "summary": "compile failed"}},
            )

            tree = load_event_tree(run_path)

        call = tree.by_id["compile-call"]
        self.assertEqual(call.status, "error")
        self.assertEqual(call.error, {"type": "OutputStatus", "msg": "compile failed"})

    def test_execution_graph_uses_editor_labels_and_pending_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preset_root = root / "presets"
            preset_root.mkdir()
            (preset_root / "demo.yaml").write_text(
                textwrap.dedent(
                    """
                    workflow: proofstack.agents.dag_workflow.DAGWorkflow
                    dag:
                      nodes:
                        - id: solver
                          kind: agent
                          name: cfg_solver
                          ui:
                            label: Draft solver
                            subtitle: Writes the first proof.
                        - id: checker
                          kind: agent
                          needs: [solver]
                          name: cfg_checker
                          ui:
                            label: Verify proof
                            subtitle: Checks the draft.
                    """
                ),
                encoding="utf-8",
            )
            run_path = root / "run"
            run_path.mkdir()
            events_path = run_path / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="run.start",
                payload={"preset": "demo"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:01.000Z",
                kind="agent.start",
                call_id="workflow",
                parent_call_id=None,
                agent="DAGWorkflow",
                agent_path="DAGWorkflow",
                execution_mode="workflow",
                payload={},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:02.000Z",
                kind="dag.node_started",
                call_id=None,
                parent_call_id="workflow",
                payload={"node": "solver", "kind": "agent"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:03.000Z",
                kind="agent.start",
                call_id="solver-call",
                parent_call_id="workflow",
                agent="cfg_solver",
                agent_path="DAGWorkflow.cfg_solver",
                execution_mode="agent",
                payload={},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:01:03.000Z",
                kind="model.call",
                call_id="model-call",
                parent_call_id="solver-call",
                payload={"cost_usd": 0.001},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:01:04.000Z",
                kind="agent.end",
                call_id="solver-call",
                parent_call_id="workflow",
                payload={},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:01:05.000Z",
                kind="dag.node_done",
                call_id=None,
                parent_call_id="workflow",
                payload={"node": "solver", "kind": "agent"},
            )

            tree = load_event_tree(run_path)
            graph = load_execution_graph(run_path, tree=tree, preset_root=preset_root)

        self.assertEqual([node.label for node in graph.roots], ["Draft solver", "Verify proof"])
        self.assertEqual(graph.by_id["solver"].status, "ok")
        self.assertEqual(graph.by_id["solver"].call_id, "solver-call")
        self.assertAlmostEqual(graph.by_id["solver"].cost_usd, 0.001)
        self.assertEqual(graph.by_id["checker"].status, "pending")

    def test_subworkflow_dag_nodes_render_under_workflow_ref_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preset_root = root / "presets"
            preset_root.mkdir()
            (preset_root / "demo.yaml").write_text(
                textwrap.dedent(
                    """
                    workflow: proofstack.agents.dag_workflow.DAGWorkflow
                    dag:
                      nodes:
                        - id: verify_improve_loop
                          kind: repeat
                          body:
                            nodes:
                              - id: verify_improve
                                kind: workflow_ref
                                preset: verify_improve
                                ui:
                                  label: Verify / Improve
                    """
                ),
                encoding="utf-8",
            )
            run_path = root / "run"
            run_path.mkdir()
            events_path = run_path / "events.jsonl"
            _write_event(events_path, kind="run.start", payload={"preset": "demo"})
            _write_event(
                events_path,
                kind="agent.start",
                call_id="workflow",
                parent_call_id=None,
                execution_mode="workflow",
                payload={},
            )
            _write_event(
                events_path,
                kind="dag.node_started",
                parent_call_id="workflow",
                payload={"node": "verify_improve_loop", "kind": "repeat"},
            )
            _write_event(
                events_path,
                kind="dag.node_started",
                parent_call_id="workflow",
                payload={"node": "verify_improve", "kind": "workflow_ref"},
            )
            _write_event(
                events_path,
                kind="agent.start",
                call_id="subflow-call",
                parent_call_id="workflow",
                execution_mode="workflow",
                payload={},
            )
            _write_event(
                events_path,
                kind="dag.node_started",
                parent_call_id="subflow-call",
                payload={"node": "verifier", "kind": "agent"},
            )
            _write_event(
                events_path,
                kind="agent.start",
                call_id="verifier-call",
                parent_call_id="subflow-call",
                execution_mode="agent",
                payload={},
            )
            _write_event(
                events_path,
                kind="agent.end",
                call_id="verifier-call",
                parent_call_id="subflow-call",
                payload={},
            )
            _write_event(
                events_path,
                kind="dag.node_done",
                parent_call_id="subflow-call",
                payload={"node": "verifier", "kind": "agent"},
            )

            tree = load_event_tree(run_path)
            graph = load_execution_graph(run_path, tree=tree, preset_root=preset_root)

        loop = graph.by_id["verify_improve_loop"]
        subworkflow = loop.children[0]

        self.assertEqual(subworkflow.raw_id, "verify_improve")
        self.assertEqual(subworkflow.call_id, "subflow-call")
        self.assertEqual([child.raw_id for child in subworkflow.children], ["verifier"])
        self.assertEqual(subworkflow.children[0].call_id, "verifier-call")

    def test_repeated_node_executions_create_separate_graph_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preset_root = root / "presets"
            preset_root.mkdir()
            (preset_root / "demo.yaml").write_text(
                textwrap.dedent(
                    """
                    workflow: proofstack.agents.dag_workflow.DAGWorkflow
                    dag:
                      nodes:
                        - id: verify_improve_loop
                          kind: repeat
                          body:
                            nodes:
                              - id: verify_improve
                                kind: workflow_ref
                                preset: verify_improve
                                ui:
                                  label: Verify / Improve
                    """
                ),
                encoding="utf-8",
            )
            run_path = root / "run"
            run_path.mkdir()
            events_path = run_path / "events.jsonl"
            _write_event(events_path, kind="run.start", payload={"preset": "demo"})
            _write_event(
                events_path,
                kind="agent.start",
                call_id="workflow",
                parent_call_id=None,
                execution_mode="workflow",
                payload={},
            )
            _write_event(
                events_path,
                kind="dag.node_started",
                parent_call_id="workflow",
                payload={"node": "verify_improve_loop", "kind": "repeat"},
            )
            for index in (1, 2):
                _write_event(
                    events_path,
                    kind="dag.node_started",
                    parent_call_id="workflow",
                    payload={"node": "verify_improve", "kind": "workflow_ref"},
                )
                _write_event(
                    events_path,
                    kind="agent.start",
                    call_id=f"subflow-{index}",
                    parent_call_id="workflow",
                    execution_mode="workflow",
                    payload={},
                )
                _write_event(
                    events_path,
                    kind="dag.node_started",
                    parent_call_id=f"subflow-{index}",
                    payload={"node": "verifier", "kind": "agent"},
                )
                _write_event(
                    events_path,
                    kind="dag.node_done",
                    parent_call_id=f"subflow-{index}",
                    payload={"node": "verifier", "kind": "agent"},
                )
                _write_event(
                    events_path,
                    kind="agent.end",
                    call_id=f"subflow-{index}",
                    parent_call_id="workflow",
                    execution_mode="workflow",
                    payload={},
                )
                _write_event(
                    events_path,
                    kind="dag.node_done",
                    parent_call_id="workflow",
                    payload={"node": "verify_improve", "kind": "workflow_ref"},
                )

            graph = load_execution_graph(run_path, preset_root=preset_root)

        loop = graph.by_id["verify_improve_loop"]
        executions = [child for child in loop.children if child.raw_id == "verify_improve"]

        self.assertEqual([node.execution_index for node in executions], [1, 2])
        self.assertEqual([node.call_id for node in executions], ["subflow-1", "subflow-2"])
        self.assertEqual([node.children[0].raw_id for node in executions], ["verifier", "verifier"])

    def test_skipped_child_in_later_repeat_instance_does_not_overwrite_first_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preset_root = root / "presets"
            preset_root.mkdir()
            (preset_root / "demo.yaml").write_text(
                textwrap.dedent(
                    """
                    workflow: proofstack.agents.dag_workflow.DAGWorkflow
                    dag:
                      nodes:
                        - id: loop
                          kind: repeat
                          body:
                            nodes:
                              - id: round
                                kind: workflow_ref
                                preset: round
                    """
                ),
                encoding="utf-8",
            )
            run_path = root / "run"
            run_path.mkdir()
            events_path = run_path / "events.jsonl"
            _write_event(events_path, kind="run.start", payload={"preset": "demo"})
            _write_event(
                events_path,
                kind="agent.start",
                call_id="workflow",
                parent_call_id=None,
                execution_mode="workflow",
                payload={},
            )
            _write_event(
                events_path,
                kind="dag.node_started",
                parent_call_id="workflow",
                payload={"node": "loop", "kind": "repeat"},
            )
            _write_event(
                events_path,
                kind="dag.node_started",
                parent_call_id="workflow",
                payload={"node": "round", "kind": "workflow_ref"},
            )
            _write_event(
                events_path,
                kind="agent.start",
                call_id="round-1",
                parent_call_id="workflow",
                execution_mode="workflow",
                payload={},
            )
            _write_event(
                events_path,
                kind="dag.node_started",
                parent_call_id="round-1",
                payload={"node": "worker", "kind": "agent"},
            )
            _write_event(
                events_path,
                kind="agent.start",
                call_id="worker-1",
                parent_call_id="round-1",
                execution_mode="agent",
                payload={},
            )
            _write_event(
                events_path,
                kind="agent.end",
                call_id="worker-1",
                parent_call_id="round-1",
                execution_mode="agent",
                payload={},
            )
            _write_event(
                events_path,
                kind="dag.node_done",
                parent_call_id="round-1",
                payload={"node": "worker", "kind": "agent"},
            )
            _write_event(
                events_path,
                kind="agent.end",
                call_id="round-1",
                parent_call_id="workflow",
                execution_mode="workflow",
                payload={},
            )
            _write_event(
                events_path,
                kind="dag.node_done",
                parent_call_id="workflow",
                payload={"node": "round", "kind": "workflow_ref"},
            )
            _write_event(
                events_path,
                kind="dag.node_started",
                parent_call_id="workflow",
                payload={"node": "round", "kind": "workflow_ref"},
            )
            _write_event(
                events_path,
                kind="agent.start",
                call_id="round-2",
                parent_call_id="workflow",
                execution_mode="workflow",
                payload={},
            )
            _write_event(
                events_path,
                kind="dag.node_skipped",
                parent_call_id="round-2",
                payload={"node": "worker", "kind": "agent"},
            )
            _write_event(
                events_path,
                kind="agent.end",
                call_id="round-2",
                parent_call_id="workflow",
                execution_mode="workflow",
                payload={},
            )
            _write_event(
                events_path,
                kind="dag.node_done",
                parent_call_id="workflow",
                payload={"node": "round", "kind": "workflow_ref"},
            )

            tree = load_event_tree(run_path)
            graph = load_execution_graph(run_path, tree=tree, preset_root=preset_root)

        rounds = [child for child in graph.by_id["loop"].children if child.raw_id == "round"]
        workers = [round_node.children[0] for round_node in rounds]

        self.assertEqual([round_node.call_id for round_node in rounds], ["round-1", "round-2"])
        self.assertEqual([worker.status for worker in workers], ["ok", "skipped"])
        self.assertEqual([worker.call_id for worker in workers], ["worker-1", None])

    def test_parallel_repeat_zones_with_same_body_id_keep_separate_executions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preset_root = root / "presets"
            preset_root.mkdir()
            (preset_root / "demo.yaml").write_text(
                textwrap.dedent(
                    """
                    workflow: proofstack.agents.dag_workflow.DAGWorkflow
                    dag:
                      nodes:
                        - id: first_loop
                          kind: repeat
                          ui:
                            label: First loop
                          body:
                            nodes:
                              - id: verify_improve
                                kind: workflow_ref
                                preset: verify_improve
                                ui:
                                  label: Verify / Improve
                        - id: second_loop
                          kind: repeat
                          ui:
                            label: Second loop
                          body:
                            nodes:
                              - id: verify_improve
                                kind: workflow_ref
                                preset: verify_improve
                                ui:
                                  label: Verify / Improve
                    """
                ),
                encoding="utf-8",
            )
            run_path = root / "run"
            run_path.mkdir()
            events_path = run_path / "events.jsonl"
            _write_event(events_path, kind="run.start", payload={"preset": "demo"})
            _write_event(
                events_path,
                kind="agent.start",
                call_id="workflow",
                parent_call_id=None,
                execution_mode="workflow",
                payload={},
            )
            _write_event(
                events_path,
                kind="dag.node_started",
                parent_call_id="workflow",
                payload={"node": "first_loop", "kind": "repeat"},
            )
            _write_event(
                events_path,
                kind="dag.node_started",
                parent_call_id="workflow",
                payload={"node": "second_loop", "kind": "repeat"},
            )
            _write_event(
                events_path,
                kind="dag.node_started",
                parent_call_id="workflow",
                payload={"node": "verify_improve", "kind": "workflow_ref"},
            )
            _write_event(
                events_path,
                kind="dag.node_started",
                parent_call_id="workflow",
                payload={"node": "verify_improve", "kind": "workflow_ref"},
            )
            for call_id in ("first-call", "second-call"):
                _write_event(
                    events_path,
                    kind="agent.start",
                    call_id=call_id,
                    parent_call_id="workflow",
                    execution_mode="workflow",
                    payload={},
                )
            for loop_id, call_id in (("first_loop", "first-call"), ("second_loop", "second-call")):
                _write_event(
                    events_path,
                    kind="agent.end",
                    call_id=call_id,
                    parent_call_id="workflow",
                    execution_mode="workflow",
                    payload={},
                )
                _write_event(
                    events_path,
                    kind="dag.node_done",
                    parent_call_id="workflow",
                    payload={"node": "verify_improve", "kind": "workflow_ref"},
                )
                _write_event(
                    events_path,
                    kind="dag.node_done",
                    parent_call_id="workflow",
                    payload={"node": loop_id, "kind": "repeat"},
                )

            graph = load_execution_graph(run_path, preset_root=preset_root)

        first = [child for child in graph.by_id["first_loop"].children if child.raw_id == "verify_improve"]
        second = [child for child in graph.by_id["second_loop"].children if child.raw_id == "verify_improve"]

        self.assertEqual([node.call_id for node in first], ["first-call"])
        self.assertEqual([node.execution_index for node in first], [1])
        self.assertEqual([node.call_id for node in second], ["second-call"])
        self.assertEqual([node.execution_index for node in second], [1])

    def test_execution_graph_keeps_event_only_nodes_when_preset_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_path = Path(tmp) / "run"
            run_path.mkdir()
            events_path = run_path / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="dag.node_started",
                parent_call_id="workflow",
                payload={"node": "budget_fallback", "kind": "if_else", "label": "Budget fallback"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:01.000Z",
                kind="dag.node_pruned",
                parent_call_id="workflow",
                payload={"node": "budget_fallback", "reason": "normal path finished"},
            )

            graph = load_execution_graph(run_path)

        self.assertEqual(len(graph.roots), 1)
        self.assertEqual(graph.roots[0].label, "Budget fallback")
        self.assertEqual(graph.roots[0].status, "skipped")
        self.assertEqual(graph.roots[0].reason, "normal path finished")

    def test_execution_graph_uses_call_errors_and_editor_label_for_failed_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preset_root = root / "presets"
            preset_root.mkdir()
            (preset_root / "demo.yaml").write_text(
                textwrap.dedent(
                    """
                    workflow: proofstack.agents.dag_workflow.DAGWorkflow
                    dag:
                      nodes:
                        - id: prompt
                          kind: agent
                          name: cfg_prompt
                          ui:
                            label: Verifier
                    """
                ),
                encoding="utf-8",
            )
            run_path = root / "run"
            run_path.mkdir()
            events_path = run_path / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="run.start",
                payload={"preset": "demo"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:01.000Z",
                kind="agent.start",
                call_id="workflow",
                parent_call_id=None,
                agent="DAGWorkflow",
                agent_path="DAGWorkflow",
                execution_mode="workflow",
                payload={},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:02.000Z",
                kind="dag.node_started",
                parent_call_id="workflow",
                payload={"node": "prompt", "kind": "agent"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:03.000Z",
                kind="agent.start",
                call_id="prompt-call",
                parent_call_id="workflow",
                agent="cfg_prompt",
                agent_path="DAGWorkflow.cfg_prompt",
                execution_mode="agent",
                payload={"input": {"solution": "draft"}},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:04.000Z",
                kind="agent.error",
                call_id="prompt-call",
                parent_call_id="workflow",
                agent="cfg_prompt",
                agent_path="DAGWorkflow.cfg_prompt",
                payload={"type": "KeyError", "msg": "'problem'"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:05.000Z",
                kind="workflow.last_gasp",
                parent_call_id="workflow",
                payload={"type": "KeyError", "msg": "'problem'"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:06.000Z",
                kind="run.end",
                payload={"status": "ok"},
            )

            tree = load_event_tree(run_path)
            graph = load_execution_graph(run_path, tree=tree, preset_root=preset_root)

        self.assertEqual(graph.by_id["prompt"].status, "error")
        self.assertEqual(graph.by_id["prompt"].reason, "'problem'")
        self.assertEqual(tree.by_id["prompt-call"].display_name, "Verifier")

    def test_execution_graph_shows_empty_model_response_as_error_even_when_node_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preset_root = root / "presets"
            preset_root.mkdir()
            (preset_root / "demo.yaml").write_text(
                textwrap.dedent(
                    """
                    workflow: proofstack.agents.dag_workflow.DAGWorkflow
                    dag:
                      nodes:
                        - id: prompt
                          kind: agent
                          name: cfg_prompt
                          ui:
                            label: Verifier
                    """
                ),
                encoding="utf-8",
            )
            run_path = root / "run"
            run_path.mkdir()
            events_path = run_path / "events.jsonl"
            _write_event(
                events_path,
                ts="2026-05-08T09:00:00.000Z",
                kind="run.start",
                payload={"preset": "demo"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:01.000Z",
                kind="agent.start",
                call_id="workflow",
                parent_call_id=None,
                agent="DAGWorkflow",
                agent_path="DAGWorkflow",
                execution_mode="workflow",
                payload={},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:02.000Z",
                kind="dag.node_started",
                parent_call_id="workflow",
                payload={"node": "prompt", "kind": "agent"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:03.000Z",
                kind="agent.start",
                call_id="prompt-call",
                parent_call_id="workflow",
                agent="cfg_prompt",
                agent_path="DAGWorkflow.cfg_prompt",
                execution_mode="agent",
                payload={},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:04.000Z",
                kind="model.empty_response",
                call_id="model-call",
                parent_call_id="prompt-call",
                agent="cfg_prompt",
                agent_path="DAGWorkflow.cfg_prompt",
                payload={"type": "EmptyResponse", "msg": "model fake returned an empty response"},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:05.000Z",
                kind="agent.end",
                call_id="prompt-call",
                parent_call_id="workflow",
                agent="cfg_prompt",
                agent_path="DAGWorkflow.cfg_prompt",
                payload={"output": {"verification": "", "verdict": ""}},
            )
            _write_event(
                events_path,
                ts="2026-05-08T09:00:06.000Z",
                kind="dag.node_done",
                parent_call_id="workflow",
                payload={"node": "prompt", "kind": "agent"},
            )

            tree = load_event_tree(run_path)
            graph = load_execution_graph(run_path, tree=tree, preset_root=preset_root)

        self.assertEqual(tree.by_id["prompt-call"].status, "error")
        self.assertEqual(graph.by_id["prompt"].status, "error")
        self.assertEqual(graph.by_id["prompt"].reason, "model fake returned an empty response")


if __name__ == "__main__":
    unittest.main()
