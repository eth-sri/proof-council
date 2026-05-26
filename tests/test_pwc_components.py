from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.pwc.workspace import (  # noqa: E402
    PWCStashAnswer,
    PWCWorkspaceInit,
    PWCWorkspaceSnapshot,
    pwc_workspace_path,
)
from proofstack.context import RunContext  # noqa: E402
from proofstack.registry import load_preset  # noqa: E402
from proofstack.sandbox.base import SandboxSpec  # noqa: E402
from proofstack.sandbox.docker import DockerSandbox  # noqa: E402
from app.dev_data import validate_preset_yaml  # noqa: E402


class PWCComponentTests(unittest.TestCase):
    def test_workspace_init_creates_canonical_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(run_id="test", root_workdir=temp_dir, flat=True)
            out = asyncio.run(
                PWCWorkspaceInit(ctx)(problem="Prove X.", problem_id="pwc test/1")
            )

            self.assertTrue((out.workspace / "answer.tex").exists())
            self.assertTrue((out.workspace / "research_notes.tex").exists())
            self.assertTrue((out.workspace / "references.bib").exists())
            self.assertTrue((out.workspace / "problem.txt").exists())

    def test_worker_component_docker_sandbox_disables_no_new_privileges(self) -> None:
        round_preset = load_preset("pwc_round")
        spec = SandboxSpec(**round_preset.component_configs["cfg_pwc_worker"]["sandbox"])
        with tempfile.TemporaryDirectory() as temp_dir:
            sandbox = DockerSandbox(spec, root=Path(temp_dir))
            cmd = sandbox._build_docker_cmd(  # type: ignore[attr-defined]
                ["codex", "--version"],
                env_extra={},
                extra_path=[],
                cwd=None,
                interactive=False,
                container_name="proofstack-test",
            )

            self.assertIn("--cap-drop", cmd)
            self.assertNotIn("no-new-privileges", cmd)

    def test_workspace_snapshot_accepts_missing_prior_notes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(run_id="test", root_workdir=temp_dir, flat=True)
            workspace = asyncio.run(PWCWorkspaceInit(ctx)(problem="Prove X.", problem_id="p1")).workspace

            PWCWorkspaceSnapshot.Inputs(
                problem_id="p1",
                round=0,
                plan_md=None,
                review_md=None,
                workspace=workspace,
                status=None,
                diff_summary=None,
            )

    def test_stash_answer_embeds_bbl_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(run_id="test", root_workdir=temp_dir, flat=True)
            compile_tex = ctx.root_workdir / "agents" / "compile" / "main.tex"
            compile_tex.parent.mkdir(parents=True)
            compile_tex.write_text("placeholder", encoding="utf-8")
            compile_tex.with_suffix(".bbl").write_text(
                "\\begin{thebibliography}{1}\\bibitem{x} X.\\end{thebibliography}",
                encoding="utf-8",
            )
            bib = ctx.root_workdir / "references.bib"
            bib.write_text("@article{x, title={X}}", encoding="utf-8")
            workspace = asyncio.run(PWCWorkspaceInit(ctx)(problem="P", problem_id="p1")).workspace
            (workspace / "references.bib").write_text(bib.read_text(encoding="utf-8"), encoding="utf-8")

            out = asyncio.run(
                PWCStashAnswer(ctx)(
                    problem_id="p1",
                    tex=(
                        "\\documentclass{article}\\begin{document}"
                        "\\cite{x}\\bibliographystyle{plain}\\bibliography{references}"
                        "\\end{document}"
                    ),
                    compile_tex_path=compile_tex,
                    workspace=workspace,
                )
            )

            final_tex = out.answer_tex.read_text(encoding="utf-8")
            self.assertIn("\\begin{thebibliography}", final_tex)
            self.assertNotIn("\\bibliography{references}", final_tex)

    def test_pwc_workflow_preset_loads_as_dag_workflow(self) -> None:
        preset = load_preset("pwc_workflow")
        round_preset = load_preset("pwc_round")

        self.assertEqual(preset.workflow_cls.__name__, "DAGWorkflow")
        self.assertIn("DAGWorkflow", preset.component_configs)
        self.assertIs(
            round_preset.component_configs["cfg_pwc_worker"]["sandbox"]["docker_no_new_privileges"],
            False,
        )

    def test_pwc_workflow_canvas_uses_workspace_state(self) -> None:
        report = validate_preset_yaml((ROOT / "configs/workflows/pwc_workflow.yaml").read_text())
        nodes = {node["id"]: node for node in report["nodes"]}
        outputs = sorted(report["workflow_outputs"])

        self.assertIn("workspace", nodes["init_workspace"]["output_fields"])
        self.assertEqual(nodes["pwc_loop"]["input_fields"], ["control", "workspace"])
        self.assertEqual(nodes["pwc_loop"]["output_fields"], ["control", "workspace"])
        self.assertNotIn("bundle", nodes["pwc_loop"]["input_fields"])
        self.assertNotIn("bundle", nodes["pwc_loop"]["output_fields"])
        self.assertEqual(
            outputs,
            ["compiled", "final_critic_findings", "pages", "pdf_path", "solution_tex", "tex_path"],
        )

    def test_pwc_round_continue_branch_does_not_prune_review_path(self) -> None:
        raw = yaml.safe_load((ROOT / "configs/workflows/pwc_round.yaml").read_text())
        nodes = {node["id"]: node for node in raw["dag"]["nodes"]}

        self.assertNotIn("worker_route", nodes)
        self.assertNotIn("early_stop", nodes)
        self.assertNotIn("coalesce", (ROOT / "configs/workflows/pwc_round.yaml").read_text())

        self.assertEqual(nodes["critic"]["when"]["inputs"]["status"], "$node.worker.status")

        self.assertEqual(nodes["snapshot"]["when"]["inputs"]["status"], "$node.worker.status")

    def test_pwc_round_worker_has_visible_workspace_input(self) -> None:
        raw_text = (ROOT / "configs/workflows/pwc_round.yaml").read_text()
        raw = yaml.safe_load(raw_text)
        report = validate_preset_yaml(raw_text)
        nodes = {node["id"]: node for node in report["nodes"]}
        raw_nodes = {node["id"]: node for node in raw["dag"]["nodes"]}

        self.assertTrue(report["ok"], report.get("errors"))
        self.assertIn("workspace", nodes["worker"]["input_fields"])
        self.assertEqual(raw_nodes["worker"]["inputs"]["workspace"], "$node.read_workspace.workspace")


if __name__ == "__main__":
    unittest.main()
