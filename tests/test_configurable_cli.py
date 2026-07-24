from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.configurable_cli import ConfigurableCLIAgent  # noqa: E402
from proofstack.context import RunContext  # noqa: E402
from proofstack.registry import load_preset  # noqa: E402
from app.dev_data import validate_preset_yaml  # noqa: E402


class ConfigurableCLITests(unittest.TestCase):
    def test_cli_agent_collects_file_and_done_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": [
                            "sh",
                            "-c",
                            (
                                "cat > prompt.txt; "
                                "printf 'generated proof' > answer.tex; "
                                "finish '{\"status\":\"done\",\"summary\":\"ok\","
                                "\"open_questions\":[\"q\"]}'"
                            ),
                        ],
                        "prompt": "Problem: {problem}",
                        "sandbox": {"backend": "subprocess"},
                        "input_schema": {"problem": "string"},
                        "output_schema": {
                            "workspace": "string",
                            "answer_tex": "string",
                            "prompt_text": "string",
                            "answer_tex_path": "string",
                            "status": "string",
                            "summary": "string",
                            "open_questions": {"type": "array", "items": {}},
                        },
                        "output_files": {
                            "answer_tex": "answer.tex",
                            "prompt_text": "prompt.txt",
                            "answer_tex_path": {"path": "answer.tex", "type": "path"},
                        },
                        "done_outputs": {
                            "status": "status",
                            "summary": "summary",
                            "open_questions": "open_questions",
                        },
                    }
                },
            )

            out = asyncio.run(ConfigurableCLIAgent(ctx, name="cfg_cli")(problem="P"))

            self.assertEqual(out.answer_tex, "generated proof")
            self.assertEqual(out.prompt_text, "Problem: P\n")
            self.assertTrue(Path(out.answer_tex_path).exists())
            self.assertEqual(out.status, "done")
            self.assertEqual(out.summary, "ok")
            self.assertEqual(out.open_questions, ["q"])
            self.assertTrue(Path(out.workspace).exists())

    def test_contract_auto_appends_output_contract_to_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": [
                            "sh",
                            "-c",
                            (
                                "cat > prompt.txt; "
                                "printf 'brief' > hint.txt; "
                                "finish '{\"status\":\"done\",\"summary\":\"ok\"}'"
                            ),
                        ],
                        "contract": "auto",
                        "prompt": "You are the specialist. Problem: {problem}",
                        "sandbox": {"backend": "subprocess"},
                        "input_schema": {"problem": "string"},
                        "output_schema": {
                            "workspace": "string",
                            "hint": "string",
                            "prompt_text": "string",
                            "hint_path": "string",
                            "status": "string",
                        },
                        "output_files": {
                            "hint": "hint.txt",
                            "prompt_text": "prompt.txt",
                            "hint_path": {"path": "hint.txt", "type": "path"},
                        },
                        "done_outputs": {"status": "status"},
                    }
                },
            )

            out = asyncio.run(ConfigurableCLIAgent(ctx, name="cfg_cli")(problem="P"))

            piped = out.prompt_text
            self.assertIn("You are the specialist. Problem: P", piped)
            self.assertIn("HOW TO DELIVER YOUR OUTPUT", piped)
            self.assertIn("hint.txt", piped)
            self.assertIn('finish \'{"status":"done","summary":', piped)
            # Passively-collected kinds (path/exists/listing) are not part of
            # what the model must write, so they are not listed.
            self.assertNotIn("hint_path", piped)
            self.assertEqual(out.hint, "brief")
            self.assertEqual(out.status, "done")

    def test_contract_auto_requests_configured_done_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["true"],
                        "contract": "auto",
                        "prompt": "Task.",
                        "output_schema": {
                            "workspace": "string",
                            "status": "string",
                            "notes": "string",
                            "questions": "string",
                        },
                        "output_files": {"notes": "notes.md"},
                        "done_outputs": {
                            "status": "status",
                            "questions": {"field": "open_questions", "join": True},
                            "changes": "diff_summary",
                        },
                    }
                },
            )
            tail = ConfigurableCLIAgent(ctx, name="cfg_cli")._contract_tail()

            # the finish example must ask for every configured done field, or
            # the model never supplies it and the output silently defaults
            self.assertIn('"open_questions"', tail)
            self.assertIn('"diff_summary"', tail)
            self.assertIn('"status":"done"', tail)
            self.assertNotIn('"artifacts"', tail)  # not configured

    def test_contract_auto_defaults_to_status_summary_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["true"],
                        "contract": "auto",
                        "prompt": "Task.",
                        "output_schema": {"workspace": "string", "answer_tex": "string"},
                        "output_files": {"answer_tex": "answer.tex"},
                        "done_outputs": {"status": "status"},
                    }
                },
            )
            tail = ConfigurableCLIAgent(ctx, name="cfg_cli")._contract_tail()

            self.assertIn('"status":"done"', tail)
            self.assertIn('"summary"', tail)
            for absent in ('"open_questions"', '"diff_summary"', '"artifacts"'):
                self.assertNotIn(absent, tail)

    def test_prompt_unchanged_without_contract_auto(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["sh", "-c", "cat > prompt.txt"],
                        "prompt": "Problem: {problem}",
                        "sandbox": {"backend": "subprocess"},
                        "input_schema": {"problem": "string"},
                        "output_schema": {"workspace": "string", "prompt_text": "string"},
                        "output_files": {"prompt_text": "prompt.txt"},
                    }
                },
            )

            out = asyncio.run(ConfigurableCLIAgent(ctx, name="cfg_cli")(problem="P"))

            self.assertEqual(out.prompt_text, "Problem: P\n")

    def test_cli_agent_uses_configured_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["sh", "-c", "printf ok > marker.txt"],
                        "prompt": "{problem_id}",
                        "workspace_root": "workspaces/{problem_id}",
                        "sandbox": {"backend": "subprocess"},
                        "input_schema": {"problem_id": "string"},
                    }
                },
            )

            out = asyncio.run(ConfigurableCLIAgent(ctx, name="cfg_cli")(problem_id="abc"))

            self.assertEqual(
                Path(out.workspace).resolve(),
                (ctx.root_workdir / "workspaces" / "abc").resolve(),
            )
            self.assertTrue((Path(out.workspace) / "marker.txt").exists())

    def test_cli_agent_schema_infers_configured_inputs_and_outputs(self) -> None:
        report = validate_preset_yaml(
            """
            workflow: proofstack.agents.dag_workflow.DAGWorkflow
            inputs:
              problem: ''
            components:
              cfg_cli:
                cmd: [sh, -c, "printf ok > answer.tex"]
                prompt: "Problem: {problem}"
                input_schema:
                  problem: string
                output_schema:
                  workspace: string
                  answer_tex: string
                  status: string
                output_files:
                  answer_tex: answer.tex
                done_outputs:
                  status: status
            dag:
              nodes:
                - id: cli
                  kind: agent
                  agent: proofstack.agents.configurable_cli.ConfigurableCLIAgent
                  name: cfg_cli
              outputs:
                answer_tex: $node.cli.answer_tex
            """
        )

        nodes = {node["id"]: node for node in report["nodes"]}
        self.assertTrue(report["ok"], report.get("errors"))
        self.assertEqual(nodes["cli"]["output_fields"], ["answer_tex", "status", "workspace"])

    def test_codex_exec_reads_configured_prompt_from_stdin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": [
                            "codex",
                            "-c",
                            'model_reasoning_effort="low"',
                            "exec",
                            "-m",
                            "gpt-5.4-mini",
                            "--json",
                        ],
                        "codex_sandbox": "auto",
                        "prompt": "Problem: {problem}",
                        "input_schema": {"problem": "string"},
                    }
                },
            )
            agent = ConfigurableCLIAgent(ctx, name="cfg_cli")

            cmd = agent._command_for(agent.Inputs(problem="P"))

            self.assertEqual(cmd[-2:], ["--dangerously-bypass-approvals-and-sandbox", "-"])

    def test_codex_exec_model_and_reasoning_effort_are_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": [
                            "codex",
                            "-c",
                            'model_reasoning_effort="low"',
                            "exec",
                            "-m",
                            "gpt-5.4-mini",
                            "--json",
                        ],
                        "model": "gpt-5.5-pro",
                        "model_reasoning_effort": "xhigh",
                        "prompt": "Problem: {problem}",
                        "input_schema": {"problem": "string"},
                    }
                },
            )
            agent = ConfigurableCLIAgent(ctx, name="cfg_cli")

            cmd = agent._command_for(agent.Inputs(problem="P"))

            self.assertIn("-m", cmd)
            self.assertEqual(cmd[cmd.index("-m") + 1], "gpt-5.5-pro")
            self.assertNotIn("gpt-5.4-mini", cmd)
            self.assertIn('model_reasoning_effort="xhigh"', cmd)
            self.assertNotIn('model_reasoning_effort="low"', cmd)

    def test_claude_node_model_override_rewrites_cmd_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["claude", "-p", "--model", "{claude_model}", "--permission-mode", "acceptEdits"],
                        "model": "opus",  # per-node override
                        "prompt": "Problem: {problem}",
                        "input_schema": {"problem": "string", "claude_model": "string"},
                    }
                },
            )
            agent = ConfigurableCLIAgent(ctx, name="cfg_cli")

            cmd = agent._command_for(agent.Inputs(problem="P", claude_model="sonnet"))

            self.assertIn("--model", cmd)
            self.assertEqual(cmd[cmd.index("--model") + 1], "opus")
            self.assertNotIn("sonnet", cmd)  # the override beats the workflow default

    def test_claude_node_reasoning_effort_sets_effort_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["claude", "-p", "--model", "sonnet"],
                        "model_reasoning_effort": "low",
                        "prompt": "Problem: {problem}",
                        "input_schema": {"problem": "string"},
                    }
                },
            )
            agent = ConfigurableCLIAgent(ctx, name="cfg_cli")

            cmd = agent._command_for(agent.Inputs(problem="P"))

            self.assertIn("--effort", cmd)
            self.assertEqual(cmd[cmd.index("--effort") + 1], "low")

    def test_claude_effort_maps_codex_minimal_and_rewrites_existing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["claude", "-p", "--effort", "high"],
                        "model_reasoning_effort": "minimal",  # codex vocab
                        "prompt": "Problem: {problem}",
                        "input_schema": {"problem": "string"},
                    }
                },
            )
            agent = ConfigurableCLIAgent(ctx, name="cfg_cli")

            cmd = agent._command_for(agent.Inputs(problem="P"))

            self.assertEqual(cmd.count("--effort"), 1)
            self.assertEqual(cmd[cmd.index("--effort") + 1], "low")

    def test_claude_node_without_override_uses_workflow_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["claude", "-p", "--model", "{claude_model}"],
                        "prompt": "Problem: {problem}",
                        "input_schema": {"problem": "string", "claude_model": "string"},
                    }
                },
            )
            agent = ConfigurableCLIAgent(ctx, name="cfg_cli")

            cmd = agent._command_for(agent.Inputs(problem="P", claude_model="haiku"))

            self.assertEqual(cmd[cmd.index("--model") + 1], "haiku")

    def test_compile_codex_docker_sandbox_allows_node_exec(self) -> None:
        preset = load_preset(str(ROOT / "tests" / "fixtures" / "compile_cli_agent.yaml"))
        sandbox = preset.component_configs["cfg_compile_latex"]["sandbox"]

        self.assertIs(sandbox["docker_no_new_privileges"], False)

    def test_copy_codex_auth_creates_and_scrubs_codex_home_without_host_auth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": [
                            "sh",
                            "-c",
                            "test -d \"$CODEX_HOME\" && printf '%s' \"$CODEX_HOME\" > codex_home.txt",
                        ],
                        "copy_codex_auth": True,
                        "sandbox": {"backend": "subprocess"},
                        "output_schema": {
                            "workspace": "string",
                            "codex_home": "string",
                        },
                        "output_files": {
                            "codex_home": "codex_home.txt",
                        },
                    }
                },
            )

            with mock.patch.object(Path, "home", return_value=Path(temp_dir) / "no-home-auth"):
                out = asyncio.run(ConfigurableCLIAgent(ctx, name="cfg_cli")())
            codex_home = Path(out.codex_home)

            self.assertEqual(codex_home, Path(out.workspace) / ".codex-home")
            self.assertFalse(codex_home.exists())

    def test_env_template_can_use_parent_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["sh", "-c", "printf '%s' \"$HOME\" > home.txt"],
                        "env": {"HOME": "{env:PROOFSTACK_TEST_HOME}"},
                        "sandbox": {"backend": "subprocess"},
                        "output_schema": {
                            "workspace": "string",
                            "home": "string",
                        },
                        "output_files": {"home": "home.txt"},
                    }
                },
            )

            with mock.patch.dict(os.environ, {"PROOFSTACK_TEST_HOME": "/portable/home"}):
                out = asyncio.run(ConfigurableCLIAgent(ctx, name="cfg_cli")())

            self.assertEqual(out.home, "/portable/home")


if __name__ == "__main__":
    unittest.main()
