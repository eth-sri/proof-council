from __future__ import annotations

import asyncio
import json
import os
import stat
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

    def test_file_completion_contract_needs_no_shell_tool(self) -> None:
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
                                "printf '%s' '{\"status\":\"done\","
                                "\"summary\":\"written\"}' > done.json"
                            ),
                        ],
                        "contract": "auto",
                        "completion_signal": "file",
                        "prompt": "Task: {problem}",
                        "sandbox": {"backend": "subprocess"},
                        "input_schema": {"problem": "string"},
                        "output_schema": {
                            "workspace": "string",
                            "hint": "string",
                            "prompt_text": "string",
                            "status": "string",
                            "summary": "string",
                        },
                        "output_files": {
                            "hint": "hint.txt",
                            "prompt_text": "prompt.txt",
                        },
                        "done_outputs": {
                            "status": "status",
                            "summary": "summary",
                        },
                    }
                },
            )

            out = asyncio.run(ConfigurableCLIAgent(ctx, name="cfg_cli")(problem="P"))

            self.assertIn("Write this completion record as valid JSON to done.json", out.prompt_text)
            self.assertNotIn("shell command", out.prompt_text)
            self.assertNotIn("finish '", out.prompt_text)
            self.assertEqual(out.hint, "brief")
            self.assertEqual(out.status, "done")
            self.assertEqual(out.summary, "written")

    def test_file_completion_contract_uses_persistent_runtime_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["true"],
                        "workspace_root": "workspace",
                        "contract": "auto",
                        "completion_signal": "file",
                        "prompt": "Task.",
                        "output_schema": {"workspace": "string"},
                    }
                },
            )
            agent = ConfigurableCLIAgent(ctx, name="cfg_cli")
            sandbox = mock.Mock(root=(ctx.root_workdir / "workspace").resolve())

            async def render_tail() -> str:
                inp = agent.Inputs()
                await agent.setup(sandbox, inp)
                try:
                    return agent._contract_tail()
                finally:
                    await agent.teardown(sandbox, inp)

            tail = asyncio.run(render_tail())

            self.assertIn(".pwc/runtime/done.json", tail)

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

    def test_copy_codex_auth_refuses_to_run_without_host_auth(self) -> None:
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
                with self.assertRaisesRegex(RuntimeError, "Run `codex login` first"):
                    asyncio.run(ConfigurableCLIAgent(ctx, name="cfg_cli")())

    def test_copy_codex_auth_strips_openai_api_key_from_all_env_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["codex", "exec"],
                        "copy_codex_auth": True,
                        "env": {
                            "OPENAI_API_KEY": "{env:OPENAI_API_KEY}",
                            "CODEX_HOME": "/unverified/codex-home",
                        },
                        "sandbox": {
                            "backend": "subprocess",
                            "env_allowlist": ["PATH", "OPENAI_API_KEY", "CODEX_HOME"],
                            "extra_env": {
                                "OPENAI_API_KEY": "sandbox-key",
                                "CODEX_HOME": "/other/unverified-home",
                            },
                            "provider_keys": ["OPENAI_API_KEY", "GOOGLE_API_KEY"],
                        },
                    }
                },
            )
            agent = ConfigurableCLIAgent(ctx, name="cfg_cli")

            with mock.patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "must-not-leak",
                    "GOOGLE_API_KEY": "keep-for-mixed-tool-workflow",
                    "CODEX_HOME": "/parent/unverified-home",
                },
            ):
                env = agent.SANDBOX.build_env(sandbox_root=Path(temp_dir))
                verified_home = Path(temp_dir) / "verified-home"
                agent._codex_home = (verified_home, str(verified_home))
                extra_env = agent.extra_env(
                    mock.Mock(root=Path(temp_dir)),
                    agent.Inputs(),
                )

            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("OPENAI_API_KEY", extra_env)
            self.assertNotIn("CODEX_HOME", env)
            self.assertEqual(extra_env["CODEX_HOME"], str(verified_home))
            self.assertEqual(env["GOOGLE_API_KEY"], "keep-for-mixed-tool-workflow")

    def test_copy_codex_auth_mounts_private_home_for_docker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["codex", "exec"],
                        "copy_codex_auth": True,
                        "sandbox": {"backend": "docker"},
                    }
                },
            )
            agent = ConfigurableCLIAgent(ctx, name="cfg_cli")
            parent = agent._codex_auth_parent
            self.assertIsNotNone(parent)
            assert parent is not None
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(parent.stat().st_mode), 0o700)

            mount_pairs = list(
                zip(
                    agent.SANDBOX.docker_extra_args,
                    agent.SANDBOX.docker_extra_args[1:],
                )
            )
            self.assertIn(
                ("-v", f"{parent}:/proofstack-codex-home"),
                mount_pairs,
            )
            sandbox = mock.Mock()
            sandbox.spec = agent.SANDBOX
            with mock.patch(
                "proofstack.agents.configurable_cli.resolve_backend",
                return_value="docker",
            ):
                host_home, env_home = agent._new_codex_home(sandbox)

            self.assertEqual(host_home.parent, parent)
            self.assertEqual(
                env_home,
                f"/proofstack-codex-home/{host_home.name}",
            )
            agent._codex_auth_finalizer()
            self.assertFalse(parent.exists())

    def test_claude_subscription_usage_strips_anthropic_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["claude", "-p"],
                        "usage": {"type": "claude_json"},
                        "env": {"ANTHROPIC_API_KEY": "{env:ANTHROPIC_API_KEY}"},
                        "sandbox": {
                            "backend": "subprocess",
                            "env_allowlist": ["PATH", "ANTHROPIC_API_KEY"],
                            "extra_env": {"ANTHROPIC_API_KEY": "sandbox-key"},
                            "provider_keys": ["ANTHROPIC_API_KEY"],
                        },
                    }
                },
            )
            agent = ConfigurableCLIAgent(ctx, name="cfg_cli")

            with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "must-not-leak"}):
                env = agent.SANDBOX.build_env(sandbox_root=Path(temp_dir))
                extra_env = agent.extra_env(
                    mock.Mock(root=Path(temp_dir)),
                    agent.Inputs(),
                )

            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("ANTHROPIC_API_KEY", extra_env)

    def test_subscription_claude_nodes_are_write_only(self) -> None:
        nodes = [
            (load_preset("claude_subscription_min"), "claude_solver"),
            (load_preset("human_loop_demo"), "claude_attempt"),
            (load_preset("human_loop_demo"), "claude_finalize"),
        ]

        for preset, name in nodes:
            cfg = preset.component_configs[name]
            cmd = cfg["cmd"]
            self.assertEqual(cfg["sandbox"]["provider_keys"], [], name)
            self.assertEqual(cmd[cmd.index("--tools") + 1], "Write", name)
            self.assertIn("--safe-mode", cmd, name)
            self.assertIn("--strict-mcp-config", cmd, name)
            self.assertIn("--no-session-persistence", cmd, name)
            self.assertNotIn("--allowedTools", cmd, name)
            self.assertFalse(any("Bash" in str(part) for part in cmd), name)
            self.assertEqual(cfg["contract"], "auto", name)
            self.assertEqual(cfg["completion_signal"], "file", name)

    def test_copy_codex_auth_rejects_api_key_login(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir) / "home"
            auth = home / ".codex" / "auth.json"
            auth.parent.mkdir(parents=True)
            auth.write_text(
                json.dumps(
                    {
                        "auth_mode": "apikey",
                        "OPENAI_API_KEY": "paid-key",
                    }
                ),
                encoding="utf-8",
            )
            ctx = RunContext.create(
                run_id="test",
                root_workdir=Path(temp_dir) / "run",
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": ["codex", "exec"],
                        "copy_codex_auth": True,
                        "sandbox": {"backend": "subprocess"},
                    }
                },
            )

            with mock.patch.object(Path, "home", return_value=home):
                with self.assertRaisesRegex(RuntimeError, "not a ChatGPT login"):
                    asyncio.run(ConfigurableCLIAgent(ctx, name="cfg_cli")())

    def test_copy_codex_auth_accepts_chatgpt_login(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir) / "home"
            auth = home / ".codex" / "auth.json"
            auth.parent.mkdir(parents=True)
            auth.write_text(
                json.dumps(
                    {
                        "auth_mode": "chatgpt",
                        "OPENAI_API_KEY": None,
                        "tokens": {"access_token": "subscription-token"},
                    }
                ),
                encoding="utf-8",
            )
            ctx = RunContext.create(
                run_id="test",
                root_workdir=Path(temp_dir) / "run",
                flat=True,
                component_configs={
                    "cfg_cli": {
                        "cmd": [
                            "sh",
                            "-c",
                            "test -f \"$CODEX_HOME/auth.json\" && "
                            "printf '%s' \"$CODEX_HOME\" > codex_home.txt",
                        ],
                        "copy_codex_auth": True,
                        "sandbox": {"backend": "subprocess"},
                        "output_schema": {
                            "workspace": "string",
                            "codex_home": "string",
                        },
                        "output_files": {"codex_home": "codex_home.txt"},
                    }
                },
            )

            with mock.patch.object(Path, "home", return_value=home):
                out = asyncio.run(ConfigurableCLIAgent(ctx, name="cfg_cli")())

            codex_home = Path(out.codex_home)
            workspace = Path(out.workspace).resolve()
            self.assertNotEqual(codex_home.parent, workspace)
            self.assertNotIn(workspace, codex_home.parents)
            self.assertFalse((workspace / ".codex-home").exists())
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
