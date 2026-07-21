"""Executor-switching battery.

Executor switching is deliberately limited to Claude CLI <-> Codex CLI. Both
run the same ConfigurableCLIAgent and read outputs identically (files +
done.json), so the swap only exchanges the command / auth / usage scaffold and
touches none of the component's task or output configuration — that is what
makes it safe, where general API/human/CLI translation was a combinatorial
correctness hazard (kept, for a future rework, on feat/executor-switching).
Every other executor change is refused. This battery pins both.
"""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from proofstack.context import RunContext  # noqa: E402
from proofstack.registry import PresetError  # noqa: E402
from proofstack.agents.configurable_prompt import ConfigurablePromptAgent  # noqa: E402

from app.dev_data import (  # noqa: E402
    CONFIGURABLE_CLI_AGENT,
    CONFIGURABLE_PROMPT_AGENT,
    HUMAN_AGENT,
    _infer_executor,
    _op_set_executor,
    mutate_preset_yaml,
    validate_preset_yaml,
)


def _claude_cli_cfg() -> dict[str, Any]:
    return {
        "cmd": ["claude", "-p", "--model", "{claude_model}", "--permission-mode", "acceptEdits"],
        "env": {"HOME": "{env:HOME}"},
        "usage": {"type": "claude_json"},
        "contract": "auto",
        "soft_timeout_s": 780,
        "sandbox": {"backend": "subprocess", "timeout_s": 900},
        "prompt": "Solve the problem: {problem}",
        "input_schema": {"problem": "string", "workspace": "string"},
        "input_files": {"context.txt": {"from_input": "context"}},
        "output_schema": {"workspace": "string", "status": "string",
                          "answer_tex": "string", "notes": "string"},
        "output_files": {"answer_tex": {"path": "answer.tex", "type": "text"},
                         "notes": "notes.md"},
        "done_outputs": {"status": "status", "summary": "summary"},
    }


def _raw_for(cfg: dict[str, Any], agent: str = CONFIGURABLE_CLI_AGENT) -> dict[str, Any]:
    return {
        "components": {"cfg": copy.deepcopy(cfg)},
        "dag": {"nodes": [{"id": "n", "kind": "agent", "agent": agent, "name": "cfg"}]},
    }


def _swap(raw: dict[str, Any], executor: str) -> None:
    _op_set_executor(raw, {"op": "set_executor", "name": "cfg", "executor": executor})


def _dump(raw: dict[str, Any]) -> str:
    return yaml.safe_dump(raw, sort_keys=False)


# fields the swap must leave byte-for-byte identical (task + output config)
_PRESERVED = ("prompt", "input_schema", "input_files", "output_schema",
              "output_files", "done_outputs")


class CliSwitchTests(unittest.TestCase):
    def test_claude_to_codex_swaps_command_preserves_everything_else(self) -> None:
        raw = _raw_for(_claude_cli_cfg())
        before = copy.deepcopy(raw["components"]["cfg"])
        _swap(raw, "codex_cli")
        cfg = raw["components"]["cfg"]
        # command/auth/usage now Codex
        self.assertEqual(cfg["cmd"][0], "codex")
        self.assertIs(cfg["copy_codex_auth"], True)
        self.assertEqual(cfg["usage"]["type"], "codex_jsonl")
        self.assertNotIn("env", cfg)  # codex auth travels via CODEX_HOME
        # task + every output binding untouched
        for key in _PRESERVED:
            self.assertEqual(cfg.get(key), before.get(key), key)
        # node still runs the CLI agent
        self.assertEqual(raw["dag"]["nodes"][0]["agent"], CONFIGURABLE_CLI_AGENT)

    def test_codex_to_claude_reverse(self) -> None:
        raw = _raw_for(_claude_cli_cfg())
        _swap(raw, "codex_cli")
        outputs_mid = copy.deepcopy(raw["components"]["cfg"]["output_files"])
        _swap(raw, "claude_cli")
        cfg = raw["components"]["cfg"]
        self.assertEqual(cfg["cmd"][0], "claude")
        self.assertEqual(cfg["usage"]["type"], "claude_json")
        self.assertNotIn("copy_codex_auth", cfg)
        self.assertEqual(cfg["output_files"], outputs_mid)

    def test_round_trip_preserves_task_and_outputs(self) -> None:
        raw = _raw_for(_claude_cli_cfg())
        before = copy.deepcopy(raw["components"]["cfg"])
        _swap(raw, "codex_cli")
        _swap(raw, "claude_cli")
        cfg = raw["components"]["cfg"]
        for key in _PRESERVED:
            self.assertEqual(cfg.get(key), before.get(key), key)

    def test_switch_to_same_cli_is_a_noop(self) -> None:
        cfg = _claude_cli_cfg()
        cfg["cmd"] = ["claude", "-p", "--model", "opus"]
        raw = _raw_for(cfg)
        _swap(raw, "claude_cli")
        self.assertEqual(raw["components"]["cfg"]["cmd"], ["claude", "-p", "--model", "opus"])

    def test_string_form_codex_cmd_is_detected_as_codex(self) -> None:
        cfg = _claude_cli_cfg()
        cfg.update({"cmd": "codex exec --json", "copy_codex_auth": True})
        cfg.pop("env", None)
        self.assertEqual(_infer_executor(cfg, CONFIGURABLE_CLI_AGENT), "codex_cli")

    def test_switchable_component_validates_after_swap(self) -> None:
        raw = {
            "workflow": "proofstack.agents.dag_workflow.DAGWorkflow",
            "inputs": {"problem": ""},
            "components": {"cfg": _claude_cli_cfg()},
            "dag": {"nodes": [{"id": "n", "kind": "agent", "agent": CONFIGURABLE_CLI_AGENT,
                               "name": "cfg", "inputs": {"problem": "$input.problem"}}],
                    "outputs": {"answer_tex": "$node.n.answer_tex"}},
        }
        _swap(raw, "codex_cli")
        self.assertTrue(validate_preset_yaml(_dump(raw))["ok"])


class RefusalTests(unittest.TestCase):
    def _refuse(self, agent, target, pattern="disabled"):
        raw = _raw_for(_claude_cli_cfg() if agent == CONFIGURABLE_CLI_AGENT
                       else {"user_prompt": "x", "output": {"default_field": "a"}}, agent)
        with self.assertRaisesRegex(PresetError, pattern):
            _swap(raw, target)

    def test_cli_to_api_refused(self):
        self._refuse(CONFIGURABLE_CLI_AGENT, "api")

    def test_cli_to_human_refused(self):
        self._refuse(CONFIGURABLE_CLI_AGENT, "human")

    def test_api_to_cli_refused(self):
        self._refuse(CONFIGURABLE_PROMPT_AGENT, "codex_cli")

    def test_human_to_cli_refused(self):
        raw = _raw_for({"prompt": "x", "output_schema": {"a": "string"}}, HUMAN_AGENT)
        with self.assertRaisesRegex(PresetError, "disabled"):
            _swap(raw, "codex_cli")

    def test_custom_agent_refused(self):
        raw = {"components": {"cfg": {"prompt": "x"}},
               "dag": {"nodes": [{"id": "a", "kind": "agent",
                                  "agent": "proofstack.agents.ac.ACAuthorBlock", "name": "cfg"}]}}
        with self.assertRaisesRegex(PresetError, "custom-coded"):
            _op_set_executor(raw, {"name": "cfg", "executor": "codex_cli"})

    def test_unreferenced_component_refused(self):
        raw = {"components": {"Author": {"model": "m"}},
               "dag": {"nodes": [{"id": "a", "kind": "agent",
                                  "agent": "proofstack.agents.ac.ACAuthorBlock"}]}}
        with self.assertRaisesRegex(PresetError, "not used by any swappable node"):
            _op_set_executor(raw, {"name": "Author", "executor": "codex_cli"})

    def test_mixed_executor_nodes_refused(self):
        raw = {"components": {"cfg": _claude_cli_cfg()},
               "dag": {"nodes": [
                   {"id": "a", "kind": "agent", "agent": CONFIGURABLE_CLI_AGENT, "name": "cfg"},
                   {"id": "b", "kind": "agent", "agent": CONFIGURABLE_PROMPT_AGENT, "name": "cfg"}]}}
        with self.assertRaisesRegex(PresetError, "different executors"):
            _op_set_executor(raw, {"name": "cfg", "executor": "codex_cli"})

    def test_implicit_kind_node_is_retargeted(self):
        raw = {"components": {"cfg": _claude_cli_cfg()},
               "dag": {"nodes": [{"id": "n", "agent": CONFIGURABLE_CLI_AGENT, "name": "cfg"}]}}
        _swap(raw, "codex_cli")  # implicit kind still counts as an agent node
        self.assertEqual(raw["dag"]["nodes"][0]["agent"], CONFIGURABLE_CLI_AGENT)


class SchemaEditProjectionTests(unittest.TestCase):
    """Ordinary output_schema edits must MERGE the delivery config: preserve
    custom file specs, honor done_outputs coverage, prune removed fields, and
    only add defaults for genuinely new, uncovered fields."""

    def _cfg(self):
        return {
            "cmd": ["claude", "-p"], "contract": "auto",
            "prompt": "Review.",
            "input_schema": {"solution": "string", "workspace": "string"},
            "output_schema": {"workspace": "string", "status": "string",
                              "summary": "string", "answer_tex": "string", "notes": "string"},
            "output_files": {
                "answer_tex": {"path": "deliverables/final-answer.tex", "type": "text", "default": "NO ANSWER"},
                "notes": {"path": "scratch/reviewer-notes.md", "type": "text", "default": "NO NOTES"},
            },
            "done_outputs": {"status": "status", "summary": "summary"},
        }

    def test_adding_a_field_preserves_custom_specs_and_done_coverage(self):
        raw = _raw_for(self._cfg())
        res = mutate_preset_yaml(_dump(raw), {"op": "update_component", "name": "cfg",
            "fields": {"output_schema":
                "workspace: string\nstatus: string\nsummary: string\n"
                "answer_tex: string\nnotes: string\ncritique: string"}})
        files = yaml.safe_load(res["raw_yaml"])["components"]["cfg"]["output_files"]
        # custom dict specs untouched
        self.assertEqual(files["answer_tex"]["path"], "deliverables/final-answer.tex")
        self.assertEqual(files["notes"]["default"], "NO NOTES")
        # done-supplied summary must NOT get a shadowing file
        self.assertNotIn("summary", files)
        # only the genuinely new field gets a default file
        self.assertEqual(files["critique"], "critique.txt")

    def test_removing_a_field_prunes_only_that_field(self):
        raw = _raw_for(self._cfg())
        res = mutate_preset_yaml(_dump(raw), {"op": "update_component", "name": "cfg",
            "fields": {"output_schema":
                "workspace: string\nstatus: string\nsummary: string\nanswer_tex: string"}})
        cfg = yaml.safe_load(res["raw_yaml"])["components"]["cfg"]
        self.assertNotIn("notes", cfg["output_files"])
        self.assertEqual(cfg["output_files"]["answer_tex"]["path"], "deliverables/final-answer.tex")
        self.assertEqual(cfg["done_outputs"], {"status": "status", "summary": "summary"})


class UsageCarryTests(unittest.TestCase):
    def test_codex_metadata_dropped_on_swap_to_claude(self):
        cfg = _claude_cli_cfg()
        cfg.update({"cmd": ["codex", "exec"], "copy_codex_auth": True,
                    "usage": {"type": "codex_jsonl", "bill": False,
                              "cost_config": "models/openai/gpt-54-mini", "model": "gpt-x"}})
        cfg.pop("env", None)
        raw = _raw_for(cfg)
        _swap(raw, "claude_cli")
        self.assertEqual(raw["components"]["cfg"]["usage"], {"type": "claude_json"})

    def test_subscription_intent_survives_round_trip(self):
        raw = _raw_for(_claude_cli_cfg())  # usage: {type: claude_json}
        _swap(raw, "codex_cli")
        self.assertEqual(raw["components"]["cfg"]["usage"],
                         {"type": "codex_jsonl", "bill": False})
        _swap(raw, "claude_cli")
        self.assertEqual(raw["components"]["cfg"]["usage"], {"type": "claude_json"})


class AdjacentFixTests(unittest.TestCase):
    """Non-switch fixes made alongside the descope."""

    def _render(self, cfg, **inp):
        with tempfile.TemporaryDirectory() as d:
            ctx = RunContext.create(run_id="r", root_workdir=d, flat=True, component_configs={"cfg": cfg})
            a = ConfigurablePromptAgent(ctx, name="cfg")
            return a.render_messages(a.Inputs(**inp))

    def test_short_tag_math_still_gets_format_instruction(self):
        cfg = {"model": "m", "user_prompt": "For x<n compute n and a proof.",
               "output": {"xml_tags": ["n", "proof"], "default_field": "n"}}
        text = "\n".join(str(m["content"]) for m in self._render(cfg))
        self.assertIn("<n>", text)
        self.assertIn("<proof>", text)

    def test_assistant_prefill_gets_no_trailing_user_turn(self):
        cfg = {"model": "m",
               "messages": [{"role": "user", "content": "Solve."}, {"role": "assistant", "content": "Sure,"}],
               "output": {"xml_tags": ["a", "b"], "default_field": "a"}}
        msgs = self._render(cfg)
        self.assertEqual(msgs[-1]["role"], "assistant")

    def test_conditional_repeat_preset_validates(self):
        path = ROOT / "configs" / "workflows" / "conditional_repeat_screenshot.yaml"
        self.assertTrue(validate_preset_yaml(path.read_text())["ok"])

    def test_dispute_percent_is_escaped(self):
        from app.dev import _surface_dispute_markers
        out = _surface_dispute_markers("% >>> DISPUTE: confidence is only 50%\n")
        self.assertIn(r"50\%", out)

    def test_rename_updates_map_chain_step_refs(self):
        raw = {"components": {"cfg": {"user_prompt": "x", "output_schema": {"old": "string"},
                                     "output": {"default_field": "old"}}},
               "dag": {"nodes": [
                   {"id": "fan", "kind": "map_chain", "foreach": "$input.items",
                    "steps": [{"id": "s1", "agent": CONFIGURABLE_PROMPT_AGENT, "name": "cfg", "inputs": {}}]},
                   {"id": "join", "kind": "join_or_agent", "source": "$node.fan.finals",
                    "inputs": {"value": "$step.s1.old"}}]}}
        res = mutate_preset_yaml(_dump(raw),
            {"op": "update_component", "name": "cfg", "fields": {"__rename_output_refs": {"old": "new"}}})
        joined = yaml.safe_load(res["raw_yaml"])["dag"]["nodes"][1]["inputs"]["value"]
        self.assertEqual(joined, "$step.s1.new")


if __name__ == "__main__":
    unittest.main()
