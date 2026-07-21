"""Executor-switching battery for the narrow, refuse-when-unsure design.

Switching is deliberately limited to a small, fully-testable tier: components
whose every output is plain text and whose prompt is delivery-neutral. Within
that tier, contract and parse behavior survive any swap sequence. Everything
richer is a clean PresetError, never a silent mangle — four review rounds of
silent mangling is why. This battery pins both halves: round-trip fidelity for
the supported tier, and specific refusals for everything else.
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
    _absorbed_output_contract,
    _op_set_executor,
    _OUTPUT_MECHANICS_FIELDS,
    mutate_preset_yaml,
    validate_preset_yaml,
)

EXECUTORS = ("api", "claude_cli", "codex_cli", "human")
_HOME_AGENTS = {
    "api": CONFIGURABLE_PROMPT_AGENT,
    "claude_cli": CONFIGURABLE_CLI_AGENT,
    "codex_cli": CONFIGURABLE_CLI_AGENT,
    "human": HUMAN_AGENT,
}

# Supported tier: delivery-neutral prompts, plain-text outputs only.
SHAPES: dict[str, tuple[str, dict[str, Any]]] = {
    "single_text": (
        "api",
        {
            "model": "models/openai/gpt-54-mini",
            "system_prompt": "You are an expert assistant.",
            "user_prompt": "Complete this node's task.",
            "input_schema": {"question": "string"},
            "output": {"default_field": "answer"},
        },
    ),
    "multi_text": (
        "api",
        {
            "model": "models/openai/gpt-54-mini",
            "system_prompt": "Review the proof.",
            "user_prompt": "Proof:\n{proof}",
            "input_schema": {"proof": "string"},
            "output_schema": {"verdict": "string", "notes": "string"},
            "output": {"xml_tags": ["verdict", "notes"], "default_field": "verdict"},
        },
    ),
    "cli_text": (
        "claude_cli",
        {
            "prompt": "Solve the problem and report your result.",
            "cmd": ["claude", "-p"],
            "contract": "auto",
            "sandbox": {"backend": "subprocess"},
            "input_schema": {"problem": "string", "workspace": "string"},
            "output_schema": {"workspace": "string", "status": "string", "answer": "string"},
            "output_files": {"answer": "answer.txt"},
            "done_outputs": {"status": "status"},
        },
    ),
    "human_text": (
        "human",
        {
            "prompt": "Please assess the proof.",
            "input_schema": {"proof": "string"},
            "output_schema": {"answer": "string", "concern": "string"},
        },
    ),
}


def _raw_for(cfg: dict[str, Any], home: str) -> dict[str, Any]:
    return {
        "components": {"cfg": copy.deepcopy(cfg)},
        "dag": {"nodes": [{"id": "n", "kind": "agent", "agent": _HOME_AGENTS[home], "name": "cfg"}]},
    }


def _swap(raw: dict[str, Any], executor: str) -> None:
    _op_set_executor(raw, {"op": "set_executor", "name": "cfg", "executor": executor})


def _parse(cfg: dict[str, Any], text: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        ctx = RunContext.create(run_id="r", root_workdir=d, flat=True, component_configs={"cfg": cfg})
        a = ConfigurablePromptAgent(ctx, name="cfg")
        return {k: v for k, v in a.parse_output(text, a.Inputs()).model_dump().items() if k != "raw_text"}


def _dump(raw: dict[str, Any]) -> str:
    return yaml.safe_dump(raw, sort_keys=False)


RESPONSES = {
    "single_text": ("A full answer.", {"answer": "A full answer."}),
    "multi_text": (
        "<verdict>correct</verdict><notes>solid</notes>",
        {"verdict": "correct", "notes": "solid"},
    ),
}


class SupportedTierTests(unittest.TestCase):
    def test_contract_survives_every_round_trip(self) -> None:
        for name, (home, cfg) in SHAPES.items():
            original = _absorbed_output_contract(cfg)
            for other in EXECUTORS:
                if other == home:
                    continue
                with self.subTest(shape=name, via=other):
                    raw = _raw_for(cfg, home)
                    _swap(raw, other)
                    _swap(raw, home)
                    got = _absorbed_output_contract(raw["components"]["cfg"])
                    self.assertEqual(set(got), set(original), f"{name} via {other}")
                    for f in _OUTPUT_MECHANICS_FIELDS:
                        self.assertNotIn(f, got)

    def test_long_chain_reaches_a_fixed_point(self) -> None:
        for name, (home, cfg) in SHAPES.items():
            with self.subTest(shape=name):
                original = set(_absorbed_output_contract(cfg))
                raw = _raw_for(cfg, home)
                for ex in ("codex_cli", "human", "api", "claude_cli", home):
                    _swap(raw, ex)
                self.assertEqual(
                    set(_absorbed_output_contract(raw["components"]["cfg"])), original
                )

    def test_parse_behavior_identical_after_round_trip(self) -> None:
        for name, (response, expected) in RESPONSES.items():
            home, cfg = SHAPES[name]
            self.assertEqual(_parse(cfg, response), expected, f"{name} baseline")
            for other in EXECUTORS:
                if other == home:
                    continue
                with self.subTest(shape=name, via=other):
                    raw = _raw_for(cfg, home)
                    _swap(raw, other)
                    _swap(raw, home)
                    self.assertEqual(
                        _parse(raw["components"]["cfg"], response), expected, f"{name} via {other}"
                    )

    def test_cli_projections_are_plain_text_files(self) -> None:
        home, cfg = SHAPES["multi_text"]
        raw = _raw_for(cfg, home)
        _swap(raw, "codex_cli")
        files = raw["components"]["cfg"]["output_files"]
        self.assertEqual(files, {"verdict": "verdict.txt", "notes": "notes.txt"})
        self.assertEqual(raw["components"]["cfg"]["done_outputs"], {"status": "status"})

    def test_summary_output_survives_a_swap(self) -> None:
        cfg = {"model": "m", "user_prompt": "Summarize.",
               "output_schema": {"summary": "string"}, "output": {"default_field": "summary"}}
        raw = _raw_for(cfg, "api")
        _swap(raw, "codex_cli"); _swap(raw, "api")
        self.assertIn("summary", _absorbed_output_contract(raw["components"]["cfg"]))

    def test_switch_to_same_executor_is_a_noop(self) -> None:
        cfg = {"model": "models/openai/gpt-54-mini", "model_reasoning_effort": "high",
               "user_prompt": "x", "output": {"default_field": "answer"}}
        raw = _raw_for(cfg, "api")
        _swap(raw, "api")
        c = raw["components"]["cfg"]
        self.assertEqual(c["model"], "models/openai/gpt-54-mini")
        self.assertEqual(c["model_reasoning_effort"], "high")

    def test_implicit_kind_agent_node_is_retargeted(self) -> None:
        raw = {"components": {"cfg": {"user_prompt": "x", "output": {"default_field": "a"}}},
               "dag": {"nodes": [
                   {"id": "e", "kind": "agent", "agent": CONFIGURABLE_PROMPT_AGENT, "name": "cfg"},
                   {"id": "i", "agent": CONFIGURABLE_PROMPT_AGENT, "name": "cfg"}]}}
        _swap(raw, "codex_cli")
        self.assertTrue(all(n["agent"] == CONFIGURABLE_CLI_AGENT for n in raw["dag"]["nodes"]))

    def test_all_string_enum_can_switch_to_human(self) -> None:
        cfg = {"model": "m", "user_prompt": "x",
               "output_schema": {"verdict": {"enum": ["yes", "no"]}}, "output": {"default_field": "verdict"}}
        _swap(_raw_for(cfg, "api"), "human")  # must not raise


class EditPathTests(unittest.TestCase):
    def test_output_schema_object_edit_is_not_corrupted(self) -> None:
        raw = _raw_for(SHAPES["cli_text"][1], "claude_cli")
        res = mutate_preset_yaml(_dump(raw), {"op": "update_component", "name": "cfg",
            "fields": {"output_schema": {"workspace": "string", "status": "string", "answer": "string"}}})
        sch = yaml.safe_load(res["raw_yaml"])["components"]["cfg"]["output_schema"]
        self.assertEqual(set(sch), {"workspace", "status", "answer"})
        self.assertTrue(all(v == "string" for v in sch.values()))

    def test_adding_a_scalar_output_keeps_it_bound(self) -> None:
        cfg = {"model": "m", "user_prompt": "x", "output_schema": {"notes": "string"},
               "output": {"default_field": "notes"}}
        res = mutate_preset_yaml(_dump(_raw_for(cfg, "api")), {"op": "update_component", "name": "cfg",
            "fields": {"output_schema": "notes: string\nscore: string"}})
        c = yaml.safe_load(res["raw_yaml"])["components"]["cfg"]
        self.assertEqual(set(c["output_schema"]), {"notes", "score"})
        self.assertIn("score", set(c["output"].get("xml_tags", [])))


class RefusalTests(unittest.TestCase):
    def _refuse(self, cfg, home, target, pattern):
        with self.assertRaisesRegex(PresetError, pattern):
            _swap(_raw_for(cfg, home), target)

    def test_structured_output_refused(self):
        cfg = {"model": "m", "user_prompt": "x",
               "output": {"xml_lists": {"items": "item"}, "default_field": "text"}}
        self._refuse(cfg, "api", "codex_cli", "non-text")

    def test_messages_refused(self):
        cfg = {"model": "m", "messages": [{"role": "user", "content": "x"}], "output": {"default_field": "a"}}
        self._refuse(cfg, "api", "codex_cli", "messages")

    def test_json_merge_refused(self):
        cfg = {"model": "m", "user_prompt": "x", "output_schema": {"a": "string"},
               "output": {"json_tag": "r", "json_merge": True}}
        self._refuse(cfg, "api", "codex_cli", "json_merge")

    def test_regex_fields_refused(self):
        cfg = {"model": "m", "user_prompt": "x", "output": {"regex_fields": {"v": r"V:\s*(\w+)"}, "default_field": "a"}}
        self._refuse(cfg, "api", "codex_cli", "regex_fields")

    def test_dotted_field_refused(self):
        cfg = {"prompt": "x", "cmd": ["codex", "exec"], "contract": "auto",
               "output_schema": {"review.summary": "string"}, "output_files": {"review.summary": "s.txt"}}
        self._refuse(cfg, "codex_cli", "api", "nested")

    def test_custom_collector_kind_refused(self):
        cfg = {"prompt": "x", "cmd": ["codex", "exec"], "contract": "auto",
               "output_schema": {"pdf": "string"}, "output_files": {"pdf": {"path": "main.pdf", "type": "path"}}}
        self._refuse(cfg, "codex_cli", "api", "path")

    def test_custom_agent_refused(self):
        raw = {"components": {"cfg": {"prompt": "x"}},
               "dag": {"nodes": [{"id": "a", "kind": "agent", "agent": "proofstack.agents.ac.ACAuthorBlock", "name": "cfg"}]}}
        with self.assertRaisesRegex(PresetError, "custom-coded"):
            _op_set_executor(raw, {"name": "cfg", "executor": "api"})

    def test_component_without_swappable_node_refused(self):
        raw = {"components": {"Author": {"model": "m"}},
               "dag": {"nodes": [{"id": "a", "kind": "agent", "agent": "proofstack.agents.ac.ACAuthorBlock"}]}}
        with self.assertRaisesRegex(PresetError, "not used by any swappable node"):
            _op_set_executor(raw, {"name": "Author", "executor": "api"})

    def test_entangled_prompt_refused(self):
        cfg = {"model": "m", "user_prompt": "Emit each item in an <item> tag.",
               "output": {"xml_tags": ["item"], "default_field": "item"}}
        self._refuse(cfg, "api", "codex_cli", "hand-embeds")

    def test_structured_output_masked_by_text_schema_refused(self):
        # schema says text, but the spec ALSO declares a list — must not slip
        cfg = {"model": "m", "user_prompt": "x", "output_schema": {"answer": "string"},
               "output": {"xml_lists": {"items": "item"}, "default_field": "answer"}}
        self._refuse(cfg, "api", "codex_cli", "non-text")

    def test_cli_constant_outputs_refused(self):
        cfg = {"prompt": "x", "cmd": ["codex", "exec"], "contract": "auto",
               "output_schema": {"answer": "string"}, "output_files": {"answer": "a.txt"},
               "constant_outputs": {"approved": False}}
        self._refuse(cfg, "codex_cli", "api", "constant_outputs")

    def test_dotted_field_in_api_spec_refused(self):
        cfg = {"model": "m", "user_prompt": "x", "output": {"default_field": "review.summary"}}
        self._refuse(cfg, "api", "codex_cli", "nested")

    def test_falsy_type_specs_refused(self):
        for bad in ({}, None, False, []):
            cfg = {"model": "m", "user_prompt": "x", "output_schema": {"f": bad},
                   "output": {"default_field": "f"}}
            with self.assertRaises(PresetError):
                _swap(_raw_for(cfg, "api"), "codex_cli")

    def test_enum_with_array_type_refused(self):
        cfg = {"model": "m", "user_prompt": "x",
               "output_schema": {"v": {"type": "array", "enum": ["a", "b"]}},
               "output": {"default_field": "v"}}
        self._refuse(cfg, "api", "codex_cli", "non-text")

    def test_mixed_executor_nodes_refused(self):
        raw = {"components": {"cfg": {"user_prompt": "x", "output": {"default_field": "a"}}},
               "dag": {"nodes": [
                   {"id": "p", "kind": "agent", "agent": CONFIGURABLE_PROMPT_AGENT, "name": "cfg"},
                   {"id": "c", "kind": "agent", "agent": CONFIGURABLE_CLI_AGENT, "name": "cfg"}]}}
        with self.assertRaisesRegex(PresetError, "different executors"):
            _op_set_executor(raw, {"name": "cfg", "executor": "api"})

    def test_contract_auto_prompt_writing_a_file_refused(self):
        cfg = {"prompt": "Write the final answer to notes.md.", "cmd": ["codex", "exec"],
               "contract": "auto", "output_schema": {"notes": "string"},
               "output_files": {"notes": "notes.md"}}
        self._refuse(cfg, "codex_cli", "api", "write")


class OutputRenameTests(unittest.TestCase):
    def test_rename_updates_map_chain_step_refs(self):
        raw = {
            "components": {"cfg": {"user_prompt": "x", "output_schema": {"old": "string"},
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


class RefusalFalsePositiveTests(unittest.TestCase):
    """A refusal that blocks a legitimate switch is a real defect."""

    def test_math_prompt_with_short_tag_not_flagged(self):
        cfg = {"model": "m", "user_prompt": "For x<n, return the value.",
               "output_schema": {"n": "string"}, "output": {"default_field": "n"}}
        _swap(_raw_for(cfg, "api"), "codex_cli")  # must not raise

    def test_prefix_tag_not_flagged(self):
        cfg = {"model": "m", "user_prompt": "Discuss the <answer_tex> macro.",
               "output_schema": {"answer": "string"}, "output": {"default_field": "answer"}}
        _swap(_raw_for(cfg, "api"), "codex_cli")  # must not raise

    def test_filename_as_example_not_flagged(self):
        cfg = {"prompt": "Explain why notes.md is a good example filename.",
               "cmd": ["codex", "exec"], "contract": "auto",
               "output_schema": {"notes": "string"}, "output_files": {"notes": "notes.md"}}
        _swap(_raw_for(cfg, "codex_cli"), "api")  # must not raise

    def test_legacy_header_in_authored_prose_not_truncated(self):
        cfg = {"model": "m",
               "user_prompt": "Discuss 'Output each named result using exactly these tags:' then prove T; essential.",
               "output": {"default_field": "answer"}}
        raw = _raw_for(cfg, "api")
        _swap(raw, "codex_cli")
        self.assertIn("prove T", raw["components"]["cfg"]["prompt"])

    def test_text_file_collector_kind_allowed(self):
        # `type: string` (and other non-special kinds) read a text file — allowed
        cfg = {"prompt": "x", "cmd": ["codex", "exec"], "contract": "auto",
               "output_schema": {"answer": "string"},
               "output_files": {"answer": {"path": "answer.txt", "type": "string"}}}
        _swap(_raw_for(cfg, "codex_cli"), "api")  # must not raise

    def test_enum_choices_preserved_across_swap(self):
        cfg = {"model": "m", "user_prompt": "x",
               "output_schema": {"verdict": {"enum": ["yes", "no"]}}, "output": {"default_field": "verdict"}}
        raw = _raw_for(cfg, "api")
        _swap(raw, "codex_cli")
        self.assertEqual(raw["components"]["cfg"]["output_schema"]["verdict"], {"enum": ["yes", "no"]})

    def test_generated_cli_filenames_are_distinct(self):
        cfg = {"model": "m", "user_prompt": "x",
               "output_schema": {"notes": "string", "notes_txt": "string"},
               "output": {"xml_tags": ["notes", "notes_txt"], "default_field": "notes"}}
        raw = _raw_for(cfg, "api")
        _swap(raw, "codex_cli")
        files = raw["components"]["cfg"]["output_files"]
        self.assertEqual(len(set(files.values())), len(files), files)

    def test_string_form_codex_cmd_is_a_noop(self):
        cfg = {"model": "gpt-5.6-sol", "model_reasoning_effort": "high", "prompt": "x",
               "cmd": "codex exec --json", "copy_codex_auth": True, "contract": "auto",
               "output_schema": {"answer": "string"}, "output_files": {"answer": "answer.txt"},
               "done_outputs": {"status": "status"}}
        raw = _raw_for(cfg, "codex_cli")
        _swap(raw, "codex_cli")
        self.assertEqual(raw["components"]["cfg"]["model"], "gpt-5.6-sol")
        self.assertEqual(raw["components"]["cfg"]["model_reasoning_effort"], "high")

    def test_reserved_output_name_refused(self):
        for bad in ("status", "raw_text", "workspace"):
            cfg = {"model": "m", "user_prompt": "x", "output": {"default_field": bad}}
            with self.assertRaisesRegex(PresetError, "reserved"):
                _swap(_raw_for(cfg, "api"), "codex_cli")

    def test_component_without_any_prompt_refused(self):
        cfg = {"model": "m", "output": {"default_field": "a"}}
        with self.assertRaisesRegex(PresetError, "no authored prompt"):
            _swap(_raw_for(cfg, "api"), "codex_cli")

    def test_system_prompt_only_keeps_the_task_on_swap(self):
        cfg = {"model": "m", "system_prompt": "Solve rigorously.", "output": {"default_field": "answer"}}
        raw = _raw_for(cfg, "api")
        _swap(raw, "codex_cli")
        prompt = raw["components"]["cfg"]["prompt"]
        self.assertIn("Solve rigorously.", prompt)
        self.assertIn("{problem}", prompt)

    def test_adding_a_cli_output_creates_its_file(self):
        cfg = {"prompt": "x", "cmd": ["codex", "exec"], "contract": "auto",
               "output_schema": {"workspace": "string", "status": "string", "answer": "string"},
               "output_files": {"answer": "answer.txt"}, "done_outputs": {"status": "status"}}
        res = mutate_preset_yaml(_dump(_raw_for(cfg, "codex_cli")), {"op": "update_component", "name": "cfg",
            "fields": {"output_schema": "workspace: string\nstatus: string\nanswer: string\nnotes: string"}})
        c = yaml.safe_load(res["raw_yaml"])["components"]["cfg"]
        self.assertIn("notes", c["output_files"])

    def test_schema_edit_binding_order_is_deterministic(self):
        cfg = {"model": "m", "user_prompt": "x", "output_schema": {"notes": "string"},
               "output": {"default_field": "notes"}}
        res = mutate_preset_yaml(_dump(_raw_for(cfg, "api")), {"op": "update_component", "name": "cfg",
            "fields": {"output_schema": "notes: string\nscore: string"}})
        c = yaml.safe_load(res["raw_yaml"])["components"]["cfg"]
        self.assertEqual(c["output"]["xml_tags"], ["notes", "score"])

    def test_filename_as_example_under_contract_auto_allowed(self):
        cfg = {"prompt": "For instance notes.md would be a sensible name.",
               "cmd": ["codex", "exec"], "contract": "auto",
               "output_schema": {"notes": "string"}, "output_files": {"notes": "notes.md"}}
        _swap(_raw_for(cfg, "codex_cli"), "api")  # must not raise


class RuntimeInstructionTests(unittest.TestCase):
    def _render(self, cfg, **inp):
        with tempfile.TemporaryDirectory() as d:
            ctx = RunContext.create(run_id="r", root_workdir=d, flat=True, component_configs={"cfg": cfg})
            a = ConfigurablePromptAgent(ctx, name="cfg")
            return a.render_messages(a.Inputs(**inp))

    def test_short_tag_math_still_gets_instruction(self):
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
        self.assertIn("FORMAT", "\n".join(str(m["content"]) for m in msgs))


class DemoPresetTests(unittest.TestCase):
    def test_conditional_repeat_preset_validates(self):
        path = ROOT / "configs" / "workflows" / "conditional_repeat_screenshot.yaml"
        report = validate_preset_yaml(path.read_text())
        self.assertTrue(report["ok"], report.get("errors"))


class LatexEscapeTests(unittest.TestCase):
    def test_dispute_percent_is_escaped(self):
        from app.dev import _surface_dispute_markers
        out = _surface_dispute_markers("% >>> DISPUTE: confidence is only 50%\n")
        self.assertIn(r"50\%", out)
        self.assertNotIn("only 50%\n", out)


if __name__ == "__main__":
    unittest.main()
