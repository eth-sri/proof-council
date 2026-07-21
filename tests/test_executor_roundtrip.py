"""Round-trip property battery for executor switching.

Two invariants, checked at two levels:
  1. Contract: a component's business outputs (names AND types) survive any
     legal sequence of executor swaps; mechanics (workspace/status) never
     leak in.
  2. Behavior: parsing REAL representative responses with the API spec gives
     identical results before and after a round trip — the spec is task
     identity and persists; it is never rebuilt from field names.
Illegal conversions (structured outputs to the human form, entangled
prompts, custom-coded nodes, json_merge) are hard-refused, never mangled.
"""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

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
    _binding_literals,
    _component_prompt_text,
    _is_string_type,
    _op_set_executor,
    _op_update_component,
    _OUTPUT_MECHANICS_FIELDS,
    _prompt_component_template,
    _spec_output_contract,
)

EXECUTORS = ("api", "claude_cli", "codex_cli", "human")
_HOME_AGENTS = {
    "api": CONFIGURABLE_PROMPT_AGENT,
    "claude_cli": CONFIGURABLE_CLI_AGENT,
    "codex_cli": CONFIGURABLE_CLI_AGENT,
    "human": HUMAN_AGENT,
}

# shape name -> (home executor, component config). All prompts are
# delivery-neutral: format instructions are generated, which is what makes a
# component swappable (tier 1).
SHAPES: dict[str, tuple[str, dict[str, Any]]] = {
    "basic_io": (
        "api",
        {
            "model": "models/openai/gpt-54-mini",
            "system_prompt": "You are an expert assistant.",
            "user_prompt": "Complete this node's task.",
            "input_schema": {"question": "string"},
            "output": {"default_field": "output"},
        },
    ),
    "ideator": (
        "api",
        {
            "model": "models/openai/gpt-54-mini",
            "system_prompt": "Propose distinct proof strategies.",
            "user_prompt": "Problem:\n{problem}\nSuggest {n} approaches.",
            "input_schema": {"problem": "string", "n": "integer"},
            "output": {"xml_lists": {"approaches": "approach"}, "default_field": "text"},
        },
    ),
    "regex_verdict": (
        "api",
        {
            "model": "models/openai/gpt-54-mini",
            "system_prompt": "Review the proof and state your assessment.",
            "user_prompt": "Proof:\n{proof}",
            "input_schema": {"proof": "string"},
            "output": {
                "regex_fields": {"verdict": r"ASSESSMENT\s*=\s*(\w+)"},
                "default_field": "analysis",
            },
        },
    ),
    "json_extract": (
        "api",
        {
            "model": "models/openai/gpt-54-mini",
            "system_prompt": "Extract the data.",
            "user_prompt": "Text:\n{text}",
            "input_schema": {"text": "string"},
            "output": {"json_tags": {"data": "payload"}, "default_field": "commentary"},
        },
    ),
    "json_tag_singular": (
        "api",
        {
            "model": "models/openai/gpt-54-mini",
            "system_prompt": "Summarize as data.",
            "user_prompt": "Text:\n{text}",
            "input_schema": {"text": "string"},
            "output": {"json_tag": "result"},
        },
    ),
    "cli_mixed": (
        "claude_cli",
        {
            "prompt": "Review and report.",
            "cmd": ["claude", "-p"],
            "sandbox": {"backend": "subprocess"},
            "input_schema": {"solution": "string", "workspace": "string"},
            "output_schema": {
                "workspace": "string",
                "status": "string",
                "notes": "string",
                "points": {"type": "array", "items": {"type": "string"}},
            },
            "output_files": {
                "notes": "notes.md",
                "points": {"path": "points.json", "type": "json"},
            },
            "done_outputs": {"verdict": "summary", "status": "status"},
        },
    ),
    "human_form": (
        "human",
        {
            "prompt": "Please assess the proof.",
            "input_schema": {"proof": "string"},
            "output_schema": {"answer": "string", "notes": "string"},
        },
    ),
}


def _structured(cfg: dict[str, Any]) -> bool:
    return any(not _is_string_type(t) for t in _absorbed_output_contract(cfg).values())


def _raw_for(cfg: dict[str, Any], home: str) -> dict[str, Any]:
    return {
        "components": {"cfg": copy.deepcopy(cfg)},
        "dag": {
            "nodes": [
                {
                    "id": "work",
                    "kind": "agent",
                    "agent": _HOME_AGENTS[home],
                    "name": "cfg",
                    "inputs": {},
                }
            ]
        },
    }


def _swap(raw: dict[str, Any], executor: str) -> None:
    _op_set_executor(raw, {"op": "set_executor", "name": "cfg", "executor": executor})


def _parse_with(cfg: dict[str, Any], raw_text: str) -> dict[str, Any]:
    """Parse a model response through the REAL parser for this component."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ctx = RunContext.create(
            run_id="parse-test",
            root_workdir=temp_dir,
            flat=True,
            component_configs={"cfg": copy.deepcopy(cfg)},
        )
        agent = ConfigurablePromptAgent(ctx, name="cfg")
        parsed = agent.parse_output(raw_text, agent.Inputs())
        return {k: v for k, v in parsed.model_dump().items() if k != "raw_text"}


# representative responses per API-home shape, honoring each spec's format
RESPONSES = {
    "basic_io": "A plain full-text answer.",
    "ideator": "<approach>induction</approach>\n<approach>contradiction</approach>",
    "regex_verdict": "The proof holds.\nASSESSMENT = pass",
    "json_extract": '<payload>{"x": 1}</payload>\nSome commentary.',
    "json_tag_singular": '<result>{"summary": "ok"}</result>',
}


class ExecutorRoundTripTests(unittest.TestCase):
    def legal_targets(self, cfg: dict[str, Any]) -> list[str]:
        return [e for e in EXECUTORS if e != "human" or not _structured(cfg)]

    def assert_contract(
        self, actual: dict[str, Any], expected: dict[str, Any], note: str
    ) -> None:
        self.assertEqual(dict(actual), dict(expected), note)
        for field in _OUTPUT_MECHANICS_FIELDS:
            self.assertNotIn(field, actual, f"{note}: mechanics leaked into contract")

    def test_contract_survives_every_legal_round_trip(self) -> None:
        for name, (home, cfg) in SHAPES.items():
            original = _absorbed_output_contract(cfg)
            for other in self.legal_targets(cfg):
                if other == home:
                    continue
                with self.subTest(shape=name, via=other):
                    raw = _raw_for(cfg, home)
                    _swap(raw, other)
                    _swap(raw, home)
                    self.assert_contract(
                        _absorbed_output_contract(raw["components"]["cfg"]),
                        original,
                        f"{name}: {home} -> {other} -> {home}",
                    )

    def test_contract_survives_a_long_chain(self) -> None:
        for name, (home, cfg) in SHAPES.items():
            with self.subTest(shape=name):
                original = _absorbed_output_contract(cfg)
                raw = _raw_for(cfg, home)
                for executor in ("codex_cli", "human", "api", "claude_cli", home):
                    if executor == "human" and _structured(cfg):
                        continue
                    _swap(raw, executor)
                self.assert_contract(
                    _absorbed_output_contract(raw["components"]["cfg"]),
                    original,
                    f"{name}: swap chain back to {home}",
                )

    def test_parse_behavior_is_identical_after_round_trips(self) -> None:
        # Round-4 P1 class: the spec must parse REAL responses identically
        # after any round trip, because it persists — tag names, regex
        # patterns, and json bindings included.
        for name, response in RESPONSES.items():
            home, cfg = SHAPES[name]
            before = _parse_with(cfg, response)
            for other in self.legal_targets(cfg):
                if other == home:
                    continue
                with self.subTest(shape=name, via=other):
                    raw = _raw_for(cfg, home)
                    _swap(raw, other)
                    _swap(raw, home)
                    after = _parse_with(raw["components"]["cfg"], response)
                    self.assertEqual(before, after, f"{name} via {other}")

    def test_every_executor_can_produce_the_full_contract(self) -> None:
        for name, (home, cfg) in SHAPES.items():
            contract = set(_absorbed_output_contract(cfg))
            for executor in self.legal_targets(cfg):
                with self.subTest(shape=name, executor=executor):
                    raw = _raw_for(cfg, home)
                    _swap(raw, executor)
                    swapped = raw["components"]["cfg"]
                    if executor == "api":
                        produced = set(_spec_output_contract(swapped.get("output")))
                    elif executor == "human":
                        produced = set(swapped.get("output_schema") or {})
                    else:
                        produced = {
                            str(f).split(".", 1)[0]
                            for f in (swapped.get("output_files") or {})
                        } | set(swapped.get("done_outputs") or {})
                    self.assertLessEqual(
                        contract, produced, f"{name} on {executor}: not producible"
                    )

    def test_ideator_round_trip_restores_the_exact_spec(self) -> None:
        home, cfg = SHAPES["ideator"]
        raw = _raw_for(cfg, home)
        _swap(raw, "codex_cli")
        swapped = raw["components"]["cfg"]
        # structured output projected as parsed JSON, so the list type survives
        self.assertEqual(
            swapped["output_files"]["approaches"],
            {"path": "approaches.json", "type": "json"},
        )
        # the spec lies dormant, untouched, while on the CLI executor
        self.assertEqual(swapped["output"], cfg["output"])
        _swap(raw, "api")
        self.assertEqual(swapped["output"], cfg["output"])

    def test_renaming_an_api_output_does_not_resurrect_the_old_one(self) -> None:
        home, cfg = SHAPES["basic_io"]
        raw = _raw_for(cfg, home)
        component = raw["components"]["cfg"]
        _op_update_component(raw, {"name": "cfg", "fields": {"default_field": "answer"}})
        _swap(raw, "codex_cli")
        self.assertEqual(component["output_files"], {"answer": "answer.txt"})
        self.assertNotIn("output", component["output_schema"])

    def test_removing_the_last_api_output_empties_the_schema(self) -> None:
        home, cfg = SHAPES["basic_io"]
        raw = _raw_for(cfg, home)
        component = raw["components"]["cfg"]
        _op_update_component(raw, {"name": "cfg", "fields": {"default_field": ""}})
        self.assertEqual(component.get("output_schema", {}), {})
        _swap(raw, "codex_cli")
        self.assertEqual(component.get("output_files", {}), {})

    def test_schema_edits_prune_the_dormant_spec_on_return(self) -> None:
        # while on a CLI executor the schema is canonical; a field deleted
        # there must not come back from the dormant spec on the way home
        home, cfg = SHAPES["json_extract"]
        raw = _raw_for(cfg, home)
        component = raw["components"]["cfg"]
        _swap(raw, "codex_cli")
        _op_update_component(
            raw,
            {
                "name": "cfg",
                "fields": {
                    "output_schema": "workspace: string\nstatus: string\ncommentary: string"
                },
            },
        )
        self.assertNotIn("data", component.get("output_files", {}))
        _swap(raw, "api")
        self.assertEqual(component["output"], {"default_field": "commentary"})


class ExecutorRefusalTests(unittest.TestCase):
    def test_structured_outputs_refuse_the_human_form(self) -> None:
        home, cfg = SHAPES["ideator"]
        raw = _raw_for(cfg, home)
        with self.assertRaisesRegex(PresetError, "non-text output"):
            _swap(raw, "human")

    def test_entangled_prompt_refuses_conversion_and_names_the_literal(self) -> None:
        home, cfg = SHAPES["ideator"]
        entangled = copy.deepcopy(cfg)
        entangled["system_prompt"] += " Emit each strategy in its own <approach> tag."
        raw = _raw_for(entangled, home)
        with self.assertRaisesRegex(PresetError, "<approach>"):
            _swap(raw, "codex_cli")

    def test_regex_anchor_in_prompt_refuses_conversion(self) -> None:
        home, cfg = SHAPES["regex_verdict"]
        entangled = copy.deepcopy(cfg)
        entangled["user_prompt"] += (
            '\nEnd with a line "ASSESSMENT = pass" or "ASSESSMENT = fail".'
        )
        raw = _raw_for(entangled, home)
        with self.assertRaisesRegex(PresetError, "delivery"):
            _swap(raw, "codex_cli")

    def test_json_merge_refuses_conversion(self) -> None:
        home, cfg = SHAPES["json_tag_singular"]
        merged = copy.deepcopy(cfg)
        merged["output"] = {"json_tag": "result", "json_merge": True}
        raw = _raw_for(merged, home)
        with self.assertRaisesRegex(PresetError, "json_merge"):
            _swap(raw, "claude_cli")

    def test_component_without_swappable_node_is_refused(self) -> None:
        # e.g. the First Proof engine's model-override components
        raw = {
            "components": {"Author": {"model": "models/openai/gpt-56-sol-pro"}},
            "dag": {
                "nodes": [
                    {
                        "id": "a",
                        "kind": "agent",
                        "agent": "proofstack.agents.ac.ACAuthorBlock",
                    }
                ]
            },
        }
        with self.assertRaisesRegex(PresetError, "not used by any swappable node"):
            _op_set_executor(raw, {"name": "Author", "executor": "codex_cli"})

    def test_component_run_by_custom_agent_is_refused(self) -> None:
        raw = {
            "components": {"cfg": {"prompt": "work"}},
            "dag": {
                "nodes": [
                    {
                        "id": "a",
                        "kind": "agent",
                        "agent": "proofstack.agents.ac.ACAuthorBlock",
                        "name": "cfg",
                    }
                ]
            },
        }
        with self.assertRaisesRegex(PresetError, "custom-coded"):
            _op_set_executor(raw, {"name": "cfg", "executor": "api"})

    def test_legacy_generated_format_block_is_stripped_not_refused(self) -> None:
        home, cfg = SHAPES["ideator"]
        legacy = copy.deepcopy(cfg)
        legacy["user_prompt"] += (
            "\n\nOutput each named result using exactly these tags:\n"
            "<approach>...</approach>"
        )
        raw = _raw_for(legacy, home)
        _swap(raw, "codex_cli")  # must not raise: the block is generated mechanics
        # (CLI convention merges system/user prompts into `prompt`)
        self.assertNotIn("<approach>", raw["components"]["cfg"]["prompt"])


class TemplateNeutralityTests(unittest.TestCase):
    def test_every_builtin_template_is_delivery_neutral(self) -> None:
        # Shipped templates must be tier 1 — freely executor-swappable — so
        # their prompts may not hand-embed delivery format: the runtime
        # generates format instructions from the output spec instead.
        for template in ("prompt_agent", "ideator", "budget_fallback", "loop_step", "default"):
            with self.subTest(template=template):
                cfg = _prompt_component_template(template)
                literals = _binding_literals(
                    cfg.get("output"), _component_prompt_text(cfg)
                )
                self.assertEqual(literals, [], f"{template}: prompt embeds {literals}")


class GeneratedInstructionTests(unittest.TestCase):
    def test_neutral_ideator_gets_runtime_format_instruction(self) -> None:
        _, cfg = SHAPES["ideator"]
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="fmt-test",
                root_workdir=temp_dir,
                flat=True,
                component_configs={"cfg": copy.deepcopy(cfg)},
            )
            agent = ConfigurablePromptAgent(ctx, name="cfg")
            messages = agent.render_messages(agent.Inputs(problem="P", n=2))
        text = "\n".join(str(m.get("content")) for m in messages)
        self.assertIn("<approach>", text)  # generated, not hand-written
        self.assertIn("FORMAT YOUR ANSWER", text)


if __name__ == "__main__":
    unittest.main()
