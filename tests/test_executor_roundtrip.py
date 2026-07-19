"""Round-trip property battery for executor switching.

The invariant that three rounds of review findings circled: a component's
business output contract (field names AND types) must survive any sequence of
executor swaps, and executor mechanics (workspace/status) must never leak into
it. Every shape here mirrors a real component style — the Basic I/O template,
the ideator (xml_lists, no schema), regex/json output specs, mixed CLI
sources, a typed human form.
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from app.dev_data import (  # noqa: E402
    _absorbed_output_contract,
    _op_set_executor,
    _op_update_component,
    _OUTPUT_MECHANICS_FIELDS,
)

EXECUTORS = ("api", "claude_cli", "codex_cli", "human")

# shape name -> (home executor, component config)
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
            "system_prompt": "Propose strategies in <approach> tags.",
            "user_prompt": "Problem:\n{problem}\nSuggest {n} approaches.",
            "input_schema": {"problem": "string", "n": "integer"},
            "output": {"xml_lists": {"approaches": "approach"}, "default_field": "text"},
        },
    ),
    "regex_verdict": (
        "api",
        {
            "model": "models/openai/gpt-54-mini",
            "system_prompt": "Review the proof.",
            "user_prompt": "Proof:\n{proof}",
            "input_schema": {"proof": "string"},
            "output": {
                "regex_fields": {"verdict": r"VERDICT:\s*(\w+)"},
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
            "output": {"json_tags": {"data": "data"}, "default_field": "commentary"},
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
            "output_schema": {
                "answer": "string",
                "concerns": {"type": "array", "items": {"type": "string"}},
            },
        },
    ),
}


def _swap(raw: dict[str, Any], executor: str) -> None:
    _op_set_executor(raw, {"op": "set_executor", "name": "cfg", "executor": executor})


def _raw_for(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"components": {"cfg": copy.deepcopy(cfg)}, "dag": {"nodes": []}}


class ExecutorRoundTripTests(unittest.TestCase):
    def assert_contract(self, actual: dict[str, Any], expected: dict[str, Any], note: str) -> None:
        self.assertEqual(dict(actual), dict(expected), note)
        for field in _OUTPUT_MECHANICS_FIELDS:
            self.assertNotIn(field, actual, f"{note}: mechanics leaked into contract")

    def test_contract_survives_every_round_trip(self) -> None:
        for name, (home, cfg) in SHAPES.items():
            original = _absorbed_output_contract(cfg)
            for other in EXECUTORS:
                if other == home:
                    continue
                with self.subTest(shape=name, via=other):
                    raw = _raw_for(cfg)
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
                raw = _raw_for(cfg)
                for executor in ("codex_cli", "human", "api", "claude_cli", home):
                    _swap(raw, executor)
                self.assert_contract(
                    _absorbed_output_contract(raw["components"]["cfg"]),
                    original,
                    f"{name}: 5-swap chain back to {home}",
                )

    def test_every_executor_can_produce_the_full_contract(self) -> None:
        # after a swap, the new executor's own mechanics must cover every
        # contract field — the validation gap behind "outputs silently lost"
        for name, (_, cfg) in SHAPES.items():
            contract = set(_absorbed_output_contract(cfg))
            for executor in EXECUTORS:
                with self.subTest(shape=name, executor=executor):
                    raw = _raw_for(cfg)
                    _swap(raw, executor)
                    swapped = raw["components"]["cfg"]
                    if executor == "api":
                        produced = set(
                            _absorbed_output_contract({"output": swapped.get("output")})
                        )
                    elif executor == "human":
                        produced = set(swapped.get("output_schema") or {})
                    else:
                        produced = {
                            str(f).split(".", 1)[0]
                            for f in (swapped.get("output_files") or {})
                        } | set(swapped.get("done_outputs") or {})
                    self.assertLessEqual(
                        contract,
                        produced,
                        f"{name} on {executor}: contract fields not producible",
                    )

    def test_ideator_list_type_survives_api_cli_api(self) -> None:
        # the round-3 P1 reproduction: approaches must stay a list, and the
        # regenerated spec must parse it as xml_lists, not a scalar tag
        _, cfg = SHAPES["ideator"]
        raw = _raw_for(cfg)
        _swap(raw, "codex_cli")
        swapped = raw["components"]["cfg"]
        self.assertEqual(
            swapped["output_files"]["approaches"],
            {"path": "approaches.json", "type": "json"},
        )
        _swap(raw, "api")
        self.assertEqual(
            swapped["output"]["xml_lists"], {"approaches": "approach"}
        )
        self.assertEqual(
            swapped["output_schema"]["approaches"],
            {"type": "array", "items": {"type": "string"}},
        )

    def test_renaming_an_api_output_does_not_resurrect_the_old_one(self) -> None:
        # the round-3 P2a reproduction, driven through the editor op
        _, cfg = SHAPES["basic_io"]
        raw = _raw_for(cfg)
        component = raw["components"]["cfg"]
        _op_update_component(
            raw, {"name": "cfg", "fields": {"default_field": "answer"}}
        )
        _swap(raw, "codex_cli")
        self.assertEqual(component["output_files"], {"answer": "answer.txt"})
        self.assertNotIn("output", component["output_schema"])


if __name__ == "__main__":
    unittest.main()
