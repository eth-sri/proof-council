"""Regression battery for the cluster-E executor-swap fixes (R1).

Two defects, now fixed, are pinned here:

* E-B1 — ``_infer_executor`` used to default *every* non-Codex configurable
  CLI command to ``claude_cli``. A genuinely custom command (``sh -c ...``,
  neither ``claude`` nor ``codex``) was mislabeled Claude, so a swap to Codex
  was admitted and destructively overwrote the user's command. The fix returns
  ``""`` for an unrecognized command, so the swap guard refuses.

* E-B5 — the swap read only the ``agent:`` key, but the runtime
  (dag_workflow._agent_for) also accepts the ``class:`` alias. A valid
  ``class:`` node was therefore refused as custom-coded. The fix reads either
  key (``_node_agent_class``) and writes back to whichever the node uses.
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from proofstack.registry import PresetError  # noqa: E402

from app.dev_data import (  # noqa: E402
    CONFIGURABLE_CLI_AGENT,
    _infer_executor,
    _op_set_executor,
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
        "output_schema": {"workspace": "string", "answer_tex": "string", "status": "string"},
        "output_files": {"answer_tex": "answer.tex"},
        "done_outputs": {"status": "status"},
    }


def _full_raw(cfg: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    """A minimal but fully valid single-node preset for validate_preset_yaml."""
    base_node = {"id": "n", "kind": "agent", "name": "cfg",
                 "inputs": {"problem": "$input.problem"}}
    base_node.update(node)
    return {
        "workflow": "proofstack.agents.dag_workflow.DAGWorkflow",
        "inputs": {"problem": "", "claude_model": "sonnet"},
        "components": {"cfg": copy.deepcopy(cfg)},
        "dag": {"nodes": [base_node], "outputs": {"answer_tex": "$node.n.answer_tex"}},
    }


def _swap(raw: dict[str, Any], executor: str) -> None:
    _op_set_executor(raw, {"op": "set_executor", "name": "cfg", "executor": executor})


class EB1CustomCommandTests(unittest.TestCase):
    """A custom CLI command is neither Claude nor Codex — a swap must refuse
    rather than overwrite it."""

    def test_custom_sh_command_infers_no_executor(self) -> None:
        cfg = _claude_cli_cfg()
        cfg["cmd"] = ["sh", "-c", "echo hi > answer.tex"]
        cfg.pop("copy_codex_auth", None)
        self.assertEqual(_infer_executor(cfg, CONFIGURABLE_CLI_AGENT), "")

    def test_swap_of_custom_command_is_refused_and_command_untouched(self) -> None:
        cfg = _claude_cli_cfg()
        cfg["cmd"] = ["sh", "-c", "echo hi > answer.tex"]
        raw = _full_raw(cfg, {"agent": CONFIGURABLE_CLI_AGENT})
        self.assertTrue(validate_preset_yaml(yaml.safe_dump(raw))["ok"])
        with self.assertRaises(PresetError):
            _swap(raw, "codex_cli")
        # the user's command survives verbatim
        self.assertEqual(raw["components"]["cfg"]["cmd"],
                         ["sh", "-c", "echo hi > answer.tex"])

    def test_normal_claude_node_still_swaps_to_codex(self) -> None:
        raw = _full_raw(_claude_cli_cfg(), {"agent": CONFIGURABLE_CLI_AGENT})
        _swap(raw, "codex_cli")
        cfg = raw["components"]["cfg"]
        self.assertEqual(cfg["cmd"][0], "codex")
        self.assertEqual(cfg["usage"]["type"], "codex_jsonl")
        self.assertEqual(raw["dag"]["nodes"][0]["agent"], CONFIGURABLE_CLI_AGENT)


class EB5ClassAliasTests(unittest.TestCase):
    """A node using the ``class:`` alias is a first-class configurable node and
    must swap like an ``agent:`` node."""

    def _class_node_raw(self) -> dict[str, Any]:
        return _full_raw(_claude_cli_cfg(), {"class": CONFIGURABLE_CLI_AGENT})

    def test_class_alias_node_validates(self) -> None:
        raw = self._class_node_raw()
        self.assertNotIn("agent", raw["dag"]["nodes"][0])
        self.assertTrue(validate_preset_yaml(yaml.safe_dump(raw))["ok"])

    def test_class_alias_node_swaps_claude_to_codex(self) -> None:
        raw = self._class_node_raw()
        _swap(raw, "codex_cli")
        cfg = raw["components"]["cfg"]
        node = raw["dag"]["nodes"][0]
        # command scaffold flipped to Codex
        self.assertEqual(cfg["cmd"][0], "codex")
        self.assertEqual(cfg["usage"]["type"], "codex_jsonl")
        # class key updated, no stale-alias-plus-fresh-agent duplication
        self.assertEqual(node.get("class"), CONFIGURABLE_CLI_AGENT)
        self.assertNotIn("agent", node)
        # result still validates
        self.assertTrue(validate_preset_yaml(yaml.safe_dump(raw))["ok"])

    def test_class_alias_round_trip_back_to_claude(self) -> None:
        raw = self._class_node_raw()
        _swap(raw, "codex_cli")
        _swap(raw, "claude_cli")
        cfg = raw["components"]["cfg"]
        node = raw["dag"]["nodes"][0]
        self.assertEqual(cfg["cmd"][0], "claude")
        self.assertEqual(cfg["usage"]["type"], "claude_json")
        self.assertEqual(node.get("class"), CONFIGURABLE_CLI_AGENT)
        self.assertNotIn("agent", node)


if __name__ == "__main__":
    unittest.main()
