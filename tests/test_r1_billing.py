from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from proofstack.agents.configurable_cli import ConfigurableCLIAgent  # noqa: E402
from proofstack.context import RunContext  # noqa: E402
from proofstack.kinds.cli import CLIDoneRecord  # noqa: E402
from app.dev_data import _cli_component_template  # noqa: E402

# One completed codex turn with real tokens -> nonzero cost when billed.
_CODEX_STDOUT = json.dumps(
    {
        "type": "turn.completed",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "reasoning_output_tokens": 100,
        },
    }
) + "\n"


def _meter(cfg: dict) -> tuple[float, int]:
    """Run the metering path for a codex node and return (usd, tokens)."""
    with tempfile.TemporaryDirectory() as td:
        ctx = RunContext.create(
            run_id="t",
            root_workdir=td,
            flat=True,
            component_configs={"cfg_x": cfg},
        )
        agent = ConfigurableCLIAgent(ctx, name="cfg_x")
        agent._active_workspace_root = Path(td)
        asyncio.run(agent.record_cli_usage(_CODEX_STDOUT, "", CLIDoneRecord()))
        return agent.tracker.counters.usd, agent.tracker.counters.tokens


class R1BillingTests(unittest.TestCase):
    def test_subscription_node_without_explicit_bill_meters_tokens_but_no_usd(self) -> None:
        # copy_codex_auth == subscription auth: USD must stay zero even when the
        # usage dict forgets `bill: false`. Tokens are still metered.
        cfg = {
            "copy_codex_auth": True,
            "usage": {
                "type": "codex_jsonl",
                "model": "gpt-5.4-mini",
                "cost_config": "models/openai/gpt-54-mini",
            },
        }
        usd, tokens = _meter(cfg)
        self.assertEqual(usd, 0.0)
        self.assertEqual(tokens, 1500)

    def test_key_based_billed_node_still_charges_usd(self) -> None:
        # No copy_codex_auth and no bill:false -> a genuine key-based API node
        # that should still be charged. Guards against over-suppression.
        cfg = {
            "usage": {
                "type": "codex_jsonl",
                "model": "gpt-5.4-mini",
                "cost_config": "models/openai/gpt-54-mini",
            },
        }
        usd, tokens = _meter(cfg)
        self.assertGreater(usd, 0.0)
        self.assertEqual(tokens, 1500)

    def test_editor_add_node_template_carries_bill_false(self) -> None:
        cfg = _cli_component_template()
        self.assertTrue(cfg.get("copy_codex_auth"))
        self.assertEqual(cfg["usage"].get("bill"), False)
        # And end-to-end: the editor template itself must not charge USD.
        usd, tokens = _meter(cfg)
        self.assertEqual(usd, 0.0)
        self.assertEqual(tokens, 1500)


if __name__ == "__main__":
    unittest.main()
