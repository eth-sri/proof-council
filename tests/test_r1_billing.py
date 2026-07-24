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
from proofstack.budget import BudgetExhausted, BudgetSpec, BudgetTracker  # noqa: E402
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


def _meter(cfg: dict, copied: bool = False) -> tuple[float, int]:
    """Run the metering path for a codex node and return (usd, tokens).

    ``copied`` mirrors what ``setup()`` leaves behind: whether codex subscription
    auth was actually copied into the sandbox (only then is the run un-billed)."""
    with tempfile.TemporaryDirectory() as td:
        ctx = RunContext.create(
            run_id="t",
            root_workdir=td,
            flat=True,
            component_configs={"cfg_x": cfg},
        )
        agent = ConfigurableCLIAgent(ctx, name="cfg_x")
        agent._active_workspace_root = Path(td)
        agent._copied_codex_auth = copied
        asyncio.run(agent.record_cli_usage(_CODEX_STDOUT, "", CLIDoneRecord()))
        return agent.tracker.counters.usd, agent.tracker.counters.tokens


class R1BillingTests(unittest.TestCase):
    def test_subscription_node_without_explicit_bill_meters_tokens_but_no_usd(self) -> None:
        # copy_codex_auth actually copied == subscription auth: USD must stay zero
        # even when the usage dict forgets `bill: false`. Tokens are still metered.
        cfg = {
            "copy_codex_auth": True,
            "usage": {
                "type": "codex_jsonl",
                "model": "gpt-5.4-mini",
                "cost_config": "models/openai/gpt-54-mini",
            },
        }
        usd, tokens = _meter(cfg, copied=True)
        self.assertEqual(usd, 0.0)
        self.assertEqual(tokens, 1500)

    def test_copy_codex_auth_flag_but_auth_not_copied_still_bills(self) -> None:
        # B2: copy_codex_auth: true but no host auth.json to copy -> the run fell
        # back to a billable key. Billing must key on whether auth was ACTUALLY
        # copied (_copied_codex_auth), not the config flag, or a key run goes free.
        cfg = {
            "copy_codex_auth": True,
            "usage": {
                "type": "codex_jsonl",
                "model": "gpt-5.4-mini",
                "cost_config": "models/openai/gpt-54-mini",
            },
        }
        usd, tokens = _meter(cfg, copied=False)
        self.assertGreater(usd, 0.0)
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

    def test_bill_false_never_masks_missing_subscription_auth(self) -> None:
        cfg = _cli_component_template()
        self.assertTrue(cfg.get("copy_codex_auth"))
        self.assertEqual(cfg["usage"].get("bill"), False)
        self.assertEqual(cfg["sandbox"].get("provider_keys"), [])

        # The declaration is not authoritative: without observed copied auth,
        # metering must account for a possible paid-key run.
        usd, tokens = _meter(cfg, copied=False)
        self.assertGreater(usd, 0.0)
        self.assertEqual(tokens, 1500)

        # With observed subscription auth, the same template remains unbilled.
        subscription_usd, _ = _meter(cfg, copied=True)
        self.assertEqual(subscription_usd, 0.0)

    def test_zero_usd_budget_is_a_hard_zero_after_spend(self) -> None:
        tracker = BudgetTracker(scope="run", spec=BudgetSpec(max_usd=0.0))
        self.assertEqual(tracker.check(), [])
        tracker.add_usd(0.01)
        with self.assertRaises(BudgetExhausted):
            tracker.check()


if __name__ == "__main__":
    unittest.main()
