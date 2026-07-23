"""R2 Round-3: complete the pacer discrete-block / claim-expiry model.

Round-2 (test_r1_pacer_state) fixed four boundaries on the PROBE-sourced and
own-spend paths. A fresh review found the sibling paths were left open:

  A2 — a CLI-detected 429 reset must bound the discrete block too (not just a
       probe reset), so pre-reset usage does not re-block the fresh window.
  A3/B1 — a saturated probe with no reset epoch must age out; codex only
          re-probes by harvesting a rollout AFTER a node runs, so an eternal
          block would park the provider forever with no way to refresh.
  A4 — the ACCOUNT gate wait must be bounded by the soonest claim expiry too,
       or a 30-second claim projects to the hours-away provider reset.
  A1 — rate-limit ceiling calibration must count only the current block; usage
       from a prior, already-reset block inflates the ceiling and later
       UNDER-blocks the fresh window.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from proofstack.subscription import (  # noqa: E402
    SubscriptionPacer,
    SubscriptionStore,
)

NOW = 3_000_000.0
FIVE_HOURS = 5 * 3600.0


class _PacerCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SubscriptionStore(home=Path(self._tmp.name))
        self.pacer = SubscriptionPacer(self.store, provider="claude")
        self.now = NOW

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _window(self, decision, name):
        return next(st for st in decision.windows if st.window == name)


class CliDetectedResetBoundaryTests(_PacerCase):
    """A2: a CLI-detected 429 reset bounds the block like a probe reset (B3)."""

    def setUp(self) -> None:
        super().setUp()
        self.store.save_settings(
            {
                "enabled": True,
                "ceilings": {"five_hour": 1000},
                "cap_pct": {"five_hour": 100},
                "account_cap_pct": {"five_hour": 100},
                "node_estimate_tokens": 50,
            }
        )

    def test_pre_reset_usage_excluded_after_cli_block_lapses(self) -> None:
        reset = self.now + 1800
        # heavy spend booked inside the block that is about to reset
        self.store.append_usage(provider="claude", model="opus", tokens=5000, now=self.now)
        self.store.record_rate_limit(
            provider="claude", window="five_hour", observed_ceiling=0, reset_at=reset, now=self.now
        )
        # during the block: blocked_until holds it shut
        self.assertFalse(self.pacer.decide(model="opus", now=self.now).admit)
        # after the reset the pre-reset 5000 belongs to the prior block, so the
        # fresh window is clean and admits (without block_start it would re-block)
        after = self.pacer.decide(model="opus", now=reset + 1)
        self.assertEqual(self._window(after, "five_hour").usage, 0)
        self.assertTrue(after.admit)


class NoResetProbeSelfHealTests(_PacerCase):
    """A3/B1: a saturated no-reset probe ages out so codex can re-probe."""

    def test_codex_no_reset_probe_heals_after_ttl(self) -> None:
        codex = SubscriptionPacer(self.store, provider="codex")
        self.store.save_settings(
            {
                "enabled": True,
                "account_cap_pct": {"weekly": 90},
                "node_estimate_tokens": 100,
                "probe_ttl_s": 600,
            }
        )
        self.store.record_probe(
            provider="codex",
            windows={"weekly": {"used_pct": 100.0, "resets_at": None}},
            now=self.now,
        )
        # fresh 100% probe blocks; still blocked within the TTL horizon...
        self.assertFalse(codex.decide(model="gpt-5.6-sol", now=self.now).admit)
        self.assertFalse(codex.decide(model="gpt-5.6-sol", now=self.now + 599).admit)
        # ...but past the TTL the stale probe is dropped so one node may run and
        # re-harvest a fresh rollout, instead of a permanent park
        self.assertTrue(codex.decide(model="gpt-5.6-sol", now=self.now + 601).admit)


class AccountGateClaimBoundTests(_PacerCase):
    """A4: the account gate wait is bounded by the soonest claim expiry."""

    def test_short_claim_does_not_project_to_provider_reset(self) -> None:
        self.store.save_settings(
            {
                "enabled": True,
                "ceilings": {"five_hour": 1000},
                "cap_pct": {"five_hour": 100},
                "account_cap_pct": {"five_hour": 90},
                "node_estimate_tokens": 100,
            }
        )
        reset = self.now + 14400  # provider reset four hours out
        self.store.record_probe(
            provider="claude",
            windows={"five_hour": {"used_pct": 85.0, "resets_at": reset}},
            now=self.now,
        )
        # 85% probe + a claim's drift tips the account gate over its 90% cap
        self.store.add_claim(
            provider="claude", model="opus", est_tokens=600, ttl_s=30, now=self.now
        )
        decision = self.pacer.decide(model="opus", now=self.now)
        self.assertFalse(decision.admit)
        self.assertEqual(decision.blocking_window, "five_hour")
        # bounded to the 30s claim, not the 4h provider reset
        self.assertAlmostEqual(decision.wait_s, 30.0, delta=1)

    def test_no_claim_keeps_the_provider_reset_wait(self) -> None:
        # own usage alone (no claim) tips the account gate: the wait is the true
        # provider reset, unbounded by any claim.
        self.store.save_settings(
            {
                "enabled": True,
                "ceilings": {"five_hour": 1000},
                "cap_pct": {"five_hour": 100},
                "account_cap_pct": {"five_hour": 90},
                "node_estimate_tokens": 100,
            }
        )
        reset = self.now + 14400
        self.store.record_probe(
            provider="claude",
            windows={"five_hour": {"used_pct": 95.0, "resets_at": reset}},
            now=self.now,
        )
        decision = self.pacer.decide(model="opus", now=self.now)
        self.assertFalse(decision.admit)
        self.assertAlmostEqual(decision.wait_s, 14400.0, delta=2)


class BlockAwareCeilingCalibrationTests(unittest.TestCase):
    """A1: a real-limit ceiling calibration counts only the current block."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._home = Path(self._tmp.name)
        self._prev = os.environ.get("PROOFCOUNCIL_HOME")
        os.environ["PROOFCOUNCIL_HOME"] = str(self._home)

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop("PROOFCOUNCIL_HOME", None)
        else:
            os.environ["PROOFCOUNCIL_HOME"] = self._prev
        self._tmp.cleanup()

    def test_pre_reset_usage_excluded_from_observed_ceiling(self) -> None:
        # detect_rate_limit stamps the hit against real wall-clock, so this test
        # uses real-time-relative ledger timestamps rather than a synthetic now.
        from proofstack.agents.configurable_cli import ConfigurableCLIAgent
        from proofstack.context import RunContext

        store = SubscriptionStore(home=self._home)
        now = time.time()
        reset = now + 3600  # < 5h -> maps to the five_hour window
        # block_start = reset - 5h = now - 14400: the first entry predates it
        store.append_usage(provider="claude", model="opus", tokens=9000, now=now - 16000)
        store.append_usage(provider="claude", model="opus", tokens=100, now=now - 60)

        ctx = RunContext.create(
            run_id="a1",
            root_workdir=self._home / "run",
            flat=True,
            component_configs={"cli": {"cmd": ["claude"]}},
        )
        agent = ConfigurableCLIAgent(ctx, name="cli")
        stdout = f"Claude AI usage limit reached|{int(reset)}"
        asyncio.run(agent._scan_claude_rate_limit(stdout, ""))

        cal = store.load_settings()["calibration"]["claude"]["five_hour"]
        self.assertEqual(cal["observed_ceiling"], 100)


if __name__ == "__main__":
    unittest.main()
