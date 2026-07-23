"""R2 Round-4: the discrete-block boundary is DURABLE and separate from probes.

Round-3 merged the CLI-detected reset and the probe reset into one value used
for BOTH the ledger block boundary AND probe-utilisation freshness. A fresh
review found that conflation broken:

  B1 — a later reset-less 429 overwrote a still-future reset, cutting the block
       short and admitting into an exhausted window.
  B2/A2 — a stale CLI boundary made a fresh reset-less probe look stale (wrongly
       admitting), while a reset-less probe replacing a probe-sourced boundary
       erased it and resurrected pre-reset usage.
  A1 — a reset-less 429 calibrated the ceiling from the whole rolling ledger.

Fix: the boundary is persisted durably (by record_probe / record_rate_limit,
only ever from an explicit reset, never erased by a reset-less observation);
probe freshness keys off the probe's OWN reset/TTL; a reset-less 429 does not
calibrate a discrete ceiling.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from proofstack.subscription import SubscriptionPacer, SubscriptionStore  # noqa: E402

NOW = 3_000_000.0


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


class FollowupNoEpochKeepsFutureResetTests(_PacerCase):
    """B1: a reset-less follow-up 429 must not shorten a known future block."""

    def test_future_reset_survives_a_no_epoch_followup(self) -> None:
        reset = self.now + 3600
        self.store.save_settings(
            {"enabled": True, "ceilings": {"five_hour": 1000}, "node_estimate_tokens": 50}
        )
        self.store.record_rate_limit(
            provider="claude", window="five_hour", observed_ceiling=0, reset_at=reset, now=self.now
        )
        self.store.record_rate_limit(
            provider="claude", window="five_hour", observed_ceiling=0, reset_at=None, now=self.now + 10
        )
        # still blocked until the original future reset, not cut to a 60s recheck
        self.assertFalse(self.pacer.decide(model="opus", now=self.now + 71).admit)
        cal = self.store.load_settings()["calibration"]["claude"]["five_hour"]
        self.assertAlmostEqual(cal["blocked_until"], reset, delta=1)
        self.assertAlmostEqual(cal["reset_at"], reset, delta=1)
        # and it does self-heal once the real reset passes
        self.assertTrue(self.pacer.decide(model="opus", now=reset + 1).admit)


class StaleBoundaryKeepsFreshProbeTests(_PacerCase):
    """B2: an old CLI boundary must not discard a fresh reset-less probe."""

    def test_fresh_no_reset_probe_still_gates_despite_old_cli_boundary(self) -> None:
        codex = SubscriptionPacer(self.store, provider="codex")
        self.store.save_settings(
            {
                "enabled": True,
                "account_cap_pct": {"weekly": 90},
                "node_estimate_tokens": 100,
                "probe_ttl_s": 600,
            }
        )
        # an OLD CLI-detected boundary, well in the past
        self.store.record_rate_limit(
            provider="codex", window="weekly", observed_ceiling=0,
            reset_at=self.now - 5000, now=self.now - 5001,
        )
        # a FRESH reset-less probe at 100%
        self.store.record_probe(
            provider="codex", windows={"weekly": {"used_pct": 100.0, "resets_at": None}}, now=self.now
        )
        decision = codex.decide(model="gpt-5.6-sol", now=self.now)
        self.assertFalse(decision.admit)  # fresh probe still gates
        self.assertEqual(self._window(decision, "weekly").probe_pct, 100.0)


class DurableBoundarySurvivesResetlessProbeTests(_PacerCase):
    """A2: a reset-less probe must not erase a durable probe-sourced boundary."""

    def test_pre_reset_usage_stays_excluded_after_a_reset_less_probe(self) -> None:
        self.store.save_settings(
            {
                "enabled": True,
                "ceilings": {"weekly": 1000},
                "cap_pct": {"weekly": 100},
                "account_cap_pct": {"weekly": 100},
                "node_estimate_tokens": 50,
            }
        )
        # a probe establishes the block boundary (reset 100s ago)
        self.store.record_probe(
            provider="claude", windows={"weekly": {"used_pct": 30.0, "resets_at": self.now - 100}},
            now=self.now - 200,
        )
        # heavy spend BEFORE that reset belongs to the prior block
        self.store.append_usage(provider="claude", model="sonnet", tokens=5000, now=self.now - 3600)
        # a later reset-less probe must NOT erase the durable boundary
        self.store.record_probe(
            provider="claude", windows={"weekly": {"used_pct": 32.0, "resets_at": None}}, now=self.now
        )
        decision = self.pacer.decide(model="sonnet", now=self.now)
        self.assertEqual(self._window(decision, "weekly").usage, 0)
        self.assertTrue(decision.admit)


if __name__ == "__main__":
    unittest.main()
