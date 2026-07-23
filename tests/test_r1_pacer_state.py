"""R2 Part B: the consolidated pacer window state machine.

One coherent WindowStatus derivation replaces the scattered conditionals across
`_window_statuses` / `_decide_from` / `_account_gate_wait`. These table-style
tests pin the four boundaries that the ad-hoc code left open:

  A8 — a real 0% cap hard-blocks (allowed=0), it does not silently admit; a
       probe-only window (no ceiling) is allowed=None and stays account-gated.
  A9 — a detected limit with no reset timestamp still produces a block, and the
       block self-heals after one recheck interval.
  B3 — once a discrete provider block has reset, pre-reset usage does not
       resurrect (even with a stale probe still on disk).
  A11 — an own-spend wait is bounded by the soonest claim expiry, so a
        30-second claim can't project an hours-long wait and force a park.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from proofstack.subscription import (  # noqa: E402
    DEFAULT_RECHECK_S,
    SubscriptionPacer,
    SubscriptionStore,
)

NOW = 3_000_000.0
FIVE_HOURS = 5 * 3600.0
ONE_WEEK = 7 * 86400.0


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


class ZeroCapHardBlockTests(_PacerCase):
    """A8: allowed=None (no cap) vs allowed=0 (hard block)."""

    def _decide_at_cap(self, cap_pct: int):
        self.store.save_settings(
            {
                "enabled": True,
                "ceilings": {"five_hour": 1000},
                "cap_pct": {"five_hour": cap_pct},
                "node_estimate_tokens": 100,
            }
        )
        return self.pacer.decide(model="opus", now=self.now)

    def test_cap_pct_admission_table(self) -> None:
        # a fresh (untouched) window: an estimate alone must not stall a real
        # cap, but a 0% cap must hard-block regardless of the untouched rule.
        for cap_pct, want_admit, want_allowed in [
            (0, False, 0),
            (50, True, 500),
            (100, True, 1000),
        ]:
            with self.subTest(cap_pct=cap_pct):
                decision = self._decide_at_cap(cap_pct)
                five = self._window(decision, "five_hour")
                self.assertEqual(five.allowed, want_allowed)
                self.assertEqual(decision.admit, want_admit)
                if not want_admit:
                    self.assertEqual(decision.blocking_window, "five_hour")

    def test_zero_cap_rechecks_so_raising_it_admits(self) -> None:
        decision = self._decide_at_cap(0)
        self.assertFalse(decision.admit)
        # bounded to a recheck interval, not an unbounded park
        self.assertAlmostEqual(decision.wait_s, DEFAULT_RECHECK_S, delta=1)
        # raising the cap live admits on the next pass, no restart
        self.store.save_settings({"cap_pct": {"five_hour": 100}})
        self.assertTrue(self.pacer.decide(model="opus", now=self.now).admit)

    def test_probe_only_window_is_allowed_none_not_zero(self) -> None:
        # codex weekly has no ceiling/seed -> probe-only; allowed must be None
        # (governed by the account gate), never a spurious 0 hard block.
        codex = SubscriptionPacer(self.store, provider="codex")
        self.store.save_settings(
            {"enabled": True, "account_cap_pct": {"weekly": 90}, "node_estimate_tokens": 0}
        )
        self.store.record_probe(
            provider="codex",
            windows={"weekly": {"used_pct": 10.0, "resets_at": self.now + 86400}},
            now=self.now,
        )
        decision = codex.decide(model="gpt-5.6-sol", now=self.now)
        weekly = self._window(decision, "weekly")
        self.assertIsNone(weekly.allowed)
        self.assertTrue(decision.admit)  # 10% is under the 90% account cap


class DetectedLimitNoResetTests(_PacerCase):
    """A9: a detected limit with no reset epoch still blocks."""

    def test_cli_limit_without_reset_blocks_then_self_heals(self) -> None:
        self.store.save_settings(
            {"enabled": True, "cap_pct": {"five_hour": 50}, "node_estimate_tokens": 100}
        )
        # a real limit hit that reported no reset time, and no local usage
        self.store.record_rate_limit(
            provider="claude", window="five_hour", observed_ceiling=0, reset_at=None, now=self.now
        )
        blocked = self.pacer.decide(model="opus", now=self.now)
        self.assertFalse(blocked.admit)
        self.assertEqual(blocked.blocking_window, "five_hour")
        self.assertAlmostEqual(blocked.wait_s, DEFAULT_RECHECK_S, delta=1)
        # after one recheck interval the block lapses so a re-probe / re-hit,
        # not a permanent lockout, decides the next pass
        healed = self.pacer.decide(model="opus", now=self.now + DEFAULT_RECHECK_S + 1)
        self.assertTrue(healed.admit)

    def test_full_probe_without_reset_records_a_block(self) -> None:
        self.store.save_settings({"enabled": True})
        self.store.record_probe(
            provider="claude",
            windows={"five_hour": {"used_pct": 100.0, "resets_at": None}},
            now=self.now,
        )
        cal = self.store.load_settings()["calibration"]["claude"]["five_hour"]
        self.assertIsNotNone(cal.get("blocked_until"))
        self.assertAlmostEqual(cal["blocked_until"], self.now + DEFAULT_RECHECK_S, delta=1)


class DiscreteResetResurrectionTests(_PacerCase):
    """B3: pre-reset usage must not resurrect after a discrete reset."""

    def setUp(self) -> None:
        super().setUp()
        self.store.save_settings(
            {
                "enabled": True,
                "ceilings": {"weekly": 1000},
                "cap_pct": {"weekly": 100},
                "account_cap_pct": {"weekly": 100},  # keep the account gate open
                "node_estimate_tokens": 50,
            }
        )

    def _probe_weekly_reset_at(self, reset_at: float, at: float) -> None:
        self.store.record_probe(
            provider="claude",
            windows={"weekly": {"used_pct": 30.0, "resets_at": reset_at}},
            now=at,
        )

    def test_usage_before_a_past_reset_is_excluded(self) -> None:
        # a stale probe: it said the block reset 100s ago
        self._probe_weekly_reset_at(self.now - 100, at=self.now - 200)
        # heavy spend booked BEFORE that reset belongs to the prior block
        self.store.append_usage(
            provider="claude", model="sonnet", tokens=5000, now=self.now - 3600
        )
        decision = self.pacer.decide(model="sonnet", now=self.now)
        weekly = self._window(decision, "weekly")
        self.assertEqual(weekly.usage, 0)
        self.assertTrue(decision.admit)

    def test_usage_after_the_reset_still_counts(self) -> None:
        # same past reset, but the spend landed AFTER it -> current block
        self._probe_weekly_reset_at(self.now - 3600, at=self.now - 100)
        self.store.append_usage(
            provider="claude", model="sonnet", tokens=5000, now=self.now - 1800
        )
        decision = self.pacer.decide(model="sonnet", now=self.now)
        weekly = self._window(decision, "weekly")
        self.assertEqual(weekly.usage, 5000)
        self.assertFalse(decision.admit)


class ClaimExpiryBoundsWaitTests(_PacerCase):
    """A11: an own-spend wait is bounded by the soonest claim expiry."""

    def setUp(self) -> None:
        super().setUp()
        self.store.save_settings(
            {
                "enabled": True,
                "ceilings": {"five_hour": 1000},
                "cap_pct": {"five_hour": 100},
                "account_cap_pct": {"five_hour": 100},
                "node_estimate_tokens": 100,
            }
        )
        # own usage under the cap; a claim is what tips it over
        self.store.append_usage(provider="claude", model="opus", tokens=500, now=self.now - 10)

    def test_wait_tracks_claim_ttl(self) -> None:
        # deficit is 500+600+100-1000=200, covered by the 500-token ledger entry
        # whose rolling age-out is ~17990s; the claim frees the headroom sooner.
        for ttl in (30.0, 300.0):
            with self.subTest(ttl=ttl):
                cid = self.store.add_claim(
                    provider="claude", model="opus", est_tokens=600, ttl_s=ttl, now=self.now
                )
                decision = self.pacer.decide(model="opus", now=self.now)
                self.assertFalse(decision.admit)
                self.assertAlmostEqual(decision.wait_s, ttl, delta=1)
                self.store.remove_claim(cid)

    def test_a_far_claim_does_not_shorten_a_genuine_ageout_wait(self) -> None:
        # claim expires far out (well past the ledger age-out): the wait is the
        # true ledger age-out, so an unrelated claim never inflates it.
        self.store.add_claim(
            provider="claude", model="opus", est_tokens=600, ttl_s=ONE_WEEK, now=self.now
        )
        decision = self.pacer.decide(model="opus", now=self.now)
        self.assertFalse(decision.admit)
        self.assertAlmostEqual(decision.wait_s, FIVE_HOURS - 10, delta=2)

    def test_no_claim_keeps_the_ledger_ageout_wait(self) -> None:
        # push over the cap on own usage alone (no claim): the wait is the
        # rolling ledger age-out, unbounded by any claim.
        self.store.append_usage(provider="claude", model="opus", tokens=450, now=self.now - 10)
        decision = self.pacer.decide(model="opus", now=self.now)
        self.assertFalse(decision.admit)  # 950 + 100 > 1000
        self.assertAlmostEqual(decision.wait_s, FIVE_HOURS - 10, delta=2)


if __name__ == "__main__":
    unittest.main()
