"""R2 Round-4 focused-delta review: three regressions the durable-boundary /
B3-label commits themselves introduced.

F1 — _window_statuses read the block boundary only from calibration, but
     pre-delta stores persisted a window's reset solely in probes.json. After
     upgrade, until the next probe rewrote it durably, prior-block usage was
     re-counted. The probe's own reset is now a boundary fallback.

F2 — record_probe now persists a reset on every routine probe, creating
     calibration["claude"]; _provider_calibration then returned only that
     scoped dict, shadowing legacy pre-namespacing top-level entries (dropping
     a weekly ceiling / active block). Legacy entries are now merged beneath.

F3 — the B3 label read the whole surrounding excerpt, so unrelated "weekly"
     prose near an explicit "5-hour limit reached" flipped the window. A label
     inside the matched signal is now authoritative.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from proofstack.subscription import (  # noqa: E402
    SubscriptionPacer,
    SubscriptionStore,
    detect_rate_limit,
)

NOW = 3_000_000.0


class PreDeltaProbeBoundaryTests(unittest.TestCase):
    """F1: a reset persisted only in probes.json still bounds the block."""

    def test_pre_delta_probe_reset_excludes_prior_block_usage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SubscriptionStore(home=Path(td))
            store.save_settings(
                {"enabled": True, "node_estimate_tokens": 50, "ceilings": {"weekly": 1_000_000}}
            )
            # pre-delta persisted probe: reset lives in probes.json, not calibration
            probes = {
                "claude": {
                    "weekly": {
                        "used_pct": 20.0,
                        "resets_at": NOW - 100,  # last reset already passed
                        "ts": NOW - 200,
                        "own_tokens": 0,
                    },
                    "_meta": {"last_attempt_ts": NOW - 200, "last_error": None},
                }
            }
            (Path(td) / "probes.json").write_text(json.dumps(probes), encoding="utf-8")
            # 5000 tokens spent BEFORE that reset -> prior block, must not count
            store.append_usage(provider="claude", model="opus", tokens=5000, now=NOW - 3600)
            dec = SubscriptionPacer(store, provider="claude").decide(model="opus", now=NOW)
            wk = next(st for st in dec.windows if st.window == "weekly")
            self.assertEqual(wk.usage, 0)


class LegacyCalibrationMergeTests(unittest.TestCase):
    """F2: creating the provider namespace must not shadow legacy top-level cal."""

    def test_routine_probe_does_not_drop_legacy_weekly_block(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SubscriptionStore(home=Path(td))
            store.save_settings(
                {
                    "enabled": True,
                    "node_estimate_tokens": 50,
                    "ceilings": {"five_hour": 1_000_000, "weekly": 1_000_000},
                    # pre-namespacing (top-level) weekly block
                    "calibration": {
                        "weekly": {
                            "observed_ceiling": 1_000_000,
                            "blocked_until": NOW + 1000,
                            "reset_at": NOW + 1000,
                        }
                    },
                }
            )
            self.assertFalse(
                SubscriptionPacer(store, provider="claude").decide(model="opus", now=NOW).admit
            )
            # a routine low-util five-hour probe (normal operation) persists a
            # scoped calibration["claude"] — the legacy weekly block must survive
            store.record_probe(
                provider="claude",
                windows={"five_hour": {"used_pct": 10.0, "resets_at": NOW + 3600}},
                now=NOW,
            )
            self.assertFalse(
                SubscriptionPacer(store, provider="claude").decide(model="opus", now=NOW).admit
            )


class DetectLabelInsideMatchWinsTests(unittest.TestCase):
    """F3: an explicit label in the matched signal beats unrelated nearby prose."""

    def test_five_hour_match_not_flipped_by_nearby_weekly_prose(self) -> None:
        hit = detect_rate_limit(
            "Weekly usage remains available.\n5-hour limit reached", now=NOW
        )
        self.assertIsNotNone(hit)
        self.assertEqual(hit.window_guess, "five_hour")

    def test_unlabeled_match_still_reads_weekly_prefix_from_context(self) -> None:
        # the labelled-prefix case B3 originally fixed still works
        hit = detect_rate_limit(f"Weekly usage limit reached|{int(NOW + 3600)}", now=NOW)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.window_guess, "weekly")


if __name__ == "__main__":
    unittest.main()
