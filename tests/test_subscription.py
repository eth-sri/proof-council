from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.configurable_cli import ConfigurableCLIAgent  # noqa: E402
from proofstack.context import RunContext  # noqa: E402
from proofstack.subscription import (  # noqa: E402
    DEFAULT_NODE_ESTIMATE_TOKENS,
    DEFAULT_PARK_AFTER_S,
    SubscriptionPacer,
    SubscriptionParked,
    SubscriptionStore,
    detect_rate_limit,
    parse_claude_probe_json,
    parse_codex_rollout_rate_limits,
)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SubscriptionStore(home=Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_window_entries_filter_by_time_provider_and_model(self) -> None:
        now = 1_000_000.0
        self.store.append_usage(provider="claude", model="opus", tokens=100, now=now - 10)
        self.store.append_usage(provider="claude", model="claude-fable-5", tokens=200, now=now - 20)
        self.store.append_usage(provider="claude", model="opus", tokens=999, now=now - 7200)
        self.store.append_usage(provider="codex", model="gpt-5.6-sol", tokens=50, now=now - 10)

        hour = self.store.window_entries(provider="claude", seconds=3600, now=now)
        self.assertEqual(sum(e.tokens for e in hour), 300)

        fable = self.store.window_entries(
            provider="claude", seconds=3600, model_filter="fable", now=now
        )
        self.assertEqual([e.tokens for e in fable], [200])

        both = self.store.window_entries(provider="claude", seconds=86400, now=now)
        self.assertEqual(sum(e.tokens for e in both), 1299)

    def test_claims_expire_and_release(self) -> None:
        now = 1_000_000.0
        cid = self.store.add_claim(
            provider="claude", model="opus", est_tokens=500, ttl_s=100, now=now
        )
        self.assertEqual(self.store.claims_total(provider="claude", now=now + 50), 500)
        self.assertEqual(self.store.claims_total(provider="claude", now=now + 150), 0)
        self.store.remove_claim(cid)
        self.assertEqual(self.store.claims_total(provider="claude", now=now + 50), 0)

    def test_settings_roundtrip_and_merge(self) -> None:
        settings = self.store.load_settings()
        self.assertFalse(settings["enabled"])
        self.assertEqual(settings["park_after_s"], DEFAULT_PARK_AFTER_S)

        self.store.save_settings({"enabled": True, "cap_pct": {"five_hour": 25}})
        settings = self.store.load_settings()
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["cap_pct"]["five_hour"], 25)
        # untouched keys keep their defaults after a partial save
        self.assertEqual(settings["cap_pct"]["weekly"], 30)

        # unknown keys are dropped, not persisted
        self.store.save_settings({"bogus": 1})
        self.assertNotIn("bogus", self.store.load_settings())

    def test_record_rate_limit_keeps_max_observed(self) -> None:
        self.store.record_rate_limit(
            provider="claude", window="five_hour", observed_ceiling=900, reset_at=None
        )
        self.store.record_rate_limit(
            provider="claude", window="five_hour", observed_ceiling=400, reset_at=123.0
        )
        cal = self.store.load_settings()["calibration"]["claude"]["five_hour"]
        self.assertEqual(cal["observed_ceiling"], 900)
        self.assertEqual(cal["blocked_until"], 123.0)


class DetectRateLimitTests(unittest.TestCase):
    def test_claude_cli_epoch_format(self) -> None:
        now = time.time()
        reset = int(now + 3000)
        hit = detect_rate_limit(f"Claude AI usage limit reached|{reset}", now=now)
        assert hit is not None
        self.assertEqual(hit.reset_at, float(reset))
        self.assertEqual(hit.window_guess, "five_hour")

    def test_millisecond_epoch_and_weekly_guess(self) -> None:
        now = time.time()
        reset_ms = int((now + 2 * 86400) * 1000)
        hit = detect_rate_limit(f"usage limit reached|{reset_ms}", now=now)
        assert hit is not None
        self.assertAlmostEqual(hit.reset_at or 0, now + 2 * 86400, delta=2)
        self.assertEqual(hit.window_guess, "weekly")

    def test_weekly_phrase_without_epoch(self) -> None:
        hit = detect_rate_limit("Error: weekly limit reached, try later")
        assert hit is not None
        self.assertIsNone(hit.reset_at)
        self.assertEqual(hit.window_guess, "weekly")

    def test_implausible_epoch_is_dropped_but_hit_kept(self) -> None:
        now = time.time()
        hit = detect_rate_limit(f"usage limit reached|{int(now - 500)}", now=now)
        assert hit is not None
        self.assertIsNone(hit.reset_at)

    def test_no_match(self) -> None:
        self.assertIsNone(detect_rate_limit("all fine, wrote answer.tex"))
        self.assertIsNone(detect_rate_limit(""))


class PacerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SubscriptionStore(home=Path(self._tmp.name))
        self.pacer = SubscriptionPacer(self.store)
        self.now = 2_000_000.0
        self.store.save_settings(
            {
                "enabled": True,
                "plan": "max_5x",
                "ceilings": {"five_hour": 1000, "weekly": 10000},
                "cap_pct": {"five_hour": 50, "weekly": 50},
                "node_estimate_tokens": 100,
            }
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_untouched_window_always_admits(self) -> None:
        # estimate exceeds allowed, but an empty window must never stall
        self.store.save_settings({"node_estimate_tokens": 999_999})
        decision = self.pacer.decide(model="opus", now=self.now)
        self.assertTrue(decision.admit)

    def test_admits_under_cap(self) -> None:
        self.store.append_usage(provider="claude", model="opus", tokens=300, now=self.now - 60)
        decision = self.pacer.decide(model="opus", now=self.now)
        self.assertTrue(decision.admit)  # 300 + 100 <= 500

    def test_blocks_over_cap_with_ageout_wait(self) -> None:
        self.store.append_usage(provider="claude", model="opus", tokens=450, now=self.now - 1000)
        decision = self.pacer.decide(model="opus", now=self.now)
        self.assertFalse(decision.admit)  # 450 + 100 > 500
        self.assertEqual(decision.blocking_window, "five_hour")
        # admits once the 450-token entry ages out of the 5h window
        self.assertAlmostEqual(decision.wait_s, 5 * 3600 - 1000, delta=1)

    def test_claims_count_against_headroom(self) -> None:
        self.store.append_usage(provider="claude", model="opus", tokens=200, now=self.now - 60)
        self.store.add_claim(
            provider="claude", model="opus", est_tokens=250, ttl_s=3600, now=self.now - 1
        )
        decision = self.pacer.decide(model="opus", now=self.now)
        self.assertFalse(decision.admit)  # 200 + 250 + 100 > 500

    def test_cap_pct_scales_allowed(self) -> None:
        self.store.save_settings({"cap_pct": {"five_hour": 100}})
        self.store.append_usage(provider="claude", model="opus", tokens=850, now=self.now - 60)
        decision = self.pacer.decide(model="opus", now=self.now)
        self.assertTrue(decision.admit)  # 850 + 100 <= 1000 at 100%

    def test_calibrated_ceiling_beats_seed_and_manual_beats_calibrated(self) -> None:
        self.store.save_settings(
            {
                "ceilings": {"five_hour": None, "weekly": None},
                "cap_pct": {"five_hour": 100, "weekly": 100},
            }
        )
        self.store.record_rate_limit(
            provider="claude", window="five_hour", observed_ceiling=600, reset_at=None
        )
        status = self.pacer.status(now=self.now)
        five = next(w for w in status["windows"] if w["window"] == "five_hour")
        self.assertEqual(five["ceiling"], 600)
        self.assertEqual(five["ceiling_source"], "calibrated")

        self.store.save_settings({"ceilings": {"five_hour": 2000}})
        status = self.pacer.status(now=self.now)
        five = next(w for w in status["windows"] if w["window"] == "five_hour")
        self.assertEqual(five["ceiling"], 2000)
        self.assertEqual(five["ceiling_source"], "manual")

    def test_blocked_until_from_calibration_blocks_and_expires(self) -> None:
        self.store.record_rate_limit(
            provider="claude",
            window="five_hour",
            observed_ceiling=400,
            reset_at=self.now + 1800,
        )
        # a usage entry so the untouched-window rule doesn't apply
        self.store.append_usage(provider="claude", model="opus", tokens=10, now=self.now - 60)
        decision = self.pacer.decide(model="opus", now=self.now)
        self.assertFalse(decision.admit)
        self.assertAlmostEqual(decision.wait_s, 1800, delta=1)
        decision = self.pacer.decide(model="opus", now=self.now + 1900)
        self.assertTrue(decision.admit)

    def test_model_class_window_only_applies_to_matching_models(self) -> None:
        self.store.save_settings(
            {"ceilings": {"weekly_fable": 100}, "cap_pct": {"weekly_fable": 100}}
        )
        self.store.append_usage(
            provider="claude", model="claude-fable-5", tokens=90, now=self.now - 60
        )
        blocked = self.pacer.decide(model="claude-fable-5", now=self.now)
        self.assertFalse(blocked.admit)
        self.assertEqual(blocked.blocking_window, "weekly_fable")
        # a non-fable model is not constrained by the fable window
        ok = self.pacer.decide(model="opus", now=self.now)
        self.assertTrue(ok.admit)

    def test_blocking_window_names_the_longest_wait(self) -> None:
        # weekly blocks for days; five_hour blocks briefly — the decision
        # must name weekly, not whichever window happens to iterate last
        self.store.save_settings({"ceilings": {"five_hour": 1000, "weekly": 300}})
        self.store.append_usage(provider="claude", model="opus", tokens=450, now=self.now - 60)
        decision = self.pacer.decide(model="opus", now=self.now)
        self.assertFalse(decision.admit)
        self.assertEqual(decision.blocking_window, "weekly")
        self.assertAlmostEqual(decision.wait_s, 7 * 86400 - 60, delta=1)

    def test_try_claim_is_atomic_against_the_last_slot(self) -> None:
        # allowed=500, est=100: after 3 admits (usage 200 + claims 300 = 500)
        # a 4th must be denied — its own estimate no longer fits.
        self.store.append_usage(provider="claude", model="opus", tokens=200, now=self.now - 60)
        admitted = []
        for _ in range(4):
            claim_id, decision = self.pacer.try_claim(model="opus", now=self.now)
            if claim_id is not None:
                admitted.append(claim_id)
            else:
                self.assertEqual(decision.blocking_window, "five_hour")
        self.assertEqual(len(admitted), 3)
        # releasing one claim frees the slot again
        self.pacer.release(admitted[0])
        claim_id, _ = self.pacer.try_claim(model="opus", now=self.now)
        self.assertIsNotNone(claim_id)

    def test_estimate_prefers_manual_then_median_then_default(self) -> None:
        self.assertEqual(self.pacer.estimate_tokens(), 100)
        self.store.save_settings({"node_estimate_tokens": None})
        for tokens in (10, 20, 1000):
            self.store.append_usage(provider="claude", model="opus", tokens=tokens, now=self.now)
        self.assertEqual(self.pacer.estimate_tokens(), 20)

        empty = SubscriptionPacer(SubscriptionStore(home=Path(self._tmp.name) / "empty"))
        self.assertEqual(empty.estimate_tokens(), DEFAULT_NODE_ESTIMATE_TOKENS)

    def test_enabled_env_override(self) -> None:
        with mock.patch.dict(os.environ, {"PROOFCOUNCIL_PACING": "off"}):
            self.assertFalse(self.pacer.enabled())
        self.store.save_settings({"enabled": False})
        with mock.patch.dict(os.environ, {"PROOFCOUNCIL_PACING": "on"}):
            self.assertTrue(self.pacer.enabled())


class ProbeParsingTests(unittest.TestCase):
    def test_claude_probe_json(self) -> None:
        text = json.dumps(
            {
                "five_hour": {"utilization": 18.0, "resets_at": "2026-07-16T17:19:59+00:00"},
                "seven_day": {"utilization": 3.0, "resets_at": None},
                "seven_day_opus": None,
                "unrelated_key": {"utilization": 50},
            }
        )
        parsed = parse_claude_probe_json(text)
        self.assertEqual(set(parsed), {"five_hour", "weekly"})
        self.assertEqual(parsed["five_hour"]["used_pct"], 18.0)
        self.assertGreater(parsed["five_hour"]["resets_at"], 1.7e9)
        self.assertIsNone(parsed["weekly"]["resets_at"])
        self.assertEqual(parse_claude_probe_json("not json"), {})

    def test_codex_rollout_rate_limits(self) -> None:
        line = json.dumps(
            {
                "payload": {
                    "rate_limits": {
                        "primary": {
                            "used_percent": 12.5,
                            "window_minutes": 10080,
                            "resets_at": 1784810119,
                        },
                        "secondary": {
                            "used_percent": 40.0,
                            "window_minutes": 300,
                            "resets_in_seconds": 1800,
                        },
                    }
                }
            }
        )
        parsed = parse_codex_rollout_rate_limits("junk\n" + line, now=1_000_000.0)
        self.assertEqual(parsed["weekly"]["used_pct"], 12.5)
        self.assertEqual(parsed["weekly"]["resets_at"], 1784810119.0)
        self.assertEqual(parsed["five_hour"]["used_pct"], 40.0)
        self.assertEqual(parsed["five_hour"]["resets_at"], 1_000_000.0 + 1800)
        self.assertEqual(parse_codex_rollout_rate_limits("no snapshots here"), {})


class ProbeAndAccountGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SubscriptionStore(home=Path(self._tmp.name))
        self.pacer = SubscriptionPacer(self.store)
        self.now = 3_000_000.0
        self.store.save_settings(
            {
                "enabled": True,
                "ceilings": {"five_hour": 1000, "weekly": 100000},
                "cap_pct": {"five_hour": 100, "weekly": 100},
                "account_cap_pct": {"five_hour": 80, "weekly": 90},
                "node_estimate_tokens": 100,
            }
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_account_gate_blocks_on_outside_usage(self) -> None:
        # own ledger empty, but the account is already 85% used elsewhere
        self.store.record_probe(
            provider="claude",
            windows={"five_hour": {"used_pct": 85.0, "resets_at": self.now + 1200}},
            now=self.now,
        )
        decision = self.pacer.decide(model="opus", now=self.now)
        self.assertFalse(decision.admit)
        self.assertEqual(decision.blocking_window, "five_hour")
        self.assertAlmostEqual(decision.wait_s, 1200, delta=1)
        # after the reset the probe is history and the gate admits again
        decision = self.pacer.decide(model="opus", now=self.now + 1300)
        self.assertTrue(decision.admit)

    def test_account_gate_counts_own_drift_since_probe(self) -> None:
        self.store.record_probe(
            provider="claude",
            windows={"five_hour": {"used_pct": 70.0, "resets_at": self.now + 3600}},
            now=self.now,
        )
        # 70% + (40 own + 100 est)/1000 = 84% > 80% cap
        self.store.append_usage(provider="claude", model="opus", tokens=40, now=self.now + 10)
        decision = self.pacer.decide(model="opus", now=self.now + 20)
        self.assertFalse(decision.admit)
        # without the drift it would admit: 70 + 100/1000*100 = 80 <= 80
        self.store2 = None

    def test_probe_delta_calibrates_ceiling(self) -> None:
        self.store.save_settings({"ceilings": {"five_hour": None}})
        self.store.record_probe(
            provider="claude",
            windows={"five_hour": {"used_pct": 10.0, "resets_at": self.now + 9000}},
            now=self.now,
        )
        self.store.append_usage(
            provider="claude", model="opus", tokens=50_000, now=self.now + 60
        )
        self.store.record_probe(
            provider="claude",
            windows={"five_hour": {"used_pct": 20.0, "resets_at": self.now + 9000}},
            now=self.now + 120,
        )
        cal = self.store.load_settings()["calibration"]["claude"]["five_hour"]
        # 50k tokens moved the needle 10 points -> ceiling ~500k
        self.assertEqual(cal["observed_ceiling"], 500_000)
        self.assertEqual(cal["source"], "probe")

    def test_full_probe_marks_window_blocked(self) -> None:
        self.store.record_probe(
            provider="claude",
            windows={"five_hour": {"used_pct": 100.0, "resets_at": self.now + 2400}},
            now=self.now,
        )
        cal = self.store.load_settings()["calibration"]["claude"]["five_hour"]
        self.assertEqual(cal["blocked_until"], self.now + 2400)

    def test_block_alignment_uses_reset_time(self) -> None:
        # entry 4h old; block started 2h ago (resets in 3h) -> usage excluded
        self.store.record_probe(
            provider="claude",
            windows={"five_hour": {"used_pct": 5.0, "resets_at": self.now + 3 * 3600}},
            now=self.now,
        )
        self.store.append_usage(
            provider="claude", model="opus", tokens=900, now=self.now - 4 * 3600
        )
        decision = self.pacer.decide(model="opus", now=self.now)
        five = next(st for st in decision.windows if st.window == "five_hour")
        self.assertEqual(five.usage, 0)
        self.assertTrue(decision.admit)

    def test_codex_probe_only_window_gates_on_account_pct(self) -> None:
        codex = SubscriptionPacer(self.store, provider="codex")
        self.store.record_probe(
            provider="codex",
            windows={"weekly": {"used_pct": 95.0, "resets_at": self.now + 86400}},
            now=self.now,
        )
        decision = codex.decide(model="gpt-5.6-sol", now=self.now)
        self.assertFalse(decision.admit)
        self.assertEqual(decision.blocking_window, "weekly")
        # calibration namespaces keep codex data away from claude's
        self.assertNotIn(
            "weekly", (self.store.load_settings()["calibration"].get("claude") or {})
        )

    def test_probe_ttl_throttles_failed_probes(self) -> None:
        self.store.save_settings({"usage_probe_cmd": "false", "probe_ttl_s": 600})
        self.pacer.ensure_fresh_probe(now=self.now)
        meta = self.store.read_probes()["claude"]["_meta"]
        self.assertEqual(meta["last_attempt_ts"], self.now)
        self.assertIn("probe exited", meta["last_error"])
        # within the TTL the failing command is not re-run
        self.pacer.ensure_fresh_probe(now=self.now + 60)
        self.assertEqual(
            self.store.read_probes()["claude"]["_meta"]["last_attempt_ts"], self.now
        )

    def test_probe_cmd_records_windows(self) -> None:
        payload = json.dumps({"five_hour": {"utilization": 42.0, "resets_at": None}})
        self.store.save_settings({"usage_probe_cmd": f"printf '%s' '{payload}'"})
        self.pacer.ensure_fresh_probe(now=self.now)
        probe = self.store.read_probes()["claude"]["five_hour"]
        self.assertEqual(probe["used_pct"], 42.0)


def _claude_stub_cmd(extra_stdout: str = "") -> list[str]:
    result = json.dumps(
        {
            "type": "result",
            "num_turns": 2,
            "total_cost_usd": 0.0,
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
                "output_tokens": 40,
            },
        }
    )
    script = f"cat > /dev/null; echo '{result}'; "
    if extra_stdout:
        script += f"echo \"{extra_stdout}\"; "
    script += "finish '{\"status\":\"done\",\"summary\":\"ok\"}'"
    return ["sh", "-c", script]


def _claude_component(cmd: list[str]) -> dict:
    return {
        "cfg_cli": {
            "cmd": cmd,
            "model": "opus",
            "prompt": "Problem: {problem}",
            "sandbox": {"backend": "subprocess"},
            "usage": {"type": "claude_json"},
            "input_schema": {"problem": "string"},
            "output_schema": {"workspace": "string", "status": "string"},
            "done_outputs": {"status": "status"},
        }
    }


class GateIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name) / "pchome"
        self._env = mock.patch.dict(os.environ, {"PROOFCOUNCIL_HOME": str(self.home)})
        self._env.start()
        self.store = SubscriptionStore(home=self.home)

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def _run_node(self, cmd: list[str], workdir: str):
        ctx = RunContext.create(
            run_id="test-sub",
            root_workdir=workdir,
            flat=True,
            component_configs=_claude_component(cmd),
        )
        return asyncio.run(ConfigurableCLIAgent(ctx, name="cfg_cli")(problem="P"))

    def test_metered_node_appends_to_ledger_and_releases_claim(self) -> None:
        self.store.save_settings({"enabled": True, "node_estimate_tokens": 10})
        with tempfile.TemporaryDirectory() as workdir:
            out = self._run_node(_claude_stub_cmd(), workdir)
        self.assertEqual(out.status, "done")
        entries = self.store.window_entries(provider="claude", seconds=3600)
        self.assertEqual([e.tokens for e in entries], [100])  # 10+20+30+40
        self.assertEqual(entries[0].model, "opus")
        self.assertEqual(entries[0].run_id, "test-sub")
        self.assertEqual(self.store.claims_total(provider="claude"), 0)

    def test_pacing_disabled_records_ledger_but_never_gates(self) -> None:
        # over-cap ledger + pacing disabled -> node still runs
        self.store.save_settings({"enabled": False, "ceilings": {"five_hour": 1}})
        self.store.append_usage(provider="claude", model="opus", tokens=999_999)
        with tempfile.TemporaryDirectory() as workdir:
            out = self._run_node(_claude_stub_cmd(), workdir)
        self.assertEqual(out.status, "done")

    def test_exhausted_window_parks_run(self) -> None:
        self.store.save_settings(
            {
                "enabled": True,
                "ceilings": {"five_hour": 1000},
                "cap_pct": {"five_hour": 50},
                "node_estimate_tokens": 100,
                "park_after_s": 0,
            }
        )
        self.store.append_usage(provider="claude", model="opus", tokens=480)
        with tempfile.TemporaryDirectory() as workdir:
            with self.assertRaises(SubscriptionParked) as caught:
                self._run_node(_claude_stub_cmd(), workdir)
        self.assertEqual(caught.exception.scope, "subscription")
        self.assertEqual(caught.exception.window, "five_hour")

    def test_rate_limit_in_stdout_calibrates_and_blocks(self) -> None:
        self.store.save_settings({"enabled": False})
        reset = int(time.time() + 3000)
        with tempfile.TemporaryDirectory() as workdir:
            self._run_node(
                _claude_stub_cmd(f"Claude AI usage limit reached|{reset}"), workdir
            )
        cal = self.store.load_settings()["calibration"]["claude"]["five_hour"]
        self.assertEqual(cal["blocked_until"], float(reset))
        # observed ceiling includes the just-metered node
        self.assertEqual(cal["observed_ceiling"], 100)


if __name__ == "__main__":
    unittest.main()
