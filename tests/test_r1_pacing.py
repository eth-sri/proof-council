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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.dev_data import _executor_scaffold  # noqa: E402
from proofstack.agents.configurable_cli import (  # noqa: E402
    ConfigurableCLIAgent,
    _answer_free_stdout,
)
from proofstack.context import RunContext  # noqa: E402
from proofstack.subscription import (  # noqa: E402
    SubscriptionPacer,
    SubscriptionStore,
    detect_rate_limit,
)


class ProbeOnlyAccountCapTests(unittest.TestCase):
    """S-A1: a probe-only (ceiling==0) window at the hard account cap must not
    admit pending spend, but must still admit when idle or below the cap."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SubscriptionStore(home=Path(self._tmp.name))
        self.codex = SubscriptionPacer(self.store, provider="codex")
        self.now = 3_000_000.0

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _probe(self, used_pct: float) -> None:
        # no ceilings/seeds for codex -> the weekly window is probe-only
        self.store.record_probe(
            provider="codex",
            windows={"weekly": {"used_pct": used_pct, "resets_at": self.now + 86400}},
            now=self.now,
        )

    def test_pending_est_at_cap_blocks(self) -> None:
        self.store.save_settings(
            {"enabled": True, "account_cap_pct": {"weekly": 90}, "node_estimate_tokens": 50_000}
        )
        self._probe(90.0)
        decision = self.codex.decide(model="gpt-5.6-sol", now=self.now)
        self.assertFalse(decision.admit)
        self.assertEqual(decision.blocking_window, "weekly")
        self.assertGreater(decision.wait_s, 0.0)
        five = next(st for st in decision.windows if st.window == "weekly")
        self.assertEqual(five.ceiling, 0)  # probe-only

    def test_pending_claim_at_cap_blocks(self) -> None:
        self.store.save_settings(
            {"enabled": True, "account_cap_pct": {"weekly": 90}, "node_estimate_tokens": 0}
        )
        self._probe(90.0)
        self.store.add_claim(
            provider="codex", model="gpt-5.6-sol", est_tokens=2_000_000, ttl_s=300, now=self.now
        )
        decision = self.codex.decide(model="gpt-5.6-sol", now=self.now)
        self.assertFalse(decision.admit)
        self.assertEqual(decision.blocking_window, "weekly")

    def test_idle_at_cap_still_admits(self) -> None:
        # zero pending tokens (no est, no claims, no own spend) -> nothing to gate
        self.store.save_settings(
            {"enabled": True, "account_cap_pct": {"weekly": 90}, "node_estimate_tokens": 0}
        )
        self._probe(90.0)
        decision = self.codex.decide(model="gpt-5.6-sol", now=self.now)
        self.assertTrue(decision.admit)

    def test_below_cap_still_admits(self) -> None:
        self.store.save_settings(
            {"enabled": True, "account_cap_pct": {"weekly": 90}, "node_estimate_tokens": 50_000}
        )
        self._probe(50.0)
        decision = self.codex.decide(model="gpt-5.6-sol", now=self.now)
        self.assertTrue(decision.admit)


class CodexScaffoldTests(unittest.TestCase):
    """S-A2: the editor's codex scaffold must not emit --ephemeral, which would
    kill the session rollout the pacer harvests for codex probes."""

    def test_codex_cli_scaffold_omits_ephemeral(self) -> None:
        cmd = _executor_scaffold("codex_cli")["cmd"]
        self.assertIn("codex", cmd)
        self.assertIn("exec", cmd)
        self.assertNotIn("--ephemeral", cmd)


class DiscreteResetWaitTests(unittest.TestCase):
    """S-A5: own-spend deficit on a window with a discrete provider reset caps
    the wait at reset time, not the ~7-day rolling ledger age-out."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SubscriptionStore(home=Path(self._tmp.name))
        self.pacer = SubscriptionPacer(self.store, provider="claude")
        self.now = 3_000_000.0

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_own_spend_wait_capped_at_discrete_reset(self) -> None:
        self.store.save_settings(
            {
                "enabled": True,
                "ceilings": {"weekly": 1000},
                "cap_pct": {"weekly": 100},
                "account_cap_pct": {"weekly": 100},  # keep the account gate open
                "node_estimate_tokens": 50,
            }
        )
        # provider reports a discrete reset in one hour, low utilization
        self.store.record_probe(
            provider="claude",
            windows={"weekly": {"used_pct": 50.0, "resets_at": self.now + 3600}},
            now=self.now,
        )
        # own spend booked BEFORE the probe (own_since_probe stays 0), inside the block
        self.store.append_usage(
            provider="claude", model="sonnet", tokens=1500, now=self.now - 100_000
        )
        decision = self.pacer.decide(model="sonnet", now=self.now)
        self.assertFalse(decision.admit)
        self.assertEqual(decision.blocking_window, "weekly")
        # capped at reset (~3600s), NOT the rolling age-out (~504800s)
        self.assertAlmostEqual(decision.wait_s, 3600, delta=2)


class AnswerFreeScanTests(unittest.TestCase):
    """S-A9: a rate-limit phrase quoted inside an assistant stream-json envelope
    must not be scanned as a real limit; a bare-line limit still is."""

    def _scan(self, stdout_text: str, stderr_text: str = "") -> object:
        # mirrors configurable_cli._scan_claude_rate_limit's scan expression
        return detect_rate_limit(_answer_free_stdout(stdout_text) + "\n" + stderr_text)

    def test_phrase_inside_assistant_answer_is_ignored(self) -> None:
        reset = int(time.time() + 3000)
        answer = f"The docs say a real hit prints Claude AI usage limit reached|{reset} on stderr."
        stdout = "\n".join(
            [
                json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": answer}]}}),
                json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "done"}),
            ]
        )
        self.assertIsNone(self._scan(stdout))

    def test_phrase_inside_user_turn_is_ignored(self) -> None:
        stdout = json.dumps(
            {"type": "user", "message": {"content": [{"type": "text", "text": "weekly limit reached?"}]}}
        )
        self.assertIsNone(self._scan(stdout))

    def test_real_bare_line_limit_still_detected(self) -> None:
        now = time.time()
        reset = int(now + 3000)
        stdout = "\n".join(
            [
                json.dumps({"type": "result", "subtype": "success", "result": "ok"}),
                f"Claude AI usage limit reached|{reset}",  # bare, non-JSON line
            ]
        )
        hit = detect_rate_limit(_answer_free_stdout(stdout), now=now)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.reset_at, float(reset))


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


class AnswerQuoteIntegrationTests(unittest.TestCase):
    """S-A9 end-to-end: a successful node whose assistant answer quotes the
    limit phrase records usage but writes NO blocking calibration."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name) / "pchome"
        self._env = mock.patch.dict(os.environ, {"PROOFCOUNCIL_HOME": str(self.home)})
        self._env.start()
        self.store = SubscriptionStore(home=self.home)

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def _cmd(self, reset: int) -> list[str]:
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
        # phrase lives ONLY in an assistant envelope; no single quotes in the text
        assistant = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"A real hit prints Claude AI usage limit reached|{reset} then stops.",
                        }
                    ]
                },
            }
        )
        script = (
            f"cat > /dev/null; printf '%s\\n' '{result}'; printf '%s\\n' '{assistant}'; "
            "finish '{\"status\":\"done\",\"summary\":\"ok\"}'"
        )
        return ["sh", "-c", script]

    def test_quoted_phrase_records_no_calibration(self) -> None:
        self.store.save_settings({"enabled": False})
        reset = int(time.time() + 3000)
        with tempfile.TemporaryDirectory() as workdir:
            ctx = RunContext.create(
                run_id="test-r1",
                root_workdir=workdir,
                flat=True,
                component_configs=_claude_component(self._cmd(reset)),
            )
            out = asyncio.run(ConfigurableCLIAgent(ctx, name="cfg_cli")(problem="P"))
        self.assertEqual(out.status, "done")
        # usage still metered
        entries = self.store.window_entries(provider="claude", seconds=3600)
        self.assertEqual([e.tokens for e in entries], [100])
        # but NO limit was recorded from the quoted answer
        claude_cal = self.store.load_settings().get("calibration", {}).get("claude", {})
        self.assertNotIn("five_hour", claude_cal)
        self.assertNotIn("weekly", claude_cal)


class PartialAccountCapMergeTests(unittest.TestCase):
    """S-A10: a partial account_cap_pct update merges per window, like cap_pct."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = SubscriptionStore(home=Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_partial_update_retains_hidden_windows(self) -> None:
        self.store.save_settings(
            {
                "account_cap_pct": {"five_hour": 90, "weekly": 90, "weekly_opus": 40, "weekly_fable": 55},
                "cap_pct": {"five_hour": 50, "weekly": 30, "weekly_opus": 10, "weekly_fable": 12},
            }
        )
        # UI saves only the two visible windows
        self.store.save_settings(
            {"account_cap_pct": {"five_hour": 80, "weekly": 70}, "cap_pct": {"five_hour": 55, "weekly": 33}}
        )
        settings = self.store.load_settings()
        self.assertEqual(
            settings["account_cap_pct"],
            {"five_hour": 80, "weekly": 70, "weekly_opus": 40, "weekly_fable": 55},
        )
        # parity: cap_pct already behaved this way
        self.assertEqual(settings["cap_pct"]["weekly_opus"], 10)


if __name__ == "__main__":
    unittest.main()
