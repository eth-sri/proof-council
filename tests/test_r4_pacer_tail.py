"""R2 Round-4 tail: two pre-existing pacer edge cases.

B3 — detect_rate_limit read the window label off only ``m.group(0)``. The
     first pattern captures from "usage limit reached" on, so a
     "Weekly usage limit reached|<epoch>" hit lost its "Weekly" prefix and,
     if the weekly reset happened to be soon, was calibrated as five_hour.
     The label is now read off the surrounding text.

B7 — _acquire_subscription_slot registered a claim via try_claim, then
     emitted "pacing.admit" before returning it to run(). A cancellation
     landing during that emit left the claim registered but never handed to
     run()'s finally, so it held phantom headroom until its TTL. The claim is
     now released if the emit is cancelled.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import proofstack.kinds.cli as cli_module  # noqa: E402
from proofstack.agents.configurable_cli import ConfigurableCLIAgent  # noqa: E402
from proofstack.context import RunContext  # noqa: E402
from proofstack.subscription import (  # noqa: E402
    FIVE_HOURS_S,
    PacingDecision,
    detect_rate_limit,
)


class DetectWindowLabelTests(unittest.TestCase):
    """B3: an explicit window label out of the matched span still classifies."""

    def test_explicit_weekly_with_soon_reset_is_weekly(self) -> None:
        now = 1_700_000_000.0
        soon = int(now + 3600)  # 1h out -> the delta heuristic alone says five_hour
        hit = detect_rate_limit(f"Weekly usage limit reached|{soon}", now=now)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.window_guess, "weekly")

    def test_plain_five_hour_message_stays_five_hour(self) -> None:
        now = 1_700_000_000.0
        soon = int(now + 3600)
        hit = detect_rate_limit(f"Claude AI usage limit reached|{soon}", now=now)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.window_guess, "five_hour")

    def test_labelled_five_hour_far_reset_stays_five_hour(self) -> None:
        # a "5-hour" label wins over the delta heuristic (which, at a far reset,
        # would otherwise guess weekly)
        now = 1_700_000_000.0
        far = int(now + FIVE_HOURS_S + 7200)
        hit = detect_rate_limit(f"5-hour limit reached, resets {far}", now=now)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.window_guess, "five_hour")


async def _immediate_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


class ClaimLeakOnCancelTests(unittest.TestCase):
    """B7: a cancel during the admit emit releases the just-registered claim."""

    def test_cancel_during_admit_event_releases_new_claim(self) -> None:
        live_claims: set[str] = set()
        emit_started = asyncio.Event()

        class FakePacer:
            def __init__(self, *, provider):
                self.provider = provider

            def gate_config(self):
                return True, 300.0

            def try_claim(self, **kwargs):
                live_claims.add("claim")
                return "claim", PacingDecision(admit=True, est_tokens=100)

            def release(self, claim_id):
                live_claims.discard(claim_id)

        async def drive() -> None:
            with tempfile.TemporaryDirectory() as td:
                ctx = RunContext.create(
                    run_id="claim",
                    root_workdir=Path(td),
                    flat=True,
                    component_configs={
                        "cli": {"cmd": ["claude"], "usage": {"type": "claude_json"}}
                    },
                )
                agent = ConfigurableCLIAgent(ctx, name="cli")

                async def blocking_emit(kind, *args, **kwargs):
                    if kind == "pacing.admit":
                        emit_started.set()
                        await asyncio.Event().wait()

                agent.events.emit = blocking_emit
                task = asyncio.create_task(agent._acquire_subscription_slot())
                await asyncio.wait_for(emit_started.wait(), 1)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        orig = asyncio.to_thread
        asyncio.to_thread = _immediate_to_thread
        real_pacer = cli_module.SubscriptionPacer
        cli_module.SubscriptionPacer = FakePacer
        try:
            asyncio.run(drive())
        finally:
            cli_module.SubscriptionPacer = real_pacer
            asyncio.to_thread = orig

        self.assertEqual(live_claims, set())


if __name__ == "__main__":
    unittest.main()
