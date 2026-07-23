"""R2 Round-3 (A7): a cancellation landing DURING metering must not lose it.

R1-A2 fixed the cancel-before-metering case with a finally fallback, but marked
``usage_recorded = True`` BEFORE record_cli_usage ran. A cancellation arriving
while record_cli_usage was in flight (e.g. mid rate-limit scan) therefore
skipped the fallback and dropped the metering — including any detected provider
limit. Metering now runs as a retained, shielded task the finally awaits, so it
completes exactly once whether or not a cancel lands mid-metering.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.configurable_cli import ConfigurableCLIAgent  # noqa: E402
from proofstack.context import RunContext  # noqa: E402


class _SlowMeterCLIAgent(ConfigurableCLIAgent):
    """record_cli_usage blocks until released, so a cancel can land inside it."""

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.usage_calls: list[str] = []
        self.metering_started = asyncio.Event()
        self.release_metering = asyncio.Event()

    async def record_cli_usage(self, stdout_text, stderr_text, done) -> None:
        self.metering_started.set()
        await self.release_metering.wait()
        self.usage_calls.append(done.status)


def _ctx(tmp: str, cmd: list[str]) -> RunContext:
    return RunContext.create(
        run_id="a7",
        root_workdir=tmp,
        flat=True,
        component_configs={"cfg_cli": {"cmd": cmd, "sandbox": {"backend": "subprocess"}}},
    )


class _SleepMeterCLIAgent(ConfigurableCLIAgent):
    """record_cli_usage takes a beat, so two cancels can straddle it."""

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.usage_calls: list[str] = []
        self.metering_started = asyncio.Event()

    async def record_cli_usage(self, stdout_text, stderr_text, done) -> None:
        self.metering_started.set()
        await asyncio.sleep(0.25)
        self.usage_calls.append(done.status)


class DoubleCancelTests(unittest.TestCase):
    def test_second_cancel_does_not_abandon_metering(self) -> None:
        async def drive() -> list[str]:
            with tempfile.TemporaryDirectory() as tmp:
                ctx = _ctx(tmp, ["sh", "-c", "finish '{\"status\":\"done\",\"summary\":\"ok\"}'"])
                agent = _SleepMeterCLIAgent(ctx, name="cfg_cli")
                task = asyncio.ensure_future(agent())
                await asyncio.wait_for(agent.metering_started.wait(), timeout=10.0)
                task.cancel()
                await asyncio.sleep(0.05)  # still draining the retained meter task
                self.assertFalse(task.done())
                task.cancel()  # a SECOND cancel must not abandon it
                with self.assertRaises(asyncio.CancelledError):
                    await task
                return agent.usage_calls

        self.assertEqual(asyncio.run(drive()), ["done"])


class CancelDuringMeteringTests(unittest.TestCase):
    def test_cancel_while_metering_still_records_once(self) -> None:
        async def drive() -> list[str]:
            with tempfile.TemporaryDirectory() as tmp:
                ctx = _ctx(tmp, ["sh", "-c", "finish '{\"status\":\"done\",\"summary\":\"ok\"}'"])
                agent = _SlowMeterCLIAgent(ctx, name="cfg_cli")
                task = asyncio.ensure_future(agent())
                # wait until metering is in flight, THEN cancel: pre-fix this
                # dropped the metering entirely.
                await asyncio.wait_for(agent.metering_started.wait(), timeout=10.0)
                task.cancel()
                agent.release_metering.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                return agent.usage_calls

        calls = asyncio.run(drive())
        # metered exactly once (the real done record), neither lost nor doubled
        self.assertEqual(calls, ["done"])


if __name__ == "__main__":
    unittest.main()
