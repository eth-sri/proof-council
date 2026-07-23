"""R1-A2: cancelling a CLI node mid-run must still meter the partial usage the
CLI already spent, BEFORE the pacing claim is released. Otherwise the subscription
pacer loses those tokens and over-admits the next run.

The normal metering happens at ``record_cli_usage`` (kinds/cli.py, right after
``_wait_for_done``). A cancellation lands inside ``_wait_for_done`` and skips it,
so the fix re-meters from the finally block. These tests pin both: the fallback
fires on cancel, and it does NOT double-meter on a clean completion.
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
from proofstack.kinds.cli import CLIAgent  # noqa: E402


class _RecordingCLIAgent(ConfigurableCLIAgent):
    """Counts record_cli_usage calls and signals once the child has spawned."""

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.usage_calls: list[str] = []
        self.spawned = asyncio.Event()

    def cli_input(self, inp):  # called after the stream is launched
        text = super().cli_input(inp)
        self.spawned.set()
        return text

    async def record_cli_usage(self, stdout_text, stderr_text, done) -> None:
        self.usage_calls.append(done.status)


def _ctx(tmp: str, cmd: list[str]) -> RunContext:
    return RunContext.create(
        run_id="a2",
        root_workdir=tmp,
        flat=True,
        component_configs={
            "cfg_cli": {
                "cmd": cmd,
                "sandbox": {"backend": "subprocess"},
            }
        },
    )


class CLICancelMetersPartialUsageTests(unittest.TestCase):
    def test_cancel_mid_run_still_meters_partial_usage(self) -> None:
        async def drive() -> list[str]:
            with tempfile.TemporaryDirectory() as tmp:
                # Never writes done.json -> run() parks in _wait_for_done.
                ctx = _ctx(tmp, ["sh", "-c", "sleep 30"])
                agent = _RecordingCLIAgent(ctx, name="cfg_cli")
                with __import__("unittest.mock", fromlist=["patch"]).patch.object(
                    CLIAgent, "POLL_INTERVAL_S", 0.02
                ):
                    task = asyncio.ensure_future(agent())
                    await asyncio.wait_for(agent.spawned.wait(), timeout=10.0)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                return agent.usage_calls

        calls = asyncio.run(drive())
        # Pre-fix: [] (usage silently dropped). Post-fix: one partial metering.
        self.assertEqual(calls, ["partial"])

    def test_clean_completion_meters_exactly_once(self) -> None:
        async def drive() -> list[str]:
            with tempfile.TemporaryDirectory() as tmp:
                ctx = _ctx(
                    tmp,
                    ["sh", "-c", "finish '{\"status\":\"done\",\"summary\":\"ok\"}'"],
                )
                agent = _RecordingCLIAgent(ctx, name="cfg_cli")
                await agent()
                return agent.usage_calls

        calls = asyncio.run(drive())
        # The finally must NOT double-meter after a normal record_cli_usage.
        self.assertEqual(calls, ["done"])


if __name__ == "__main__":
    unittest.main()
