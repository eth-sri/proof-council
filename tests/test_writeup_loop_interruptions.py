"""Completed replies must survive interrupted bookkeeping and parallel reuse."""

import asyncio
from unittest.mock import patch

import pytest

from proofstack.agents.writeup_codex_seat import CodexSeatResult
from proofstack.agents.writeup_loop import GateResult, WriteupLoop
from proofstack.context import RunContext
from proofstack.events import EventEmitter


DOC = "\\documentclass{article}\n\\begin{document}\nOriginal.\n\\end{document}\n"
REWRITE = DOC.replace("Original.", "Rewritten, with an incorrect lemma.")
GOOD = GateResult(ok=True, compiled=True, pages=1)


@pytest.mark.parametrize("interruption", [asyncio.CancelledError, OSError])
@pytest.mark.parametrize("event", ["model.call", "agent.end", "agent.cache_hit"])
def test_completed_api_unable_survives_bookkeeping(tmp_path, interruption, event):
    replies = iter([
        REWRITE,
        "ERRORS FOUND\nLemma is false.",
        REWRITE + "UNABLE: Lemma is false.\n",
    ])
    seen = []

    class FakeClient:
        model = "offline-test"

        def run_queries(self, queries, **kwargs):
            reply = next(replies)
            seen.append(reply)
            yield 0, [{"role": "assistant", "content": reply}], {
                "cost": 0.01, "input_tokens": 10, "output_tokens": 10,
            }

        def terminate(self):
            pass

    ctx = RunContext.create(
        root_workdir=tmp_path, flat=True,
        api_client_factory=lambda cfg: FakeClient())
    loop = WriteupLoop(ctx)
    original_emit = EventEmitter.emit

    async def interrupt(self, kind, payload=None, **kwargs):
        if self.agent == "repair-r1" and kind == event:
            raise interruption("interrupted while recording a completed repair")
        return await original_emit(self, kind, payload, **kwargs)

    with patch.object(WriteupLoop, "_gate", return_value=GOOD):
        if event == "agent.cache_hit":
            asyncio.run(loop(document_text=DOC, rounds=1))
        with patch.object(EventEmitter, "emit", interrupt):
            out = asyncio.run(loop(document_text=DOC, rounds=1))

    assert len(seen) == 3
    assert out.document_text == DOC
    assert not out.improved
    assert "UNABLE" in out.shipped
    assert out.usd_total == pytest.approx(0.03)
    assert loop._discarded_reply is None


def test_parallel_invocations_keep_their_own_reply_and_deadline(tmp_path):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True)
    loop = WriteupLoop(ctx)
    loop.seat = "cli"
    deadlines = {}

    async def scenario():
        repair_arrived = asyncio.Event()
        second_started = asyncio.Event()
        never = asyncio.Event()
        replies = iter([
            REWRITE,
            "ERRORS FOUND\nLemma is false.",
            REWRITE + "UNABLE: Lemma is false.\n",
        ])
        record_usage = loop._record_cli_usage

        async def fake_cli(prompt, **kwargs):
            if asyncio.current_task().get_name() == "document-B":
                deadlines["B"] = loop._deadline
                second_started.set()
                await never.wait()
            return CodexSeatResult(text=next(replies))

        async def suspended_usage(name, result):
            if name == "repair-r1":
                assert "UNABLE:" in loop._discarded_reply[1]
                deadlines["A-before"] = loop._deadline
                repair_arrived.set()
                try:
                    await never.wait()
                finally:
                    deadlines["A-after"] = loop._deadline
            await record_usage(name, result)

        def remaining():
            return 7200 if asyncio.current_task().get_name() == "document-B" else 3600

        with patch("proofstack.agents.writeup_loop.run_codex_seat", fake_cli), \
             patch("proofstack.agents.writeup_loop.redact_seat_text", lambda text: text), \
             patch.object(loop, "_record_cli_usage", suspended_usage), \
             patch.object(loop, "_remaining_wallclock", remaining), \
             patch.object(WriteupLoop, "_gate", return_value=GOOD):
            first = asyncio.create_task(loop(document_text=DOC, rounds=1), name="document-A")
            second = None
            try:
                await asyncio.wait_for(repair_arrived.wait(), timeout=5)
                second = asyncio.create_task(
                    loop(document_text=DOC.replace("Original", "Other"), rounds=1),
                    name="document-B")
                await asyncio.wait_for(second_started.wait(), timeout=5)
                first.cancel()
                return await first
            finally:
                tasks = [task for task in (first, second) if task is not None]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    out = asyncio.run(scenario())
    assert out.document_text == DOC
    assert "UNABLE" in out.shipped
    assert deadlines["A-before"] == deadlines["A-after"]
    assert deadlines["B"] > deadlines["A-before"] + 3500
    assert loop._discarded_reply is None
