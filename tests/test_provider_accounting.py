"""Synthetic receipt races and real event-writer cancellation; no paid calls."""
import asyncio
from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import pytest

from mathagents.provider_trace import ProviderTrace, latest_attempts
from proofstack.agents.ac.ac_workflow import ACWorkflow, _sum_logged_model_cost
from proofstack.agents.ac.critic import ACCritic
from proofstack.context import RunContext
from proofstack.events import new_call_id
from proofstack.kinds.api_call import APICallAgent
from proofstack.provider_accounting import provider_usage_snapshot, record_provider_usage, settle_provider_usage


@pytest.fixture
def ctx(tmp_path):
    return RunContext.create(root_workdir=tmp_path, flat=True,
                             api_client_factory=lambda cfg: pytest.fail("no provider calls"))


def receipt(ctx, call_id="critic-call", cost=2):
    trace = ProviderTrace(ctx.root_workdir / f"agents/{call_id}/provider-attempts.jsonl", call_id=call_id)
    trace.update(("request", 0), response_id="saved", status="completed", cost=cost,
                 usage_unavailable=False, input_tokens=10, output_tokens=20, reasoning_tokens=5)
    return trace


def recovery(ctx, trace, mode):
    if mode == "outer":
        return ACWorkflow(ctx)._apply_resume_budget_offset
    critic = ACCritic(ctx)

    async def reconcile():
        await critic._reconcile_recovery(trace.path, latest_attempts(trace.path))

    return reconcile


def events(ctx):
    return [json.loads(line) for line in (ctx.root_workdir / "events.jsonl").read_text().splitlines()]


def test_call_ids_use_128_bits_of_randomness(monkeypatch):
    sizes = []

    def token_hex(size):
        sizes.append(size)
        return "ab" * size

    monkeypatch.setattr("proofstack.events.secrets.token_hex", token_hex)
    assert new_call_id() == "ab" * 16
    assert sizes == [16]


@pytest.mark.parametrize("first_id", ["abcdef", None])
def test_old_and_new_call_ids_keep_independent_charges(ctx, first_id):
    first_id = first_id or new_call_id()
    second_id = new_call_id()

    async def exercise():
        for call_id, cost in [(first_id, 4), (second_id, 7)]:
            receipt(ctx, call_id, cost)
            await record_provider_usage(ctx, ctx.budgets.root(), {"cost_usd": cost},
                                        call_id=call_id, emitter=ctx.events)
        await settle_provider_usage(ctx, ctx.budgets.root())
        assert ctx.budgets.root().counters.usd == 11

    asyncio.run(exercise())
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == 11


@pytest.mark.parametrize("metered", [None, "25", "25.0"])
def test_historical_numeric_strings_are_normalized_for_all_accounting(ctx, metered):
    trace = receipt(ctx, cost=2.5)
    payload = {"cost_usd": "2.5", "in_tokens": "10", "out_tokens": "20.0", "reasoning_tokens": "5"}
    if metered is not None:
        payload["metered_tokens"] = metered
    path = ctx.root_workdir / "events.jsonl"
    path.write_text(json.dumps({"kind": "model.call", "call_id": "critic-call", "payload": payload}) + "\n")
    snapshot = provider_usage_snapshot(ctx.root_workdir)
    assert snapshot.cost_usd == 2.5 and not snapshot.adjustments and not snapshot.damaged
    assert snapshot.tokens == (25 if metered is not None else 30)
    assert snapshot.billed_usage["critic-call"] == {
        "cost_usd": 2.5, "in_tokens": 10, "out_tokens": 20, "reasoning_tokens": 5}

    async def exercise():
        await recovery(ctx, trace, "outer")()
        await record_provider_usage(ctx, ctx.budgets.root(), {k: v for k, v in payload.items() if k != "metered_tokens"},
                                    call_id="critic-call", emitter=ctx.events)
        await record_provider_usage(ctx, ctx.budgets.root(), {"cost_usd": 4, "in_tokens": 3, "out_tokens": 2},
                                    call_id="new-call", emitter=ctx.events)
        await settle_provider_usage(ctx, ctx.budgets.root())
        assert ctx.budgets.root().counters.usd == 6.5
        assert ctx.budgets.root().counters.tokens == snapshot.tokens + 5

    asyncio.run(exercise())
    assert _sum_logged_model_cost(path) == 6.5


@pytest.mark.parametrize("field,value", [
    ("cost_usd", "not-a-number"), ("cost_usd", "NaN"), ("cost_usd", "Infinity"),
    ("cost_usd", -1), ("in_tokens", []), ("out_tokens", "2.5"),
    ("reasoning_tokens", {}), ("metered_tokens", "invalid"), ("in_tokens", True),
])
def test_invalid_historical_usage_is_reported_and_recovered_from_receipts(ctx, field, value):
    trace = receipt(ctx)
    path = ctx.root_workdir / "events.jsonl"
    path.write_text(json.dumps({"kind": "model.call", "call_id": "critic-call",
                               "payload": {"cost_usd": 2, field: value}}) + "\n")
    asyncio.run(recovery(ctx, trace, "outer")())
    assert ctx.budgets.root().counters.usd == 2
    assert ctx.budgets.root().counters.tokens == 30
    assert provider_usage_snapshot(ctx.root_workdir).damaged == [1]
    assert any(row["kind"] == "accounting.events_corrupt" for row in events(ctx))


def test_null_historical_usage_is_zero(ctx):
    path = ctx.root_workdir / "events.jsonl"
    path.write_text(json.dumps({"kind": "model.call", "call_id": "old", "payload": {
        key: None for key in ("cost_usd", "in_tokens", "out_tokens", "reasoning_tokens", "metered_tokens")}}))
    snapshot = provider_usage_snapshot(ctx.root_workdir)
    assert snapshot.cost_usd == snapshot.tokens == 0
    assert not snapshot.damaged


@pytest.mark.parametrize("first", ["none", "ordinary", "critic", "outer", "resume"])
@pytest.mark.parametrize("warning_fails", [False, True])
def test_unreadable_history_keeps_known_charges_and_later_settles_once(ctx, monkeypatch, caplog, first, warning_fails):
    from proofstack import provider_accounting

    trace = receipt(ctx, cost=4)
    original_read = provider_accounting._usage_events
    original_emit = ctx.events.emit

    def failed_read(path):
        raise OSError("synthetic history read failure")

    async def emit(kind, *args, **kwargs):
        if warning_fails and kind == "accounting.events_unreadable":
            raise OSError("synthetic diagnostic write failure")
        await original_emit(kind, *args, **kwargs)

    async def exercise():
        nonlocal ctx
        if first in {"ordinary", "resume"}:
            await record_provider_usage(ctx, ctx.budgets.root(), {"cost_usd": 4, "in_tokens": 10, "out_tokens": 20},
                                        call_id="critic-call", emitter=ctx.events)
        if first == "resume":
            ctx = RunContext.create(root_workdir=ctx.root_workdir, flat=True)
        if first in {"outer", "critic", "resume"}:
            await recovery(ctx, trace, "outer" if first == "resume" else first)()
        trace.update(("request", 0), cost=6, output_tokens=30)
        receipt(ctx, "new-call", cost=7)
        monkeypatch.setattr(provider_accounting, "_usage_events", failed_read)
        monkeypatch.setattr(ctx.events, "emit", emit)
        for _ in range(2):
            await record_provider_usage(ctx, ctx.budgets.root(), {"cost_usd": 6, "in_tokens": 10, "out_tokens": 30},
                                        call_id="critic-call", emitter=ctx.events)
            await record_provider_usage(ctx, ctx.budgets.root(), {"cost_usd": 7, "in_tokens": 10, "out_tokens": 20},
                                        call_id="new-call", emitter=ctx.events)
            assert ctx.budgets.root().counters.usd == 13
            assert ctx.budgets.root().counters.tokens == 70
        monkeypatch.setattr(provider_accounting, "_usage_events", original_read)
        for _ in range(2):
            await recovery(ctx, trace, "critic")()
            await settle_provider_usage(ctx, ctx.budgets.root())
            await record_provider_usage(ctx, ctx.budgets.root(), {"cost_usd": 6, "in_tokens": 10, "out_tokens": 30},
                                        call_id="critic-call", emitter=ctx.events)
            assert ctx.budgets.root().counters.usd == 13
            assert ctx.budgets.root().counters.tokens == 70
        resumed = RunContext.create(root_workdir=ctx.root_workdir, flat=True)
        await settle_provider_usage(resumed, resumed.budgets.root())
        assert resumed.budgets.root().counters.usd == 13
        assert resumed.budgets.root().counters.tokens == 70

    asyncio.run(exercise())
    assert "usage settlement deferred" in caplog.text
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == 13
    warnings = [row for row in events(ctx) if row["kind"] == "accounting.events_unreadable"]
    assert len(warnings) == (0 if warning_fails else 4)


def test_cancellation_still_propagates_during_history_read_fallback(ctx, monkeypatch):
    def failed_read(path):
        raise OSError("synthetic history read failure")

    monkeypatch.setattr("proofstack.provider_accounting._usage_events", failed_read)

    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()
        emit = ctx.events.emit

        async def blocked_emit(*args, **kwargs):
            started.set()
            await release.wait()
            await emit(*args, **kwargs)

        monkeypatch.setattr(ctx.events, "emit", blocked_emit)
        task = asyncio.create_task(record_provider_usage(ctx, ctx.budgets.root(), {"cost_usd": 4},
                                                         call_id="paid-call", emitter=ctx.events))
        try:
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert ctx.budgets.root().counters.usd == 4
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)

    asyncio.run(exercise())


@pytest.mark.parametrize("mode", ["outer", "critic"])
@pytest.mark.parametrize("fragment", ['{"kind":', '{"kind":{"kind":"run.start"}', '[]'])
def test_damaged_interior_event_warns_without_blocking_recovery(ctx, mode, fragment):
    trace = receipt(ctx)
    path = ctx.root_workdir / "events.jsonl"
    path.write_text(fragment + '\n{"kind":"agent.start"}\n')
    reconcile = recovery(ctx, trace, mode)

    async def exercise():
        await reconcile()
        await reconcile()

    asyncio.run(exercise())
    assert ctx.budgets.root().counters.usd == 2
    assert _sum_logged_model_cost(path) == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()[1:]]
    assert rows[0]["kind"] == "agent.start"
    warnings = [r for r in rows if r["kind"] == "accounting.events_corrupt"]
    assert warnings and all(r["payload"]["skipped_lines"] == [1] for r in warnings)
    assert len([r for r in rows if r["kind"] == "model.call"]) == 1


@pytest.mark.parametrize("last", ['{"kind":', '{"kind":"run.end"}'])
def test_append_separates_unterminated_event_from_resumed_start(ctx, last):
    path = ctx.root_workdir / "events.jsonl"
    path.write_text(last)
    asyncio.run(ctx.events.emit("run.start"))
    rows = path.read_text().splitlines()
    assert rows[0] == last
    assert json.loads(rows[1])["kind"] == "run.start"


@pytest.mark.parametrize("mode", ["outer", "critic"])
def test_corrupt_interior_provider_receipt_still_blocks_recovery(ctx, mode):
    trace = receipt(ctx)
    trace.path.write_text('{"attempt_id":\n' + trace.path.read_text())
    with pytest.raises(json.JSONDecodeError):
        asyncio.run(recovery(ctx, trace, mode)())
    assert ctx.budgets.root().counters.usd == 0


def test_outer_resume_and_events_use_the_same_receipt_snapshot(ctx, monkeypatch):
    trace = receipt(ctx)
    emit = ctx.events.emit

    async def late_receipt(kind, *args, **kwargs):
        await emit(kind, *args, **kwargs)
        if kind == "model.call":
            trace.update(("request", 0), cost=3, output_tokens=30)

    monkeypatch.setattr(ctx.events, "emit", late_receipt)

    async def exercise():
        await recovery(ctx, trace, "outer")()
        assert ctx.budgets.root().counters.usd == 2
        await recovery(ctx, trace, "critic")()
        assert ctx.budgets.root().counters.usd == 3
        await recovery(ctx, trace, "outer")()
        await recovery(ctx, trace, "critic")()
        assert ctx.budgets.root().counters.usd == 3

    asyncio.run(exercise())
    calls = [r for r in events(ctx) if r["kind"] == "model.call"]
    assert [r["payload"]["cost_usd"] for r in calls] == [2, 1]
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == 3


@pytest.mark.parametrize("mode", ["outer", "critic"])
@pytest.mark.parametrize("blocked_at", ["lock", "writer"])
def test_cancellation_joins_settlement_before_propagating(ctx, monkeypatch, mode, blocked_at):
    trace = receipt(ctx)
    reconcile = recovery(ctx, trace, mode)
    sink = ctx.events.sink

    async def exercise():
        started = asyncio.Event()
        release = threading.Event()
        write, append = sink.write, sink._append
        if blocked_at == "lock":
            await sink._lock.acquire()

            async def blocked_write(record):
                started.set()
                await write(record)

            monkeypatch.setattr(sink, "write", blocked_write)
        else:
            loop = asyncio.get_running_loop()

            def blocked_append(line):
                loop.call_soon_threadsafe(started.set)
                assert release.wait(5), "test did not release event writer"
                append(line)

            monkeypatch.setattr(sink, "_append", blocked_append)

        task = asyncio.create_task(reconcile())
        try:
            await asyncio.wait_for(started.wait(), 5)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done(), "cancellation abandoned a pending settlement"
            assert ctx.budgets.root().counters.usd == 0
        finally:
            if blocked_at == "lock":
                sink._lock.release()
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert ctx.budgets.root().counters.usd == 2
        if mode == "critic":
            assert ctx.budgets.root().counters.tokens == 30  # Reasoning is a subset of output.
        await reconcile()
        await recovery(ctx, trace, "critic")()
        assert ctx.budgets.root().counters.usd == 2
        assert len([r for r in events(ctx) if r["kind"] == "model.call"]) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("mode", ["outer", "critic"])
def test_failed_append_retries_only_unsettled_costs(ctx, monkeypatch, mode):
    receipt(ctx, "a", 2)
    receipt(ctx, "b", 3)
    sink, tracker = ctx.events.sink, ctx.budgets.root()
    append = sink._append
    call_ids = {"a", "b"} if mode == "critic" else None

    def fail_second(line):
        if json.loads(line)["call_id"] == "b":
            raise OSError("synthetic append failure")
        append(line)

    monkeypatch.setattr(sink, "_append", fail_second)

    async def exercise():
        with pytest.raises(OSError, match="synthetic"):
            await settle_provider_usage(ctx, tracker, call_ids=call_ids)
        assert tracker.counters.usd == (2 if mode == "critic" else 0)
        monkeypatch.setattr(sink, "_append", append)
        await settle_provider_usage(ctx, tracker, call_ids=call_ids)
        await settle_provider_usage(ctx, tracker, call_ids=call_ids)
        assert tracker.counters.usd == 5

    asyncio.run(exercise())
    assert [r["call_id"] for r in events(ctx)] == ["a", "b"]


def test_concurrent_reconciliation_shares_lock_across_child_contexts(ctx):
    trace = receipt(ctx)

    async def exercise():
        await asyncio.gather(recovery(ctx, trace, "critic")(), recovery(replace(ctx), trace, "critic")())
        assert ctx.budgets.root().counters.usd == 2
        assert ctx.budgets.root().counters.tokens == 30

    asyncio.run(exercise())
    assert len(events(ctx)) == 1


def test_legacy_debit_is_restored_once_without_materializing_an_event(ctx):
    receipt(ctx)
    (ctx.root_workdir / "recovery-debits.json").write_text(json.dumps({
        "legacy": {"cost_usd": 3, "reason": "synthetic reconciled charge"},
    }))

    async def exercise():
        for _ in range(2):
            total = await settle_provider_usage(ctx, ctx.budgets.root())
            assert total == ctx.budgets.root().counters.usd == 5

    asyncio.run(exercise())
    assert [r["call_id"] for r in events(ctx)] == ["critic-call"]
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == 5


def test_outer_resume_freezes_counter_baseline_before_awaiting(ctx, monkeypatch):
    trace = receipt(ctx, cost=10)
    emit, tracker = ctx.events.emit, ctx.budgets.root()

    async def concurrent_charge(kind, *args, **kwargs):
        if kind == "model.call":
            # A native CLI call has no provider receipt and can finish while
            # recovery awaits the event sink. Its reasoning is already output.
            tracker.add_usd(3)
            tracker.add_tokens(7)
            await emit("model.call", {"cost_usd": 3, "in_tokens": 4, "out_tokens": 3,
                                      "reasoning_tokens": 2}, call_id="native-call")
        await emit(kind, *args, **kwargs)

    monkeypatch.setattr(ctx.events, "emit", concurrent_charge)

    async def exercise():
        await recovery(ctx, trace, "outer")()
        assert tracker.counters.usd == 13
        assert tracker.counters.tokens == 37
        await recovery(ctx, trace, "outer")()
        assert tracker.counters.usd == 13
        assert tracker.counters.tokens == 37

    asyncio.run(exercise())
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == 13


@pytest.mark.parametrize("mode", ["outer", "critic", "ordinary"])
@pytest.mark.parametrize("cancel", ["repeated", "timeout"])
def test_writer_failure_does_not_replace_cancellation_or_trigger_retry(ctx, monkeypatch, mode, cancel):
    from proofstack.agents.dag_workflow import _call_with_retries

    trace = receipt(ctx)
    tracker, sink = ctx.budgets.root(), ctx.events.sink
    attempts = 0

    async def exercise():
        started, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def failed_append(line):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5), "test did not release writer"
            raise OSError("synthetic event writer failure")

        monkeypatch.setattr(sink, "_append", failed_append)

        async def attempt():
            nonlocal attempts
            attempts += 1
            if mode == "ordinary":
                await record_provider_usage(ctx, tracker, {"cost_usd": 2}, call_id="critic-call", emitter=ctx.events)
            else:
                await recovery(ctx, trace, mode)()

        deadline = asyncio.timeout(None)

        async def run():
            async with deadline:
                await _call_with_retries(attempt, retries=1, retry_delay_s=0)

        task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(started.wait(), 5)
            if cancel == "timeout":
                deadline.reschedule(loop.time())
                await asyncio.sleep(0)
            else:
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(TimeoutError if cancel == "timeout" else asyncio.CancelledError) as raised:
            await asyncio.wait_for(task, 5)
        error = raised.value.__cause__ if cancel == "timeout" else raised.value
        assert isinstance(error, asyncio.CancelledError)
        assert isinstance(error.__cause__, OSError)
        assert attempts == 1
        assert tracker.counters.usd == (2 if mode == "ordinary" else 0)
        assert tracker.counters.tokens == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("first", ["ordinary", "recovery"])
@pytest.mark.parametrize("mode", ["outer", "critic"])
@pytest.mark.parametrize("final_cost", [4, 6])
def test_normal_api_call_and_recovery_share_one_charge(ctx, monkeypatch, first, mode, final_cost):
    agent = ACCritic(replace(ctx))
    agent._client = SimpleNamespace(model="synthetic")
    sink, tracker = ctx.events.sink, ctx.budgets.root()
    write = sink.write

    async def exercise():
        ready, deliver, writing = asyncio.Event(), asyncio.Event(), asyncio.Event()
        trace = None

        async def query(client, messages, query_fn, *, call_id):
            nonlocal trace
            trace = receipt(ctx, call_id, cost=4)
            await sink._lock.acquire()
            ready.set()
            await deliver.wait()
            trace.update(("request", 0), cost=final_cost, input_tokens=12, output_tokens=25)
            return 0, [{"role": "assistant", "content": "<answer_ready>true</answer_ready>"}], {
                "cost": final_cost, "input_tokens": 12, "output_tokens": 25, "reasoning_tokens": 5}

        async def blocked_write(record):
            if record["kind"] == "model.call":
                writing.set()
            await write(record)

        monkeypatch.setattr(agent, "_query", query)
        monkeypatch.setattr(sink, "write", blocked_write)
        ordinary = asyncio.create_task(APICallAgent.run(agent, agent.Inputs(problem="Synthetic question")))
        await asyncio.wait_for(ready.wait(), 5)
        try:
            if first == "ordinary":
                deliver.set()
                await asyncio.wait_for(writing.wait(), 5)
                reconcile = asyncio.create_task(recovery(ctx, trace, mode)())
            else:
                reconcile = asyncio.create_task(recovery(ctx, trace, mode)())
                await asyncio.wait_for(writing.wait(), 5)
                deliver.set()
            await asyncio.sleep(0)
            assert tracker.counters.usd == (final_cost if first == "ordinary" else 0)
        finally:
            sink._lock.release()
            deliver.set()
        await asyncio.wait_for(asyncio.gather(ordinary, reconcile), 5)
        assert tracker.counters.usd == final_cost
        assert tracker.counters.tokens == 37
        await recovery(ctx, trace, "outer")()
        assert tracker.counters.usd == final_cost
        assert tracker.counters.tokens == 37

    asyncio.run(exercise())
    calls = [r for r in events(ctx) if r["kind"] == "model.call"]
    assert sum(r["payload"]["cost_usd"] for r in calls) == final_cost
    normal = next(r["payload"] for r in calls if r["payload"].get("status") == "completed")
    assert normal["cost_usd"] == (final_cost if first == "ordinary" else final_cost - 4)
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == final_cost


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_failed_call_logging_deduplicates_already_recovered_usage(ctx, status):
    trace = receipt(ctx, cost=4)
    agent = ACCritic(ctx)

    async def exercise():
        await recovery(ctx, trace, "outer")()
        await agent._charge_failed_provider_trace(trace, SimpleNamespace(model="fake"), "critic-call", status)
        trace.update(("request", 0), cost=6, output_tokens=30)
        await agent._charge_failed_provider_trace(trace, SimpleNamespace(model="fake"), "critic-call", status)
        # A different call must still be fully charged.
        other = receipt(ctx, "other-call", cost=3)
        await agent._charge_failed_provider_trace(other, SimpleNamespace(model="fake"), "other-call", status)
        await recovery(ctx, trace, "outer")()
        assert ctx.budgets.root().counters.usd == 9
        assert ctx.budgets.root().counters.tokens == 70

    asyncio.run(exercise())
    calls = [r for r in events(ctx) if r["kind"] == "model.call"]
    assert [r["payload"]["cost_usd"] for r in calls] == [4, 0, 2, 3]
    assert all(r["payload"]["status"] == status for r in calls[1:])


def test_resume_uses_native_metered_tokens_without_double_counting_reasoning(ctx):
    async def exercise():
        await ctx.events.emit("model.call", {"cost_usd": 2, "in_tokens": 20, "out_tokens": 10,
                                            "reasoning_tokens": 5, "metered_tokens": 25})
        await settle_provider_usage(ctx, ctx.budgets.root())
        await settle_provider_usage(ctx, ctx.budgets.root())
        assert ctx.budgets.root().counters.usd == 2
        assert ctx.budgets.root().counters.tokens == 25

    asyncio.run(exercise())


@pytest.mark.parametrize("mode", ["outer", "critic", "ordinary"])
@pytest.mark.parametrize("after_append", [False, True])
def test_paid_usage_survives_failed_log_without_charging_twice(ctx, monkeypatch, mode, after_append):
    trace = receipt(ctx, cost=4)
    agent = ACCritic(replace(ctx))
    append = ctx.events.sink._append

    def failed_write(line):
        if after_append:
            append(line)
        raise OSError("synthetic writer failure")

    monkeypatch.setattr(ctx.events.sink, "_append", failed_write)

    async def log_call():
        await agent._charge_failed_provider_trace(trace, SimpleNamespace(model="fake"), "critic-call", "failed")

    async def exercise():
        with pytest.raises(OSError, match="synthetic"):
            await log_call()
        assert ctx.budgets.root().counters.usd == 4
        assert ctx.budgets.root().counters.tokens == 30
        monkeypatch.setattr(ctx.events.sink, "_append", append)
        if mode != "ordinary":
            await recovery(ctx, trace, mode)()
        await log_call()
        assert ctx.budgets.root().counters.usd == 4
        assert ctx.budgets.root().counters.tokens == 30
        trace.update(("request", 0), cost=6, output_tokens=30)
        if mode != "ordinary":
            await recovery(ctx, trace, mode)()
        await log_call()
        await recovery(ctx, trace, "outer")()
        assert ctx.budgets.root().counters.usd == 6
        assert ctx.budgets.root().counters.tokens == 40

    asyncio.run(exercise())
    assert _sum_logged_model_cost(ctx.root_workdir / "events.jsonl") == 6
