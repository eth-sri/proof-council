import asyncio
import threading
from types import SimpleNamespace

import pytest

from proofstack.agents.ac.critic import ACCritic
from proofstack.budget import BudgetSpec
from proofstack.context import RunContext
from mathagents.provider_trace import active_trace


def test_cancelled_query_preserves_stop_when_accounting_fails(tmp_path, monkeypatch):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True)
    agent = ACCritic(ctx)
    entered, stopped = threading.Event(), threading.Event()
    client = SimpleNamespace(model="fake", terminate=stopped.set)

    def query(client, messages):
        trace = active_trace.get()
        trace.update(("request", 0), response_id="saved", status="completed", cost=2,
                     usage_unavailable=False)
        trace.completed_report = "Paid review before cancellation"
        entered.set()
        assert stopped.wait(5), "provider thread was not terminated"

    def failed_append(line):
        raise OSError("synthetic event writer failure")

    monkeypatch.setattr(ctx.events.sink, "_append", failed_append)

    async def exercise():
        task = asyncio.create_task(agent._query(client, [], query, call_id="cancelled-query"))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(task, 5)
        assert isinstance(raised.value.__cause__, OSError)
        assert raised.value.completed_report == "Paid review before cancellation"
        assert ctx.budgets.root().counters.usd == 2

    asyncio.run(exercise())


def test_cancelled_query_terminates_provider_polling(tmp_path):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True,
                            run_budget=BudgetSpec(max_wallclock_s=60))
    agent = ACCritic(ctx)
    entered, stopped, finished = threading.Event(), threading.Event(), threading.Event()
    client = SimpleNamespace(model="fake", terminate=stopped.set, timeout=100, max_wallclock_per_call_s=100)

    def query(client, messages):
        entered.set()
        try:
            assert stopped.wait(timeout=2), "provider thread did not receive termination"
        finally:
            finished.set()

    async def run():
        task = asyncio.create_task(agent._query(client, [], query))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(finished.wait, 2)

    asyncio.run(run())
    assert stopped.is_set()
    assert 0 < client.timeout <= 60
    assert "model.call.cancelled" in (tmp_path / "events.jsonl").read_text()
    assert "usage_unavailable" in (tmp_path / "events.jsonl").read_text()


def test_reused_client_timeout_shrinks_with_remaining_budget(tmp_path, monkeypatch):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True)
    agent = ACCritic(ctx)
    monkeypatch.setattr(agent.tracker, "remaining_wallclock_s", lambda: 3)
    client = SimpleNamespace(timeout=90, max_wallclock_per_call_s=60)
    result = asyncio.run(agent._query(client, [], lambda *_: "result"))
    assert result == "result"
    assert client.timeout == client.max_wallclock_per_call_s == 3


def test_configured_shorter_timeout_is_preserved(tmp_path, monkeypatch):
    agent = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    monkeypatch.setattr(agent.tracker, "remaining_wallclock_s", lambda: 3)
    config = agent._limit_client_deadline({"timeout": 1, "max_wallclock_per_call_s": 2})
    assert config == {"timeout": 1, "max_wallclock_per_call_s": 2}


def test_default_client_deadlines_are_not_widened(tmp_path, monkeypatch):
    agent = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    monkeypatch.setattr(agent.tracker, "remaining_wallclock_s", lambda: 23 * 3600)
    assert agent._limit_client_deadline({}) == {}
    client = SimpleNamespace(timeout=30000, max_wallclock_per_call_s=600)
    asyncio.run(agent._query(client, [], lambda *_: None))
    assert client.max_wallclock_per_call_s == 600
    assert client.timeout == 30000


def test_expired_budget_overrun_keeps_provider_timeout(tmp_path, monkeypatch):
    from proofstack.budget import allow_budget_overrun

    ctx = RunContext.create(root_workdir=tmp_path, flat=True, run_budget=BudgetSpec(max_wallclock_s=0))
    agent = ACCritic(ctx)
    client = SimpleNamespace(timeout=30, max_wallclock_per_call_s=20)
    with allow_budget_overrun():
        config = agent._limit_client_deadline({"timeout": 30, "max_wallclock_per_call_s": 20})
        assert asyncio.run(agent._query(client, [], lambda *_: "fallback")) == "fallback"
    assert config == {"timeout": 30, "max_wallclock_per_call_s": 20}
    assert client.timeout == 30 and client.max_wallclock_per_call_s == 20


@pytest.mark.parametrize("configured,cap,expected", [(10, 100, 10), (100, 10, 10), (None, 1000, 600)])
def test_editor_seat_cap_only_narrows_model_configuration(tmp_path, monkeypatch, configured, cap, expected):
    import mathagents
    from proofstack.agents.writeup_loop import RewriteSeat

    config = {"model": "fake", "timeout": 30}
    if configured is not None:
        config["max_wallclock_per_call_s"] = configured
    monkeypatch.setattr(mathagents, "load_solver_config", lambda _: config)
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, api_client_factory=lambda cfg: cfg)
    seat = RewriteSeat(ctx)
    seat.wallclock_cap_s = cap
    client = seat._build_client("fake")
    assert client["max_wallclock_per_call_s"] == expected
    assert client["timeout"] == min(30, cap)
