import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from mathagents.api_client import APIClient
from mathagents.provider_trace import ProviderTrace, active_trace, latest_attempts, usage_adjustments


def response(status="completed", *, tool=False):
    output = ([{"type": "function_call", "name": "research", "arguments": "{}", "id": "fc", "call_id": "call"}]
              if tool else [{"type": "message", "id": "msg", "content": [{"type": "output_text", "text": "Finished proof report"}]}])
    usage = {"input_tokens": 100, "output_tokens": 20}
    raw = {"id": "resp_one", "status": status, "output": output if status == "completed" else [],
           "usage": usage if status == "completed" else None}
    items = []
    for item in raw["output"]:
        values = dict(item)
        if "content" in values:
            values["content"] = [SimpleNamespace(**p) for p in values["content"]]
        items.append(SimpleNamespace(**values))
    return SimpleNamespace(**{**raw, "output": items, "usage": SimpleNamespace(**usage) if raw["usage"] else None},
                           model_dump=lambda: raw)


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr("mathagents.api_client.request_logger.log_request", lambda **kw: None)
    monkeypatch.setattr("mathagents.api_client.request_logger.log_response", lambda **kw: None)
    return APIClient(model="gpt-6-astra--max", api="custom", background=True, use_openai_responses_api=True,
                     timeout=2, max_wallclock_per_call_s=2, max_retries_inner=0)


@pytest.mark.parametrize("tool", [False, True])
def test_cancel_completion_race_recovers_without_another_request_or_tool(api, monkeypatch, tmp_path, tool):
    clock = [0.0]
    monkeypatch.setattr("mathagents.api_client.time.time", lambda: clock[0])
    monkeypatch.setattr("mathagents.api_client.time.sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    calls = []
    def create(**kwargs):
        calls.append("create")
        return response("in_progress")
    def cancel(*args, **kwargs):
        calls.append("cancel")
        raise RuntimeError("already completed")
    def retrieve(*args, **kwargs):
        calls.append("retrieve")
        return response(tool=tool)
    trace = ProviderTrace(tmp_path / "agents/seat/provider-attempts.jsonl", call_id="c")
    token = active_trace.set(trace)
    try:
        result = api._openai_query_responses_api(SimpleNamespace(responses=SimpleNamespace(
            create=create, cancel=cancel, retrieve=retrieve)), 0, [{"role": "user", "content": "Q"}])
    finally:
        active_trace.reset(token)
    assert calls == ["create", "cancel", "retrieve"]
    assert result.input_tokens == 100 and result.output_tokens == 20
    assert latest_attempts(trace.path)[0]["status"] == "completed"
    assert not latest_attempts(trace.path)[0]["reconciliation_pending"]
    assert bool(trace.completed_report) is not tool
    if not tool:
        assert "Finished proof report" in result.conversation[-1]["content"]
        saved = json.loads(next(trace.path.parent.glob("provider-completed-*.json")).read_text())
        assert saved["report"] == "Finished proof report"
    event = {"kind": "model.call", "call_id": "c", "payload": {"cost_usd": trace.totals()["cost"],
              "in_tokens": 100, "out_tokens": 20}}
    assert usage_adjustments(tmp_path, [event]) == []


def test_unresolved_cancellation_keeps_response_id_and_known_usage(api, tmp_path):
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl")
    trace.response(api, "request", 0, {"id": "resp_one", "usage": {"input_tokens": 100, "output_tokens": 20}})
    def fail(*a, **kw):
        raise RuntimeError("transport unavailable")
    token = active_trace.set(trace)
    try:
        assert api._cancel_background_response(SimpleNamespace(responses=SimpleNamespace(cancel=fail, retrieve=fail)),
                                               "resp_one", "request", 0) is None
    finally:
        active_trace.reset(token)
    row = latest_attempts(trace.path)[0]
    assert row["response_id"] == "resp_one" and row["reconciliation_pending"]
    assert row["input_tokens"] == 100


def test_cancel_returning_completed_still_retrieves_code_interpreter_outputs(api, tmp_path):
    operations = []
    completed = response()
    evidence = {"type": "code_interpreter_call", "status": "completed", "container_id": "container",
                "code": "print('synthetic')", "outputs": [{"type": "logs", "logs": "synthetic"}]}

    def cancel(*args, **kwargs):
        operations.append("cancel")
        return completed

    def retrieve(*args, **kwargs):
        assert kwargs["include"] == ["code_interpreter_call.outputs"]
        operations.append("retrieve")
        raw = completed.model_dump()
        raw["output"] = [evidence, *raw["output"]]
        return SimpleNamespace(model_dump=lambda: raw, status="completed", usage=completed.usage)

    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl")
    token = active_trace.set(trace)
    try:
        api._cancel_background_response(SimpleNamespace(responses=SimpleNamespace(cancel=cancel, retrieve=retrieve)),
                                        "resp_one", "request", 0)
    finally:
        active_trace.reset(token)
    assert operations == ["cancel", "retrieve"]
    assert trace.tool_evidence() == [evidence]
    report = json.loads(next(tmp_path.glob("provider-completed-*.json")).read_text())
    assert report["tool_evidence"] == [evidence]


def test_delivery_interval_removes_research_tools_but_keeps_publication(api, monkeypatch):
    api.tool_wrapup_reserve_s = 2
    api.tool_descriptions = [{"type": "code_interpreter", "container": {"type": "auto"}},
        {"type": "function", "function": {"name": "publish_artifact", "description": "publish",
                                         "parameters": {"type": "object", "properties": {}}}}]
    api.tool_functions = {"publish_artifact": lambda: None}
    payloads = []
    def create(**kwargs):
        payloads.append(kwargs)
        return response()
    result = api._openai_query_responses_api(SimpleNamespace(responses=SimpleNamespace(create=create)), 0,
                                            [{"role": "user", "content": "Q"}])
    assert len(payloads) == 1
    assert [t.get("name") for t in payloads[0]["tools"]] == ["publish_artifact"]
    assert "reserved for delivery" in payloads[0]["input"][-1]["content"]
    assert "Finished proof report" in result.conversation[-1]["content"]


def test_old_response_cannot_replace_latest_recoverable_report(api, tmp_path):
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl")
    trace.request(api, "first", 0, {})
    trace.response(api, "first", 0, response().model_dump())
    assert trace.completed_report
    trace.request(api, "second", 0, {})
    trace.response(api, "first", 0, response().model_dump())
    assert trace.completed_report is None
    assert trace.totals()["input_tokens"] == 100
    current = response().model_dump()
    current["output"][0]["content"][0]["text"] = "Current report"
    trace.response(api, "second", 0, current)
    trace.response(api, "first", 0, response().model_dump())
    assert trace.completed_report == "Current report"
    assert trace.totals()["input_tokens"] == 200


def test_report_disk_failure_does_not_erase_provider_usage(api, tmp_path, monkeypatch):
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl")
    warnings = []
    monkeypatch.setattr("mathagents.provider_trace.logger.warning", lambda *args: warnings.append(args))
    def fail(*a, **kw):
        raise OSError("disk failure")
    monkeypatch.setattr("mathagents.provider_trace.os.replace", fail)
    monkeypatch.setattr("mathagents.provider_trace.os.unlink", fail)
    trace.response(api, "request", 0, response().model_dump())
    assert trace.completed_report == "Finished proof report"
    assert latest_attempts(trace.path)[0]["input_tokens"] == 100
    assert len(warnings) == 2


def test_mandatory_receipt_failure_still_blocks_provider_retry(api, tmp_path, monkeypatch):
    from mathagents.provider_trace import ProviderAccountingError
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl")
    def fail(*a, **kw):
        raise OSError("receipt disk failure")
    monkeypatch.setattr("mathagents.provider_trace.os.write", fail)
    with pytest.raises(ProviderAccountingError):
        trace.response(api, "request", 0, response().model_dump())


def test_live_failure_event_is_flushed_while_query_is_still_running(api, tmp_path):
    from proofstack.agents.batch3_critic import Batch3CleanupCritic
    from proofstack.context import RunContext

    ctx = RunContext.create(root_workdir=tmp_path, flat=True)
    seat = Batch3CleanupCritic(ctx)
    def query(client, messages):
        trace = active_trace.get()
        trace.request(client, "request", 0, {})
        trace.failure(("request", 0), error_code="context_length_exceeded", retry_decision="stop")
        # The call has not returned, but a monitor can already see the failure.
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        event = next(e for e in events if e["kind"] == "model.attempt.failed")
        assert event["call_id"] == "critic-call"
        assert event["payload"]["retry_decision"] == "stop"
        raise ValueError("context_length_exceeded")

    with pytest.raises(ValueError, match="context_length_exceeded"):
        asyncio.run(seat._query(api, [], query, call_id="critic-call"))


def test_live_monitor_failure_does_not_change_retry_or_accounting(api, tmp_path):
    def fail(payload):
        raise OSError("monitor unavailable")
    trace = ProviderTrace(tmp_path / "attempts.jsonl", on_failure=fail)
    trace.request(api, "request", 0, {})
    trace.response(api, "request", 0, response().model_dump())
    trace.failure(("request", 0), retry_decision="stop")
    assert trace.totals()["input_tokens"] == 100
    assert latest_attempts(trace.path)[0]["retry_decision"] == "stop"


def test_cancellation_reconciliation_bounds_hung_transports(api):
    release = threading.Event()
    def hang(*a, **kw):
        release.wait(10)
    started = time.monotonic()
    try:
        result = api._cancel_background_response(SimpleNamespace(responses=SimpleNamespace(cancel=hang, retrieve=hang)),
                                                 "resp_one", "request", 0)
        assert result is None
        assert time.monotonic() - started < 3.5
    finally:
        release.set()


def test_api_cancellation_carries_completed_report_and_charges_once(api, tmp_path):
    from proofstack.agents.ac.multi_author import SubAuthorSeat
    from proofstack.budget import BudgetSpec
    from proofstack.context import RunContext
    ctx = RunContext.create(root_workdir=tmp_path, flat=True, run_budget=BudgetSpec(max_wallclock_s=100))
    seat = SubAuthorSeat(ctx, model_ref="models/openai/gpt-54", name="seat")
    started = threading.Event()
    def query(client, messages):
        trace = active_trace.get()
        trace.request(client, "request", 0, {})
        started.set()
        while not client.terminated:
            time.sleep(.005)
        trace.response(client, "request", 0, response().model_dump())
        return None
    async def scenario():
        task = asyncio.create_task(seat._query(api, [], query, call_id="c"))
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc:
            await task
        assert exc.value.completed_report == "Finished proof report"
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        calls = [e for e in events if e["kind"] == "model.call"]
        assert len(calls) == 1 and calls[0]["payload"]["in_tokens"] == 100
        assert usage_adjustments(tmp_path, events) == []
    asyncio.run(scenario())
