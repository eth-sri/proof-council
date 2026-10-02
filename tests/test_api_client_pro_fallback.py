"""Offline request-level regressions for the Astra Pro -> standard fallback."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mathagents.api_client import APIClient, _responses_retry_state
from mathagents.config_loader import load_solver_config
from mathagents.provider_trace import ProviderTrace, active_trace, latest_attempts


def _api(**overrides):
    config = {
        key: value for key, value in load_solver_config("models/openai/gpt-6-astra-pro").items()
        if not key.startswith("__")
    }
    config.update(sleep_after_request=0, sleep_on_error=0, max_retries_inner=2)
    config.update(overrides)
    return APIClient(**config)


def _query(text="problem"):
    return [{"role": "developer", "content": "Prove it."}, {"role": "user", "content": text}]


def _reply(*, status="completed", text="done", code="server_error", output=None, usage=True):
    if output is None:
        output = [] if status == "failed" else [SimpleNamespace(
            type="message", id="msg_done",
            content=[SimpleNamespace(type="output_text", text=text)],
        )]
    return SimpleNamespace(
        id="resp_test", status=status, output=output,
        error=SimpleNamespace(code=code, message=code),
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15) if usage else None,
        model_dump=lambda: {"status": status},
    )


class _Responses:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.payloads = []
        self.cancelled = []

    def create(self, **payload):
        self.payloads.append(deepcopy(payload))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def retrieve(self, response_id, **kwargs):
        return _reply(status="in_progress", output=[], usage=False)

    def cancel(self, response_id, **kwargs):
        self.cancelled.append(response_id)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("mathagents.api_client.time.sleep", lambda _: None)
    monkeypatch.setattr("mathagents.api_client.request_logger", Mock())
    # Older test modules install RateLimitError=Exception at collection time.
    # Keep unrelated HTTP failures distinct regardless of collection order.
    monkeypatch.setattr("mathagents.api_client.RateLimitError", type("RateLimitError", (Exception,), {}))


def _run(monkeypatch, api, responses, query=None):
    monkeypatch.setattr("mathagents.api_client.OpenAI", lambda **_: SimpleNamespace(responses=responses))
    return api._run_query_with_retry(0, query or _query())


def _modes(responses):
    return [payload["reasoning"]["mode"] for payload in responses.payloads]


@pytest.mark.parametrize("stream", [False, True])
def test_chat_context_400_halves_output_limit_and_retries(monkeypatch, stream):
    api = APIClient(model="chat-test", api="openai", stream_openai_chat_completions=stream,
                    max_tokens=128, max_retries_inner=2, sleep_on_error=0)
    error = RuntimeError("maximum context length exceeded")
    error.status_code = 400
    payloads = []
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=2, model_dump=lambda: {})
    def create(**payload):
        payloads.append(payload)
        if len(payloads) == 1:
            raise error
        if stream:
            return iter([SimpleNamespace(usage=usage, choices=[SimpleNamespace(delta=SimpleNamespace(content="done"))])])
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(
            content="done", tool_calls=None,
            model_dump=lambda: {"role": "assistant", "content": "done"}))], model_dump=lambda: {})
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = api._openai_query_chat_completions_api(client, 0, _query())
    assert result.conversation[-1]["content"] == "done"
    assert [p[api.max_tokens_param] for p in payloads] == [128, 64]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("limit", [None, 1])
def test_chat_context_error_without_reducible_limit_does_not_mask_original(stream, limit):
    api = APIClient(model="chat-test", api="openai", stream_openai_chat_completions=stream)
    api.kwargs[api.max_tokens_param] = limit
    error = RuntimeError("context_length_exceeded")
    error.status_code = 400
    create = Mock(side_effect=error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(RuntimeError, match="context_length_exceeded"):
        api._openai_query_chat_completions_api(client, 0, _query())
    create.assert_called_once()


@pytest.mark.parametrize("inner_retries", [0, 2])
@pytest.mark.parametrize("failure", ["transport", "server_error", "rate_limit_exceeded"])
def test_two_failures_switch_third_attempt_including_across_outer_retries(monkeypatch, inner_retries, failure):
    responses = _Responses([
        RuntimeError("503 temporary outage") if failure == "transport"
        else _reply(status="failed", code=failure, usage=False)
        for _ in range(2)
    ] + [_reply()])
    api = _api(max_retries_inner=inner_retries)
    warning = Mock()
    monkeypatch.setattr("mathagents.api_client.logger.warning", warning)
    result = _run(monkeypatch, api, responses)

    assert _modes(responses) == ["pro", "pro", "standard"]
    assert result.conversation[-1]["content"].strip() == "done"
    for payload in responses.payloads:
        assert payload["model"] == "gpt-6-astra"
        assert payload["input"] == _query()
        assert payload["reasoning"]["effort"] == "max"
        assert payload["reasoning"]["summary"] == "auto"
        assert "openai_pro_fallback_after_failures" not in payload
    assert sum("Pro fallback" in str(call) for call in warning.call_args_list) == 1
    assert api.kwargs["reasoning"]["mode"] == "pro"
    assert _responses_retry_state.get() is None


class _Clock:
    def __init__(self, monkeypatch):
        self.now = 0.0
        monkeypatch.setattr("mathagents.api_client.time.time", lambda: self.now)
        monkeypatch.setattr("mathagents.api_client.time.sleep", self.sleep)

    def sleep(self, seconds):
        self.now += seconds


@pytest.mark.parametrize("server_kill", [False, True])
@pytest.mark.parametrize("inner_retries", [0, 2])
def test_timeouts_keep_effort_downgrade_and_cancel_before_standard(monkeypatch, server_kill, inner_retries):
    _Clock(monkeypatch)
    if server_kill:
        first = _reply(status="failed", usage=False)
    else:
        first = _reply(status="queued", output=[], usage=False)
    responses = _Responses([first, first, _reply()])
    api = _api(
        timeout=1, max_wallclock_per_call_s=600, max_retries_inner=inner_retries,
        background_server_kill_after_s=0 if server_kill else 3300,
    )
    _run(monkeypatch, api, responses)
    assert _modes(responses) == ["pro", "pro", "standard"]
    assert [p["reasoning"]["effort"] for p in responses.payloads] == ["max", "high", "high"]
    assert responses.cancelled == ([] if server_kill else ["resp_test", "resp_test"])


def test_poll_transport_failures_do_not_recreate_or_downgrade_response(monkeypatch):
    responses = _Responses([_reply(status="queued", output=[], usage=False)])
    responses.retrieve = Mock(side_effect=[RuntimeError("503"), RuntimeError("503"), _reply()])
    result = _run(monkeypatch, _api(), responses)
    assert _modes(responses) == ["pro"]
    assert responses.retrieve.call_count == 3
    assert result.conversation[-1]["content"].strip() == "done"


def test_poll_404_status_retries_existing_response(monkeypatch):
    error = RuntimeError("Response not found")
    error.status_code = 404
    responses = _Responses([_reply(status="queued", output=[], usage=False)])
    responses.retrieve = Mock(side_effect=[error, _reply()])
    _run(monkeypatch, _api(), responses)
    assert _modes(responses) == ["pro"]
    assert [call.args[0] for call in responses.retrieve.call_args_list] == ["resp_test", "resp_test"]


def test_persistent_poll_404_is_deadline_bounded(monkeypatch):
    _Clock(monkeypatch)
    error = RuntimeError("Error code: 404")
    responses = _Responses([_reply(status="queued", output=[], usage=False)])
    responses.retrieve = Mock(side_effect=error)
    with pytest.raises(ValueError, match="Max outer retries"):
        _run(monkeypatch, _api(max_wallclock_per_call_s=100), responses)
    assert _modes(responses) == ["pro"]
    assert responses.cancelled == ["resp_test"]


def test_persistent_poll_failures_still_reach_deadline(monkeypatch):
    _Clock(monkeypatch)
    responses = _Responses([_reply(status="queued", output=[], usage=False)])
    responses.retrieve = Mock(side_effect=RuntimeError("503"))
    with pytest.raises(ValueError, match="Max outer retries"):
        _run(monkeypatch, _api(max_wallclock_per_call_s=100), responses)
    assert _modes(responses) == ["pro"]
    assert responses.cancelled == ["resp_test"]


def test_outer_retry_does_not_reset_fallback_deadline(monkeypatch):
    clock = _Clock(monkeypatch)
    responses = _Responses([RuntimeError("503"), RuntimeError("503"), _reply(status="queued", output=[], usage=False)])
    api = _api(max_retries_inner=0, max_wallclock_per_call_s=200)
    with pytest.raises(ValueError, match="Max outer retries"):
        _run(monkeypatch, api, responses)
    assert _modes(responses) == ["pro", "pro", "standard"]
    assert responses.cancelled == ["resp_test"]
    assert clock.now <= 320  # one polling interval and existing retry backoff


@pytest.mark.parametrize("error", ["401 unauthorized", "model_not_found", "insufficient_quota", "billing_hard_limit_reached"])
def test_terminal_errors_still_fail_fast(monkeypatch, error):
    responses = _Responses([RuntimeError(error)])
    with pytest.raises(RuntimeError, match=error):
        _run(monkeypatch, _api(), responses)
    assert _modes(responses) == ["pro"]
    assert _responses_retry_state.get() is None


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("code", ["context_length_exceeded", "maximum context length exceeded", "input token count exceeds limit"])
def test_context_failure_never_retries_or_switches_to_standard(monkeypatch, tmp_path, background, code):
    api = _api(max_retries_inner=25, openai_pro_fallback_after_failures=1)
    failed = _reply(status="failed", code=code, usage=False)
    if background:
        responses = _Responses([_reply(status="queued", output=[], usage=False)])
        responses.retrieve = Mock(return_value=failed)
    else:
        responses = _Responses([ValueError(code)])
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl")
    token = active_trace.set(trace)
    sleep = Mock()
    monkeypatch.setattr(api, "_sleep", sleep)
    try:
        with pytest.raises(ValueError, match=code):
            _run(monkeypatch, api, responses)
    finally:
        active_trace.reset(token)
    assert _modes(responses) == ["pro"]
    sleep.assert_not_called()
    assert _responses_retry_state.get() is None
    row = latest_attempts(trace.path)[0]
    assert row["error_code"] == "context_length_exceeded"
    assert row["retry_decision"] == "stop" and row["usage_unavailable"]


def test_retry_decision_is_reported_before_retry_sleep(monkeypatch, tmp_path):
    api = _api()
    responses = _Responses([ValueError("503 unavailable"), _reply()])
    events = []
    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl", on_failure=events.append)
    token = active_trace.set(trace)
    def sleep(seconds):
        assert events and events[0]["retry_decision"] == "retry"
        assert events[0]["error_message"] == "503 unavailable"
    monkeypatch.setattr(api, "_sleep", sleep)
    try:
        _run(monkeypatch, api, responses)
    finally:
        active_trace.reset(token)
    assert len(events) == 1
    assert _modes(responses) == ["pro", "pro"]


@pytest.mark.parametrize("error_dict", [False, True])
def test_failed_context_with_partial_output_keeps_usage_but_is_not_a_success(monkeypatch, tmp_path, error_dict):
    failed = _reply(text="Unfinished review")
    failed.status = "failed"
    error = {"code": "context_length_exceeded", "message": "Too long"}
    failed.error = error if error_dict else SimpleNamespace(**error)
    failed.model_dump = lambda: {
        "id": "resp_test", "status": "failed", "error": error,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    responses = _Responses([failed])
    trace = ProviderTrace(tmp_path / "attempts.jsonl")
    token = active_trace.set(trace)
    try:
        with pytest.raises(ValueError, match="context_length_exceeded") as raised:
            _run(monkeypatch, _api(), responses)
    finally:
        active_trace.reset(token)
    assert _modes(responses) == ["pro"]
    assert raised.value.partial_usage["input_tokens"] == 10
    assert raised.value.partial_usage["cost"] == pytest.approx(_api()._get_cost(10, 5))
    assert not raised.value.partial_usage["usage_unavailable"]
    assert trace.completed_report is None


def test_salvaged_output_does_not_trigger_another_attempt(monkeypatch):
    reply = _reply()
    reply.status = "failed"
    responses = _Responses([reply])
    result = _run(monkeypatch, _api(), responses)
    assert _modes(responses) == ["pro"]
    assert result.conversation[-1]["content"].strip() == "done"
    assert result.cost_usd == pytest.approx(_api()._get_cost(10, 5))


def test_fallback_persists_through_tool_and_wrapup_turns_with_usage(monkeypatch):
    executions = []

    def compute():
        executions.append("executed")
        return "42"

    api = _api(tools=[(compute, {
        "type": "function", "function": {"name": "compute", "parameters": {"type": "object", "properties": {}}},
    })], max_tool_calls=1)
    tool_reply = _reply(output=[SimpleNamespace(
        type="function_call", id="fc_1", call_id="call_1", name="compute", arguments="{}",
    )])
    incomplete = _reply(status="incomplete", text="partial")
    incomplete.incomplete_details = SimpleNamespace(reason="max_output_tokens")
    responses = _Responses([RuntimeError("503"), RuntimeError("503"), tool_reply, incomplete, _reply()])
    result = _run(monkeypatch, api, responses)

    assert _modes(responses) == ["pro", "pro", "standard", "standard", "standard"]
    assert [p["reasoning"]["effort"] for p in responses.payloads] == ["max"] * 4 + ["high"]
    assert executions == ["executed"]
    assert any(item.get("type") == "function_call_output" for item in responses.payloads[3]["input"])
    assert result.conversation[:2] == _query()
    assert (result.input_tokens, result.output_tokens) == (30, 15)
    assert result.cost_usd == pytest.approx(3 * api._get_cost(10, 5))


def test_successful_tool_turn_before_failures_is_preserved(monkeypatch):
    api = _api(tools=[(lambda: "42", {
        "type": "function", "function": {"name": "compute", "parameters": {"type": "object", "properties": {}}},
    })], max_tool_calls=1)
    tool_reply = _reply(output=[SimpleNamespace(
        type="function_call", id="fc_1", call_id="call_1", name="compute", arguments="{}",
    )])
    responses = _Responses([tool_reply, RuntimeError("503"), RuntimeError("503"), _reply()])
    result = _run(monkeypatch, api, responses)
    assert _modes(responses) == ["pro", "pro", "pro", "standard"]
    assert responses.payloads[1]["input"] == responses.payloads[3]["input"]
    assert result.cost_usd == pytest.approx(2 * api._get_cost(10, 5))


def test_fallback_is_opt_in_and_later_queries_start_pro(monkeypatch):
    api = _api(openai_pro_fallback_after_failures=0)
    responses = _Responses([RuntimeError("503"), RuntimeError("503"), _reply()])
    _run(monkeypatch, api, responses)
    assert _modes(responses) == ["pro", "pro", "pro"]

    api = _api()
    responses = _Responses([RuntimeError("503"), RuntimeError("503"), _reply(), _reply()])
    _run(monkeypatch, api, responses)
    _run(monkeypatch, api, responses)
    assert _modes(responses) == ["pro", "pro", "standard", "pro"]


def test_parallel_queries_on_same_client_have_independent_fallback(monkeypatch):
    barrier = Barrier(2)
    payloads = {"failing": [], "healthy": []}

    def create(**payload):
        name = payload["input"][-1]["content"]
        payloads[name].append(deepcopy(payload))
        if name == "failing":
            if len(payloads[name]) <= 2:
                raise RuntimeError("503 temporary outage")
            barrier.wait(timeout=5)
        else:
            barrier.wait(timeout=5)
        return _reply()

    monkeypatch.setattr("mathagents.api_client.OpenAI", lambda **_: SimpleNamespace(responses=SimpleNamespace(create=create)))
    api = _api()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(api._run_query_with_retry, idx, _query(name)) for idx, name in enumerate(payloads)]
        for future in futures:
            assert future.result(timeout=10).conversation[-1]["content"].strip() == "done"
    assert [p["reasoning"]["mode"] for p in payloads["failing"]] == ["pro", "pro", "standard"]
    assert [p["reasoning"]["mode"] for p in payloads["healthy"]] == ["pro"]
    assert api.kwargs["reasoning"]["mode"] == "pro"
