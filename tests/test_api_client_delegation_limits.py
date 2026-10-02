"""Offline request-level coverage for delegation resource controls."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from mathagents.api_client import APIClient, RequiredHostedToolUnavailable


HOSTED_TOOLS = [
    (None, {"type": "code_interpreter", "container": {"type": "auto"}}),
    (None, {"type": "web_search_preview"}),
]


def _local(fn):
    return (fn, {"type": "function", "function": {
        "name": "delegate", "description": "Delegate.",
        "parameters": {"type": "object", "properties": {}},
    }})


def _reply(*, hosted=False, local=False, incomplete=False, arguments="{}"):
    output = []
    if hosted:
        output.append(SimpleNamespace(type="code_interpreter_call", id="ci", code="print(1)",
                                      container_id="cntr", status="completed"))
    if local:
        output.append(SimpleNamespace(type="function_call", id="fc", call_id="call", name="delegate",
                                      arguments=arguments))
    else:
        output.append(SimpleNamespace(type="message", id="msg", content=[SimpleNamespace(type="output_text", text="done")]))
    return SimpleNamespace(output=output, usage={"input_tokens": 1, "output_tokens": 1},
                           status="incomplete" if incomplete else "completed",
                           incomplete_details=SimpleNamespace(reason="max_output_tokens") if incomplete else None,
                           model_dump=lambda: {})


class _Responses:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.payloads = []

    def create(self, **payload):
        self.payloads.append(payload)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def _no_logs_or_sleeps(monkeypatch):
    monkeypatch.setattr("mathagents.api_client.request_logger.log_request", lambda **kw: None)
    monkeypatch.setattr("mathagents.api_client.request_logger.log_response", lambda **kw: None)
    monkeypatch.setattr("mathagents.api_client.time.sleep", lambda seconds: None)


def _api(**kwargs):
    return APIClient(model="fake", api="custom", use_openai_responses_api=True,
                     sleep_after_request=0, sleep_on_error=0, **kwargs)


def _run(api, responses):
    return api._openai_query_responses_api(SimpleNamespace(responses=responses), 0,
                                         [{"role": "user", "content": "problem"}])


def test_tool_cannot_spoof_container_provenance_or_deadline():
    seen = []
    def delegate(*, messages=None, call_deadline_monotonic_s=None):
        seen.append((messages, call_deadline_monotonic_s))
        return "done"
    api = _api(tools=[_local(delegate)])
    trusted = [{"type": "code_interpreter_call", "container_id": "own-container"}]
    api._execute_tool_function("delegate", {"messages": [{"container_id": "other-helper"}],
                                          "call_deadline_monotonic_s": float("inf")},
                               trusted, call_deadline_monotonic_s=12.0)
    assert seen == [(trusted, 12.0)]


@pytest.mark.parametrize("limit", [-1, 1.5, True, "1"])
def test_hosted_limit_requires_nonnegative_integer(limit):
    with pytest.raises(ValueError, match="nonnegative integer"):
        _api(max_hosted_tool_calls=limit)


@pytest.mark.parametrize("options", [
    {"api": "anthropic", "use_openai_responses_api": True},
    {"api": "custom", "use_openai_responses_api": False},
    {"api": "custom", "use_openai_responses_api": True, "batch_processing": True},
])
def test_hosted_limit_rejects_unsupported_paths(options):
    with pytest.raises(ValueError, match="requires the OpenAI Responses API"):
        APIClient(model="fake", max_hosted_tool_calls=1, **options)


def test_hosted_allowance_reaches_provider_and_decreases_after_local_turn():
    calls = []
    api = _api(max_hosted_tool_calls=2, max_tool_calls=1,
               tools=HOSTED_TOOLS + [_local(lambda: calls.append(1) or "report")])
    responses = _Responses([_reply(hosted=True, local=True), _reply(hosted=True)])
    _run(api, responses)
    assert calls == [1]
    assert [p["max_tool_calls"] for p in responses.payloads] == [2, 1]
    assert all("max_hosted_tool_calls" not in p for p in responses.payloads)
    assert all(tool["type"] != "function" for tool in responses.payloads[1]["tools"])


def test_zero_hosted_allowance_keeps_local_functions_available():
    calls = []
    api = _api(max_hosted_tool_calls=0, tools=HOSTED_TOOLS + [_local(lambda: calls.append(1) or "report")])
    responses = _Responses([_reply(local=True), _reply()])
    _run(api, responses)
    assert calls == [1]
    assert all([tool["type"] for tool in p["tools"]] == ["function"] for p in responses.payloads)
    assert all("max_tool_calls" not in p for p in responses.payloads)


def test_hosted_allowance_cannot_restart_on_output_continuation():
    api = _api(max_hosted_tool_calls=1, tools=HOSTED_TOOLS, openai_continue_on_max_output_tokens=True)
    responses = _Responses([_reply(hosted=True, incomplete=True), _reply()])
    _run(api, responses)
    assert responses.payloads[0]["max_tool_calls"] == 1
    assert responses.payloads[1]["tools"] == []
    assert "max_tool_calls" not in responses.payloads[1]


def test_ambiguous_failed_request_cannot_reuse_hosted_allowance():
    api = _api(max_hosted_tool_calls=1, tools=HOSTED_TOOLS)
    responses = _Responses([RuntimeError("connection lost after request submission"), _reply()])
    _run(api, responses)
    assert responses.payloads[0]["max_tool_calls"] == 1
    assert responses.payloads[1]["tools"] == []


def test_mandatory_verification_cannot_retry_without_hosted_tools():
    api = _api(max_hosted_tool_calls=12, tools=HOSTED_TOOLS,
               required_hosted_tool_types=["code_interpreter"])
    responses = _Responses([RuntimeError("connection lost after request submission"), _reply()])
    with pytest.raises(RequiredHostedToolUnavailable):
        _run(api, responses)
    assert len(responses.payloads) == 1
    assert responses.payloads[0]["max_tool_calls"] == 12


def test_completed_tool_can_wrap_up_without_refunding_uncertain_allowance():
    api = _api(max_hosted_tool_calls=1, tools=HOSTED_TOOLS,
               required_hosted_tool_types=["code_interpreter"], openai_continue_on_max_output_tokens=True)
    responses = _Responses([_reply(hosted=True, incomplete=True), _reply()])
    _run(api, responses)
    assert responses.payloads[1]["tools"] == []
    code = next(item for item in responses.payloads[1]["input"] if item.get("type") == "code_interpreter_call")
    assert "outputs" in code and code["outputs"] is None


@pytest.mark.parametrize("verified", [False, True])
def test_mandatory_verification_requires_evidence_not_just_any_completed_cell(verified):
    api = _api(max_hosted_tool_calls=1, tools=HOSTED_TOOLS,
               required_hosted_tool_types=["code_interpreter"],
               required_hosted_tool_validator=lambda: verified, openai_continue_on_max_output_tokens=True)
    responses = _Responses([_reply(hosted=True, incomplete=True), _reply()])
    if verified:
        _run(api, responses)
        assert len(responses.payloads) == 2
    else:
        with pytest.raises(RequiredHostedToolUnavailable):
            _run(api, responses)
        assert len(responses.payloads) == 1


@pytest.mark.parametrize("status", ["completed", "failed", "incomplete"])
def test_continuation_preserves_supported_code_outputs_and_status(status):
    api = _api(tools=HOSTED_TOOLS, openai_continue_on_max_output_tokens=True)
    first = _reply(hosted=True, incomplete=True)
    first.output[0].status = status
    first.output[0].outputs = [
        {"type": "logs", "logs": "computed checksum", "extra": "not an input field"},
        SimpleNamespace(model_dump=lambda: {"type": "image", "url": "https://example.invalid/plot.png"}),
    ]
    responses = _Responses([first, _reply()])
    _run(api, responses)
    code = next(item for item in responses.payloads[1]["input"] if item.get("type") == "code_interpreter_call")
    assert code["status"] == status
    assert code["outputs"] == [{"type": "logs", "logs": "computed checksum"},
                                {"type": "image", "url": "https://example.invalid/plot.png"}]


def test_outer_retries_share_allowance_but_new_queries_reset_it(monkeypatch):
    api = _api(max_hosted_tool_calls=1, max_retries=2, max_retries_inner=0,
               tools=HOSTED_TOOLS + [_local(lambda: "report")])
    responses = _Responses([_reply(hosted=True, local=True), RuntimeError("temporary error"), _reply(), _reply()])
    monkeypatch.setattr(api, "_run_query", lambda *args, **kwargs: _run(api, responses))
    api._run_query_with_retry(0, [{"role": "user", "content": "first"}])
    api._run_query_with_retry(1, [{"role": "user", "content": "second"}])
    assert responses.payloads[0]["max_tool_calls"] == 1
    assert all(t["type"] == "function" for t in responses.payloads[2]["tools"])
    assert "max_tool_calls" not in responses.payloads[2]
    assert responses.payloads[3]["max_tool_calls"] == 1


def test_local_tools_receive_original_call_deadline_across_retry(monkeypatch):
    clock = {"wall": 1000.0, "mono": 500.0}
    monkeypatch.setattr("mathagents.api_client.time.time", lambda: clock["wall"])
    monkeypatch.setattr("mathagents.api_client.time.monotonic", lambda: clock["mono"])
    deadlines = []

    def delegate(*, call_deadline_monotonic_s=None):
        deadlines.append(call_deadline_monotonic_s)
        return "report"

    api = _api(max_wallclock_per_call_s=600, max_retries=2, tools=[_local(delegate)])
    responses = _Responses([_reply(local=True, arguments='{"call_deadline_monotonic_s": 999999}'), _reply(local=True), _reply()])
    attempts = []

    def query(*args, **kwargs):
        attempts.append(1)
        clock["wall"] += 100
        clock["mono"] += 100
        if len(attempts) == 1:
            raise RuntimeError("retry before first response")
        return _run(api, responses)

    monkeypatch.setattr(api, "_run_query", query)
    api._run_query_with_retry(0, [{"role": "user", "content": "problem"}])
    assert deadlines == [1100.0, 1100.0]
    assert all("call_deadline_monotonic_s" not in p for p in responses.payloads)


def test_deadline_injection_is_optional_and_does_not_reach_unrelated_tools():
    api = _api(tools=[_local(lambda: "plain")])
    assert api._execute_tool_function("delegate", {}, [], call_deadline_monotonic_s=10) == "plain"
    received = []
    api = _api(max_wallclock_per_call_s=None,
               tools=[_local(lambda call_deadline_monotonic_s=None: received.append(call_deadline_monotonic_s) or "report")])
    _run(api, _Responses([_reply(local=True), _reply()]))
    assert received == [None]


def test_short_responses_deadline_prevents_real_lead_wave_launch(tmp_path, monkeypatch):
    from proofstack.agents.ac.author import Author
    from proofstack.agents.ac.multi_author import MultiAuthor
    from proofstack.budget import BudgetSpec
    from proofstack.context import RunContext

    monkeypatch.setattr("mathagents.api_client.time.time", lambda: 1000.0)

    async def scenario():
        ctx = RunContext.create(
            root_workdir=tmp_path, flat=True,
            run_budget=BudgetSpec(max_wallclock_s=80000),
            component_configs={"Author": {"delegation": {"enabled": True}}},
        )
        lead = MultiAuthor(ctx, name="Author")
        lead._render_container_messages(Author.Inputs(problem="P", round=1, n_rounds=2), "")

        async def forbidden_wave(*args, **kwargs):
            pytest.fail("a 600-second lead call cannot afford the 1500-second synthesis reserve")

        monkeypatch.setattr(lead, "_run_wave", forbidden_wave)
        api = _api(max_wallclock_per_call_s=600, tools=[lead._delegate_tool()])
        responses = _Responses([
            _reply(local=True, arguments='{"tasks":[{"role":"prover","task":"Prove L"}]}'),
            _reply(),
        ])
        monkeypatch.setattr(api, "_run_query", lambda *args, **kwargs: _run(api, responses))
        result = await asyncio.to_thread(api._run_query_with_retry, 0, [{"role": "user", "content": "problem"}])
        assert lead._waves_done == 0
        assert len(responses.payloads) == 2
        tool_reply = next(m for m in responses.payloads[1]["input"] if m.get("type") == "function_call_output")
        assert "Not enough wallclock budget" in tool_reply["output"]
        assert result.conversation[-1]["content"].strip() == "done"

    asyncio.run(scenario())


@pytest.mark.parametrize("optional_form", ["empty", "null", "omitted"])
def test_async_optional_arguments_survive_responses_tool_loop(tmp_path, monkeypatch, optional_form):
    from proofstack.agents.ac.async_helpers import helper_scope
    from proofstack.agents.ac.multi_author import SubAuthorSeat
    from test_async_helpers import lead_at

    async def scenario():
        async def work(seat, inp):
            return seat.Outputs(report="verified")

        monkeypatch.setattr(SubAuthorSeat, "run", work)
        async with helper_scope():
            lead = lead_at(tmp_path, max_threads=4, max_tasks_per_turn=4)
            tasks = [{"role": role, "task": f"Verify n={n}", "include_workspace": True, "depends_on": []}
                     for n, role in zip((12, 13, 14, 15), ("explorer", "prover", "checker", "prover"))]
            status_args, read_args = {}, {"path": "manifest.json", "offset": 0, "max_chars": 24000}
            if optional_form != "omitted":
                for task in tasks:
                    task["agent_id"] = "" if optional_form == "empty" else None
                status_args["agent_ids"] = [] if optional_form == "empty" else None
                read_args["revision"] = "" if optional_form == "empty" else None
            calls = [("delegate", {"tasks": tasks}), ("helper_status", status_args), ("read_context", read_args)]
            replies = []
            for name, arguments in calls:
                reply = _reply(local=True, arguments=json.dumps(arguments))
                reply.output[0].name = name
                replies.append(reply)
            responses = _Responses([*replies, _reply()])
            api = _api(tools=lead._context_tools() + lead._helper_tools())
            await asyncio.to_thread(_run, api, responses)
            outputs = [m["output"] for m in responses.payloads[-1]["input"]
                       if m.get("type") == "function_call_output"]
            results = [json.JSONDecoder().raw_decode(body)[0] for body in outputs]
            assert len(results) == 3 and all(not result.get("error") for result in results)
            assert len(results[0]["helpers"]) == len(results[1]["helpers"]) == 4
            assert json.loads(results[2]["content"])["files"]
            descriptions = {t["name"]: t["parameters"] for t in responses.payloads[0]["tools"]}
            assert descriptions["read_context"]["properties"]["revision"]["type"] == ["string", "null"]
            assert descriptions["helper_status"]["properties"]["agent_ids"]["type"] == ["array", "null"]
            assert descriptions["delegate"]["properties"]["tasks"]["items"]["properties"]["agent_id"]["type"] == ["string", "null"]
            await asyncio.gather(*lead._async_session.tasks.values())
            assert all(row["status"] == "completed" for row in lead._async_session.records.values())
    asyncio.run(scenario())


@pytest.mark.parametrize("builder", ["inline", "openai_files", "anthropic_files"])
@pytest.mark.parametrize("all_invalid", [False, True])
def test_delegate_validation_errors_do_not_spend_real_waves(tmp_path, monkeypatch, builder, all_invalid):
    from proofstack.agents.ac.author import Author
    from proofstack.agents.ac.multi_author import MultiAuthor
    from proofstack.context import RunContext

    async def scenario():
        configs, waves = [], []

        def factory(cfg):
            configs.append(cfg)
            return SimpleNamespace(tool_descriptions=[], model="fake")

        ctx = RunContext.create(root_workdir=tmp_path, flat=True, api_client_factory=factory,
            component_configs={"Author": {"delegation": {"enabled": True, "max_waves": 2}}})
        lead = MultiAuthor(ctx, name="Author")
        lead._render_container_messages(Author.Inputs(problem="P", round=1, n_rounds=2), "")

        async def wave(number, *args):
            waves.append(number)
            return f"Completed wave {number}"

        monkeypatch.setattr(lead, "_run_wave", wave)
        if builder == "inline":
            cfg = lead.extra_client_kwargs()
        else:
            if builder == "openai_files":
                lead._build_api_client_with_file_ids([])
            else:
                lead._build_anthropic_api_client_with_files()
            cfg = configs[-1]
        api = _api(tools=cfg["tools"], max_tool_calls=cfg["max_tool_calls"], max_wallclock_per_call_s=7200)
        valid = {"tasks": [{"role": "prover", "task": "Prove L"}]}
        invalid = {"tasks": [{"role": "checker", "task": "Check L", "depends_on": ["missing"]}]}
        attempts = [invalid] * 5 if all_invalid else [valid, invalid, valid, valid, valid]
        responses = _Responses([
            *[_reply(local=True, arguments=json.dumps(args)) for args in attempts], _reply(),
        ])
        await asyncio.to_thread(_run, api, responses)

        assert waves == ([] if all_invalid else [1, 2])
        assert lead._waves_done == len(waves)
        assert cfg["max_tool_calls"]["delegate"] == 5
        names = lambda payload: [t.get("name") for t in payload["tools"] if t["type"] == "function"]
        assert "delegate" in names(responses.payloads[2])  # correction remains possible
        assert "delegate" not in names(responses.payloads[-1])  # attempts remain bounded
        outputs = [m["output"] for m in responses.payloads[-1]["input"] if m.get("type") == "function_call_output"]
        assert "depends_on must name completed agents" in outputs[0 if all_invalid else 1]
        if not all_invalid:
            assert "Completed wave 2" in outputs[2]
            assert "No delegation waves left" in outputs[3]

    asyncio.run(scenario())


@pytest.mark.parametrize("provider", ["anthropic", "chat_completions"])
def test_other_provider_tools_receive_original_deadline_after_outer_retry(provider, monkeypatch):
    clock = {"wall": 1000.0, "mono": 500.0}
    monkeypatch.setattr("mathagents.api_client.time.time", lambda: clock["wall"])
    monkeypatch.setattr("mathagents.api_client.time.monotonic", lambda: clock["mono"])
    received = []

    def delegate(call_deadline_monotonic_s=None):
        received.append(call_deadline_monotonic_s)
        return "report"

    api = APIClient(model="fake", api="custom", max_wallclock_per_call_s=600,
                    max_retries=2, sleep_after_request=0, sleep_on_error=0,
                    tools=[_local(delegate)])
    if provider == "anthropic":
        tool_data = {"type": "tool_use", "id": "tool1", "name": "delegate",
                     "input": {"call_deadline_monotonic_s": 999999}}
        text_data = {"type": "text", "text": "done"}
        tool = SimpleNamespace(**tool_data, model_dump=lambda: tool_data)
        text = SimpleNamespace(**text_data, model_dump=lambda: text_data)
        replies = [
            SimpleNamespace(content=[tool], usage={"input_tokens": 1, "output_tokens": 1},
                            stop_reason="tool_use", model_dump=lambda: {}),
            SimpleNamespace(content=[text], usage={"input_tokens": 1, "output_tokens": 1},
                            stop_reason="end_turn", model_dump=lambda: {}),
        ]
        responses = _Responses(replies)
        monkeypatch.setattr("mathagents.api_client.anthropic.Anthropic",
                            lambda **kwargs: SimpleNamespace(messages=responses))
        provider_query = lambda: api._anthropic_query_with_tools(0, [{"role": "user", "content": "problem"}])
    else:
        tool = SimpleNamespace(id="tool1", type="function", function=SimpleNamespace(
            name="delegate", arguments='{"call_deadline_monotonic_s": 999999}',
        ))
        tool_message = SimpleNamespace(tool_calls=[tool], model_dump=lambda: {
            "role": "assistant", "content": "", "tool_calls": [{"id": "tool1", "type": "function",
                "function": {"name": "delegate", "arguments": "{}"}}],
        })
        final_message = SimpleNamespace(tool_calls=[], model_dump=lambda: {"role": "assistant", "content": "done"})
        responses = _Responses([
            SimpleNamespace(choices=[SimpleNamespace(message=m)], usage={"input_tokens": 1, "output_tokens": 1},
                            model_dump=lambda: {})
            for m in (tool_message, final_message)
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=responses))
        provider_query = lambda: api._openai_query_chat_completions_api(client, 0, [{"role": "user", "content": "problem"}])

    attempts = []

    def query(*args, **kwargs):
        attempts.append(1)
        clock["wall"] += 100
        clock["mono"] += 100
        if len(attempts) == 1:
            raise RuntimeError("retry before first response")
        return provider_query()

    monkeypatch.setattr(api, "_run_query", query)
    api._run_query_with_retry(0, [{"role": "user", "content": "problem"}])
    assert received == [1100.0]
    assert len(responses.payloads) == 2
    assert all("call_deadline_monotonic_s" not in payload for payload in responses.payloads)
