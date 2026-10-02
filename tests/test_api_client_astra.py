from __future__ import annotations

import json
import os
import subprocess
import sys
from unittest.mock import patch

import httpx
import openai
import pytest

from mathagents.api_client import APIClient
from mathagents.config_loader import load_solver_config
from proofstack.registry import load_preset


def _client(config="models/openai/gpt-6-astra-pro", **overrides):
    cfg = {k: v for k, v in load_solver_config(config).items() if not k.startswith("__")}
    cfg["base_url"] = "https://astra.invalid/v1"
    cfg.update(overrides)
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-astra-key"}):
        return APIClient(**cfg)


@pytest.mark.parametrize(
    ("config", "effort", "mode"),
    [
        ("gpt-6-astra", None, None),
        ("gpt-6-astra-max", "max", None),
        ("gpt-6-astra-pro", "max", "pro"),
    ],
)
def test_astra_configs_preserve_mode_effort_and_timeout_policy(config, effort, mode):
    api = _client(f"models/openai/{config}")
    assert api.model == "gpt-6-astra"
    assert api.kwargs["reasoning"].get("mode") == mode
    assert api.kwargs["reasoning"].get("effort") == effort
    assert api.kwargs["reasoning"]["summary"] == "auto"
    assert api.kwargs["max_output_tokens"] == 128000
    assert api.background and api.use_openai_responses_api
    assert not api.batch_processing
    assert api.timeout == 11400
    assert api.max_wallclock_per_call_s == 14000
    assert api.throw_error_on_failure
    if effort:
        retry = api._kwargs_for_background_timeout_retry(1)
        assert retry["reasoning"]["effort"] == "high"
        assert retry["reasoning"].get("mode") == mode
        assert api.kwargs["reasoning"]["effort"] == "max"


@pytest.mark.parametrize(
    "preset_name",
    ["author_critic", "firstproof_submission", "author_critic_fable_author", "author_critic_fable_council"],
)
def test_research_presets_use_astra_pro_roles_and_astra_xhigh_compute(preset_name):
    preset = load_preset(preset_name)
    expected_author = (
        "models/anthropic/fable_51_max"
        if preset_name == "author_critic_fable_author"
        else "models/openai/gpt-6-astra-pro"
    )
    assert preset.component_configs["Author"]["model"] == expected_author
    assert preset.component_configs["ACCritic"]["model"] == "models/openai/gpt-6-astra-pro"
    assert preset.inputs["council_models"] == [
        "models/openai/gpt-6-astra-pro",
        "models/anthropic/fable_51",
        "models/gemini/gemini-31-pro",
    ]
    assert preset.inputs["compute_model"] == "gpt-6-astra"
    assert preset.inputs["compute_reasoning_effort"] == "xhigh"
    assert preset.inputs["compute_cost_config"] == "models/openai/gpt-6-astra"


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6-astra-2026-09-05"])
@pytest.mark.parametrize("responses", [True, False])
def test_astra_parameters_and_developer_messages(model, responses):
    api = _client(
        "models/openai/gpt-6-astra",
        model=model,
        background=responses,
        use_openai_responses_api=responses,
        temperature=0.5,
        top_p=0.9,
        top_logprobs=3,
        logprobs=True,
        include=["reasoning.encrypted_content", "message.output_text.logprobs"],
    )
    assert not {"temperature", "top_p", "top_logprobs", "logprobs"} & api.kwargs.keys()
    assert api.kwargs["include"] == ["reasoning.encrypted_content"]
    token_param = "max_output_tokens" if responses else "max_completion_tokens"
    assert api.kwargs[token_param] == 128000
    assert "max_tokens" not in api.kwargs
    query = [{"role": "developer", "content": "Write a proof."}, {"role": "user", "content": "P"}]
    assert api._validate_and_prepare_query(query) == query


def test_other_models_keep_sampling_and_message_behavior():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        api = APIClient(model="gpt-4.1", temperature=0.5, top_p=0.9, max_tokens=100)
    assert api.kwargs["temperature"] == 0.5
    assert api.kwargs["top_p"] == 0.9
    assert api.kwargs["max_tokens"] == 100
    assert api._validate_and_prepare_query([{"role": "developer", "content": "P"}]) == [
        {"role": "system", "content": "P"}
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_openai_responses_api": False},
        {"batch_processing": True},
        {"use_openai_responses_api": False, "reasoning": {}, "tools": [(None, {"type": "web_search"})]},
        {"batch_processing": True, "reasoning": {}, "tools": [(None, {"type": "code_interpreter"})]},
    ],
)
def test_astra_pro_and_tools_reject_chat_completions_path(overrides):
    with pytest.raises(ValueError, match="require the Responses API"):
        _client(**overrides)


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "cached", "cache_writes", "expected"),
    [
        (1000, 100, 200, 300, 0.01395),
        (272000, 1000, 100000, 50000, 1.995),
        (272001, 1000, 100000, 50000, 3.96502),
    ],
)
def test_astra_pricing_handles_cache_categories_and_long_context(input_tokens, output_tokens, cached, cache_writes, expected):
    assert _client()._get_cost(input_tokens, output_tokens, cached, cache_writes) == pytest.approx(expected)


def test_astra_pro_sdk_serializes_background_tool_loop_and_bills_each_response():
    # Older test modules install global SDK stubs during collection.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys\nrunpy.run_path(sys.argv[1])['_check_astra_sdk_tool_loop']()",
            __file__,
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_astra_sdk_tool_loop():
    requests = []
    tool_executions = []

    def compute_sum(a, b):
        tool_executions.append((a, b))
        return a + b

    function = {
        "type": "function",
        "function": {
            "name": "compute_sum",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
    api = _client(
        tools=[
            (compute_sum, function),
            (None, {"type": "web_search"}),
            (None, {"type": "code_interpreter", "container": {"type": "auto"}}),
        ],
        max_retries_inner=0,
        max_tool_calls=1,
        temperature=0.4,
        top_p=0.8,
    )
    outputs = [
        [
            {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "opaque-reasoning"},
            {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "compute_sum", "arguments": '{"a": 2, "b": 3}'},
        ],
        [{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "5", "annotations": []}]}],
    ]

    def handle(request):
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body))
        if request.method == "POST":
            index = sum(method == "POST" for method, _, _ in requests) - 1
            assert request.url.path == "/v1/responses"
            assert index < 2, "unexpected retry"
            status, output, usage = "queued", [], None
        else:
            index = int(request.url.path.rsplit("resp_", 1)[1])
            assert request.method == "GET"
            status, output = "completed", outputs[index]
            usage = {
                "input_tokens": 200000,
                "input_tokens_details": {"cached_tokens": 50000, "cache_write_tokens": 25000},
                "output_tokens": (index + 1) * 1000,
                "output_tokens_details": {"reasoning_tokens": (index + 1) * 1000 - 100},
                "total_tokens": 200000 + (index + 1) * 1000,
            }
        return httpx.Response(200, json={
            "id": f"resp_{index}", "object": "response", "created_at": 1788566400,
            "model": "gpt-6-astra", "status": status, "output": output,
            "usage": usage, "error": None,
        })

    with (
        openai.OpenAI(
            api_key="test-astra-key", base_url="https://astra.invalid/v1", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as sdk,
        patch("mathagents.api_client.OpenAI", return_value=sdk),
        patch("mathagents.api_client.time.sleep"),
        patch("mathagents.api_client.request_logger"),
    ):
        query = api._validate_and_prepare_query([
            {"role": "developer", "content": "Use compute_sum to add the integers."},
            {"role": "user", "content": "2 + 3"},
        ])
        result = api._openai_query_with_tools(0, query)

    assert [method for method, _, _ in requests] == ["POST", "GET", "POST", "GET"]
    payloads = [body for method, _, body in requests if method == "POST"]
    for body in payloads:
        assert body["model"] == "gpt-6-astra"
        assert body["reasoning"] == {"mode": "pro", "effort": "max", "summary": "auto"}
        assert body["background"] is True
        assert body["max_output_tokens"] == 128000
        assert body["input"][0]["role"] == "developer"
        assert not {"temperature", "top_p", "max_tokens", "max_completion_tokens"} & body.keys()
    assert {tool["type"] for tool in payloads[0]["tools"]} == {"function", "web_search", "code_interpreter"}
    assert {tool["type"] for tool in payloads[1]["tools"]} == {"web_search", "code_interpreter"}
    assert any(item.get("encrypted_content") == "opaque-reasoning" for item in payloads[1]["input"])
    assert any(item.get("type") == "function_call_output" and item["call_id"] == "call_1" for item in payloads[1]["input"])
    assert tool_executions == [(2, 3)]
    assert result.conversation[-1]["content"].strip() == "5"
    assert result.input_tokens == 400000
    assert result.cached_input_tokens == 100000
    assert result.cached_write_tokens == 50000
    assert result.output_tokens == 3000
    assert result.reasoning_tokens == 2800
    # Each request is below the long-context threshold; reasoning is already in output usage.
    assert result.cost_usd == pytest.approx(3.375)
