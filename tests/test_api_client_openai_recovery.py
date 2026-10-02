"""Recovery paths for long OpenAI Responses calls and provider tools on council seats.

- server-side ~60-min kill (status=failed, server_error, empty output) is
  classified as a background timeout so the retry uses the downgraded effort;
- a ``max_output_tokens`` cutoff (status=incomplete) gets one wrap-up turn;
- the Gemini native path maps provider tools, carries the developer message
  as ``systemInstruction`` and records code-execution parts;
- council seats declare the code sandbox + web search tools.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathagents.api_client import (  # noqa: E402
    APIClient,
    _OPENAI_MAX_OUTPUT_TOKENS_WRAPUP_PROMPT,
)
from mathagents.config_loader import load_solver_config  # noqa: E402
from proofstack.agents.ac.council import CouncilMember  # noqa: E402
from proofstack.agents.ac.critic import ACCritic  # noqa: E402
from proofstack.kinds.api_call import _assistant_text  # noqa: E402


class LongCallConfigTests(unittest.TestCase):
    def test_astra_and_sol_configs_enable_server_kill_and_max_output_recovery(self) -> None:
        for config in ("gpt-6-astra", "gpt-6-astra-max", "gpt-6-astra-pro", "gpt-56-sol-pro"):
            cfg = {k: v for k, v in load_solver_config(f"models/openai/{config}").items() if not k.startswith("__")}
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                api = APIClient(**cfg)
            with self.subTest(config=config):
                self.assertEqual(api.background_server_kill_after_s, 3300)
                self.assertTrue(api.openai_continue_on_max_output_tokens)
                self.assertEqual(api.openai_max_output_token_continuations, 1)
                self.assertEqual(api.openai_max_output_token_continuation_effort, "high")
                if config != "gpt-6-astra":
                    self.assertEqual(api.background_timeout_reasoning_efforts, ["high"])


def _message(text: str, msg_id: str = "msg_1"):
    return SimpleNamespace(
        type="message",
        id=msg_id,
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def _reasoning(item_id: str):
    return SimpleNamespace(type="reasoning", id=item_id, summary=[], encrypted_content=None)


def _usage(in_tokens=10, out_tokens=5):
    return SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens, total_tokens=in_tokens + out_tokens)


def _completed(text: str, resp_id: str = "resp_done"):
    return SimpleNamespace(
        id=resp_id,
        status="completed",
        output=[_message(text)],
        usage=_usage(),
        model_dump=lambda: {"status": "completed"},
    )


def _failed_kill(resp_id: str = "resp_killed"):
    return SimpleNamespace(
        id=resp_id,
        status="failed",
        error=SimpleNamespace(code="server_error", message="An error occurred while processing your request."),
        output=[],
        usage=None,
        model_dump=lambda: {"status": "failed"},
    )


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _patched(clock: _FakeClock):
    return (
        patch("mathagents.api_client.request_logger.log_request"),
        patch("mathagents.api_client.request_logger.log_response"),
        patch("mathagents.api_client.time.time", clock.time),
        patch("mathagents.api_client.time.sleep", clock.sleep),
    )


class ServerKillClassificationTests(unittest.TestCase):
    def _api(self, **overrides):
        kwargs = dict(
            model="gpt-6-astra--max",
            api="custom",
            use_openai_responses_api=True,
            background=True,
            timeout=11400,
            max_wallclock_per_call_s=14000,
            background_timeout_downgrade_after=1,
            background_timeout_reasoning_efforts=["high"],
            background_server_kill_after_s=3300,
        )
        kwargs.update(overrides)
        return APIClient(**kwargs)

    def _run(self, api, fail_after_s: float):
        clock = _FakeClock()

        class _Responses:
            def __init__(self):
                self.payloads = []
                self.cancelled = []

            def create(self, **payload):
                self.payloads.append(payload)
                if len(self.payloads) == 1:
                    return SimpleNamespace(id="resp_1", status="queued", model_dump=lambda: {})
                return _completed("done")

            def retrieve(self, response_id, **kwargs):
                if clock.now >= fail_after_s:
                    return _failed_kill(response_id)
                return SimpleNamespace(id=response_id, status="in_progress", model_dump=lambda: {})

            def cancel(self, response_id, **kwargs):
                self.cancelled.append(response_id)

        responses = _Responses()
        patches = _patched(clock)
        with patches[0], patches[1], patches[2], patches[3]:
            result = api._openai_query_responses_api(
                SimpleNamespace(responses=responses), 0, [{"role": "user", "content": "hi"}]
            )
        return responses, result

    def test_kill_after_threshold_downgrades_effort_on_retry(self) -> None:
        responses, result = self._run(self._api(), fail_after_s=3620)

        self.assertEqual(len(responses.payloads), 2)
        self.assertEqual(responses.payloads[0]["reasoning"]["effort"], "max")
        self.assertEqual(responses.payloads[1]["reasoning"]["effort"], "high")
        self.assertEqual(responses.cancelled, [])
        self.assertEqual(result.conversation[-1]["content"].strip(), "done")

    def test_early_failure_is_a_plain_retry_at_same_effort(self) -> None:
        responses, _ = self._run(self._api(), fail_after_s=120)

        self.assertEqual(len(responses.payloads), 2)
        self.assertEqual(responses.payloads[1]["reasoning"]["effort"], "max")

    def test_classification_is_off_without_threshold(self) -> None:
        responses, _ = self._run(self._api(background_server_kill_after_s=None), fail_after_s=3620)

        self.assertEqual(len(responses.payloads), 2)
        self.assertEqual(responses.payloads[1]["reasoning"]["effort"], "max")


class MaxOutputTokensContinuationTests(unittest.TestCase):
    def _api(self, **overrides):
        kwargs = dict(
            model="gpt-6-astra--max",
            api="custom",
            use_openai_responses_api=True,
            background=False,
            timeout=100,
            max_wallclock_per_call_s=1000,
            openai_continue_on_max_output_tokens=True,
            openai_max_output_token_continuations=1,
            openai_max_output_token_continuation_effort="high",
        )
        kwargs.update(overrides)
        return APIClient(**kwargs)

    @staticmethod
    def _incomplete():
        return SimpleNamespace(
            id="resp_incomplete",
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output=[
                _reasoning("rs_1"),
                _message("partial text", "msg_partial"),
                _reasoning("rs_2"),
                _reasoning("rs_3"),
            ],
            usage=_usage(100, 128000),
            model_dump=lambda: {"status": "incomplete"},
        )

    def _run(self, api, replies, messages=None):
        clock = _FakeClock()
        payloads = []

        def create(**payload):
            payloads.append(payload)
            reply = replies[len(payloads) - 1]
            if isinstance(reply, Exception):
                raise reply
            return reply

        patches = _patched(clock)
        with patches[0], patches[1], patches[2], patches[3]:
            result = api._openai_query_responses_api(
                SimpleNamespace(responses=SimpleNamespace(create=create)),
                0,
                messages if messages is not None else [{"role": "user", "content": "write the proof"}],
            )
        return payloads, result

    def test_failed_wrapup_keeps_accrued_usage_and_partial_output(self) -> None:
        payloads, result = self._run(
            self._api(max_retries_inner=0),
            [self._incomplete(), RuntimeError("provider outage")],
        )

        self.assertEqual(len(payloads), 2)
        self.assertEqual(result.conversation[-1]["id"], "msg_partial")
        self.assertNotIn(_OPENAI_MAX_OUTPUT_TOKENS_WRAPUP_PROMPT, [m.get("content") for m in result.conversation])
        self.assertEqual((result.input_tokens, result.output_tokens), (100, 128000))

    def test_terminal_wrapup_failure_keeps_usage_without_retrying(self) -> None:
        api = self._api(read_cost=10, write_cost=50, max_retries_inner=25)
        payloads, result = self._run(
            api, [self._incomplete(), RuntimeError("401 unauthorized")],
        )

        self.assertEqual(len(payloads), 2)
        self.assertEqual(result.conversation[-1]["id"], "msg_partial")
        self.assertEqual((result.input_tokens, result.output_tokens), (100, 128000))
        self.assertAlmostEqual(result.cost_usd, api._get_cost(100, 128000))
        self.assertEqual(result.n_retries, 1)

    def test_terminal_wrapup_usage_reaches_public_query_accounting(self) -> None:
        api = self._api(read_cost=10, write_cost=50)

        def run_query(idx, messages, ignore_tool_calls=False):
            return self._run(
                api, [self._incomplete(), RuntimeError("401 unauthorized")], messages,
            )[1]

        with patch.object(api, "_run_query", side_effect=run_query) as query:
            records = list(api.run_queries(
                [[{"role": "user", "content": "write the proof"}]], no_tqdm=True,
            ))
        self.assertEqual(query.call_count, 1)
        self.assertEqual(len(records), 1)
        _idx, conversation, cost = records[0]
        self.assertEqual(_assistant_text(conversation).strip(), "partial text")
        self.assertEqual((cost["input_tokens"], cost["output_tokens"]), (100, 128000))
        self.assertAlmostEqual(cost["cost"], api._get_cost(100, 128000))

    def test_terminal_failure_without_a_partial_result_still_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "401 unauthorized"):
            self._run(self._api(), [RuntimeError("401 unauthorized")])

    def test_empty_recovery_never_reuses_a_prior_critic_verdict(self) -> None:
        messages = [
            {"role": "user", "content": "Review draft one"},
            {"role": "assistant", "content": "Old review. <answer_ready>true</answer_ready>"},
            {"role": "user", "content": "Review the revised draft"},
        ]
        for failure in (False, True):
            with self.subTest(terminal_wrapup_failure=failure):
                first = self._incomplete()
                first.output = [_reasoning("rs_first")]
                second = self._incomplete()
                second.output = [_reasoning("rs_wrapup")]
                if failure:
                    second = RuntimeError("401 unauthorized")
                payloads, result = self._run(self._api(), [first, second], messages)
                self.assertEqual(len(payloads), 2)
                self.assertEqual(_assistant_text(result.conversation), "")
                self.assertEqual(result.output_tokens, 128000 if failure else 256000)
                self.assertEqual(result.conversation[:len(messages)], messages)
                critic = object.__new__(ACCritic)
                critic.ctx = SimpleNamespace(component_config_for=lambda _: {})
                verdict = critic.parse_output(
                    _assistant_text(result.conversation), ACCritic.Inputs(problem="P"),
                )
                self.assertTrue(verdict.parse_failed)
                self.assertFalse(verdict.answer_ready)

    def test_wrapup_turn_replays_history_without_trailing_reasoning(self) -> None:
        payloads, result = self._run(self._api(), [self._incomplete(), _completed("final text")])

        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["reasoning"]["effort"], "max")
        self.assertEqual(payloads[1]["reasoning"]["effort"], "high")
        replayed = payloads[1]["input"]
        self.assertEqual(replayed[-1], {"role": "user", "content": _OPENAI_MAX_OUTPUT_TOKENS_WRAPUP_PROMPT})
        self.assertEqual(replayed[-2]["id"], "msg_partial")
        self.assertEqual([item["id"] for item in replayed if item.get("type") == "reasoning"], ["rs_1"])
        self.assertEqual(result.conversation[-1]["content"].strip(), "final text")
        self.assertEqual(result.input_tokens, 110)
        self.assertEqual(result.output_tokens, 128005)

    def test_continuation_budget_is_bounded(self) -> None:
        payloads, result = self._run(self._api(), [self._incomplete(), self._incomplete()])

        self.assertEqual(len(payloads), 2)
        self.assertEqual(result.conversation[-1]["id"], "msg_partial")

    def test_disabled_continuation_accepts_truncated_output(self) -> None:
        payloads, result = self._run(
            self._api(openai_continue_on_max_output_tokens=False), [self._incomplete()]
        )

        self.assertEqual(len(payloads), 1)
        self.assertEqual(result.conversation[-1]["id"], "msg_partial")


class GoogleNativeToolsTests(unittest.TestCase):
    def _api(self):
        cfg = {
            k: v for k, v in load_solver_config("models/gemini/gemini-31-pro").items()
            if not k.startswith("__")
        }
        critic = object.__new__(ACCritic)
        critic.ctx = SimpleNamespace(model_for=lambda *_: "models/gemini/gemini-31-pro")
        cfg.update(critic.extra_client_kwargs())
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-google-key"}):
            return APIClient(**cfg)

    def _reply(self, parts):
        return SimpleNamespace(status_code=200, text="", json=lambda: {
            "candidates": [{"content": {"role": "model", "parts": parts}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        })

    def test_stateful_critic_sends_all_history_turns(self) -> None:
        api = self._api()
        critic = object.__new__(ACCritic)
        critic.ctx = SimpleNamespace(component_config_for=lambda _: {})
        prior = [
            {"role": "developer", "content": "Review carefully."},
            {"role": "user", "content": "Review draft one"},
            {"role": "assistant", "content": "There is a gap. <answer_ready>false</answer_ready>"},
        ]
        inp = ACCritic.Inputs(problem="P", mode="stateful", prior_messages=prior)
        messages = critic.render_messages(inp)
        with patch("mathagents.api_client.requests.post", return_value=self._reply([
            {"text": "Revised review. <answer_ready>true</answer_ready>"},
        ])) as post, patch("mathagents.api_client.request_logger"):
            result = api._run_query(0, api._validate_and_prepare_query(messages))

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["contents"], [
            {"role": "user", "parts": [{"text": prior[1]["content"]}]},
            {"role": "model", "parts": [{"text": prior[2]["content"]}]},
            {"role": "user", "parts": [{"text": messages[-1]["content"]}]},
        ])
        self.assertEqual(payload["systemInstruction"], {"parts": [{"text": prior[0]["content"]}]})
        self.assertTrue(critic.parse_output(_assistant_text(result.conversation), inp).answer_ready)
        self.assertEqual(len(prior), 3)

    def test_native_history_replays_tool_parts_and_signatures_once(self) -> None:
        api = self._api()
        parts = [
            {"thought": True, "text": "checking", "thoughtSignature": "opaque-1"},
            {"executableCode": {"language": "PYTHON", "code": "print(4)"}},
            {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "4"}},
            {"text": "The answer ", "thoughtSignature": "opaque-2"},
            {"text": "is 4."},
            {"thoughtSignature": "opaque-3"},
        ]
        messages = [{"role": "user", "content": "Compute"}]
        with patch("mathagents.api_client.requests.post", side_effect=[
            self._reply(parts), self._reply([{"text": "Checked again."}]),
        ]) as post, patch("mathagents.api_client.request_logger"):
            first = api._run_query(0, messages)
            self.assertEqual(_assistant_text(first.conversation), "The answer is 4.")
            second = api._run_query(1, first.conversation + [{"role": "user", "content": "Check again"}])

        self.assertEqual(post.call_args.kwargs["json"]["contents"], [
            {"role": "user", "parts": [{"text": "Compute"}]},
            {"role": "model", "parts": parts},
            {"role": "user", "parts": [{"text": "Check again"}]},
        ])
        self.assertEqual(_assistant_text(second.conversation), "Checked again.")
        self.assertEqual(messages, [{"role": "user", "content": "Compute"}])

    def test_native_reasoning_only_reply_is_not_a_visible_answer(self) -> None:
        api = self._api()
        messages = [
            {"role": "user", "content": "Review draft one"},
            {"role": "assistant", "content": "An old verdict."},
            {"role": "user", "content": "Review draft two"},
        ]
        parts = [{"thought": True, "text": "Still checking", "thoughtSignature": "opaque"}]
        with patch("mathagents.api_client.requests.post", return_value=self._reply(parts)), patch(
            "mathagents.api_client.request_logger"
        ):
            result = api._run_query(0, messages)
        self.assertEqual(_assistant_text(result.conversation), "")
        self.assertEqual(result.conversation[-1]["google_parts"], parts)

    def test_native_request_maps_tools_system_instruction_and_code_parts(self) -> None:
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json
            body = {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"thought": True, "text": "let me compute"},
                                {"executableCode": {"language": "PYTHON", "code": "print(2+2)"}},
                                {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "4\n"}},
                                {"text": "The answer is 4."},
                            ],
                        },
                        "groundingMetadata": {"webSearchQueries": ["shearer bound"]},
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 50,
                    "toolUsePromptTokenCount": 40,
                    "cachedContentTokenCount": 30,
                    "candidatesTokenCount": 20,
                    "thoughtsTokenCount": 7,
                },
            }
            return SimpleNamespace(status_code=200, json=lambda: body, text="")

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-google-key"}):
            api = APIClient(
                model="gemini-3.1-pro-preview",
                api="google",
                use_gdm_tools=True,
                max_tokens=65536,
                tools=[
                    (None, {"type": "code_interpreter", "container": {"type": "auto"}}),
                    (None, {"type": "web_search_preview"}),
                ],
                extra_body={"extra_body": {"google": {"thinking_config": {"include_thoughts": True, "thinking_level": "high"}}}},
            )
        self.assertTrue(api.use_google_internal_tools)

        with patch("mathagents.api_client.requests.post", fake_post), patch(
            "mathagents.api_client.request_logger.log_request"
        ), patch("mathagents.api_client.request_logger.log_response"):
            result = api._run_query(
                0,
                [
                    {"role": "developer", "content": "You are a council member."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
            )

        payload = captured["payload"]
        self.assertIn("gemini-3.1-pro-preview:generateContent", captured["url"])
        self.assertEqual(payload["systemInstruction"], {"parts": [{"text": "You are a council member."}]})
        self.assertEqual(payload["contents"], [{"role": "user", "parts": [{"text": "What is 2+2?"}]}])
        self.assertEqual(payload["tools"], [{"codeExecution": {}}, {"googleSearch": {}}])
        self.assertEqual(
            payload["generationConfig"],
            {"maxOutputTokens": 65536, "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "high"}},
        )

        kinds = [(m.get("role"), m.get("type")) for m in result.conversation[2:]]
        self.assertEqual(
            kinds,
            [
                ("assistant", "web_search_call"),
                ("assistant", "cot"),
                ("assistant", "code_interpreter_call"),
                ("tool", "code_interpreter_result"),
                ("assistant", "response"),
            ],
        )
        self.assertEqual(result.conversation[-1]["content"], "The answer is 4.")
        self.assertEqual(result.conversation[-2]["content"], "4\n")
        # Gemini bills thinking at the output rate (20 candidate + 7 thought tokens)
        # and built-in tool re-prompts at the input rate (50 prompt + 40 tool-use).
        self.assertEqual((result.input_tokens, result.output_tokens, result.reasoning_tokens), (90, 27, 7))
        self.assertEqual(result.cached_input_tokens, 30)

    def test_native_routing_only_for_provider_managed_tools(self) -> None:
        provider_tools = [
            (None, {"type": "code_interpreter", "container": {"type": "auto"}}),
            (None, {"type": "web_search_preview"}),
        ]
        local_tool = (
            lambda: "ok",
            {"type": "function", "function": {"name": "probe", "description": "", "parameters": {"type": "object", "properties": {}}}},
        )
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-google-key"}):
            native = APIClient(model="gemini-3.1-pro-preview", api="google", use_gdm_tools=True, tools=provider_tools)
            local_only = APIClient(model="gemini-3.1-pro-preview", api="google", use_gdm_tools=True, tools=[local_tool])
            no_tools = APIClient(model="gemini-3.1-pro-preview", api="google", use_gdm_tools=True)
            # Neither endpoint can serve provider-managed and local tools
            # together, so a mixed set fails fast at construction.
            with self.assertRaisesRegex(ValueError, "cannot be combined with local function tools"):
                APIClient(
                    model="gemini-3.1-pro-preview", api="google", use_gdm_tools=True, tools=provider_tools + [local_tool]
                )

        self.assertTrue(native.use_google_internal_tools)
        self.assertEqual(native.api, "google")
        # Local function tools keep the Chat Completions tool loop, which
        # the native path does not provide.
        self.assertFalse(local_only.use_google_internal_tools)
        self.assertEqual(local_only.api, "openai")
        self.assertFalse(no_tools.use_google_internal_tools)
        self.assertEqual(no_tools.api, "openai")


class CouncilToolsTests(unittest.TestCase):
    def test_council_member_declares_code_sandbox_and_web_search(self) -> None:
        kwargs = CouncilMember.extra_client_kwargs(object.__new__(CouncilMember))

        self.assertEqual(
            [desc["type"] for _fn, desc in kwargs["tools"]],
            ["code_interpreter", "web_search_preview"],
        )
        self.assertEqual(kwargs["max_tool_calls"], CouncilMember.MAX_TOOL_CALLS)


if __name__ == "__main__":
    unittest.main()
