from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if "anthropic" not in sys.modules:
    anthropic = types.ModuleType("anthropic")
    anthropic.NOT_GIVEN = object()
    anthropic.Anthropic = object
    sys.modules["anthropic"] = anthropic

    anthropic_types = types.ModuleType("anthropic.types")
    anthropic_types.TextBlock = type("TextBlock", (), {})
    anthropic_types.ThinkingBlock = type("ThinkingBlock", (), {})
    sys.modules["anthropic.types"] = anthropic_types

    anthropic_message_params = types.ModuleType("anthropic.types.message_create_params")
    anthropic_message_params.MessageCreateParamsNonStreaming = dict
    sys.modules["anthropic.types.message_create_params"] = anthropic_message_params

    anthropic_batch_params = types.ModuleType("anthropic.types.messages.batch_create_params")
    anthropic_batch_params.Request = dict
    sys.modules["anthropic.types.messages.batch_create_params"] = anthropic_batch_params

if "openai" not in sys.modules:
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    openai.RateLimitError = RuntimeError
    sys.modules["openai"] = openai

if "together" not in sys.modules:
    together = types.ModuleType("together")
    together.Together = object
    sys.modules["together"] = together

if "transformers" not in sys.modules:
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = object
    sys.modules["transformers"] = transformers

if "loguru" not in sys.modules:
    loguru = types.ModuleType("loguru")

    class _Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    loguru.logger = _Logger()
    sys.modules["loguru"] = loguru

from mathagents.api_client import APIClient  # noqa: E402


def _function_tool(name: str = "list_persisted_files") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "List files.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


def _message(text: str):
    return SimpleNamespace(
        type="message",
        id="msg_1",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def _function_call(name: str = "list_persisted_files"):
    return SimpleNamespace(
        type="function_call",
        id="fc_1",
        call_id="call_1",
        name=name,
        arguments="{}",
    )


def _web_search_call():
    return SimpleNamespace(
        type="web_search_call",
        id="ws_1",
        status="completed",
        action=SimpleNamespace(type="search", queries=["irrationality of sqrt 2"]),
    )


def _code_interpreter_call():
    return SimpleNamespace(
        type="code_interpreter_call",
        id="ci_1",
        code='print("ok")',
        container_id="cntr_1",
        status="completed",
        results=[],
    )


def _reasoning():
    return SimpleNamespace(
        type="reasoning",
        id="rs_1",
        summary=[SimpleNamespace(text="thinking", type="summary_text")],
        content=None,
        encrypted_content=None,
    )


def _response(*outputs):
    return SimpleNamespace(
        output=list(outputs),
        usage={"input_tokens": 1, "output_tokens": 1},
        model_dump=lambda: {"output": [getattr(out, "type", "") for out in outputs]},
    )


class _Responses:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def create(self, **payload):
        self.payloads.append(payload)
        if not self.responses:
            raise AssertionError("unexpected extra Responses API call")
        return self.responses.pop(0)


class ResponsesToolLoopTests(unittest.TestCase):
    def test_default_tool_budget_is_unbounded_and_reaches_final_message(self) -> None:
        def list_persisted_files() -> dict:
            return {"ok": True, "files": []}

        api = APIClient(
            model="fake",
            api="custom",
            use_openai_responses_api=True,
            tools=[(list_persisted_files, _function_tool())],
        )
        responses = _Responses([
            _response(_function_call()),
            _response(_message("<solution>done</solution>")),
        ])
        client = SimpleNamespace(responses=responses)

        with patch("mathagents.api_client.request_logger.log_request"), patch(
            "mathagents.api_client.request_logger.log_response"
        ):
            result = api._openai_query_responses_api(client, 0, [{"role": "user", "content": "hi"}])

        self.assertEqual(len(responses.payloads), 2)
        self.assertEqual(api.max_tool_calls["any"], float("inf"))
        self.assertEqual(result.conversation[-1]["content"].strip(), "<solution>done</solution>")
        self.assertEqual(result.conversation[-2]["type"], "function_call_output")
        self.assertIn("'files': []", result.conversation[-2]["output"])

    def test_zero_tool_budget_removes_local_functions_from_responses_payload(self) -> None:
        def list_persisted_files() -> dict:
            return {"ok": True, "files": []}

        api = APIClient(
            model="fake",
            api="custom",
            use_openai_responses_api=True,
            max_tool_calls=0,
            tools=[
                (list_persisted_files, _function_tool()),
                (None, {"type": "code_interpreter", "container": {"type": "auto"}}),
            ],
        )
        responses = _Responses([_response(_message("done"))])
        client = SimpleNamespace(responses=responses)

        with patch("mathagents.api_client.request_logger.log_request"), patch(
            "mathagents.api_client.request_logger.log_response"
        ):
            api._openai_query_responses_api(client, 0, [{"role": "user", "content": "hi"}])

        self.assertEqual(responses.payloads[0]["tools"], [{"type": "code_interpreter", "container": {"type": "auto"}}])

    def test_provider_web_search_call_is_replayed_without_query_field(self) -> None:
        def list_persisted_files() -> dict:
            return {"ok": True, "files": []}

        api = APIClient(
            model="fake",
            api="custom",
            use_openai_responses_api=True,
            tools=[
                (list_persisted_files, _function_tool()),
                (None, {"type": "web_search_preview"}),
                (None, {"type": "code_interpreter", "container": {"type": "auto"}}),
            ],
        )
        responses = _Responses([
            _response(_reasoning(), _web_search_call(), _code_interpreter_call(), _function_call()),
            _response(_message("<solution>done</solution>")),
        ])
        client = SimpleNamespace(responses=responses)

        with patch("mathagents.api_client.request_logger.log_request"), patch(
            "mathagents.api_client.request_logger.log_response"
        ):
            api._openai_query_responses_api(client, 0, [{"role": "user", "content": "hi"}])

        replayed = responses.payloads[1]["input"]
        web_items = [item for item in replayed if item.get("type") == "web_search_call"]
        self.assertEqual(web_items, [{"type": "web_search_call", "id": "ws_1", "status": "completed"}])
        self.assertNotIn("query", web_items[0])
        code_items = [item for item in replayed if item.get("type") == "code_interpreter_call"]
        self.assertEqual(
            code_items,
            [
                {
                    "type": "code_interpreter_call",
                    "id": "ci_1",
                    "code": 'print("ok")',
                    "container_id": "cntr_1",
                    "status": "completed",
                }
            ],
        )
        self.assertNotIn("results", code_items[0])
        reasoning_items = [item for item in replayed if item.get("type") == "reasoning"]
        self.assertEqual(reasoning_items, [{"type": "reasoning", "id": "rs_1", "summary": [{"text": "thinking", "type": "summary_text"}]}])
        self.assertNotIn("content", reasoning_items[0])


class SalvagePrefersOutputTests(unittest.TestCase):
    """When OpenAI returns ``response.status='failed'`` with
    ``rate_limit_exceeded`` but ``response.output`` contains the model's
    actual reply, we must salvage the output rather than retry the call.

    The old order checked the rate-limit branch first and discarded a
    fully-formed response; that was responsible for two ~$0 / no-content
    Author rounds in the hilbert_admissible_v1 production run.
    """

    def test_salvage_takes_precedence_over_retry_when_output_present(self) -> None:
        queued = SimpleNamespace(id="resp_test_123", status="queued")
        failed_with_output = SimpleNamespace(
            id="resp_test_123",
            status="failed",
            error=SimpleNamespace(
                code="rate_limit_exceeded",
                message="Rate limit reached for gpt-5.5-pro. Please try again in 1.573s.",
            ),
            output=[_message("the complete model reply")],
            usage=None,
            model_dump=lambda: {"output": ["message"], "status": "failed"},
        )

        class _BackgroundResponses:
            def __init__(self):
                self.create_calls = 0
                self.retrieve_calls = 0

            def create(self, **payload):
                self.create_calls += 1
                return queued

            def retrieve(self, response_id):
                self.retrieve_calls += 1
                return failed_with_output

        responses = _BackgroundResponses()
        client = SimpleNamespace(responses=responses)

        api = APIClient(
            model="fake-pro",
            api="custom",
            use_openai_responses_api=True,
            background=True,
        )

        with patch("mathagents.api_client.request_logger.log_request"), patch(
            "mathagents.api_client.request_logger.log_response"
        ), patch("mathagents.api_client.time.sleep"):
            result = api._openai_query_responses_api(
                client, 0, [{"role": "user", "content": "hi"}]
            )

        self.assertEqual(
            responses.create_calls,
            1,
            "rate_limit_exceeded with non-empty output must NOT trigger a retry — "
            "the existing output is the salvageable response.",
        )
        self.assertEqual(
            result.conversation[-1]["content"].strip(),
            "the complete model reply",
            "salvaged output must reach the final conversation",
        )

    def test_retry_still_fires_when_output_is_empty(self) -> None:
        """Sanity: retry-on-rate-limit still works when there's nothing to salvage."""
        failed_empty = SimpleNamespace(
            id="resp_test_456",
            status="failed",
            error=SimpleNamespace(
                code="rate_limit_exceeded",
                message="Rate limit reached. Please try again in 1.0s.",
            ),
            output=[],
            usage=None,
            model_dump=lambda: {"output": [], "status": "failed"},
        )
        completed_after_retry = SimpleNamespace(
            id="resp_test_789",
            status="completed",
            output=[_message("recovered on retry")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
            model_dump=lambda: {"output": ["message"], "status": "completed"},
        )

        class _BackgroundResponses:
            def __init__(self):
                self.create_calls = 0
                self.retrieve_calls = 0

            def create(self, **payload):
                self.create_calls += 1
                # Each create starts a new queued response; the retrieve
                # sequence below decides what state it ends in.
                return SimpleNamespace(
                    id=f"resp_{self.create_calls}", status="queued"
                )

            def retrieve(self, response_id):
                self.retrieve_calls += 1
                # First poll: failed-with-no-output (triggers retry).
                # Second poll: a clean completed response on the retry.
                if self.retrieve_calls == 1:
                    return failed_empty
                return completed_after_retry

        responses = _BackgroundResponses()
        client = SimpleNamespace(responses=responses)

        api = APIClient(
            model="fake-pro",
            api="custom",
            use_openai_responses_api=True,
            background=True,
        )

        with patch("mathagents.api_client.request_logger.log_request"), patch(
            "mathagents.api_client.request_logger.log_response"
        ), patch("mathagents.api_client.time.sleep"):
            result = api._openai_query_responses_api(
                client, 0, [{"role": "user", "content": "hi"}]
            )

        self.assertEqual(
            responses.create_calls,
            2,
            "rate_limit_exceeded with EMPTY output should still trigger a retry",
        )
        self.assertEqual(
            result.conversation[-1]["content"].strip(),
            "recovered on retry",
        )


class BackgroundTimeoutRetryTests(unittest.TestCase):
    def test_timeout_retry_cancels_and_downgrades_reasoning_effort(self) -> None:
        queued = SimpleNamespace(
            id="resp_timeout_1",
            status="queued",
            model_dump=lambda: {"status": "queued"},
        )
        completed = SimpleNamespace(
            id="resp_done_1",
            status="completed",
            output=[_message("done at lower effort")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
            model_dump=lambda: {"output": ["message"], "status": "completed"},
        )

        class _BackgroundResponses:
            def __init__(self):
                self.create_calls = 0
                self.retrieve_calls = 0
                self.cancelled = []
                self.payloads = []

            def create(self, **payload):
                self.create_calls += 1
                self.payloads.append(payload)
                if self.create_calls == 1:
                    return queued
                return completed

            def retrieve(self, response_id):
                self.retrieve_calls += 1
                return queued

            def cancel(self, response_id):
                self.cancelled.append(response_id)

        clock = {"now": 0.0}

        def fake_time():
            return clock["now"]

        def fake_sleep(seconds):
            clock["now"] += seconds

        responses = _BackgroundResponses()
        client = SimpleNamespace(responses=responses)

        api = APIClient(
            model="gpt-5.5-pro--xhigh",
            api="custom",
            use_openai_responses_api=True,
            background=True,
            timeout=1,
            max_wallclock_per_call_s=300,
            background_timeout_downgrade_after=1,
            background_timeout_reasoning_efforts=["high"],
        )

        with patch("mathagents.api_client.request_logger.log_request"), patch(
            "mathagents.api_client.request_logger.log_response"
        ), patch("mathagents.api_client.time.time", fake_time), patch(
            "mathagents.api_client.time.sleep", fake_sleep
        ):
            result = api._openai_query_responses_api(
                client, 0, [{"role": "user", "content": "hi"}]
            )

        self.assertEqual(responses.create_calls, 2)
        self.assertEqual(responses.cancelled, ["resp_timeout_1"])
        self.assertEqual(responses.payloads[0]["reasoning"]["effort"], "xhigh")
        self.assertEqual(responses.payloads[1]["reasoning"]["effort"], "high")
        self.assertEqual(
            result.conversation[-1]["content"].strip(),
            "done at lower effort",
        )

    def test_poll_deadline_capped_by_remaining_wallclock_budget(self) -> None:
        # Per-attempt timeout is much larger than the inner-loop
        # wallclock budget. Without the wallclock-aware cap, the
        # polling loop would wait the full ``timeout`` before raising,
        # letting a stuck call hold the worker slot well past the
        # ``max_wallclock_per_call_s`` budget. The fix caps each
        # poll's deadline at ``min(timeout, remaining_wallclock)``.
        queued = SimpleNamespace(
            id="resp_wallclock_1",
            status="queued",
            model_dump=lambda: {"status": "queued"},
        )

        class _BackgroundResponses:
            def __init__(self):
                self.create_calls = 0
                self.cancelled = []

            def create(self, **payload):
                self.create_calls += 1
                return queued

            def retrieve(self, response_id):
                return queued

            def cancel(self, response_id):
                self.cancelled.append(response_id)

        clock = {"now": 0.0}

        def fake_time():
            return clock["now"]

        def fake_sleep(seconds):
            clock["now"] += seconds

        responses = _BackgroundResponses()
        client = SimpleNamespace(responses=responses)

        api = APIClient(
            model="gpt-5.5-pro--xhigh",
            api="custom",
            use_openai_responses_api=True,
            background=True,
            timeout=11400,
            max_wallclock_per_call_s=180,
        )

        with patch("mathagents.api_client.request_logger.log_request"), patch(
            "mathagents.api_client.request_logger.log_response"
        ), patch("mathagents.api_client.time.time", fake_time), patch(
            "mathagents.api_client.time.sleep", fake_sleep
        ):
            with self.assertRaises(ValueError):
                api._openai_query_responses_api(
                    client, 0, [{"role": "user", "content": "hi"}]
                )

        # The poll must have cancelled the stuck call rather than
        # waiting out the 11400s per-attempt timeout.
        self.assertGreaterEqual(len(responses.cancelled), 1)
        self.assertEqual(responses.cancelled[0], "resp_wallclock_1")
        # And the whole inner loop must have exited well before
        # ``self.timeout`` — within a small multiple of the wallclock
        # budget, not the per-attempt timeout.
        self.assertLess(clock["now"], 600.0)

    def test_first_attempt_runs_to_per_attempt_timeout(self) -> None:
        # Regression: the polling deadline used to be a duration that
        # shrunk as time passed (``min(self.timeout, max_wallclock -
        # elapsed)``), which truncated attempt 1 to roughly half the
        # inner-loop wallclock budget instead of the configured
        # per-attempt timeout. Pin the absolute-deadline behavior:
        # with no prior calls, attempt 1 must run until ~self.timeout
        # even when max_wallclock is comfortably larger.
        timeout = 4000
        max_wallclock = 5000

        queued = SimpleNamespace(
            id="resp_full_attempt_1",
            status="queued",
            model_dump=lambda: {"status": "queued"},
        )

        cancelled_at: list[float] = []

        class _BackgroundResponses:
            def create(self, **payload):
                return queued

            def retrieve(self, response_id):
                return queued

            def cancel(self, response_id):
                cancelled_at.append(clock["now"])

        clock = {"now": 0.0}

        def fake_time():
            return clock["now"]

        def fake_sleep(seconds):
            clock["now"] += seconds

        client = SimpleNamespace(responses=_BackgroundResponses())

        api = APIClient(
            model="gpt-5.5-pro--xhigh",
            api="custom",
            use_openai_responses_api=True,
            background=True,
            timeout=timeout,
            max_wallclock_per_call_s=max_wallclock,
        )

        with patch("mathagents.api_client.request_logger.log_request"), patch(
            "mathagents.api_client.request_logger.log_response"
        ), patch("mathagents.api_client.time.time", fake_time), patch(
            "mathagents.api_client.time.sleep", fake_sleep
        ):
            with self.assertRaises(ValueError):
                api._openai_query_responses_api(
                    client, 0, [{"role": "user", "content": "hi"}]
                )

        # Bugged version cancelled around t = max_wallclock / 2 (~2520s
        # for the production config). Pin the fix: first cancel must be
        # at or after ``self.timeout``, not at the wallclock midpoint.
        self.assertGreater(cancelled_at[0], float(timeout) - 60.0)
        self.assertLess(cancelled_at[0], float(timeout) + 120.0)


if __name__ == "__main__":
    unittest.main()
