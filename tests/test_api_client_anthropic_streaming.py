from __future__ import annotations

import os
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
    msg_params = types.ModuleType("anthropic.types.message_create_params")
    msg_params.MessageCreateParamsNonStreaming = dict
    sys.modules["anthropic.types.message_create_params"] = msg_params
    batch_params = types.ModuleType("anthropic.types.messages.batch_create_params")
    batch_params.Request = dict
    sys.modules["anthropic.types.messages.batch_create_params"] = batch_params

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

from mathagents import api_client as api_client_module  # noqa: E402
from mathagents.api_client import APIClient, _ProviderWallclockTimeout  # noqa: E402


class _FakeUsage:
    def __init__(
        self,
        input_tokens=7,
        output_tokens=11,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens

    def model_dump(self):
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
        }


class _FakeTextBlock:
    type = "text"

    def __init__(self, text="streamed answer"):
        self.text = text

    def model_dump(self):
        return {"type": self.type, "text": self.text}


class _FakeThinkingBlock:
    type = "thinking"

    def __init__(self, thinking="", signature="sig"):
        self.thinking = thinking
        self.signature = signature

    def model_dump(self):
        return {"type": self.type, "thinking": self.thinking, "signature": self.signature}


class _FakeServerBlock:
    def __init__(self, data):
        self._data = dict(data)
        self.type = self._data.get("type", "")

    def model_dump(self):
        return dict(self._data)


class _FakeMessage:
    def __init__(self, content=None, usage=None, stop_reason="end_turn"):
        self.usage = usage or _FakeUsage()
        self.content = content or [_FakeTextBlock()]
        self.stop_reason = stop_reason

    def model_dump(self):
        return {
            "usage": self.usage.model_dump(),
            "content": [block.model_dump() for block in self.content],
            "stop_reason": self.stop_reason,
        }


class _FakeStream:
    def __init__(self, message=None):
        self.message = message or _FakeMessage()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        return None

    def __iter__(self):
        yield object()

    def get_final_message(self):
        return self.message


class _FakeMessages:
    def __init__(self, stream_messages=None):
        self.create_calls = 0
        self.stream_calls = 0
        self.last_payload = None
        self.payloads = []
        self.stream_messages = list(stream_messages or [])

    def create(self, **payload):
        self.create_calls += 1
        raise AssertionError("non-streaming Anthropic path should not be used")

    def stream(self, **payload):
        self.stream_calls += 1
        self.last_payload = payload
        self.payloads.append(payload)
        message = self.stream_messages.pop(0) if self.stream_messages else None
        return _FakeStream(message=message)


class _FakeAnthropic:
    last_instance = None
    stream_messages = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = _FakeMessages(stream_messages=self.stream_messages)
        _FakeAnthropic.last_instance = self


class AnthropicStreamingTests(unittest.TestCase):
    def test_anthropic_maps_provider_tools_to_server_tools(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}):
            client = APIClient(
                model="claude-test",
                api="anthropic",
                max_tokens=16,
                tools=[
                    (None, {"type": "web_search_preview"}),
                    (None, {"type": "code_interpreter", "container": {"type": "auto"}}),
                ],
            )

        self.assertEqual(
            client._anthropic_tool_descriptions(),
            [
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "code_execution_20250825", "name": "code_execution"},
            ],
        )

    def test_anthropic_streaming_uses_stream_manager(self) -> None:
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}),
            patch.object(api_client_module.anthropic, "Anthropic", _FakeAnthropic),
            patch.object(api_client_module.request_logger, "log_request") as log_request,
            patch.object(api_client_module.request_logger, "log_response") as log_response,
        ):
            client = APIClient(
                model="claude-test",
                api="anthropic",
                max_tokens=16,
                timeout=1800,
                max_retries=1,
                max_retries_inner=0,
                max_wallclock_per_call_s=1800,
                stream_anthropic_messages=True,
            )
            result = client._anthropic_query_with_tools(
                3,
                [{"role": "user", "content": "hello"}],
            )

        messages = _FakeAnthropic.last_instance.messages
        self.assertEqual(messages.stream_calls, 1)
        self.assertEqual(messages.create_calls, 0)
        self.assertLessEqual(messages.last_payload["timeout"], 1800)
        self.assertGreater(messages.last_payload["timeout"], 1790)
        self.assertEqual(result.conversation[-1]["content"], "streamed answer")
        self.assertEqual(result.input_tokens, 7)
        self.assertEqual(result.output_tokens, 11)
        self.assertTrue(log_request.call_args.kwargs["stream_anthropic_messages"])
        log_response.assert_called_once()

    def test_anthropic_streaming_enforces_total_wallclock(self) -> None:
        client = APIClient.__new__(APIClient)
        client.stream_anthropic_messages = True
        client.max_wallclock_per_call_s = 10

        fake_client = types.SimpleNamespace(messages=_FakeMessages())
        with patch.object(api_client_module.time, "time", return_value=111.0):
            with self.assertRaises(_ProviderWallclockTimeout):
                client._create_anthropic_message(fake_client, {}, inner_start=100.0)

    def test_anthropic_streaming_preserves_start_usage_counts(self) -> None:
        client = APIClient.__new__(APIClient)
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=5,
                output_tokens=128000,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
        )
        start_usage = SimpleNamespace(
            input_tokens=9000,
            cache_creation_input_tokens=7000,
            cache_read_input_tokens=2000,
        )

        client._merge_anthropic_stream_start_usage(response, start_usage)

        self.assertEqual(response.usage.input_tokens, 9000)
        self.assertEqual(response.usage.cache_creation_input_tokens, 7000)
        self.assertEqual(response.usage.cache_read_input_tokens, 2000)
        self.assertEqual(response.usage.output_tokens, 128000)

    def test_anthropic_streaming_cache_creation_is_billed_in_run_queries(self) -> None:
        message = _FakeMessage(
            content=[_FakeTextBlock("streamed answer")],
            usage=_FakeUsage(
                input_tokens=5,
                output_tokens=11,
                cache_creation_input_tokens=1000,
            ),
        )

        class _CachedAnthropic(_FakeAnthropic):
            stream_messages = [message]

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}),
            patch.object(api_client_module.anthropic, "Anthropic", _CachedAnthropic),
            patch.object(api_client_module.request_logger, "log_request"),
            patch.object(api_client_module.request_logger, "log_response"),
        ):
            client = APIClient(
                model="claude-test",
                api="anthropic",
                max_tokens=16,
                timeout=1800,
                max_retries=1,
                max_retries_inner=0,
                max_wallclock_per_call_s=1800,
                stream_anthropic_messages=True,
                read_cost=5,
                write_cost=25,
                cache_write_cost=6.25,
            )
            results = list(
                client.run_queries(
                    [[{"role": "user", "content": "hello"}]],
                    no_tqdm=True,
                )
            )

        _, _, detailed_cost = results[0]
        self.assertEqual(detailed_cost["input_tokens"], 5)
        self.assertEqual(detailed_cost["output_tokens"], 11)
        self.assertEqual(detailed_cost["cached_write_tokens"], 1000)
        expected = (5 * 5 + 11 * 25 + 1000 * 6.25) / 1_000_000
        self.assertAlmostEqual(detailed_cost["cost"], expected)

    def test_anthropic_streaming_salvages_empty_max_token_response(self) -> None:
        first = _FakeMessage(
            content=[_FakeThinkingBlock()],
            usage=_FakeUsage(input_tokens=9000, output_tokens=128000),
            stop_reason="max_tokens",
        )
        second = _FakeMessage(
            content=[_FakeTextBlock("final council opinion")],
            usage=_FakeUsage(input_tokens=10, output_tokens=4),
        )

        class _SalvageAnthropic(_FakeAnthropic):
            stream_messages = [first, second]

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}),
            patch.object(api_client_module.anthropic, "Anthropic", _SalvageAnthropic),
            patch.object(api_client_module.request_logger, "log_request") as log_request,
            patch.object(api_client_module.request_logger, "log_response") as log_response,
        ):
            client = APIClient(
                model="claude-test",
                api="anthropic",
                max_tokens=128000,
                timeout=1800,
                max_retries=1,
                max_retries_inner=0,
                max_wallclock_per_call_s=1800,
                stream_anthropic_messages=True,
                anthropic_salvage_empty_max_tokens=True,
                thinking={"type": "adaptive", "display": "omitted"},
                output_config={"effort": "max"},
            )
            result = client._anthropic_query_with_tools(
                3,
                [{"role": "user", "content": "hard problem"}],
            )

        messages = _FakeAnthropic.last_instance.messages
        self.assertEqual(messages.stream_calls, 2)
        self.assertEqual(result.conversation[-1]["content"], "final council opinion")
        self.assertEqual(result.output_tokens, 128004)
        self.assertIn("thinking", messages.payloads[0])
        self.assertIn("output_config", messages.payloads[0])
        self.assertNotIn("thinking", messages.payloads[1])
        self.assertNotIn("output_config", messages.payloads[1])
        self.assertTrue(log_request.call_args.kwargs["anthropic_final_answer_salvage"])
        self.assertEqual(log_response.call_count, 2)

    def test_anthropic_streaming_continues_pause_turn_with_server_blocks(self) -> None:
        first = _FakeMessage(
            content=[
                _FakeServerBlock(
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {"query": "test"},
                    }
                ),
                _FakeServerBlock(
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "title": "Result",
                                "url": "https://example.com",
                                "encrypted_content": "opaque",
                            }
                        ],
                    }
                ),
            ],
            stop_reason="pause_turn",
        )
        second = _FakeMessage(content=[_FakeTextBlock("done")])

        class _PauseAnthropic(_FakeAnthropic):
            stream_messages = [first, second]

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}),
            patch.object(api_client_module.anthropic, "Anthropic", _PauseAnthropic),
            patch.object(api_client_module.request_logger, "log_request"),
            patch.object(api_client_module.request_logger, "log_response"),
        ):
            client = APIClient(
                model="claude-test",
                api="anthropic",
                max_tokens=16,
                timeout=1800,
                max_retries=1,
                max_retries_inner=0,
                max_wallclock_per_call_s=1800,
                stream_anthropic_messages=True,
                tools=[(None, {"type": "web_search_preview"})],
            )
            result = client._anthropic_query_with_tools(
                3,
                [{"role": "user", "content": "search"}],
            )

        messages = _PauseAnthropic.last_instance.messages
        self.assertEqual(messages.stream_calls, 2)
        self.assertEqual(result.conversation[-1]["content"], "done")
        replay = messages.payloads[1]["messages"][1]
        self.assertEqual(replay["role"], "assistant")
        self.assertEqual(replay["content"][0]["type"], "server_tool_use")
        self.assertEqual(replay["content"][1]["content"][0]["encrypted_content"], "opaque")

    def test_anthropic_streaming_continues_visible_max_tokens_response(self) -> None:
        first = _FakeMessage(
            content=[_FakeTextBlock("part one")],
            usage=_FakeUsage(input_tokens=20, output_tokens=16),
            stop_reason="max_tokens",
        )
        second = _FakeMessage(
            content=[_FakeTextBlock("part two")],
            usage=_FakeUsage(input_tokens=6, output_tokens=4),
        )

        class _ContinueAnthropic(_FakeAnthropic):
            stream_messages = [first, second]

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}),
            patch.object(api_client_module.anthropic, "Anthropic", _ContinueAnthropic),
            patch.object(api_client_module.request_logger, "log_request"),
            patch.object(api_client_module.request_logger, "log_response"),
        ):
            client = APIClient(
                model="claude-test",
                api="anthropic",
                max_tokens=16,
                timeout=1800,
                max_retries=1,
                max_retries_inner=0,
                max_wallclock_per_call_s=1800,
                stream_anthropic_messages=True,
                anthropic_continue_on_max_tokens=True,
            )
            result = client._anthropic_query_with_tools(
                3,
                [{"role": "user", "content": "write a long answer"}],
            )

        messages = _ContinueAnthropic.last_instance.messages
        self.assertEqual(messages.stream_calls, 2)
        self.assertEqual(result.conversation[-1]["content"], "part one\n\npart two")
        self.assertEqual(result.output_tokens, 20)
        continue_turn = messages.payloads[1]["messages"][-2]
        self.assertEqual(continue_turn["role"], "user")
        self.assertIn("Please continue", continue_turn["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
