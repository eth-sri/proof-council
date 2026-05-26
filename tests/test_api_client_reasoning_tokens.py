"""Lock the behaviour of ``APIClient._extract_reasoning_tokens``.

The First Proof spec requires logging input + output + reasoning tokens
per API call. ``_extract_reasoning_tokens`` is the provider-agnostic
extractor; if a provider tweak silently changes its return shape, the
adapter starts billing reasoning as 0 again. These tests pin the
mapping for every nested-field convention we currently support.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# Reuse the same SDK shims as test_api_client_responses_tools.py so the
# module imports without real provider SDKs installed in the test env.
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


def _client() -> APIClient:
    """Build an APIClient bypassing __init__ so we can call instance
    methods without provider env vars."""
    instance = APIClient.__new__(APIClient)
    return instance


class ExtractReasoningTokensTests(unittest.TestCase):
    def test_none_usage_returns_zero(self) -> None:
        self.assertEqual(_client()._extract_reasoning_tokens(None), 0)

    def test_empty_dict_returns_zero(self) -> None:
        self.assertEqual(_client()._extract_reasoning_tokens({}), 0)

    def test_openai_responses_output_tokens_details(self) -> None:
        usage = {
            "input_tokens": 100,
            "output_tokens": 250,
            "output_tokens_details": {"reasoning_tokens": 175},
        }
        self.assertEqual(_client()._extract_reasoning_tokens(usage), 175)

    def test_openai_chat_completions_completion_tokens_details(self) -> None:
        usage = {
            "prompt_tokens": 80,
            "completion_tokens": 200,
            "completion_tokens_details": {"reasoning_tokens": 132},
        }
        self.assertEqual(_client()._extract_reasoning_tokens(usage), 132)

    def test_gemini_thoughts_token_count(self) -> None:
        usage = {
            "promptTokenCount": 50,
            "candidatesTokenCount": 90,
            "thoughtsTokenCount": 415,
        }
        self.assertEqual(_client()._extract_reasoning_tokens(usage), 415)

    def test_gemini_thinking_tokens_alias(self) -> None:
        usage = {"input_tokens": 1, "output_tokens": 2, "thinking_tokens": 33}
        self.assertEqual(_client()._extract_reasoning_tokens(usage), 33)

    def test_codex_reasoning_out_tokens_alias(self) -> None:
        # The Compute Worker parses codex JSONL into reasoning_out_tokens.
        usage = {"input_tokens": 1, "output_tokens": 2, "reasoning_out_tokens": 88}
        self.assertEqual(_client()._extract_reasoning_tokens(usage), 88)

    def test_top_level_reasoning_tokens(self) -> None:
        usage = {"input_tokens": 1, "output_tokens": 2, "reasoning_tokens": 7}
        self.assertEqual(_client()._extract_reasoning_tokens(usage), 7)

    def test_object_with_output_tokens_details_attr(self) -> None:
        details = SimpleNamespace(reasoning_tokens=42)
        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            output_tokens_details=details,
            model_dump=lambda: {
                "input_tokens": 10,
                "output_tokens": 20,
                "output_tokens_details": {"reasoning_tokens": 42},
            },
        )
        self.assertEqual(_client()._extract_reasoning_tokens(usage), 42)

    def test_object_without_model_dump_preserves_nested_details(self) -> None:
        """Regression: some third-party OpenAI / Gemini wrappers return
        a duck-typed usage object that has neither ``model_dump`` nor
        nested ``__dict__`` exposure but does expose
        ``output_tokens_details`` as an attribute. The ``_usage_to_dict``
        fallback must preserve that nested object so
        ``_extract_reasoning_tokens`` finds the reasoning count.
        """
        details = SimpleNamespace(reasoning_tokens=99)
        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            output_tokens_details=details,
        )
        # Note: no model_dump method — exercises the fallback branch.
        self.assertFalse(hasattr(usage, "model_dump"))
        self.assertEqual(_client()._extract_reasoning_tokens(usage), 99)

    def test_object_without_model_dump_picks_up_completion_details(self) -> None:
        """Same regression but for OpenAI Chat-Completions' field
        (``completion_tokens_details``)."""
        details = SimpleNamespace(reasoning_tokens=58)
        usage = SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=33,
            completion_tokens_details=details,
        )
        self.assertFalse(hasattr(usage, "model_dump"))
        self.assertEqual(_client()._extract_reasoning_tokens(usage), 58)

    def test_object_without_model_dump_picks_up_camelcase_thoughts(self) -> None:
        """Same regression for Gemini's native ``thoughtsTokenCount``."""
        usage = SimpleNamespace(
            promptTokenCount=11,
            candidatesTokenCount=22,
            thoughtsTokenCount=300,
        )
        self.assertFalse(hasattr(usage, "model_dump"))
        self.assertEqual(_client()._extract_reasoning_tokens(usage), 300)

    def test_anthropic_without_thinking_returns_zero(self) -> None:
        # Anthropic does not separately report thinking tokens.
        usage = {"input_tokens": 50, "output_tokens": 200, "cache_read_input_tokens": 10}
        self.assertEqual(_client()._extract_reasoning_tokens(usage), 0)

    def test_negative_or_invalid_clamped_to_zero(self) -> None:
        self.assertEqual(
            _client()._extract_reasoning_tokens({"reasoning_tokens": -5}),
            0,
        )
        self.assertEqual(
            _client()._extract_reasoning_tokens({"output_tokens_details": {"reasoning_tokens": "not-a-number"}}),
            0,
        )

    def test_responses_details_wins_over_top_level_fallback(self) -> None:
        usage = {
            "output_tokens_details": {"reasoning_tokens": 100},
            "reasoning_tokens": 999,  # would-be fallback; should NOT win
        }
        self.assertEqual(_client()._extract_reasoning_tokens(usage), 100)


class InternalRequestResultPropagationTests(unittest.TestCase):
    def test_reasoning_tokens_default_zero(self) -> None:
        result = APIClient.InternalRequestResult(
            conversation=[],
            input_tokens=10,
            output_tokens=20,
        )
        self.assertEqual(result.reasoning_tokens, 0)

    def test_reasoning_tokens_passes_through_constructor(self) -> None:
        result = APIClient.InternalRequestResult(
            conversation=[],
            input_tokens=10,
            output_tokens=20,
            reasoning_tokens=512,
        )
        self.assertEqual(result.reasoning_tokens, 512)


if __name__ == "__main__":
    unittest.main()
