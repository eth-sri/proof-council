from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from proofstack.agents.ac.ac_workflow import DEFAULT_COUNCIL_MODELS  # noqa: E402
from proofstack.registry import load_preset  # noqa: E402
from app.dev import _api_key_requirements_for_preset  # noqa: E402
from mathagents.config_loader import load_solver_config  # noqa: E402


class FableModelConfigTests(unittest.TestCase):
    def _assert_common_fable_fields(self, cfg: dict) -> None:
        self.assertEqual(cfg["model"], "claude-fable-5")
        self.assertEqual(cfg["api"], "anthropic")
        self.assertEqual(cfg["max_tokens"], 128000)
        # Fable 5 always thinks; adaptive is the only accepted explicit
        # config. budget_tokens or type: disabled are rejected with a 400.
        self.assertEqual(cfg["thinking"]["type"], "adaptive")
        self.assertNotIn("budget_tokens", cfg["thinking"])
        self.assertTrue(cfg["stream_anthropic_messages"])
        self.assertTrue(cfg["anthropic_salvage_empty_max_tokens"])
        self.assertEqual(cfg["read_cost"], 10)
        self.assertEqual(cfg["write_cost"], 50)
        self.assertEqual(cfg["cache_read_cost"], 1.0)
        self.assertEqual(cfg["cache_write_cost"], 12.5)

    def test_fable_council_seat_uses_xhigh_streaming_config(self) -> None:
        cfg = load_solver_config("models/anthropic/fable_5")

        self._assert_common_fable_fields(cfg)
        self.assertEqual(cfg["thinking"]["display"], "omitted")
        self.assertEqual(cfg["output_config"]["effort"], "xhigh")
        self.assertEqual(cfg["timeout"], 1800)
        self.assertEqual(cfg["max_wallclock_per_call_s"], 1800)
        self.assertEqual(cfg["max_retries"], 1)
        self.assertEqual(cfg["max_retries_inner"], 1)

    def test_fable_author_uses_max_effort_with_pro_parity_leash(self) -> None:
        cfg = load_solver_config("models/anthropic/fable_5_max")

        self._assert_common_fable_fields(cfg)
        # Reasoning summaries visible for the Author, matching the
        # gpt-55-pro Author's `reasoning: {summary: auto}`.
        self.assertEqual(cfg["thinking"]["display"], "summarized")
        self.assertEqual(cfg["output_config"]["effort"], "max")
        # Author-role leash at parity with gpt-55-pro.
        self.assertEqual(cfg["timeout"], 11400)
        self.assertEqual(cfg["max_wallclock_per_call_s"], 14000)
        self.assertEqual(cfg["max_retries"], 2)
        self.assertEqual(cfg["max_retries_inner"], 1)


class FablePresetTests(unittest.TestCase):
    def test_fable_author_preset_swaps_author_only(self) -> None:
        preset = load_preset("author_critic_fable_author")

        self.assertEqual(
            preset.component_configs["Author"]["model"],
            "models/anthropic/fable_5_max",
        )
        self.assertEqual(
            preset.inputs["council_models"],
            list(DEFAULT_COUNCIL_MODELS),
        )

    def test_fable_council_preset_swaps_anthropic_seat_only(self) -> None:
        preset = load_preset("author_critic_fable_council")

        self.assertEqual(
            preset.component_configs["Author"]["model"],
            "models/openai/gpt-55-pro",
        )
        self.assertEqual(
            preset.inputs["council_models"],
            [
                "models/openai/gpt-55-pro",
                "models/anthropic/fable_5",
                "models/gemini/gemini-31-pro",
            ],
        )
        self.assertNotIn(
            "models/anthropic/opus_47_max",
            preset.inputs["council_models"],
        )

    def test_fable_presets_report_all_provider_keys(self) -> None:
        for name in (
            "author_critic_fable_author",
            "author_critic_fable_council",
        ):
            requirements = _api_key_requirements_for_preset(
                name,
                env={"__TEST_EMPTY_ENV__": "1"},
            )

            self.assertEqual(
                {item["env"] for item in requirements},
                {"ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"},
                msg=name,
            )


if __name__ == "__main__":
    unittest.main()
