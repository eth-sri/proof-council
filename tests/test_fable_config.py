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
    def _assert_common_fable_fields(self, cfg: dict, *, model: str, cache_read_cost: float) -> None:
        self.assertEqual(cfg["model"], model)
        self.assertEqual(cfg["api"], "anthropic")
        self.assertEqual(cfg["max_tokens"], 128000)
        # Fable always thinks; adaptive is the only accepted explicit
        # config. budget_tokens or type: disabled are rejected with a 400.
        self.assertEqual(cfg["thinking"]["type"], "adaptive")
        self.assertNotIn("budget_tokens", cfg["thinking"])
        self.assertTrue(cfg["stream_anthropic_messages"])
        self.assertTrue(cfg["anthropic_salvage_empty_max_tokens"])
        self.assertTrue(cfg["anthropic_continue_on_max_tokens"])
        self.assertEqual(cfg["read_cost"], 10)
        self.assertEqual(cfg["write_cost"], 50)
        self.assertEqual(cfg["cache_read_cost"], cache_read_cost)
        self.assertEqual(cfg["cache_write_cost"], 12.5)

    def _assert_council_seat_leash(self, cfg: dict, *, cap_s: int = 1800) -> None:
        self.assertEqual(cfg["thinking"]["display"], "omitted")
        self.assertEqual(cfg["output_config"]["effort"], "max")
        self.assertEqual(cfg["timeout"], cap_s)
        self.assertEqual(cfg["max_wallclock_per_call_s"], cap_s)
        self.assertEqual(cfg["max_retries"], 1)
        self.assertEqual(cfg["max_retries_inner"], 1)

    def _assert_author_leash(self, cfg: dict) -> None:
        # Reasoning summaries visible for the Author, matching the
        # gpt-6-astra-pro Author's `reasoning: {summary: auto}`.
        self.assertEqual(cfg["thinking"]["display"], "summarized")
        self.assertEqual(cfg["output_config"]["effort"], "max")
        # Author-role leash at parity with gpt-6-astra-pro.
        self.assertEqual(cfg["timeout"], 11400)
        self.assertEqual(cfg["max_wallclock_per_call_s"], 14000)
        self.assertEqual(cfg["max_retries"], 2)
        self.assertEqual(cfg["max_retries_inner"], 1)
        self.assertTrue(cfg["throw_error_on_failure"])

    def test_fable_council_seat_uses_max_streaming_config(self) -> None:
        cfg = load_solver_config("models/anthropic/fable_5")

        self._assert_common_fable_fields(cfg, model="claude-fable-5", cache_read_cost=1.0)
        self._assert_council_seat_leash(cfg)

    def test_fable_author_uses_max_effort_with_pro_parity_leash(self) -> None:
        cfg = load_solver_config("models/anthropic/fable_5_max")

        self._assert_common_fable_fields(cfg, model="claude-fable-5", cache_read_cost=1.0)
        self._assert_author_leash(cfg)

    def test_fable_51_council_seat_has_45_min_cap_and_cheaper_cache_reads(self) -> None:
        cfg = load_solver_config("models/anthropic/fable_51")

        self._assert_common_fable_fields(cfg, model="claude-fable-5-1", cache_read_cost=0.25)
        # Leave time for long thinking streams and mid-stream recovery.
        self._assert_council_seat_leash(cfg, cap_s=2700)

    def test_fable_51_author_matches_fable_5_author_leash(self) -> None:
        cfg = load_solver_config("models/anthropic/fable_51_max")

        self._assert_common_fable_fields(cfg, model="claude-fable-5-1", cache_read_cost=0.25)
        self._assert_author_leash(cfg)


class FablePresetTests(unittest.TestCase):
    def test_fable_author_preset_swaps_author_only(self) -> None:
        preset = load_preset("author_critic_fable_author")

        self.assertEqual(
            preset.component_configs["Author"]["model"],
            "models/anthropic/fable_51_max",
        )
        self.assertNotIn("USE_CONTAINER_FILES", preset.component_configs["Author"])
        self.assertEqual(
            preset.inputs["council_models"],
            list(DEFAULT_COUNCIL_MODELS),
        )

    def test_fable_council_preset_matches_runtime_council(self) -> None:
        preset = load_preset("author_critic_fable_council")

        self.assertEqual(
            preset.component_configs["Author"]["model"],
            "models/openai/gpt-6-astra-pro",
        )
        self.assertEqual(
            preset.inputs["council_models"],
            [
                "models/openai/gpt-6-astra-pro",
                "models/anthropic/fable_51",
                "models/gemini/gemini-31-pro",
            ],
        )
        self.assertEqual(preset.inputs["council_models"], list(DEFAULT_COUNCIL_MODELS))

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
