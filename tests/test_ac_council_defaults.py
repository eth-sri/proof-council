from __future__ import annotations

import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from proofstack.agents.ac.ac_workflow import ACWorkflow, DEFAULT_COUNCIL_MODELS  # noqa: E402
from proofstack.registry import load_preset  # noqa: E402
from app.dev import _api_key_requirements_for_preset  # noqa: E402
from mathagents.config_loader import load_solver_config  # noqa: E402
from proofstack.agents.ac.council import _strip_visible_thought_blocks  # noqa: E402


class ACCouncilDefaultsTests(unittest.TestCase):
    def test_runtime_default_council_includes_gemini(self) -> None:
        self.assertEqual(
            DEFAULT_COUNCIL_MODELS,
            (
                "models/openai/gpt-55-pro",
                "models/anthropic/opus_47_max",
                "models/gemini/gemini-31-pro",
            ),
        )
        self.assertEqual(
            ACWorkflow.Inputs(problem="P", problem_id="p").council_models,
            list(DEFAULT_COUNCIL_MODELS),
        )

    def test_normal_author_critic_preset_uses_runtime_council(self) -> None:
        preset = load_preset("author_critic")

        self.assertEqual(
            preset.inputs["council_models"],
            list(DEFAULT_COUNCIL_MODELS),
        )

    def test_opus_council_member_uses_max_adaptive_streaming_config(self) -> None:
        cfg = load_solver_config("models/anthropic/opus_47_max")

        self.assertEqual(cfg["model"], "claude-opus-4-7")
        self.assertEqual(cfg["max_tokens"], 128000)
        self.assertEqual(cfg["output_config"]["effort"], "max")
        self.assertNotEqual(cfg["output_config"]["effort"], "xhigh")
        self.assertEqual(cfg["thinking"]["type"], "adaptive")
        self.assertEqual(cfg["thinking"]["display"], "omitted")
        self.assertTrue(cfg["stream_anthropic_messages"])
        self.assertTrue(cfg["anthropic_salvage_empty_max_tokens"])
        self.assertEqual(cfg["timeout"], 1800)
        self.assertEqual(cfg["max_wallclock_per_call_s"], 1800)
        self.assertEqual(cfg["max_retries"], 1)
        self.assertEqual(cfg["max_retries_inner"], 1)

    def test_normal_author_critic_preset_reports_all_provider_keys(self) -> None:
        requirements = _api_key_requirements_for_preset(
            "author_critic",
            env={"__TEST_EMPTY_ENV__": "1"},
        )

        self.assertEqual(
            {item["env"] for item in requirements},
            {"ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"},
        )

    def test_workflow_fans_default_models_out_to_council(self) -> None:
        from proofstack.context import RunContext

        async def run_check() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                ctx = RunContext.create(
                    run_id="test",
                    root_workdir=temp_dir,
                    flat=True,
                )
                workflow = ACWorkflow(ctx)
                seen: list[list[str]] = []

                async def fake_critic_run(self, inp):
                    return self.Outputs(
                        review_md="ok",
                        answer_ready=False,
                        mode=inp.mode,
                    )

                async def fake_council_run(self, inp):
                    seen.append(list(inp.member_models))
                    return self.Outputs(replies=[])

                workflow.critic.run = types.MethodType(
                    fake_critic_run,
                    workflow.critic,
                )
                workflow.council.run = types.MethodType(
                    fake_council_run,
                    workflow.council,
                )
                author_out = type(
                    "AuthorOut",
                    (),
                    {
                        "council_to": [],
                        "council_question": "q",
                        "answer_tex": "A",
                        "research_notes_tex": "N",
                        "references_bib": "B",
                        "thinking_summary": "",
                    },
                )()

                await workflow._gather_critic_council(
                    inp=ACWorkflow.Inputs(problem="P", problem_id="p"),
                    workspace=Path(temp_dir),
                    author_k=author_out,
                    critic_conversation=[],
                    mode="fresh",
                    round=1,
                    run_council=True,
                )

                self.assertEqual(seen, [list(DEFAULT_COUNCIL_MODELS)])

        asyncio.run(run_check())

    def test_council_strips_visible_provider_thought_blocks(self) -> None:
        self.assertEqual(
            _strip_visible_thought_blocks(
                "<thought>internal notes</thought>\nUseful council reply."
            ),
            "Useful council reply.",
        )
        self.assertEqual(
            _strip_visible_thought_blocks("Useful council reply."),
            "Useful council reply.",
        )


if __name__ == "__main__":
    unittest.main()
