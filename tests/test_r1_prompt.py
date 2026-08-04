from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.agents.configurable_prompt import ConfigurablePromptAgent  # noqa: E402
from proofstack.context import RunContext  # noqa: E402


def _rendered_text(messages) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages)


class FormatInstructionTemplateScopeTests(unittest.TestCase):
    """FMT-A8: the auto format-instruction skip must inspect the authored
    prompt template, not rendered user input."""

    def test_input_quoting_output_tag_still_gets_format_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=Path(tmp),
                flat=True,
                component_configs={
                    "judge": {
                        "messages": [{"role": "user", "content": "Judge: {candidate}"}],
                        "output": {
                            "xml_tags": ["verdict", "feedback"],
                            "default_field": "feedback",
                        },
                    }
                },
            )
            agent = ConfigurablePromptAgent(ctx, name="judge")

            # User input echoes a prior output tag; this must NOT suppress the
            # verdict instruction (input is not the authored template).
            inp = agent.Inputs(candidate="They wrote <verdict>incorrect</verdict> earlier.")
            messages = agent.render_messages(inp)
            text = _rendered_text(messages)

            self.assertIn("<verdict>incorrect</verdict>", text)  # input preserved
            self.assertIn("FORMAT YOUR ANSWER:", text)
            self.assertIn("Wrap that output in <verdict>...</verdict>.", text)
            self.assertIn("Wrap that output in <feedback>...</feedback>.", text)

            # And the field still parses out of a model reply.
            out = agent.parse_output(
                "<verdict>correct</verdict><feedback>ok</feedback>", inp
            )
            self.assertEqual(out.verdict, "correct")
            self.assertEqual(out.feedback, "ok")

    def test_authored_template_mentioning_tag_suppresses_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext.create(
                run_id="test",
                root_workdir=Path(tmp),
                flat=True,
                component_configs={
                    "judge": {
                        "messages": [
                            {
                                "role": "user",
                                "content": "Judge: {candidate}. Put the answer in a <verdict> tag.",
                            }
                        ],
                        "output": {
                            "xml_tags": ["verdict"],
                            "default_field": "feedback",
                        },
                    }
                },
            )
            agent = ConfigurablePromptAgent(ctx, name="judge")

            inp = agent.Inputs(candidate="the sum is 5")
            messages = agent.render_messages(inp)
            text = _rendered_text(messages)

            # Hand-written format prompt keeps working: no auto instruction added.
            self.assertNotIn("FORMAT YOUR ANSWER:", text)
            self.assertNotIn("Wrap that output in <verdict>...</verdict>.", text)


if __name__ == "__main__":
    unittest.main()
