"""P2 regression test — uploaded files cleaned up on Author failure.

Before the fix in ``proofstack.agents.ac.author``, ``bridge.cleanup()``
was only reached after the API call, cost accounting, response parsing,
and container download all succeeded. If ``_one_shot_query()`` (or any
later step) raised, the uploaded OpenAI ``user_data`` files leaked.

This test mocks ``_one_shot_query`` to raise mid-call and verifies that
``files.delete`` was still invoked for every uploaded file.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# --- stub external deps so project modules import cleanly ----------------

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
    openai.OpenAI = MagicMock()
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
    loguru.logger = MagicMock()
    sys.modules["loguru"] = loguru

from proofstack.agents.ac.author import Author  # noqa: E402
from proofstack.agents.ac.blocks import CANONICAL_FILES  # noqa: E402
from proofstack.context import RunContext  # noqa: E402


class AuthorContainerCleanupTests(unittest.TestCase):
    """Verify ``bridge.cleanup()`` always fires, even on mid-call errors."""

    def test_cleanup_runs_when_api_call_raises(self) -> None:
        """``_one_shot_query`` raising must not skip the cleanup ``finally``."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = RunContext.create(
                run_id="test_author_cleanup",
                root_workdir=Path(temp_dir),
                flat=True,
            )
            author = Author(ctx)

            uploaded_ids = [f"file-{name}_id" for name in CANONICAL_FILES]
            mock_oai_client = MagicMock()
            mock_oai_client.files.create.side_effect = [
                SimpleNamespace(id=fid) for fid in uploaded_ids
            ]

            os.environ["OPENAI_API_KEY"] = "test-key-not-used"

            inp = Author.Inputs(
                problem="Prove X.",
                round=0,
                n_rounds=1,
                budget_used_usd=0.0,
                budget_max_usd=10.0,
                answer_tex="\\documentclass{article}\\begin{document}x\\end{document}",
                research_notes_tex="notes",
                references_bib="@article{x}",
                prev_critique="",
                prev_council="",
            )

            simulated_error = RuntimeError("simulated API failure")
            with patch.object(
                sys.modules["openai"], "OpenAI",
                MagicMock(return_value=mock_oai_client),
            ), patch.object(
                author, "_build_api_client_with_file_ids",
                return_value=MagicMock(model="gpt-test"),
            ), patch(
                "proofstack.agents.ac.author._one_shot_query",
                side_effect=simulated_error,
            ):
                with self.assertRaises(RuntimeError) as cm:
                    asyncio.run(author._run_with_container_files(inp))
                self.assertEqual(str(cm.exception), "simulated API failure")

            # The whole point of the fix: cleanup must have fired.
            self.assertEqual(
                mock_oai_client.files.delete.call_count,
                len(CANONICAL_FILES),
                "bridge.cleanup() did not delete uploaded files on Author "
                "failure — the P2 leak fix is regressed.",
            )
            deleted_ids = {
                call.args[0]
                for call in mock_oai_client.files.delete.call_args_list
            }
            self.assertEqual(deleted_ids, set(uploaded_ids))


if __name__ == "__main__":
    unittest.main()
