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
import zipfile
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

from proofstack.agents.ac.author import Author, _is_attachment_rejection  # noqa: E402
from proofstack.agents.ac.blocks import CANONICAL_FILES  # noqa: E402
from proofstack.agents.ac.container_files import (  # noqa: E402
    ContainerFileBridge,
    _classify_attachment_upload_error,
)
from proofstack.agents.ac.compute import (  # noqa: E402
    COMPUTE_HANDOFF_MAX_FILES,
    Compute,
    inspect_compute_handoff,
    render_compute_reply_for_author,
)
from proofstack.budget import BudgetExhausted, BudgetSpec  # noqa: E402
from proofstack.context import RunContext  # noqa: E402
from mathagents.api_client import ProviderAttachmentRejectedError  # noqa: E402


class AuthorContainerCleanupTests(unittest.TestCase):
    """Verify ``bridge.cleanup()`` always fires, even on mid-call errors."""

    def test_attachment_retry_requires_typed_or_structured_rejection(self) -> None:
        typed = ProviderAttachmentRejectedError("file rejected", cost={})
        structured = RuntimeError("provider request failed")
        structured.body = {"error": {"code": "array_above_max_length"}}
        too_large = RuntimeError("provider request failed")
        too_large.status_code = 413

        self.assertTrue(_is_attachment_rejection(typed))
        self.assertTrue(_is_attachment_rejection(structured))
        self.assertTrue(_is_attachment_rejection(too_large))
        self.assertFalse(
            _is_attachment_rejection(RuntimeError("attachment service unavailable"))
        )
        self.assertFalse(_is_attachment_rejection(RuntimeError("invalid file id")))

    def test_generic_upload_validation_error_is_not_permanently_quarantined(
        self,
    ) -> None:
        error = RuntimeError("provider rejected malformed request metadata")
        error.status_code = 400

        self.assertEqual(_classify_attachment_upload_error(error), "unknown")

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

    def test_oversized_compute_archive_is_not_uploaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            oversized = temp / "old_compute_workspace.zip"
            with zipfile.ZipFile(oversized, "w") as zf:
                for idx in range(COMPUTE_HANDOFF_MAX_FILES + 1):
                    zf.writestr(f"code/generated_{idx:04d}.py", "")

            ctx = RunContext.create(
                run_id="test_compute_archive_preflight",
                root_workdir=temp / "run",
                flat=True,
            )
            author = Author(ctx)
            inp = Author.Inputs(
                problem="Prove X.",
                round=1,
                n_rounds=2,
                compute_zip_path=oversized,
            )

            self.assertEqual(author._extra_attachments(inp), [])
            rendered = render_compute_reply_for_author(
                Compute.Outputs(zip_path=oversized, response_md="result")
            )
            self.assertIn("workspace zip: omitted", rendered)

    def test_optional_compute_upload_failure_keeps_canonical_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in CANONICAL_FILES:
                (root / name).write_text(name, encoding="utf-8")
            compute_zip = root / "compute.zip"
            with zipfile.ZipFile(compute_zip, "w") as zf:
                zf.writestr("notes/result.txt", "result")

            client = MagicMock()
            client.files.create.side_effect = [
                *[
                    SimpleNamespace(id=f"file-{idx}")
                    for idx in range(len(CANONICAL_FILES))
                ],
                RuntimeError("provider rejected optional attachment"),
            ]
            bridge = ContainerFileBridge(
                openai_client=client,
                workspace=root,
                names=CANONICAL_FILES,
                extra_attachments=[(compute_zip, "compute artifact")],
            )

            uploaded_ids = bridge.upload()

            self.assertEqual(len(uploaded_ids), len(CANONICAL_FILES))
            self.assertEqual(len(bridge.uploaded), len(CANONICAL_FILES))
            self.assertEqual(len(bridge.extra_upload_failures), 1)
            self.assertEqual(bridge.extra_upload_failures[0].name, "compute.zip")
            self.assertIn(
                "provider rejected optional attachment",
                bridge.extra_upload_failures[0].message,
            )
            self.assertEqual(
                bridge.extra_upload_failures[0].disposition,
                "unknown",
            )

    def test_transient_optional_upload_failure_is_not_quarantined(self) -> None:
        class TransientUploadError(RuntimeError):
            status_code = 503

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in CANONICAL_FILES:
                (root / name).write_text(name, encoding="utf-8")
            compute_zip = root / "compute.zip"
            with zipfile.ZipFile(compute_zip, "w") as zf:
                zf.writestr("notes/result.txt", "result")

            client = MagicMock()
            client.files.create.side_effect = [
                *[
                    SimpleNamespace(id=f"file-{idx}")
                    for idx in range(len(CANONICAL_FILES))
                ],
                TransientUploadError("service temporarily unavailable"),
            ]
            bridge = ContainerFileBridge(
                openai_client=client,
                workspace=root,
                names=CANONICAL_FILES,
                extra_attachments=[(compute_zip, "compute artifact")],
            )
            bridge.upload()
            failure = bridge.extra_upload_failures[0]
            self.assertEqual(failure.disposition, "transient")

            ctx = RunContext.create(
                run_id="test_transient_attachment",
                root_workdir=root / "run",
                flat=True,
            )
            author = Author(ctx)
            events: list[tuple[str, dict]] = []

            async def emit(kind, payload, **kwargs):
                events.append((kind, payload))

            author.events = SimpleNamespace(emit=emit)
            asyncio.run(
                author._emit_attachment_upload_failures(
                    [failure],
                    provider="openai",
                )
            )

            self.assertTrue(inspect_compute_handoff(compute_zip).attachable)
            self.assertEqual(events[0][0], "ac.author.attachment_unavailable")
            self.assertTrue(events[0][1]["retry_next_round"])

    def test_author_retries_provider_attachment_rejection_without_compute_zip(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            compute_zip = temp / "compute.zip"
            with zipfile.ZipFile(compute_zip, "w") as zf:
                zf.writestr("notes/result.txt", "result")
            ctx = RunContext.create(
                run_id="test_attachment_retry",
                root_workdir=temp / "run",
                flat=True,
            )
            author = Author(ctx)
            client = MagicMock()
            client.files.create.side_effect = [
                SimpleNamespace(id=f"file-{idx}")
                for idx in range(len(CANONICAL_FILES) + 1)
            ]
            configured_ids: list[list[str]] = []

            def build_client(file_ids):
                configured_ids.append(list(file_ids))
                return MagicMock(model="gpt-test")

            inp = Author.Inputs(
                problem="Prove X.",
                round=1,
                n_rounds=2,
                answer_tex="old answer",
                research_notes_tex="old notes",
                references_bib="old refs",
                compute_zip_path=compute_zip,
            )
            query_result = (
                0,
                [{"role": "assistant", "content": "<ready>false</ready>"}],
                {"cost": 0.0, "input_tokens": 1, "output_tokens": 1},
            )
            rejected = ProviderAttachmentRejectedError(
                "container file is too large",
                cost={
                    "cost": 1.25,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "reasoning_tokens": 10,
                },
            )

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch.object(
                sys.modules["openai"],
                "OpenAI",
                MagicMock(return_value=client),
            ), patch.object(
                author,
                "_build_api_client_with_file_ids",
                side_effect=build_client,
            ), patch(
                "proofstack.agents.ac.author._one_shot_query",
                side_effect=[
                    rejected,
                    query_result,
                ],
            ):
                out = asyncio.run(author._run_with_container_files(inp))

            self.assertEqual(out.answer_tex, "old answer")
            self.assertEqual(
                [len(file_ids) for file_ids in configured_ids],
                [len(CANONICAL_FILES) + 1, len(CANONICAL_FILES)],
            )
            self.assertEqual(client.files.delete.call_count, len(CANONICAL_FILES) + 1)
            self.assertFalse(inspect_compute_handoff(compute_zip).attachable)
            self.assertEqual(author._extra_attachments(inp), [])
            self.assertEqual(author.tracker.counters.usd, 1.25)
            self.assertEqual(author.tracker.counters.tokens, 122)

    def test_attachment_is_quarantined_before_failed_call_exhausts_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            compute_zip = temp / "compute.zip"
            with zipfile.ZipFile(compute_zip, "w") as zf:
                zf.writestr("notes/result.txt", "result")
            ctx = RunContext.create(
                run_id="test_attachment_budget",
                root_workdir=temp / "run",
                run_budget=BudgetSpec(max_usd=1.0),
                flat=True,
            )
            author = Author(ctx)
            client = MagicMock()
            client.files.create.side_effect = [
                SimpleNamespace(id=f"file-{idx}")
                for idx in range(len(CANONICAL_FILES) + 1)
            ]
            api_client = MagicMock(model="gpt-test")
            rejected = ProviderAttachmentRejectedError(
                "container file is too large",
                cost={
                    "cost": 1.25,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "reasoning_tokens": 10,
                },
            )
            inp = Author.Inputs(
                problem="Prove X.",
                round=1,
                n_rounds=2,
                compute_zip_path=compute_zip,
            )

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch.object(
                sys.modules["openai"],
                "OpenAI",
                MagicMock(return_value=client),
            ), patch.object(
                author,
                "_build_api_client_with_file_ids",
                return_value=api_client,
            ), patch(
                "proofstack.agents.ac.author._one_shot_query",
                side_effect=rejected,
            ) as query:
                with self.assertRaises(BudgetExhausted):
                    asyncio.run(author._run_with_container_files(inp))

            self.assertEqual(query.call_count, 1)
            self.assertFalse(inspect_compute_handoff(compute_zip).attachable)
            self.assertEqual(author._extra_attachments(inp), [])
            self.assertEqual(client.files.delete.call_count, len(CANONICAL_FILES) + 1)
            self.assertEqual(author.tracker.counters.usd, 1.25)


if __name__ == "__main__":
    unittest.main()
