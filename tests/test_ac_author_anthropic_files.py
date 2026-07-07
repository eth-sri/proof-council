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
sys.path.insert(0, str(ROOT))

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
from proofstack.agents.ac.container_files import (  # noqa: E402
    ANTHROPIC_FILES_BETA,
    AnthropicContainerFileBridge,
)
from proofstack.context import RunContext  # noqa: E402


class _FakeDownloadResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body


class _FakeAnthropicFiles:
    def __init__(self):
        self.upload_calls: list[tuple[str, list[str]]] = []
        self.retrieve_calls: list[tuple[str, list[str]]] = []
        self.download_calls: list[tuple[str, list[str]]] = []
        self.delete_calls: list[tuple[str, list[str]]] = []
        self.generated: dict[str, tuple[str, str]] = {}
        self.fail_upload_after: int | None = None
        self._next_upload = 0

    def upload(self, *, file, betas):
        self._next_upload += 1
        name = Path(getattr(file, "name", f"upload_{self._next_upload}")).name
        file.read()
        if self.fail_upload_after is not None and self._next_upload > self.fail_upload_after:
            raise RuntimeError("simulated upload failure")
        file_id = f"file_input_{self._next_upload}"
        self.upload_calls.append((name, list(betas)))
        return SimpleNamespace(id=file_id, filename=name)

    def retrieve_metadata(self, file_id, *, betas):
        self.retrieve_calls.append((file_id, list(betas)))
        filename, _body = self.generated[file_id]
        return SimpleNamespace(id=file_id, filename=filename)

    def download(self, file_id, *, betas):
        self.download_calls.append((file_id, list(betas)))
        _filename, body = self.generated[file_id]
        return _FakeDownloadResponse(body)

    def delete(self, file_id, *, betas):
        self.delete_calls.append((file_id, list(betas)))
        return SimpleNamespace(id=file_id, deleted=True)


class _FakeAnthropicClient:
    def __init__(self):
        self.files = _FakeAnthropicFiles()
        self.beta = SimpleNamespace(files=self.files)


class AnthropicContainerFileBridgeTests(unittest.TestCase):
    def test_upload_blocks_download_generated_canonical_files_and_cleanup(self) -> None:
        fake_client = _FakeAnthropicClient()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in CANONICAL_FILES:
                (root / name).write_text(f"old {name}", encoding="utf-8")
            compute_zip = root / "compute.zip"
            compute_zip.write_bytes(b"zip")

            bridge = AnthropicContainerFileBridge(
                anthropic_client=fake_client,
                workspace=root,
                names=CANONICAL_FILES,
                extra_attachments=[(compute_zip, "compute artifact")],
            )
            uploaded_ids = bridge.upload()

            self.assertEqual(len(uploaded_ids), 4)
            self.assertEqual(
                [name for name, _betas in fake_client.files.upload_calls],
                ["answer.tex", "research_notes.tex", "references.bib", "compute.zip"],
            )
            self.assertTrue(
                all(betas == [ANTHROPIC_FILES_BETA] for _name, betas in fake_client.files.upload_calls)
            )
            self.assertEqual(
                bridge.render_container_upload_blocks(),
                [{"type": "container_upload", "file_id": file_id} for file_id in uploaded_ids],
            )
            self.assertIn("generated file named exactly `answer.tex`", bridge.render_workspace_listing())

            fake_client.files.generated = {
                "file_generated_answer": ("answer.tex", "new answer"),
                "file_generated_notes": ("/tmp/research_notes.tex", "new notes"),
                "file_generated_ignore": ("scratch.txt", "ignore me"),
            }
            conversation = [
                {"role": "user", "type": "container_upload", "file_id": uploaded_ids[0]},
                {
                    "role": "assistant",
                    "type": "bash_code_execution_tool_result",
                    "content": [
                        {"type": "file", "file_id": "file_generated_answer"},
                        {"type": "file", "file_id": "file_generated_notes"},
                        {"type": "file", "file_id": "file_generated_ignore"},
                        {"type": "file", "file_id": uploaded_ids[1]},
                    ],
                },
            ]

            downloaded = bridge.download(conversation)
            bridge.cleanup()

        self.assertEqual(
            downloaded,
            {
                "answer.tex": "new answer",
                "research_notes.tex": "new notes",
            },
        )
        self.assertEqual(
            [file_id for file_id, _betas in fake_client.files.download_calls],
            ["file_generated_answer", "file_generated_notes"],
        )
        self.assertEqual(
            {file_id for file_id, _betas in fake_client.files.delete_calls},
            {
                *uploaded_ids,
                "file_generated_answer",
                "file_generated_notes",
            },
        )
        self.assertTrue(
            all(betas == [ANTHROPIC_FILES_BETA] for _file_id, betas in fake_client.files.delete_calls)
        )


class AuthorAnthropicFilesTests(unittest.TestCase):
    def _input(self) -> Author.Inputs:
        return Author.Inputs(
            problem="Prove X.",
            round=0,
            n_rounds=1,
            budget_used_usd=0.0,
            budget_max_usd=10.0,
            answer_tex="old answer",
            research_notes_tex="old notes",
            references_bib="old refs",
            prev_critique="",
            prev_council="",
        )

    def test_fable_author_selects_anthropic_bridge_and_updates_generated_answer(self) -> None:
        fake_client = _FakeAnthropicClient()
        fake_client.files.generated = {
            "file_generated_answer": ("answer.tex", "new answer")
        }

        built_configs: list[dict] = []

        def factory(cfg):
            built_configs.append(cfg)
            return SimpleNamespace(model=cfg["model"])

        conversation = [
            {
                "role": "assistant",
                "type": "bash_code_execution_tool_result",
                "content": [{"type": "file", "file_id": "file_generated_answer"}],
            },
            {"role": "assistant", "content": "<ready>true</ready>"},
        ]

        with tempfile.TemporaryDirectory() as td:
            ctx = RunContext.create(
                run_id="test_author_anthropic_files",
                root_workdir=Path(td),
                flat=True,
                api_client_factory=factory,
                component_configs={
                    "Author": {"model": "models/anthropic/fable_5_max"}
                },
            )
            author = Author(ctx)

            anthropic_ctor = MagicMock(return_value=fake_client)
            with (
                patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-not-used"}),
                patch.object(
                    sys.modules["anthropic"],
                    "Anthropic",
                    anthropic_ctor,
                ),
                patch(
                    "proofstack.agents.ac.author._one_shot_query",
                    return_value=(
                        0,
                        conversation,
                        {
                            "cost": 0.0,
                            "input_tokens": 1,
                            "output_tokens": 2,
                            "reasoning_tokens": 3,
                        },
                    ),
                ),
            ):
                out = asyncio.run(author._run_with_container_files(self._input()))

        self.assertEqual(out.answer_tex, "new answer")
        self.assertEqual(out.research_notes_tex, "old notes")
        self.assertEqual(out.references_bib, "old refs")
        self.assertEqual(out.files_changed, ["answer.tex"])
        self.assertTrue(out.ready)
        self.assertEqual(out.via, "anthropic_container_files")
        self.assertIn(ANTHROPIC_FILES_BETA, built_configs[0]["anthropic_betas"])
        self.assertEqual(
            [tool[1]["type"] for tool in built_configs[0]["tools"]],
            ["code_interpreter", "web_search_preview"],
        )
        self.assertEqual(built_configs[0]["tools"][1][1]["max_uses"], Author.MAX_TOOL_CALLS)
        anthropic_ctor.assert_called_once_with(api_key="test-key-not-used")
        self.assertEqual(len(fake_client.files.upload_calls), len(CANONICAL_FILES))
        self.assertEqual(len(fake_client.files.delete_calls), len(CANONICAL_FILES) + 1)

    def test_fable_author_round0_raises_when_no_generated_files(self) -> None:
        fake_client = _FakeAnthropicClient()

        with tempfile.TemporaryDirectory() as td:
            ctx = RunContext.create(
                run_id="test_author_anthropic_no_files",
                root_workdir=Path(td),
                flat=True,
                api_client_factory=lambda cfg: SimpleNamespace(model=cfg["model"]),
                component_configs={
                    "Author": {"model": "models/anthropic/fable_5_max"}
                },
            )
            author = Author(ctx)

            with (
                patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-not-used"}),
                patch.object(
                    sys.modules["anthropic"],
                    "Anthropic",
                    MagicMock(return_value=fake_client),
                ),
                patch(
                    "proofstack.agents.ac.author._one_shot_query",
                    return_value=(
                        0,
                        [{"role": "assistant", "content": "<ready>false</ready>"}],
                        {"cost": 0.0, "input_tokens": 1, "output_tokens": 2},
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "no generated canonical files",
                ):
                    asyncio.run(author._run_with_container_files(self._input()))

        self.assertEqual(len(fake_client.files.upload_calls), len(CANONICAL_FILES))
        self.assertEqual(len(fake_client.files.delete_calls), len(CANONICAL_FILES))

    def test_fable_author_cleans_up_uploaded_inputs_when_call_raises(self) -> None:
        fake_client = _FakeAnthropicClient()

        with tempfile.TemporaryDirectory() as td:
            ctx = RunContext.create(
                run_id="test_author_anthropic_files_failure",
                root_workdir=Path(td),
                flat=True,
                api_client_factory=lambda cfg: SimpleNamespace(model=cfg["model"]),
                component_configs={
                    "Author": {"model": "models/anthropic/fable_5_max"}
                },
            )
            author = Author(ctx)

            with (
                patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-not-used"}),
                patch.object(
                    sys.modules["anthropic"],
                    "Anthropic",
                    MagicMock(return_value=fake_client),
                ),
                patch(
                    "proofstack.agents.ac.author._one_shot_query",
                    side_effect=RuntimeError("simulated Fable failure"),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    asyncio.run(author._run_with_container_files(self._input()))

        self.assertEqual(len(fake_client.files.upload_calls), len(CANONICAL_FILES))
        self.assertEqual(len(fake_client.files.delete_calls), len(CANONICAL_FILES))

    def test_fable_author_cleans_up_partial_upload_when_upload_raises(self) -> None:
        fake_client = _FakeAnthropicClient()
        fake_client.files.fail_upload_after = 1

        with tempfile.TemporaryDirectory() as td:
            ctx = RunContext.create(
                run_id="test_author_anthropic_upload_failure",
                root_workdir=Path(td),
                flat=True,
                api_client_factory=lambda cfg: SimpleNamespace(model=cfg["model"]),
                component_configs={
                    "Author": {"model": "models/anthropic/fable_5_max"}
                },
            )
            author = Author(ctx)

            with (
                patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-not-used"}),
                patch.object(
                    sys.modules["anthropic"],
                    "Anthropic",
                    MagicMock(return_value=fake_client),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    asyncio.run(author._run_with_container_files(self._input()))

        self.assertEqual(len(fake_client.files.upload_calls), 1)
        self.assertEqual(len(fake_client.files.delete_calls), 1)


if __name__ == "__main__":
    unittest.main()
