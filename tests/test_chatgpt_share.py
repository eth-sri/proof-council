from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.harness.chatgpt_share import (  # noqa: E402
    ShareFetchError,
    download_shared_file,
    extract_result,
    share_contains_token,
    share_id,
    validate_result,
)

FIXTURE = json.loads(
    (ROOT / "tests" / "fixtures" / "chatgpt_share_sample.json").read_text(encoding="utf-8")
)


class ShareIdTests(unittest.TestCase):
    def test_full_url_and_bare_uuid(self) -> None:
        uuid = "6a6ef011-902c-83eb-b009-7c85c80b2324"
        self.assertEqual(share_id(f"https://chatgpt.com/share/{uuid}"), uuid)
        self.assertEqual(share_id(uuid), uuid)
        self.assertEqual(share_id(f"  https://chatgpt.com/share/{uuid}?x=1 "), uuid)

    def test_garbage_is_none(self) -> None:
        self.assertIsNone(share_id("https://example.com/nope"))
        self.assertIsNone(share_id(""))


class ExtractResultTests(unittest.TestCase):
    def test_joins_assistant_messages_after_last_user(self) -> None:
        result = extract_result(FIXTURE)
        self.assertIn("Progress note", result["assistant_text"])
        self.assertIn("<answer>The lemma holds.</answer>", result["assistant_text"])
        self.assertNotIn("hidden bootstrap", result["assistant_text"])
        self.assertNotIn("Please follow", result["assistant_text"])

    def test_model_metadata(self) -> None:
        result = extract_result(FIXTURE)
        self.assertEqual(result["model_slug"], "gpt-5-6-sol-pro")
        self.assertEqual(result["effort"], "max")
        self.assertEqual(result["default_model_slug"], "gpt-5-6-sol")

    def test_generated_files_are_structured_with_message_ids(self) -> None:
        result = extract_result(FIXTURE)
        self.assertEqual(
            result["generated_files"],
            [
                {
                    "message_id": "msg-final",
                    "sandbox_path": "/mnt/data/report.pdf",
                    "name": "report.pdf",
                }
            ],
        )

    def test_sandbox_artifacts_from_assistant_only(self) -> None:
        result = extract_result(FIXTURE)
        self.assertIn("report.pdf", result["sandbox_artifacts"])
        self.assertTrue(
            any(a.startswith("sandbox:/") for a in result["sandbox_artifacts"])
        )
        # The operator's own upload is not a sandbox artifact.
        self.assertNotIn("instruction.txt", result["sandbox_artifacts"])


def _node(node_id: str, parent: str | None, role: str, text: str, meta: dict | None = None) -> dict:
    return {
        "id": node_id,
        "parent": parent,
        "message": {
            "author": {"role": role},
            "content": {"parts": [text]},
            "metadata": meta or {},
        },
    }


class BranchWalkTests(unittest.TestCase):
    def test_mapping_fallback_walks_active_branch_only(self) -> None:
        data = {
            "mapping": {
                "u1": _node("u1", None, "user", "question"),
                "a-abandoned": _node("a-abandoned", "u1", "assistant", "OLD BRANCH"),
                "a-final": _node("a-final", "u1", "assistant", "NEW ANSWER"),
            },
            "current_node": "a-final",
        }
        result = extract_result(data)
        self.assertEqual(result["assistant_text"], "NEW ANSWER")
        self.assertNotIn("OLD BRANCH", result["assistant_text"])

    def test_malformed_nodes_are_skipped(self) -> None:
        data = {
            "linear_conversation": [
                "garbage",
                {"message": "not a dict"},
                {"message": {"author": "nope", "content": None}},
                _node("u1", None, "user", "q"),
                _node("a1", "u1", "assistant", "ANSWER"),
            ]
        }
        result = extract_result(data)
        self.assertEqual(result["assistant_text"], "ANSWER")


class ArtifactScopeTests(unittest.TestCase):
    def test_pre_final_turn_files_and_citations_are_not_artifacts(self) -> None:
        data = {
            "linear_conversation": [
                _node("u1", None, "user", "first question"),
                _node(
                    "a1", "u1", "assistant", "early note",
                    {"attachments": [{"name": "old_output.csv"}]},
                ),
                _node("u2", "a1", "user", "follow-up"),
                _node(
                    "a2", "u2", "assistant", "final answer",
                    {"content_references": [
                        {"type": "file-citation", "name": "uploaded_source.pdf"},
                    ]},
                ),
            ]
        }
        result = extract_result(data)
        self.assertEqual(result["sandbox_artifacts"], [])

    def test_answer_turn_file_refs_are_artifacts(self) -> None:
        data = {
            "linear_conversation": [
                _node("u1", None, "user", "question"),
                _node(
                    "a1", "u1", "assistant", "final answer",
                    {
                        "attachments": [{"name": "plot.png"}],
                        "content_references": [{"type": "file", "name": "out.csv"}],
                    },
                ),
            ]
        }
        result = extract_result(data)
        self.assertIn("plot.png", result["sandbox_artifacts"])
        self.assertIn("out.csv", result["sandbox_artifacts"])


class CitationSourcesTests(unittest.TestCase):
    def test_reference_urls_are_appended_as_sources(self) -> None:
        data = {
            "linear_conversation": [
                _node("u1", None, "user", "question"),
                _node(
                    "a1", "u1", "assistant", "answer with citation",
                    {"content_references": [
                        {
                            "type": "grouped_webpages",
                            "items": [
                                {"title": "Some Paper", "url": "https://arxiv.org/abs/1234.5678"}
                            ],
                        }
                    ]},
                ),
            ]
        }
        result = extract_result(data)
        self.assertEqual(
            result["sources"], ["Some Paper — https://arxiv.org/abs/1234.5678"]
        )
        self.assertIn("Sources cited", result["assistant_text"])
        self.assertIn("https://arxiv.org/abs/1234.5678", result["assistant_text"])


class ValidateResultTests(unittest.TestCase):
    def _result(self, **overrides):
        base = {
            "assistant_text": "fine answer <answer_ready>true</answer_ready>",
            "model_slug": "gpt-5-6-sol-pro",
            "default_model_slug": "gpt-5-6-sol",
            "sandbox_artifacts": [],
        }
        base.update(overrides)
        return base

    def test_empty_answer_is_hard_error(self) -> None:
        with self.assertRaises(ShareFetchError):
            validate_result(self._result(assistant_text="  "), {})

    def test_clean_result_has_no_warnings(self) -> None:
        task = {
            "browser": {"expected_model_slugs": ["gpt-5-6-sol"]},
            "expected": {"markers": ["<answer_ready>"]},
        }
        self.assertEqual(validate_result(self._result(), task), [])

    def test_slug_mismatch_warns(self) -> None:
        task = {"browser": {"expected_model_slugs": ["gpt-5-6-sol"]}}
        warnings = validate_result(self._result(model_slug="o4-mini"), task)
        self.assertTrue(any("o4-mini" in w for w in warnings))

    def test_missing_slug_warns_when_expected(self) -> None:
        task = {"browser": {"expected_model_slugs": ["gpt-5-6-sol"]}}
        warnings = validate_result(
            self._result(model_slug=None, default_model_slug=None), task
        )
        self.assertTrue(any("no model slug" in w for w in warnings))

    def test_missing_fenced_files_warn(self) -> None:
        task = {"expected": {"fenced_files": ["answer.tex", "references.bib"]}}
        warnings = validate_result(self._result(), task)
        self.assertTrue(any("file path=" in w for w in warnings))
        with_file = self._result(
            assistant_text="```file path=answer.tex\nx\n```"
        )
        self.assertEqual(validate_result(with_file, task), [])

    def test_missing_marker_warns(self) -> None:
        task = {"expected": {"markers": ["<answer_ready>"]}}
        warnings = validate_result(self._result(assistant_text="no verdict"), task)
        self.assertTrue(any("<answer_ready>" in w for w in warnings))

    def test_sandbox_artifacts_warn(self) -> None:
        warnings = validate_result(
            self._result(sandbox_artifacts=["report.pdf"]), {}
        )
        self.assertTrue(any("report.pdf" in w for w in warnings))

    def test_slug_matching_is_delimiter_aware(self) -> None:
        task = {"browser": {"expected_model_slugs": ["gpt-5-6-sol-pro"]}}
        # Longer actual slug with a dash boundary matches …
        self.assertEqual(
            validate_result(self._result(model_slug="gpt-5-6-sol-pro-high"), task), []
        )
        # … but a shorter actual slug must not pass for a longer expected one.
        warnings = validate_result(self._result(model_slug="gpt-5"), task)
        self.assertTrue(any("gpt-5" in w for w in warnings))

    def test_effort_validation(self) -> None:
        task = {"browser": {"expected_efforts": ["max"]}}
        self.assertEqual(validate_result(self._result(effort="max"), task), [])
        warnings = validate_result(self._result(effort="medium"), task)
        self.assertTrue(any("medium" in w for w in warnings))
        warnings = validate_result(self._result(), task)
        self.assertTrue(any("effort" in w for w in warnings))

    def test_required_files_warn_individually(self) -> None:
        task = {"expected": {"required_files": ["answer.tex"], "fenced_files": ["answer.tex", "research_notes.tex"]}}
        # A response with only research_notes.tex passes the at-least-one
        # check but must still warn about the missing required answer.tex.
        result = self._result(
            assistant_text="```file path=research_notes.tex\nx\n```"
        )
        warnings = validate_result(result, task)
        self.assertTrue(any("required file 'answer.tex'" in w for w in warnings))

    def test_mixed_answer_models_warn(self) -> None:
        warnings = validate_result(
            self._result(answer_models=["gpt-5-6-sol-pro", "o4-mini"]), {}
        )
        self.assertTrue(any("mixes turns" in w for w in warnings))

    def test_task_token_verified_by_uploaded_instruction_file(self) -> None:
        result = extract_result(FIXTURE)
        # The fixture's user turn uploaded instruction.txt.
        self.assertTrue(share_contains_token(result, "tokX", "instruction.txt"))
        self.assertFalse(share_contains_token(result, "tokX", "instruction_tokX.txt"))
        task = {"task_token": "tokX", "instruction_file": "instruction_tokX.txt"}
        warnings = validate_result(result, task)
        self.assertTrue(any("could not verify" in w for w in warnings))
        task_ok = {"task_token": "tokX", "instruction_file": "instruction.txt"}
        warnings_ok = validate_result(result, task_ok)
        self.assertFalse(any("could not verify" in w for w in warnings_ok))

    def test_task_token_verified_by_message_text(self) -> None:
        data = {
            "linear_conversation": [
                _node("u1", None, "user", "[ProofCouncil task tokY]\nplease work"),
                _node("a1", "u1", "assistant", "done"),
            ]
        }
        result = extract_result(data)
        self.assertTrue(share_contains_token(result, "tokY", "instruction_tokY.txt"))


class DownloadGuardrailTests(unittest.TestCase):
    def test_rejects_non_allowlisted_hosts(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for url in (
                "https://evil.example.com/file",
                "http://sub.oaiusercontent.com/file",
                "https://oaiusercontent.com.evil.net/file",
                "",
            ):
                with self.assertRaises(ShareFetchError):
                    download_shared_file({"download_url": url}, Path(tmp))

    def test_rejects_oversized_files_before_downloading(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ShareFetchError):
                download_shared_file(
                    {
                        "download_url": "https://sub.oaiusercontent.com/f",
                        "file_size_bytes": 10**12,
                    },
                    Path(tmp),
                )


if __name__ == "__main__":
    unittest.main()
