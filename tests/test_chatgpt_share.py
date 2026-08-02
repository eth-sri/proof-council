from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.harness.chatgpt_share import (  # noqa: E402
    ShareFetchError,
    extract_result,
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

    def test_sandbox_artifacts_from_assistant_only(self) -> None:
        result = extract_result(FIXTURE)
        self.assertIn("report.pdf", result["sandbox_artifacts"])
        self.assertTrue(
            any(a.startswith("sandbox:/") for a in result["sandbox_artifacts"])
        )
        # The operator's own upload is not a sandbox artifact.
        self.assertNotIn("instruction.txt", result["sandbox_artifacts"])


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


if __name__ == "__main__":
    unittest.main()
