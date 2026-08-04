"""Codex auth.json classification — one decision drives both the sandbox
env and the billing. Must recognize the CURRENT token-based schema (no
auth_mode field), the legacy auth_mode schema, and always treat an
embedded API key as paid."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.codex_auth import classify_codex_auth  # noqa: E402


class ClassifyCodexAuthTests(unittest.TestCase):
    def test_current_token_schema_is_subscription(self) -> None:
        # The shape `codex login` writes today: no auth_mode at all.
        text = json.dumps(
            {
                "OPENAI_API_KEY": None,
                "tokens": {
                    "id_token": "x",
                    "access_token": "y",
                    "refresh_token": "z",
                    "account_id": "a",
                },
                "last_refresh": "2026-08-01T00:00:00Z",
            }
        )
        self.assertEqual(classify_codex_auth(text), "subscription")

    def test_legacy_auth_mode_schema_is_subscription(self) -> None:
        text = json.dumps({"auth_mode": "chatgpt", "OPENAI_API_KEY": None})
        self.assertEqual(classify_codex_auth(text), "subscription")

    def test_embedded_api_key_always_wins(self) -> None:
        for extra in ({}, {"auth_mode": "chatgpt"}, {"tokens": {"access_token": "t"}}):
            text = json.dumps({"OPENAI_API_KEY": "sk-paid", **extra})
            self.assertEqual(classify_codex_auth(text), "api_key", extra)

    def test_garbage_and_absent(self) -> None:
        self.assertEqual(classify_codex_auth("not json"), "unknown")
        self.assertEqual(classify_codex_auth(json.dumps({"tokens": {}})), "unknown")
        self.assertEqual(classify_codex_auth(json.dumps([1, 2])), "unknown")
        self.assertEqual(classify_codex_auth(None), "absent")


if __name__ == "__main__":
    unittest.main()
