from __future__ import annotations

import json

from proofstack.codex_auth import (
    classify_codex_auth,
    extract_codex_auth_secrets,
    redact_codex_secrets,
)


def test_current_token_schema_is_subscription() -> None:
    text = json.dumps(
        {
            "OPENAI_API_KEY": None,
            "tokens": {
                "id_token": "x",
                "access_token": "y",
                "refresh_token": "z",
            },
        }
    )
    assert classify_codex_auth(text) == "subscription"


def test_legacy_auth_mode_schema_is_subscription() -> None:
    text = json.dumps({"auth_mode": "chatgpt", "OPENAI_API_KEY": None})
    assert classify_codex_auth(text) == "subscription"


def test_embedded_api_key_always_wins() -> None:
    for extra in ({}, {"auth_mode": "chatgpt"}, {"tokens": {"access_token": "t"}}):
        text = json.dumps({"OPENAI_API_KEY": "sk-paid", **extra})
        assert classify_codex_auth(text) == "api_key"


def test_garbage_and_absent_are_not_accepted() -> None:
    assert classify_codex_auth("not json") == "unknown"
    assert classify_codex_auth(json.dumps({"tokens": {}})) == "unknown"
    assert classify_codex_auth(json.dumps([1, 2])) == "unknown"
    assert classify_codex_auth(None) == "absent"


def test_extract_and_redact_nested_codex_credentials() -> None:
    access = "access-token-that-must-not-leak"
    refresh = "refresh-token-that-must-not-leak"
    api_key = "sk-additional-secret"
    auth_text = json.dumps(
        {
            "tokens": {
                "access_token": access,
                "nested": [{"refresh_token": refresh}],
            }
        }
    )

    secrets = extract_codex_auth_secrets(auth_text, additional=(api_key, "short"))

    assert set(secrets) == {access, refresh, api_key}
    redacted = redact_codex_secrets(
        f"access={access} refresh={refresh} key={api_key}", secrets
    )
    assert access not in redacted
    assert refresh not in redacted
    assert api_key not in redacted
    assert redacted.count("[redacted-codex-credential]") == 3
