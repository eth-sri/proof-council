"""Classify Codex CLI authentication consistently for access and billing."""
from __future__ import annotations

import json
from collections.abc import Iterable


_SECRET_KEY_PARTS = ("api_key", "access_token", "refresh_token", "id_token")
_REDACTION = "[redacted-codex-credential]"


def classify_codex_auth(auth_text: str | None) -> str:
    """Return ``subscription``, ``api_key``, ``unknown``, or ``absent``."""
    if auth_text is None:
        return "absent"
    try:
        data = json.loads(auth_text)
    except json.JSONDecodeError:
        return "unknown"
    if not isinstance(data, dict):
        return "unknown"
    api_key = data.get("OPENAI_API_KEY")
    if isinstance(api_key, str) and api_key.strip():
        return "api_key"
    if str(data.get("auth_mode") or "").lower() == "chatgpt":
        return "subscription"
    tokens = data.get("tokens")
    if isinstance(tokens, dict) and any(
        isinstance(tokens.get(key), str) and tokens.get(key)
        for key in ("access_token", "id_token", "refresh_token")
    ):
        return "subscription"
    return "unknown"


def extract_codex_auth_secrets(
    auth_text: str | None,
    *,
    additional: Iterable[str] = (),
) -> tuple[str, ...]:
    """Extract credential values that must never enter worker artifacts."""
    values: set[str] = {
        value.strip()
        for value in additional
        if isinstance(value, str) and len(value.strip()) >= 8
    }
    if auth_text is not None:
        try:
            data = json.loads(auth_text)
        except json.JSONDecodeError:
            data = None

        def visit(value, key: str = "") -> None:
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    visit(child_value, str(child_key).lower())
                return
            if isinstance(value, list):
                for child in value:
                    visit(child, key)
                return
            if (
                isinstance(value, str)
                and len(value.strip()) >= 8
                and any(part in key for part in _SECRET_KEY_PARTS)
            ):
                values.add(value.strip())

        visit(data)
    return tuple(sorted(values, key=lambda value: (-len(value), value)))


def redact_codex_secrets(text: str, secrets: Iterable[str]) -> str:
    """Replace known credential values without logging or returning them."""
    redacted = str(text)
    for secret in sorted(
        {value for value in secrets if isinstance(value, str) and len(value) >= 8},
        key=lambda value: (-len(value), value),
    ):
        redacted = redacted.replace(secret, _REDACTION)
    return redacted


__all__ = [
    "classify_codex_auth",
    "extract_codex_auth_secrets",
    "redact_codex_secrets",
]
