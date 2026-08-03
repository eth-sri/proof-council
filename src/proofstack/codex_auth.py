"""Single source of truth for classifying a codex CLI auth.json.

Both the sandbox environment decision (may provider keys pass?) and the
billing decision (subscription $0 vs paid API spend) must come from ONE
read of ONE file, or the two can disagree — the exact
paid-key-billed-as-subscription hazard this module exists to prevent.
"""
from __future__ import annotations

import json


def classify_codex_auth(auth_text: str | None) -> str:
    """Classify auth.json content: 'subscription', 'api_key', 'unknown', 'absent'.

    Known schemas:
      - legacy:  {"auth_mode": "chatgpt", ...}
      - current: {"OPENAI_API_KEY": null, "tokens": {"id_token": ...,
                  "access_token": ..., "refresh_token": ...}, ...}

    An embedded non-empty API key always wins ('api_key'): it is paid auth
    and must never be billed as subscription, whatever else the file says.
    """
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


__all__ = ["classify_codex_auth"]
