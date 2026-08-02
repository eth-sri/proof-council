"""Fetch a public ChatGPT share link and reconstruct the consultation.

Uses the undocumented ``chatgpt.com/backend-api/share/<uuid>`` JSON
endpoint; it may break without notice, so every consumer must keep the
manual-paste path as a first-class fallback. Sandbox-generated files
are never exposed by the share page — ``extract_result`` only detects
that they exist so the operator can be told to download them by hand.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

SHARE_ID_PATTERN = re.compile(
    r"(?:https?://chatgpt\.com/share/)?"
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)

# Cloudflare serves a challenge to non-browser UAs (2026-07-23).
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

_SANDBOX_LINK_RE = re.compile(r"sandbox:/[^\s)\"'>]+")
_FILE_FENCE_RE = re.compile(r"```file\s+path=(?P<path>[A-Za-z0-9._/-]+)")


class ShareFetchError(Exception):
    """Retrieval or validation failure with an operator-friendly message."""


def share_id(url_or_id: str) -> str | None:
    m = SHARE_ID_PATTERN.search(str(url_or_id).strip())
    return m.group(1) if m else None


def fetch_share(url_or_id: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
    sid = share_id(url_or_id)
    if sid is None:
        raise ShareFetchError(
            f"not a ChatGPT share URL or UUID: {str(url_or_id)[:200]!r}"
        )
    req = urllib.request.Request(
        f"https://chatgpt.com/backend-api/share/{sid}",
        headers={
            "Accept": "application/json",
            "User-Agent": _BROWSER_UA,
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"https://chatgpt.com/share/{sid}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ShareFetchError(
                "share link not found (deleted, never published, or mistyped)"
            ) from e
        if e.code == 403:
            raise ShareFetchError(
                "the share endpoint refused the request (Cloudflare challenge "
                "or endpoint change) — use the manual paste fallback"
            ) from e
        raise ShareFetchError(f"share fetch failed with HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise ShareFetchError(f"network error fetching share link: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise ShareFetchError(
            "share endpoint returned non-JSON (endpoint change?) — use the "
            "manual paste fallback"
        ) from e


def visible_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = data.get("linear_conversation") or list(
        (data.get("mapping") or {}).values()
    )
    out: list[dict[str, Any]] = []
    for node in nodes:
        msg = node.get("message")
        if not msg:
            continue
        meta = msg.get("metadata") or {}
        role = (msg.get("author") or {}).get("role")
        if role not in ("user", "assistant"):
            continue
        if meta.get("is_visually_hidden_from_conversation"):
            continue
        parts = (msg.get("content") or {}).get("parts") or []
        text = "".join(p for p in parts if isinstance(p, str))
        if not text.strip() and not meta.get("attachments"):
            continue
        out.append(
            {
                "role": role,
                "text": text,
                "model": meta.get("resolved_model_slug") or meta.get("model_slug"),
                "effort": meta.get("thinking_effort"),
                "attachments": list(meta.get("attachments") or []),
                "content_references": list(meta.get("content_references") or []),
            }
        )
    return out


def extract_result(data: dict[str, Any]) -> dict[str, Any]:
    msgs = visible_messages(data)
    last_user = -1
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            last_user = i
    answer_msgs = [m for m in msgs[last_user + 1 :] if m["role"] == "assistant"]
    assistant_text = "\n\n".join(
        m["text"] for m in answer_msgs if m["text"].strip()
    )

    model_slug = None
    effort = None
    for m in reversed(answer_msgs or [m for m in msgs if m["role"] == "assistant"]):
        model_slug = model_slug or m["model"]
        effort = effort or m["effort"]

    sandbox_artifacts: list[str] = []
    for m in msgs:
        if m["role"] != "assistant":
            continue
        for att in m["attachments"]:
            name = att.get("name") if isinstance(att, dict) else None
            sandbox_artifacts.append(str(name or att))
        for ref in m["content_references"]:
            if isinstance(ref, dict) and "file" in str(ref.get("type") or ""):
                sandbox_artifacts.append(str(ref.get("name") or ref.get("type")))
    sandbox_artifacts.extend(_SANDBOX_LINK_RE.findall(assistant_text))
    seen: set[str] = set()
    sandbox_artifacts = [
        a for a in sandbox_artifacts if not (a in seen or seen.add(a))
    ]

    return {
        "title": data.get("title") or "Shared conversation",
        "assistant_text": assistant_text,
        "model_slug": model_slug,
        "effort": effort,
        "default_model_slug": data.get("default_model_slug"),
        "models_seen": sorted(
            {
                (m["model"], m["effort"])
                for m in msgs
                if m["role"] == "assistant" and m["model"]
            },
            key=lambda t: tuple(str(x) for x in t),
        ),
        "sandbox_artifacts": sandbox_artifacts,
        "messages": msgs,
    }


def validate_result(result: dict[str, Any], task: dict[str, Any]) -> list[str]:
    """Operator-facing warnings; an empty answer is a hard error."""
    text = str(result.get("assistant_text") or "")
    if not text.strip():
        raise ShareFetchError(
            "the shared conversation contains no assistant answer after your "
            "message — did you share before the model finished?"
        )
    warnings: list[str] = []

    browser = task.get("browser") or {}
    expected_slugs = [s for s in (browser.get("expected_model_slugs") or []) if s]
    slug = result.get("model_slug") or result.get("default_model_slug")
    if expected_slugs:
        if not slug:
            warnings.append(
                "the share carried no model slug, so the model used cannot be "
                f"verified (expected one of {expected_slugs})"
            )
        elif not any(str(slug).startswith(s) or s.startswith(str(slug)) for s in expected_slugs):
            warnings.append(
                f"the conversation ran on model {slug!r}, expected one of "
                f"{expected_slugs} — check the model picker"
            )

    expected = task.get("expected") or {}
    fenced = [str(f) for f in (expected.get("fenced_files") or [])]
    if fenced:
        present = set(_FILE_FENCE_RE.findall(text))
        if not present.intersection(fenced):
            warnings.append(
                "no fenced ``file path=...`` blocks found for "
                f"{fenced} — if the model offered file downloads instead, "
                "download them from the chat and add them via the manual upload"
            )
    for marker in expected.get("markers") or []:
        if str(marker) not in text:
            warnings.append(f"expected marker {str(marker)!r} not found in the answer")

    if result.get("sandbox_artifacts"):
        listed = ", ".join(str(a) for a in result["sandbox_artifacts"][:8])
        warnings.append(
            "the model created files in its sandbox that share links cannot "
            f"retrieve ({listed}) — download any you need from the chat and "
            "add them via the manual upload"
        )
    return warnings


__all__ = [
    "SHARE_ID_PATTERN",
    "ShareFetchError",
    "share_id",
    "fetch_share",
    "visible_messages",
    "extract_result",
    "validate_result",
]
