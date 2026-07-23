#!/usr/bin/env python3
"""Fetch a public ChatGPT share link and reconstruct the conversation.

CLI lead-researcher helper: the human runs a browser consultation,
clicks Share, pastes the link; the lead runs this to file the answer
verbatim. Uses the undocumented backend-api/share/<uuid> JSON endpoint
(may break without notice; fallback = human pastes content manually).

Usage:
  python3 scripts/fetch_chatgpt_share.py <share-url-or-uuid> [outfile]

Prints (or writes) markdown: title, model metadata, then each visible
user/assistant message. Exits nonzero on retrieval failure so callers
can fall back to manual paste.
"""

import json
import re
import sys
import urllib.request

SHARE_ID_PATTERN = re.compile(
    r"(?:https?://chatgpt\.com/share/)?"
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


def fetch(url_or_id: str) -> dict:
    m = SHARE_ID_PATTERN.search(url_or_id.strip())
    if not m:
        raise SystemExit(f"not a ChatGPT share URL/UUID: {url_or_id!r}")
    endpoint = f"https://chatgpt.com/backend-api/share/{m.group(1)}"
    req = urllib.request.Request(
        endpoint,
        headers={
            "Accept": "application/json",
            # Cloudflare serves a challenge to non-browser UAs (2026-07-23)
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"https://chatgpt.com/share/{m.group(1)}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def visible_messages(data: dict):
    nodes = data.get("linear_conversation") or list(
        (data.get("mapping") or {}).values()
    )
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
        if not text.strip():
            continue
        yield {
            "role": role,
            "text": text,
            "model": meta.get("resolved_model_slug") or meta.get("model_slug"),
            "effort": meta.get("thinking_effort"),
        }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    data = fetch(sys.argv[1])
    lines = [f"# {data.get('title', 'Shared conversation')}"]
    lines.append(f"default_model_slug: {data.get('default_model_slug')}")
    models = set()
    body = []
    for item in visible_messages(data):
        models.add((item["model"], item["effort"]))
        body.append(f"\n## {item['role']}\n")
        body.append(item["text"])
    lines.append(f"message_models: {sorted(models, key=lambda t: tuple(str(x) for x in t))}")
    out = "\n".join(lines) + "\n" + "\n".join(body) + "\n"
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", encoding="utf-8") as fh:
            fh.write(out)
        print(
            f"wrote {len(out)} chars; models seen: {sorted(models, key=lambda t: tuple(str(x) for x in t))}"
        )
    else:
        print(out)


if __name__ == "__main__":
    main()
