#!/usr/bin/env python3
"""Fetch a public ChatGPT share link and reconstruct the conversation.

CLI lead-researcher helper: the human runs a browser consultation,
clicks Share, pastes the link; the lead runs this to file the answer
verbatim. Thin wrapper around proofstack.harness.chatgpt_share (also
used by the dashboard's browser-harness flow). Exits nonzero on
retrieval failure so callers can fall back to manual paste.

Usage:
  python3 scripts/fetch_chatgpt_share.py <share-url-or-uuid> [outfile]

Prints (or writes) markdown: title, model metadata, then each visible
user/assistant message.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proofstack.harness.chatgpt_share import (  # noqa: E402
    ShareFetchError,
    fetch_share,
    visible_messages,
)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    try:
        data = fetch_share(sys.argv[1])
    except ShareFetchError as e:
        raise SystemExit(f"error: {e}")
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
