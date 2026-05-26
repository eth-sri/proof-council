"""Wiring-check + PDF visual-parsing test for gpt-5.5-pro.

One-shot Pro call with ``code_interpreter`` and ``web_search_preview``
both enabled. Asks the model to locate Figure 8 in a public PDF and
describe its visual contents. Two questions answered in one shot:

1. Does our APIClient accept both hosted tools simultaneously on the
   Responses API for gpt-5.5-pro?
2. Can Pro + Code Interpreter actually do meaningful visual PDF
   parsing (i.e. render a page image and "see" it)?

Expected ground truth (per user): Figure 8 is a schematic genus-g
surface drawn as a donut with two holes on the left and right
connected by dots, labeled "g holes" with arrows; it appears on
page 21.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from _env import load_dotenv_file  # noqa: E402
load_dotenv_file(REPO_ROOT / ".env")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mathagents import load_solver_config  # noqa: E402
from mathagents.api_client import APIClient  # noqa: E402


SYSTEM = """\
You are a research mathematician with access to two tools:
- code_interpreter (Python sandbox with TeX Live and standard libs)
- web_search_preview

Use whichever combination you need. Be specific about what you observe
visually; do not summarize the prose around the figure. If you cannot
actually see the figure pixels, say so explicitly rather than
fabricating from context.
"""

USER = """\
The PDF at https://johannesschmitt.gitlab.io/ModCurves/Script.pdf
contains a Figure 8.

Tasks, in order:

1. Find which page Figure 8 is on.
2. Describe its visual contents in detail. I want a description of the
   actual drawn picture (shapes, layout, labels, arrows), not the
   surrounding caption or prose.
3. Tell me what kind of mathematical object it depicts.

Show your work — I want to see whether you used code_interpreter to
render the page, web_search to find context, both, or neither.
"""


def main() -> int:
    cfg = load_solver_config("models/openai/gpt-55-pro")
    cfg = {k: v for k, v in cfg.items() if not k.startswith("__")}
    cfg["tools"] = [
        (None, {"type": "code_interpreter", "container": {"type": "auto"}}),
        (None, {"type": "web_search_preview"}),
    ]
    cfg["max_tool_calls"] = 30
    # Pro xhigh + dual hosted tools commonly runs 15-25 min in our
    # tests; bump the per-call wallclock from the 600 s default so
    # the call has room to finish.
    cfg["max_wallclock_per_call_s"] = 1800.0

    client = APIClient(**cfg)
    messages = [
        {"role": "developer", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]
    print("Issuing one-shot gpt-5.5-pro call with CI + web_search...")
    start = time.monotonic()
    iterator = client.run_queries([messages], no_tqdm=True)
    idx, conversation, cost = next(iter(iterator))
    elapsed = time.monotonic() - start

    final_text = ""
    tool_call_summary: list[str] = []
    for msg in conversation:
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            final_text = msg["content"]
        elif msg.get("type") == "code_interpreter_call":
            code = msg.get("code", "") or ""
            head = code.strip().splitlines()[:3]
            tool_call_summary.append(f"CI cell ({len(code)} chars): " + " | ".join(head))
        elif msg.get("type") == "web_search_call":
            tool_call_summary.append(f"WebSearch query: {msg.get('query','')[:120]}")

    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Cost: ${cost.get('cost', 0):.3f}")
    print(f"Input tokens: {cost.get('input_tokens', 0)}")
    print(f"Output tokens: {cost.get('output_tokens', 0)}")
    print(f"Tool-call trail ({len(tool_call_summary)} entries):")
    for line in tool_call_summary:
        print(f"  - {line}")
    print()
    print("=" * 70)
    print("Final assistant text:")
    print("=" * 70)
    print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
