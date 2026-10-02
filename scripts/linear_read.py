"""Run the prefix-only define-before-use reader on a .tex file.

See proofstack.agents.linear_read for the method; CleanupSession exposes the
same reader to the editor as mcp__cleanup__linear_read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from proofstack.agents.linear_read import linear_read  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("document", help="Path to a .tex file")
    parser.add_argument("--model", default="models/anthropic/sonnet_5")
    parser.add_argument("--output", required=True, help="Output directory (created)")
    parser.add_argument("--max-blocks", type=int, default=0, help="Debug: only the first N passages")
    args = parser.parse_args()

    result = linear_read(Path(args.document).read_text(), args.model,
                         title=args.document, max_blocks=args.max_blocks)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(result["report"])
    (out / "report.json").write_text(json.dumps({"document": args.document, **result}, indent=1))
    print(f"{result['passages']} passages, {len(result['findings'])} findings, "
          f"${result['cost_usd']:.3f} -> {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
