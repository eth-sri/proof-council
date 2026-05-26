"""Single-call Pro smoke for the v1 container-files Author.

Exercises just the new code path — one Author.__call__ against
gpt-5.5-pro with code_interpreter + web_search_preview and three
canonical files uploaded as read-only attachments — and reports:

  - whether Pro followed the "write to /mnt/data/<name> canonical path"
    instruction (visible in Author.Outputs.files_changed and the
    container's file listing post-call);
  - the container_id returned;
  - per-canonical-file new-vs-old byte diff;
  - cost and duration.

This validates Pro respects the new prompt template and round-trips
files correctly, without committing to a full ACWorkflow run.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
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

from proofstack import BudgetSpec, RunContext  # noqa: E402
from proofstack.agents.ac.author import Author  # noqa: E402


PROBLEM = (
    "Prove that there are infinitely many primes p such that p+2 is "
    "also prime (twin-prime conjecture), OR if you cannot do that, "
    "carefully state the historical status, recent partial results "
    "(Zhang 2013, Maynard, Polymath 8b), and the precise current "
    "lower bound on lim inf (p_{n+1} - p_n). Cap your answer at 3 "
    "pages of standard article-class output. This is a smoke test "
    "of the workflow infrastructure, not a research deliverable — a "
    "correct historical survey is sufficient."
)


async def main() -> int:
    # Full date + hex nonce — short ``HHMMSS`` collides across days and
    # would let ``RunContext`` pick up a prior run's ``resume_cache``,
    # returning cached Author outputs instead of making the live
    # Pro/container-files call this script is supposed to exercise.
    run_id = (
        f"smoke_pro_container_{time.strftime('%Y%m%d-%H%M%S')}"
        f"_{secrets.token_hex(2)}"
    )
    print(f"run_id: {run_id}")
    out_dir = REPO_ROOT / "outputs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext.create(
        run_id=run_id,
        root_workdir=out_dir,
        flat=False,
        run_budget=BudgetSpec(max_usd=20.0, max_wallclock_s=2400),
        config_snapshot={"smoke": "pro+container_files"},
    )
    print(f"output: {out_dir}")

    author = Author(ctx)
    print(f"Author USE_CONTAINER_FILES = {Author.USE_CONTAINER_FILES}")

    start = time.monotonic()
    out = await author(
        problem=PROBLEM,
        round=0,
        n_rounds=1,
        budget_used_usd=0.0,
        budget_max_usd=20.0,
        answer_tex="",
        research_notes_tex="",
        references_bib="",
        prev_critique="",
        prev_council="",
    )
    elapsed = time.monotonic() - start

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"elapsed: {elapsed:.1f}s")
    print(f"via: {out.via}")
    print(f"container_id: {out.container_id}")
    print(f"files_changed: {out.files_changed}")
    print(f"ready: {out.ready}")
    print(f"council_question: {out.council_question!r}")
    print(f"parse_warnings: {out.parse_warnings}")
    print(f"raw_text length: {len(out.raw_text)} chars")
    print()
    print(f"answer.tex bytes: {len(out.answer_tex)}")
    print(f"research_notes.tex bytes: {len(out.research_notes_tex)}")
    print(f"references.bib bytes: {len(out.references_bib)}")
    print()
    print("answer.tex head:")
    print("----")
    print(out.answer_tex[:500])
    print("----")
    print()
    print("thinking_summary:")
    print("----")
    print(out.thinking_summary[:1000])
    print("----")

    # Save for inspection
    (out_dir / "author_outputs.json").write_text(
        out.model_dump_json(indent=2), encoding="utf-8"
    )
    (out_dir / "answer.tex").write_text(out.answer_tex, encoding="utf-8")
    (out_dir / "research_notes.tex").write_text(out.research_notes_tex, encoding="utf-8")
    (out_dir / "references.bib").write_text(out.references_bib, encoding="utf-8")
    print(f"\nAll artifacts saved under {out_dir}")

    # The whole point of this script is to validate the round trip. A
    # missing container_id, an Author that fell back to the inline path,
    # an empty answer.tex, or any parse warning means the v1 path did
    # NOT actually work — fail loudly so wrappers / CI cannot mistake
    # them for success.
    failures: list[str] = []
    if out.via != "container_files":
        failures.append(f"Author.via = {out.via!r}, expected 'container_files'")
    if out.container_id is None:
        failures.append("Author.container_id is None — no code_interpreter_call in the response")
    if not out.files_changed:
        failures.append("Author.files_changed is empty — model wrote no canonical files at /mnt/data/*")
    if not out.answer_tex.strip():
        failures.append("answer.tex came back empty after download")
    if out.parse_warnings:
        failures.append(f"parse_warnings: {out.parse_warnings}")

    if failures:
        print()
        print("FAIL — container-files round trip did not validate cleanly:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print()
    print("PASS — container-files round trip validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
