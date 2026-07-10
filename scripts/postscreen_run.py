"""Post-run audit: hand one completed workflow run's logs to a frontier
model (with web_search + code_interpreter) and get back a full audit report
plus any artifacts (progress curve, round-by-round CSV) the model generates.

This automates the manual "upload the run zip to a chat assistant and ask
for an audit" process, with cost and token accounting kept in ``APIClient``.

Usage::

    # zip a run folder on the fly (curated bundle) and audit it
    uv run python scripts/postscreen_run.py \\
        --run-dir outputs/<run_id> \\
        --slug <problem_id>

    # or audit an already-built zip
    uv run python scripts/postscreen_run.py --zip /path/run.zip --slug <problem_id>

What it does:

  1. Builds (or takes) a zip of the run. By default it excludes the
     per-round ``compute_workspace_round_*.zip`` snapshots (already
     compressed, redundant) and paper PDFs (their extracted ``.txt`` are
     kept) — this drops a typical run from ~280 MB to ~12 MB without
     losing the reasoning trace.
  2. Uploads the zip to OpenAI and attaches it to a code_interpreter
     container.
  3. Runs a single audit call (default ``models/openai/gpt-55-pro``,
     override with ``--model``) through ``APIClient`` (so cost and token
     accounting stay centralized).
  4. Downloads whatever the model saved under ``/mnt/data/audit_artifacts/``
     (the progress PNG, the CSV, ...).
  5. Writes ``report.md``, ``artifacts/``, ``conversation.json`` and
     ``audit_meta.json`` into the output directory.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from _env import load_dotenv_file  # noqa: E402

load_dotenv_file(REPO_ROOT / ".env")

from mathagents import APIClient, load_solver_config  # noqa: E402
from openai import OpenAI  # noqa: E402


DEFAULT_MODEL = "models/openai/gpt-55-pro"

# Bulky, redundant artifacts excluded from the audit bundle by default.
# The per-round workspace snapshots are already-compressed zips that just
# duplicate earlier workspace state; the paper PDFs have ``.txt`` siblings.
WORKSPACE_SNAPSHOT_GLOB = "compute_workspace_round_*.zip"

SYSTEM_PROMPT = (
    "You are a senior research mathematician and ML-systems auditor "
    "reviewing the logs of an autonomous multi-agent workflow that attacks "
    "open research-level mathematics problems over many rounds (an Author "
    "drafts a solution, a Critic reviews it, a Council of frontier models "
    "advises, and a Compute agent runs code). Be precise, concrete, and "
    "honest; cite specific files/rounds for every claim. Use $...$ and "
    "$$...$$ for math so the report renders through pandoc."
)

# The audit prompt itself, kept verbatim from the manual ChatGPT-Pro process.
AUDIT_PROMPT = """\
We are testing a new workflow for iterative, agent based solutions of hard \
math research questions. Appended is the output/log folder of one such \
sustained attempt on an open conjecture. To optimize our processes, it would \
be extremely helpful if you could read through these files very carefully, \
trace how the reasoning went, if any technical problems arose, if the models \
encountered any non-anticipated blockers in our agent setup etc. Basically \
anything that could help us obtain a smoother, more productive workflow. Also \
analyze in detail the progress that was obtained over the rounds and how each \
added round influenced the quality. If possible, setting 100% as a complete \
solution, try to estimate how much progress was achieved after each round, \
and draw a respective graph.
Please perform a complete audit, including tracing through mathematical \
exploration directions and how issues arise, resolve, possibly regress etc. \
Then give me a full report, including any recommendations based on your \
findings."""

# Operational scaffolding appended to the audit prompt so the model knows
# where the data is, what was intentionally excluded, and how to hand back
# its figures/tables so we can retrieve them.
OPERATOR_INSTRUCTIONS = """\

---
Operational notes (not part of the math, but follow them):

- The run's logs are in a single zip uploaded to your code_interpreter \
sandbox under /mnt/data (its basename carries a platform-id prefix; it is \
the only .zip directly under /mnt/data). Unzip it to a working directory and \
explore the tree before reading. Read the run metadata, events, every \
Author/Critic/Council round's input/output/messages, the compute agent's \
cli_stdout logs and reports, the canonical answer.tex / research_notes.tex \
progression, and the final solution.
- {exclusions_note}
- Deliverables: create the directory /mnt/data/audit_artifacts/ and save \
into it (a) the progress-vs-round curve as a PNG and (b) the round-by-round \
progress estimates as a CSV. Anything you put there will be retrieved and \
filed next to your report, so reference these files by name in the report.
- Return the full audit as your final assistant message in GitHub-flavored \
Markdown."""


def _build_bundle(
    run_dir: Path,
    out_zip: Path,
    *,
    include_workspace_zips: bool,
    include_pdfs: bool,
    exclude: list[str] | None = None,
) -> tuple[int, int]:
    """Zip ``run_dir`` into ``out_zip`` under a top-level folder named after
    the run dir. ``exclude`` holds fnmatch globs tested against the path
    relative to ``run_dir`` (posix form). Returns (files_written,
    files_skipped)."""
    root_name = run_dir.name
    written = skipped = 0
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    out_zip_resolved = out_zip.resolve()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            # Never zip the archive into itself — possible when --out-dir is
            # placed inside --run-dir.
            if path.resolve() == out_zip_resolved:
                skipped += 1
                continue
            name = path.name
            if not include_pdfs and name.lower().endswith(".pdf"):
                skipped += 1
                continue
            if not include_workspace_zips and fnmatch.fnmatch(name, WORKSPACE_SNAPSHOT_GLOB):
                skipped += 1
                continue
            rel = path.relative_to(run_dir).as_posix()
            if exclude and any(fnmatch.fnmatch(rel, pat) for pat in exclude):
                skipped += 1
                continue
            arcname = Path(root_name) / path.relative_to(run_dir)
            zf.write(path, arcname.as_posix())
            written += 1
    return written, skipped


def _exclusions_note(
    *,
    from_zip: bool,
    include_pdfs: bool,
    include_workspace_zips: bool,
    exclude: list[str] | None = None,
) -> str:
    """Describe what was actually excluded from the bundle, so the operator
    note never claims a curation that didn't happen."""
    if from_zip:
        return (
            "This bundle was prepared externally, so its exact contents are "
            "unknown to this tool; audit whatever is present and do not assume "
            "any particular file is missing by design."
        )
    excluded = []
    if not include_workspace_zips:
        excluded.append("per-round `compute_workspace_round_*.zip` workspace snapshots")
    if not include_pdfs:
        excluded.append("downloaded paper PDFs (their extracted `.txt` are kept)")
    for pat in exclude or []:
        excluded.append(f"files matching `{pat}`")
    if not excluded:
        return "The bundle is complete: nothing was excluded when it was built."
    return (
        "To keep the upload small, these bulky, redundant artifacts were "
        "excluded when building this bundle: " + " and ".join(excluded)
        + ". Do NOT treat their absence as a workflow defect."
    )


def _container_id_from_conversation(conversation: list[dict]) -> str | None:
    for msg in reversed(conversation):
        if msg.get("type") == "code_interpreter_call" and msg.get("container_id"):
            return msg["container_id"]
    return None


def _assistant_text(conversation: list[dict]) -> str:
    for msg in reversed(conversation):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        parts.append(block["text"])
                    elif block.get("type") == "output_text":
                        parts.append(block.get("text", ""))
            if parts:
                return "\n".join(parts)
    return ""


def _read_body(resp) -> bytes:
    if hasattr(resp, "read"):
        return resp.read()
    if hasattr(resp, "content"):
        return resp.content
    return bytes(resp)


def _download_artifacts(oc: OpenAI, container_id: str, bundle_name: str, dest: Path) -> list[str]:
    """Download the deliverables the model wrote to /mnt/data/audit_artifacts/.

    If the model ignored that directory and dropped figures/tables at the
    /mnt/data/ root instead, fall back to those — but only then, so we don't
    also pull lower-res intermediates the model left at the root alongside a
    proper audit_artifacts/ copy.
    """
    try:
        files = list(oc.containers.files.list(container_id))
    except Exception as e:  # container may have expired
        print(f"  WARN: could not list container files: {e}", file=sys.stderr)
        return []

    deliverable_exts = (".png", ".csv", ".svg", ".pdf", ".json", ".xlsx", ".jpg", ".jpeg")
    in_artifacts = [cf for cf in files if "/audit_artifacts/" in str(getattr(cf, "path", "") or "")]
    if in_artifacts:
        chosen = in_artifacts
    else:
        chosen = [
            cf for cf in files
            if (path := str(getattr(cf, "path", "") or "")).startswith("/mnt/data/")
            and path.count("/") == 3
            and path.rsplit("/", 1)[-1].lower().endswith(deliverable_exts)
            and path.rsplit("/", 1)[-1] != bundle_name
        ]

    dest.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    seen: set[str] = set()
    for cf in chosen:
        base = str(getattr(cf, "path", "") or "").rsplit("/", 1)[-1]
        if base in seen:
            continue
        seen.add(base)
        try:
            body = _read_body(oc.containers.files.content.retrieve(cf.id, container_id=container_id))
        except Exception as e:
            print(f"  WARN: could not download {base}: {e}", file=sys.stderr)
            continue
        (dest / base).write_bytes(body)
        saved.append(base)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="run folder to zip + audit")
    src.add_argument("--zip", type=Path, help="pre-built zip to audit")
    parser.add_argument("--slug", required=True, help="problem slug, for naming/labels")
    parser.add_argument("--out-dir", type=Path, default=None, help="output dir (default: outputs/postscreen-<slug>-<ts>)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model spec (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt-file", type=Path, default=None, help="override the audit prompt")
    parser.add_argument("--include-workspace-zips", action="store_true", help="include compute_workspace_round_*.zip snapshots")
    parser.add_argument("--include-pdfs", action="store_true", help="include paper PDFs")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="fnmatch glob on the run-dir-relative path to skip when bundling "
        "(repeatable); use to stay under the code_interpreter container's "
        "1000-file limit, e.g. --exclude 'events_blobs/*'",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (REPO_ROOT / "outputs" / f"postscreen-{args.slug}-{ts}")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve the bundle.
    if args.zip:
        bundle = args.zip.resolve()
        if not bundle.is_file():
            raise SystemExit(f"--zip not found: {bundle}")
        print(f"using bundle: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)")
    else:
        run_dir = args.run_dir.resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"--run-dir not found: {run_dir}")
        bundle = out_dir / f"{args.slug}_run_audit.zip"
        written, skipped = _build_bundle(
            run_dir, bundle,
            include_workspace_zips=args.include_workspace_zips,
            include_pdfs=args.include_pdfs,
            exclude=args.exclude,
        )
        print(f"built bundle: {bundle.name} — {written} files ({skipped} excluded), {bundle.stat().st_size / 1e6:.1f} MB")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    oc = OpenAI(api_key=api_key)

    # 2. Upload + attach to a code_interpreter container.
    with open(bundle, "rb") as fh:
        uploaded = oc.files.create(file=fh, purpose="user_data")
    print(f"uploaded: file_id={uploaded.id} bytes={uploaded.bytes}")

    tool_pairs = [
        (None, {"type": "code_interpreter", "container": {"type": "auto", "file_ids": [uploaded.id]}}),
        (None, {"type": "web_search_preview"}),
    ]

    # 3. Build the APIClient from the proven Pro model config + our tools.
    cfg = {k: v for k, v in load_solver_config(args.model).items() if not k.startswith("__")}
    cfg["tools"] = tool_pairs
    client = APIClient(**cfg)

    audit_prompt = (
        args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else AUDIT_PROMPT
    )
    operator_note = OPERATOR_INSTRUCTIONS.replace(
        "{exclusions_note}",
        _exclusions_note(
            from_zip=bool(args.zip),
            include_pdfs=args.include_pdfs,
            include_workspace_zips=args.include_workspace_zips,
            exclude=args.exclude,
        ),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": audit_prompt + operator_note},
    ]

    print(f"calling {cfg.get('model')} (background, may run 15-60+ min)...")
    start = datetime.now()
    _idx, conversation, cost = next(iter(client.run_queries([messages], no_tqdm=True)))
    elapsed = (datetime.now() - start).total_seconds()

    report = _assistant_text(conversation).strip()
    container_id = _container_id_from_conversation(conversation)
    print(f"  done in {elapsed / 60:.1f} min; ${cost.get('cost', 0):.2f}; report {len(report)} chars; container={container_id}")

    # 4. Download artifacts the model produced.
    artifacts: list[str] = []
    if container_id:
        artifacts = _download_artifacts(oc, container_id, bundle.name, out_dir / "artifacts")
        print(f"  artifacts: {artifacts or 'none'}")
    else:
        print("  WARN: no container_id in conversation; cannot fetch artifacts", file=sys.stderr)

    # 5. Persist everything.
    (out_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    (out_dir / "conversation.json").write_text(
        json.dumps(conversation, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    meta = {
        "slug": args.slug,
        "model": cfg.get("model"),
        "timestamp": ts,
        "bundle": bundle.name,
        "bundle_bytes": bundle.stat().st_size,
        "file_id": uploaded.id,
        "container_id": container_id,
        "elapsed_s": round(elapsed, 1),
        "cost": cost,
        "artifacts": artifacts,
        "excluded": {
            "workspace_zips": not args.include_workspace_zips,
            "pdfs": not args.include_pdfs,
            "globs": args.exclude,
        },
        "report_chars": len(report),
    }
    (out_dir / "audit_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print("\npostscreen: done")
    print(f"  report:    {out_dir / 'report.md'}")
    if artifacts:
        print(f"  artifacts: {out_dir / 'artifacts'} ({', '.join(artifacts)})")
    print(f"  meta:      {out_dir / 'audit_meta.json'}")
    if not report:
        print("  WARN: report is empty; inspect conversation.json", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
