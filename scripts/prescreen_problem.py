"""Run the prescreen workflow on one submitted problem.

Usage::

    uv run python scripts/prescreen_problem.py \\
        --problem-dir test_run_problems/<slug>

Reads ``submission/problem.{txt,tex,md}`` from the slug folder, runs the
``prescreen`` workflow preset (gpt-5.5-pro--xhigh with web_search and
code_interpreter), writes ``prescreen/{report.md,report.pdf,response.json}``
plus ``cleaned/problem_clean.tex``, and merges the structured verdict into
``metadata.yaml``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from _env import load_dotenv_file  # noqa: E402

load_dotenv_file(REPO_ROOT / ".env")

from proofstack import RunContext  # noqa: E402
from proofstack.registry import load_preset  # noqa: E402


PROBLEM_CANDIDATES = ("problem.txt", "problem.tex", "problem.md")


def _find_problem_file(submission_dir: Path) -> Path:
    for name in PROBLEM_CANDIDATES:
        path = submission_dir / name
        if path.exists():
            return path
    matches = sorted(
        p for p in submission_dir.iterdir()
        if p.is_file() and p.suffix in {".txt", ".tex", ".md"}
    )
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(
        f"could not locate a problem file in {submission_dir}; "
        f"expected one of {PROBLEM_CANDIDATES} or a single .txt/.tex/.md"
    )


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    s = text.strip()
    m = _FENCE_RE.match(s)
    return m.group(1).strip() if m else s


def _render_pdf(report_md: Path, out_pdf: Path) -> None:
    cmd = [
        "pandoc", str(report_md), "-o", str(out_pdf),
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=1in",
        "-V", "mainfont=DejaVu Sans",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or "pandoc failed")


_TRUTHY = {"true", "yes", "y", "1"}
_FALSY = {"false", "no", "n", "0", ""}


def _coerce_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _TRUTHY:
            return True
        if key in _FALSY:
            return False
    raise ValueError(
        f"verdict {field!r} could not be coerced to bool: {value!r}"
    )


def _update_metadata(metadata_path: Path, verdict: dict[str, Any]) -> None:
    data = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    ps = data.get("prescreen") or {}
    ps["status"] = "done"
    ps["date"] = date.today().isoformat()
    ps["model"] = "gpt-5.5-pro--xhigh"
    ps["source"] = "workflow"
    ps["verdict"] = verdict.get("verdict", "")
    ps["summary"] = verdict.get("summary", "")
    ps["suitable_for_test"] = _coerce_bool(
        verdict.get("suitable_for_test"), field="suitable_for_test"
    )
    flags = verdict.get("flags") or []
    ps["flags"] = [str(f) for f in flags]
    data["prescreen"] = ps
    metadata_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


async def amain() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-dir", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing prescreen/response.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="run-log output directory (default: outputs/)",
    )
    args = parser.parse_args()

    problem_dir = args.problem_dir.resolve()
    submission_dir = problem_dir / "submission"
    if not submission_dir.is_dir():
        raise SystemExit(f"{problem_dir} has no submission/ subdir")

    prescreen_dir = problem_dir / "prescreen"
    cleaned_dir = problem_dir / "cleaned"
    prescreen_dir.mkdir(exist_ok=True)
    cleaned_dir.mkdir(exist_ok=True)

    response_path = prescreen_dir / "response.json"
    if response_path.exists() and not args.force:
        raise SystemExit(
            f"{response_path} already exists; rerun with --force to overwrite"
        )

    src_path = _find_problem_file(submission_dir)
    problem_text = src_path.read_text(encoding="utf-8").strip()
    print(f"prescreen: reading {src_path.relative_to(problem_dir)}", file=sys.stderr)

    preset = load_preset("prescreen")
    run_id = f"prescreen-{problem_dir.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    args.output.mkdir(parents=True, exist_ok=True)

    ctx = RunContext.create(
        run_id=run_id,
        root_workdir=args.output,
        flat=False,
        run_budget=preset.budget,
        model_overrides=dict(preset.model_overrides),
        component_configs=dict(preset.component_configs),
        config_snapshot={
            "preset": preset.name,
            "problem_id": problem_dir.name,
            "submission_path": str(src_path),
        },
    )

    built_inputs = preset.build_inputs(
        problem=problem_text,
        problem_id=problem_dir.name,
        cli_overrides={},
    )

    wf_cls = preset.workflow_cls
    wf = wf_cls(ctx)
    out = await wf(**built_inputs)
    out_json = out.model_dump(mode="json") if hasattr(out, "model_dump") else out

    response_path.write_text(
        json.dumps(out_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not isinstance(out_json, dict):
        raise SystemExit(
            f"workflow returned non-dict output; raw saved to {response_path}"
        )

    report_md = (out_json.get("report") or "").strip()
    cleaned_block = (out_json.get("cleaned_latex") or "").strip()
    verdict_block = (out_json.get("verdict") or "").strip()

    missing = [
        name for name, val in
        (("report", report_md), ("cleaned_latex", cleaned_block), ("verdict", verdict_block))
        if not val
    ]
    if missing:
        raise SystemExit(
            f"prescreen output missing fields: {missing}; "
            f"raw response at {response_path}. Rerun with --force after "
            f"checking the response."
        )

    (prescreen_dir / "report.md").write_text(report_md + "\n", encoding="utf-8")
    cleaned_tex = _strip_fence(cleaned_block)
    (cleaned_dir / "problem_clean.tex").write_text(cleaned_tex + "\n", encoding="utf-8")

    verdict_text = _strip_fence(verdict_block)
    try:
        verdict_dict = yaml.safe_load(verdict_text) or {}
    except yaml.YAMLError as e:
        raise SystemExit(
            f"verdict YAML invalid: {e}\nraw verdict block:\n{verdict_block}"
        )

    pdf_path = prescreen_dir / "report.pdf"
    try:
        _render_pdf(prescreen_dir / "report.md", pdf_path)
        pdf_status = str(pdf_path.relative_to(problem_dir))
    except (RuntimeError, OSError) as e:
        first_line = (str(e).splitlines() or ["unknown"])[0] or "unknown"
        pdf_status = f"FAILED ({first_line})"

    metadata_path = problem_dir / "metadata.yaml"
    if metadata_path.exists():
        _update_metadata(metadata_path, verdict_dict)

    print("prescreen: done")
    print(f"  verdict:           {verdict_dict.get('verdict')}")
    print(f"  summary:           {verdict_dict.get('summary')}")
    print(f"  suitable_for_test: {verdict_dict.get('suitable_for_test')}")
    print(f"  flags:             {verdict_dict.get('flags')}")
    print(f"  files:")
    print(f"    {prescreen_dir.relative_to(problem_dir)}/report.md")
    print(f"    {pdf_status}")
    print(f"    {cleaned_dir.relative_to(problem_dir)}/problem_clean.tex")
    print(f"    {response_path.relative_to(problem_dir)}")
    if metadata_path.exists():
        print(f"  metadata updated:  {metadata_path.relative_to(problem_dir)}")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
