"""Run the prescreen workflow on one submitted problem.

Usage::

    uv run python scripts/prescreen_problem.py \\
        --problem-dir path/to/<problem-dir>

The problem dir is expected to contain a single ``.txt/.tex/.md`` file in
``submission/``. For revised submissions, select the live statement with
``--problem-file submission/problem_conj1_only.tex`` (relative to the problem
directory, or absolute). Ambiguous directories require explicit selection.
The script runs the
``prescreen`` workflow preset (a referee-style model call with web_search and
code_interpreter), then writes ``prescreen/{report.md,report.pdf,response.json}``
and ``cleaned/problem_clean.tex`` back into the problem dir. If the dir has a
``metadata.yaml``, the structured verdict is merged into its ``prescreen:``
block; otherwise that step is skipped.

If filing fails after a response is saved, rerun with ``--resume-from RUN_DIR``
to file that response without another model call. Use the same statement and
``--problem-file`` selection; the saved model and statement fingerprint are
checked. Add ``--force`` only to replace an already filed result.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
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


PROBLEM_SUFFIXES = {".txt", ".tex", ".md"}


def _find_problem_file(submission_dir: Path) -> Path:
    matches = sorted(
        p for p in submission_dir.iterdir()
        if p.is_file() and p.suffix.lower() in PROBLEM_SUFFIXES
    )
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(
        f"could not locate a problem file in {submission_dir}; "
        "expected a single .txt/.tex/.md; use --problem-file to select "
        "the current statement explicitly"
    )


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    s = text.strip()
    m = _FENCE_RE.match(s)
    return m.group(1).strip() if m else s


def _render_pdf(report_md: Path, out_pdf: Path, *, timeout: float = 120) -> None:
    cmd = [
        "pandoc", str(report_md), "-o", str(out_pdf),
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=1in",
        "-V", "mainfont=DejaVu Sans",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        _, stderr = proc.communicate(timeout=timeout)
    except BaseException:
        # Pandoc spawns XeLaTeX; killing only Pandoc leaves the compiler alive.
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(OSError):
            proc.kill()
        proc.wait()
        raise
    finally:
        proc.stdout.close()
        proc.stderr.close()
    if proc.returncode != 0:
        raise RuntimeError(stderr.strip() or "pandoc failed")


def _write_text_atomic(path: Path, text: str) -> None:
    path = path.resolve()
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as work:
        staged = Path(work) / path.name
        staged.write_text(text, encoding="utf-8")
        if path.exists():
            staged.chmod(path.stat().st_mode & 0o777)
        staged.replace(path)


_TRUTHY = {"true", "yes", "y", "1"}
_FALSY = {"false", "no", "n", "0", ""}


def _coerce_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
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


def _parse_verdict(text: str) -> dict[str, Any]:
    verdict = yaml.safe_load(_strip_fence(text))
    if not isinstance(verdict, dict):
        raise ValueError("verdict must be a YAML mapping")
    for field in ("verdict", "summary", "suitable_for_test", "flags"):
        if field not in verdict:
            raise ValueError(f"verdict is missing {field!r}")
    for field in ("verdict", "summary"):
        if not isinstance(verdict[field], str) or not verdict[field].strip():
            raise ValueError(f"verdict {field!r} must be nonempty text")
    if verdict["verdict"] not in {
        "well-posed", "typo-or-clarification", "mathematical-issues", "known-answer", "easy",
    }:
        raise ValueError(f"unknown verdict: {verdict['verdict']!r}")
    verdict["suitable_for_test"] = _coerce_bool(
        verdict["suitable_for_test"], field="suitable_for_test"
    )
    flags = verdict["flags"]
    if not isinstance(flags, list) or any(
        not isinstance(flag, str) or not flag.strip() for flag in flags
    ):
        raise ValueError("verdict flags must be a list of nonempty strings")
    return verdict


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("metadata must be a YAML mapping")
    if data.get("prescreen") is not None and not isinstance(data["prescreen"], dict):
        raise ValueError("metadata prescreen must be a YAML mapping")
    return data


def _metadata_text(
    metadata_path: Path, verdict: dict[str, Any], model_spec: str, submission_path: str,
) -> str:
    data = _load_metadata(metadata_path)
    ps = dict(data.get("prescreen") or {})
    ps["status"] = "done"
    ps["date"] = date.today().isoformat()
    ps["model"] = model_spec
    ps["source"] = "workflow"
    ps["submission_path"] = submission_path
    for field in ("verdict", "summary", "suitable_for_test", "flags"):
        ps[field] = verdict[field]
    data["prescreen"] = ps
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


async def amain() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-dir", type=Path, required=True)
    parser.add_argument(
        "--problem-file", type=Path,
        help="live statement under submission/, relative to --problem-dir or absolute",
    )
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
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--model",
        default=None,
        help="model spec overriding the preset's prescreen model "
        "(e.g. models/openai/gpt-55-pro)",
    )
    source.add_argument(
        "--resume-from", type=Path,
        help="file the saved response in RUN_DIR without calling a model",
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

    if args.problem_file is None:
        src_path = _find_problem_file(submission_dir).resolve()
    else:
        src_path = (problem_dir / args.problem_file).resolve()
    if (
        not src_path.is_relative_to(submission_dir.resolve())
        or not src_path.is_file()
        or src_path.suffix.lower() not in PROBLEM_SUFFIXES
    ):
        raise SystemExit("--problem-file must name a .txt/.tex/.md file under submission/")
    # Logical path for logs and metadata: submission/ may itself be a symlink
    # whose resolved target is not under problem_dir.
    src_rel = Path("submission") / src_path.relative_to(submission_dir.resolve())
    problem_text = src_path.read_text(encoding="utf-8").strip()
    if not problem_text:
        raise SystemExit(f"problem statement is empty: {src_path}")
    metadata_path = problem_dir / "metadata.yaml"
    if metadata_path.exists():
        try:
            _load_metadata(metadata_path)
        except (ValueError, yaml.YAMLError, OSError) as e:
            raise SystemExit(f"invalid metadata: {e}") from e
    print(f"prescreen: reading {src_rel}", file=sys.stderr)

    provenance = {
        "problem_id": problem_dir.name,
        "submission_path": src_rel.as_posix(),
        "sha256": hashlib.sha256(problem_text.encode("utf-8")).hexdigest(),
    }
    if args.resume_from is not None:
        raw_path = args.resume_from.resolve() / "prescreen-response.json"
        try:
            saved = json.loads((raw_path.parent / "prescreen-input.json").read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or any(saved.get(k) != v for k, v in provenance.items()):
                raise ValueError("saved response does not match the selected statement")
            model_spec = saved.get("model")
            if not isinstance(model_spec, str) or not model_spec.strip():
                raise ValueError("saved response has no model provenance")
            raw_text = raw_path.read_text(encoding="utf-8")
            out_json = json.loads(raw_text)
        except (OSError, ValueError) as e:
            raise SystemExit(f"cannot resume filing: {e}") from e
    else:
        preset = load_preset("prescreen")
        component_configs = {
            name: dict(cfg) for name, cfg in preset.component_configs.items()
        }
        if args.model:
            component_configs["cfg_prescreen"]["model"] = args.model
        model_spec = component_configs["cfg_prescreen"]["model"]

        run_id = f"prescreen-{problem_dir.name}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        args.output.mkdir(parents=True, exist_ok=True)

        ctx = RunContext.create(
            run_id=run_id,
            root_workdir=args.output,
            flat=False,
            run_budget=preset.budget,
            model_overrides=dict(preset.model_overrides),
            component_configs=component_configs,
            config_snapshot={
                "preset": preset.name,
                "problem_id": problem_dir.name,
                "submission_path": str(src_path),
                "model": model_spec,
            },
        )
        _write_text_atomic(
            ctx.root_workdir / "prescreen-input.json",
            json.dumps({**provenance, "model": model_spec}, ensure_ascii=False, indent=2),
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

        # Keep diagnostics without replacing the previous accepted intake files.
        raw_path = ctx.root_workdir / "prescreen-response.json"
        raw_text = json.dumps(out_json, ensure_ascii=False, indent=2)
        _write_text_atomic(raw_path, raw_text)
    print(f"prescreen: saved response at {raw_path}; recover filing with "
          f"--resume-from {raw_path.parent} (no model call)", file=sys.stderr)

    if not isinstance(out_json, dict):
        raise SystemExit(
            f"workflow returned non-dict output; raw saved to {raw_path}"
        )
    if out_json.get("error") or out_json.get("last_gasp"):
        raise SystemExit(f"prescreen workflow failed; raw response at {raw_path}")

    for field in ("report", "cleaned_latex", "verdict"):
        if not isinstance(out_json.get(field), str):
            raise SystemExit(f"prescreen output {field!r} must be text; raw response at {raw_path}")
    report_md = out_json["report"].strip()
    cleaned_block = out_json["cleaned_latex"].strip()
    verdict_block = out_json["verdict"].strip()

    missing = [
        name for name, val in
        (("report", report_md), ("cleaned_latex", cleaned_block), ("verdict", verdict_block))
        if not val
    ]
    if missing:
        raise SystemExit(
            f"prescreen output missing fields: {missing}; "
            f"raw response at {raw_path}. Check the response before rerunning."
        )

    cleaned_tex = _strip_fence(cleaned_block)
    if not cleaned_tex:
        raise SystemExit(f"cleaned_latex is empty; raw response at {raw_path}")
    try:
        verdict_dict = _parse_verdict(verdict_block)
    except (ValueError, yaml.YAMLError) as e:
        raise SystemExit(f"verdict invalid: {e}; raw response at {raw_path}") from e

    metadata_text = None
    if metadata_path.exists():
        try:
            metadata_text = _metadata_text(
                metadata_path, verdict_dict, model_spec, src_rel.as_posix(),
            )
        except (ValueError, yaml.YAMLError, OSError) as e:
            raise SystemExit(f"invalid metadata: {e}; raw response at {raw_path}") from e

    # Invalidate an older completion marker before replacing any filed output.
    response_path.unlink(missing_ok=True)
    _write_text_atomic(prescreen_dir / "report.md", report_md + "\n")
    _write_text_atomic(cleaned_dir / "problem_clean.tex", cleaned_tex + "\n")

    pdf_path = prescreen_dir / "report.pdf"
    try:
        # A failed rerender must not leave a previous report's PDF in place.
        with tempfile.TemporaryDirectory(prefix="prescreen-pdf-", dir=prescreen_dir) as work:
            rendered = Path(work) / "report.pdf"
            _render_pdf(prescreen_dir / "report.md", rendered)
            rendered.replace(pdf_path)
        pdf_status = str(pdf_path.relative_to(problem_dir))
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        pdf_path.unlink(missing_ok=True)
        first_line = (str(e).splitlines() or ["unknown"])[0] or "unknown"
        pdf_status = f"FAILED ({first_line})"

    if metadata_text is not None:
        _write_text_atomic(metadata_path, metadata_text)
    _write_text_atomic(response_path, raw_text)

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
