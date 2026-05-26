from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from proofstack.agent import Agent


CANONICAL_FILES = (
    "answer.tex",
    "research_notes.tex",
    "references.bib",
    "report.md",
)

AUX_DIRS = ("papers", "code", "data", "notes")

EMPTY_TEMPLATES: dict[str, str] = {
    "answer.tex": (
        r"\documentclass[11pt]{article}" + "\n"
        r"\usepackage[utf8]{inputenc}" + "\n"
        r"\usepackage[T1]{fontenc}" + "\n"
        r"\usepackage{amsmath,amssymb,amsthm}" + "\n"
        r"\title{Solution}" + "\n"
        r"\date{}" + "\n"
        r"\begin{document}" + "\n"
        r"\maketitle" + "\n\n"
        "% This file is the candidate solution. The Worker will fill it in.\n"
        r"\end{document}" + "\n"
    ),
    "research_notes.tex": (
        r"\documentclass[11pt]{article}" + "\n"
        r"\usepackage[utf8]{inputenc}" + "\n"
        r"\usepackage[T1]{fontenc}" + "\n"
        r"\usepackage{amsmath,amssymb,amsthm}" + "\n"
        r"\title{Research notes}" + "\n"
        r"\date{}" + "\n"
        r"\begin{document}" + "\n"
        r"\maketitle" + "\n\n"
        "% Persistent research log: lemmas, definitions, summaries,\n"
        "% experiment outputs, and anything that should survive across\n"
        "% rounds.\n"
        r"\end{document}" + "\n"
    ),
    "references.bib": (
        "% Shared BibTeX file. Add entries used in answer.tex and\n"
        "% research_notes.tex.\n"
    ),
    "report.md": "# Round report\n\n_(rewritten each round)_\n",
}


class PWCWorkspaceFiles(BaseModel):
    workspace: Path
    answer_tex: str = ""
    research_notes_tex: str = ""
    references_bib: str = ""
    report_md: str = ""
    workspace_listing: str = ""
    answer_path: Path
    research_notes_path: Path
    references_bib_path: Path
    report_path: Path


class PWCWorkspaceInit(Agent):
    description: ClassVar[str] = "Create or read the persistent PWC workspace."
    execution_mode: ClassVar[str] = "deterministic_tool"
    cache_enabled: ClassVar[bool] = False

    class Inputs(BaseModel):
        problem: str
        problem_id: str

    class Outputs(BaseModel):
        workspace: Path

    async def run(self, inp):  # type: ignore[override]
        workspace = pwc_workspace_path(self.ctx.root_workdir, inp.problem_id)
        bootstrap_workspace(workspace, problem_text=inp.problem)
        return self.Outputs(workspace=workspace)


class PWCWorkspaceRead(Agent):
    description: ClassVar[str] = "Read the current PWC workspace files."
    execution_mode: ClassVar[str] = "deterministic_tool"
    cache_enabled: ClassVar[bool] = False

    class Inputs(BaseModel):
        workspace: Path

    class Outputs(PWCWorkspaceFiles):
        pass

    async def run(self, inp):  # type: ignore[override]
        bootstrap_workspace(inp.workspace)
        return self.Outputs(**workspace_files(inp.workspace).model_dump())


class PWCWorkspaceSnapshot(Agent):
    description: ClassVar[str] = "Snapshot PWC round artifacts under .pwc/round-K."
    execution_mode: ClassVar[str] = "deterministic_tool"
    cache_enabled: ClassVar[bool] = False

    class Inputs(BaseModel):
        problem_id: str
        round: int
        plan_md: str | None = ""
        review_md: str | None = ""
        workspace: Path
        status: str | None = ""
        diff_summary: str | None = ""
        open_questions: list[str] = Field(default_factory=list)

    class Outputs(PWCWorkspaceFiles):
        round: int

    async def run(self, inp):  # type: ignore[override]
        workspace = inp.workspace
        bootstrap_workspace(workspace)
        snap = workspace / ".pwc" / f"round-{inp.round}"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "plan.md").write_text(inp.plan_md or "", encoding="utf-8")
        if inp.review_md:
            (snap / "review.md").write_text(inp.review_md, encoding="utf-8")
        for name in CANONICAL_FILES:
            src = workspace / name
            if src.exists():
                try:
                    shutil.copyfile(src, snap / name)
                except OSError:
                    pass
        done = {
            "status": inp.status or "",
            "diff_summary": inp.diff_summary or "",
            "open_questions": inp.open_questions,
        }
        (snap / "done.json").write_text(json.dumps(done, indent=2), encoding="utf-8")
        return self.Outputs(round=inp.round, **workspace_files(workspace).model_dump())


class PWCStashAnswer(Agent):
    description: ClassVar[str] = "Write the final PWC answer under solutions/."
    execution_mode: ClassVar[str] = "deterministic_tool"
    cache_enabled: ClassVar[bool] = False

    class Inputs(BaseModel):
        problem_id: str
        tex: str
        workspace: Path
        compile_tex_path: Path | None = None
        ship_bib_alongside: bool = False

    class Outputs(BaseModel):
        answer_tex: Path

    async def run(self, inp):  # type: ignore[override]
        solutions_dir = self.ctx.root_workdir / "solutions"
        solutions_dir.mkdir(parents=True, exist_ok=True)
        safe = safe_id(inp.problem_id)
        bbl_path = inp.compile_tex_path.with_suffix(".bbl") if inp.compile_tex_path else None
        bib_path = inp.workspace / "references.bib"
        final_tex = embed_or_ship_bibliography(
            inp.tex,
            bbl_path=bbl_path if bbl_path and bbl_path.exists() else None,
            bib_path=bib_path if bib_path.exists() else None,
            ship_bib_alongside=inp.ship_bib_alongside,
            safe_id=safe,
            solutions_dir=solutions_dir,
        )
        path = solutions_dir / f"{safe}.tex"
        path.write_text(final_tex, encoding="utf-8")
        return self.Outputs(answer_tex=path)


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_id(problem_id: str) -> str:
    cleaned = _SAFE_ID_RE.sub("_", problem_id).strip("._")
    return cleaned or "problem"


def pwc_workspace_path(root_workdir: Path, problem_id: str) -> Path:
    return root_workdir / "pwc_workspaces" / safe_id(problem_id)


def bootstrap_workspace(workspace: Path, *, problem_text: str | None = None) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    if problem_text is not None:
        problem_path = workspace / "problem.txt"
        if not problem_path.exists():
            problem_path.write_text(problem_text, encoding="utf-8")
    for name, body in EMPTY_TEMPLATES.items():
        path = workspace / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
    for dirname in AUX_DIRS:
        (workspace / dirname).mkdir(parents=True, exist_ok=True)
    (workspace / ".pwc").mkdir(parents=True, exist_ok=True)


def workspace_files(workspace: Path) -> PWCWorkspaceFiles:
    return PWCWorkspaceFiles(
        workspace=workspace,
        answer_tex=safe_read(workspace / "answer.tex"),
        research_notes_tex=safe_read(workspace / "research_notes.tex"),
        references_bib=safe_read(workspace / "references.bib"),
        report_md=safe_read(workspace / "report.md"),
        workspace_listing=workspace_listing(workspace),
        answer_path=workspace / "answer.tex",
        research_notes_path=workspace / "research_notes.tex",
        references_bib_path=workspace / "references.bib",
        report_path=workspace / "report.md",
    )


def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, FileNotFoundError):
        return ""


def workspace_has_progress(workspace: Path) -> bool:
    for name, template in EMPTY_TEMPLATES.items():
        actual = safe_read(workspace / name)
        if actual.strip() and _normalize(actual) != _normalize(template):
            return True
    return False


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def workspace_listing(workspace: Path, *, max_entries: int = 200) -> str:
    lines: list[str] = []
    for sub in AUX_DIRS:
        directory = workspace / sub
        if not directory.exists():
            continue
        entries = sorted(directory.rglob("*"))
        if not entries:
            continue
        lines.append(f"{sub}/")
        for entry in entries[:max_entries]:
            rel = entry.relative_to(directory)
            try:
                size = entry.stat().st_size if entry.is_file() else 0
            except OSError:
                size = 0
            kind = "d" if entry.is_dir() else "f"
            lines.append(f"  {kind} {size:>10d}  {rel}")
        if len(entries) > max_entries:
            lines.append(f"  ... {len(entries) - max_entries} more entries omitted")
    return "\n".join(lines) if lines else "(no auxiliary files yet)"


_BIBLIOGRAPHY_RE = re.compile(r"\\bibliography\s*\{[^}]+\}")
_BIBLIOGRAPHYSTYLE_RE = re.compile(r"\\bibliographystyle\s*\{[^}]+\}\s*\n?")
_ADDBIBRESOURCE_RE = re.compile(r"\\addbibresource\s*\{[^}]+\}")


def embed_or_ship_bibliography(
    tex_body: str,
    *,
    bbl_path: Path | None,
    bib_path: Path | None,
    ship_bib_alongside: bool,
    safe_id: str,
    solutions_dir: Path,
) -> str:
    has_classic_bib = _BIBLIOGRAPHY_RE.search(tex_body) is not None
    has_biblatex = _ADDBIBRESOURCE_RE.search(tex_body) is not None or "{biblatex}" in tex_body
    if not has_classic_bib and not has_biblatex:
        return tex_body

    use_alongside = ship_bib_alongside or has_biblatex
    have_bib = bib_path is not None and bib_path.exists() and bib_path.stat().st_size > 0
    have_bbl = bbl_path is not None and bbl_path.exists() and bbl_path.stat().st_size > 0

    if use_alongside and have_bib:
        bib_out = solutions_dir / f"{safe_id}.bib"
        shutil.copyfile(bib_path, bib_out)
        out = _BIBLIOGRAPHY_RE.sub(lambda _m: f"\\bibliography{{{safe_id}}}", tex_body)
        return _ADDBIBRESOURCE_RE.sub(lambda _m: f"\\addbibresource{{{safe_id}.bib}}", out)

    if has_classic_bib and have_bbl:
        bbl_text = bbl_path.read_text(encoding="utf-8", errors="replace")
        out = _BIBLIOGRAPHYSTYLE_RE.sub("", tex_body)
        return _BIBLIOGRAPHY_RE.sub(lambda _m: bbl_text, out)

    return tex_body


__all__ = [
    "AUX_DIRS",
    "CANONICAL_FILES",
    "EMPTY_TEMPLATES",
    "PWCStashAnswer",
    "PWCWorkspaceFiles",
    "PWCWorkspaceInit",
    "PWCWorkspaceRead",
    "PWCWorkspaceSnapshot",
    "bootstrap_workspace",
    "pwc_workspace_path",
    "safe_id",
    "safe_read",
    "workspace_files",
    "workspace_has_progress",
    "workspace_listing",
]
