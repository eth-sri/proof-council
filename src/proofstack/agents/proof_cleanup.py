from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from proofstack.agent import Agent
from proofstack.agents.dag_workflow import DAGWorkflow


FORMAL_ENVIRONMENTS = {
    "assumption",
    "claim",
    "conjecture",
    "corollary",
    "definition",
    "example",
    "hypothesis",
    "lemma",
    "observation",
    "problem",
    "proposition",
    "question",
    "remark",
    "theorem",
}
MATH_ENVIRONMENTS = {
    "align",
    "align*",
    "alignat",
    "alignat*",
    "displaymath",
    "eqnarray",
    "eqnarray*",
    "equation",
    "equation*",
    "flalign",
    "flalign*",
    "gather",
    "gather*",
    "multline",
    "multline*",
}
ARTIFACT_ENVIRONMENTS = {
    "Verbatim",
    "algorithm",
    "algorithm*",
    "figure",
    "figure*",
    "lstlisting",
    "minted",
    "picture",
    "table",
    "table*",
    "thebibliography",
    "tikzpicture",
    "verbatim",
}
VERBATIM_ENVIRONMENTS = {"Verbatim", "lstlisting", "minted", "verbatim"}


def _is_escaped(tex: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and tex[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _strip_tex_comments(tex: str) -> str:
    cleaned: list[str] = []
    index = 0
    while index < len(tex):
        verbatim_name = ""
        for name in VERBATIM_ENVIRONMENTS:
            opening = f"\\begin{{{name}}}"
            if tex.startswith(opening, index) and not _is_escaped(tex, index):
                verbatim_name = name
                break
        if verbatim_name:
            closing = f"\\end{{{verbatim_name}}}"
            closing_start = tex.find(closing, index)
            if closing_start >= 0:
                end = closing_start + len(closing)
                cleaned.append(tex[index:end])
                index = end
                continue
        if tex.startswith("\\verb", index) and not _is_escaped(tex, index):
            delimiter_index = index + 5
            if delimiter_index < len(tex) and tex[delimiter_index] == "*":
                delimiter_index += 1
            if delimiter_index < len(tex):
                delimiter = tex[delimiter_index]
                if not delimiter.isspace() and not delimiter.isalnum():
                    end = tex.find(delimiter, delimiter_index + 1)
                    if end >= 0:
                        cleaned.append(tex[index : end + 1])
                        index = end + 1
                        continue
        if tex[index] == "%" and not _is_escaped(tex, index):
            while index < len(tex) and tex[index] != "\n":
                index += 1
            continue
        cleaned.append(tex[index])
        index += 1
    return "".join(cleaned)


def _normalized_tex(fragment: str) -> str:
    return re.sub(r"\s+", " ", _strip_tex_comments(fragment)).strip()


def _mask_verbatim(tex: str) -> str:
    names = "|".join(re.escape(name) for name in sorted(VERBATIM_ENVIRONMENTS))
    pattern = re.compile(
        rf"\\begin\{{(?P<verbatim>{names})\}}.*?\\end\{{(?P=verbatim)\}}",
        re.DOTALL,
    )
    characters = list(tex)
    for match in pattern.finditer(tex):
        for index in range(match.start(), match.end()):
            if characters[index] != "\n":
                characters[index] = " "
    source = "".join(characters)
    characters = list(source)
    index = 0
    while index < len(source):
        if source.startswith("\\verb", index) and not _is_escaped(source, index):
            delimiter_index = index + 5
            if delimiter_index < len(source) and source[delimiter_index] == "*":
                delimiter_index += 1
            if delimiter_index < len(source):
                delimiter = source[delimiter_index]
                if not delimiter.isspace() and not delimiter.isalnum():
                    end = source.find(delimiter, delimiter_index + 1)
                    if end >= 0:
                        for masked in range(index, end + 1):
                            if characters[masked] != "\n":
                                characters[masked] = " "
                        index = end + 1
                        continue
        index += 1
    return "".join(characters)


def _environment_blocks(tex: str, names: set[str]) -> Counter[str]:
    if not names:
        return Counter()
    alternatives = "|".join(
        sorted((re.escape(name) for name in names), key=len, reverse=True)
    )
    pattern = re.compile(
        rf"\\begin\{{({alternatives})\}}.*?\\end\{{\1\}}", re.DOTALL
    )
    source = _strip_tex_comments(tex)
    return Counter(_normalized_tex(match.group(0)) for match in pattern.finditer(source))


def _exact_environment_blocks(tex: str, names: set[str]) -> Counter[str]:
    if not names:
        return Counter()
    alternatives = "|".join(
        sorted((re.escape(name) for name in names), key=len, reverse=True)
    )
    pattern = re.compile(
        rf"\\begin\{{({alternatives})\}}.*?\\end\{{\1\}}", re.DOTALL
    )
    source = _strip_tex_comments(tex)
    return Counter(match.group(0) for match in pattern.finditer(source))


def _inline_verbatim_commands(tex: str) -> Counter[str]:
    source = _strip_tex_comments(tex)
    commands: Counter[str] = Counter()
    index = 0
    while index < len(source):
        if not source.startswith("\\verb", index) or _is_escaped(source, index):
            index += 1
            continue
        delimiter_index = index + 5
        if delimiter_index < len(source) and source[delimiter_index] == "*":
            delimiter_index += 1
        if delimiter_index >= len(source):
            index += 1
            continue
        delimiter = source[delimiter_index]
        if delimiter.isspace() or delimiter.isalnum():
            index += 1
            continue
        end = source.find(delimiter, delimiter_index + 1)
        if end < 0:
            index += 1
            continue
        commands[source[index : end + 1]] += 1
        index = end + 1
    return commands


def _math_fragments(tex: str) -> Counter[str]:
    source = _mask_verbatim(_strip_tex_comments(tex))
    fragments: Counter[str] = Counter()
    index = 0
    while index < len(source):
        opening = ""
        closing = ""
        if source.startswith("\\(", index) and not _is_escaped(source, index):
            opening, closing = "\\(", "\\)"
        elif source.startswith("\\[", index) and not _is_escaped(source, index):
            opening, closing = "\\[", "\\]"
        elif source.startswith("$$", index) and not _is_escaped(source, index):
            opening = closing = "$$"
        elif source[index] == "$" and not _is_escaped(source, index):
            opening = closing = "$"
        if not opening:
            index += 1
            continue
        search = index + len(opening)
        end = -1
        while search < len(source):
            if source.startswith(closing, search) and not _is_escaped(source, search):
                if closing != "$" or not (
                    (search and source[search - 1] == "$")
                    or (search + 1 < len(source) and source[search + 1] == "$")
                ):
                    end = search
                    break
            search += 1
        if end < 0:
            index += len(opening)
            continue
        fragments[_normalized_tex(source[index + len(opening) : end])] += 1
        index = end + len(closing)
    return fragments


def _protected_commands(tex: str) -> Counter[str]:
    command = re.compile(
        r"\\(?:auto|c|C|eq|name|page|v|V)?ref\*?(?:\[[^\]]*\])?\{[^{}]*\}"
        r"|\\label\{[^{}]*\}"
        r"|\\(?:auto|foot|full|paren|smart|super|text)?cite\*?"
        r"(?:\[[^\]]*\]\s*){0,2}\{[^{}]*\}"
        r"|\\(?:citealp|citeauthor|citep|citet|citeyear|citeyearpar|nocite)\*?"
        r"(?:\[[^\]]*\]\s*){0,2}\{[^{}]*\}"
        r"|\\(?:addbibresource|bibliography|bibliographystyle|include|includegraphics|input|url)"
        r"(?:\[[^\]]*\])?\{[^{}]*\}"
        r"|\\href\{[^{}]*\}(?:\{[^{}]*\})?"
        r"|\\hyperref\[[^\]]*\](?:\{[^{}]*\})?"
        r"|\\printbibliography(?:\[[^\]]*\])?"
    )
    source = _strip_tex_comments(tex)
    return Counter(_normalized_tex(match.group(0)) for match in command.finditer(source))


def _citation_contexts(tex: str) -> Counter[str]:
    citation = re.compile(
        r"\\(?:auto|foot|full|paren|smart|super|text)?cite\*?"
        r"(?:\[[^\]]*\]\s*){0,2}\{[^{}]*\}"
        r"|\\(?:citealp|citeauthor|citep|citet|citeyear|citeyearpar|nocite)\*?"
        r"(?:\[[^\]]*\]\s*){0,2}\{[^{}]*\}"
    )
    source = _strip_tex_comments(tex)
    contexts: Counter[str] = Counter()
    for match in citation.finditer(source):
        punctuation_start = max(source.rfind(mark, 0, match.start()) for mark in ".!?") + 1
        paragraph_start = source.rfind("\n\n", 0, match.start()) + 2
        start = max(punctuation_start, paragraph_start)
        endings = [source.find(mark, match.end()) for mark in ".!?"]
        endings = [end for end in endings if end >= 0]
        end = min(endings) + 1 if endings else len(source)
        contexts[_normalized_tex(source[start:end])] += 1
    return contexts


def _balanced_group_end(tex: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    index = start
    while index < len(tex):
        character = tex[index]
        escaped = _is_escaped(tex, index)
        if not escaped and character == opening:
            depth += 1
        elif not escaped and character == closing:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return start


def _command_definitions(tex: str) -> Counter[str]:
    start_pattern = re.compile(
        r"\\(?:DeclareDocumentCommand|DeclareMathOperator|DeclareRobustCommand|"
        r"NewDocumentCommand|ProvideDocumentCommand|RenewDocumentCommand|"
        r"declaretheorem|newcommand|newenvironment|newtheorem|providecommand|"
        r"renewcommand|renewenvironment)\*?"
        r"|\\(?:(?:global|long|outer)\s*\\)*(?:def|edef|gdef|xdef)\s*\\[A-Za-z@]+"
    )
    definitions: Counter[str] = Counter()
    source = _strip_tex_comments(tex)
    for match in start_pattern.finditer(source):
        end = match.end()
        if re.search(r"(?:e|g|x)?def\s*\\[A-Za-z@]+$", match.group(0)):
            while end < len(source) and source[end] != "{":
                end += 1
        groups = 0
        while end < len(source):
            while end < len(source) and source[end].isspace():
                end += 1
            if end >= len(source) or source[end] not in "[{":
                break
            opening = source[end]
            closing = "]" if opening == "[" else "}"
            group_end = _balanced_group_end(source, end, opening, closing)
            if group_end == end:
                break
            end = group_end
            groups += 1
        if groups:
            definitions[_normalized_tex(source[match.start() : end])] += 1
    return definitions


def _preserves_tex_invariants(baseline: str, candidate: str) -> bool:
    baseline_without_comments = _strip_tex_comments(baseline)
    declared = set(
        re.findall(r"\\newtheorem\*?\s*\{([^{}]+)\}", baseline_without_comments)
    )
    declared.update(
        re.findall(
            r"\\declaretheorem\*?(?:\[[^\]]*\])?\s*\{([^{}]+)\}",
            baseline_without_comments,
        )
    )
    formal_names = FORMAL_ENVIRONMENTS | declared
    pairs = (
        (_environment_blocks(baseline, MATH_ENVIRONMENTS), _environment_blocks(candidate, MATH_ENVIRONMENTS)),
        (_environment_blocks(baseline, formal_names), _environment_blocks(candidate, formal_names)),
        (_environment_blocks(baseline, ARTIFACT_ENVIRONMENTS), _environment_blocks(candidate, ARTIFACT_ENVIRONMENTS)),
        (_exact_environment_blocks(baseline, VERBATIM_ENVIRONMENTS), _exact_environment_blocks(candidate, VERBATIM_ENVIRONMENTS)),
        (_inline_verbatim_commands(baseline), _inline_verbatim_commands(candidate)),
        (_protected_commands(baseline), _protected_commands(candidate)),
        (_citation_contexts(baseline), _citation_contexts(candidate)),
        (_command_definitions(baseline), _command_definitions(candidate)),
        (_math_fragments(baseline), _math_fragments(candidate)),
    )
    return all(left == right for left, right in pairs)


def _has_unresolved_tex_warnings(log: str) -> bool:
    lowered = log.lower()
    return any(
        warning in lowered
        for warning in (
            "there were undefined references",
            "there were undefined citations",
            "rerun to get cross-references right",
            "rerun to get outlines right",
            "multiply defined",
            "destination with the same identifier",
            "missing character:",
            "overfull \\hbox",
            "overfull \\vbox",
            "please (re)run bibtex",
            "please (re)run biber",
        )
    ) or ("undefined" in lowered and ("citation" in lowered or "reference" in lowered))


def _standalone_tex_compiles(tex: str, run_dir: Path) -> bool:
    compiler = shutil.which("pdflatex")
    if compiler is None:
        return False
    try:
        with tempfile.TemporaryDirectory(prefix=".proof-cleanup-", dir=run_dir) as raw_dir:
            workdir = Path(raw_dir)
            source = workdir / "cleaned_proof.tex"
            source.write_text(tex, encoding="utf-8")
            command = [
                compiler,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                source.name,
            ]
            for _ in range(3):
                result = subprocess.run(
                    command,
                    cwd=workdir,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                )
                if result.returncode:
                    return False
            pdf = workdir / "cleaned_proof.pdf"
            log = workdir / "cleaned_proof.log"
            if not pdf.is_file() or not pdf.stat().st_size or not log.is_file():
                return False
            return not _has_unresolved_tex_warnings(
                log.read_text(encoding="utf-8", errors="replace")
            )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _safe_problem_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "proof"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class CleanupInputBlock(Agent):
    description = "Select proof text or load a preceding workflow's proof artifact."
    execution_mode = "deterministic_tool"

    class Inputs(BaseModel):
        problem: str
        solution: Path | str | None = None

    class Outputs(BaseModel):
        proof: str

    async def run(self, inp: Inputs) -> Outputs:
        proof = inp.problem
        if isinstance(inp.solution, Path):
            path = inp.solution.resolve()
            try:
                path.relative_to(self.ctx.root_workdir.resolve())
            except ValueError as exc:
                raise ValueError("solution path escapes the run directory") from exc
            try:
                proof = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"solution is not a readable UTF-8 proof: {exc}") from exc
        elif inp.solution is not None and inp.solution.strip():
            proof = inp.solution
        if not proof.strip():
            raise ValueError("proof is empty")
        return self.Outputs(proof=proof)


class CleanupAuditCompletionBlock(Agent):
    description = "Require a completed, nonempty scholarship audit."
    execution_mode = "deterministic_tool"
    cache_enabled = False

    class Inputs(BaseModel):
        workspace: str
        audit: str
        status: str
        completion_receipt: dict[str, Any]

    class Outputs(BaseModel):
        workspace: str
        audit: str

    async def run(self, inp: Inputs) -> Outputs:
        if inp.status != "done":
            raise RuntimeError(f"scholarship auditor returned status {inp.status!r}")
        if inp.completion_receipt.get("status") != "done":
            raise RuntimeError("scholarship auditor did not write a completion receipt")
        if not inp.audit.strip():
            raise RuntimeError("scholarship auditor produced an empty audit")
        return self.Outputs(workspace=inp.workspace, audit=inp.audit)


class CleanupEditCompletionBlock(Agent):
    description = "Require a completed, nonempty scholarship edit."
    execution_mode = "deterministic_tool"
    cache_enabled = False

    class Inputs(BaseModel):
        workspace: str
        cleaned_proof: str
        status: str
        completion_receipt: dict[str, Any]

    class Outputs(BaseModel):
        workspace: str
        cleaned_proof: str

    async def run(self, inp: Inputs) -> Outputs:
        if inp.status != "done":
            raise RuntimeError(f"scholarship editor returned status {inp.status!r}")
        if inp.completion_receipt.get("status") != "done":
            raise RuntimeError("scholarship editor did not write a completion receipt")
        if not inp.cleaned_proof.strip():
            raise RuntimeError("scholarship editor produced an empty proof")
        return self.Outputs(
            workspace=inp.workspace,
            cleaned_proof=inp.cleaned_proof,
        )


class CleanupPublicationBlock(Agent):
    description = "Validate TeX preservation and compilation, then publish."
    execution_mode = "deterministic_tool"
    cache_enabled = False

    class Inputs(BaseModel):
        problem_id: str
        scholarship_tex: str
        cleaned_proof: str
        organization_completed: bool = False

    cache_enabled = False

    class Outputs(BaseModel):
        problem_id: str
        cleaned_proof: str
        solution_tex: str
        compiled: bool

    async def run(self, inp: Inputs) -> Outputs:
        if not inp.organization_completed:
            raise RuntimeError("organization writer did not complete")
        if not inp.scholarship_tex.strip() or not inp.cleaned_proof.strip():
            raise RuntimeError("workflow produced an empty proof")
        if not _preserves_tex_invariants(inp.scholarship_tex, inp.cleaned_proof):
            raise RuntimeError("organization writer changed protected mathematical material")
        compiled = await asyncio.to_thread(
            _standalone_tex_compiles, inp.cleaned_proof, self.workdir
        )
        if not compiled:
            raise RuntimeError("cleaned proof did not compile cleanly")

        problem_id = _safe_problem_id(inp.problem_id)
        solution = self.ctx.root_workdir / "solutions" / f"{problem_id}.tex"
        _atomic_write(solution, inp.cleaned_proof.encode("utf-8"))
        return self.Outputs(
            problem_id=problem_id,
            cleaned_proof=inp.cleaned_proof,
            solution_tex=str(solution),
            compiled=True,
        )


class ProofCleanupDAGWorkflow(DAGWorkflow):
    description = "Audit, repair, and copy-edit one standalone LaTeX proof."
    cache_enabled = False

    Inputs = DAGWorkflow.Inputs
    Outputs = CleanupPublicationBlock.Outputs

    def __init__(self, ctx, **kwargs: Any):
        super().__init__(ctx, **kwargs)
        if self.component_config.get("unsafe_bypass_codex_sandbox") or sys.platform == "win32":
            for name in ("scholarship_auditor", "proof_editor", "organization_editor"):
                config = dict(ctx.component_configs.get(name, {}))
                config["codex_sandbox"] = "bypass"
                ctx.component_configs[name] = config

    async def _last_gasp(self, inp, state: dict[str, Any], error: Exception):
        await self.events.emit(
            "workflow.last_gasp_disabled",
            {"type": type(error).__name__, "msg": str(error)},
        )
        raise error

    def _stash_solution(self, problem_id: str, tex_body: str) -> Path:
        raise RuntimeError("unvalidated solution stashing is disabled")

__all__ = [
    "CleanupAuditCompletionBlock",
    "CleanupEditCompletionBlock",
    "CleanupInputBlock",
    "CleanupPublicationBlock",
    "ProofCleanupDAGWorkflow",
]
