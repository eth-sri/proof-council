"""ACWorkflow — Author/Critic iterative loop with optional Advisory Council.

Shape::

  Round 0:                          Author --> Critic (fresh)
  Rounds k = 1..N:                  Author --> parallel(Critic, Council if requested)
                                    + forced-fresh promotion on stateful-agree
  After loop:                       compile-check [+ optional final_critic]

The Critic runs in two modes:

- **fresh**: brand-new instance, no prior conversation. Used at round
  0, at K-reset boundaries (every ``full_critic_interval`` rounds),
  on the terminal in-loop review, and on forced-fresh promotion.
- **stateful**: continuation of an existing instance, with the prior
  conversation passed in via ``prior_messages``.

Early-stop logic (when ``author.ready=True``):

- last Critic was *fresh* and answer_ready=True → ship (subject to
  deterministic gate)
- last Critic was *fresh* and answer_ready=False → Author turn, loop
  continues
- last Critic was *stateful* and answer_ready=True → run a forced-fresh
  Critic on the same files (no author turn between); if it agrees,
  ship; if not, Author turn with **both** reviews in context
- last Critic was *stateful* and answer_ready=False → Author turn,
  loop continues

The Author is stateless across rounds: each call sees only the
inputs pasted into its prompt (problem, current files, latest
critique, latest council replies). Anything the Author wants to
carry forward should be written into ``research_notes.tex``.

Reuses ``BudgetTracker``, the last-gasp salvage path, and main's
``embed_or_ship_bibliography`` helper (from ``proofstack.agents.pwc.
workspace``). The previous ``LatexCompileFix`` sub-agent was removed
in main's YAML-first restructuring; AC uses an inline
``_simple_compile_latex`` helper (single-shot, no model-driven repair).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self

from pydantic import BaseModel, Field, model_validator

from proofstack.agent import Agent
from proofstack.agents.ac.author import Author
from proofstack.agents.ac.blocks import CANONICAL_FILES
from proofstack.agents.ac.compute import (
    DEFAULT_COST_CONFIG as DEFAULT_COMPUTE_COST_CONFIG,
)
from proofstack.agents.ac.compute import (
    DEFAULT_MODEL as DEFAULT_COMPUTE_MODEL,
)
from proofstack.agents.ac.compute import (
    DEFAULT_REASONING_EFFORT as DEFAULT_COMPUTE_REASONING_EFFORT,
)
from proofstack.agents.ac.compute import (
    DEFAULT_HARD_TIMEOUT_S as DEFAULT_COMPUTE_HARD_TIMEOUT_S,
)
from proofstack.agents.ac.compute import (
    DEFAULT_SOFT_TIMEOUT_S as DEFAULT_COMPUTE_SOFT_TIMEOUT_S,
)
from proofstack.agents.ac.compute import (
    Compute,
    render_compute_reply_for_author,
)
from proofstack.agents.ac.council import (
    Council,
    CouncilReply,
    render_council_replies_for_author,
)
from proofstack.agents.ac.critic import ACCritic
from proofstack.agents.dag_workflow import DAGWorkflow, _bare_wrap
from proofstack.agents.pwc.workspace import embed_or_ship_bibliography
from proofstack.budget import BudgetExhausted
from proofstack.latex_contract import (
    DEFAULT_FIRSTPROOF_PAGE_LIMIT,
    normalize_firstproof_latex,
)


# --- Local LaTeX compile helper -------------------------------------------
#
# Replaces the previous ``LatexCompileFix`` sub-agent (removed from main
# during its YAML-first restructuring). Single-shot compile only: no
# model-driven repair. If pdflatex fails, the workflow ships the .tex
# as-is and records ``compiled=False`` rather than burning model budget
# trying to mutate the deliverable Critic just signed off on.


@dataclass
class _CompileResult:
    """Lean ``LatexCompileFix.Outputs`` replacement.

    Fields:
      - ``tex``      — the final tex content (possibly bare-wrapped).
      - ``tex_path`` — historical field; always ``None`` now that the
        compile work happens inside a transient ``TemporaryDirectory``.
        Kept on the dataclass for callers that still pass it through.
      - ``pdf_path`` — same: always ``None`` post-cleanup.
      - ``bbl_path`` — the bibliography file's content survives the
        temp-dir cleanup via a ``NamedTemporaryFile(delete=False,
        suffix='.bbl')`` copy; consumers (``_stash_answer`` →
        ``embed_or_ship_bibliography``) read this. ``None`` when the
        compile did not produce a .bbl.
      - ``compiled`` — True iff pdflatex produced a PDF from at least
        one successful pass. Undefined-reference warnings are not fatal.
      - ``pages``    — page count of the produced PDF, 0 if not compiled.
    """

    tex: str
    tex_path: Path | None
    pdf_path: Path | None
    compiled: bool
    pages: int
    bbl_path: Path | None = None
    compile_log: str = ""
    normalization_removals: list[str] = field(default_factory=list)


def _passthrough_compile(tex_text: str) -> _CompileResult:
    """Used when we cannot or do not want to compile (e.g. budget
    exhausted before the final compile step). Ships the tex as-is."""
    return _CompileResult(
        tex=tex_text or "",
        tex_path=None,
        pdf_path=None,
        compiled=False,
        pages=0,
    )


_BIBLIOGRAPHY_DIRECTIVE_RE = re.compile(r"\\bibliography\s*\{[^}]+\}")
_ADDBIBRESOURCE_DIRECTIVE_RE = re.compile(r"\\addbibresource\s*\{[^}]+\}")
_USEPACKAGE_RE = re.compile(r"\\usepackage\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
_LATEX_REPAIR_PACKAGES: tuple[tuple[str, re.Pattern[str], tuple[str, ...]], ...] = (
    ("graphicx", re.compile(r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{"), ("graphicx",)),
    ("hyperref", re.compile(r"\\(?:url|href)\s*\{"), ("hyperref", "url")),
    ("xcolor", re.compile(r"\\(?:textcolor|color)\s*(?:\[[^\]]*\])?\s*\{"), ("xcolor", "color")),
    ("cleveref", re.compile(r"\\[cC](?:ref|pageref)\s*\{"), ("cleveref",)),
)


@dataclass
class _LatexRun:
    returncode: int
    timed_out: bool = False
    missing_binary: bool = False


def _latex_packages(tex: str) -> set[str]:
    packages: set[str] = set()
    for match in _USEPACKAGE_RE.finditer(tex):
        packages.update(name.strip() for name in match.group(1).split(",") if name.strip())
    return packages


def _insert_latex_preamble_lines(tex: str, lines: list[str]) -> str:
    if not lines:
        return tex
    marker = r"\begin{document}"
    idx = tex.find(marker)
    insert = "\n".join(lines) + "\n"
    if idx < 0:
        return tex.rstrip() + "\n" + insert
    return tex[:idx] + insert + tex[idx:]


def _repair_latex_for_compile(tex: str) -> str:
    repaired = re.sub(
        r"^[ \t]*\\geometry\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}[ \t]*\n?",
        "",
        tex,
        flags=re.MULTILINE,
    )
    repaired = re.sub(
        r"^[ \t]*\\(?:double|onehalf|single)spacing\b[ \t]*\n?",
        "",
        repaired,
        flags=re.MULTILINE,
    )
    repaired = re.sub(
        r"^[ \t]*\\setstretch\s*\{[^{}]*\}[ \t]*\n?",
        "",
        repaired,
        flags=re.MULTILINE,
    )
    packages = _latex_packages(repaired)
    insertions: list[str] = []
    for package, trigger, alternatives in _LATEX_REPAIR_PACKAGES:
        if trigger.search(repaired) and not any(existing in packages for existing in alternatives):
            insertions.append(f"\\usepackage{{{package}}}")
            packages.add(package)
    return _insert_latex_preamble_lines(repaired, insertions)


def _count_pdf_pages(pdf_path: Path) -> int:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import fitz  # type: ignore[import-not-found]

        with fitz.open(pdf_path) as doc:
            return int(doc.page_count)
    except Exception:
        try:
            proc = subprocess.run(
                ["pdfinfo", str(pdf_path)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode == 0:
                match = re.search(r"^Pages:\s*(\d+)\s*$", proc.stdout, re.MULTILINE)
                if match:
                    return int(match.group(1))
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        try:
            data = pdf_path.read_bytes()
        except OSError:
            return 0
        return len(re.findall(rb"/Type\s*/Page(?!s)", data))


def _simple_compile_latex(
    tex_body: str,
    *,
    bib_path: Path | None,
    page_limit: int,
    is_full_document: bool = True,
) -> _CompileResult:
    """Compile a single tex body to PDF and return a ``_CompileResult``.

    Single attempt; no model-driven repair. If ``tex_body`` is not a
    full document (no ``\\documentclass``), we wrap it with the
    standard preamble via ``_bare_wrap``. If the tex references an
    external bib (via ``\\bibliography`` or ``\\addbibresource``) and a
    ``bib_path`` was provided, we run the standard pdflatex / bibtex /
    pdflatex / pdflatex dance; otherwise three pdflatex passes for
    cross-references.

    Returns a result whose ``compiled`` flag is True when a PDF exists
    from a successful pdflatex pass. Undefined references remain in the
    log for diagnosis but do not turn a real answer into a rejected
    artifact.
    """
    raw_tex = tex_body if is_full_document and r"\documentclass" in tex_body else _bare_wrap(tex_body or "")
    normalization_removals: list[str] = []
    final_tex = normalize_firstproof_latex(raw_tex, removals=normalization_removals)
    final_tex = _repair_latex_for_compile(final_tex)

    # Run the whole compile inside a ``TemporaryDirectory`` so .aux,
    # .log, .pdf, intermediate .toc / .bbl etc. are cleaned up
    # automatically. On a 24-hour run we'd otherwise leak many MB per
    # compile into /tmp; small individually, but the deterministic gate
    # alone fires several times per problem per round.
    pages = 0
    compiled = False
    bbl_persisted_path: Path | None = None
    compile_log = ""
    with tempfile.TemporaryDirectory(prefix="ac_compile_") as work_str:
        work = Path(work_str)
        tex_path = work / "main.tex"
        tex_path.write_text(final_tex, encoding="utf-8")
        if bib_path is not None and bib_path.exists() and bib_path.stat().st_size > 0:
            shutil.copyfile(bib_path, work / "references.bib")

        uses_biblatex = _ADDBIBRESOURCE_DIRECTIVE_RE.search(final_tex) is not None
        uses_bibtex = _BIBLIOGRAPHY_DIRECTIVE_RE.search(final_tex) is not None and not uses_biblatex
        bib_dance = (work / "references.bib").exists() and (uses_bibtex or uses_biblatex)

        def _run(cmd: list[str], timeout: int = 300) -> _LatexRun:
            try:
                r = subprocess.run(cmd, cwd=work, capture_output=True, timeout=timeout)
                return _LatexRun(returncode=r.returncode)
            except subprocess.TimeoutExpired:
                return _LatexRun(returncode=124, timed_out=True)
            except FileNotFoundError:
                return _LatexRun(returncode=127, missing_binary=True)

        rc = _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
        pdf_candidate = work / "main.pdf"
        had_successful_pdf = rc.returncode == 0 and pdf_candidate.exists()
        timed_out = rc.timed_out
        if bib_dance and rc.returncode == 0:
            bib_tool = "biber" if uses_biblatex else "bibtex"
            target = "main"
            _run([bib_tool, target], timeout=60)
            rc = _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
            had_successful_pdf = had_successful_pdf or (rc.returncode == 0 and pdf_candidate.exists())
            timed_out = timed_out or rc.timed_out
        remaining_pdflatex_passes = 1 if bib_dance else 2
        for _ in range(remaining_pdflatex_passes):
            if rc.returncode != 0:
                break
            rc = _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
            had_successful_pdf = had_successful_pdf or (rc.returncode == 0 and pdf_candidate.exists())
            timed_out = timed_out or rc.timed_out

        log_path = work / "main.log"
        if log_path.exists():
            try:
                compile_log = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                compile_log = ""

        pdf_path: Path | None = work / "main.pdf"
        if not pdf_path.exists():
            pdf_path = None

        compiled = pdf_path is not None and (
            rc.returncode == 0 or (timed_out and had_successful_pdf)
        )

        if compiled:
            pages = _count_pdf_pages(pdf_path)

        # Preserve the .bbl into a stable location BEFORE the
        # TemporaryDirectory cleanup runs. ``_stash_answer`` ->
        # ``embed_or_ship_bibliography`` reads the file path's contents
        # to inline the bibliography into the final shipped .tex.
        bbl_src = work / "main.bbl"
        if bbl_src.exists() and bbl_src.stat().st_size > 0:
            try:
                fd, dst = tempfile.mkstemp(prefix="ac_bbl_", suffix=".bbl")
                os.close(fd)
                shutil.copyfile(bbl_src, dst)
                bbl_persisted_path = Path(dst)
            except OSError:
                bbl_persisted_path = None

    return _CompileResult(
        tex=final_tex,
        tex_path=None,
        pdf_path=None,
        compiled=compiled,
        pages=pages,
        bbl_path=bbl_persisted_path,
        compile_log=compile_log,
        normalization_removals=normalization_removals,
    )


DEFAULT_PAGE_LIMIT = DEFAULT_FIRSTPROOF_PAGE_LIMIT
DEFAULT_COUNCIL_MODELS: tuple[str, ...] = (
    "models/openai/gpt-56-sol-pro",
    "models/anthropic/opus_47_max",
    "models/gemini/gemini-31-pro",
)


class ACWorkflow(Agent):
    """Iterative Author/Critic loop with optional Advisory Council."""

    description: ClassVar[str] = (
        "Iterative Author/Critic loop with optional Advisory Council."
    )
    execution_mode: ClassVar[str] = "workflow"
    cache_enabled: ClassVar[bool] = False

    DEFAULT_N_ROUNDS: ClassVar[int] = 5
    DEFAULT_FULL_CRITIC_INTERVAL: ClassVar[int] = 3

    class Inputs(BaseModel):
        problem: str
        problem_id: str
        n_rounds: int = Field(default=5, ge=1, le=500)
        full_critic_interval: int = Field(default=3, ge=1, le=20)
        enable_council: bool = True
        council_models: list[str] = Field(default_factory=lambda: list(DEFAULT_COUNCIL_MODELS))
        # Reserved for a future Pro-vetted summarizer of council replies;
        # ignored when None (current default — raw replies pass through).
        synthesizer_model: str | None = None
        # Out-of-band codex CLI worker invoked by Author
        # ``<compute_agent>`` blocks. Fans out in parallel with Critic
        # and Council; reply lands in the next Author turn (text +
        # workspace zip attachment).
        enable_compute: bool = True
        compute_model: str = DEFAULT_COMPUTE_MODEL
        compute_reasoning_effort: str = DEFAULT_COMPUTE_REASONING_EFFORT
        compute_cost_config: str = DEFAULT_COMPUTE_COST_CONFIG
        compute_soft_timeout_s: int = Field(default=DEFAULT_COMPUTE_SOFT_TIMEOUT_S, ge=0)
        compute_hard_timeout_s: int = Field(default=DEFAULT_COMPUTE_HARD_TIMEOUT_S, ge=1)
        # ``docker`` (default) or ``subprocess`` (host CLI; needed when
        # the pwc sandbox image is unavailable). Forwarded to
        # ``Compute.Inputs.sandbox_backend``.
        compute_sandbox_backend: str = "docker"
        compute_docker_image: str = "proofstack-pwc-sandbox:latest"
        # Codex sandbox flag: ``auto`` | ``workspace-write`` |
        # ``docker-bypass`` | ``none``. ``auto`` resolves correctly
        # based on the backend.
        compute_codex_sandbox: str = "auto"
        # Dev-only post-loop audit. False by default: competition runs
        # use the in-loop terminal Critic as the canonical final review.
        enable_final_critic: bool = False
        page_limit: int = Field(default=DEFAULT_PAGE_LIMIT, ge=1, le=200)
        ship_bib_alongside: bool = False
        # Set by run_workflow.py's --restart-from path. When true, the
        # workflow restores the latest AC checkpoint from this run dir
        # and continues from the next missing step instead of starting
        # at round 0.
        resume_run: bool = False
        additional_instructions: str = ""
        # First Proof staged runs need intermediate batch boundaries to end
        # after a Critic review, not with the normal final Author-only turn.
        stop_after_review_round: bool = False

        @model_validator(mode="after")
        def validate_compute_timeouts(self) -> Self:
            if self.compute_soft_timeout_s >= self.compute_hard_timeout_s:
                raise ValueError(
                    "compute_soft_timeout_s must be less than compute_hard_timeout_s"
                )
            return self

    class Outputs(BaseModel):
        problem_id: str
        answer_tex: Path
        research_notes_tex: Path
        references_bib: Path
        compiled: bool = False
        pages: int = 0
        rounds_completed: int = 0
        early_stopped: bool = False
        last_critic_accepted: bool | None = None
        # Final-Critic results — populated only when
        # ``enable_final_critic=True``. Otherwise empty.
        final_critic_answer_ready: bool = False
        final_critic_mode_run: str = "not_run"
        final_critic_review_md: str = ""
        last_gasp: bool = False
        error: str | None = None

    def __init__(self, ctx, **kw):
        super().__init__(ctx, **kw)
        self.author = Author(ctx, parent_budget_scope=self.tracker.scope)
        self.critic = ACCritic(ctx, parent_budget_scope=self.tracker.scope)
        self.council = Council(ctx, parent_budget_scope=self.tracker.scope)
        self.compute = Compute(ctx, parent_budget_scope=self.tracker.scope)
        self._resume_cost_offset_applied = False
        # No sub-agent for latex compile any more — ``_simple_compile_latex``
        # is a plain helper. Cost-tracked as part of the workflow's own
        # wallclock; no separate budget bucket needed.

    # --- main loop ---------------------------------------------------------

    async def run(self, inp):  # type: ignore[override]
        workspace = self._workspace_path(inp.problem_id, inp.problem)
        resume_state = self._load_resume_state(workspace) if inp.resume_run else None
        if inp.resume_run and resume_state is None:
            legacy_workspace = self._workspace_path(inp.problem_id)
            legacy_state = (
                self._load_resume_state(legacy_workspace)
                if legacy_workspace != workspace
                else None
            )
            legacy_hash = (
                str(legacy_state.get("problem_hash") or "").strip()
                if isinstance(legacy_state, dict)
                else ""
            )
            legacy_problem_text = _safe_read(legacy_workspace / "problem.txt").strip()
            expected_hash = _problem_hash(inp.problem)
            if legacy_state is not None and (
                legacy_hash == expected_hash
                or legacy_problem_text == str(inp.problem or "").strip()
            ):
                workspace = legacy_workspace
                resume_state = legacy_state
        resume_stop_round = _resume_stop_round(resume_state)
        original_problem = inp.problem
        effective_problem = self._problem_with_run_notes(
            inp, resume_stop_round=resume_stop_round
        )
        if effective_problem != inp.problem:
            inp = inp.model_copy(update={"problem": effective_problem})
        # ``problem.txt`` always preserves the original statement; the
        # resume note / additional instructions the agents actually see
        # are recorded separately so the artifact reflects the live run
        # without mutating the canonical problem.
        await self._init_workspace(workspace, problem_text=original_problem)
        self._write_effective_problem(
            workspace, original=original_problem, effective=effective_problem
        )
        if resume_state is not None:
            self._restore_workspace_from_resume(workspace, resume_state)
            self._apply_resume_budget_offset()
            await self.events.emit(
                "ac.resume",
                {
                    "stop_round": resume_stop_round,
                    "next_round": resume_state.get("next_round"),
                    "awaiting_review_round": resume_state.get("awaiting_review_round"),
                    "source": resume_state.get("source", "checkpoint"),
                },
            )

        review_history: list[ACCritic.Outputs] = [
            ACCritic.Outputs.model_validate(raw)
            for raw in (resume_state or {}).get("review_history", [])
            if isinstance(raw, dict)
        ]
        # Stateful Critic state. ``critic_conversation`` is the running
        # message list (alternating user/assistant) of the current
        # Critic instance; ``critic_instance_turn`` is the 0-indexed
        # turn within that instance. Reset to ``[]`` / ``0`` on K-reset,
        # last-critic forced-fresh, or forced-fresh promotion.
        critic_conversation: list[dict] = list(
            (resume_state or {}).get("critic_conversation", [])
        )
        critic_instance_turn = int(
            (resume_state or {}).get("critic_instance_turn", 0) or 0
        )
        pending_council_text = str(
            (resume_state or {}).get("pending_council_text", "") or ""
        )
        # Compute worker reply pending for the next Author turn. Cleared
        # at the start of each Author call after being consumed.
        pending_compute_text = str(
            (resume_state or {}).get("pending_compute_text", "") or ""
        )
        pending_compute_zip_path = self._decode_run_path(
            (resume_state or {}).get("pending_compute_zip_path")
        )
        pending_workflow_feedback = str(
            (resume_state or {}).get("pending_workflow_feedback", "") or ""
        )
        # When non-empty, overrides ``review_history[-1].review_md`` as
        # the prev_critique fed to the next Author. Used in two cases:
        # forced-fresh disagreement (Author sees both reviews) and
        # deterministic-gate rejection (Author sees workflow reasons too).
        pending_critique = str(
            (resume_state or {}).get("pending_critique", "") or ""
        )
        last_round_run = _state_int(resume_state, "last_round_run", -1)
        next_round = _state_int(resume_state, "next_round", 0)
        early_stopped = bool((resume_state or {}).get("early_stopped", False))
        awaiting_review_round_raw = (resume_state or {}).get("awaiting_review_round")
        awaiting_review_round = (
            int(awaiting_review_round_raw)
            if awaiting_review_round_raw is not None
            else None
        )
        awaiting_review_kind = str(
            (resume_state or {}).get("awaiting_review_kind", "") or ""
        )
        awaiting_author_raw = (resume_state or {}).get("awaiting_author")
        awaiting_author = (
            Author.Outputs.model_validate(awaiting_author_raw)
            if isinstance(awaiting_author_raw, dict)
            else None
        )
        resume_ready_to_noop = self._resume_ready_to_noop(inp, resume_state)
        awaiting_finalization = bool(
            (resume_state or {}).get("awaiting_finalization", False)
        ) or (
            resume_state is not None
            and early_stopped
            and not resume_ready_to_noop
        )

        if self._resume_has_no_new_work(
            inp=inp,
            state=resume_state,
            next_round=next_round,
            awaiting_review_round=awaiting_review_round,
            awaiting_finalization=awaiting_finalization,
            early_stopped=early_stopped,
        ):
            await self.events.emit(
                "ac.resume_noop",
                {
                    "reason": self._resume_noop_reason(
                        inp=inp,
                        state=resume_state,
                        next_round=next_round,
                        early_stopped=early_stopped,
                    ),
                    "next_round": next_round,
                    "n_rounds": inp.n_rounds,
                    "early_stopped": early_stopped,
                },
            )
            return self._outputs_from_existing_resume(
                inp=inp,
                workspace=workspace,
                state=resume_state or {},
            )

        previous_round_bound = _state_int(
            resume_state, "n_rounds_at_checkpoint", inp.n_rounds
        )
        has_new_instruction = bool(
            str(getattr(inp, "additional_instructions", "") or "").strip()
        )
        finalize_without_rounds = (
            awaiting_finalization
            and not has_new_instruction
            and inp.n_rounds <= previous_round_bound
        )
        if awaiting_finalization and not finalize_without_rounds:
            early_stopped = False

        try:
            if awaiting_review_round is not None and awaiting_author is not None:
                await self.events.emit(
                    "ac.resume_missing_review_start",
                    {
                        "round": awaiting_review_round,
                        "kind": awaiting_review_kind or "round_review",
                        "n_rounds": inp.n_rounds,
                    },
                )
                mode, critic_conversation, critic_instance_turn, omit_thinking = (
                    self._critic_mode_for_resume_review(
                        inp=inp,
                        round=awaiting_review_round,
                        critic_conversation=critic_conversation,
                        critic_instance_turn=critic_instance_turn,
                        awaiting_review_kind=awaiting_review_kind,
                    )
                )
                requested_council = (
                    inp.enable_council
                    and awaiting_author.council_question
                    and bool(inp.council_models)
                )
                requested_compute = (
                    inp.enable_compute
                    and bool(awaiting_author.compute_instructions)
                )
                terminal_auxiliary_blocked, kinds, pending_terminal_auxiliary = (
                    self._terminal_auxiliary_decision(
                        inp=inp,
                        round=awaiting_review_round,
                        requested_council=bool(requested_council),
                        requested_compute=bool(requested_compute),
                    )
                )
                if terminal_auxiliary_blocked:
                    await self.events.emit(
                        "ac.terminal_auxiliary_suppressed",
                        {"round": awaiting_review_round, "kinds": kinds},
                    )
                    pending_critique = pending_terminal_auxiliary
                run_council = bool(requested_council) and not terminal_auxiliary_blocked
                run_compute = bool(requested_compute) and not terminal_auxiliary_blocked
                review_k, council_replies, compute_out = await self._gather_critic_council(
                    inp=inp, workspace=workspace,
                    author_k=awaiting_author,
                    critic_conversation=critic_conversation,
                    mode=mode,
                    round=awaiting_review_round,
                    run_council=run_council,
                    run_compute=run_compute,
                    omit_author_thinking=omit_thinking,
                )
                if council_replies:
                    pending_council_text = render_council_replies_for_author(
                        council_replies
                    )
                if compute_out is not None:
                    pending_compute_text = render_compute_reply_for_author(
                        compute_out
                    )
                    pending_compute_zip_path = compute_out.zip_path
                    self._write_compute_artifacts(
                        workspace, compute_out, round=awaiting_review_round
                    )
                review_history.append(review_k)
                critic_conversation = list(review_k.messages_after)
                critic_instance_turn += 1
                self._write_review_artifacts(
                    workspace, review_k, round=awaiting_review_round
                )
                self._snapshot_round(
                    workspace, round=awaiting_review_round,
                    author=awaiting_author, review=review_k,
                    council_replies=council_replies,
                    compute_out=compute_out,
                )
                last_round_run = max(last_round_run, awaiting_review_round)
                next_round = max(next_round, awaiting_review_round + 1)
                if not terminal_auxiliary_blocked:
                    pending_critique = ""
                self._save_resume_state(
                    workspace,
                    inp=inp,
                    last_round_run=last_round_run,
                    next_round=next_round,
                    review_history=review_history,
                    critic_conversation=critic_conversation,
                    critic_instance_turn=critic_instance_turn,
                    pending_council_text=pending_council_text,
                    pending_compute_text=pending_compute_text,
                    pending_compute_zip_path=pending_compute_zip_path,
                    pending_critique=pending_critique,
                    early_stopped=early_stopped,
                    pending_workflow_feedback=pending_workflow_feedback,
                )

            if next_round <= 0 and not finalize_without_rounds:
                # ---- Round 0: bootstrap (Author + fresh Critic) -----------
                await self.events.emit(
                    "ac.round_start", {"round": 0, "n_rounds": inp.n_rounds}
                )
                author_0 = await self.author(
                    **self._author_inputs(
                        inp=inp, workspace=workspace,
                        prev_critique="", prev_council="",
                        workflow_feedback=pending_workflow_feedback,
                        prev_compute_response="",
                        compute_zip_path=None,
                        round=0,
                    )
                )
                self._write_files_from_author(workspace, author_0)
                self._write_author_artifacts(workspace, author_0, round=0)
                pending_workflow_feedback = await self._compile_feedback_after_author(
                    workspace, page_limit=inp.page_limit, round=0
                )
                self._save_resume_state(
                    workspace,
                    inp=inp,
                    last_round_run=-1,
                    next_round=0,
                    review_history=review_history,
                    critic_conversation=critic_conversation,
                    critic_instance_turn=critic_instance_turn,
                    pending_council_text=pending_council_text,
                    pending_compute_text=pending_compute_text,
                    pending_compute_zip_path=pending_compute_zip_path,
                    pending_critique=pending_critique,
                    early_stopped=early_stopped,
                    awaiting_review_round=0,
                    awaiting_review_kind="round_review",
                    awaiting_author=author_0,
                    pending_workflow_feedback=pending_workflow_feedback,
                )
                review_0 = await self.critic(
                    **self._critic_inputs(
                        inp=inp, workspace=workspace,
                        author_thinking=author_0.thinking_summary,
                        mode="fresh",
                        prior_messages=[],
                        omit_author_thinking=False,
                        round=0,
                    )
                )
                review_history.append(review_0)
                critic_conversation = list(review_0.messages_after)
                critic_instance_turn = 1
                self._write_review_artifacts(workspace, review_0, round=0)
                self._snapshot_round(
                    workspace, round=0, author=author_0, review=review_0,
                    council_replies=[],
                )
                last_round_run = 0
                next_round = 1
                self._save_resume_state(
                    workspace,
                    inp=inp,
                    last_round_run=last_round_run,
                    next_round=next_round,
                    review_history=review_history,
                    critic_conversation=critic_conversation,
                    critic_instance_turn=critic_instance_turn,
                    pending_council_text=pending_council_text,
                    pending_compute_text=pending_compute_text,
                    pending_compute_zip_path=pending_compute_zip_path,
                    pending_critique=pending_critique,
                    early_stopped=early_stopped,
                    pending_workflow_feedback=pending_workflow_feedback,
                )

            # ---- Rounds 1..N ------------------------------------------
            first_round = (
                inp.n_rounds + 1 if finalize_without_rounds else max(1, next_round)
            )
            for k in range(first_round, inp.n_rounds + 1):
                await self.events.emit(
                    "ac.round_start", {"round": k, "n_rounds": inp.n_rounds}
                )
                # Author input: prefer pending_critique (set by
                # forced-fresh disagreement or gate rejection) over the
                # last review's prose.
                prev_critique_for_author = (
                    pending_critique
                    or (review_history[-1].review_md if review_history else "")
                )
                pending_critique = ""

                author_k = await self.author(
                    **self._author_inputs(
                        inp=inp, workspace=workspace,
                        prev_critique=prev_critique_for_author,
                        prev_council=pending_council_text,
                        workflow_feedback=pending_workflow_feedback,
                        prev_compute_response=pending_compute_text,
                        compute_zip_path=pending_compute_zip_path,
                        round=k,
                    )
                )
                self._write_files_from_author(workspace, author_k)
                self._write_author_artifacts(workspace, author_k, round=k)
                pending_workflow_feedback = await self._compile_feedback_after_author(
                    workspace, page_limit=inp.page_limit, round=k
                )

                # Consume pending council + compute; both re-set below
                # if the new Author turn requests them again.
                pending_council_text = ""
                pending_compute_text = ""
                pending_compute_zip_path = None
                self._save_resume_state(
                    workspace,
                    inp=inp,
                    last_round_run=last_round_run,
                    next_round=k,
                    review_history=review_history,
                    critic_conversation=critic_conversation,
                    critic_instance_turn=critic_instance_turn,
                    pending_council_text=pending_council_text,
                    pending_compute_text=pending_compute_text,
                    pending_compute_zip_path=pending_compute_zip_path,
                    pending_critique=pending_critique,
                    early_stopped=early_stopped,
                    awaiting_review_round=k,
                    awaiting_review_kind="round_review",
                    awaiting_author=author_k,
                    pending_workflow_feedback=pending_workflow_feedback,
                )

                mode, critic_conversation, critic_instance_turn = (
                    self._critic_mode_for_round(
                        inp=inp,
                        round=k,
                        critic_conversation=critic_conversation,
                        critic_instance_turn=critic_instance_turn,
                    )
                )

                requested_council = (
                    inp.enable_council
                    and author_k.council_question
                    and bool(inp.council_models)
                )
                requested_compute = (
                    inp.enable_compute
                    and bool(author_k.compute_instructions)
                )
                terminal_auxiliary_blocked, kinds, pending_terminal_auxiliary = (
                    self._terminal_auxiliary_decision(
                        inp=inp,
                        round=k,
                        requested_council=bool(requested_council),
                        requested_compute=bool(requested_compute),
                    )
                )
                if terminal_auxiliary_blocked:
                    await self.events.emit(
                        "ac.terminal_auxiliary_suppressed",
                        {"round": k, "kinds": kinds},
                    )
                    pending_critique = pending_terminal_auxiliary
                run_council = bool(requested_council) and not terminal_auxiliary_blocked
                run_compute = bool(requested_compute) and not terminal_auxiliary_blocked
                review_k, council_replies, compute_out = await self._gather_critic_council(
                    inp=inp, workspace=workspace,
                    author_k=author_k,
                    critic_conversation=critic_conversation,
                    mode=mode,
                    round=k,
                    run_council=run_council,
                    run_compute=run_compute,
                    omit_author_thinking=False,
                )
                if council_replies:
                    pending_council_text = render_council_replies_for_author(
                        council_replies
                    )
                if compute_out is not None:
                    pending_compute_text = render_compute_reply_for_author(
                        compute_out
                    )
                    pending_compute_zip_path = compute_out.zip_path
                    self._write_compute_artifacts(
                        workspace, compute_out, round=k
                    )
                review_history.append(review_k)
                critic_conversation = list(review_k.messages_after)
                critic_instance_turn += 1
                self._write_review_artifacts(workspace, review_k, round=k)

                # ---- Early-stop logic ------------------------------
                forced_fresh_for_snapshot: ACCritic.Outputs | None = None
                # Defer ready+ready early-stop when a compute reply is
                # pending: the Author commissioned a worker this round
                # but its findings (possibly a verification failure or
                # counterexample) have not yet been fed back. Continue
                # to the next round so the Author can react before
                # shipping.
                compute_blocks_ship = compute_out is not None
                if compute_blocks_ship and author_k.ready and review_k.answer_ready:
                    await self.events.emit(
                        "ac.early_stop_deferred_for_compute",
                        {
                            "round": k,
                            "compute_status": compute_out.status if compute_out is not None else None,
                        },
                    )
                if terminal_auxiliary_blocked and author_k.ready and review_k.answer_ready:
                    await self.events.emit(
                        "ac.early_stop_deferred_for_terminal_auxiliary",
                        {"round": k},
                    )
                if (
                    author_k.ready
                    and review_k.answer_ready
                    and not compute_blocks_ship
                    and not terminal_auxiliary_blocked
                ):
                    review_for_gate = review_k
                    if review_k.mode == "stateful":
                        # Forced-fresh promotion: fresh instance, same
                        # files, no author thinking, no bias note.
                        forced = await self.critic(
                            **self._critic_inputs(
                                inp=inp, workspace=workspace,
                                author_thinking="",
                                mode="fresh",
                                prior_messages=[],
                                omit_author_thinking=True,
                                round=k,
                            )
                        )
                        await self.events.emit(
                            "ac.forced_fresh_review",
                            {
                                "round": k,
                                "answer_ready": forced.answer_ready,
                                "parse_failed": forced.parse_failed,
                            },
                        )
                        self._write_forced_fresh_artifacts(
                            workspace, forced, round=k
                        )
                        forced_fresh_for_snapshot = forced
                        if not forced.answer_ready:
                            # Disagreement: forced-fresh becomes the
                            # new active instance (the stateful one
                            # endorsed and was just refuted; can't keep
                            # it as the conversation root).
                            critic_conversation = list(forced.messages_after)
                            critic_instance_turn = 1
                            pending_critique = (
                                "## Stateful reviewer's report\n\n"
                                + (review_k.review_md or "")
                                + "\n\n---\n\n"
                                + "## Independent fresh reviewer's report\n\n"
                                + (forced.review_md or "")
                            )
                            self._snapshot_round(
                                workspace, round=k, author=author_k, review=review_k,
                                council_replies=council_replies,
                                forced_fresh_review=forced,
                                compute_out=compute_out,
                            )
                            last_round_run = k
                            self._save_resume_state(
                                workspace,
                                inp=inp,
                                last_round_run=last_round_run,
                                next_round=k + 1,
                                review_history=review_history,
                                critic_conversation=critic_conversation,
                                critic_instance_turn=critic_instance_turn,
                                pending_council_text=pending_council_text,
                                pending_compute_text=pending_compute_text,
                                pending_compute_zip_path=pending_compute_zip_path,
                                pending_critique=pending_critique,
                                early_stopped=early_stopped,
                                pending_workflow_feedback=pending_workflow_feedback,
                            )
                            continue
                        review_for_gate = forced

                    # Critic side has signed off (fresh, or forced-fresh-
                    # confirmed). Deterministic gate next.
                    gate_ok, gate_reasons = await self._deterministic_ready(
                        workspace, page_limit=inp.page_limit
                    )
                    if gate_ok:
                        await self.events.emit(
                            "ac.early_stop_agreed",
                            {
                                "round": k,
                                "author_ready": True,
                                "critic_ready": True,
                                "critic_mode": review_for_gate.mode,
                            },
                        )
                        self._snapshot_round(
                            workspace, round=k, author=author_k, review=review_k,
                            council_replies=council_replies,
                            forced_fresh_review=forced_fresh_for_snapshot,
                            compute_out=compute_out,
                        )
                        last_round_run = k
                        early_stopped = True
                        self._save_resume_state(
                            workspace,
                            inp=inp,
                            last_round_run=last_round_run,
                            next_round=k + 1,
                            review_history=review_history,
                            critic_conversation=critic_conversation,
                            critic_instance_turn=critic_instance_turn,
                            pending_council_text=pending_council_text,
                            pending_compute_text=pending_compute_text,
                            pending_compute_zip_path=pending_compute_zip_path,
                            pending_critique=pending_critique,
                            early_stopped=early_stopped,
                            pending_workflow_feedback=pending_workflow_feedback,
                            awaiting_finalization=True,
                        )
                        break
                    # Gate blocked — push the reasons to the next
                    # Author turn so it can fix them instead of seeing
                    # only the Critic's positive review.
                    await self.events.emit(
                        "ac.deterministic_gate_blocked",
                        {"round": k, "reasons": gate_reasons},
                    )
                    pending_critique = (
                        (review_for_gate.review_md or "")
                        + "\n\n## Workflow rejection\n\n"
                        + "The ship-gate blocked early-stop for the following reasons: "
                        + ", ".join(gate_reasons)
                        + ".\nPlease address these before retrying."
                    )

                self._snapshot_round(
                    workspace, round=k, author=author_k, review=review_k,
                    council_replies=council_replies,
                    forced_fresh_review=forced_fresh_for_snapshot,
                    compute_out=compute_out,
                )
                last_round_run = k
                self._save_resume_state(
                    workspace,
                    inp=inp,
                    last_round_run=last_round_run,
                    next_round=k + 1,
                    review_history=review_history,
                    critic_conversation=critic_conversation,
                    critic_instance_turn=critic_instance_turn,
                    pending_council_text=pending_council_text,
                    pending_compute_text=pending_compute_text,
                    pending_compute_zip_path=pending_compute_zip_path,
                    pending_critique=pending_critique,
                    early_stopped=early_stopped,
                    pending_workflow_feedback=pending_workflow_feedback,
                )

            # ---- Final compile + (optional) final_critic --------
            answer_tex_text = (
                (workspace / "answer.tex").read_text(encoding="utf-8", errors="replace")
                if (workspace / "answer.tex").exists()
                else ""
            )
            bib_path = workspace / "references.bib"
            bib_arg = bib_path if bib_path.exists() and bib_path.stat().st_size > 0 else None
            fixed = await asyncio.to_thread(
                _simple_compile_latex,
                answer_tex_text,
                bib_path=bib_arg,
                page_limit=inp.page_limit,
                is_full_document=True,
            )
            ac_dir = workspace / ".ac"
            ac_dir.mkdir(parents=True, exist_ok=True)
            self._write_compile_artifact(
                ac_dir / "final-compile.log",
                fixed,
                page_limit=inp.page_limit,
                title="Final compile check",
            )
            (workspace / "answer.tex").write_text(fixed.tex, encoding="utf-8")
            page_overflow = fixed.compiled and fixed.pages > inp.page_limit
            await self.events.emit(
                "ac.final_compile",
                {
                    "compiled": fixed.compiled,
                    "pages": fixed.pages,
                    "page_limit": inp.page_limit,
                    "page_overflow": page_overflow,
                },
            )
            if page_overflow:
                fixed = _CompileResult(
                    tex=fixed.tex,
                    tex_path=fixed.tex_path,
                    pdf_path=fixed.pdf_path,
                    compiled=False,
                    pages=fixed.pages,
                    bbl_path=fixed.bbl_path,
                    compile_log=fixed.compile_log,
                    normalization_removals=list(fixed.normalization_removals),
                )

            last_critic_accepted = (
                review_history[-1].answer_ready if review_history else None
            )
            final_critic_answer_ready = False
            final_critic_mode_run = "not_run"
            final_critic_review_md = ""
            terminal_review_history = review_history
            terminal_critic_conversation = critic_conversation
            terminal_critic_instance_turn = critic_instance_turn
            terminal_pending_council_text = pending_council_text
            terminal_pending_compute_text = pending_compute_text
            terminal_pending_compute_zip_path = pending_compute_zip_path
            terminal_pending_critique = pending_critique
            terminal_pending_workflow_feedback = pending_workflow_feedback
            # Dev-only audit. Forced-fresh on the shipped document; not
            # a gate — purely for the record.
            if inp.enable_final_critic:
                try:
                    final_review = await self.critic(
                        **self._critic_inputs(
                            inp=inp, workspace=workspace,
                            author_thinking="",
                            mode="fresh",
                            prior_messages=[],
                            omit_author_thinking=True,
                            round=last_round_run + 1,
                        )
                    )
                    final_critic_answer_ready = final_review.answer_ready
                    final_critic_mode_run = "run"
                    final_critic_review_md = final_review.review_md
                    ac_dir = workspace / ".ac"
                    ac_dir.mkdir(parents=True, exist_ok=True)
                    (ac_dir / "final-review.md").write_text(
                        final_review.review_md, encoding="utf-8"
                    )
                    (ac_dir / "final-review.json").write_text(
                        final_review.model_dump_json(indent=2), encoding="utf-8"
                    )
                    await self.events.emit(
                        "ac.final_critic_review",
                        {
                            "answer_ready": final_review.answer_ready,
                            "parse_failed": final_review.parse_failed,
                        },
                    )
                    terminal_review_history = [*review_history, final_review]
                    terminal_critic_conversation = list(final_review.messages_after)
                    terminal_critic_instance_turn = 1
                    terminal_pending_council_text = ""
                    terminal_pending_compute_text = ""
                    terminal_pending_compute_zip_path = None
                    terminal_pending_critique = final_review.review_md
                except BudgetExhausted as e:
                    if e.scope == "run":
                        raise
                    await self.events.emit(
                        "ac.final_critic_skipped",
                        {"reason": "budget_exhausted", "scope": e.scope},
                    )
                    final_critic_mode_run = "skipped_budget_exhausted"
                except Exception as e:
                    await self.events.emit(
                        "ac.final_critic_failed",
                        {"type": type(e).__name__, "msg": str(e)},
                    )
                    final_critic_mode_run = "failed"

            # ``fixed.bbl_path`` is persisted by ``_simple_compile_latex``
            # via a ``mkstemp``-allocated file outside its TemporaryDirectory
            # so it survives the compile cleanup. Clean it up after the
            # stash call so we don't leak one .bbl per problem under /tmp.
            final_bbl: Path | None = None
            if fixed and getattr(fixed, "bbl_path", None) and fixed.bbl_path.exists():
                final_bbl = fixed.bbl_path
            answer_path = self._stash_answer(
                inp.problem_id,
                fixed.tex,
                bbl_path=final_bbl,
                bib_path=bib_arg,
                ship_bib_alongside=inp.ship_bib_alongside,
            )
            terminal_outputs = {
                "answer_tex": self._encode_run_path(answer_path),
                "compiled": fixed.compiled,
                "pages": fixed.pages,
                "rounds_completed": last_round_run,
                "early_stopped": early_stopped,
                "last_critic_accepted": last_critic_accepted,
                "final_critic_answer_ready": final_critic_answer_ready,
                "final_critic_mode_run": final_critic_mode_run,
                "final_critic_review_md": final_critic_review_md,
                "last_gasp": False,
                "error": None,
            }
            self._save_resume_state(
                workspace,
                inp=inp,
                last_round_run=last_round_run,
                next_round=last_round_run + 1,
                review_history=terminal_review_history,
                critic_conversation=terminal_critic_conversation,
                critic_instance_turn=terminal_critic_instance_turn,
                pending_council_text=terminal_pending_council_text,
                pending_compute_text=terminal_pending_compute_text,
                pending_compute_zip_path=terminal_pending_compute_zip_path,
                pending_critique=terminal_pending_critique,
                early_stopped=early_stopped,
                pending_workflow_feedback=terminal_pending_workflow_feedback,
                terminal_outputs=terminal_outputs,
            )
            if final_bbl is not None:
                try:
                    final_bbl.unlink()
                except OSError:
                    pass
            return self.Outputs(
                problem_id=inp.problem_id,
                answer_tex=answer_path,
                research_notes_tex=workspace / "research_notes.tex",
                references_bib=workspace / "references.bib",
                compiled=fixed.compiled,
                pages=fixed.pages,
                rounds_completed=last_round_run,
                early_stopped=early_stopped,
                last_critic_accepted=last_critic_accepted,
                final_critic_answer_ready=final_critic_answer_ready,
                final_critic_mode_run=final_critic_mode_run,
                final_critic_review_md=final_critic_review_md,
                last_gasp=False,
            )

        except (BudgetExhausted, asyncio.TimeoutError, Exception) as e:
            # ``asyncio.CancelledError`` is INTENTIONALLY not caught
            # here: external cancellation (runner shutdown, server stop,
            # parent task cancel) must reach the caller as a real
            # cancellation, not get converted into a faux-successful
            # workflow output via last-gasp salvage. Internal fanout
            # cancellation (Critic cancelled because Council errored
            # first) is converted to a real Exception inside
            # ``_gather_critic_council`` so it lands here as one of the
            # caught types.
            await self.events.emit(
                "ac.last_gasp",
                {
                    "type": type(e).__name__,
                    "msg": str(e),
                    "rounds_completed": max(last_round_run, 0),
                },
            )
            tex_text = _safe_read(workspace / "answer.tex")
            wrapped = await self._last_gasp_finalize(
                inp.problem, tex_text, error=e, workspace=workspace
            )
            ws_bbl = workspace / "answer.bbl"
            ws_bib = workspace / "references.bib"
            answer_path = self._stash_answer(
                inp.problem_id,
                wrapped,
                bbl_path=ws_bbl if ws_bbl.exists() else None,
                bib_path=ws_bib if ws_bib.exists() else None,
                ship_bib_alongside=inp.ship_bib_alongside,
            )
            error_str = f"{type(e).__name__}: {e}"
            return self.Outputs(
                problem_id=inp.problem_id,
                answer_tex=answer_path,
                research_notes_tex=workspace / "research_notes.tex",
                references_bib=workspace / "references.bib",
                compiled=False,
                pages=0,
                rounds_completed=max(last_round_run, 0),
                early_stopped=False,
                last_critic_accepted=None,
                final_critic_answer_ready=False,
                final_critic_mode_run="not_run",
                final_critic_review_md="",
                last_gasp=True,
                error=error_str,
            )

    # --- workspace + I/O helpers ---------------------------------------

    def _workspace_path(self, problem_id: str, problem_text: str = "") -> Path:
        safe = _safe_id(problem_id)
        if not problem_text:
            return self.ctx.root_workdir / "ac_workspaces" / safe
        return (
            self.ctx.root_workdir
            / "ac_workspaces"
            / f"{safe}-{_problem_hash(problem_text)}"
        )

    async def _init_workspace(self, workspace: Path, *, problem_text: str) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        problem_path = workspace / "problem.txt"
        marker_path = workspace / ".ac" / "problem-hash.txt"
        expected_hash = _problem_hash(problem_text)
        existing_hash = ""
        if marker_path.exists():
            try:
                existing_hash = marker_path.read_text(encoding="utf-8").strip()
            except OSError:
                existing_hash = ""
        if existing_hash and existing_hash != expected_hash:
            raise RuntimeError(
                "AC workspace problem hash mismatch: "
                f"expected {expected_hash}, found {existing_hash}"
            )
        _write_text_atomic(problem_path, problem_text)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(marker_path, expected_hash + "\n")

    def _write_effective_problem(
        self, workspace: Path, *, original: str, effective: str
    ) -> None:
        """Record the problem text the agents actually see this run.

        Only written when it diverges from ``problem.txt`` (i.e. a resume
        note and/or additional instructions were prepended), so plain
        fresh runs stay uncluttered.
        """
        effective_path = workspace / "problem-effective.txt"
        if effective.strip() == original.strip():
            return
        try:
            effective_path.write_text(effective, encoding="utf-8")
        except OSError:
            pass

    def _author_inputs(
        self, *, inp, workspace: Path,
        prev_critique: str, prev_council: str,
        workflow_feedback: str,
        prev_compute_response: str,
        compute_zip_path: Path | None,
        round: int,
    ) -> dict:
        # Report run-scope spend so the Author can pace itself; walk up
        # the budget tree to the root.
        node = self.tracker
        while node.parent is not None:
            node = node.parent
        used = float(node.counters.usd)
        max_usd = (
            float(node.spec.max_usd)
            if node.spec is not None and node.spec.max_usd is not None
            else 0.0
        )
        return {
            "problem": inp.problem,
            "round": round,
            "n_rounds": inp.n_rounds,
            "page_limit": inp.page_limit,
            "budget_used_usd": used,
            "budget_max_usd": max_usd,
            "answer_tex": _safe_read(workspace / "answer.tex"),
            "research_notes_tex": _safe_read(workspace / "research_notes.tex"),
            "references_bib": _safe_read(workspace / "references.bib"),
            "prev_critique": prev_critique,
            "workflow_feedback": workflow_feedback,
            "prev_council": prev_council,
            "prev_compute_response": prev_compute_response,
            "compute_zip_path": compute_zip_path,
        }

    def _critic_inputs(
        self, *, inp, workspace: Path, author_thinking: str,
        mode: str, prior_messages: list[dict],
        omit_author_thinking: bool = False,
        round: int,
    ) -> dict:
        return {
            "problem": inp.problem,
            "round": round,
            "n_rounds": inp.n_rounds,
            "page_limit": inp.page_limit,
            "mode": mode,
            "answer_tex": _safe_read(workspace / "answer.tex"),
            "research_notes_tex": _safe_read(workspace / "research_notes.tex"),
            "references_bib": _safe_read(workspace / "references.bib"),
            "author_thinking": author_thinking,
            "prior_messages": list(prior_messages),
            "omit_author_thinking": omit_author_thinking,
        }

    def _resume_has_no_new_work(
        self,
        *,
        inp,
        state: dict[str, Any] | None,
        next_round: int,
        awaiting_review_round: int | None,
        awaiting_finalization: bool,
        early_stopped: bool,
    ) -> bool:
        if state is None or awaiting_review_round is not None or awaiting_finalization:
            return False
        if not self._resume_ready_to_noop(inp, state):
            return False
        if next_round > inp.n_rounds:
            return True
        previous_bound = _state_int(state, "n_rounds_at_checkpoint", inp.n_rounds)
        has_new_instruction = bool(
            str(getattr(inp, "additional_instructions", "") or "").strip()
        )
        return (
            early_stopped
            and not has_new_instruction
            and inp.n_rounds <= previous_bound
        )

    def _resume_ready_to_noop(
        self, inp, state: dict[str, Any] | None
    ) -> bool:
        if state is None or not self._resume_answer_exists(inp, state):
            return False
        if isinstance(state.get("terminal_outputs"), dict):
            return True
        metadata = _read_json_if_exists(self.ctx.root_workdir / "run-metadata.json")
        return isinstance(metadata.get("outputs"), dict) if isinstance(metadata, dict) else False

    def _resume_answer_exists(self, inp, state: dict[str, Any]) -> bool:
        default = self.ctx.root_workdir / "solutions" / f"{_safe_id(inp.problem_id)}.tex"
        if default.exists():
            return True
        candidates: list[Any] = []
        state_outputs = state.get("terminal_outputs")
        if isinstance(state_outputs, dict):
            candidates.append(state_outputs.get("answer_tex"))
        metadata = _read_json_if_exists(self.ctx.root_workdir / "run-metadata.json")
        metadata_outputs = (
            metadata.get("outputs")
            if isinstance(metadata, dict) and isinstance(metadata.get("outputs"), dict)
            else {}
        )
        if isinstance(metadata_outputs, dict):
            candidates.append(metadata_outputs.get("answer_tex"))
        for candidate in candidates:
            path = self._decode_run_path(candidate)
            if path is not None and path.exists():
                return True
        return False

    def _resume_noop_reason(
        self,
        *,
        inp,
        state: dict[str, Any] | None,
        next_round: int,
        early_stopped: bool,
    ) -> str:
        if next_round > inp.n_rounds:
            return "round_bound_reached"
        previous_bound = _state_int(state, "n_rounds_at_checkpoint", inp.n_rounds)
        if early_stopped and inp.n_rounds <= previous_bound:
            return "already_early_stopped"
        return "no_new_work"

    def _outputs_from_existing_resume(
        self,
        *,
        inp,
        workspace: Path,
        state: dict[str, Any],
    ) -> "ACWorkflow.Outputs":
        metadata = _read_json_if_exists(self.ctx.root_workdir / "run-metadata.json")
        prior_outputs = (
            metadata.get("outputs")
            if isinstance(metadata, dict) and isinstance(metadata.get("outputs"), dict)
            else {}
        )
        state_outputs = (
            state.get("terminal_outputs")
            if isinstance(state.get("terminal_outputs"), dict)
            else {}
        )
        outputs = {**prior_outputs, **state_outputs}
        answer_path = (
            self.ctx.root_workdir / "solutions" / f"{_safe_id(inp.problem_id)}.tex"
        )
        prior_answer = outputs.get("answer_tex") if isinstance(outputs, dict) else None
        if not answer_path.exists() and prior_answer:
            candidate = self._decode_run_path(prior_answer) or Path(str(prior_answer))
            if candidate.exists():
                answer_path = candidate
        return self.Outputs(
            problem_id=inp.problem_id,
            answer_tex=answer_path,
            research_notes_tex=workspace / "research_notes.tex",
            references_bib=workspace / "references.bib",
            compiled=bool(outputs.get("compiled", False)),
            pages=int(outputs.get("pages", 0) or 0),
            rounds_completed=int(
                outputs.get(
                    "rounds_completed",
                    _state_int(state, "last_round_run", 0),
                )
                or 0
            ),
            early_stopped=bool(
                outputs.get("early_stopped", state.get("early_stopped", False))
            ),
            last_critic_accepted=(
                bool(outputs["last_critic_accepted"])
                if "last_critic_accepted" in outputs
                and outputs.get("last_critic_accepted") is not None
                else None
            ),
            final_critic_answer_ready=bool(
                outputs.get("final_critic_answer_ready", False)
            ),
            final_critic_mode_run=str(
                outputs.get("final_critic_mode_run", "not_run") or "not_run"
            ),
            final_critic_review_md=str(
                outputs.get("final_critic_review_md", "") or ""
            ),
            last_gasp=bool(outputs.get("last_gasp", False)),
            error=outputs.get("error"),
        )

    def _problem_with_run_notes(self, inp, *, resume_stop_round: int | None) -> str:
        parts = [str(inp.problem or "").rstrip()]
        if inp.resume_run and resume_stop_round is not None:
            parts.append(
                "This run is resumed from an earlier workflow that terminated "
                f"at round {resume_stop_round}. The new upper round bound is "
                f"round {inp.n_rounds}; continue the existing draft and "
                "round history instead of restarting the solution."
            )
        extra = str(getattr(inp, "additional_instructions", "") or "").strip()
        if extra:
            parts.append(
                "Additional instructions for this continuation:\n"
                f"{extra}"
            )
        return "\n\n".join(part for part in parts if part).strip()

    def _critic_mode_for_resume_review(
        self,
        *,
        inp,
        round: int,
        critic_conversation: list[dict],
        critic_instance_turn: int,
        awaiting_review_kind: str,
    ) -> tuple[str, list[dict], int, bool]:
        if awaiting_review_kind == "final_author":
            return "fresh", [], 0, True
        return (
            *self._critic_mode_for_round(
                inp=inp,
                round=round,
                critic_conversation=critic_conversation,
                critic_instance_turn=critic_instance_turn,
            ),
            False,
        )

    def _terminal_auxiliary_decision(
        self,
        *,
        inp,
        round: int,
        requested_council: bool,
        requested_compute: bool,
    ) -> tuple[bool, list[str], str]:
        kinds = []
        if requested_council:
            kinds.append("Council")
        if requested_compute:
            kinds.append("Compute")
        blocked = (
            round == inp.n_rounds
            and not inp.stop_after_review_round
            and bool(kinds)
        )
        if not blocked:
            return False, kinds, ""
        return True, kinds, (
            "## Workflow continuation required\n\n"
            "The terminal Author turn requested "
            + " and ".join(kinds)
            + ", but there is no following Author turn in this pass "
            "to consume auxiliary results. The auxiliary fanout was "
            "suppressed so hidden results cannot be ignored at ship time. "
            "Continue with another Author round if this advice or "
            "verification is still needed."
        )

    def _critic_mode_for_round(
        self,
        *,
        inp,
        round: int,
        critic_conversation: list[dict],
        critic_instance_turn: int,
    ) -> tuple[str, list[dict], int]:
        K = max(inp.full_critic_interval, 1)
        is_last_critic_round = round == inp.n_rounds and not inp.stop_after_review_round
        if (
            critic_instance_turn >= K
            or (is_last_critic_round and critic_instance_turn > 0)
        ):
            critic_conversation = []
            critic_instance_turn = 0
        mode = "fresh" if critic_instance_turn == 0 else "stateful"
        return mode, critic_conversation, critic_instance_turn

    def _save_resume_state(
        self,
        workspace: Path,
        *,
        inp,
        last_round_run: int,
        next_round: int,
        review_history: list[ACCritic.Outputs],
        critic_conversation: list[dict],
        critic_instance_turn: int,
        pending_council_text: str,
        pending_compute_text: str,
        pending_compute_zip_path: Path | None,
        pending_critique: str,
        early_stopped: bool,
        awaiting_review_round: int | None = None,
        awaiting_review_kind: str | None = None,
        awaiting_author: Author.Outputs | None = None,
        pending_workflow_feedback: str = "",
        awaiting_finalization: bool = False,
        terminal_outputs: dict[str, Any] | None = None,
    ) -> None:
        ac_dir = workspace / ".ac"
        ac_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 1,
            "source": "checkpoint",
            "problem_id": inp.problem_id,
            "problem_hash": _problem_hash(inp.problem),
            "n_rounds_at_checkpoint": inp.n_rounds,
            "last_round_run": last_round_run,
            "next_round": next_round,
            "awaiting_review_round": awaiting_review_round,
            "awaiting_review_kind": awaiting_review_kind,
            "awaiting_author": (
                awaiting_author.model_dump(mode="json")
                if awaiting_author is not None
                else None
            ),
            "awaiting_finalization": awaiting_finalization,
            "review_history": [
                _review_resume_record(review) for review in review_history
            ],
            "critic_conversation": critic_conversation,
            "critic_instance_turn": critic_instance_turn,
            "pending_council_text": pending_council_text,
            "pending_compute_text": pending_compute_text,
            "pending_compute_zip_path": self._encode_run_path(
                pending_compute_zip_path
            ),
            "pending_critique": pending_critique,
            "pending_workflow_feedback": pending_workflow_feedback,
            "early_stopped": early_stopped,
            "terminal_outputs": terminal_outputs,
        }
        (ac_dir / "resume-state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _load_resume_state(self, workspace: Path) -> dict[str, Any] | None:
        state_path = workspace / ".ac" / "resume-state.json"
        if state_path.exists():
            try:
                raw = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = None
            if isinstance(raw, dict):
                raw.setdefault("source", "checkpoint")
                return raw
        return self._load_legacy_resume_state(workspace)

    def _load_legacy_resume_state(self, workspace: Path) -> dict[str, Any] | None:
        ac_dir = workspace / ".ac"
        if not ac_dir.exists():
            return None
        rounds: list[tuple[int, Path]] = []
        for path in ac_dir.glob("round-*"):
            if not path.is_dir():
                continue
            try:
                rounds.append((int(path.name.removeprefix("round-")), path))
            except ValueError:
                continue
        if not rounds:
            return None
        rounds.sort()
        review_history: list[dict[str, Any]] = []
        for _, snap in rounds:
            raw_review = _read_json_if_exists(snap / "review_outputs.json")
            if isinstance(raw_review, dict):
                review_history.append(raw_review)

        last_round, last_snap = rounds[-1]
        for name in CANONICAL_FILES:
            dst = workspace / name
            src = last_snap / name
            if src.exists() and not dst.exists():
                try:
                    shutil.copyfile(src, dst)
                except OSError:
                    pass

        author_raw = _read_json_if_exists(last_snap / "author_outputs.json")
        review_raw = _read_json_if_exists(last_snap / "review_outputs.json")
        forced_raw = _read_json_if_exists(last_snap / "forced_fresh_review_outputs.json")
        final_review_raw = _read_json_if_exists(ac_dir / "final-review.json")
        pending_council_text = _safe_read(last_snap / "council_replies.md")
        pending_compute_text = _safe_read(last_snap / "compute_response.md")
        pending_compute_zip_path = None
        for zip_path in sorted(last_snap.glob("compute_workspace_round_*.zip")):
            pending_compute_zip_path = self._encode_run_path(zip_path)
            break

        pending_critique = ""
        active_review = review_raw
        active_turn = _critic_turn_from_messages(
            active_review.get("messages_after", []) if isinstance(active_review, dict) else []
        )
        active_conversation = (
            list(active_review.get("messages_after", []))
            if isinstance(active_review, dict)
            else []
        )
        if isinstance(forced_raw, dict) and forced_raw.get("answer_ready") is False:
            pending_critique = (
                "## Stateful reviewer's report\n\n"
                + str((review_raw or {}).get("review_md", ""))
                + "\n\n---\n\n"
                + "## Independent fresh reviewer's report\n\n"
                + str(forced_raw.get("review_md", ""))
            )
            active_conversation = list(forced_raw.get("messages_after", []))
            active_turn = _critic_turn_from_messages(active_conversation)

        if not isinstance(review_raw, dict):
            if isinstance(final_review_raw, dict):
                review_history.append(final_review_raw)
                return {
                    "version": 1,
                    "source": "legacy_final_review",
                    "last_round_run": last_round,
                    "next_round": last_round + 1,
                    "review_history": review_history,
                    "critic_conversation": list(final_review_raw.get("messages_after", [])),
                    "critic_instance_turn": _critic_turn_from_messages(
                        final_review_raw.get("messages_after", [])
                    ),
                    "pending_council_text": pending_council_text,
                    "pending_compute_text": pending_compute_text,
                    "pending_compute_zip_path": pending_compute_zip_path,
                    "pending_critique": str(final_review_raw.get("review_md", "")),
                    "early_stopped": False,
                }
            if isinstance(author_raw, dict):
                return {
                    "version": 1,
                    "source": "legacy_final_author",
                    "last_round_run": last_round - 1,
                    "next_round": last_round,
                    "awaiting_review_round": last_round,
                    "awaiting_review_kind": "final_author",
                    "awaiting_author": author_raw,
                    "review_history": review_history,
                    "critic_conversation": active_conversation,
                    "critic_instance_turn": active_turn,
                    "pending_council_text": "",
                    "pending_compute_text": "",
                    "pending_compute_zip_path": None,
                    "pending_critique": "",
                    "early_stopped": False,
                }
            return None

        return {
            "version": 1,
            "source": "legacy_round_snapshot",
            "last_round_run": last_round,
            "next_round": last_round + 1,
            "review_history": review_history,
            "critic_conversation": active_conversation,
            "critic_instance_turn": active_turn,
            "pending_council_text": pending_council_text,
            "pending_compute_text": pending_compute_text,
            "pending_compute_zip_path": pending_compute_zip_path,
            "pending_critique": pending_critique,
            "early_stopped": False,
        }

    def _restore_workspace_from_resume(
        self, workspace: Path, state: dict[str, Any]
    ) -> None:
        awaiting_author = state.get("awaiting_author")
        if isinstance(awaiting_author, dict):
            for field, name in (
                ("answer_tex", "answer.tex"),
                ("research_notes_tex", "research_notes.tex"),
                ("references_bib", "references.bib"),
            ):
                if field in awaiting_author:
                    (workspace / name).write_text(
                        str(awaiting_author.get(field) or ""),
                        encoding="utf-8",
                    )

    def _encode_run_path(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return Path(path).relative_to(self.ctx.root_workdir).as_posix()
        except ValueError:
            return str(path)

    def _decode_run_path(self, raw: Any) -> Path | None:
        if not raw:
            return None
        path = Path(str(raw))
        if path.is_absolute():
            return path
        return self.ctx.root_workdir / path

    def _apply_resume_budget_offset(self) -> None:
        if self._resume_cost_offset_applied:
            return
        self._resume_cost_offset_applied = True
        prior_cost = _sum_logged_model_cost(self.ctx.root_workdir / "events.jsonl")
        if prior_cost > 0:
            self.tracker.add_usd(prior_cost)

    async def _gather_critic_council(
        self,
        *,
        inp,
        workspace: Path,
        author_k: "Author.Outputs",
        critic_conversation: list[dict],
        mode: str,
        round: int,
        run_council: bool,
        run_compute: bool = False,
        omit_author_thinking: bool = False,
    ) -> tuple["ACCritic.Outputs", list[CouncilReply], "Compute.Outputs | None"]:
        """Run Critic, optional Council, and optional Compute worker in
        parallel with proper sibling cancellation.

        Uses ``asyncio.wait(return_when=FIRST_EXCEPTION)`` so that if
        Critic raises, the auxiliaries are canceled before they keep
        spending. Auxiliary (Council / Compute) tasks are wrapped so
        their non-fatal exceptions never trigger FIRST_EXCEPTION — only
        run-scope ``BudgetExhausted`` and ``CancelledError`` are
        propagated from auxiliaries. This prevents an isolated auxiliary
        startup error (e.g. missing docker image, missing codex binary)
        from cancelling the Critic and aborting the whole round.
        """
        critic_task = asyncio.create_task(
            self.critic(
                **self._critic_inputs(
                    inp=inp, workspace=workspace,
                    author_thinking=author_k.thinking_summary,
                    mode=mode,
                    prior_messages=critic_conversation,
                    round=round,
                    omit_author_thinking=omit_author_thinking,
                )
            ),
            name=f"ACCritic-r{round}",
        )
        council_task: asyncio.Task | None = None
        if run_council:
            member_models = (
                author_k.council_to or list(inp.council_models)
            )
            council_task = asyncio.create_task(
                self._safe_council(
                    round=round,
                    author_question=author_k.council_question or "",
                    answer_tex=author_k.answer_tex,
                    research_notes_tex=author_k.research_notes_tex,
                    references_bib=author_k.references_bib,
                    member_models=member_models,
                ),
                name=f"Council-r{round}",
            )
        compute_task: asyncio.Task | None = None
        if run_compute:
            compute_workspace = workspace / "compute"
            compute_workspace.mkdir(parents=True, exist_ok=True)
            compute_task = asyncio.create_task(
                self._safe_compute(
                    inp=inp,
                    round=round,
                    instructions=author_k.compute_instructions or "",
                    answer_tex=author_k.answer_tex,
                    research_notes_tex=author_k.research_notes_tex,
                    references_bib=author_k.references_bib,
                    compute_workspace=compute_workspace,
                ),
                name=f"Compute-r{round}",
            )

        tasks: set[asyncio.Task] = {critic_task}
        if council_task is not None:
            tasks.add(council_task)
        if compute_task is not None:
            tasks.add(compute_task)
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        except BaseException:
            # Defensive: asyncio.wait itself shouldn't raise here, but
            # if it does, cancel everything.
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise

        # Cancel anything still pending. With the safe wrappers above,
        # auxiliaries only raise on cancellation or run-scope budget
        # exhaustion — so reaching here with auxiliaries still pending
        # means Critic just raised, and the safe wrappers will catch
        # the resulting CancelledError and emit a `_failed` event.
        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            if t.cancelled():
                continue
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        # Run-scope BudgetExhausted from ANY task must propagate to
        # ACWorkflow's outer last-gasp salvage.
        for t in tasks:
            if t.cancelled():
                continue
            exc = t.exception()
            if isinstance(exc, BudgetExhausted) and exc.scope == "run":
                raise exc

        # External cancellation reaches the caller as a real cancel
        # (runner shutdown, parent task cancel). We deliberately do
        # not synthesize a RuntimeError from auxiliary failures, since
        # the safe wrappers swallow them.
        if critic_task.cancelled():
            raise asyncio.CancelledError(
                "Critic cancelled with no identifiable internal cause"
            )

        # Critic failures bubble up; the workflow's outer except runs
        # last-gasp salvage on whatever the workspace currently holds.
        review_k = critic_task.result()
        council_replies: list[CouncilReply] = []
        if council_task is not None and not council_task.cancelled():
            try:
                council_out = council_task.result()
                if council_out is not None:
                    council_replies = list(council_out.replies)
            except (asyncio.CancelledError, Exception):
                # Safe wrapper already emitted the failure event.
                pass
        compute_out: Compute.Outputs | None = None
        if compute_task is not None and not compute_task.cancelled():
            try:
                compute_out = compute_task.result()
            except (asyncio.CancelledError, Exception):
                # Safe wrapper already emitted the failure event;
                # leave compute_out at None so the round proceeds
                # without a compute reply.
                compute_out = None
        return review_k, council_replies, compute_out

    async def _safe_council(
        self,
        *,
        round: int,
        author_question: str,
        answer_tex: str,
        research_notes_tex: str,
        references_bib: str,
        member_models: list[str],
    ):
        """Run Council with non-fatal exceptions swallowed.

        Only ``CancelledError`` and run-scope ``BudgetExhausted``
        propagate; everything else is logged as ``ac.council_failed``
        and returns ``None`` so the Critic can complete unmolested.
        """
        try:
            return await self.council(
                author_question=author_question,
                answer_tex=answer_tex,
                research_notes_tex=research_notes_tex,
                references_bib=references_bib,
                member_models=member_models,
            )
        except asyncio.CancelledError:
            raise
        except BudgetExhausted as e:
            if e.scope == "run":
                raise
            await self.events.emit(
                "ac.council_failed",
                {"round": round, "type": "BudgetExhausted", "scope": e.scope, "msg": str(e)},
            )
            return None
        except Exception as e:
            await self.events.emit(
                "ac.council_failed",
                {"round": round, "type": type(e).__name__, "msg": str(e)},
            )
            return None

    async def _safe_compute(
        self,
        *,
        inp,
        round: int,
        instructions: str,
        answer_tex: str,
        research_notes_tex: str,
        references_bib: str,
        compute_workspace: Path,
    ) -> "Compute.Outputs | None":
        """Run Compute with non-fatal exceptions swallowed.

        Same contract as ``_safe_council``: only ``CancelledError``
        and run-scope ``BudgetExhausted`` propagate. Agent-scope budget
        exhaust and arbitrary CLI / setup errors yield a
        ``Compute.Outputs(status="error", error=...)`` value so the
        round can continue with the auxiliary failure visible in the
        next Author prompt.
        """
        try:
            return await self.compute(
                problem=inp.problem,
                problem_id=inp.problem_id,
                round=round,
                instructions=instructions,
                answer_tex=answer_tex,
                research_notes_tex=research_notes_tex,
                references_bib=references_bib,
                compute_workspace=compute_workspace,
                model=inp.compute_model,
                reasoning_effort=inp.compute_reasoning_effort,
                cost_config=inp.compute_cost_config,
                soft_timeout_s=inp.compute_soft_timeout_s,
                hard_timeout_s=inp.compute_hard_timeout_s,
                sandbox_backend=inp.compute_sandbox_backend,
                docker_image=inp.compute_docker_image,
                codex_sandbox=inp.compute_codex_sandbox,
            )
        except asyncio.CancelledError:
            raise
        except BudgetExhausted as e:
            if e.scope == "run":
                raise
            await self.events.emit(
                "ac.compute_failed",
                {"round": round, "type": "BudgetExhausted", "scope": e.scope, "msg": str(e)},
            )
            return Compute.Outputs(
                response_md="",
                zip_path=None,
                status="error",
                summary="",
                workspace=None,
                error=f"BudgetExhausted({e.scope}): {e}",
            )
        except Exception as e:
            await self.events.emit(
                "ac.compute_failed",
                {"round": round, "type": type(e).__name__, "msg": str(e)},
            )
            return Compute.Outputs(
                response_md="",
                zip_path=None,
                status="error",
                summary="",
                workspace=None,
                error=f"{type(e).__name__}: {e}",
            )

    def _write_files_from_author(self, workspace: Path, author: Author.Outputs) -> None:
        (workspace / "answer.tex").write_text(author.answer_tex, encoding="utf-8")
        (workspace / "research_notes.tex").write_text(
            author.research_notes_tex, encoding="utf-8"
        )
        (workspace / "references.bib").write_text(
            author.references_bib, encoding="utf-8"
        )

    def _write_author_artifacts(
        self, workspace: Path, author: Author.Outputs, *, round: int
    ) -> None:
        ac_dir = workspace / ".ac"
        ac_dir.mkdir(parents=True, exist_ok=True)
        (ac_dir / f"author-round-{round}.md").write_text(
            (
                f"# Author round {round}\n\n"
                f"Files changed: {author.files_changed}\n"
                f"Ready: {author.ready}\n"
                f"Council question: {author.council_question or '(none)'}\n"
                f"Council to: {author.council_to or '(default models)'}\n"
                f"Parse warnings: {author.parse_warnings or '(none)'}\n\n"
                f"## Thinking summary\n\n{author.thinking_summary or '(empty)'}\n"
            ),
            encoding="utf-8",
        )

    def _write_review_artifacts(
        self, workspace: Path, review: ACCritic.Outputs, *, round: int
    ) -> None:
        ac_dir = workspace / ".ac"
        ac_dir.mkdir(parents=True, exist_ok=True)
        (ac_dir / f"review-round-{round}.md").write_text(
            review.review_md, encoding="utf-8"
        )

    def _snapshot_round(
        self, workspace: Path, *, round: int,
        author: Author.Outputs,
        review: ACCritic.Outputs | None,
        council_replies: list[CouncilReply],
        forced_fresh_review: ACCritic.Outputs | None = None,
        compute_out: "Compute.Outputs | None" = None,
    ) -> None:
        snap = workspace / ".ac" / f"round-{round}"
        snap.mkdir(parents=True, exist_ok=True)
        for name in CANONICAL_FILES:
            src = workspace / name
            if src.exists():
                try:
                    shutil.copyfile(src, snap / name)
                except OSError:
                    pass
        (snap / "author_outputs.json").write_text(
            author.model_dump_json(indent=2), encoding="utf-8"
        )
        if review is not None:
            (snap / "review.md").write_text(review.review_md, encoding="utf-8")
            (snap / "review_outputs.json").write_text(
                review.model_dump_json(indent=2), encoding="utf-8"
            )
        if forced_fresh_review is not None:
            (snap / "forced_fresh_review.md").write_text(
                forced_fresh_review.review_md, encoding="utf-8"
            )
            (snap / "forced_fresh_review_outputs.json").write_text(
                forced_fresh_review.model_dump_json(indent=2), encoding="utf-8"
            )
        if council_replies:
            (snap / "council_replies.md").write_text(
                render_council_replies_for_author(council_replies),
                encoding="utf-8",
            )
        if compute_out is not None:
            (snap / "compute_response.md").write_text(
                compute_out.response_md or "", encoding="utf-8"
            )
            if compute_out.zip_path is not None:
                try:
                    src_zip = Path(compute_out.zip_path)
                    if src_zip.exists():
                        shutil.copyfile(src_zip, snap / src_zip.name)
                except OSError:
                    pass

    def _write_forced_fresh_artifacts(
        self, workspace: Path, review: ACCritic.Outputs, *, round: int
    ) -> None:
        ac_dir = workspace / ".ac"
        ac_dir.mkdir(parents=True, exist_ok=True)
        (ac_dir / f"forced-fresh-round-{round}.md").write_text(
            review.review_md, encoding="utf-8"
        )

    def _write_compute_artifacts(
        self, workspace: Path, compute_out: "Compute.Outputs", *, round: int
    ) -> None:
        ac_dir = workspace / ".ac"
        ac_dir.mkdir(parents=True, exist_ok=True)
        body = compute_out.response_md or ""
        header = (
            f"# Compute worker reply — round {round}\n"
            f"status: {compute_out.status}\n"
            f"error: {compute_out.error or '(none)'}\n"
            f"workspace: {compute_out.workspace}\n"
            f"zip: {compute_out.zip_path}\n\n---\n\n"
        )
        (ac_dir / f"compute-round-{round}.md").write_text(
            header + body, encoding="utf-8"
        )

    async def _compile_feedback_after_author(
        self, workspace: Path, *, page_limit: int, round: int
    ) -> str:
        """Compile after an Author turn and return feedback for next round.

        This is intentionally deterministic and model-free. It normalizes
        answer.tex to the First Proof LaTeX contract before the Critic sees
        it, records the compile log under ``.ac/``, and gives the next
        Author concrete page/compile/format feedback.
        """
        answer_path = workspace / "answer.tex"
        if not answer_path.exists():
            return "answer.tex is missing."
        body = answer_path.read_text(encoding="utf-8", errors="replace")
        bib_path = workspace / "references.bib"
        bib_arg = bib_path if bib_path.exists() and bib_path.stat().st_size > 0 else None
        try:
            out = await asyncio.to_thread(
                _simple_compile_latex,
                body,
                bib_path=bib_arg,
                page_limit=page_limit,
                is_full_document=True,
            )
        except Exception as e:
            return f"LaTeX compile check failed before pdflatex: {type(e).__name__}: {e}"

        answer_path.write_text(out.tex, encoding="utf-8")
        ac_dir = workspace / ".ac"
        ac_dir.mkdir(parents=True, exist_ok=True)
        self._write_compile_artifact(
            ac_dir / f"compile-round-{round}.log",
            out,
            page_limit=page_limit,
            title=f"Author round {round} compile check",
        )

        messages: list[str] = []
        if out.normalization_removals:
            messages.append(
                "The workflow normalized answer.tex to the First Proof LaTeX "
                "contract: " + "; ".join(out.normalization_removals) + "."
            )
        if not out.compiled:
            messages.append(
                f"answer.tex did not compile under pdflatex after normalization; "
                f"inspect .ac/compile-round-{round}.log and repair it."
            )
        elif out.pages > page_limit:
            messages.append(
                f"answer.tex compiled to {out.pages} pages after normalization, "
                f"above the First Proof limit of {page_limit}. Shorten or "
                "restructure the final deliverable."
            )
        if out.bbl_path is not None:
            try:
                out.bbl_path.unlink()
            except OSError:
                pass
        return "\n".join(messages) if messages else "No LaTeX compile or formatting issues detected."

    @staticmethod
    def _write_compile_artifact(
        path: Path, out: _CompileResult, *, page_limit: int, title: str
    ) -> None:
        body = [
            f"# {title}",
            f"compiled: {out.compiled}",
            f"pages: {out.pages}",
            f"page_limit: {page_limit}",
            "normalization_removals:",
        ]
        if out.normalization_removals:
            body.extend(f"- {item}" for item in out.normalization_removals)
        else:
            body.append("- none")
        body.append("")
        body.append("## pdflatex log")
        body.append(out.compile_log or "(empty)")
        path.write_text("\n".join(body), encoding="utf-8")

    # --- deterministic gate ----------------------------------------------

    async def _deterministic_ready(
        self, workspace: Path, *, page_limit: int
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        answer_path = workspace / "answer.tex"
        if not answer_path.exists():
            return False, ["answer_missing"]
        body = answer_path.read_text(encoding="utf-8", errors="replace")
        if not body.strip():
            return False, ["answer_empty"]
        if r"\textbf{Open:}" in body or r"\textbf{Open}" in body:
            reasons.append("open_gap_marker")
        bib_path = workspace / "references.bib"
        bib_arg = bib_path if bib_path.exists() and bib_path.stat().st_size > 0 else None
        try:
            out = await asyncio.to_thread(
                _simple_compile_latex,
                body,
                bib_path=bib_arg,
                page_limit=page_limit,
                is_full_document=True,
            )
        except Exception as e:
            reasons.append(f"compile_failed:{type(e).__name__}")
            return False, reasons
        try:
            if not out.compiled:
                reasons.append("compile_failed")
                return False, reasons
            if out.pages > page_limit:
                reasons.append(f"page_overflow:{out.pages}>{page_limit}")
            return (not reasons), reasons
        finally:
            # _simple_compile_latex persists ``out.bbl_path`` outside its
            # TemporaryDirectory so ``_stash_answer`` can read it; the
            # deterministic gate doesn't ship the result anywhere, so the
            # .bbl would otherwise leak under /tmp until container exit.
            if out.bbl_path is not None:
                try:
                    out.bbl_path.unlink()
                except OSError:
                    pass

    # --- last-gasp -------------------------------------------------------

    async def _last_gasp_finalize(
        self,
        problem: str,
        tex: str,
        *,
        error: Exception,
        workspace: Path | None = None,
    ) -> str:
        if isinstance(error, BudgetExhausted) and error.scope == "run":
            return normalize_firstproof_latex(_bare_wrap(tex))
        bib_arg: Path | None = None
        if workspace is not None:
            bib_path = workspace / "references.bib"
            if bib_path.exists() and bib_path.stat().st_size > 0:
                bib_arg = bib_path
        out = None
        try:
            out = await asyncio.to_thread(
                _simple_compile_latex,
                tex,
                bib_path=bib_arg,
                page_limit=999,
                is_full_document=True,
            )
            return out.tex
        except Exception:
            return normalize_firstproof_latex(_bare_wrap(tex))
        finally:
            # Same .bbl-leak fix as _deterministic_ready: last-gasp does
            # not pass the result to _stash_answer, so the persisted
            # .bbl never gets cleaned up there.
            if out is not None and out.bbl_path is not None:
                try:
                    out.bbl_path.unlink()
                except OSError:
                    pass

    def _stash_answer(
        self,
        problem_id: str,
        tex_body: str,
        *,
        bbl_path: Path | None = None,
        bib_path: Path | None = None,
        ship_bib_alongside: bool = False,
    ) -> Path:
        solutions_dir = self.ctx.root_workdir / "solutions"
        solutions_dir.mkdir(parents=True, exist_ok=True)
        safe = _safe_id(problem_id)
        final_tex = embed_or_ship_bibliography(
            tex_body,
            bbl_path=bbl_path,
            bib_path=bib_path,
            ship_bib_alongside=ship_bib_alongside,
            safe_id=safe,
            solutions_dir=solutions_dir,
        )
        path = solutions_dir / f"{safe}.tex"
        path.write_text(final_tex, encoding="utf-8")
        return path


# --- helpers ----------------------------------------------------------------


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_id(problem_id: str) -> str:
    cleaned = _SAFE_ID_RE.sub("_", problem_id).strip("._")
    return cleaned or "problem"


def _problem_hash(problem_text: str) -> str:
    normalized = str(problem_text or "").strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:12]


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(tmp).replace(path)
    except BaseException:
        try:
            Path(tmp).unlink()
        except OSError:
            pass
        raise


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, FileNotFoundError):
        return ""


def _read_json_if_exists(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _critic_turn_from_messages(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    return max(0, len(messages) // 2)


def _resume_stop_round(state: dict[str, Any] | None) -> int | None:
    if not state:
        return None
    awaiting = state.get("awaiting_review_round")
    if awaiting is not None:
        try:
            return int(awaiting)
        except (TypeError, ValueError):
            pass
    last_round = state.get("last_round_run")
    try:
        return int(last_round)
    except (TypeError, ValueError):
        return None


def _state_int(state: dict[str, Any] | None, key: str, default: int) -> int:
    if not state or key not in state:
        return default
    try:
        return int(state[key])
    except (TypeError, ValueError):
        return default


def _review_resume_record(review: ACCritic.Outputs) -> dict[str, Any]:
    return review.model_dump(mode="json", exclude={"messages_after"})


def _sum_logged_model_cost(events_path: Path) -> float:
    total = 0.0
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0.0
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("kind") != "model.call":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        try:
            total += float(payload.get("cost_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return total



class ACDAGWorkflow(DAGWorkflow):
    description: ClassVar[str] = ACWorkflow.description
    execution_mode: ClassVar[str] = "workflow"
    cache_enabled: ClassVar[bool] = False

    Inputs = ACWorkflow.Inputs
    Outputs = ACWorkflow.Outputs

    async def _last_gasp(self, inp, state: dict[str, Any], error: Exception):
        helper = ACWorkflow(self.ctx, name="ac_last_gasp")
        ac_state = _ac_visual_state_from_dag_state(state)
        workspace = (
            Path(str(ac_state.get("workspace")))
            if ac_state.get("workspace")
            else helper._workspace_path(inp.problem_id, inp.problem)
        )
        await self.events.emit(
            "ac.last_gasp",
            {
                "type": type(error).__name__,
                "msg": str(error),
                "rounds_completed": max(int(ac_state.get("last_round_run", 0) or 0), 0),
            },
        )
        tex_text = _safe_read(workspace / "answer.tex")
        wrapped = await helper._last_gasp_finalize(
            inp.problem, tex_text, error=error, workspace=workspace
        )
        ws_bbl = workspace / "answer.bbl"
        ws_bib = workspace / "references.bib"
        answer_path = helper._stash_answer(
            inp.problem_id,
            wrapped,
            bbl_path=ws_bbl if ws_bbl.exists() else None,
            bib_path=ws_bib if ws_bib.exists() else None,
            ship_bib_alongside=bool(getattr(inp, "ship_bib_alongside", False)),
        )
        return self.Outputs(
            problem_id=inp.problem_id,
            answer_tex=answer_path,
            research_notes_tex=workspace / "research_notes.tex",
            references_bib=workspace / "references.bib",
            compiled=False,
            pages=0,
            rounds_completed=max(int(ac_state.get("last_round_run", 0) or 0), 0),
            early_stopped=False,
            last_critic_accepted=None,
            final_critic_answer_ready=False,
            final_critic_mode_run="not_run",
            final_critic_review_md="",
            last_gasp=True,
            error=f"{type(error).__name__}: {error}",
        )


def _ac_visual_state_from_dag_state(state: dict[str, Any]) -> dict[str, Any]:
    node = state.get("node") if isinstance(state, dict) else {}
    if isinstance(node, dict):
        loop = node.get("author_critic_loop")
        if hasattr(loop, "state") and isinstance(loop.state, dict):
            return loop.state
        if isinstance(loop, dict) and isinstance(loop.get("state"), dict):
            return loop["state"]
        problem = node.get("problem")
        if hasattr(problem, "state") and isinstance(problem.state, dict):
            return problem.state
        if isinstance(problem, dict) and isinstance(problem.get("state"), dict):
            return problem["state"]
    return {}


__all__ = ["ACWorkflow", "ACDAGWorkflow"]
