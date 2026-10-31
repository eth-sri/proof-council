"""Research, stateful editorial review, and deadline-triggered partial cleanup."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from proofstack.agent import Agent
from proofstack.agents.ac.ac_workflow import (
    ACWorkflow, _safe_id, _problem_hash, _is_programming_error,
)
from proofstack.agents.ac.critic import ACCritic
from proofstack.agents.ac.author import Author
from proofstack.agents.batch3_critic import Batch3CleanupCritic, CleanupContextTooLarge
from proofstack.agents.proof_simplification import (
    SIMPLIFICATION_INSTRUCTIONS, SimplificationCritic, SimplificationInterrupted, SimplificationState,
)
from proofstack.agents.final_stateless_review import FinalStatelessCritic, FinalReviewBlocked, FinalReviewState
from proofstack.agents.ac.critic import CriticContextTooLarge
from proofstack.agents.pwc.workspace import embed_or_ship_bibliography
from proofstack.agents.dag_workflow import _deep_merge
from proofstack.agents.writeup_loop import (
    RewriteSeat, RepairSeat, _assemble, _extract_tex, _defuse_proof_sectioning, _declares,
    _compile_raw, _GateCanceller,
)
from proofstack.budget import BudgetExhausted, BudgetRegistry, BudgetSpec, SubscriptionParked, monetary_limit_exhausted
from proofstack.latex_contract import normalize_submission_latex, render_firstproof_latex_contract
from proofstack.registry import load_preset


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def _save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def published_document_path(root, problem_id, *, partial, digest, version=None):
    if version not in (None, 1) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        return None
    directory = root / ("partials" if partial else "submissions")
    safe_id = _safe_id(problem_id)
    if version == 1:
        return directory / safe_id / f"{digest}.tex"
    return directory / f"{safe_id}.tex"


class _Schedule(BaseModel):
    problem_hash: str
    research_deadline_unix_s: float = Field(gt=0, allow_inf_nan=False)
    run_deadline_unix_s: float = Field(gt=0, allow_inf_nan=False)
    initial_usd: float = Field(gt=0, allow_inf_nan=False)
    partial_reserve_usd: float = Field(ge=0, allow_inf_nan=False)
    final_stateless_review: bool = False
    final_review_reserve_usd: float = Field(default=0, ge=0, allow_inf_nan=False)
    final_review_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    final_review_export_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)


class _AcceptedBaseline(BaseModel):
    problem_hash: str
    problem_id: str
    document: str
    sha256: str
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_bib_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    review_md: str
    round: int = Field(ge=0)
    simplified: bool = False


class FirstProofBatch3Workflow(Agent):
    description = "Research with stateful rewrite review; one unreviewed partial rewrite at the research cutoff."
    execution_mode: ClassVar[str] = "workflow"
    cache_enabled: ClassVar[bool] = False

    class Inputs(ACWorkflow.Inputs):
        compute_sandbox_backend: str = "subprocess"
        # Zero disables the whole shared memory guard in the generic workflow.
        compute_max_parallel_workers: int = Field(default=4, ge=1)
        max_wallclock_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
        page_limit: int = Field(default=16, ge=1, le=16)
        research_seconds: float = Field(default=82800, ge=0, allow_inf_nan=False)
        # Absolute batch deadlines prevent queued subprocesses resetting the clock.
        research_deadline_unix_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
        run_deadline_unix_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
        partial_cleanup_reserve_fraction: float = Field(default=0.15, gt=0, lt=1)
        # Two hours: 90 minutes editing, 25 finishing/repairs, five for export.
        partial_cleanup_seconds: float = Field(default=7200, ge=0, allow_inf_nan=False)
        min_partial_continuation_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)
        max_cleanup_repairs: int = Field(default=8, ge=0, le=30)
        max_partial_cleanup_repairs: int = Field(default=2, ge=0, le=5)
        cleanup_page_headroom: int = Field(default=1, ge=0, le=20)
        cleanup_backend: Literal["api", "claude_code"] = "claude_code"
        max_recovery_attempts: int = Field(default=2, ge=0, le=10)
        # Editorial failures of an accepted draft are counted separately, so
        # they cannot use up the recoveries kept for research failures.
        max_cleanup_handoffs: int = Field(default=4, ge=0, le=20)
        simplification_rounds: int = Field(
            default=0, ge=0, le=10, strict=True,
            description="Optional post-acceptance Author/fresh-Critic attempts before cleanup; set 3 to enable.",
        )
        # Simplification always enables this gate, regardless of this switch.
        final_stateless_review: bool = False
        final_review_reserve_usd: float = Field(default=100, gt=0, allow_inf_nan=False)
        final_review_seconds: float = Field(default=5400, gt=0, allow_inf_nan=False)
        final_review_export_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)

    class Outputs(BaseModel):
        problem_id: str
        answer_tex: Path | None = None
        publication_version: Literal[1] | None = None
        submission_approved: bool = False
        submission_sha256: str | None = None
        partial_ready: bool = False
        partial_sha256: str | None = None
        output_kind: str = "abstention"
        partial_reason: str = ""
        abstained: bool = True
        compiled: bool = False
        pages: int = 0
        rounds_completed: int = 0
        early_stopped: bool = False
        final_critic_answer_ready: bool = False
        final_critic_review_md: str = ""
        accepted_baseline_pending: bool = False
        accepted_baseline_path: Path | None = None
        accepted_baseline_sha256: str | None = None
        cleanup_errors: list[str] = Field(default_factory=list)
        error: str | None = None
        error_retryable: bool = True

    def _remaining_usd(self) -> float:
        limits = [n.spec.max_usd - n.planning_usd for n in self.tracker.chain()
                  if n.spec is not None and n.spec.max_usd is not None and n.spec.max_usd >= 0]
        if not limits or not math.isfinite(min(limits)):
            raise ValueError("Batch 3 requires an explicit finite run USD budget")
        return max(0.0, min(limits))

    @staticmethod
    def _requires_final_review(inp):
        return inp.final_stateless_review or inp.simplification_rounds > 0

    def _final_review_identity(self, inp, document):
        critic = FinalStatelessCritic(self.ctx, name="final_stateless_critic")
        review_input = critic.Inputs(problem=inp.problem, recovery_problem=inp.problem,
                                    answer_tex=document, page_limit=inp.page_limit,
                                    mode="fresh", prior_messages=[], omit_author_thinking=True)
        key = critic._recovery_checkpoint(review_input).stem
        return key, review_input

    def _final_review_path(self, key):
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise FinalReviewBlocked("Invalid final-review checkpoint key")
        return self.ctx.root_workdir / "final_stateless_review" / f"{key}.json"

    def _final_review_pointer(self):
        return self.ctx.root_workdir / "batch3-final-review-pending.json"

    def _final_approval(self, inp, document):
        key, review_input = self._final_review_identity(inp, document)
        path = self._final_review_path(key)
        if not path.exists():
            return None
        state = FinalReviewState.model_validate_json(_read(path))
        if state.key != key or state.review_input != review_input:
            raise FinalReviewBlocked("Final-review checkpoint does not match the exact review packet")
        review = state.review
        return review if (state.status == "completed" and review is not None
                          and review.mode == "fresh" and review.answer_ready and not review.parse_failed) else None

    def _restore_output(self, inp):
        try:
            out = self.Outputs.model_validate_json(_read(self.ctx.root_workdir / "batch3-output.json"))
            revocation = json.loads(_read(self._accepted_checkpoint()) or "{}")
            if (out.problem_id == inp.problem_id and out.submission_approved
                    and isinstance(revocation, dict) and revocation.get("revoked")
                    and revocation.get("problem_hash") == _problem_hash(inp.problem)
                    and revocation.get("sha256") == out.submission_sha256):
                out = self.Outputs(problem_id=inp.problem_id)
                # The outer adapter also reads this manifest directly.
                _save_json(self.ctx.root_workdir / "batch3-output.json", out.model_dump(mode="json"))
                return out
            partial = out.partial_ready and not out.submission_approved
            digest = out.partial_sha256 if partial else out.submission_sha256
            path = published_document_path(self.ctx.root_workdir, inp.problem_id, partial=partial,
                                           digest=digest, version=out.publication_version)
            text = _read(path) if path is not None else ""
            if (out.problem_id == inp.problem_id and out.compiled and out.pages > 0
                    and (partial or out.submission_approved) and text
                    and hashlib.sha256(text.encode("utf-8")).hexdigest() == digest
                    and normalize_submission_latex(text) == text):
                if out.pages > inp.page_limit:
                    if out.submission_approved:
                        raise FinalReviewBlocked(
                            "Requested page-limit configuration excludes the saved approved manuscript; "
                            "preserved the artifact and manifest. Reconcile before resuming."
                        )
                    raise ValueError("Saved partial exceeds the requested page limit")
                if out.submission_approved and self._requires_final_review(inp):
                    try:
                        approval = self._final_approval(inp, text)
                    except (ValueError, OSError) as exc:
                        raise FinalReviewBlocked("Saved final approval is unreadable; preserved the approved manifest") from exc
                    if not approval:
                        raise FinalReviewBlocked(
                            "Saved approval does not match the current final-review configuration; "
                            "preserved the approved artifact and manifest. Reconcile before resuming."
                        )
                return out.model_copy(update={"answer_tex": path})
        except ValueError:
            pass
        out = self.Outputs(problem_id=inp.problem_id)
        if self._requires_final_review(inp):
            _save_json(self.ctx.root_workdir / "batch3-output.json", out.model_dump(mode="json"))
        return out

    def _phase_context(self, name: str, *, usd: float, seconds: float):
        if usd <= 0 or seconds <= 0:
            raise TimeoutError(f"no budget or time left for {name}")
        registry = BudgetRegistry()
        root = registry.register_root("run", BudgetSpec(max_usd=usd, max_wallclock_s=seconds))
        root.parent = self.tracker
        return replace(self.ctx, budgets=registry)

    async def _phase(self, name, *, usd, deadline, call):
        ctx = self._phase_context(name, usd=usd, seconds=deadline - time.monotonic())
        await self.events.emit("batch3.phase_start", {"phase": name, "max_usd": usd, "max_wallclock_s": deadline - time.monotonic()})
        try:
            result = await asyncio.wait_for(call(ctx), timeout=max(0.0, deadline - time.monotonic()))
            if asyncio.current_task().cancelling():
                raise asyncio.CancelledError
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{name} deadline reached")
            return result
        finally:
            tracker = ctx.budgets.root()
            await self.events.emit("batch3.phase_end", {"phase": name, "cost_usd": tracker.counters.usd,
                **({"api_equivalent_usd": tracker.planning_usd, "subscription": True} if tracker.is_subscription else {})})

    def _workspace(self, inp):
        return self.ctx.root_workdir / "ac_workspaces" / f"{_safe_id(inp.problem_id)}-{_problem_hash(inp.problem)}"

    def _research_state(self, inp):
        try:
            state = json.loads(_read(self._workspace(inp) / ".ac/resume-state.json"))
            return state if isinstance(state, dict) else {}
        except ValueError:
            return {}

    def _accepted_checkpoint(self):
        return self.ctx.root_workdir / "batch3-accepted-baseline.json"

    @staticmethod
    def _snapshot_hashes(snapshot):
        try:
            source = (snapshot / "answer.tex").read_bytes()
            try:
                bibliography = (snapshot / "references.bib").read_bytes()
            except FileNotFoundError:
                bibliography = b""
            return (hashlib.sha256(source).hexdigest(), hashlib.sha256(bibliography).hexdigest())
        except OSError:
            return None

    def _restore_accepted(self, inp):
        try:
            baseline = _AcceptedBaseline.model_validate_json(_read(self._accepted_checkpoint()))
            if (baseline.problem_id == inp.problem_id and baseline.problem_hash == _problem_hash(inp.problem)
                    and baseline.document.strip()
                    and hashlib.sha256(baseline.document.encode()).hexdigest() == baseline.sha256
                    and normalize_submission_latex(baseline.document) == baseline.document):
                if baseline.source_sha256 is not None and baseline.source_bib_sha256 is not None:
                    return baseline
                # Recover legacy fingerprints from the reviewed snapshot,
                # never from a live bibliography that may have changed since.
                hashes = self._snapshot_hashes(self._workspace(inp) / ".ac" / f"round-{baseline.round}")
                if (hashes is not None
                        and baseline.source_sha256 in (None, hashes[0])
                        and baseline.source_bib_sha256 in (None, hashes[1])):
                    return baseline.model_copy(update={"source_sha256": hashes[0], "source_bib_sha256": hashes[1]})
        except ValueError:
            pass
        return None

    def _remember_accepted(self, inp, document, review_md):
        round_no = int(self._research_state(inp).get("last_round_run") or 0)
        hashes = self._snapshot_hashes(self._workspace(inp) / ".ac" / f"round-{round_no}")
        if hashes is None:
            # Cleanup can still earn acceptance, but no fallback may be retained
            # without the source that the research critics actually reviewed.
            return
        baseline = _AcceptedBaseline(
            problem_hash=_problem_hash(inp.problem), problem_id=inp.problem_id, document=document,
            sha256=hashlib.sha256(document.encode()).hexdigest(), review_md=review_md,
            source_sha256=hashes[0], source_bib_sha256=hashes[1], round=round_no,
        )
        # A crash between revocation and the research handoff must not turn
        # the old terminal AC checkpoint into a fresh mathematical acceptance.
        try:
            previous = json.loads(_read(self._accepted_checkpoint()))
        except ValueError:
            previous = {}
        if (isinstance(previous, dict) and previous.get("revoked")
                and previous.get("problem_hash") == baseline.problem_hash
                and (previous.get("sha256") == baseline.sha256
                     or (previous.get("source_sha256") == baseline.source_sha256
                         and previous.get("source_bib_sha256") in (None, baseline.source_bib_sha256)))
                and baseline.round <= previous.get("round", -1)):
            raise RuntimeError("Cannot reuse the research acceptance revoked by cleanup; a fresh review is required")
        _save_json(self._accepted_checkpoint(), baseline.model_dump())
        self._accepted_baseline = baseline

    def _revoke_accepted(self, inp, baseline, review, *, round=None):
        retained = self._accepted_baseline
        self._accepted_baseline = None
        _save_json(self._accepted_checkpoint(), {
            "revoked": True, "problem_hash": _problem_hash(inp.problem),
            "sha256": hashlib.sha256(baseline.encode()).hexdigest(),
            "source_sha256": retained.source_sha256 if retained is not None else None,
            "source_bib_sha256": retained.source_bib_sha256 if retained is not None else None,
            "round": int(self._research_state(inp).get("last_round_run") or 0) if round is None else round,
            "review_md": review.review_md,
        })

    def _reconcile_accepted(self, inp, *, exported_document=None):
        baseline = self._accepted_baseline
        if baseline is None:
            return
        # Only a normally completed research phase supplies its export. Error
        # and last-gasp outputs may contain edits that have not been reviewed.
        exported_round = (self._research_state(inp).get("last_round_run")
                          if exported_document is not None
                          and hashlib.sha256(exported_document.encode()).hexdigest() == baseline.sha256 else None)
        snapshots = []
        for path in (self._workspace(inp) / ".ac").glob("round-*"):
            match = re.fullmatch(r"round-(\d+)", path.name)
            if match and int(match[1]) > baseline.round:
                snapshots.append((int(match[1]), path))
        for round_no, snapshot in sorted(snapshots):
            hashes = self._snapshot_hashes(snapshot)
            if hashes is None or not (
                hashes == (baseline.source_sha256, baseline.source_bib_sha256)
                or hashes[0] == baseline.sha256 or round_no == exported_round
            ):
                continue
            for name in ("review_outputs.json", "forced_fresh_review_outputs.json"):
                try:
                    raw = json.loads(_read(snapshot / name))
                    # Missing fields and malformed verdicts are not rejections.
                    if not isinstance(raw, dict) or raw.get("answer_ready") is not False:
                        continue
                    review = ACCritic.Outputs.model_validate(raw)
                except ValueError:
                    continue
                if not review.parse_failed:
                    self._revoke_accepted(inp, baseline.document, review, round=round_no)
                    return

    async def _review_final(self, inp, document, *, deadline):
        key, review_input = self._final_review_identity(inp, document)
        path = self._final_review_path(key)
        if path.exists():
            state = FinalReviewState.model_validate_json(_read(path))
            if state.key != key or state.review_input != review_input:
                raise FinalReviewBlocked("Final-review checkpoint does not match the exact review packet")
        else:
            state = FinalReviewState(
                key=key, review_input=review_input, max_usd=inp.final_review_reserve_usd,
                initial_spend=self.tracker.planning_usd,
                deadline_unix_s=time.time() + min(inp.final_review_seconds,
                    deadline - inp.final_review_export_seconds - time.monotonic()),
            )
            _save_json(path, state.model_dump(mode="json"))
        _save_json(self._final_review_pointer(), {"key": key})
        if state.status == "completed":
            if state.review is None:
                raise FinalReviewBlocked("Completed final review is missing its verdict")
            if not state.review.parse_failed or state.attempt >= 1:
                return state.review
        phase_end = min(deadline - inp.final_review_export_seconds,
                        time.monotonic() + state.deadline_unix_s - time.time())
        allowance = min(self._remaining_usd(),
                        state.max_usd - max(0.0, self.tracker.planning_usd - state.initial_spend))
        if state.status == "completed":
            if allowance <= 0 or phase_end <= time.monotonic():
                return state.review
            state.previous_review = state.review
            state.attempt += 1
            state.status, state.review = "pending", None
            _save_json(path, state.model_dump(mode="json"))

        async def call(ctx):
            critic = FinalStatelessCritic(ctx, name="final_stateless_critic")
            critic.verdict_attempt = state.attempt
            try:
                review = await critic(**state.review_input.model_dump())
            except BudgetExhausted as exc:
                completed = getattr(exc, "completed_output", None) or getattr(critic, "_completed_review", None)
                if (not monetary_limit_exhausted(exc)
                        or not isinstance(completed, ACCritic.Outputs)):
                    raise
                review = completed
            # Save before event logging/publication. The regular critic owns
            # provider reconciliation if interruption happened before this point.
            state.status, state.review = "completed", review
            _save_json(path, state.model_dump(mode="json"))
            await self.events.emit("batch3.final_stateless_review", {
                "key": key, "sha256": hashlib.sha256(document.encode()).hexdigest(),
                "accepted": review.answer_ready and not review.parse_failed and review.mode == "fresh",
                "parse_failed": review.parse_failed,
            })
            return review

        review = await self._phase("final_stateless_review", usd=allowance, deadline=phase_end, call=call)
        if review.parse_failed and state.attempt == 0:
            return await self._review_final(inp, document, deadline=deadline)
        return review

    async def _finish_final_candidate(self, inp, out, document, *, deadline):
        try:
            ok, pages, detail = await self._check(document, inp, deadline)
            if not ok or normalize_submission_latex(document) != document:
                raise FinalReviewBlocked("Final manuscript failed the exact export check: " + detail)
            review = await self._review_final(inp, document, deadline=deadline)
            if review.parse_failed or review.mode != "fresh":
                raise FinalReviewBlocked("Final stateless critic did not return a valid fresh verdict")
            if not review.answer_ready:
                # A cold rejection does not identify which earlier version
                # introduced the flaw. Do not silently fall back to that family.
                baseline = self._accepted_baseline.document if self._accepted_baseline else document
                self._handoff(inp, baseline, document, review, origin="Final stateless critic")
                _save_json(self._final_review_pointer(), {})
                out.final_critic_answer_ready = False
                out.final_critic_review_md = review.review_md
                await self._fallback(inp, out, document, deadline=deadline)
                await self.events.emit("batch3.final_review_rejected", {"review_md": review.review_md})
                return None
            if time.monotonic() >= deadline:
                raise FinalReviewBlocked("Run deadline reached before final-reviewed export")
            out.final_critic_review_md = review.review_md
            published = self._publish(inp, out, document, pages, partial=False)
            _save_json(self._final_review_pointer(), {})
            return published
        except SubscriptionParked:
            raise
        except Exception as exc:
            # No further paid phase after an unresolved, invalid or unavailable
            # final review. Keep a clearly unapproved artifact for inspection.
            error = f"Final stateless review: {type(exc).__name__}: {exc}"
            out.cleanup_errors.append(error)
            if out.submission_approved:
                out.answer_tex, out.submission_sha256 = None, None
                out.submission_approved = out.final_critic_answer_ready = out.early_stopped = False
                out.abstained, out.compiled, out.output_kind = True, False, "abstention"
            await self._fallback(inp, out, document, deadline=deadline)
            out.error, out.partial_reason, out.error_retryable = error, error, False
            _save_json(self.ctx.root_workdir / "batch3-output.json", out.model_dump(mode="json"))
            return out

    async def _publish_retained(self, inp, out, *, deadline, allow_final_review=False):
        self._reconcile_accepted(inp)
        baseline = self._accepted_baseline
        if baseline is None:
            return None
        if self._requires_final_review(inp) and not self._final_approval(inp, baseline.document):
            if allow_final_review:
                return await self._finish_final_candidate(inp, out, baseline.document, deadline=deadline)
            return None
        return await self._publish_accepted(inp, out, baseline.document, deadline=deadline,
                                            review_md=baseline.review_md)

    async def _research(self, ctx, inp, *, resume, round_bound, out, deadline):
        async def checkpoint(workspace, round):
            if workspace != self._workspace(inp):
                raise ValueError("Author checkpoint belongs to a different workspace")
            _, pages, detail = await self._fallback(inp, out, _read(workspace / "answer.tex"), deadline=deadline)
            feedback = (
                f"Final export check, including the unreviewed partial notice: {detail}\n"
                + self._page_guidance(inp, pages)
                + "\nThis is a format check, not mathematical acceptance."
            )
            (workspace / ".ac" / f"partial-export-round-{round}.log").write_text(feedback, encoding="utf-8")
            return feedback

        preset = load_preset("author_critic")
        ctx = replace(ctx, component_configs=_deep_merge(preset.component_configs, ctx.component_configs),
                      model_overrides={**preset.model_overrides, **ctx.model_overrides},
                      author_checkpoint=checkpoint)
        inputs = preset.build_inputs(cli_overrides={
            **inp.model_dump(), "resume_run": resume, "n_rounds": round_bound,
            "stop_after_review_round": True, "enable_final_critic": False, "ship_bib_alongside": False,
        })
        return await preset.workflow_cls(ctx, name=f"batch3_research_{round_bound}")(**inputs)

    @staticmethod
    def _page_guidance(inp, pages=None):
        target = max(1, inp.page_limit - inp.cleanup_page_headroom)
        measurement = f"The supplied manuscript measured {pages} pages. " if pages else "No successful page count is available for the supplied manuscript. "
        return (
            measurement + f"Aim for at most {target} pages, leaving headroom below the hard {inp.page_limit}-page cap "
            "for the title, references and partial-result notice. Shorten repetition and exposition, not essential "
            "proof steps, hypotheses or caveats. Do not shrink fonts or change margins/spacing to fit."
        )

    async def _edit(self, ctx, inp, document, *, name, notes="", report="", partial=False, restore_baseline="",
                    mechanical=False, measured_pages=None, session_key="standalone", session_baseline="",
                    editor_feedback=None, finishing_only=False):
        constraints = render_firstproof_latex_contract(inp.page_limit, can_compile=inp.cleanup_backend == "claude_code") + (
            "\nPreserve claims, quantifiers, hypotheses, essential proof steps and acknowledged gaps. "
            "Keep an inline bibliography; do not use external bibliography or input files."
        )
        constraints += "\nOriginal problem:\n" + inp.problem
        constraints += "\n" + self._page_guidance(inp, measured_pages)
        if restore_baseline and not partial:
            unable_signal = ("write status=unable in completion.json" if inp.cleanup_backend == "claude_code"
                             else "return UNABLE:")
            constraints += (
                "\nRestore only the specific baseline proof passages identified by the critic. "
                "Preserve unaffected improvements in the candidate. Do not invent new mathematics "
                f"or silently repair a baseline flaw. If restoration is insufficient, {unable_signal} "
                "and explain the mathematical gap. The restored candidate must be reviewed again.\n"
                "Pre-rewrite baseline (source for targeted restoration, not blanket authority):\n"
                + restore_baseline
            )
        elif report and not partial:
            constraints += "\nMake only targeted editorial repairs to the findings. Do not rewrite the manuscript wholesale or introduce new mathematics."
        if partial:
            constraints += (
                "\nThis is an UNSOLVED attempt. Produce an honest partial-results write-up, "
                "not a purported complete solution. Preserve explicit unresolved gaps, conditional "
                "claims and critic objections. Do not invent repairs or claim a new mathematical "
                "result. This rewrite will NOT receive another mathematical review.\n"
                "The workflow places a one-line partial-result notice directly after the title and "
                "re-inserts it after your edit; leave space for it inside the page limit. Do not write "
                "your own notice, status banner or remark about review status; state what is proved "
                "and what remains open in the abstract and introduction.\n"
            )
            if not mechanical:
                constraints += "Latest unresolved findings:\n" + report
        if inp.cleanup_backend == "claude_code":
            from proofstack.agents.cleanup_session import CleanupSession
            if mechanical:
                constraints += "\nMake only the requested mechanical compilation/page repairs, not a new editorial rewrite."
            result = await CleanupSession(ctx)(
                problem=inp.problem, document=document, baseline=session_baseline or document,
                research_notes=notes, findings=report, constraints=constraints,
                partial=partial, page_limit=inp.page_limit, session_key=session_key,
                finishing_only=finishing_only,
            )
            if result.status == "unable":
                raise ValueError("editor could not complete the requested changes: " + result.summary)
            if r"\begin{document}" not in result.answer_tex or r"\end{document}" not in result.answer_tex:
                raise ValueError("editor did not return a complete LaTeX manuscript")
            if editor_feedback is not None:
                editor_feedback.append(result.feedback_md or result.summary)
            return normalize_submission_latex(_defuse_proof_sectioning(result.answer_tex))
        template = "partial-repair.txt" if mechanical else ("repair.txt" if report and not partial else "rewrite-wrapper.txt")
        subs = {"document": document}
        if template == "repair.txt":
            subs["referee findings"] = report
        elif mechanical:
            subs["mechanical findings"] = report
        prompt = _assemble(template, subs, research_notes=notes, writing_constraints=constraints)
        seat_cls = RepairSeat if template in ("repair.txt", "partial-repair.txt") else RewriteSeat
        seat = seat_cls(ctx, name=name)
        seat.wallclock_cap_s = ctx.budgets.root().remaining_wallclock_s()
        seat.stage = name
        try:
            reply = await seat(prompt=prompt)
        except BudgetExhausted as exc:
            completed = getattr(exc, "completed_output", None)
            # A partial needs no further paid review. Keep its already charged
            # reply, but still run the normal parsing and deterministic gates.
            if (not partial or not monetary_limit_exhausted(exc)
                    or not isinstance(completed, seat.Outputs)):
                raise
            reply = completed
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        tex, declaration = _extract_tex(reply.text)
        if _declares(declaration, "UNABLE:"):
            raise ValueError("editor could not complete the requested changes: " + declaration)
        if r"\begin{document}" not in tex or r"\end{document}" not in tex:
            raise ValueError("editor did not return a complete LaTeX manuscript")
        if editor_feedback is not None:
            editor_feedback.append(declaration)
        return normalize_submission_latex(_defuse_proof_sectioning(tex))

    async def _standalone(self, inp, document, *, deadline):
        document = normalize_submission_latex(document)
        if not re.search(r"\\bibliography\s*\{", document):
            return document
        workspace = self._workspace(inp)
        bbl = self.workdir / "fallback.bbl"
        bbl.unlink(missing_ok=True)
        canceller = _GateCanceller()
        try:
            await asyncio.to_thread(
                _compile_raw, document, _read(workspace / "references.bib"),
                deadline=deadline, canceller=canceller, bbl_output=bbl, secure=True,
            )
            document = embed_or_ship_bibliography(
                document, bbl_path=bbl, bib_path=None, ship_bib_alongside=False,
                safe_id=_safe_id(inp.problem_id), solutions_dir=self.workdir,
            )
            return normalize_submission_latex(document)
        except asyncio.CancelledError:
            canceller.cancel()
            raise
        finally:
            bbl.unlink(missing_ok=True)

    async def _compile(self, document, *, deadline):
        canceller = _GateCanceller()
        try:
            return await asyncio.to_thread(_compile_raw, document, None, deadline=deadline, canceller=canceller,
                                           secure=True)
        except asyncio.CancelledError:
            canceller.cancel()
            raise

    async def _check(self, document, inp, deadline):
        if re.search(r"\\(?:bibliography|addbibresource|input|include)\b", document):
            return False, 0, "external LaTeX/bibliography files are not permitted"
        compiled, pages, detail = await self._compile(document, deadline=deadline)
        return compiled and 0 < pages <= inp.page_limit, pages, f"compiled={compiled}, pages={pages}; {detail}"

    def _retain_simplified(self, state):
        baseline = self._accepted_baseline
        if baseline is None:
            raise RuntimeError("Simplification requires an accepted baseline")
        baseline = baseline.model_copy(update={
            "document": state.document, "sha256": hashlib.sha256(state.document.encode()).hexdigest(),
            "review_md": state.review_md,
            "simplified": baseline.simplified or hashlib.sha256(state.document.encode()).hexdigest() != state.source_sha256,
        })
        _save_json(self._accepted_checkpoint(), baseline.model_dump())
        self._accepted_baseline = baseline

    def _recover_simplification_call(self, inp, state, path):
        """Reconcile saved local evidence without issuing another model call."""
        if state.stage in {"author", "critic"} or state.call_pending:
            role = state.call_role or state.stage
            folders = list((self.ctx.root_workdir / "agents").glob(f"{state.call_agent}-c*")) if state.call_agent else []
            settled = False
            subscription_output = None
            if len(folders) == 1:
                folder = folders[0]
                for name in ("provider-settlement.json", "subscription-settlement.json"):
                    if (folder / name).exists():
                        evidence = json.loads(_read(folder / name))
                        settled = evidence.get("settled") is True
                        if name == "subscription-settlement.json" and evidence.get("receipt_id"):
                            from proofstack.subscription import recover_packet
                            settled, subscription_output = recover_packet(self.ctx.root_workdir, evidence["receipt_id"])
                if role == "critic":
                    raw = _read(folder / "completed_review.json") or _read(folder / "output.json")
                    if raw:
                        state.review = Batch3CleanupCritic.Outputs.model_validate_json(raw)
                    elif subscription_output is not None:
                        critic = SimplificationCritic(self.ctx, name=state.call_agent)
                        review_input = critic.Inputs.model_validate_json(_read(folder / "input.json"))
                        state.review = critic.parse_output(subscription_output["text"], review_input)
                    elif (folder / "provider-attempts.jsonl").exists():
                        from mathagents.provider_trace import latest_attempts
                        rows = latest_attempts(folder / "provider-attempts.jsonl")
                        if rows:
                            report_path = folder / f"provider-completed-{rows[-1]['invocation_id']}.json"
                            if report_path.exists():
                                report = json.loads(_read(report_path))["report"]
                                critic = SimplificationCritic(self.ctx, name=state.call_agent)
                                review_input = critic.Inputs.model_validate_json(_read(folder / "input.json"))
                                state.review = critic.parse_output(report, review_input)
                elif settled:
                    raw = _read(folder / "completed_author.json") or _read(folder / "output.json")
                    if raw:
                        state.candidate = Author.Outputs.model_validate_json(raw)
            # A completed rejection matters even if accounting is still pending.
            if (self._accepted_baseline is not None and state.review is not None and not state.review.parse_failed
                    and state.review.disposition == "research"):
                self._handoff(inp, state.document,
                              normalize_submission_latex(state.candidate.answer_tex) if state.candidate else state.document,
                              state.review)
            if not settled:
                raise SimplificationInterrupted(
                    f"Interrupted simplification {state.stage} call; reconcile provider work before paid retries: {path}"
                )
            state.call_pending = False
            state.stop_reason = "Interrupted call settled; optional simplification stopped"
            state.stage = "reviewed" if state.review is not None else "done"
        if state.stage in {"reviewed", "rejected"}:
            review = state.review
            if review is not None and not review.parse_failed and review.disposition == "research":
                state.stage = "rejected"
                _save_json(path, state.model_dump(mode="json"))
                if self._accepted_baseline is not None:
                    self._handoff(inp, state.document,
                                  normalize_submission_latex(state.candidate.answer_tex) if state.candidate else state.document,
                                  review)
                return
            if state.stage == "rejected":
                raise SimplificationInterrupted("Rejected simplification checkpoint has no valid verdict")
            if review is not None and review.answer_ready and not review.parse_failed and review.disposition == "accept":
                if state.candidate is None:
                    raise SimplificationInterrupted("Accepted simplification checkpoint is missing its candidate")
                state.document = normalize_submission_latex(state.candidate.answer_tex)
                state.notes, state.review_md = state.candidate.research_notes_tex, review.review_md
            state.completed += 1
            state.stage, state.candidate, state.review = "ready", None, None
            if state.stop_reason:
                state.stage = "done"
        _save_json(path, state.model_dump(mode="json"))
        if self._accepted_baseline is not None:
            self._retain_simplified(state)

    def _reconcile_simplifications(self, inp):
        for path in sorted((self.ctx.root_workdir / "proof_simplification").glob("*/state.json")):
            state = SimplificationState.model_validate_json(path.read_text(encoding="utf-8"))
            baseline = self._accepted_baseline
            same_baseline = (baseline is not None and path.parent.name.startswith(f"{baseline.round}-")
                             and baseline.sha256 in {state.source_sha256, hashlib.sha256(state.document.encode()).hexdigest()})
            if state.stage in {"reviewed", "rejected"} and not same_baseline and not state.call_pending:
                continue
            if (state.stage in {"author", "critic"} or state.call_pending) and baseline is not None and not same_baseline:
                raise SimplificationInterrupted("Unresolved simplification belongs to another accepted baseline")
            if state.stage in {"author", "critic", "reviewed", "rejected"} or state.call_pending:
                self._recover_simplification_call(inp, state, path)

    def _preserve_pending_baseline(self, inp, out, error):
        baseline = self._accepted_baseline
        if baseline is not None:
            path = self.ctx.root_workdir / "accepted_pending" / f"{baseline.sha256}.tex"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(baseline.document, encoding="utf-8")
            # This is a previously accepted proof, not a final-approved submission
            # or an unreviewed partial. Keep it outside both export directories.
            out = out.model_copy(update={
                "answer_tex": path, "submission_approved": False, "submission_sha256": None,
                "partial_ready": False, "partial_sha256": None, "abstained": True,
                "final_critic_answer_ready": False, "output_kind": "accepted_baseline_needs_final_review",
                "accepted_baseline_pending": True, "accepted_baseline_path": path,
                "accepted_baseline_sha256": baseline.sha256,
            })
        out.error, out.partial_reason, out.error_retryable = str(error), str(error), False
        _save_json(self.ctx.root_workdir / "batch3-output.json", out.model_dump(mode="json"))
        return out

    async def _simplify_accepted(self, inp, document, notes, report, out, *, deadline, reserve):
        if not inp.simplification_rounds or self._accepted_baseline is None:
            return document, notes, None
        source = hashlib.sha256(document.encode()).hexdigest()
        key = f"{self._accepted_baseline.round}-{source}"
        root = self.ctx.root_workdir / "proof_simplification" / key
        path = root / "state.json"
        if path.exists():
            state = SimplificationState.model_validate_json(path.read_text(encoding="utf-8"))
            if state.source_sha256 != source:
                raise ValueError("Simplification checkpoint belongs to another manuscript")
            if state.stage in {"author", "critic", "reviewed", "rejected"} or state.call_pending:
                self._recover_simplification_call(inp, state, path)
        else:
            seconds = deadline - time.monotonic()
            available = self._remaining_usd() - reserve
            if seconds <= 0 or available <= 0:
                return document, notes, None
            # This optional phase cannot consume the cleanup reserve or reset
            # its allowance on resume. Keep half the disposable time/money for
            # editorial work; the global research cutoff still applies.
            state = SimplificationState(
                source_sha256=source, rounds=inp.simplification_rounds,
                document=document, notes=notes, review_md=report,
                max_usd=available / 2, initial_spend=self.tracker.planning_usd,
                deadline_unix_s=time.time() + seconds / 2,
            )
            _save_json(path, state.model_dump(mode="json"))
        if (state.stage in {"reviewed", "rejected"} and state.review is not None
                and not state.review.parse_failed and state.review.disposition == "research"):
            state.stage = "rejected"
            _save_json(path, state.model_dump(mode="json"))
            self._handoff(inp, state.document,
                          normalize_submission_latex(state.candidate.answer_tex) if state.candidate else state.document,
                          state.review)
            return state.document, state.notes, state.review
        self._retain_simplified(state)
        if state.stage in {"author", "critic"}:
            raise SimplificationInterrupted(
                f"Interrupted simplification {state.stage} call; reconcile provider work before paid retries: {path}"
            )
        if state.stage == "rejected":
            raise ValueError("Rejected simplification checkpoint has no valid baseline rejection")
        if state.stage == "done":
            return state.document, state.notes, None
        phase_end = min(deadline, time.monotonic() + state.deadline_unix_s - time.time())
        allowance = min(self._remaining_usd() - reserve,
                        state.max_usd - max(0.0, self.tracker.planning_usd - state.initial_spend))
        try:
            await self._phase(
                "proof_simplification", usd=allowance, deadline=phase_end,
                call=lambda ctx: self._simplification_loop(ctx, inp, state, path, deadline=phase_end),
            )
        except SubscriptionParked:
            raise
        except Exception as exc:
            interrupted_call = state.stage in {"author", "critic"} or state.call_pending
            if interrupted_call:
                self._recover_simplification_call(inp, state, path)
            if not interrupted_call and not isinstance(exc, (BudgetExhausted, TimeoutError)):
                raise
            state.stop_reason = f"{type(exc).__name__}: {exc}"
            out.cleanup_errors.append("proof simplification: " + state.stop_reason)
        if state.stage == "rejected":
            return state.document, state.notes, state.review
        state.stage = "done"
        _save_json(path, state.model_dump(mode="json"))
        await self.events.emit("batch3.simplification_end", {
            "completed": state.completed, "requested": state.rounds, "reason": state.stop_reason,
            "accepted_sha256": self._accepted_baseline.sha256,
        })
        return state.document, state.notes, None

    async def _simplification_loop(self, ctx, inp, state, path, *, deadline):
        while state.completed < state.rounds:
            ctx.budgets.root().check()
            attempt = state.completed + 1
            if state.stage == "ready":
                await self.events.emit("batch3.simplification_round", {"round": attempt, "rounds": state.rounds})
                state.stage = "author"
                state.call_pending, state.call_role = True, "author"
                state.call_agent = f"simplification_author-{path.parent.name}-{attempt}"
                _save_json(path, state.model_dump(mode="json"))
                author = Author(ctx, name=state.call_agent)
                try:
                    state.candidate = await author(
                        problem=inp.problem, recovery_problem=inp.problem,
                        round=attempt, n_rounds=state.rounds, page_limit=inp.page_limit,
                        answer_tex=state.document, research_notes_tex=state.notes,
                        workflow_feedback=SIMPLIFICATION_INSTRUCTIONS,
                        prev_critique=state.feedback,
                        budget_used_usd=ctx.budgets.root().planning_usd,
                        budget_max_usd=ctx.budgets.root().spec.max_usd,
                    )
                except BudgetExhausted as exc:
                    completed = getattr(exc, "completed_output", None)
                    if isinstance(completed, Author.Outputs):
                        state.candidate = completed
                        state.stage = "candidate"
                        state.call_pending = False
                        _save_json(path, state.model_dump(mode="json"))
                    raise
                state.stage = "candidate"
                state.call_pending = False
                _save_json(path, state.model_dump(mode="json"))
            candidate = state.candidate
            if candidate is None:
                raise ValueError("Simplification checkpoint is missing its candidate")
            # Normalize before review and export exactly those reviewed bytes.
            document = normalize_submission_latex(candidate.answer_tex)
            ok, pages, feedback = await self._check(document, inp, deadline)
            reason = ""
            if not candidate.ready or candidate.artifact_status == "failed":
                reason = "Author did not deliver a complete candidate"
            elif candidate.council_question or candidate.compute_instructions:
                reason = "Candidate requests auxiliary work not completed in this phase"
            elif document == state.document:
                reason = "No change to the accepted manuscript"
            elif not ok:
                reason = feedback
            stopped = None
            if not reason and state.stage != "reviewed":
                state.stage = "critic"
                state.call_pending, state.call_role = True, "critic"
                state.call_agent = f"simplification_critic-{path.parent.name}-{attempt}"
                _save_json(path, state.model_dump(mode="json"))
                critic = SimplificationCritic(ctx, name=state.call_agent)
                try:
                    state.review = await critic(
                        problem=inp.problem, answer_tex=document, baseline_tex=state.document,
                        page_limit=inp.page_limit, mode="fresh", prior_messages=[],
                        compile_feedback=feedback, editor_response=candidate.thinking_summary,
                    )
                except BaseException as exc:
                    completed = critic.completed_review or getattr(exc, "completed_output", None)
                    if isinstance(completed, Batch3CleanupCritic.Outputs) and completed.disposition == "research":
                        state.review, state.stage = completed, "rejected"
                        _save_json(path, state.model_dump(mode="json"))
                        self._handoff(inp, state.document, document, completed)
                    if (isinstance(exc, BudgetExhausted) and not isinstance(exc, SubscriptionParked)
                            and isinstance(completed, Batch3CleanupCritic.Outputs)):
                        state.review, stopped = completed, exc
                    else:
                        raise
                state.stage = "reviewed"
                state.call_pending = False
                _save_json(path, state.model_dump(mode="json"))
            review = state.review if not reason else None
            if review is not None and not review.parse_failed and review.disposition == "research":
                state.stage = "rejected"
                _save_json(path, state.model_dump(mode="json"))
                self._handoff(inp, state.document, document, review)
                return
            accepted = bool(review is not None and review.answer_ready and not review.parse_failed
                            and review.disposition == "accept" and ok)
            _save_json(path.parent / f"round-{attempt}.json", {
                "author": candidate.model_dump(mode="json"),
                "review": review.model_dump(mode="json") if review else None,
                "candidate_tex": document, "accepted": accepted, "pages": pages, "reason": reason,
            })
            if accepted:
                state.document, state.notes, state.review_md = document, candidate.research_notes_tex, review.review_md
            state.feedback = reason or (review.review_md if review else "No valid review")
            state.completed += 1
            state.stage, state.candidate, state.review = "ready", None, None
            _save_json(path, state.model_dump(mode="json"))
            if accepted:
                self._retain_simplified(state)
            await self.events.emit("batch3.simplification_review", {
                "round": attempt, "accepted": accepted, "pages": pages, "reason": state.feedback,
            })
            if stopped is not None:
                raise stopped

    async def _candidate_loop(self, ctx, inp, baseline, notes, *, episode, deadline):
        history = list(self._research_state(inp).get("critic_conversation") or [])
        provenance = ""
        if getattr(self, "_accepted_baseline", None) is not None and self._accepted_baseline.simplified:
            provenance = (
                "Baseline provenance: the baseline supplied below is the retained accepted manuscript. "
                "Earlier conversation may concern a different version. Its separate review follows "
                "for context, not as proof of correctness:\n" + self._accepted_baseline.review_md
            )
        session_key = "solved-" + hashlib.sha256(baseline.encode()).hexdigest()
        editor_feedback = []
        _, baseline_pages, _ = await self._check(baseline, inp, deadline)
        candidate = await self._edit(ctx, inp, baseline, name=f"rewrite-{episode}", notes=notes,
                                     measured_pages=baseline_pages, session_key=session_key, session_baseline=baseline,
                                     editor_feedback=editor_feedback)
        for revision in range(inp.max_cleanup_repairs + 1):
            path = self.workdir / f"candidate-{episode}-{revision}.tex"
            path.write_text(candidate, encoding="utf-8")
            ok, pages, feedback = await self._check(candidate, inp, deadline)
            if asyncio.current_task().cancelling():
                raise asyncio.CancelledError
            critic = Batch3CleanupCritic(ctx, name=f"cleanup_critic-{episode}-{revision}")
            try:
                review = await critic(
                    problem=inp.problem, answer_tex=candidate, page_limit=inp.page_limit,
                    baseline_tex=baseline, prior_messages=history,
                    baseline_provenance=provenance,
                    mode="stateful", compile_feedback=feedback, editor_response="\n\n".join(editor_feedback),
                )
            except BaseException as exc:
                completed = critic.completed_review or getattr(exc, "completed_output", None)
                if isinstance(completed, Batch3CleanupCritic.Outputs) and completed.disposition == "research":
                    self._handoff(inp, baseline, candidate, completed)
                if (monetary_limit_exhausted(exc) and ok
                        and isinstance(completed, Batch3CleanupCritic.Outputs)
                        and completed.disposition == "accept"):
                    review = completed
                else:
                    raise
            history = review.messages_after
            _save_json(self.workdir / f"review-{episode}-{revision}.json", review.model_dump(mode="json"))
            if review.disposition == "research":
                # Persist the handoff before any later logging/cancellation can lose it.
                self._handoff(inp, baseline, candidate, review)
                return "research", candidate, pages, review
            if review.disposition == "accept" and ok:
                return "accept", candidate, pages, review
            report = review.review_md + "\nMechanical checks:\n" + feedback
            try:
                if review.parse_failed:
                    raise ValueError("cleanup critic returned an invalid routing verdict")
                if revision == inp.max_cleanup_repairs:
                    raise RuntimeError("cleanup repair limit reached without acceptance")
                editor_feedback.clear()
                repaired = await self._edit(
                    ctx, inp, candidate, name=f"repair-{episode}-{revision}", report=report,
                    measured_pages=pages, session_key=session_key, session_baseline=baseline,
                    editor_feedback=editor_feedback,
                    **({"restore_baseline": baseline} if review.disposition == "restore" else {}),
                )
                if repaired == candidate:
                    await self.events.emit("cleanup.repair_unchanged", {
                        "episode": episode, "revision": revision, "disposition": review.disposition,
                        "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                    })
                    raise RuntimeError("cleanup repair returned an unchanged manuscript; no repeat critic call was made")
                candidate = repaired
            except SubscriptionParked:
                raise
            except Exception as exc:
                feedback_review = review.model_copy(update={"review_md": report + f"\nEditorial interruption: {exc}"})
                # A restore verdict identifies an error in the rewrite, not
                # the accepted baseline. Failure to restore it changes neither.
                self._handoff(inp, baseline, candidate, feedback_review, mathematical=False)
                raise

    def _handoff(self, inp, baseline, candidate, review, *, mathematical=True, origin="Cleanup"):
        if mathematical:
            self._revoke_accepted(inp, baseline, review)
        workspace = self._workspace(inp)
        state = self._research_state(inp)
        if not state:
            raise RuntimeError("cannot return to research without its checkpoint")
        reason = (f"{origin} found a substantive mathematical flaw. Prior acceptance is revoked. "
                  "Repair the mathematics, not just the presentation. " if mathematical else
                  "Editorial cleanup could not finish. The edited candidate is not approved for submission. "
                  "Address the reported presentation/technical issues and obtain a fresh review. "
                  "This is not by itself a mathematical rejection; the exact accepted baseline remains a fallback. ")
        feedback = (reason + "Current answer.tex is the "
                    "edited candidate.\n\n" + review.review_md + "\n\nPre-rewrite manuscript:\n" + baseline)
        rejection = ACCritic.Outputs(review_md=feedback, answer_ready=False, mode="stateful",
                                     messages_after=review.messages_after)
        state.update(early_stopped=False, terminal_outputs=None, awaiting_finalization=False,
                     awaiting_author=None, awaiting_review_round=None, awaiting_review_kind="",
                     pending_critique=feedback, critic_conversation=review.messages_after,
                     critic_instance_turn=max(1, int(state.get("critic_instance_turn") or 0)))
        state["review_history"] = [*state.get("review_history", []), rejection.model_dump(mode="json", exclude={"messages_after"})]
        (workspace / "answer.tex").write_text(candidate, encoding="utf-8")
        _save_json(workspace / ".ac/resume-state.json", state)
        _save_json(self.workdir / "research-handoff.json", {"review": review.model_dump(mode="json"), "accepted": False})

    def _publish(self, inp, out, document, pages, *, partial):
        if not partial and self._requires_final_review(inp):
            review = self._final_approval(inp, document)
            if review is None:
                raise FinalReviewBlocked("Cannot publish an accepted solution without its exact final stateless approval")
            out.final_critic_review_md = review.review_md
        digest = hashlib.sha256(document.encode("utf-8")).hexdigest()
        path = published_document_path(self.ctx.root_workdir, inp.problem_id, partial=partial,
                                       digest=digest, version=1)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(document, encoding="utf-8")
        tmp.replace(path)
        updates = dict(
            answer_tex=path, publication_version=1, compiled=True, pages=pages, abstained=False,
            output_kind="partial_unreviewed" if partial else "accepted_solution",
            partial_ready=partial, submission_approved=not partial,
            partial_sha256=digest if partial else None, submission_sha256=None if partial else digest,
            early_stopped=not partial, final_critic_answer_ready=not partial,
            accepted_baseline_pending=False, accepted_baseline_path=None, accepted_baseline_sha256=None,
        )
        if not partial:
            updates.update(partial_reason="", error=None)
        # The manifest is the commit point. Neither an interrupted write nor
        # an orphaned new revision may invalidate the previously published one.
        published = out.model_copy(update=updates)
        _save_json(self.ctx.root_workdir / "batch3-output.json", published.model_dump(mode="json"))
        for field, value in updates.items():
            setattr(out, field, value)
        return out

    @staticmethod
    def _label_partial(document):
        document = normalize_submission_latex(document)
        legacy = ("\n\\section*{Partial result: not a complete solution}\n"
                  "This attempt did not receive final mathematical acceptance. "
                  "Its deadline cleanup has not been mathematically reviewed.\n")
        notice = r"\noindent\textbf{Partial result: not a complete solution (unreviewed).}\par"
        document = document.replace(legacy, "\n")
        # Ignore title commands in comments or verbatim examples. Keep offsets
        # intact so the original TeX, not a reconstructed document, is labelled.
        protected = re.compile(
            r"\\begin\{(verbatim\*?|lstlisting|comment)\}.*?\\end\{\1\}"
            r"|\\verb\*?(?P<delimiter>[^\w\s]).*?(?P=delimiter)"
            r"|\\[%\\]|%[^\n]*", re.DOTALL,
        )
        masked = protected.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), document)
        begin = re.search(r"\\begin\{document\}", masked)
        if begin is None:
            raise ValueError("No active \\begin{document}; repair the document structure before labelling")
        position = begin.end()
        title = r"\\maketitle\b(?:[ \t]*\[[^\]\n]*\])?"
        # Recognize a simple leading title group, not macro argument groups.
        grouped = re.match(r"\s*\{\s*" + title + r"\s*\}", masked[position:])
        if grouped:
            position += grouped.end()
        else:
            depth = 0
            for token in re.finditer(
                r"\\[{}]|[{}]|" + title
                + r"|\\if[A-Za-z@]*\b|\\(?:else|fi|begingroup|endgroup)\b|\\end\{document\}",
                masked[position:],
            ):
                if token[0] == "{":
                    depth += 1
                elif token[0] == "}":
                    depth -= 1
                elif depth == 0 and token[0].startswith(r"\maketitle"):
                    position += token.end()
                    break
                elif (token[0].startswith(r"\if")
                      or (depth == 0 and token[0] not in (r"\{", r"\}")
                          and not token[0].startswith(r"\maketitle"))):
                    # Do not place a notice inside a conditional or after the
                    # document end. TeX conditionals are not brace-delimited.
                    break
        if re.match(r"\s*" + re.escape(notice), masked[position:]):
            return document
        # Relocate old notices too: their mere presence in a macro or false
        # conditional does not mean the compiled PDF contains a visible label.
        for match in reversed(list(re.finditer(re.escape(notice), masked))):
            document = document[:match.start()] + document[match.end():]
            if match.start() < position:
                position -= len(notice)
        # Text before article's maketitle forces the title onto a new page.
        document = document[:position] + "\n" + notice + "\n" + document[position:]
        return normalize_submission_latex(document)

    async def _publish_accepted(self, inp, out, document, *, deadline, review_md=""):
        # The critic accepted this manuscript and only its editorial rewrite
        # failed, so it is submitted unpolished rather than labelled partial.
        if time.monotonic() >= deadline:
            return None
        if self._requires_final_review(inp) and not self._final_approval(inp, document):
            return None
        try:
            ok, pages, detail = await self._check(document, inp, deadline)
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            out.cleanup_errors.append(f"accepted baseline: {detail}")
            return None
        await self.events.emit("batch3.accepted_baseline_published", {"pages": pages, "reason": out.partial_reason})
        out.final_critic_review_md = review_md
        return self._publish(inp, out, document, pages, partial=False)

    async def _fallback(self, inp, out, document, *, deadline):
        if not document.strip() or time.monotonic() >= deadline:
            return document, 0, "no manuscript or no time left for export check"
        pages = 0
        try:
            document = await self._standalone(inp, document, deadline=deadline)
            labelled = self._label_partial(document)
            ok, pages, detail = await self._check(labelled, inp, deadline)
            if ok:
                self._publish(inp, out, labelled, pages, partial=True)
            elif f"fallback: {detail}" not in out.cleanup_errors:
                out.cleanup_errors.append(f"fallback: {detail}")
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            out.cleanup_errors.append(f"fallback: {detail}")
        return document, pages, detail

    async def _partial(self, inp, out, document, notes, report, *, deadline):
        if not document.strip():
            out.error = "No saved research attempt was available for partial cleanup"
            return out
        document, pages, detail = await self._fallback(inp, out, document, deadline=deadline)
        try:
            document = self._label_partial(document)
        except ValueError as exc:
            # Keep malformed input available to the editor instead of losing
            # the whole cleanup before it can repair the document structure.
            detail += f"\nPartial notice: {exc}"
        repairs = inp.max_partial_cleanup_repairs
        weights = ([1.0] if repairs == 0 else [0.75, 0.25] if repairs == 1 else
                   [0.75, 0.15] + [0.10 / (repairs - 1)] * (repairs - 1))
        export_seconds = min(300.0, max(0.0, deadline - time.monotonic()) * 0.1)
        session_key = "partial-" + hashlib.sha256(document.encode()).hexdigest()[:16]
        continuation = False
        continuation_mechanical = False
        continued = False
        try:
            candidate = document
            for revision, weight in enumerate(weights):
                seconds = max(0.0, deadline - time.monotonic())
                if seconds <= export_seconds or self._remaining_usd() <= 0:
                    break
                # Reserve both money and time for mechanical repairs, then an
                # exact-byte compile/export window after every model call.
                fraction = weight / sum(weights[revision:])
                # Keep a repair slot after a continuation: 90/15/10 model
                # minutes by default, followed by the separate export window.
                time_fraction = 18 / 23 if revision == 0 and repairs else fraction
                edit_deadline = time.monotonic() + (seconds - export_seconds) * time_fraction
                name = "partial-rewrite" if revision == 0 else f"partial-repair-{revision}"
                findings = report + "\nInitial mechanical checks:\n" + detail if revision == 0 or continuation else (
                    "Harness compilation/page check of this exact candidate:\n" + detail
                )
                # Both interrupted continuations and mechanical repairs finish
                # existing work; neither may launch optional reviewer tasks.
                try:
                    candidate = await self._phase(
                        "partial_continue" if continuation else "partial_rewrite" if revision == 0 else "partial_repair",
                        usd=self._remaining_usd() * fraction, deadline=edit_deadline,
                        call=lambda ctx: self._edit(ctx, inp, candidate, notes=notes if revision == 0 else "",
                                                    report=findings, partial=True, name=name,
                                                    mechanical=continuation_mechanical if continuation else revision > 0,
                                                    measured_pages=pages,
                                                    session_key=session_key, session_baseline=document,
                                                    finishing_only=revision > 0),
                    )
                except Exception as exc:
                    from proofstack.agents.cleanup_session import (
                        CleanupIncomplete, cleanup_accounting_unresolved, _read_file,
                    )
                    if (inp.cleanup_backend != "claude_code" or continued or revision + 2 > repairs
                            or not isinstance(exc, (TimeoutError, CleanupIncomplete))
                            or (deadline - time.monotonic() - export_seconds) * (
                                weights[revision + 1] / sum(weights[revision + 1:])
                            ) < inp.min_partial_continuation_seconds
                            or self._remaining_usd() <= 0
                            or cleanup_accounting_unresolved(self.ctx.root_workdir)):
                        raise
                    root = self.ctx.root_workdir / "cleanup_sessions" / session_key
                    state = json.loads(_read_file(root, "session.json"))
                    if not state.get("resumable") or state.get("in_flight", True):
                        raise
                    # These bytes are not published until the resumed editor
                    # completes the review protocol and exact export gates.
                    candidate = _read_file(root / "workspace", "answer.tex")
                    continuation_mechanical = revision > 0
                    continuation = continued = True
                    out.cleanup_errors.append(f"partial cleanup continued after {type(exc).__name__}: {exc}")
                    await self.events.emit("batch3.partial_continuation", {"reason": str(exc), "session_key": session_key})
                    continue
                continuation = False
                try:
                    candidate = self._label_partial(candidate)
                except ValueError as exc:
                    label_error = str(exc)
                else:
                    label_error = None
                (self.workdir / f"partial-candidate-{revision}.tex").write_text(candidate, encoding="utf-8")
                if label_error:
                    ok, pages, detail = False, 0, label_error
                else:
                    ok, pages, detail = await self._check(candidate, inp, deadline)
                _save_json(self.workdir / f"partial-check-{revision}.json", {
                    "ok": ok, "pages": pages, "page_limit": inp.page_limit, "detail": detail,
                })
                if ok:
                    return self._publish(inp, out, candidate, pages, partial=True)
                out.cleanup_errors.append(detail)
                if time.monotonic() >= deadline or self._remaining_usd() <= 0:
                    break
        except SubscriptionParked:
            raise
        except Exception as exc:
            out.cleanup_errors.append(f"partial cleanup: {type(exc).__name__}: {exc}")
        if out.answer_tex is None:
            out.error = "No compiling partial within the page limit; abstaining"
        _save_json(self.ctx.root_workdir / "batch3-output.json", out.model_dump(mode="json"))
        return out

    async def run(self, inp):
        schedule_path = self.ctx.root_workdir / "batch3-schedule.json"
        saved_schedule = _Schedule.model_validate_json(_read(schedule_path)) if inp.resume_run else None
        if saved_schedule is not None and saved_schedule.final_stateless_review:
            # Resume cannot turn off the gate or replenish its saved reserves.
            inp = inp.model_copy(update={
                "final_stateless_review": True,
                "final_review_reserve_usd": saved_schedule.final_review_reserve_usd,
                "final_review_seconds": saved_schedule.final_review_seconds,
                "final_review_export_seconds": saved_schedule.final_review_export_seconds,
            })
        final_required = self._requires_final_review(inp)
        final_reserve = inp.final_review_reserve_usd if final_required else 0
        final_window = inp.final_review_seconds + inp.final_review_export_seconds if final_required else 0
        out = self.Outputs(problem_id=inp.problem_id)
        self._accepted_baseline = self._restore_accepted(inp) if inp.resume_run else None
        self._reconcile_accepted(inp)
        if inp.resume_run:
            try:
                self._reconcile_simplifications(inp)
            except SimplificationInterrupted as exc:
                return self._preserve_pending_baseline(inp, out, exc)
        accounting_blocked = (self.ctx.root_workdir / "cleanup-accounting-uncertain.json").exists()
        if not accounting_blocked and (self.ctx.root_workdir / "cleanup_sessions").exists():
            from proofstack.agents.cleanup_session import cleanup_accounting_unresolved
            accounting_blocked = cleanup_accounting_unresolved(self.ctx.root_workdir)
        if accounting_blocked:
            out = self._restore_output(inp)
            out.error = "Cleanup usage is unresolved; reconcile before resuming paid work"
            out.error_retryable = False
            out.cleanup_errors.append(out.error)
            return out
        remaining = self.tracker.remaining_wallclock_s()
        if remaining is None or not math.isfinite(remaining) or remaining <= 0:
            raise ValueError("Batch 3 requires a finite run time budget")
        now = time.monotonic()
        deadline = now + remaining
        if inp.max_wallclock_s is not None:
            deadline = min(deadline, now + inp.max_wallclock_s)
        if inp.run_deadline_unix_s is not None:
            deadline = min(deadline, now + inp.run_deadline_unix_s - time.time())
        research_end = min(deadline, now + inp.research_seconds)
        if inp.research_deadline_unix_s is not None:
            research_end = min(research_end, now + inp.research_deadline_unix_s - time.time())
        research_end = min(research_end, deadline - inp.partial_cleanup_seconds - final_window)
        reserve = self._remaining_usd() * inp.partial_cleanup_reserve_fraction
        if inp.resume_run:
            schedule = saved_schedule
            if schedule.problem_hash != _problem_hash(inp.problem):
                raise ValueError("Batch 3 checkpoint belongs to a different problem")
            unix_now = time.time()
            deadline = min(deadline, now + schedule.run_deadline_unix_s - unix_now)
            research_end = min(research_end, now + schedule.research_deadline_unix_s - unix_now)
            research_end = min(research_end, deadline - inp.partial_cleanup_seconds - final_window)
            reserve = schedule.partial_reserve_usd
            if self.ctx.execution_backend != "subscription":
                from proofstack.provider_accounting import settle_provider_usage
                spent = await settle_provider_usage(self.ctx, self.tracker)
                self.tracker.counters.usd = max(self.tracker.counters.usd, spent)
            self.tracker.spec = (self.tracker.spec or BudgetSpec()).model_copy(
                update={"max_usd": min(schedule.initial_usd, self._remaining_usd() + self.tracker.planning_usd)})
        else:
            if final_required and (self._remaining_usd() <= reserve + final_reserve
                                   or deadline - now <= inp.partial_cleanup_seconds + final_window):
                raise ValueError("Run budget/time must exceed the final-review and cleanup reserves")
            schedule = _Schedule(
                problem_hash=_problem_hash(inp.problem),
                research_deadline_unix_s=time.time() + research_end - now,
                run_deadline_unix_s=time.time() + deadline - now,
                initial_usd=self._remaining_usd(), partial_reserve_usd=reserve,
            )
        new_final_policy = final_required and (saved_schedule is None or not saved_schedule.final_stateless_review)
        if new_final_policy:
            schedule = schedule.model_copy(update={
                "final_stateless_review": True, "final_review_reserve_usd": final_reserve,
                "final_review_seconds": inp.final_review_seconds,
                "final_review_export_seconds": inp.final_review_export_seconds,
                "research_deadline_unix_s": min(schedule.research_deadline_unix_s,
                                                time.time() + research_end - time.monotonic()),
            })
        if saved_schedule is None or new_final_policy:
            _save_json(schedule_path, schedule.model_dump())
        reserve += final_reserve
        _save_json(self.workdir / "schedule.json", schedule.model_dump())
        if inp.resume_run:
            out = self._restore_output(inp)
            if out.submission_approved:
                _save_json(self._final_review_pointer(), {})
                return out
            if final_required and self._final_review_pointer().exists():
                pending = json.loads(_read(self._final_review_pointer()))
                if pending.get("key"):
                    final_state = FinalReviewState.model_validate_json(_read(self._final_review_path(pending["key"])))
                    key, review_input = self._final_review_identity(inp, final_state.review_input.answer_tex)
                    if key != final_state.key or review_input != final_state.review_input:
                        raise FinalReviewBlocked("Pending final review configuration changed; reconcile before resuming")
                    result = await self._finish_final_candidate(
                        inp, out, final_state.review_input.answer_tex, deadline=deadline)
                    if result is not None:
                        return result
        if not inp.resume_run:
            _save_json(self._accepted_checkpoint(), {})
        # An explicit resume may follow a code fix; only this invocation's
        # failure classification should govern automatic retries.
        out.error_retryable = True
        document = notes = report = ""
        state = self._research_state(inp)
        round_bound = min(500, max(inp.n_rounds, int(state.get("n_rounds_at_checkpoint") or 0),
                                   int(state.get("next_round") or 0)))
        episode, recoveries, handoffs = 0, 0, 0
        while time.monotonic() < research_end and self._remaining_usd() > reserve:
            episode += 1
            in_cleanup = False
            retryable = True
            try:
                try:
                    research = await self._phase(
                        "research", usd=self._remaining_usd() - reserve, deadline=research_end,
                        call=lambda ctx: self._research(ctx, inp, resume=bool(self._research_state(inp)), round_bound=round_bound,
                                                         out=out, deadline=research_end),
                    )
                finally:
                    # Check every saved round even when a later round errors,
                    # times out, or the enclosing workflow is cancelled.
                    self._reconcile_accepted(inp)
                document = _read(Path(research.answer_tex))
                notes = _read(Path(research.research_notes_tex))
                out.rounds_completed = research.rounds_completed
                out.partial_reason = ""
                out.error = None
                state = self._research_state(inp)
                reviews = state.get("review_history") or []
                report = str(reviews[-1].get("review_md", "")) if reviews else ""
                if research.error:
                    retryable = research.error_retryable
                    raise RuntimeError(research.error)
                document, _, _ = await self._fallback(inp, out, document, deadline=deadline)
                if research.last_gasp:
                    out.partial_reason = "Research ended with a best-effort draft"
                    break
                self._reconcile_accepted(inp, exported_document=document)
                if research.early_stopped and research.last_critic_accepted is True and research.compiled:
                    in_cleanup = True
                    self._remember_accepted(inp, document, report)
                    document, notes, rejection = await self._simplify_accepted(
                        inp, document, notes, report, out, deadline=research_end, reserve=reserve,
                    )
                    if rejection is not None:
                        # The simplification critic found a flaw in the baseline,
                        # not just in its attempted replacement.
                        report = rejection.review_md
                        if out.rounds_completed >= 500:
                            out.partial_reason = "Research reached the supported 500-round limit"
                            break
                        round_bound = min(500, max(round_bound + inp.n_rounds,
                                                   out.rounds_completed + inp.n_rounds))
                        continue
                    # Preserve time for the one-shot partial rewrite even if an
                    # editorial candidate arrives near the research cutoff.
                    cleanup_end = deadline - max(inp.partial_cleanup_seconds,
                                                 min(3600.0, max(0.0, deadline - time.monotonic()) * 0.1))
                    cleanup_end = min(cleanup_end, deadline - final_window)
                    disposition, candidate, pages, review = await self._phase(
                        "candidate_cleanup", usd=self._remaining_usd() - reserve, deadline=cleanup_end,
                        call=lambda ctx: self._candidate_loop(ctx, inp, document, notes, episode=episode, deadline=cleanup_end),
                    )
                    out.final_critic_review_md = review.review_md
                    if disposition == "accept":
                        if not final_required:
                            return self._publish(inp, out, candidate, pages, partial=False)
                        result = await self._finish_final_candidate(inp, out, candidate, deadline=deadline)
                        if result is not None:
                            return result
                        if out.rounds_completed >= 500:
                            break
                        round_bound = min(500, max(round_bound + inp.n_rounds,
                                                   out.rounds_completed + inp.n_rounds))
                        continue
                    document, report = candidate, review.review_md
                    await self.events.emit("batch3.return_to_research", {"episode": episode, "round": out.rounds_completed})
                if not self._research_state(inp):
                    raise RuntimeError("research did not leave a resumable checkpoint")
                if out.rounds_completed >= 500:
                    out.partial_reason = "Research reached the supported 500-round limit"
                    break
                round_bound = min(500, max(round_bound + inp.n_rounds, out.rounds_completed + inp.n_rounds))
            except SubscriptionParked:
                raise
            except Exception as exc:
                out.partial_reason = f"{type(exc).__name__}: {exc}"
                out.cleanup_errors.append(out.partial_reason)
                if (not retryable or _is_programming_error(exc) or isinstance(exc, (CleanupContextTooLarge, CriticContextTooLarge))
                        or getattr(exc, "requires_reconciliation", False)):
                    # Repeating a paid phase cannot repair a broken code path.
                    # Preserve the last draft without another model invocation.
                    error = out.partial_reason
                    out.error = error
                    out.error_retryable = False
                    if isinstance(exc, SimplificationInterrupted):
                        return self._preserve_pending_baseline(inp, out, exc)
                    state = self._research_state(inp)
                    out.rounds_completed = max(out.rounds_completed, int(state.get("last_round_run") or 0))
                    _save_json(self.ctx.root_workdir / "batch3-output.json", out.model_dump(mode="json"))
                    if not await self._publish_retained(inp, out, deadline=deadline):
                        await self._fallback(inp, out, _read(self._workspace(inp) / "answer.tex") or document,
                                             deadline=deadline)
                    # Publishing clears the error; the operator must still see it.
                    out.error, out.partial_reason, out.error_retryable = error, error, False
                    _save_json(self.ctx.root_workdir / "batch3-output.json", out.model_dump(mode="json"))
                    await self.events.emit("batch3.recovery_blocked", {
                        "reason": out.error, "retryable": False, "operator_action_required": True,
                    })
                    return out
                if (not isinstance(exc, BudgetExhausted)
                        and (handoffs < inp.max_cleanup_handoffs if in_cleanup else recoveries < inp.max_recovery_attempts)
                        and time.monotonic() < research_end and self._remaining_usd() > reserve):
                    if in_cleanup:
                        handoffs += 1
                    else:
                        recoveries += 1
                    state = self._research_state(inp)
                    if in_cleanup and state:
                        # A latched mathematical rejection has already updated
                        # the checkpoint; don't overwrite its stronger findings.
                        if state.get("early_stopped"):
                            review = Batch3CleanupCritic.Outputs(
                                review_md=out.partial_reason, disposition="repair", answer_ready=False,
                                messages_after=state.get("critic_conversation") or [],
                            )
                            self._handoff(inp, document, document, review, mathematical=False)
                    elif state:
                        state.update(terminal_outputs=None, early_stopped=False, awaiting_finalization=False)
                        _save_json(self._workspace(inp) / ".ac/resume-state.json", state)
                    round_bound = min(500, max(round_bound, int(state.get("next_round") or 0)))
                    if int(state.get("next_round") or 0) > 500:
                        break
                    await self._fallback(inp, out, _read(self._workspace(inp) / "answer.tex") or document, deadline=deadline)
                    await self.events.emit("batch3.recovery", {"attempt": handoffs if in_cleanup else recoveries,
                                                            "cleanup_handoff": in_cleanup, "reason": out.partial_reason})
                    continue
                break
        if await self._publish_retained(inp, out, deadline=deadline, allow_final_review=True):
            return out
        workspace = self._workspace(inp)
        document = _read(workspace / "answer.tex") or document
        notes = _read(workspace / "research_notes.tex") or notes
        bibliography = _read(workspace / "references.bib")
        if bibliography:
            notes += "\n\nReference data for inlining citations (do not invent sources):\n" + bibliography
        state = self._research_state(inp)
        reviews = state.get("review_history") or []
        report = str(state.get("pending_critique") or (reviews[-1].get("review_md") if reviews else "") or report)
        if state.get("awaiting_review_round") is not None and state.get("awaiting_author"):
            report = (
                "Review provenance: the saved Author manuscript is UNREVIEWED. Any critic findings "
                "below refer to an earlier revision, not this draft. Compare them with the supplied "
                "manuscript; do not reintroduce obsolete caveats or assume claimed repairs are verified.\n\n"
                + report
            )
        out.rounds_completed = max(out.rounds_completed, int(state.get("last_round_run") or 0))
        out.partial_reason = out.partial_reason or "Research cutoff or reserved cleanup budget reached"
        report += "\n\nReason for stopping: " + out.partial_reason
        await self.events.emit("batch3.partial_cleanup", {"reason": out.partial_reason, "critic_follows": False})
        return await self._partial(inp, out, document, notes, report, deadline=deadline)


class FirstProofBatch3RehearsalWorkflow(FirstProofBatch3Workflow):
    """Explicit non-competition entry point; never selected by the submission image."""

    class Inputs(FirstProofBatch3Workflow.Inputs):
        page_limit: int = Field(default=16, ge=1, le=200)
