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
from proofstack.agents.batch3_critic import Batch3CleanupCritic, CleanupContextTooLarge
from proofstack.agents.ac.critic import CriticContextTooLarge
from proofstack.agents.pwc.workspace import embed_or_ship_bibliography
from proofstack.agents.dag_workflow import _deep_merge
from proofstack.agents.writeup_loop import (
    RewriteSeat, RepairSeat, _assemble, _extract_tex, _defuse_proof_sectioning, _declares,
    _compile_raw, _GateCanceller,
)
from proofstack.budget import BudgetExhausted, BudgetRegistry, BudgetSpec, SubscriptionParked
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


class _AcceptedBaseline(BaseModel):
    problem_hash: str
    problem_id: str
    document: str
    sha256: str
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_bib_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    review_md: str
    round: int = Field(ge=0)


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
        cleanup_errors: list[str] = Field(default_factory=list)
        error: str | None = None
        error_retryable: bool = True

    def _remaining_usd(self) -> float:
        limits = [n.spec.max_usd - n.counters.usd for n in self.tracker.chain()
                  if n.spec is not None and n.spec.max_usd is not None and n.spec.max_usd >= 0]
        if not limits or not math.isfinite(min(limits)):
            raise ValueError("Batch 3 requires an explicit finite run USD budget")
        return max(0.0, min(limits))

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
            if (out.problem_id == inp.problem_id and out.compiled and 0 < out.pages <= inp.page_limit
                    and (partial or out.submission_approved) and text
                    and hashlib.sha256(text.encode("utf-8")).hexdigest() == digest
                    and normalize_submission_latex(text) == text):
                return out.model_copy(update={"answer_tex": path})
        except ValueError:
            pass
        return self.Outputs(problem_id=inp.problem_id)

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
            await self.events.emit("batch3.phase_end", {"phase": name, "cost_usd": ctx.budgets.root().counters.usd})

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

    async def _publish_retained(self, inp, out, *, deadline):
        self._reconcile_accepted(inp)
        baseline = self._accepted_baseline
        if baseline is None:
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
            if (not partial or isinstance(exc, SubscriptionParked) or exc.limit_kind != "usd"
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

    async def _candidate_loop(self, ctx, inp, baseline, notes, *, episode, deadline):
        history = list(self._research_state(inp).get("critic_conversation") or [])
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
                    mode="stateful", compile_feedback=feedback, editor_response="\n\n".join(editor_feedback),
                )
            except BaseException as exc:
                completed = critic.completed_review or getattr(exc, "completed_output", None)
                if isinstance(completed, Batch3CleanupCritic.Outputs) and completed.disposition == "research":
                    self._handoff(inp, baseline, candidate, completed)
                if (isinstance(exc, BudgetExhausted) and not isinstance(exc, SubscriptionParked)
                        and exc.limit_kind == "usd" and ok
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

    def _handoff(self, inp, baseline, candidate, review, *, mathematical=True):
        if mathematical:
            self._revoke_accepted(inp, baseline, review)
        workspace = self._workspace(inp)
        state = self._research_state(inp)
        if not state:
            raise RuntimeError("cannot return to research without its checkpoint")
        reason = ("Cleanup found a substantive mathematical flaw. Prior acceptance is revoked. "
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
        out = self.Outputs(problem_id=inp.problem_id)
        self._accepted_baseline = self._restore_accepted(inp) if inp.resume_run else None
        self._reconcile_accepted(inp)
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
        research_end = min(research_end, deadline - inp.partial_cleanup_seconds)
        reserve = self._remaining_usd() * inp.partial_cleanup_reserve_fraction
        schedule_path = self.ctx.root_workdir / "batch3-schedule.json"
        if inp.resume_run:
            schedule = _Schedule.model_validate_json(_read(schedule_path))
            if schedule.problem_hash != _problem_hash(inp.problem):
                raise ValueError("Batch 3 checkpoint belongs to a different problem")
            unix_now = time.time()
            deadline = min(deadline, now + schedule.run_deadline_unix_s - unix_now)
            research_end = min(research_end, now + schedule.research_deadline_unix_s - unix_now)
            reserve = schedule.partial_reserve_usd
            from proofstack.provider_accounting import settle_provider_usage
            spent = await settle_provider_usage(self.ctx, self.tracker)
            self.tracker.counters.usd = max(self.tracker.counters.usd, spent)
            self.tracker.spec = (self.tracker.spec or BudgetSpec()).model_copy(
                update={"max_usd": min(schedule.initial_usd, self._remaining_usd() + self.tracker.counters.usd)})
            out = self._restore_output(inp)
            if out.submission_approved:
                return out
        else:
            schedule = _Schedule(
                problem_hash=_problem_hash(inp.problem),
                research_deadline_unix_s=time.time() + research_end - now,
                run_deadline_unix_s=time.time() + deadline - now,
                initial_usd=self._remaining_usd(), partial_reserve_usd=reserve,
            )
            _save_json(schedule_path, schedule.model_dump())
        _save_json(self.workdir / "schedule.json", schedule.model_dump())
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
                    # Preserve time for the one-shot partial rewrite even if an
                    # editorial candidate arrives near the research cutoff.
                    cleanup_end = deadline - max(inp.partial_cleanup_seconds,
                                                 min(3600.0, max(0.0, deadline - time.monotonic()) * 0.1))
                    disposition, candidate, pages, review = await self._phase(
                        "candidate_cleanup", usd=self._remaining_usd() - reserve, deadline=cleanup_end,
                        call=lambda ctx: self._candidate_loop(ctx, inp, document, notes, episode=episode, deadline=cleanup_end),
                    )
                    out.final_critic_review_md = review.review_md
                    if disposition == "accept":
                        return self._publish(inp, out, candidate, pages, partial=False)
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
        if await self._publish_retained(inp, out, deadline=deadline):
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
