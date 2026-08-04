"""Executable visual Author/Critic workflow blocks."""
from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from proofstack.agents.ac.ac_workflow import (
    ACWorkflow,
    _CompileResult,
    _problem_hash,
    _resume_stop_round,
    _safe_read,
    _simple_compile_latex,
    _state_int,
)
from proofstack.agents.ac.author import Author
from proofstack.agents.ac.compute import Compute, render_compute_reply_for_author
from proofstack.agents.ac.council import render_council_replies_for_author
from proofstack.agents.ac.critic import ACCritic
from proofstack.budget import BudgetExhausted


class ACStateInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: dict[str, Any] = Field(default_factory=dict)


class ACJoinInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base_state: dict[str, Any] = Field(default_factory=dict)
    stateful_state: dict[str, Any] = Field(default_factory=dict)
    fresh_state: dict[str, Any] = Field(default_factory=dict)
    council_state: dict[str, Any] = Field(default_factory=dict)
    compute_state: dict[str, Any] = Field(default_factory=dict)


class ACCompileGateInput(ACStateInput):
    ready: bool = False


class ACStateOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: dict[str, Any] = Field(default_factory=dict)


class ACInitBlock(ACWorkflow):
    description: ClassVar[str] = "Initialize Author/Critic run state and workspace."
    PALETTE: ClassVar[dict[str, str]] = {
        "id": "ac_init_block",
        "label": "Problem",
        "group": "Author/Critic",
        "description": "Initializes the problem workspace and resume state.",
        "keywords": "author critic problem init resume workspace",
    }

    Inputs = ACWorkflow.Inputs
    Outputs = ACStateOutput

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
        await self._init_workspace(workspace, problem_text=original_problem)
        self._write_effective_problem(
            workspace, original=original_problem, effective=effective_problem
        )
        if resume_state is not None:
            self._restore_workspace_from_resume(workspace, resume_state)
            # Prior spend is re-seeded centrally by the launcher
            # (run_workflow._seed_budget_from_prior_events), not here.
            await self.events.emit(
                "ac.resume",
                {
                    "stop_round": resume_stop_round,
                    "next_round": resume_state.get("next_round"),
                    "awaiting_review_round": resume_state.get("awaiting_review_round"),
                    "source": resume_state.get("source", "checkpoint"),
                },
            )

        review_history = [
            ACCritic.Outputs.model_validate(raw).model_dump(mode="json")
            for raw in (resume_state or {}).get("review_history", [])
            if isinstance(raw, dict)
        ]
        pending_compute_zip_path = self._decode_run_path(
            (resume_state or {}).get("pending_compute_zip_path")
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
        resume_ready_to_noop = self._resume_ready_to_noop(inp, resume_state)
        awaiting_finalization = bool(
            (resume_state or {}).get("awaiting_finalization", False)
        ) or (
            resume_state is not None
            and early_stopped
            and not resume_ready_to_noop
        )
        noop = self._resume_has_no_new_work(
            inp=inp,
            state=resume_state,
            next_round=next_round,
            awaiting_review_round=awaiting_review_round,
            awaiting_finalization=awaiting_finalization,
            early_stopped=early_stopped,
        )
        if noop:
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

        state = {
            "inputs": inp.model_dump(mode="json"),
            "original_problem": original_problem,
            "workspace": str(workspace),
            "review_history": review_history,
            "critic_conversation": list((resume_state or {}).get("critic_conversation", [])),
            "critic_instance_turn": int((resume_state or {}).get("critic_instance_turn", 0) or 0),
            "pending_council_text": str((resume_state or {}).get("pending_council_text", "") or ""),
            "pending_compute_text": str((resume_state or {}).get("pending_compute_text", "") or ""),
            "pending_compute_zip_path": self._encode_run_path(pending_compute_zip_path),
            "pending_workflow_feedback": str((resume_state or {}).get("pending_workflow_feedback", "") or ""),
            "pending_critique": str((resume_state or {}).get("pending_critique", "") or ""),
            "last_round_run": last_round_run,
            "next_round": next_round,
            "n_rounds": inp.n_rounds,
            "early_stopped": early_stopped,
            "awaiting_review_round": awaiting_review_round,
            "awaiting_review_kind": str((resume_state or {}).get("awaiting_review_kind", "") or ""),
            "awaiting_author": (resume_state or {}).get("awaiting_author"),
            "awaiting_finalization": awaiting_finalization,
            "finalize_without_rounds": finalize_without_rounds,
            "noop": noop,
            "loop_done": noop or finalize_without_rounds,
        }
        return self.Outputs(state=state)


class _ACVisualStep(ACWorkflow):
    cache_enabled: ClassVar[bool] = False

    Inputs = ACStateInput
    Outputs = ACStateOutput
    HIDDEN_GRAPH_INPUTS: ClassVar[set[str]] = {"state"}

    def _state(self, raw: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(raw or {})

    def _inp(self, state: dict[str, Any]) -> ACWorkflow.Inputs:
        return ACWorkflow.Inputs.model_validate(state.get("inputs") or {})

    def _workspace(self, state: dict[str, Any]) -> Path:
        return Path(str(state.get("workspace") or ""))

    def _reviews(self, state: dict[str, Any]) -> list[ACCritic.Outputs]:
        return [
            ACCritic.Outputs.model_validate(raw)
            for raw in state.get("review_history", [])
            if isinstance(raw, dict)
        ]

    def _author(self, state: dict[str, Any]) -> Author.Outputs | None:
        raw = state.get("current_author") or state.get("awaiting_author")
        if not isinstance(raw, dict):
            return None
        return Author.Outputs.model_validate(raw)

    def _review(self, raw: Any) -> ACCritic.Outputs | None:
        if not isinstance(raw, dict):
            return None
        return ACCritic.Outputs.model_validate(raw)

    def _compute(self, raw: Any) -> Compute.Outputs | None:
        if not isinstance(raw, dict):
            return None
        return Compute.Outputs.model_validate(raw)

    def _save_visual_resume(
        self,
        state: dict[str, Any],
        *,
        awaiting_review_round: int | None = None,
        awaiting_review_kind: str | None = None,
        awaiting_author: Author.Outputs | None = None,
        awaiting_finalization: bool = False,
        terminal_outputs: dict[str, Any] | None = None,
    ) -> None:
        self._save_resume_state(
            self._workspace(state),
            inp=self._inp(state),
            last_round_run=int(state.get("last_round_run", -1) or -1),
            next_round=int(state.get("next_round", 0) or 0),
            review_history=self._reviews(state),
            critic_conversation=list(state.get("critic_conversation") or []),
            critic_instance_turn=int(state.get("critic_instance_turn", 0) or 0),
            pending_council_text=str(state.get("pending_council_text") or ""),
            pending_compute_text=str(state.get("pending_compute_text") or ""),
            pending_compute_zip_path=self._decode_run_path(state.get("pending_compute_zip_path")),
            pending_critique=str(state.get("pending_critique") or ""),
            early_stopped=bool(state.get("early_stopped", False)),
            awaiting_review_round=awaiting_review_round,
            awaiting_review_kind=awaiting_review_kind,
            awaiting_author=awaiting_author,
            pending_workflow_feedback=str(state.get("pending_workflow_feedback") or ""),
            awaiting_finalization=awaiting_finalization,
            terminal_outputs=terminal_outputs,
        )


class ACAuthorBlock(_ACVisualStep):
    description: ClassVar[str] = "Run the Author turn in the visual Author/Critic loop."
    PALETTE: ClassVar[dict[str, str]] = {
        "id": "ac_author_block",
        "label": "Author",
        "group": "Author/Critic",
        "description": "Writes the answer/research-notes/BibTeX triple for the current round.",
        "keywords": "author critic firstproof write proof",
    }

    class Outputs(ACStateOutput):
        answer_tex: str = ""
        research_notes_tex: str = ""
        references_bib: str = ""
        ready: bool = False
        council_request: str = ""
        compute_request: str = ""

    async def run(self, inp):  # type: ignore[override]
        state = self._state(inp.state)
        if state.get("loop_done"):
            return self.Outputs(state=state)
        ac_inp = self._inp(state)
        workspace = self._workspace(state)

        awaiting_review_round = state.get("awaiting_review_round")
        awaiting_author_raw = state.get("awaiting_author")
        if awaiting_review_round is not None and isinstance(awaiting_author_raw, dict):
            k = int(awaiting_review_round)
            author = Author.Outputs.model_validate(awaiting_author_raw)
            await self.events.emit(
                "ac.resume_missing_review_start",
                {
                    "round": k,
                    "kind": state.get("awaiting_review_kind") or "round_review",
                    "n_rounds": ac_inp.n_rounds,
                },
            )
        else:
            k = int(state.get("next_round", 0) or 0)
            if k > ac_inp.n_rounds:
                state["loop_done"] = True
                return self.Outputs(state=state)
            await self.events.emit("ac.round_start", {"round": k, "n_rounds": ac_inp.n_rounds})
            if k <= 0:
                prev_critique = ""
                prev_council = ""
                prev_compute = ""
                compute_zip_path = None
            else:
                reviews = self._reviews(state)
                prev_critique = state.get("pending_critique") or (reviews[-1].review_md if reviews else "")
                state["pending_critique"] = ""
                prev_council = str(state.get("pending_council_text") or "")
                prev_compute = str(state.get("pending_compute_text") or "")
                compute_zip_path = self._decode_run_path(state.get("pending_compute_zip_path"))

            author = await self._call_author_with_operator_comments(
                **self._author_inputs(
                    inp=ac_inp,
                    workspace=workspace,
                    prev_critique=prev_critique,
                    prev_council=prev_council,
                    workflow_feedback=str(state.get("pending_workflow_feedback") or ""),
                    prev_compute_response=prev_compute,
                    compute_zip_path=compute_zip_path,
                    round=k,
                )
            )
            self._write_files_from_author(workspace, author)
            self._write_author_artifacts(workspace, author, round=k)
            state["pending_workflow_feedback"] = await self._compile_feedback_after_author(
                workspace, page_limit=ac_inp.page_limit, round=k
            )
            if k >= 1:
                state["pending_council_text"] = ""
                state["pending_compute_text"] = ""
                state["pending_compute_zip_path"] = None
            self._save_visual_resume(
                state,
                awaiting_review_round=k,
                awaiting_review_kind="round_review",
                awaiting_author=author,
            )

        if state.get("awaiting_review_round") is not None:
            mode, conversation, instance_turn, omit_thinking = self._critic_mode_for_resume_review(
                inp=ac_inp,
                round=k,
                critic_conversation=list(state.get("critic_conversation") or []),
                critic_instance_turn=int(state.get("critic_instance_turn", 0) or 0),
                awaiting_review_kind=str(state.get("awaiting_review_kind") or ""),
            )
        elif k <= 0:
            mode, conversation, instance_turn, omit_thinking = "fresh", [], 0, False
        else:
            mode, conversation, instance_turn = self._critic_mode_for_round(
                inp=ac_inp,
                round=k,
                critic_conversation=list(state.get("critic_conversation") or []),
                critic_instance_turn=int(state.get("critic_instance_turn", 0) or 0),
            )
            omit_thinking = False

        requested_council = ac_inp.enable_council and author.council_question and bool(ac_inp.council_models)
        requested_compute = ac_inp.enable_compute and bool(author.compute_instructions)
        terminal_auxiliary_blocked, kinds, pending_terminal_auxiliary = self._terminal_auxiliary_decision(
            inp=ac_inp,
            round=k,
            requested_council=bool(requested_council),
            requested_compute=bool(requested_compute),
        )
        if terminal_auxiliary_blocked:
            await self.events.emit("ac.terminal_auxiliary_suppressed", {"round": k, "kinds": kinds})
            state["pending_critique"] = pending_terminal_auxiliary

        state.update(
            {
                "current_round": k,
                "current_author": author.model_dump(mode="json"),
                "critic_mode": mode,
                "critic_conversation": conversation,
                "critic_instance_turn": instance_turn,
                "omit_author_thinking": omit_thinking,
                "run_council": bool(requested_council) and not terminal_auxiliary_blocked,
                "run_compute": bool(requested_compute) and not terminal_auxiliary_blocked,
                "terminal_auxiliary_blocked": terminal_auxiliary_blocked,
                "round_review": None,
                "stateful_review": None,
                "fresh_review": None,
                "forced_fresh_review": None,
                "council_replies": [],
                "compute_out": None,
                "ready_for_gate": False,
            }
        )
        return self.Outputs(
            state=state,
            answer_tex=author.answer_tex,
            research_notes_tex=author.research_notes_tex,
            references_bib=author.references_bib,
            ready=author.ready,
            council_request=author.council_question or "",
            compute_request=author.compute_instructions or "",
        )


class ACStatefulCriticBlock(_ACVisualStep):
    description: ClassVar[str] = "Run the continuing Critic when the current round is stateful."
    PALETTE: ClassVar[dict[str, str]] = {
        "id": "ac_stateful_critic_block",
        "label": "Stateful Critic",
        "group": "Author/Critic",
        "description": "Continuing Critic instance for non-reset rounds.",
        "keywords": "author critic stateful review referee",
    }

    class Outputs(ACStateOutput):
        review_md: str = ""
        answer_ready: bool = False

    async def run(self, inp):  # type: ignore[override]
        state = self._state(inp.state)
        author = self._author(state)
        if state.get("critic_mode") != "stateful" or author is None:
            return self.Outputs(state=state)
        ac_inp = self._inp(state)
        k = int(state.get("current_round", 0) or 0)
        review = await self.critic(
            **self._critic_inputs(
                inp=ac_inp,
                workspace=self._workspace(state),
                author_thinking=author.thinking_summary,
                mode="stateful",
                prior_messages=list(state.get("critic_conversation") or []),
                round=k,
                omit_author_thinking=False,
            )
        )
        dumped = review.model_dump(mode="json")
        state["stateful_review"] = dumped
        state["round_review"] = dumped
        return self.Outputs(state=state, review_md=review.review_md, answer_ready=review.answer_ready)


class ACFreshCriticBlock(_ACVisualStep):
    description: ClassVar[str] = "Run the fresh Critic for reset rounds or forced-fresh promotion."
    PALETTE: ClassVar[dict[str, str]] = {
        "id": "ac_fresh_critic_block",
        "label": "Fresh Critic",
        "group": "Author/Critic",
        "description": "Independent Critic instance for fresh reviews and forced-fresh promotion.",
        "keywords": "author critic fresh independent audit review",
    }

    class Outputs(ACStateOutput):
        review_md: str = ""
        answer_ready: bool = False

    async def run(self, inp):  # type: ignore[override]
        state = self._state(inp.state)
        author = self._author(state)
        if author is None:
            return self.Outputs(state=state)
        ac_inp = self._inp(state)
        k = int(state.get("current_round", 0) or 0)
        review: ACCritic.Outputs | None = None
        if state.get("critic_mode") == "fresh":
            review = await self.critic(
                **self._critic_inputs(
                    inp=ac_inp,
                    workspace=self._workspace(state),
                    author_thinking=author.thinking_summary,
                    mode="fresh",
                    prior_messages=list(state.get("critic_conversation") or []),
                    round=k,
                    omit_author_thinking=bool(state.get("omit_author_thinking", False)),
                )
            )
            dumped = review.model_dump(mode="json")
            state["fresh_review"] = dumped
            state["round_review"] = dumped
        else:
            stateful = self._review(state.get("stateful_review"))
            should_force = (
                stateful is not None
                and author.ready
                and stateful.answer_ready
                and not bool(state.get("run_compute"))
                and not bool(state.get("terminal_auxiliary_blocked"))
            )
            if should_force:
                review = await self.critic(
                    **self._critic_inputs(
                        inp=ac_inp,
                        workspace=self._workspace(state),
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
                        "answer_ready": review.answer_ready,
                        "parse_failed": review.parse_failed,
                    },
                )
                self._write_forced_fresh_artifacts(self._workspace(state), review, round=k)
                state["forced_fresh_review"] = review.model_dump(mode="json")
        return self.Outputs(
            state=state,
            review_md=review.review_md if review is not None else "",
            answer_ready=review.answer_ready if review is not None else False,
        )


class ACCouncilBlock(_ACVisualStep):
    description: ClassVar[str] = "Run the optional Advisory Council requested by the Author."
    PALETTE: ClassVar[dict[str, str]] = {
        "id": "ac_council_block",
        "label": "LLM Council",
        "group": "Author/Critic",
        "description": "Optional parallel council of additional models requested by the Author.",
        "keywords": "author critic council advisory optional parallel",
    }

    class Outputs(ACStateOutput):
        feedback: str = ""

    async def run(self, inp):  # type: ignore[override]
        state = self._state(inp.state)
        author = self._author(state)
        if not state.get("run_council") or author is None:
            return self.Outputs(state=state)
        ac_inp = self._inp(state)
        k = int(state.get("current_round", 0) or 0)
        member_models = await self._council_member_models(
            round=k,
            requested=list(author.council_to),
            allowed=list(ac_inp.council_models),
        )
        replies = await self._safe_council(
            round=k,
            author_question=author.council_question or "",
            answer_tex=author.answer_tex,
            research_notes_tex=author.research_notes_tex,
            references_bib=author.references_bib,
            member_models=member_models,
        )
        state["council_replies"] = [reply.model_dump(mode="json") for reply in (replies.replies if replies else [])]
        text = render_council_replies_for_author(replies.replies) if replies and replies.replies else ""
        return self.Outputs(state=state, feedback=text)


class ACComputeBlock(_ACVisualStep):
    description: ClassVar[str] = "Run the optional Compute worker requested by the Author."
    PALETTE: ClassVar[dict[str, str]] = {
        "id": "ac_compute_block",
        "label": "Compute Node",
        "group": "Author/Critic",
        "description": "Optional CAS/Codex worker requested by the Author.",
        "keywords": "author critic compute codex cas optional parallel",
    }

    class Outputs(ACStateOutput):
        response_md: str = ""
        workspace_zip: str | None = None
        status: str = ""

    async def run(self, inp):  # type: ignore[override]
        state = self._state(inp.state)
        author = self._author(state)
        if not state.get("run_compute") or author is None:
            return self.Outputs(state=state)
        ac_inp = self._inp(state)
        workspace = self._workspace(state)
        compute_workspace = workspace / "compute"
        compute_workspace.mkdir(parents=True, exist_ok=True)
        out = await self._safe_compute(
            inp=ac_inp,
            round=int(state.get("current_round", 0) or 0),
            instructions=author.compute_instructions or "",
            answer_tex=author.answer_tex,
            research_notes_tex=author.research_notes_tex,
            references_bib=author.references_bib,
            compute_workspace=compute_workspace,
        )
        if out is not None:
            state["compute_out"] = out.model_dump(mode="json")
        return self.Outputs(
            state=state,
            response_md=out.response_md if out is not None else "",
            workspace_zip=self._encode_run_path(out.zip_path) if out is not None else None,
            status=out.status if out is not None else "",
        )


class ACReviewJoinBlock(_ACVisualStep):
    description: ClassVar[str] = "Join Critic, Council, and Compute results for the round."
    Inputs = ACJoinInput
    HIDDEN_GRAPH_INPUTS: ClassVar[set[str]] = {
        "base_state",
        "stateful_state",
        "fresh_state",
        "council_state",
        "compute_state",
    }
    PALETTE: ClassVar[dict[str, str]] = {
        "id": "ac_review_join_block",
        "label": "Review Join",
        "group": "Author/Critic",
        "description": "Combines round review and auxiliary outputs before the ready gate.",
        "keywords": "author critic join ready gate",
    }

    class Outputs(ACStateOutput):
        ready_for_gate: bool = False
        review_md: str = ""

    async def run(self, inp):  # type: ignore[override]
        state = self._state(inp.base_state)
        for branch in (inp.stateful_state, inp.fresh_state, inp.council_state, inp.compute_state):
            for key in (
                "stateful_review",
                "fresh_review",
                "forced_fresh_review",
                "round_review",
                "council_replies",
                "compute_out",
            ):
                value = (branch or {}).get(key)
                if value:
                    state[key] = value

        author = self._author(state)
        review = self._review(state.get("round_review"))
        if author is None or review is None:
            return self.Outputs(state=state)

        workspace = self._workspace(state)
        k = int(state.get("current_round", 0) or 0)
        council_replies = state.get("council_replies") or []
        compute_out = self._compute(state.get("compute_out"))
        if council_replies:
            from proofstack.agents.ac.council import CouncilReply

            replies = [CouncilReply.model_validate(raw) for raw in council_replies]
            state["pending_council_text"] = render_council_replies_for_author(replies)
        if compute_out is not None:
            state["pending_compute_text"] = render_compute_reply_for_author(compute_out)
            state["pending_compute_zip_path"] = self._encode_run_path(compute_out.zip_path)
            self._write_compute_artifacts(workspace, compute_out, round=k)

        reviews = self._reviews(state)
        reviews.append(review)
        state["review_history"] = [item.model_dump(mode="json") for item in reviews]
        state["critic_conversation"] = list(review.messages_after)
        state["critic_instance_turn"] = int(state.get("critic_instance_turn", 0) or 0) + 1
        self._write_review_artifacts(workspace, review, round=k)

        compute_blocks_ship = compute_out is not None
        if compute_blocks_ship and author.ready and review.answer_ready:
            await self.events.emit(
                "ac.early_stop_deferred_for_compute",
                {"round": k, "compute_status": compute_out.status if compute_out is not None else None},
            )
        if state.get("terminal_auxiliary_blocked") and author.ready and review.answer_ready:
            await self.events.emit("ac.early_stop_deferred_for_terminal_auxiliary", {"round": k})

        ready_for_gate = False
        if author.ready and review.answer_ready and not compute_blocks_ship and not state.get("terminal_auxiliary_blocked"):
            if review.mode == "stateful":
                forced = self._review(state.get("forced_fresh_review"))
                if forced is not None and not forced.answer_ready:
                    state["critic_conversation"] = list(forced.messages_after)
                    state["critic_instance_turn"] = 1
                    state["pending_critique"] = (
                        "## Stateful reviewer's report\n\n"
                        + (review.review_md or "")
                        + "\n\n---\n\n"
                        + "## Independent fresh reviewer's report\n\n"
                        + (forced.review_md or "")
                    )
                elif forced is not None and forced.answer_ready:
                    state["review_for_gate"] = forced.model_dump(mode="json")
                    ready_for_gate = True
            else:
                state["review_for_gate"] = review.model_dump(mode="json")
                ready_for_gate = True
        state["ready_for_gate"] = ready_for_gate
        return self.Outputs(state=state, ready_for_gate=ready_for_gate, review_md=review.review_md)


class ACCompileGateBlock(_ACVisualStep):
    description: ClassVar[str] = "Apply the deterministic compile/page gate."
    Inputs = ACCompileGateInput
    HIDDEN_GRAPH_INPUTS: ClassVar[set[str]] = {"state", "ready"}
    PALETTE: ClassVar[dict[str, str]] = {
        "id": "ac_compile_gate_block",
        "label": "Compile Gate",
        "group": "Author/Critic",
        "description": "Checks compile/page constraints before returning or continuing.",
        "keywords": "author critic compile gate ready",
    }

    async def run(self, inp):  # type: ignore[override]
        state = self._state(inp.state)
        author = self._author(state)
        review = self._review(state.get("round_review"))
        if author is None or review is None:
            return self.Outputs(state=state)
        workspace = self._workspace(state)
        ac_inp = self._inp(state)
        k = int(state.get("current_round", 0) or 0)
        forced = self._review(state.get("forced_fresh_review"))
        compute_out = self._compute(state.get("compute_out"))
        council_replies = []
        if state.get("council_replies"):
            from proofstack.agents.ac.council import CouncilReply

            council_replies = [CouncilReply.model_validate(raw) for raw in state.get("council_replies") or []]

        if inp.ready:
            gate_ok, gate_reasons = await self._deterministic_ready(workspace, page_limit=ac_inp.page_limit)
            if gate_ok:
                review_for_gate = self._review(state.get("review_for_gate")) or review
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
                    workspace,
                    round=k,
                    author=author,
                    review=review,
                    council_replies=council_replies,
                    forced_fresh_review=forced,
                    compute_out=compute_out,
                )
                state["last_round_run"] = k
                state["early_stopped"] = True
                state["loop_done"] = True
                state["next_round"] = k + 1
                self._save_visual_resume(state, awaiting_finalization=True)
                return self.Outputs(state=state)
            await self.events.emit("ac.deterministic_gate_blocked", {"round": k, "reasons": gate_reasons})
            review_for_gate = self._review(state.get("review_for_gate")) or review
            state["pending_critique"] = (
                (review_for_gate.review_md or "")
                + "\n\n## Workflow rejection\n\n"
                + "The ship-gate blocked early-stop for the following reasons: "
                + ", ".join(gate_reasons)
                + ".\nPlease address these before retrying."
            )

        self._snapshot_round(
            workspace,
            round=k,
            author=author,
            review=review,
            council_replies=council_replies,
            forced_fresh_review=forced,
            compute_out=compute_out,
        )
        state["last_round_run"] = k
        state["next_round"] = k + 1
        state["awaiting_review_round"] = None
        state["awaiting_review_kind"] = None
        state["awaiting_author"] = None
        if not state.get("pending_critique") and not state.get("terminal_auxiliary_blocked"):
            state["pending_critique"] = ""
        self._save_visual_resume(state)
        return self.Outputs(state=state)


class ACReturnBlock(ACWorkflow):
    description: ClassVar[str] = "Finalize and return the Author/Critic workflow outputs."
    Inputs = ACStateInput
    Outputs = ACWorkflow.Outputs
    cache_enabled: ClassVar[bool] = False
    HIDDEN_GRAPH_INPUTS: ClassVar[set[str]] = {"state"}
    PALETTE: ClassVar[dict[str, str]] = {
        "id": "ac_return_block",
        "label": "Return",
        "group": "Author/Critic",
        "description": "Runs final compile/stash and returns the final files.",
        "keywords": "author critic return output final compile",
    }

    async def run(self, inp):  # type: ignore[override]
        state = copy.deepcopy(inp.state or {})
        ac_inp = ACWorkflow.Inputs.model_validate(state.get("inputs") or {})
        workspace = Path(str(state.get("workspace") or self._workspace_path(ac_inp.problem_id, ac_inp.problem)))
        if state.get("noop"):
            return self._outputs_from_existing_resume(inp=ac_inp, workspace=workspace, state=self._load_resume_state(workspace) or {})

        answer_tex_text = _safe_read(workspace / "answer.tex")
        bib_path = workspace / "references.bib"
        bib_arg = bib_path if bib_path.exists() and bib_path.stat().st_size > 0 else None
        fixed = await asyncio.to_thread(
            _simple_compile_latex,
            answer_tex_text,
            bib_path=bib_arg,
            page_limit=ac_inp.page_limit,
            is_full_document=True,
        )
        ac_dir = workspace / ".ac"
        ac_dir.mkdir(parents=True, exist_ok=True)
        self._write_compile_artifact(
            ac_dir / "final-compile.log",
            fixed,
            page_limit=ac_inp.page_limit,
            title="Final compile check",
        )
        (workspace / "answer.tex").write_text(fixed.tex, encoding="utf-8")
        page_overflow = fixed.compiled and fixed.pages > ac_inp.page_limit
        await self.events.emit(
            "ac.final_compile",
            {
                "compiled": fixed.compiled,
                "pages": fixed.pages,
                "page_limit": ac_inp.page_limit,
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

        reviews = [
            ACCritic.Outputs.model_validate(raw)
            for raw in state.get("review_history", [])
            if isinstance(raw, dict)
        ]
        last_round_run = int(state.get("last_round_run", -1) or -1)
        last_critic_accepted = reviews[-1].answer_ready if reviews else None
        final_critic_answer_ready = False
        final_critic_mode_run = "not_run"
        final_critic_review_md = ""
        terminal_review_history = reviews
        terminal_critic_conversation = list(state.get("critic_conversation") or [])
        terminal_critic_instance_turn = int(state.get("critic_instance_turn", 0) or 0)
        terminal_pending_council_text = str(state.get("pending_council_text") or "")
        terminal_pending_compute_text = str(state.get("pending_compute_text") or "")
        terminal_pending_compute_zip_path = self._decode_run_path(state.get("pending_compute_zip_path"))
        terminal_pending_critique = str(state.get("pending_critique") or "")
        terminal_pending_workflow_feedback = str(state.get("pending_workflow_feedback") or "")
        if ac_inp.enable_final_critic:
            try:
                final_review = await self.critic(
                    **self._critic_inputs(
                        inp=ac_inp,
                        workspace=workspace,
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
                (ac_dir / "final-review.md").write_text(final_review.review_md, encoding="utf-8")
                (ac_dir / "final-review.json").write_text(final_review.model_dump_json(indent=2), encoding="utf-8")
                await self.events.emit(
                    "ac.final_critic_review",
                    {"answer_ready": final_review.answer_ready, "parse_failed": final_review.parse_failed},
                )
                terminal_review_history = [*reviews, final_review]
                terminal_critic_conversation = list(final_review.messages_after)
                terminal_critic_instance_turn = 1
                terminal_pending_council_text = ""
                terminal_pending_compute_text = ""
                terminal_pending_compute_zip_path = None
                terminal_pending_critique = final_review.review_md
            except BudgetExhausted as e:
                if e.scope == "run":
                    raise
                await self.events.emit("ac.final_critic_skipped", {"reason": "budget_exhausted", "scope": e.scope})
                final_critic_mode_run = "skipped_budget_exhausted"
            except Exception as e:
                await self.events.emit("ac.final_critic_failed", {"type": type(e).__name__, "msg": str(e)})
                final_critic_mode_run = "failed"

        final_bbl = fixed.bbl_path if fixed and getattr(fixed, "bbl_path", None) and fixed.bbl_path.exists() else None
        answer_path = self._stash_answer(
            ac_inp.problem_id,
            fixed.tex,
            bbl_path=final_bbl,
            bib_path=bib_arg,
            ship_bib_alongside=ac_inp.ship_bib_alongside,
        )
        terminal_outputs = {
            "answer_tex": self._encode_run_path(answer_path),
            "compiled": fixed.compiled,
            "pages": fixed.pages,
            "rounds_completed": last_round_run,
            "early_stopped": bool(state.get("early_stopped", False)),
            "last_critic_accepted": last_critic_accepted,
            "final_critic_answer_ready": final_critic_answer_ready,
            "final_critic_mode_run": final_critic_mode_run,
            "final_critic_review_md": final_critic_review_md,
            "last_gasp": False,
            "error": None,
        }
        self._save_resume_state(
            workspace,
            inp=ac_inp,
            last_round_run=last_round_run,
            next_round=last_round_run + 1,
            review_history=terminal_review_history,
            critic_conversation=terminal_critic_conversation,
            critic_instance_turn=terminal_critic_instance_turn,
            pending_council_text=terminal_pending_council_text,
            pending_compute_text=terminal_pending_compute_text,
            pending_compute_zip_path=terminal_pending_compute_zip_path,
            pending_critique=terminal_pending_critique,
            early_stopped=bool(state.get("early_stopped", False)),
            pending_workflow_feedback=terminal_pending_workflow_feedback,
            terminal_outputs=terminal_outputs,
        )
        if final_bbl is not None:
            try:
                final_bbl.unlink()
            except OSError:
                pass
        return self.Outputs(
            problem_id=ac_inp.problem_id,
            answer_tex=answer_path,
            research_notes_tex=workspace / "research_notes.tex",
            references_bib=workspace / "references.bib",
            compiled=fixed.compiled,
            pages=fixed.pages,
            rounds_completed=last_round_run,
            early_stopped=bool(state.get("early_stopped", False)),
            last_critic_accepted=last_critic_accepted,
            final_critic_answer_ready=final_critic_answer_ready,
            final_critic_mode_run=final_critic_mode_run,
            final_critic_review_md=final_critic_review_md,
            last_gasp=False,
        )


__all__ = [
    "ACInitBlock",
    "ACAuthorBlock",
    "ACStatefulCriticBlock",
    "ACFreshCriticBlock",
    "ACCouncilBlock",
    "ACComputeBlock",
    "ACReviewJoinBlock",
    "ACCompileGateBlock",
    "ACReturnBlock",
]
