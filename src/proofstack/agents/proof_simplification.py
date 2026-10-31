"""Review protocol and durable state for optional post-acceptance simplification."""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, Field

from proofstack.agents.ac.author import Author
from proofstack.agents.batch3_critic import Batch3CleanupCritic
from proofstack.agents.dag_workflow import _deep_merge
from proofstack.context import component_config_for_class
from proofstack.latex_contract import render_firstproof_latex_contract


SIMPLIFICATION_INSTRUCTIONS = """The supplied answer.tex is the latest accepted
manuscript. This is a proof-simplification phase, not new problem-solving or an
editorial expansion. Try to simplify the mathematical argument: remove redundant
lemmas or case splits, find a shorter proof, reduce machinery, or replace a
technical construction with a more transparent one. New proof ideas are allowed.
Preserve the results, hypotheses, quantifiers and strength of the accepted
manuscript, and still solve the original problem in full. Do not hide work in
research_notes.tex, omit essential steps, or replace a proof with an unverified
citation. Fewer pages alone are not evidence of a simpler proof.
Keep answer.tex self-contained with an inline bibliography. Explain the proposed
simplification and its risks in your thinking summary. If no improvement is
justified, return the accepted manuscript unchanged. Set ready=true only for a
complete candidate. Do not request Council or Compute workers in this phase;
use your own tools. The critic will independently check the entire candidate
against the accepted baseline. Feedback below can refer to a rejected candidate;
answer.tex remains the accepted version, not that rejected candidate.
"""


class SimplificationCritic(Batch3CleanupCritic):
    """Reuse lossless context checks/accounting, but allow new proof ideas."""

    def __init__(self, ctx, **kwargs):
        defaults = component_config_for_class(ctx.component_configs, Batch3CleanupCritic, "Batch3CleanupCritic")
        configured = _deep_merge({"SimplificationCritic": defaults}, ctx.component_configs)
        overrides = dict(ctx.model_overrides)
        if "Batch3CleanupCritic" in overrides:
            overrides.setdefault("SimplificationCritic", overrides["Batch3CleanupCritic"])
        super().__init__(replace(ctx, component_configs=configured, model_overrides=overrides), **kwargs)

    def render_messages(self, inp):
        prompt = """Independently review a proposed mathematical proof simplification.
Read the complete baseline and candidate. Prior acceptance is not evidence of
correctness. The candidate may use a different proof, but must solve the original
problem and preserve the baseline's results, hypotheses, quantifiers and strength.
Check every changed argument and its dependencies, not just prose or length.
Nothing in research notes can substitute for a proof in the candidate.

Choose exactly one disposition:
- accept: the candidate is fully correct, preserves all the baseline's results,
  satisfies the LaTeX contract, AND genuinely simplifies the proof. Explain what
  became simpler. Shorter typography, omitted details, moving arguments into
  notes, or merely claiming simplicity are not improvements.
- repair: the candidate is not a verified improvement (including correctness
  gaps, weakened statements, no actual simplification, or unresolved doubts),
  but you have not found a mathematical flaw in the baseline. Reject this attempt
  and give concrete feedback for the next Author. It is fine if repair would need
  new mathematical reasoning: unlike editorial cleanup, that is allowed here.
- research: you found a substantive mathematical flaw or missing argument in the
  BASELINE itself. Identify it explicitly and explain which baseline claims it
  affects. This revokes baseline acceptance and returns the problem to research.
Do not choose research merely because the new candidate fails.
End with exactly one <cleanup_verdict>accept</cleanup_verdict>,
<cleanup_verdict>repair</cleanup_verdict>, or
<cleanup_verdict>research</cleanup_verdict> on its own line, with nothing after it.
"""
        prompt += "\n# Original problem\n" + inp.problem
        prompt += "\n# Accepted baseline (not assumed correct)\n" + inp.baseline_tex
        prompt += "\n# Candidate manuscript\n" + inp.answer_tex
        prompt += "\n# LaTeX contract\n" + render_firstproof_latex_contract(inp.page_limit)
        prompt += "\n# Mechanical checks\n" + inp.compile_feedback
        prompt += "\n# Author's explanation (unverified)\n" + inp.editor_response
        return [{"role": "user", "content": prompt}]

    def parse_output(self, raw_text, inp):
        out = super().parse_output(raw_text, inp)
        if out.disposition == "restore":
            out = out.model_copy(update={"disposition": "invalid", "parse_failed": True,
                                         "answer_ready": False})
        return out.model_copy(update={"mode": "fresh"})


class SimplificationInterrupted(RuntimeError):
    requires_reconciliation = True


class SimplificationState(BaseModel):
    source_sha256: str
    rounds: int = Field(ge=1, le=10)
    completed: int = Field(default=0, ge=0, le=10)
    stage: Literal["ready", "author", "candidate", "critic", "reviewed", "done", "rejected"] = "ready"
    document: str
    notes: str
    review_md: str
    feedback: str = ""
    candidate: Author.Outputs | None = None
    review: Batch3CleanupCritic.Outputs | None = None
    max_usd: float = Field(gt=0, allow_inf_nan=False)
    initial_spend: float = Field(ge=0, allow_inf_nan=False)
    deadline_unix_s: float = Field(gt=0, allow_inf_nan=False)
    stop_reason: str = ""
    call_agent: str = ""
    call_pending: bool = False
    call_role: Literal["", "author", "critic"] = ""
