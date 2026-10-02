"""Stateful regression review of an edited candidate, not a cold correctness vote."""
from __future__ import annotations

import json
import re
from typing import Literal

from mathagents.api_client import _is_context_length_error
from mathagents.config_loader import load_solver_config
from proofstack.agents.ac.critic import ACCritic
from proofstack.kinds.api_call import APICallAgent
from proofstack.latex_contract import render_firstproof_latex_contract


class CleanupContextTooLarge(ValueError):
    """The lossless review packet does not fit; a provider retry cannot help."""


class Batch3CleanupCritic(ACCritic):
    MODEL = "models/openai/gpt-6-astra-pro"
    completed_review = None

    class Inputs(ACCritic.Inputs):
        baseline_tex: str = ""
        compile_feedback: str = ""
        editor_response: str = ""

    class Outputs(ACCritic.Outputs):
        disposition: Literal["accept", "repair", "restore", "research", "invalid"] = "invalid"

    def cache_input_is_reusable(self, inp):
        return True

    def cache_output_is_reusable(self, out):
        # This reviewer receives baseline/candidate text, not research-note files.
        return not out.parse_failed

    def _on_response(self, raw_text, inp):
        # Each editorial invocation has its own instance. Retain substantive
        # rejections even if subsequent accounting or logging is interrupted.
        self.completed_review = self.parse_output(raw_text, inp)

    @staticmethod
    def _reviews(inp):
        # Keep every finding, including older unresolved objections. Only old
        # user packets (full manuscript snapshots) are replaced, never reviews.
        return [message for message in inp.prior_messages if message.get("role") == "assistant"]

    @staticmethod
    def _baseline(inp):
        if inp.baseline_tex:
            return inp.baseline_tex
        # Compatibility with saved conversations and callers that supplied the
        # baseline only on the first review of an editorial episode.
        marker = "\n# Pre-rewrite baseline (not assumed correct)\n"
        for message in reversed(inp.prior_messages):
            content = message.get("content", "")
            if message.get("role") == "user" and isinstance(content, str) and marker in content:
                return content.split(marker, 1)[1].split("\n# Candidate manuscript\n", 1)[0]
        return ""

    def context_preflight(self, inp):
        config = self.ctx.component_config_for(self)
        model = load_solver_config(self.ctx.model_for(self, self.MODEL))
        # Text tokenization cannot exceed the UTF-8 byte count. This deliberately
        # conservative bound needs no tokenizer download and handles LaTeX and
        # Unicode without relying on a prose chars/token average.
        try:
            window = int(config.get("context_window_tokens", 1_050_000))
            reserve = int(config.get("context_tool_reserve_tokens", 64_000))
            output = int(model.get("max_tokens") or 128_000)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CleanupContextTooLarge("Invalid cleanup critic context limits") from exc
        if window <= 0 or reserve < 0 or output <= 0:
            raise CleanupContextTooLarge("Invalid cleanup critic context limits")
        messages = self.render_messages(inp)
        upper_bound = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 1024
        requested_model, _, suffix_effort = model["model"].partition("--")
        return {
            # Requested settings, not a claim about any later provider fallback.
            "requested_model": requested_model,
            "requested_reasoning_mode": (model.get("reasoning") or {}).get("mode"),
            "requested_reasoning_effort": suffix_effort or (model.get("reasoning") or {}).get("effort") or model.get("reasoning_effort"),
            "input_token_upper_bound": upper_bound,
            "input_token_budget": window - output - reserve,
            "context_window_tokens": window,
            "output_token_reserve": output,
            "tool_token_reserve": reserve,
            "prior_review_count": len(self._reviews(inp)),
            "removed_user_packets": sum(m.get("role") == "user" for m in inp.prior_messages),
        }

    async def run(self, inp):
        preflight = self.context_preflight(inp)
        fits = preflight["input_token_upper_bound"] <= preflight["input_token_budget"]
        await self.events.emit("cleanup.context.preflight", {**preflight, "fits": fits})
        if not fits:
            raise CleanupContextTooLarge(
                "Cleanup critic context preflight rejected the lossless review packet: "
                f"{preflight['input_token_upper_bound']} token upper bound exceeds "
                f"{preflight['input_token_budget']} input budget. Old drafts were already removed; "
                "the complete baseline, candidate and critic findings were preserved. "
                "Use a larger verified context limit or a separately reviewed findings reduction; "
                "do not retry this unchanged packet."
            )
        try:
            # Editorial review owns its lossless baseline/findings policy;
            # it must not use the research critic's fresh-review fallback.
            return await APICallAgent.run(self, inp)
        except Exception as exc:
            if _is_context_length_error(exc):
                raise CleanupContextTooLarge(
                    "Provider rejected the cleanup critic context; do not retry the unchanged packet: " + str(exc)
                ) from exc
            raise

    def render_messages(self, inp):
        prompt = """Review this editorial revision against the original problem and the
pre-rewrite baseline. Re-read the entire argument, not only the changed lines.
Prior acceptance is not evidence of correctness. Check for lost proof steps,
changed assumptions or quantifiers, weakened conclusions, broken dependencies,
incorrect citations, and concealed unresolved gaps. Assess clarity as well.

Choose exactly one disposition:
- accept: the exact candidate fully solves the original problem, preserves a
  rigorous argument, is clearly presented, and satisfies the LaTeX contract.
- repair: only local wording, notation, references, or formatting need repair;
  no substantive mathematical reasoning or new lemma is needed.
- restore: a substantive regression introduced ONLY by editing. You have checked
  that the baseline contains a correct argument for every affected step and can
  identify the exact baseline passages to restore. No new mathematical reasoning
  is needed. Cite both the broken candidate passages and their baseline fixes.
- research: a substantive mathematical flaw or missing argument also exists in
  the baseline, its origin is uncertain, or restoration would require new
  mathematics. This is a mathematical rejection, routed back to Author/Critic.
  Explain the gap and whether it affects the baseline. If ANY finding meets this
  criterion, choose research even if other findings are rewrite-only regressions.
Do not classify a missing proof as a merely editorial repair. Track your earlier
findings, which were addressed, and new issues. Do not approve a partial answer.
The preceding assistant messages are the complete prior review record, in order.
They may concern obsolete drafts. Re-evaluate their findings against the current
candidate below; do not assume that an old objection remains valid or was fixed.
The editor's response is an unverified account of changes, not proof of a repair.
End with exactly one of <cleanup_verdict>accept</cleanup_verdict>,
<cleanup_verdict>repair</cleanup_verdict>,
<cleanup_verdict>restore</cleanup_verdict>, or
<cleanup_verdict>research</cleanup_verdict> on its own line.
This replaces the research-stage answer_ready protocol; do not append an
answer_ready tag or other text after the cleanup verdict.
"""
        prompt += "\n# Original problem\n" + inp.problem
        baseline = self._baseline(inp)
        if baseline:
            prompt += "\n# Pre-rewrite baseline (not assumed correct)\n" + baseline
        prompt += "\n# Candidate manuscript\n" + inp.answer_tex
        prompt += "\n# LaTeX contract\n" + render_firstproof_latex_contract(inp.page_limit)
        prompt += "\n# Mechanical checks\n" + inp.compile_feedback
        if inp.editor_response:
            prompt += "\n# Editor response (unverified)\n" + inp.editor_response
        return [*self._reviews(inp), {"role": "user", "content": prompt}]

    def parse_output(self, raw_text, inp):
        # Older research turns ask for answer_ready. Accept a consistent legacy
        # tag after the unique final verdict, but never contradictory prose/tags.
        match = re.search(
            r"(?:^|\n)[ \t]*<cleanup_verdict>(accept|repair|restore|research)</cleanup_verdict>\s*"
            r"(?:<answer_ready>(true|false)</answer_ready>\s*)?\Z", raw_text)
        disposition = match[1] if match and raw_text.count("<cleanup_verdict>") == 1 else "invalid"
        if match and match[2] is not None and (match[2] == "true") != (disposition == "accept"):
            disposition = "invalid"
        return self.Outputs(
            disposition=disposition, answer_ready=disposition == "accept",
            parse_failed=disposition == "invalid", mode="stateful", review_md=raw_text,
            messages_after=[*self.render_messages(inp), {"role": "assistant", "content": raw_text}],
        )
