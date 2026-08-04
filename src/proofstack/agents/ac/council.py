"""Advisory Council — parallel multi-model fanout.

The Council is invoked when the Author emits a ``<council>...</council>``
block in its turn. Each council member is a single API call to a
strong model (gpt-5.6-sol in Pro mode, claude-opus-4.x,
gemini-3.x-pro, …) given
the same workspace files plus the Author's question. Members run in
parallel; a placeholder ``synthesizer_model`` field is reserved for a
future Pro-vetted summarizer (the user prefers raw replies for now to
preserve entropy).
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from proofstack.agent import Agent
from proofstack.budget import BudgetExhausted
from proofstack.context import ModelSpec
from proofstack.kinds.api_call import APICallAgent


COUNCIL_MEMBER_SYSTEM = """\
You are a member of an Advisory Council. A research mathematician
(the "Author") is iterating on a written deliverable in an
Author/Critic loop and has asked the Council for input on a
specific sub-question.

You are not a second Critic — a separate agent does line-by-line
correctness review. Your role is closer to a research collaborator:
when the Author has hit a wall on a sub-problem or wants a
different angle, you suggest alternative approaches, point at
adjacent literature you happen to know about, propose
decompositions, or share intuition about which directions are
likely to be fruitful.

Even when the Author phrases the question as "is X correct?",
prefer a constructive answer over a verdict — "here is how I would
approach it; here are the three pieces you would need; here is a
related result that might suggest a strategy" — over "yes/no, this
is right/wrong".

Read the current `answer.tex`, `research_notes.tex`, and
`references.bib` for context, then engage with the Author's
question.

Be opinionated where you have a real angle to offer. If you are
not sure, say "uncertain" rather than confabulating. Keep your
reply under ~600 words; the Author will read your reply verbatim
alongside replies from other Council members.
"""


COUNCIL_MEMBER_USER = """\
### Author's question for the Council ###
{author_question}

### Current answer.tex ###
```tex
{answer_tex}
```

### Current research_notes.tex ###
```tex
{research_notes_tex}
```

### Current references.bib ###
```bibtex
{references_bib}
```

Respond with a focused expert opinion on the question above.
"""


class CouncilReply(BaseModel):
    member: str
    model_ref: str
    text: str = ""
    error: str | None = None


class CouncilMember(APICallAgent):
    """One council seat — single API call against the configured model.

    The model is set per-instance via a constructor argument so a
    single class can serve all seats; ``self.MODEL`` is shadowed at
    instance level so ``ctx.model_for(self, self.MODEL)`` resolves the
    intended ref. Each seat gets its own ``agent_path`` / ``call_id``
    in the event log, with the full model name on its ``model.call``
    payload.
    """

    description: ClassVar[str] = (
        "One Advisory Council seat — independent strong-model opinion."
    )
    SYSTEM_PROMPT: ClassVar[str] = COUNCIL_MEMBER_SYSTEM
    USER_PROMPT: ClassVar[str] = COUNCIL_MEMBER_USER
    MODEL: ClassVar[ModelSpec] = "models/openai/gpt-56-sol-pro"

    class Inputs(BaseModel):
        author_question: str
        answer_tex: str = ""
        research_notes_tex: str = ""
        references_bib: str = ""

    class Outputs(BaseModel):
        text: str = ""

    def __init__(
        self,
        ctx: Any,
        *,
        model_ref: ModelSpec | None = None,
        seat_label: str = "Member",
        **kw: Any,
    ) -> None:
        super().__init__(ctx, **kw)
        if model_ref is not None:
            self.MODEL = model_ref  # type: ignore[misc]
        self.seat_label = seat_label

    def render_messages(self, inp: BaseModel) -> list[dict[str, Any]]:
        fields = inp.model_dump(mode="json")
        for k in ("answer_tex", "research_notes_tex", "references_bib"):
            if not fields.get(k):
                fields[k] = "(empty)"
        return [
            {"role": "developer", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT.format(**fields)},
        ]

    def parse_output(self, raw_text: str, inp: BaseModel) -> BaseModel:
        return self.Outputs(text=_strip_visible_thought_blocks(raw_text))


class Council(Agent):
    """Fanout: run all configured CouncilMembers in parallel on one question."""

    description: ClassVar[str] = "Fan out a sub-question to N strong models in parallel."
    execution_mode: ClassVar[str] = "fanout"
    cache_enabled: ClassVar[bool] = False

    class Inputs(BaseModel):
        author_question: str
        answer_tex: str = ""
        research_notes_tex: str = ""
        references_bib: str = ""
        member_models: list[str] = Field(default_factory=list)

    class Outputs(BaseModel):
        replies: list[CouncilReply] = Field(default_factory=list)

    async def run(self, inp: Inputs) -> Outputs:  # type: ignore[override]
        if not inp.member_models:
            return self.Outputs(replies=[])

        async def _one(model_ref: str) -> CouncilReply:
            seat = CouncilMember(
                self.ctx,
                model_ref=model_ref,
                seat_label=_short_label(model_ref),
                parent_budget_scope=self.tracker.scope,
            )
            try:
                out = await seat(
                    author_question=inp.author_question,
                    answer_tex=inp.answer_tex,
                    research_notes_tex=inp.research_notes_tex,
                    references_bib=inp.references_bib,
                )
                # An empty reply means the provider returned an empty
                # assistant message (sometimes the result of a
                # silently-swallowed inner-loop error). Render as an
                # explicit error reply so the Author sees a marker
                # rather than a blank section.
                reply_text = (out.text or "").strip()
                if not reply_text:
                    return CouncilReply(
                        member=seat.seat_label,
                        model_ref=model_ref,
                        error="empty response from provider (no assistant text)",
                    )
                return CouncilReply(
                    member=seat.seat_label,
                    model_ref=model_ref,
                    text=out.text,
                )
            except BudgetExhausted as e:
                # Run-level exhaust must propagate so the workflow
                # last-gasp fires immediately. Agent-scoped exhaust is
                # local to this seat — downgrade to a textual error so
                # the surviving members' replies still reach the Author.
                if e.scope == "run":
                    raise
                return CouncilReply(
                    member=seat.seat_label,
                    model_ref=model_ref,
                    error=f"BudgetExhausted({e.scope}): {e}",
                )
            except Exception as e:
                return CouncilReply(
                    member=seat.seat_label,
                    model_ref=model_ref,
                    error=f"{type(e).__name__}: {e}",
                )

        # ``asyncio.gather(return_exceptions=False)`` propagates the
        # first raise but does NOT cancel siblings — so a run-scope
        # ``BudgetExhausted`` from one seat would otherwise leave the
        # other seats burning budget until their own API calls return.
        # Use ``asyncio.wait(FIRST_EXCEPTION)`` + explicit cancel.
        tasks = [
            asyncio.create_task(
                _one(ref), name=f"CouncilMember-{_short_label(ref)}"
            )
            for ref in inp.member_models
        ]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
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

        replies: list[CouncilReply] = []
        for t in tasks:
            if t.cancelled():
                continue
            exc = t.exception()
            if exc is not None:
                # ``_one`` only ever lets run-scope BudgetExhausted
                # escape; any other escape is a genuine bug worth
                # surfacing. Either way, re-raise so the workflow can
                # last-gasp on the run-scope case.
                raise exc
            replies.append(t.result())
        return self.Outputs(replies=replies)


def _short_label(model_ref: str) -> str:
    """Best-effort short label for an event-log column.

    ``models/openai/gpt-56-sol-pro`` -> ``gpt-56-sol-pro``;
    ``models/anthropic/opus_47_max`` -> ``opus_47_max``;
    fallback: the ref itself.
    """
    parts = model_ref.rsplit("/", 1)
    return parts[-1] if parts else model_ref


_VISIBLE_THOUGHT_RE = re.compile(r"<thought>.*?</thought>\s*", re.IGNORECASE | re.DOTALL)


def _strip_visible_thought_blocks(text: str) -> str:
    return _VISIBLE_THOUGHT_RE.sub("", text).strip()


def render_council_replies_for_author(replies: list[CouncilReply]) -> str:
    """Format Council replies for inclusion in the Author's next-turn prompt."""
    if not replies:
        return "(no council replies)"
    parts: list[str] = []
    for r in replies:
        header = f"### Council member: {r.member} ({r.model_ref}) ###"
        body = r.text if r.text else f"(error: {r.error})"
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


__all__ = [
    "Council",
    "CouncilMember",
    "CouncilReply",
    "render_council_replies_for_author",
]
