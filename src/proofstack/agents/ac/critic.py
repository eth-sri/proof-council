"""ACCritic — referee-style mathematical reviewer with fresh + stateful modes.

Two call shapes:

- ``mode="fresh"``: brand-new instance, no prior conversation. Used at
  round 0, at K-reset boundaries (every ``full_critic_interval`` rounds),
  before the last author round, and on forced-fresh promotion when the
  stateful critic + author both signal ready.
- ``mode="stateful"``: continuation of an existing instance. The prior
  conversation (alternating user/assistant) is passed in via
  ``Inputs.prior_messages``; the new user turn is appended automatically.

Output: free-form referee prose, ending with a single
``<answer_ready>true</answer_ready>`` or ``<answer_ready>false</answer_ready>``
tag on its own line. The workflow uses ``answer_ready`` for its
early-stop gate.

Tools: ``web_search_preview`` and ``code_interpreter`` (no container files).
"""
from __future__ import annotations

import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from proofstack.context import ModelSpec
from proofstack.kinds.api_call import APICallAgent
from proofstack.latex_contract import (
    DEFAULT_FIRSTPROOF_PAGE_LIMIT,
    render_firstproof_latex_contract,
)


CriticMode = Literal["fresh", "stateful"]


CRITIC_PROMPT_HEAD = """\
Act as a strict mathematical referee. Below you find a mathematical
problem statement together with an attempt at a solution. Perform an
in-depth review of this given answer, going paragraph by paragraph to
audit its validity. Check for any mathematical errors, gaps in given
arguments, missing assumptions when applying known results, handwaving,
unclear formulations, unproved essential lemmas, or unresolved
"Remaining open issues". Use web search to validate any cited results
from the literature, and perform cross-checks using code-interpreter
where appropriate. Your goal is to identify *any* issues which could
affect the mathematical validity of the given treatment. Then give me
a full report.

Set `<answer_ready>true</answer_ready>` only if answer.tex fully solves
the stated problem as a complete rigorous solution, with no remaining
open gaps, no unproved essential lemmas, and no missing assumptions. If
the problem statement was ambiguous, answer.tex must explicitly record
the adopted interpretation in a "Problem statement and interpretation"
section and solve that faithful interpretation. A partial final answer
that merely lists open issues is not answer-ready and must end with
`<answer_ready>false</answer_ready>`.

Also set `<answer_ready>false</answer_ready>` if answer.tex violates
the First Proof LaTeX contract supplied below: wrong document class,
font size other than 12pt, over the page limit, non-permitted
margin/layout changes, line-spacing changes, in-document font-size
changes, or any LaTeX compile failure.

End your report with exactly one of these two lines, on its own line, with no additional text:

- `<answer_ready>true</answer_ready>`
- `<answer_ready>false</answer_ready>`"""


CRITIC_FRESH_USER = """\
{prompt_head}

# Problem statement

{problem}

# First Proof LaTeX contract

{latex_contract}

# Author's solution attempt

## `answer.tex`

```latex
{answer_tex}
```

## `references.bib`

```bibtex
{references_bib}
```

# Author's working notes (background context only)

This is the Author's persistent scratchpad — not the deliverable, and not the focus of your review. Skim it; flag any fatal mathematical error you happen to spot that could be steering the proof in a wrong direction. Otherwise concentrate the review on `answer.tex`.

## `research_notes.tex`

```latex
{research_notes_tex}
```

# Author's notes on this draft

{author_thinking}
"""


CRITIC_FRESH_USER_NO_THINKING = """\
{prompt_head}

# Problem statement

{problem}

# First Proof LaTeX contract

{latex_contract}

# Author's solution attempt

## `answer.tex`

```latex
{answer_tex}
```

## `references.bib`

```bibtex
{references_bib}
```

# Author's working notes (background context only)

This is the Author's persistent scratchpad — not the deliverable, and not the focus of your review. Skim it; flag any fatal mathematical error you happen to spot that could be steering the proof in a wrong direction. Otherwise concentrate the review on `answer.tex`.

## `research_notes.tex`

```latex
{research_notes_tex}
```
"""


CRITIC_STATEFUL_USER = """\
The author has revised the proof in response to your previous review. Please review the revised draft. Re-read the proof in full — do not assume earlier concerns were resolved. Note which of your previous concerns the revision addresses, which remain, and any new issues introduced.

# First Proof LaTeX contract

{latex_contract}

## `answer.tex` (revised)

```latex
{answer_tex}
```

## `references.bib` (revised)

```bibtex
{references_bib}
```

# Author's working notes (background context only)

The Author's persistent scratchpad. Not the deliverable, not the focus of review. Skim for any fatal mathematical error that might be steering the proof in a wrong direction; otherwise concentrate on `answer.tex`.

## `research_notes.tex` (revised)

```latex
{research_notes_tex}
```

# Author's notes on the revision

{author_thinking}

Set `<answer_ready>true</answer_ready>` only if answer.tex fully solves
the stated problem as a complete rigorous solution, with no remaining
open gaps, no unproved essential lemmas, and no missing assumptions. If
the problem statement was ambiguous, answer.tex must explicitly record
the adopted interpretation in a "Problem statement and interpretation"
section and solve that faithful interpretation. A partial final answer
that merely lists open issues is not answer-ready.

End your report with `<answer_ready>true</answer_ready>` or `<answer_ready>false</answer_ready>` on its own line.
"""


_ANSWER_READY_RE = re.compile(
    r"<answer_ready>\s*(true|false)\s*</answer_ready>",
    re.IGNORECASE,
)


def _parse_answer_ready(raw_text: str) -> tuple[bool, bool]:
    """Return ``(answer_ready, parse_failed)``.

    Picks the **last** ``<answer_ready>`` tag found in the text — the
    closing verdict should be the last occurrence, but the model
    occasionally repeats the tag mid-prose. ``parse_failed`` is True
    when no tag is present.
    """
    matches = _ANSWER_READY_RE.findall(raw_text)
    if not matches:
        return False, True
    return matches[-1].strip().lower() == "true", False


def _strip_answer_ready(raw_text: str) -> str:
    return _ANSWER_READY_RE.sub("", raw_text).strip()


class ACCritic(APICallAgent):
    """Referee-style Critic with fresh + stateful conversation modes."""

    description: ClassVar[str] = (
        "Referee-style review of the Author's solution. Free-form prose "
        "ending in <answer_ready>true|false</answer_ready>. Two modes: "
        "fresh (new instance) and stateful (continuation via prior_messages)."
    )
    MODEL: ClassVar[ModelSpec] = "models/openai/gpt-56-sol-pro"
    MAX_TOOL_CALLS: ClassVar[int] = 12

    class Inputs(BaseModel):
        problem: str
        round: int = 0
        n_rounds: int = 0
        page_limit: int = DEFAULT_FIRSTPROOF_PAGE_LIMIT
        mode: CriticMode = "fresh"
        answer_tex: str = ""
        research_notes_tex: str = ""
        references_bib: str = ""
        author_thinking: str = ""
        # Stateful continuation: prior conversation alternating
        # user/assistant. The new user turn (rendered from the current
        # input fields) is appended automatically in ``render_messages``.
        prior_messages: list[dict[str, Any]] = Field(default_factory=list)
        # Fresh-only: suppress the Author's thinking summary. Used on
        # forced-fresh promotion to avoid biasing the new reviewer with
        # the Author's "looks done" framing.
        omit_author_thinking: bool = False

    class Outputs(BaseModel):
        review_md: str = ""
        answer_ready: bool = False
        mode: CriticMode = "fresh"
        parse_failed: bool = False
        # Full conversation including this turn's user message and the
        # assistant response. The workflow stores this and passes it as
        # ``prior_messages`` on the next stateful call.
        messages_after: list[dict[str, Any]] = Field(default_factory=list)

    def extra_client_kwargs(self) -> dict[str, Any]:
        return {
            "tools": [
                (None, {"type": "code_interpreter", "container": {"type": "auto"}}),
                (None, {"type": "web_search_preview"}),
            ],
            "max_tool_calls": self.MAX_TOOL_CALLS,
        }

    def render_messages(self, inp: Inputs) -> list[dict[str, Any]]:
        fields = inp.model_dump(mode="json")
        fields["latex_contract"] = render_firstproof_latex_contract(inp.page_limit)
        for k in ("answer_tex", "research_notes_tex", "references_bib", "author_thinking"):
            if not fields.get(k):
                fields[k] = "(empty)"
        fields["prompt_head"] = CRITIC_PROMPT_HEAD

        if inp.mode == "stateful":
            new_user = CRITIC_STATEFUL_USER.format(**fields)
            return list(inp.prior_messages) + [
                {"role": "user", "content": new_user}
            ]

        # Fresh mode
        if inp.omit_author_thinking:
            new_user = CRITIC_FRESH_USER_NO_THINKING.format(**fields)
        else:
            new_user = CRITIC_FRESH_USER.format(**fields)
        return [{"role": "user", "content": new_user}]

    def parse_output(self, raw_text: str, inp: Inputs) -> Outputs:
        answer_ready, parse_failed = _parse_answer_ready(raw_text)
        review_md = _strip_answer_ready(raw_text) if not parse_failed else raw_text

        if parse_failed:
            answer_ready = False
            review_md = (
                (review_md or "")
                + "\n\n*[meta] Critic verdict tag missing; the workflow is "
                "treating answer_ready as False as a defense-in-depth fallback. "
                "Inspect raw_response.txt for the unparsed model output.*"
            )

        # Re-render to recover the new user turn; same call is idempotent
        # since render_messages doesn't touch state. ``rendered[-1]`` is
        # the just-sent user message; the rest is the prior conversation.
        rendered = self.render_messages(inp)
        new_user_message = rendered[-1]
        prior = rendered[:-1]
        assistant_message = {"role": "assistant", "content": raw_text}
        messages_after = prior + [new_user_message, assistant_message]

        return self.Outputs(
            review_md=review_md,
            answer_ready=answer_ready,
            mode=inp.mode,
            parse_failed=parse_failed,
            messages_after=messages_after,
        )


__all__ = ["ACCritic", "CriticMode"]
