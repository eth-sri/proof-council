"""Author — single Pro API call with code interpreter + web search.

The Author is the mathematical writer in the Author/Critic loop. It
receives the problem, the current state of the three canonical
workspace files, the latest Critic review, and any Council replies
from the previous round. It produces a refined version of the files,
optionally a Council request for the next round, and an optional
``ready`` flag.

There are two file-IO modes, selected by the ``USE_CONTAINER_FILES``
class-level toggle:

  - **Container-files mode** (default, ``USE_CONTAINER_FILES = True``).
    Before the API call we upload the workspace's three canonical
    files via ``client.files.create(purpose="user_data")`` and pass
    them as read-only attachments through
    ``code_interpreter.container.file_ids``. They appear in the
    sandbox at ``/mnt/data/file-{platform_id}-{name}``. The Author
    edits them via ``code_interpreter`` and writes the new versions
    to canonical writable paths ``/mnt/data/{name}``. After the call
    we list+download those canonical paths via
    ``client.containers.files`` and pick up whatever the Author wrote.
    Output tokens carry only the ``thinking_summary`` and the optional
    ``<council>``/``<ready>`` control blocks — not the file bodies.

  - **Inline mode** (``USE_CONTAINER_FILES = False``). The legacy
    path: file contents are pasted into the prompt and the Author
    re-emits each file as a fenced ``file path=...`` code block.
    Heavier on output tokens but does not depend on the Files /
    Containers endpoints. Useful for tests with cheap models that
    might struggle with the full hosted-tool stack.

Tools available either way:
  - ``code_interpreter``: a Python sandbox with TeX Live and standard
    scientific libs. Use for ``pdflatex`` compile checks, ``sympy``
    verifications, ad-hoc numerics, etc.
  - ``web_search_preview``: free-form web search; the only path to
    fetch external URLs (the CI sandbox has no outbound network).

The Author is stateless across rounds by default
(``stateful_author=False`` on the workflow): each call sees only what
is pasted into its prompt and the workspace files at /mnt/data/.
Anything the Author wants to remember should be written into
``research_notes.tex``.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from proofstack.agents.ac.blocks import (
    CANONICAL_FILES,
    ComputeRequest,
    CouncilRequest,
    parse_author_output,
)
from proofstack.agents.ac.container_files import (
    ANTHROPIC_FILES_BETA,
    AnthropicContainerFileBridge,
    ContainerFileBridge,
    find_container_id,
)
from proofstack.context import ModelSpec
from proofstack.events import new_call_id
from proofstack.kinds.api_call import APICallAgent, _assistant_text, _one_shot_query
from proofstack.latex_contract import (
    DEFAULT_FIRSTPROOF_PAGE_LIMIT,
    render_firstproof_latex_contract,
)


# ----- inline-mode prompts (legacy path) ----------------------------------


AUTHOR_RESEARCH_DOCTRINE = """\
Research ambition and problem interpretation. Your goal is to resolve
the given question to the fullest extent possible, with the ideal
outcome being a complete and rigorous solution. The solution may
require novel, creative, and non-trivial elements. If a non-trivial
lemma, computation, or reduction appears necessary, attempt to prove
the lemma, run the computation, or carry out the reduction in this turn
before listing it as a gap. Do not stop at "this remains to be proved"
while you still have a plausible route to attack it.

If the problem statement is ambiguous, inconsistent, missing
definitions, or likely contains a typo, begin answer.tex with a short
section titled "Problem statement and interpretation". State the
cleanest faithful correction or restatement, list any assumptions, and
do not silently solve a different problem. Record the same
interpretation in research_notes.tex so later rounds preserve it. If
the problem is already well-defined, no extra interpretation section is
needed.

When open gaps remain in your treatment, end answer.tex with a section
titled "Remaining open issues" (or equivalent). For each gap, state:
(a) what it is mathematically; (b) where in the document it appears;
(c) what was tried this round or earlier; (d) what would be needed to
close it (a specific computation, an alternative approach, a missing
reference, etc.). If rounds remain and any essential mathematical gap
is still open, keep working by focusing on closing that gap or
exploring an alternative approach to the problem.
"""


AUTHOR_ROUND0_SYSTEM = """\
Act as a research-level mathematical proof author. Produce a complete,
technically rigorous LaTeX solution to the problem below, using
computation and search where useful.

You have two tools available:
  - **code_interpreter/code_execution**: a Python sandbox with TeX
    Live and standard scientific libs. Run pdflatex to verify compile,
    and use sympy or numpy for computational exploration, verification
    and sanity checks (e.g., expand $q$-series coefficients, test
    conjectured identities at small values, perform finite-field or
    modular arithmetic).
  - **web_search_preview/web_search**: free-form web search for
    citations.

Output exactly three files using fenced code blocks with ``file
path=...`` info strings. Each block contains the full contents of one
file. The three canonical files are:

```file path=answer.tex
% A compiling LaTeX article — your final answer.
% Use \\documentclass[12pt]{article}; fullpage is allowed, but no
% other margin/layout or line-spacing changes. Use
% \\bibliographystyle{alpha} + \\bibliography{references} for citations.
\\documentclass[12pt]{article}
...
```

```file path=research_notes.tex
% Your reasoning trace, lit notes, scratch computations.
% Need not be polished. Same compiling article structure.
...
```

```file path=references.bib
% BibTeX entries cited from answer.tex / research_notes.tex.
% Verify each entry with web_search_preview before adding it.
...
```

**``research_notes.tex`` is your persistent scratchpad across rounds.**
Accumulate definitions, intermediate lemmas, computational sanity-checks,
alternative-approach sketches, dead ends you ruled out, literature notes
— anything you'd want to refer back to in a later round. **Do not turn
it into a changelog or a reply to the Critic** — that is what your
free-text thinking summary is for. A future Author turn should be able
to read research_notes.tex and pick up where it left off.

Anything outside these three blocks is treated as a free-text
"thinking summary" you may use to flag open questions or signal
intent to a Critic who will read your work next. Keep it short.

""" + AUTHOR_RESEARCH_DOCTRINE


AUTHOR_ROUND0_USER = """\
{problem}

### First Proof LaTeX contract ###
{latex_contract}
"""


AUTHOR_LOOP_SYSTEM = """\
Act as a research-level mathematical proof author iterating on a
written deliverable in an Author/Critic loop. You have already
produced an initial version of three canonical files; a Critic has
reviewed them. Your job in each subsequent round is to refine the
files in response to the Critic's findings.

You have two tools:
  - **code_interpreter/code_execution**: a Python sandbox with TeX Live
    and standard scientific libs. Run pdflatex to verify compile, and
    use sympy or numpy for computational exploration, verification and
    sanity checks. It is ephemeral per call.
  - **web_search_preview/web_search**: free-form web search.

The canonical workspace files are ``answer.tex``,
``research_notes.tex``, and ``references.bib``. To update a file,
emit a fenced code block with a ``file path=...`` info string holding
the *full new contents* of that file. Files you do not emit are kept
unchanged on disk.

```file path=answer.tex
% Replace the full contents of answer.tex.
...
```

**``research_notes.tex`` is your persistent scratchpad across rounds.**
Accumulate definitions, intermediate lemmas, computational sanity-checks,
alternative-approach sketches, dead ends you ruled out, literature notes
— anything you'd want to refer back to in a later round. **Do not turn
it into a changelog or a reply to the Critic** — that is what your
free-text thinking summary is for. A future Author turn should be able
to read research_notes.tex and pick up where it left off.

You may additionally emit three control blocks:

  - **<council>...question...</council>** — invoke the Advisory
    Council on a specific sub-question. Up to one such block per
    round. Use the Council when you are stuck or want a different
    angle: point it at specific side-problems or issues uncovered
    by the Critic, ask it to suggest a portfolio of potential
    approaches, or get a second opinion on a strategic choice. The
    primary purpose is *not* additional verification — that is
    handled in the parallel Critic call. Council members are
    independent strong models (Pro, Gemini, Opus); their replies
    arrive at the start of your next turn.

  - **<compute_agent>...instructions...</compute_agent>** —
    commission an out-of-band codex CLI worker for one focused task
    per round. Up to one such block per round. Soft timeout 60 min,
    full network access, a persistent workspace where ``code/``,
    ``data/``, ``papers/``, ``notes/`` survive across rounds, and —
    depending on the sandbox image — possibly a richer CAS toolchain
    (e.g. SageMath, GAP, Singular, PARI/GP). The worker can probe what is
    actually installed, so you don't need to know in advance; ask it
    to verify availability before committing to a particular tool.
    Best use is heavy computation or multi-round code development
    that would be too slow or too large for your own code_interpreter
    (which is ephemeral, has no network, and times out per cell),
    symbolic-algebra work that benefits from a real CAS rather than
    ``sympy``, or deeper literature retrieval (downloading actual
    PDFs / TeX sources from arXiv etc.). Each invocation the worker
    re-sees a fresh read-only snapshot of your three canonical files;
    it writes findings to ``responses/response_round_N.md`` and the
    whole workspace is zipped and attached to your *next* turn so
    you can inspect logs, data, and downloaded papers via
    code_interpreter. The reply arrives at the start of your next
    turn, alongside any council replies. Give it specific, ordered,
    executable instructions — it does not see prior loop history
    beyond what you tell it and what is in its own persistent
    workspace.

  - **<ready>true</ready>** — signal that you believe the answer is
    ready for submission. See the readiness rules below.

Anything outside the file blocks and control blocks is kept as a
free-text "thinking summary" for the human reviewer and the Critic.
Keep it short.

---

""" + AUTHOR_RESEARCH_DOCTRINE + """\

Readiness rule. Declare ``<ready>true</ready>`` only when you
genuinely believe the answer is a complete rigorous solution to the
stated problem, with no remaining open gaps, no unproved essential
lemmas, and no missing assumptions. Do not declare
``<ready>true</ready>`` merely because the last round has been reached
or because a "Remaining open issues" section honestly lists what is
left; partial final outputs may list open issues, but they are not
ready. Even then the run terminates only if the Critic also concurs.
"""


AUTHOR_LOOP_USER = """\
### Problem ###
{problem}

### Round {round} of {n_rounds} ###
Budget used so far: ${budget_used_usd:.2f} / ${budget_max_usd:.2f}

### First Proof LaTeX contract ###
{latex_contract}

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

### Latest Critic review ###
{prev_critique}

### Workflow compile/format feedback ###
{workflow_feedback}

### Advisory Council replies (if you requested any last turn) ###
{prev_council}

### Compute worker reply (if you commissioned one last turn) ###
{prev_compute_response}

Refine the three files. Emit fenced ``file path=...`` blocks for any
file you change. Optionally invoke ``<council>`` on at most one
specific sub-question, and/or ``<compute_agent>`` on at most one
focused computation/code/literature task. Optionally set
``<ready>true</ready>`` if you believe the answer is ready for
submission.
"""


# ----- container-mode prompts ---------------------------------------------


AUTHOR_ROUND0_SYSTEM_CONTAINER = """\
Act as a research-level mathematical proof author. Produce a complete,
technically rigorous LaTeX solution to the problem below, using
computation and search where useful.

You have two tools available:
  - **code_interpreter/code_execution**: a Python sandbox with TeX Live
    and standard scientific libs. Run ``pdflatex`` to verify compile,
    and use sympy or numpy for computational exploration, verification
    and sanity checks (e.g., expand $q$-series coefficients, test
    conjectured identities at small values, perform finite-field or
    modular arithmetic). The sandbox has NO outbound network — for
    any external URL or paper, use the web search tool.
  - **web_search_preview/web_search**: free-form web search for
    citations.

Your workspace lives at ``/mnt/data/`` inside the sandbox. Three
canonical files are tracked by the infrastructure:

  - ``/mnt/data/answer.tex``         — the final compiling LaTeX answer
  - ``/mnt/data/research_notes.tex`` — your persistent scratchpad
  - ``/mnt/data/references.bib``     — BibTeX citations

**``research_notes.tex`` is your persistent scratchpad across rounds.**
Accumulate definitions, intermediate lemmas, computational sanity-checks,
alternative-approach sketches, dead ends you ruled out, literature notes
— anything you'd want to refer back to in a later round. **Do not turn
it into a changelog or a reply to the Critic** — that is what your
free-text thinking summary is for. A future Author turn should be able
to read research_notes.tex and pick up where it left off.

The user message will list, for each canonical file, either:

  - a **read-only input path** (a platform-prefixed path like
    ``/mnt/data/file-{{id}}-answer.tex`` containing whatever was on
    disk before this turn) and the **canonical write path**
    (``/mnt/data/<name>`` without prefix) where you must write the new
    contents; or
  - Anthropic Files API ``container_upload`` attachments. In that
    case, inspect the attached files in ``code_execution`` and create
    generated files named exactly ``answer.tex``,
    ``research_notes.tex``, and ``references.bib``. The
    infrastructure downloads generated files by exact filename.

How to work:
  1. Read each input file from its read-only path or attached upload
     (it will be empty on round 0).
  2. Build the new contents (in memory, or by copying then editing).
  3. Write the new contents to the canonical write path with
     ``Path("/mnt/data/<name>").write_text(...)`` when such paths are
     available. With Anthropic Files API attachments, create generated
     files named exactly ``answer.tex``, ``research_notes.tex``, and
     ``references.bib``. Do not write to ``.new`` / ``.bak`` / sibling
     paths — only the canonical filenames are picked up by the
     infrastructure after your turn.
  4. Run ``pdflatex /mnt/data/answer.tex`` from a code_interpreter
     cell to confirm it compiles before claiming readiness.

Do NOT paste the file contents in your reply text. Anything in the
reply text outside the optional control blocks below is treated as a
brief free-text "thinking summary" for the Critic.

""" + AUTHOR_RESEARCH_DOCTRINE


AUTHOR_ROUND0_USER_CONTAINER = """\
### Problem ###
{problem}

### First Proof LaTeX contract ###
{latex_contract}

### Workspace files ###
{workspace_listing}

(All three are initially placeholders for round 0.) Author your
deliverable by writing the canonical files at the listed write paths.
"""


AUTHOR_LOOP_SYSTEM_CONTAINER = """\
Act as a research-level mathematical proof author iterating on a
written deliverable in an Author/Critic loop. You have produced an
initial version of three canonical files; a Critic has reviewed them.
Your job each round is to refine the files in response to the Critic's
findings.

You have two tools:
  - **code_interpreter/code_execution**: a Python sandbox with TeX Live
    and standard scientific libs. Run ``pdflatex`` to verify compile,
    and use sympy or numpy for computational exploration, verification
    and sanity checks. The sandbox has NO outbound network — use the
    web search tool for any external URL or paper.
  - **web_search_preview/web_search**: free-form web search.

Your workspace contains three canonical files. The user message lists,
for each, either a read-only input path (holding the previous round's
contents) and a canonical write path (``/mnt/data/<name>``), or
Anthropic Files API ``container_upload`` attachments. For Anthropic
attachments, inspect the uploaded files in ``code_execution`` and
create generated files named exactly ``answer.tex``,
``research_notes.tex``, and ``references.bib``.

How to update a file:
  1. Read its current contents from the listed read-only path or
     attached upload.
  2. Build the new contents.
  3. Write to the canonical write path with
     ``Path("/mnt/data/<name>").write_text(...)`` when available. With
     Anthropic Files API attachments, create generated files named
     exactly ``answer.tex``, ``research_notes.tex``, and
     ``references.bib``. Do not write to ``.new``/``.bak``/sibling
     paths — only canonical filenames are picked up.
  4. Compile-check answer.tex with ``pdflatex`` before claiming
     readiness.

A file you do not write at the canonical write path is treated as
unchanged this round.

**``research_notes.tex`` is your persistent scratchpad across rounds.**
Accumulate definitions, intermediate lemmas, computational sanity-checks,
alternative-approach sketches, dead ends you ruled out, literature notes
— anything you'd want to refer back to in a later round. **Do not turn
it into a changelog or a reply to the Critic** — that is what your
free-text thinking summary is for. A future Author turn should be able
to read research_notes.tex and pick up where it left off.

You may also emit three control blocks in your reply text:

  - **<council>...question...</council>** — invoke the Advisory
    Council on a specific sub-question. At most one per round. Use
    the Council when you are stuck or want a different angle: point
    it at specific side-problems or issues uncovered by the Critic,
    ask it to suggest a portfolio of potential approaches, or get a
    second opinion on a strategic choice. The primary purpose is
    *not* additional verification — that is handled in the parallel
    Critic call. Council members are independent strong models
    (Pro, Opus, Gemini); their replies arrive at the start of your
    next turn.

  - **<compute_agent>...instructions...</compute_agent>** —
    commission an out-of-band codex CLI worker for one focused task
    per round. At most one per round. Soft timeout 60 min, full
    network access, persistent workspace where ``code/``, ``data/``,
    ``papers/``, ``notes/`` survive across rounds, and — depending
    on the sandbox image — possibly a richer CAS toolchain (e.g.
    SageMath, GAP, Singular, PARI/GP). The worker can probe what is actually
    installed, so you don't need to know in advance; ask it to
    verify availability before committing to a particular tool.
    Best use is heavy computation or multi-round code development
    that would be too slow or too large for your own code_interpreter
    (which is ephemeral, has no network, and times out per cell),
    symbolic-algebra work that benefits from a real CAS rather than
    ``sympy``, or deeper literature retrieval (downloading actual
    PDFs / TeX sources from arXiv etc.). Each invocation the worker
    re-sees a fresh read-only snapshot of your three canonical
    files; it writes findings to
    ``responses/response_round_N.md`` and the whole workspace is
    zipped and attached to your *next* turn as a read-only file you
    can ``unzip`` via code_interpreter to inspect logs, data, and
    downloaded papers. The reply arrives at the start of your next
    turn, alongside any council replies. Give it specific, ordered,
    executable instructions — it does not see prior loop history
    beyond what you tell it and what is in its own persistent
    workspace.

  - **<ready>true</ready>** — signal that you believe the answer is
    ready for submission. See the readiness rules below.

Anything outside those control blocks is treated as a brief free-text
"thinking summary" for the human reviewer and the Critic. Do NOT
paste file contents in your reply.

---

""" + AUTHOR_RESEARCH_DOCTRINE + """\

Readiness rule. Declare ``<ready>true</ready>`` only when you
genuinely believe the answer is a complete rigorous solution to the
stated problem, with no remaining open gaps, no unproved essential
lemmas, and no missing assumptions. Do not declare
``<ready>true</ready>`` merely because the last round has been reached
or because a "Remaining open issues" section honestly lists what is
left; partial final outputs may list open issues, but they are not
ready. Even then the run terminates only if the Critic also concurs.
"""


AUTHOR_LOOP_USER_CONTAINER = """\
### Problem ###
{problem}

### Round {round} of {n_rounds} ###
Budget used so far: ${budget_used_usd:.2f} / ${budget_max_usd:.2f}

### First Proof LaTeX contract ###
{latex_contract}

### Workspace files ###
{workspace_listing}

### Latest Critic review ###
{prev_critique}

### Workflow compile/format feedback ###
{workflow_feedback}

### Advisory Council replies (if you requested any last turn) ###
{prev_council}

### Compute worker reply (if you commissioned one last turn) ###
{prev_compute_response}

Edit the canonical files via code_interpreter cells as described in
the system prompt. Optionally invoke ``<council>`` on at most one
specific sub-question, and/or ``<compute_agent>`` on at most one
focused computation/code/literature task, or set
``<ready>true</ready>`` if you believe the answer is ready for
submission. Do NOT paste the canonical files in your reply.
"""


class Author(APICallAgent):
    """One Author API call — Pro + code interpreter + web search."""

    description: ClassVar[str] = (
        "Author/Critic-loop mathematical writer (Pro + code interpreter + web search)."
    )
    SYSTEM_PROMPT: ClassVar[str] = AUTHOR_LOOP_SYSTEM
    USER_PROMPT: ClassVar[str] = AUTHOR_LOOP_USER
    MODEL: ClassVar[ModelSpec] = "models/openai/gpt-56-sol-pro"
    # Generous tool-call budget — Author may run several CI cells (compile,
    # sympy, sanity checks) plus web searches per round.
    MAX_TOOL_CALLS: ClassVar[int] = 30
    # Container-files mode (v1): upload workspace files as read-only
    # attachments and read modified versions from the container after
    # the call, instead of pasting them into the prompt and parsing
    # fenced ``file path=...`` blocks from the response. Saves output
    # tokens and lets Pro do natural in-place edits via Path.write_text.
    USE_CONTAINER_FILES: ClassVar[bool] = True

    class Inputs(BaseModel):
        problem: str
        round: int
        n_rounds: int
        page_limit: int = DEFAULT_FIRSTPROOF_PAGE_LIMIT
        budget_used_usd: float = 0.0
        budget_max_usd: float = 0.0
        answer_tex: str = ""
        research_notes_tex: str = ""
        references_bib: str = ""
        prev_critique: str = ""
        workflow_feedback: str = ""
        prev_council: str = ""
        # Pasted text reply from the previous round's compute worker
        # (``responses/response_round_{k-1}.md``). Empty if no compute
        # request was made last round.
        prev_compute_response: str = ""
        # Path to the previous round's compute workspace zip — attached
        # to the container as a read-only user_data file when present.
        compute_zip_path: Path | None = None

    class Outputs(BaseModel):
        answer_tex: str = ""
        research_notes_tex: str = ""
        references_bib: str = ""
        files_changed: list[str] = Field(default_factory=list)
        ready: bool = False
        council_question: str | None = None
        council_to: list[str] = Field(default_factory=list)
        compute_instructions: str | None = None
        thinking_summary: str = ""
        parse_warnings: list[str] = Field(default_factory=list)
        raw_text: str = ""
        # Diagnostic metadata for the container-files path.
        container_id: str | None = None
        via: str = "inline_blocks"

    def extra_client_kwargs(self) -> dict[str, Any]:
        # Inline-mode tool config; container-files mode rebuilds the
        # client per call with the current upload's file_ids spliced in.
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
        for k in (
            "answer_tex",
            "research_notes_tex",
            "references_bib",
            "prev_critique",
            "workflow_feedback",
            "prev_council",
            "prev_compute_response",
        ):
            if not fields.get(k):
                fields[k] = "(empty)"

        if inp.round == 0:
            return [
                {"role": "developer", "content": AUTHOR_ROUND0_SYSTEM},
                {"role": "user", "content": AUTHOR_ROUND0_USER.format(**fields)},
            ]
        return [
            {"role": "developer", "content": AUTHOR_LOOP_SYSTEM},
            {"role": "user", "content": AUTHOR_LOOP_USER.format(**fields)},
        ]

    def parse_output(self, raw_text: str, inp: Inputs) -> Outputs:
        parsed = parse_author_output(raw_text)
        # Carry forward existing file contents for any file the Author
        # did not re-emit. Round 0 has no prior contents, so omitted
        # files default to empty.
        answer_tex = parsed.files.get("answer.tex", inp.answer_tex)
        research_notes_tex = parsed.files.get("research_notes.tex", inp.research_notes_tex)
        references_bib = parsed.files.get("references.bib", inp.references_bib)
        files_changed = sorted(
            name for name in CANONICAL_FILES if name in parsed.files
        )
        council_q: str | None = None
        council_to: list[str] = []
        if isinstance(parsed.council, CouncilRequest):
            council_q = parsed.council.question
            council_to = list(parsed.council.to)
        compute_instr: str | None = None
        if isinstance(parsed.compute, ComputeRequest):
            compute_instr = parsed.compute.instructions
        return self.Outputs(
            answer_tex=answer_tex,
            research_notes_tex=research_notes_tex,
            references_bib=references_bib,
            files_changed=files_changed,
            ready=parsed.ready,
            council_question=council_q,
            council_to=council_to,
            compute_instructions=compute_instr,
            thinking_summary=parsed.thinking_summary,
            parse_warnings=list(parsed.parse_warnings),
            raw_text=raw_text,
            via="inline_blocks",
        )

    # ---- container-files run path ---------------------------------------

    async def run(self, inp: Inputs) -> Outputs:  # type: ignore[override]
        if not self.USE_CONTAINER_FILES:
            return await super().run(inp)
        return await self._run_with_container_files(inp)

    async def _run_with_container_files(self, inp: Inputs) -> Outputs:
        provider_api = self._container_file_provider_api()
        if provider_api == "anthropic":
            return await self._run_with_anthropic_container_files(inp)
        if provider_api == "openai":
            return await self._run_with_openai_container_files(inp)
        await self.events.emit(
            "ac.author.container_files_unsupported_provider",
            {"api": provider_api},
        )
        return await super().run(inp)

    async def _run_with_openai_container_files(self, inp: Inputs) -> Outputs:
        # Surface budget warnings the same way APICallAgent.run does.
        for scope, kind, used, limit in self.tracker.check():
            await self.events.emit(
                "budget.warn",
                {"scope": scope, "kind": kind, "used": used, "limit": limit},
            )

        from openai import OpenAI  # imported lazily so non-CI tests don't depend on it

        with tempfile.TemporaryDirectory(prefix="ac_author_upload_") as td:
            td_path = Path(td)
            (td_path / "answer.tex").write_text(inp.answer_tex or "", encoding="utf-8")
            (td_path / "research_notes.tex").write_text(
                inp.research_notes_tex or "", encoding="utf-8"
            )
            (td_path / "references.bib").write_text(
                inp.references_bib or "", encoding="utf-8"
            )

            openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            bridge = ContainerFileBridge(
                openai_client=openai_client,
                workspace=td_path,
                names=CANONICAL_FILES,
                extra_attachments=self._extra_attachments(inp),
            )
            try:
                file_ids = bridge.upload()
                workspace_listing = bridge.render_workspace_listing()

                messages = self._render_container_messages(inp, workspace_listing)
                try:
                    (self.workdir / "messages.json").write_text(
                        json.dumps(messages, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
                except OSError:
                    pass

                api_client = self._build_api_client_with_file_ids(file_ids)

                call_id = new_call_id()
                await self.events.emit(
                    "model.call.start",
                    {"model": getattr(api_client, "model", str(self.MODEL))},
                    call_id=call_id,
                )
                start = time.monotonic()
                _idx, conversation, cost = await asyncio.to_thread(
                    _one_shot_query, api_client, messages
                )
                elapsed = time.monotonic() - start

                usd = float(cost.get("cost", 0.0))
                in_tok = int(cost.get("input_tokens", 0) or 0)
                out_tok = int(cost.get("output_tokens", 0) or 0)
                # First Proof spec requires per-call reasoning tokens.
                # The container-files Author path emits its own
                # ``model.call`` event instead of going through
                # APICallAgent.run, so reasoning_tokens has to be
                # propagated explicitly here too — otherwise the
                # dominant Author Pro calls always log 0.
                reasoning_tok = int(cost.get("reasoning_tokens", 0) or 0)
                self.tracker.add_usd(usd)
                self.tracker.add_tokens(in_tok + out_tok)
                await self.events.emit(
                    "model.call",
                    {
                        "model": getattr(api_client, "model", str(self.MODEL)),
                        "in_tokens": in_tok,
                        "out_tokens": out_tok,
                        "reasoning_tokens": reasoning_tok,
                        "cost_usd": usd,
                        "duration_s": elapsed,
                        "via": "container_files",
                    },
                    call_id=call_id,
                )
                for scope, kind, used, limit in self.tracker.check():
                    await self.events.emit(
                        "budget.warn",
                        {"scope": scope, "kind": kind, "used": used, "limit": limit},
                    )

                raw_text = _assistant_text(conversation)
                try:
                    (self.workdir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
                except OSError:
                    pass

                container_id = find_container_id(conversation)
                modified: dict[str, str] = {}
                if container_id is None:
                    await self.events.emit(
                        "ac.author.no_container_id",
                        {"reason": "no code_interpreter_call in conversation"},
                    )
                else:
                    try:
                        modified = await asyncio.to_thread(bridge.download, container_id)
                    except Exception as e:
                        await self.events.emit(
                            "ac.author.container_download_failed",
                            {
                                "container_id": container_id,
                                "type": type(e).__name__,
                                "msg": str(e),
                            },
                        )
            finally:
                # Always delete uploaded user_data files, even if the
                # API call / cost accounting / parsing / download raised.
                # Without this the uploads leak in OpenAI storage on
                # every Author failure. Download (above) runs *before*
                # cleanup on the success path because it lives inside
                # the ``try``; on the failure path there is nothing to
                # download.
                try:
                    await asyncio.to_thread(bridge.cleanup)
                except Exception as e:
                    await self.events.emit(
                        "ac.author.cleanup_failed",
                        {"type": type(e).__name__, "msg": str(e)},
                    )

        return self._build_outputs_from_container(
            inp, raw_text, modified, container_id, via="container_files"
        )

    async def _run_with_anthropic_container_files(self, inp: Inputs) -> Outputs:
        # Surface budget warnings the same way APICallAgent.run does.
        for scope, kind, used, limit in self.tracker.check():
            await self.events.emit(
                "budget.warn",
                {"scope": scope, "kind": kind, "used": used, "limit": limit},
            )

        import anthropic  # imported lazily so non-Anthropic tests stay light

        with tempfile.TemporaryDirectory(prefix="ac_author_anthropic_upload_") as td:
            td_path = Path(td)
            (td_path / "answer.tex").write_text(inp.answer_tex or "", encoding="utf-8")
            (td_path / "research_notes.tex").write_text(
                inp.research_notes_tex or "", encoding="utf-8"
            )
            (td_path / "references.bib").write_text(
                inp.references_bib or "", encoding="utf-8"
            )

            anthropic_client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )
            bridge = AnthropicContainerFileBridge(
                anthropic_client=anthropic_client,
                workspace=td_path,
                names=CANONICAL_FILES,
                extra_attachments=self._extra_attachments(inp),
            )
            try:
                bridge.upload()
                workspace_listing = bridge.render_workspace_listing()
                upload_blocks = bridge.render_container_upload_blocks()
                messages = self._render_anthropic_container_messages(
                    inp, workspace_listing, upload_blocks
                )
                try:
                    (self.workdir / "messages.json").write_text(
                        json.dumps(messages, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
                except OSError:
                    pass

                api_client = self._build_anthropic_api_client_with_files()

                call_id = new_call_id()
                await self.events.emit(
                    "model.call.start",
                    {"model": getattr(api_client, "model", str(self.MODEL))},
                    call_id=call_id,
                )
                start = time.monotonic()
                _idx, conversation, cost = await asyncio.to_thread(
                    _one_shot_query, api_client, messages
                )
                elapsed = time.monotonic() - start

                usd = float(cost.get("cost", 0.0))
                in_tok = int(cost.get("input_tokens", 0) or 0)
                out_tok = int(cost.get("output_tokens", 0) or 0)
                reasoning_tok = int(cost.get("reasoning_tokens", 0) or 0)
                self.tracker.add_usd(usd)
                self.tracker.add_tokens(in_tok + out_tok)
                await self.events.emit(
                    "model.call",
                    {
                        "model": getattr(api_client, "model", str(self.MODEL)),
                        "in_tokens": in_tok,
                        "out_tokens": out_tok,
                        "reasoning_tokens": reasoning_tok,
                        "cost_usd": usd,
                        "duration_s": elapsed,
                        "via": "anthropic_container_files",
                    },
                    call_id=call_id,
                )
                for scope, kind, used, limit in self.tracker.check():
                    await self.events.emit(
                        "budget.warn",
                        {"scope": scope, "kind": kind, "used": used, "limit": limit},
                    )

                raw_text = _assistant_text(conversation)
                try:
                    (self.workdir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
                except OSError:
                    pass

                try:
                    modified = await asyncio.to_thread(bridge.download, conversation)
                except Exception as e:
                    modified = {}
                    await self.events.emit(
                        "ac.author.anthropic_download_failed",
                        {"type": type(e).__name__, "msg": str(e)},
                    )
                if not modified:
                    await self.events.emit(
                        "ac.author.no_generated_files",
                        {"provider": "anthropic"},
                    )
                    if inp.round == 0:
                        raise RuntimeError(
                            "Anthropic Author produced no generated canonical files "
                            "on round 0; failing instead of returning empty answer.tex."
                        )
            finally:
                try:
                    await asyncio.to_thread(bridge.cleanup)
                except Exception as e:
                    await self.events.emit(
                        "ac.author.cleanup_failed",
                        {"type": type(e).__name__, "msg": str(e)},
                    )

        return self._build_outputs_from_container(
            inp, raw_text, modified, None, via="anthropic_container_files"
        )

    def _extra_attachments(self, inp: Inputs) -> list[tuple[Path, str]]:
        extra_attachments: list[tuple[Path, str]] = []
        if inp.compute_zip_path is not None:
            zip_path = Path(inp.compute_zip_path)
            if zip_path.exists():
                extra_attachments.append(
                    (
                        zip_path,
                        (
                            "Zip of the previous round's compute-worker "
                            "workspace. Unzip via code_interpreter/code_execution "
                            "to inspect responses/, code/, data/, papers/, notes/, etc."
                        ),
                    )
                )
        return extra_attachments

    def _render_container_messages(
        self, inp: Inputs, workspace_listing: str
    ) -> list[dict[str, Any]]:
        fields = inp.model_dump(mode="json")
        fields["latex_contract"] = render_firstproof_latex_contract(inp.page_limit)
        for k in (
            "prev_critique",
            "workflow_feedback",
            "prev_council",
            "prev_compute_response",
        ):
            if not fields.get(k):
                fields[k] = "(empty)"
        fields["workspace_listing"] = workspace_listing
        if inp.round == 0:
            return [
                {"role": "developer", "content": AUTHOR_ROUND0_SYSTEM_CONTAINER},
                {"role": "user", "content": AUTHOR_ROUND0_USER_CONTAINER.format(**fields)},
            ]
        return [
            {"role": "developer", "content": AUTHOR_LOOP_SYSTEM_CONTAINER},
            {"role": "user", "content": AUTHOR_LOOP_USER_CONTAINER.format(**fields)},
        ]

    def _render_anthropic_container_messages(
        self,
        inp: Inputs,
        workspace_listing: str,
        upload_blocks: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        messages = self._render_container_messages(inp, workspace_listing)
        mode_note = (
            "### Anthropic Files API workspace mode ###\n"
            "The attached `container_upload` blocks are the source of truth for "
            "the current canonical files and any compute zip. In code_execution, "
            "inspect those uploads and create generated files named exactly "
            "`answer.tex`, `research_notes.tex`, and `references.bib`. If "
            "`/mnt/data` paths are not present in this sandbox, write those exact "
            "filenames in the current working directory and compile with "
            "`pdflatex answer.tex`. Do not paste full file bodies in the reply "
            "text; generated files are downloaded by exact filename.\n\n"
        )
        for msg in messages:
            if msg.get("role") != "user":
                continue
            text = str(msg.get("content", ""))
            msg["content"] = [
                {"type": "text", "text": mode_note + text},
                *upload_blocks,
            ]
            break
        return messages

    def _container_model_config(self) -> dict[str, Any]:
        from mathagents import load_solver_config

        spec = self.ctx.model_for(self, self.MODEL)
        cfg = load_solver_config(spec)
        return {k: v for k, v in cfg.items() if not k.startswith("__")}

    def _container_file_provider_api(self) -> str:
        cfg = self._container_model_config()
        return str(cfg.get("api", "openai")).lower()

    def _build_api_client_with_file_ids(self, file_ids: list[str]):
        """Construct a fresh APIClient whose code_interpreter tool has
        the round's uploaded ``file_ids`` attached.

        We bypass the parent class's cached client because the file_ids
        change every call.
        """
        cfg = self._container_model_config()
        cfg["tools"] = [
            (
                None,
                {
                    "type": "code_interpreter",
                    "container": {"type": "auto", "file_ids": file_ids},
                },
            ),
            (None, {"type": "web_search_preview"}),
        ]
        cfg["max_tool_calls"] = self.MAX_TOOL_CALLS
        return self.ctx.api_client_factory(cfg)

    def _build_anthropic_api_client_with_files(self):
        cfg = self._container_model_config()
        cfg["tools"] = [
            (None, {"type": "code_interpreter"}),
            (None, {"type": "web_search_preview", "max_uses": self.MAX_TOOL_CALLS}),
        ]
        cfg["max_tool_calls"] = self.MAX_TOOL_CALLS
        betas = list(cfg.get("anthropic_betas") or [])
        if ANTHROPIC_FILES_BETA not in betas:
            betas.append(ANTHROPIC_FILES_BETA)
        cfg["anthropic_betas"] = betas
        return self.ctx.api_client_factory(cfg)

    def _build_outputs_from_container(
        self,
        inp: Inputs,
        raw_text: str,
        modified: dict[str, str],
        container_id: str | None,
        *,
        via: str = "container_files",
    ) -> Outputs:
        # We still parse the response text for the optional control
        # blocks (council, ready, thinking_summary). Any fenced
        # ``file path=...`` blocks emitted in container mode are
        # ignored — the container is the source of truth for files.
        parsed = parse_author_output(raw_text)

        answer_tex = modified.get("answer.tex", inp.answer_tex)
        research_notes_tex = modified.get(
            "research_notes.tex", inp.research_notes_tex
        )
        references_bib = modified.get("references.bib", inp.references_bib)

        files_changed: list[str] = []
        for name, new_val, old_val in [
            ("answer.tex", answer_tex, inp.answer_tex),
            ("research_notes.tex", research_notes_tex, inp.research_notes_tex),
            ("references.bib", references_bib, inp.references_bib),
        ]:
            if new_val != old_val:
                files_changed.append(name)

        council_q: str | None = None
        council_to: list[str] = []
        if isinstance(parsed.council, CouncilRequest):
            council_q = parsed.council.question
            council_to = list(parsed.council.to)
        compute_instr: str | None = None
        if isinstance(parsed.compute, ComputeRequest):
            compute_instr = parsed.compute.instructions

        warnings = list(parsed.parse_warnings)
        # Surface a warning if the Author emitted fenced file blocks in
        # container mode (the user should know they were ignored).
        ignored_inline_files = sorted(parsed.files.keys())
        if ignored_inline_files:
            warnings.append(
                f"container mode: ignored {len(ignored_inline_files)} fenced "
                f"file block(s) ({ignored_inline_files}) — files are read "
                f"from the container, not from the response."
            )

        return self.Outputs(
            answer_tex=answer_tex,
            research_notes_tex=research_notes_tex,
            references_bib=references_bib,
            files_changed=sorted(files_changed),
            ready=parsed.ready,
            council_question=council_q,
            council_to=council_to,
            compute_instructions=compute_instr,
            thinking_summary=parsed.thinking_summary,
            parse_warnings=warnings,
            raw_text=raw_text,
            container_id=container_id,
            via=via,
        )


__all__ = ["Author"]
