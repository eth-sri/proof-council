"""Author with research-scoped asynchronous helpers and a legacy wave mode.

Batch 3 opts into asynchronous delegation: delegate launches jobs, and
helper_status/wait_helpers/cancel_helpers let the lead coordinate them. Jobs
publish versioned artifacts and can survive Author turns until research ends.

The following describes the retained blocking mode for older component users:

The lead Author keeps its single-call shape (one Responses conversation
with code_interpreter + web search over the container files). On top of
that it gets ``read_context`` and ``delegate``, which runs a wave of
role-specialised subagents in parallel and blocks until all of them have
reported. Each subagent is a fresh API call (``SubAuthorSeat``) with the
problem, the lead's task text, an optional shared briefing and optionally
the pre-turn workspace snapshot and tool-readable shared evidence, and answers with a
structured Markdown report the lead reads verbatim. The lead alone edits
the canonical files.

Mapping to Codex multi-agent-v2: ``spawn_agent`` + ``wait_agent`` collapse
into one blocking ``delegate`` call per wave (each local tool round trip
costs a full lead request, so waves are batched); ``followup_task`` is a
task with ``agent_id`` (the seat's prompt/report history is replayed, not
its tool outputs or sandbox); depth is 1 (seats have no ``delegate``);
``author_parallelism`` / ``max_waves`` / ``job_timeout_s`` bound the fan-out.

The workflow's ``author_parallelism=1`` behaves like ``Author``; larger
values allow that many helpers while the lead waits. Direct component users
that omit it retain the legacy ``delegation.enabled`` / ``max_threads`` settings.
"""
from __future__ import annotations

import asyncio
import math
import concurrent.futures
import contextvars
from datetime import datetime, timezone
import json
import os
import time
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from proofstack.agents.ac.author import Author
from proofstack.agents.ac.container_files import (
    CONTAINER_DATA_ROOT,
    _read_container_file,
    find_container_id,
)
from proofstack.agents.ac.council import _short_label, _strip_visible_thought_blocks
from proofstack.agents.ac.delegation_context import DelegationContext, ManifestPages
from proofstack.agents.ac.delegation_snapshot import launch_snapshot
from proofstack.agents.ac.helper_sandbox import HelperSandbox
from proofstack.agents.ac.sandbox_carry import SandboxCarryState, SandboxSnapshot
from proofstack.budget import BudgetExhausted
from proofstack.context import ModelSpec
from proofstack.kinds.api_call import APICallAgent
from .async_helpers import session_for, workflow_owns_helpers


ROLE_DESCRIPTIONS: dict[str, str] = {
    "explorer": (
        "reconnaissance: is the result (or a reformulation) known, and where; "
        "exact statements of results to cite with verified references; small "
        "experiments / numerics; a genuinely different proof architecture"
    ),
    "prover": (
        "prove exactly the stated lemma or claim, in full rigour, in LaTeX; "
        "or the strongest partial result plus a precise account of the gap; "
        "or a counterexample if the statement is false"
    ),
    "checker": (
        "adversarially verify a specific, finished argument: first unsupported "
        "inference or counterexample, exact hypotheses used; verdict "
        "SOUND / REPAIRABLE / BROKEN / UNRESOLVED"
    ),
    "drafter": (
        "write a specified LaTeX section from the lead's detailed plan, in the "
        "lead's notation, marking every step the plan does not justify"
    ),
}

ROLE_INSTRUCTIONS: dict[str, str] = {
    "explorer": """\
Role: explorer. You do reconnaissance, not writing. Find out whether the
result the lead names, or a reformulation of it, is known and where;
collect the exact statements (with all hypotheses) of results the lead
wants to cite, with bibliographic data verified via web search (give
BibTeX); run small experiments or numerics to test conjectures. You may
propose a fundamentally different proof architecture or a reformulation
of the problem if the literature or your experiments suggest one. Say
clearly when you could not find something. Breadth first, then depth on
what matters most for the task.
""",
    "prover": """\
Role: prover. Prove exactly the statement the lead gives you, in full
rigour, written out in LaTeX that the lead can paste and adapt. Do not
change the statement. If you cannot prove it, give the strongest partial
result you can actually prove and a precise account of the missing step
(what is needed, what you tried). If the statement is false as given,
say so and give a counterexample; distinguish "this lemma is false" from
"the original problem is false", and give a corrected statement if one
is apparent. Test your claims on small cases or numerically in
code_interpreter when possible; numerical agreement is supporting
evidence, never a proof.
""",
    "checker": """\
Role: checker. Adversarially verify the finished argument the lead gives
you, paragraph by paragraph. Start the Summary with a verdict: SOUND,
REPAIRABLE (how), BROKEN (where), or UNRESOLVED (what you could not
decide). For BROKEN give the first unsupported inference or a
counterexample; for every cited result list the exact hypotheses used
and whether they are verified in the argument. Test claims numerically
or on small cases where possible, but numerical agreement is evidence,
not verification of a general statement. Do not rewrite the argument
unless the lead asked for a fix; do not soften a real problem.
""",
    "drafter": """\
Role: drafter. Write the LaTeX section the lead specifies, from the plan
or sketch it gives, in the notation and style of the lead's files.
Rigorous and complete; no new results beyond what the plan justifies.
Where the plan is insufficient, write \\textbf{[GAP: ...]} with a
precise description rather than glossing over it. Return the section as
one LaTeX block ready to paste.
""",
}

SEAT_SYSTEM_HEAD = """\
You are a subagent on a small team led by a research-level mathematical
Author (the "lead"), who is writing a rigorous solution to a research
problem in an Author/Critic loop. The lead has delegated ONE focused task
to you. It can read your full report and published artifacts with read_context.
It does not see your private working or files left only in your sandbox.

You have a Python sandbox (code_interpreter, with TeX Live, sympy, numpy;
no network) and web search. Work independently and check what you claim.

Report format (Markdown, in this order):
## Summary
At most 200 words: what you established, each item tagged
[established] / [likely] / [unresolved].
## Caveats
What you assumed, what you could not settle, what the lead must
double-check. Put this before the findings so the limitations are easy to find.
## Findings
The substantive content: proofs in LaTeX, computations with the code
you ran and its output, verified references (BibTeX), counterexamples.

Hard rules: never claim a proof you have not written out in full; mark
every gap; do not restate the task at length; keep the whole report
under about {max_words} words when possible, without omitting essential arguments.
Use publish_artifact to deliver longer proofs, runnable scripts and certificates
as UTF-8 text. When available, prefer publish_sandbox_artifact for computed files: save the actual
file under /mnt/data/checkpoints and let the harness download it without transcription.
A sandbox link alone DOES NOT transfer a file. Check the tool's
success response, cite its returned artifact path in your report, and explicitly
flag any failed publication. The final textual report is also saved automatically.

"""

SEAT_USER = """\
### Problem ###
{problem}

{briefing_section}\
### Your task from the lead (round {round}) ###
{task}

{workspace_section}\
Write your report now, in the required format.
"""

SEAT_BRIEFING_SECTION = """\
### Shared briefing from the lead ###
{briefing}

"""

SEAT_SANDBOX_NOTICE = """\
Sandbox safety: in Pro mode, resuming after a local function call such as
read_context or publish_artifact can replace your sandbox, even within this task.
When publish_sandbox_artifact is available, designated checkpoints are automatically
downloaded before local context/publication calls and reattached to later sandboxes.
Save important UTF-8 scripts, results and markers as flat files under
/mnt/data/checkpoints BEFORE calling a local tool. Limits: 16 files, 2 MB per file,
4 MB retained total, 64 captured versions, and 60 seconds per boundary (or less
when the call deadline is nearer). No other directory is automatically preserved.
Read sandbox_checkpoint in EVERY tool reply. Confirm publication succeeded and
the checkpoint status is passed. A failed transfer is not recovered evidence.
After a local call, check which files exist. If the sandbox changed, copy listed
read-only attachment_path files back to their sandbox_path before use; attachments
are not automatically placed at their original paths. Do not invent missing values
or regenerate a marker and call it the original. publish_artifact persists only
the UTF-8 content you supply; it does not establish sandbox-file provenance.
Report any unrecovered evidence as missing, not verified. Without checkpoint tools,
only explicitly published text is shared and there is no automatic file carry-over.

"""

SEAT_WORKSPACE_SECTION = """\
### Manuscript snapshot supplied by the lead (context only; see provenance) ###
answer.tex:
```tex
{answer_tex}
```

research_notes.tex:
```tex
{research_notes_tex}
```

references.bib:
```bibtex
{references_bib}
```

"""

SEAT_FOLLOWUP_USER = """\
{briefing_section}\
### Follow-up from the lead ###
{task}

Continue from your previous report (your sandbox state from that call is gone).
Restore useful work from that report or published artifacts before recomputing
what is missing. Reply in the same report format.
"""

DELEGATION_GUIDE = """\

### Delegation (multi-agent mode) ###
You lead a small team. Besides your own tools you have a function tool
``delegate`` that runs one wave of subagents in parallel and returns
their reports when all have finished. Roles:
{role_lines}

Subagents are fresh instances of a strong model with code_interpreter
and web search. Each one sees only the problem, your task text, the
optional shared ``briefing`` of the wave, and (by default) the pre-turn
snapshot of the three canonical files. Through read_context it can also read
the latest Critic, council and Compute replies, transferable Compute files,
workflow feedback, and completed earlier-wave reports and published artifacts.
read_context('manifest.json') lists available versioned files and omissions;
these are tool-readable files, NOT paths in your hosted sandbox. Read long files
in chunks until next_offset is null. Nothing shares live edits or private reasoning.
A task must still be self-contained: state exactly what to prove / check / find,
fix the notation, and say what form the answer should take. Use the
briefing for what all tasks share (exact hypotheses, notation, the
Critic's objection, approaches already rejected and why).

How to use it well:
- Delegate early in the turn, before heavy editing. Decide what this
  round needs (Critic review, open issues, your plan), then issue ONE
  wave with up to {max_threads} independent tasks. Good tasks: prove a
  specific lemma; find out whether a result is known and get verified
  references; run an experiment; adversarially check a finished
  argument the Critic doubts.
- Use diversity against getting stuck. When the same substantive
  objection has survived two attempted repairs, do not ask for a third
  patch: send one prover a blind alternative derivation of the disputed
  step (include_workspace false, only the statement), one explorer the
  question whether the disputed lemma or reduction is necessary at all
  or a different architecture avoids it, and one checker the current
  argument. Leave the approach to them where you can; a lead that
  prescribes every route limits the diversity it is buying.
- A checker cannot check a proof that is being written in the same
  wave. Cross-check a specific delivered result in the second wave, or
  do it yourself. Set depends_on to the earlier agent IDs so their exact reports
  and published files are included and validated. A missing dependency is an error.
- include_workspace=false is a BLIND task: no manuscript, reviews, council,
  Compute files or other helpers are shared unless explicitly named in depends_on.
  Continued seats cannot erase their prior context; use a fresh seat for blindness.
- You remain responsible for everything in answer.tex. Verify subagent
  proofs before integrating them (spot-check steps, rerun their
  numerics), rewrite them into your notation, and never paste a claimed
  proof you have not checked. Treat a report as advice from a strong
  colleague, not as truth.
- A follow-up wave can continue a subagent (pass its agent_id). It sees
  its previous task and report, not its old sandbox. You have at most
  {max_waves} waves per turn and each wave takes real time (typically
  10 to 45 minutes). Do not delegate trivialities you can do yourself,
  and do not delegate at all in a round whose Critic review only asks
  for local fixes.
- Do your file edits AFTER the final wave. The provider may give you
  a fresh sandbox after a ``delegate`` call: the wave result tells you
  whether files you wrote before the wave were carried over (as
  read-only attachments to copy back to their canonical paths) and
  which ones. Always run ``ls /mnt/data`` after a wave and re-create
  anything missing; only what is in the sandbox at the end of your
  turn is kept.
- Put what came back into research_notes.tex as mathematics (proved
  lemmas, counterexamples, dead ends and why), not as a delegation
  changelog, so later rounds do not repeat the work.
"""

ASYNC_DELEGATION_GUIDE = """\

### Delegation (asynchronous helpers) ###
You may use delegate to launch up to {max_threads} concurrent helpers. It returns
task IDs without waiting for their research. Roles available:
{role_lines}

You decide whether to delegate, how to divide the work, and when to wait.
helper_status reports current states and published artifact paths. wait_helpers
waits for a selected helper to finish (at most 600 seconds per tool call); a wait
timeout does NOT cancel it. cancel_helpers cancels selected tasks and retains
published partial work. To continue a finished task, pass its agent_id to delegate;
the continuation gets a NEW task ID and its predecessor's published work, not
private reasoning or a restored sandbox. Each task's start time and deadline
are returned on launch. Substantive helper work can take hours; an unfinished
task or a quiet interval is not evidence that it is stuck.

If a helper's result is needed for the manuscript you intend to submit next,
wait for it when your remaining allowance permits. wait_helpers returns when
ANY selected task finishes; wait again with the remaining IDs as needed.
Otherwise continue useful independent work. Do not cancel merely because a
wait expired or your turn is ending. Cancel work that is superseded, irrelevant,
or no longer worth its resource cost; you decide which work meets that test.

Helpers share your research budget and concurrency slots, including while the
critic runs. They are stopped at the research cutoff before final cleanup. You
can continue writing or return your manuscript while helpers work. Results that
arrive later are available on your next Author turn, not inserted into a draft
already under review. Leave relevant tasks running when they can aid a later
turn; record their IDs and the unresolved question in your research notes.
Review inherited task descriptions and source versions before using their
results or duplicating the assignment. You alone edit the canonical manuscript.

By default helpers receive the problem, task and briefing, a launch-time copy
of your saved canonical files when accessible,
the latest critic/council/Compute context and already published helper artifacts.
Save relevant edits before delegating. Launch metadata identifies each file's
source and digest; unavailable current files are explicitly labeled turn-start
fallbacks. Supply missing changes in the task or briefing, or defer that task.
Helpers do NOT see subsequent edits, unsaved work, or private reasoning.
Each result identifies its source round, input hash and snapshot; verify that
it still applies before using it. read_context('manifest.json') lists available
files and omissions. Start a manifest read at offset 0 with revision=null. Follow
next_offset to read full files, passing the returned revision on subsequent
manifest pages; restart at offset 0 with revision=null if it has expired.
Set include_workspace=false for blind work; only the problem, task, briefing and
explicit depends_on artifacts are shared. Dependencies must have finished.

Published files and reports are unreviewed evidence, not accepted proofs. An
interrupted job may still have useful .tex, code or data files even without a
final report. Read its artifact manifest before deciding whether to continue it.
Model spending and time limits still apply. {launch_limit}

Every local tool can replace a Pro sandbox. The tool reply includes a sandbox
note about preserved attachments. Check /mnt/data and restore missing canonical
files from those attachments before editing; do not assume live files survived.
"""


def _delegate_tool_description(roles: list[str], max_threads: int, max_waves: int) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": (
                "Run one wave of subagents in parallel and wait for all of them. "
                "Each task is a self-contained assignment for a fresh subagent with "
                "the given role; pass agent_id from an earlier report to continue "
                "that subagent instead. Blocks until every subagent has reported "
                "(typically 10-45 minutes) and returns all reports. At most "
                f"{max_threads} tasks run concurrently and at most {max_waves} "
                "waves are allowed per turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "briefing": {
                        "type": "string",
                        "description": (
                            "Optional context given verbatim to every subagent in this wave: "
                            "exact hypotheses, notation, the Critic's objection, rejected approaches."
                        ),
                    },
                    "tasks": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": roles},
                                "task": {
                                    "type": "string",
                                    "description": "Self-contained assignment: statement, notation, expected form of the answer.",
                                },
                                "agent_id": {
                                    "type": ["string", "null"],
                                    "description": "Omit or use null/empty string for a fresh subagent; use an earlier subagent ID to continue it.",
                                },
                                "include_workspace": {
                                    "type": "boolean",
                                    "description": "Share the pre-turn files, critic/council/Compute context and earlier helpers (default true). False is blind except for explicit task/briefing and depends_on artifacts.",
                                },
                                "depends_on": {
                                    "type": "array", "items": {"type": "string"},
                                    "description": "Earlier completed agent IDs whose full reports and published artifacts this task needs. No same-wave dependencies.",
                                },
                            },
                            "required": ["role", "task"],
                        },
                    },
                },
                "required": ["tasks"],
            },
        },
    }


class SubAuthorSeat(APICallAgent):
    """One subagent: a fresh strong-model call with a role and a task."""

    description: ClassVar[str] = "Role-specialised subagent working for the lead Author."
    MODEL: ClassVar[ModelSpec] = "models/openai/gpt-6-astra-pro"
    MAX_TOOL_CALLS: ClassVar[int | None] = None
    MAX_WORDS: ClassVar[int] = 2500

    class Inputs(BaseModel):
        role: str
        task: str
        problem: str
        round: int = 0
        briefing: str = ""
        answer_tex: str = ""
        research_notes_tex: str = ""
        references_bib: str = ""
        include_workspace: bool = True
        prior_messages: list[dict[str, Any]] = Field(default_factory=list)
        started_at_utc: str = ""
        deadline_utc: str = ""
        remaining_seconds: float = 0
        wrapup_seconds: float = 300
        context_notice: str = ""
        asynchronous: bool = False

    class Outputs(BaseModel):
        report: str = ""
        messages_after: list[dict[str, Any]] = Field(default_factory=list)

    def __init__(self, ctx: Any, *, model_ref: ModelSpec | None = None, **kw: Any) -> None:
        super().__init__(ctx, **kw)
        if model_ref is not None:
            self.MODEL = model_ref  # type: ignore[misc]
        self.context_view = None
        self._helper_sandbox = None
        self._seat_running = False
        self.interrupted_report = None
        self.provider_outcomes = []
        self._seat_seconds = None
        self._wrapup_seconds = 0

    async def run(self, inp):
        if self._seat_running:
            raise RuntimeError("SubAuthorSeat invocations must use separate seat instances")
        self._seat_running = True
        self._client = None
        self._helper_sandbox = None
        self.interrupted_report = None
        self.provider_outcomes = []
        self._seat_seconds = inp.remaining_seconds if inp.remaining_seconds > 0 else None
        self._wrapup_seconds = min(inp.wrapup_seconds, inp.remaining_seconds / 10) if self._seat_seconds else 0
        try:
            out = await super().run(inp)
            # Cleanup is cancellable too; latch the paid result before awaiting it.
            self.interrupted_report = out.report
            return out
        except asyncio.CancelledError as exc:
            self.interrupted_report = getattr(exc, "completed_report", None) or self.interrupted_report
            raise
        finally:
            try:
                state = self._helper_sandbox
                if state is not None:
                    ids = state.close()
                    if ids:
                        cleanup = asyncio.create_task(asyncio.to_thread(state._delete, ids))
                        cleanup.add_done_callback(lambda task: None if task.cancelled() else task.exception())
                        try:
                            await asyncio.wait_for(asyncio.shield(cleanup), timeout=3.0)
                        except Exception:
                            pass
            finally:
                self._seat_running = False

    def _build_client(self, spec):
        client = super()._build_client(spec)
        if self._helper_sandbox is not None:
            from openai import OpenAI

            descriptor = MultiAuthor._client_container_descriptor(client)
            self._helper_sandbox.bind(descriptor, lambda *, timeout: OpenAI(
                api_key=client.api_key, base_url=client.base_url, timeout=timeout, max_retries=0,
            ))
        return client

    def extra_client_kwargs(self) -> dict[str, Any]:
        from mathagents import load_solver_config

        kwargs = {
            "tools": [
                (None, {"type": "code_interpreter", "container": {"type": "auto", "file_ids": []}}),
                (None, {"type": "web_search_preview"}),
            ],
            "max_tool_calls": {"read_context": 256, "publish_artifact": 64},
        }
        cfg = load_solver_config(self.ctx.model_for(self, self.MODEL))
        responses = cfg.get("api", "openai") == "openai" and cfg.get("use_openai_responses_api", False)
        if self.context_view is not None:
            if responses:
                if self._helper_sandbox is None:
                    self._helper_sandbox = HelperSandbox(self.context_view, self.workdir)
                kwargs["tools"].extend(self._helper_sandbox.tools())
                kwargs["max_tool_calls"]["publish_sandbox_artifact"] = 64
            else:
                kwargs["tools"].extend(self.context_view.tools())
        if responses:
            kwargs["max_hosted_tool_calls"] = self.MAX_TOOL_CALLS
            if self._seat_seconds is not None:
                kwargs["max_wallclock_per_call_s"] = self._seat_seconds
                kwargs["tool_wrapup_reserve_s"] = self._wrapup_seconds
        return kwargs

    def render_messages(self, inp: Inputs) -> list[dict[str, Any]]:
        briefing_section = SEAT_BRIEFING_SECTION.format(briefing=inp.briefing) if inp.briefing.strip() else ""
        briefing_section += (
            f"\nTiming: started {inp.started_at_utc or 'not supplied'}; "
            f"hard {'task' if inp.asynchronous else 'wave'} deadline "
            f"{inp.deadline_utc or 'not supplied'}; {inp.remaining_seconds:.0f} seconds available at start. "
            "The deadline includes elapsed setup time. "
            "Use the sandbox clock to check time; publish useful partial work before the deadline.\n"
            f"Reserve the final {min(inp.wrapup_seconds, inp.remaining_seconds / 10):.0f} seconds INSIDE this allowance "
            "for publishing existing work and your final report; do not begin new long calculations then.\n"
            + inp.context_notice + "\n"
        )
        briefing_section += SEAT_SANDBOX_NOTICE
        if inp.prior_messages:
            return list(inp.prior_messages) + [
                {"role": "user", "content": SEAT_FOLLOWUP_USER.format(
                    task=inp.task, briefing_section=briefing_section,
                )}
            ]
        system = SEAT_SYSTEM_HEAD.format(max_words=self.MAX_WORDS) + ROLE_INSTRUCTIONS[inp.role]
        workspace_section = ""
        if inp.include_workspace:
            workspace_section = SEAT_WORKSPACE_SECTION.format(
                answer_tex=inp.answer_tex or "(empty)",
                research_notes_tex=inp.research_notes_tex or "(empty)",
                references_bib=inp.references_bib or "(empty)",
            )
        user = SEAT_USER.format(
            problem=inp.problem,
            round=inp.round,
            task=inp.task,
            briefing_section=briefing_section,
            workspace_section=workspace_section,
        )
        return [
            {"role": "developer", "content": system},
            {"role": "user", "content": user},
        ]

    def _on_response(self, raw_text, inp):
        self.interrupted_report = _strip_visible_thought_blocks(raw_text)

    async def _query(self, client, messages, query, *, call_id=None):
        result = await super()._query(client, messages, query, call_id=call_id)
        self.provider_outcomes = result[2].get("provider_outcomes", [])
        return result

    def parse_output(self, raw_text: str, inp: Inputs) -> Outputs:
        if _strip_visible_thought_blocks(raw_text).strip() and self._helper_sandbox and self._helper_sandbox.failures:
            raw_text += "\n\n## Harness checkpoint warnings\n" + "\n".join(
                "- " + error for error in self._helper_sandbox.failures)
        rendered = self.render_messages(inp)
        messages_after = rendered + [{"role": "assistant", "content": raw_text}]
        return self.Outputs(
            report=_strip_visible_thought_blocks(raw_text),
            messages_after=messages_after,
        )


DELEGATION_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    # Direct legacy component users retain blocking semantics; Batch 3 opts in.
    "asynchronous": False,
    "max_tasks_per_turn": None,
    "helper_timeout_s": None,
    # drafter exists but is off by default: the lead writes answer.tex itself.
    "roles": ["explorer", "prover", "checker"],
    "max_threads": 4,
    "max_waves": 2,
    "max_tasks_per_wave": 4,
    # Absolute deadline for a whole wave, from the moment delegate is called.
    "job_timeout_s": 3000,
    "wrapup_reserve_s": 300,
    # Wallclock the lead must still have after a wave to integrate and save.
    "synthesis_reserve_s": 1500,
    "subagent_model": None,
    "role_models": {},
    "seat_max_tool_calls": None,
    "max_report_chars": 14000,
    # Refresh the lead's code_interpreter container this often during a
    # wave; OpenAI expires containers idle for 20 min and a request using
    # an expired container fails.
    "container_keepalive_s": 300,
    # In Pro reasoning mode the provider starts a fresh container for the
    # request after a local tool call, discarding everything the lead
    # wrote. When on (None = auto: Pro mode), files the lead wrote before a
    # wave are copied into the next container as read-only attachments.
    "sandbox_carry_over": None,
    "carry_over_max_bytes": 2_000_000,
}

_CARRY_OVER_SUFFIXES = (".tex", ".bib", ".txt", ".md", ".py", ".json", ".csv", ".log", ".sty", ".cls", ".sage", ".gp")

_WAVE_GRACE_S = 120.0
_SNAPSHOT_ATTEMPTS = 3
_SNAPSHOT_RETRY_S = 20.0
_FILE_API_TIMEOUT_S = 15.0
_CLEANUP_WAIT_S = 20.0
_DELEGATE_CORRECTION_ATTEMPTS = 3


class MultiAuthor(Author):
    """Author plus a ``delegate`` tool that fans out to SubAuthorSeats."""

    description: ClassVar[str] = (
        "Author/Critic-loop lead writer with a delegate tool for parallel role-specialised subagents."
    )
    # None preserves legacy settings when this component is used directly.
    author_parallelism: int | None = None
    delegation: dict[str, Any] = {}

    def _delegation_cfg(self) -> dict[str, Any]:
        cfg = dict(DELEGATION_DEFAULTS)
        cfg.update({k: v for k, v in (self.delegation or {}).items() if v is not None})
        if self.author_parallelism is not None:
            parallelism = self.author_parallelism
            if type(parallelism) is not int or parallelism < 1:
                raise ValueError("author_parallelism must be a positive integer")
            cfg["enabled"] = parallelism > 1
            cfg["max_threads"] = parallelism
            if (self.delegation or {}).get("max_tasks_per_wave") is None:
                cfg["max_tasks_per_wave"] = parallelism
        cfg["roles"] = [r for r in cfg["roles"] if r in ROLE_DESCRIPTIONS] or list(DELEGATION_DEFAULTS["roles"])
        reserve = cfg["wrapup_reserve_s"]
        if not isinstance(reserve, (int, float)) or not math.isfinite(reserve) or reserve < 0:
            raise ValueError("wrapup_reserve_s must be finite and nonnegative")
        cap = cfg["seat_max_tool_calls"]
        if cap is not None and (type(cap) is not int or cap < 0):
            raise ValueError("seat_max_tool_calls must be null or a nonnegative integer")
        if type(cfg["asynchronous"]) is not bool:
            raise ValueError("asynchronous must be boolean")
        if type(cfg["max_threads"]) is not int or cfg["max_threads"] < 1:
            raise ValueError("max_threads must be a positive integer")
        cap = cfg["max_tasks_per_turn"]
        if cap is not None and (type(cap) is not int or cap < 1):
            raise ValueError("max_tasks_per_turn must be null or a positive integer")
        cap = cfg["helper_timeout_s"]
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, float))
                                or not math.isfinite(cap) or cap <= 0):
            raise ValueError("helper_timeout_s must be null or finite and positive")
        reserve = cfg["synthesis_reserve_s"]
        if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not math.isfinite(reserve) or reserve < 0:
            raise ValueError("synthesis_reserve_s must be finite and nonnegative")
        return cfg

    def _cache_config_snapshot(self) -> dict[str, Any]:
        snapshot = super()._cache_config_snapshot()
        if self.author_parallelism is not None:
            snapshot["author_parallelism"] = self.author_parallelism
        return snapshot

    def delegation_enabled(self) -> bool:
        return bool(self._delegation_cfg().get("enabled"))

    # ---- prompt hooks -----------------------------------------------------

    def _guide(self) -> str:
        cfg = self._delegation_cfg()
        role_lines = "\n".join(f"  - {r}: {ROLE_DESCRIPTIONS[r]}" for r in cfg["roles"])
        if cfg["asynchronous"]:
            cap = cfg["max_tasks_per_turn"]
            return ASYNC_DELEGATION_GUIDE.format(role_lines=role_lines, max_threads=cfg["max_threads"],
                launch_limit=(f"At most {cap} jobs per turn, including recovered launches and continuations."
                              if cap is not None else ""))
        return DELEGATION_GUIDE.format(
            role_lines=role_lines,
            max_threads=min(cfg["max_threads"], cfg["max_tasks_per_wave"]),
            max_waves=cfg["max_waves"],
        )

    def _with_guide(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.delegation_enabled():
            return messages
        out = [dict(m) for m in messages]
        for m in out:
            if m.get("role") in ("developer", "system"):
                m["content"] = str(m.get("content", "")) + self._guide()
                break
        session = getattr(self, "_async_session", None)
        if (self._delegation_cfg()["asynchronous"] and session is not None
                and (session.records or session.warnings or session.failure)):
            recent = list(session.records)[-12:]
            selected = list(dict.fromkeys([*(i for i in session.records if not session.finished(i)), *recent]))
            summary = [{**{k: session.records[i].get(k) for k in
                           ("agent_id", "status", "round", "input_hash", "snapshot", "report_path", "deadline_utc")},
                        "task": session.records[i]["task"][:300],
                        "task_path": f"helper-inputs/{i}/task.txt"} for i in selected]
            out.append({"role": "user", "content": (
                f"Existing asynchronous helpers ({len(session.records)} total; running and recent jobs below). "
                "Check whether these assignments still help the current manuscript before using results, "
                "cancelling, or duplicating work. helper_status and read_context provide current status "
                "and published work; no remote conversation or sandbox is restored after a restart.\n"
                + json.dumps({"helpers": summary, "warnings": session.warnings, "error": session.failure})
            )})
        if getattr(self, "_recovered", False):
            out.append({"role": "user", "content": (
                "RECOVERED RESEARCH: this Author invocation was interrupted. Read read_context('manifest.json') "
                "and the saved helper reports before doing further research. These are unreviewed claims, not accepted proofs. "
                "No remote sandbox or provider conversation has been restored. Read saved scripts via read_context and "
                "recreate needed files in your current sandbox. Do not repeat completed helper tasks. "
                f"Already consumed {self._waves_done} delegation wave(s); those allowances have NOT reset.\n"
                + self.delegation_summary()
            )})
        return out

    def _begin_turn(self, inp: Author.Inputs) -> None:
        previous_turn = getattr(self, "_async_turn", None)
        if previous_turn is not None:
            previous_turn["open"] = False
        self._current_inp = inp
        self._waves_done = 0
        self._agent_counter = 0
        self._seats: dict[str, dict[str, Any]] = {}
        self._delegation_log: list[dict[str, Any]] = []
        self._context = DelegationContext(self.workdir / "context", round=inp.round)
        self._manifest_pages = ManifestPages()
        self._context_reader = self._context.view()
        self._recovered = False
        if self.delegation_enabled():
            from .delegation_recovery import restore_checkpoint
            if not self._delegation_cfg()["asynchronous"]:
                self._recovered = restore_checkpoint(self)
            for name, body in {
                "problem.txt": inp.problem, "answer.tex": inp.answer_tex,
                "research_notes.tex": inp.research_notes_tex, "references.bib": inp.references_bib,
                "critic.md": inp.prev_critique, "council.md": inp.prev_council,
                "compute_response.md": inp.prev_compute_response, "workflow_feedback.md": inp.workflow_feedback,
            }.items():
                try:
                    self._context.put("round/" + name, body, source=f"Author round {inp.round} input")
                except (ValueError, OSError) as exc:
                    self._context.omissions.append(f"{name} unavailable: {type(exc).__name__}")
            if inp.compute_zip_path:
                self._context.add_compute(inp.compute_zip_path)
            if self._delegation_cfg()["asynchronous"]:
                self._async_session = session_for(self)
        self._budget_stop: BudgetExhausted | None = None
        self._active_wave: asyncio.Task | None = None
        self._sandbox_carry = SandboxCarryState()
        self._carried = self._sandbox_carry.carried
        self._carry_ids_to_delete = self._sandbox_carry.file_ids_to_delete
        self._async_turn = {"open": True}
        # The tool runs in the APIClient worker thread; the wave must run on
        # the lead's loop *and* in the lead's contextvars (agent path, parent
        # call id, workdir) so seat events and artifacts nest under the Author.
        self._ctxvars = contextvars.copy_context()
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def render_messages(self, inp: Author.Inputs) -> list[dict[str, Any]]:
        self._begin_turn(inp)
        return self._with_guide(super().render_messages(inp))

    def _render_container_messages(self, inp: Author.Inputs, workspace_listing: str) -> list[dict[str, Any]]:
        self._begin_turn(inp)
        return self._with_guide(super()._render_container_messages(inp, workspace_listing))

    # ---- client hooks -----------------------------------------------------

    def _delegate_tool(self) -> tuple[Any, dict[str, Any]]:
        cfg = self._delegation_cfg()
        if cfg["asynchronous"]:
            desc = _delegate_tool_description(cfg["roles"], cfg["max_threads"], cfg["max_waves"])
            desc["function"]["description"] = (
                "Launch asynchronous helpers and return task IDs immediately after sandbox preservation. "
                "Use helper_status/read_context for progress, wait_helpers to wait, cancel_helpers to stop. "
                "Passing agent_id starts a new continuation from a finished helper's published work."
            )
            desc["function"]["parameters"]["properties"]["tasks"]["maxItems"] = cfg["max_threads"]
            desc["function"]["parameters"]["properties"]["briefing"]["description"] = "Shared context given verbatim to every helper in this launch."
            props = desc["function"]["parameters"]["properties"]["tasks"]["items"]["properties"]
            props["agent_id"]["description"] = (
                "Omit or use null/empty string for a fresh helper; use a finished helper's ID "
                "to continue its published work under a new task ID.")
            props["depends_on"]["description"] = "Finished helper IDs whose reports and artifacts this task needs."
            props["include_workspace"]["description"] = (
                "Share saved canonical files captured at launch, critic/council/Compute context and helper artifacts. "
                "Inspect snapshot warnings for turn-start fallbacks. False gives a blind task except for task, briefing and depends_on.")
            return self._launch_helpers, desc
        return (
            self._delegate,
            _delegate_tool_description(cfg["roles"], min(cfg["max_threads"], cfg["max_tasks_per_wave"]), cfg["max_waves"]),
        )

    def extra_client_kwargs(self) -> dict[str, Any]:
        kwargs = super().extra_client_kwargs()
        if self.delegation_enabled():
            kwargs["tools"] = list(kwargs["tools"]) + self._context_tools() + self._helper_tools()
            kwargs["max_tool_calls"] = self._lead_tool_limits()
        return kwargs

    def _lead_tool_limits(self) -> dict[str, int]:
        # APIClient counts rejected requests too; _waves_done alone limits work.
        waves = self._delegation_cfg()["max_waves"]
        if self._delegation_cfg()["asynchronous"]:
            cap = self._delegation_cfg()["max_tasks_per_turn"]
            return {"delegate": cap + _DELEGATE_CORRECTION_ATTEMPTS if cap is not None else 256,
                    "read_context": 256, "helper_status": 256, "wait_helpers": 256, "cancel_helpers": 256}
        return {"delegate": waves + _DELEGATE_CORRECTION_ATTEMPTS if waves > 0 else 0, "read_context": 256}

    def _context_tools(self):
        # Some callers build client kwargs before rendering the first prompt.
        if not hasattr(self, "_context"):
            self._context = DelegationContext(self.workdir / "context", round=0)
            self._context_reader = self._context.view()
        # The inline API client is cached across turns; resolve the current
        # invocation's store when called rather than binding its first snapshot.
        _, description = self._context.view().tools()[0]
        return [(self._read_context, description)]

    def _read_context(self, path="manifest.json", offset=0, max_chars=12000, revision=None, messages=None,
                      call_deadline_monotonic_s=None):
        if self._delegation_cfg()["asynchronous"] and getattr(self, "_async_session", None) is not None:
            async def read():
                if not isinstance(path, str):
                    return {"error": "path must be a string"}
                if path.startswith(("helpers/", "helper-inputs/")):
                    return json.loads(self._async_session.read(path, offset, max_chars))
                if path == "manifest.json":
                    with self._context._lock:
                        return self._manifest_pages.read(lambda: {
                            "round": self._current_inp.round,
                            "files": list(self._context.metadata.values()) + self._async_session.manifest_entries(),
                            "omissions": list(self._context.omissions)}, offset, max_chars, revision)
                return json.loads(self._context_reader.read(path, offset, max_chars, revision))
            return self._async_tool(read, messages, call_deadline_monotonic_s)
        return self._context_reader.read(path, offset, max_chars, revision)

    def _helper_tools(self):
        from .delegation_context import _tool
        tools = [self._delegate_tool()]
        if self._delegation_cfg()["asynchronous"]:
            ids = {"agent_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                                 "description": "Explicit nonempty list of existing helper IDs."}}
            status_ids = {"agent_ids": {"type": ["array", "null"], "items": {"type": "string"},
                                        "description": "Omit, use null, or use [] for all helpers; otherwise select existing IDs."}}
            tools += [
                (self._helper_status, _tool("helper_status", "Read helper status and artifact paths; omit IDs or use null/[] for all helpers, including earlier rounds.", status_ids, [])),
                (self._wait_helpers, _tool("wait_helpers", "Wait until any selected helper finishes or timeout_s elapses. Does not cancel research.",
                 {**ids, "timeout_s": {"type": "number", "minimum": 0, "maximum": 600}}, ["agent_ids"])),
                (self._cancel_helpers, _tool("cancel_helpers", "Cancel selected helpers and retain their partial artifacts.", ids, ["agent_ids"])),
            ]
        return tools

    def _async_tool(self, action, messages=None, deadline=None):
        loop = getattr(self, "_loop", None)
        turn = getattr(self, "_async_turn", {})
        if loop is None or loop.is_closed() or not turn.get("open"):
            return json.dumps({"error": "Author invocation is closed"})
        try:
            if asyncio.get_running_loop() is loop:
                return json.dumps({"error": "Synchronous helper tools must run in the API worker thread"})
        except RuntimeError:
            pass
        container = find_container_id(messages or [])
        async def run():
            if not turn.get("open"):
                return {"error": "Author invocation is closed"}
            carry = None
            if container and self._carry_over_enabled():
                until = min(time.monotonic() + 60, deadline) if deadline is not None else time.monotonic() + 60
                carry = await self._snapshot_sandbox(container, deadline=until)
            if not turn.get("open") or (deadline is not None and time.monotonic() >= deadline):
                return {"error": "Author invocation deadline reached"}
            try:
                result = await action()
            except (ValueError, BudgetExhausted) as exc:
                result = {"error": str(exc)}
            result["sandbox_note"] = self._render_sandbox_note(carry)
            return result
        future = self._submit_in_lead_context(loop, run(), track_wave=False)
        try:
            timeout = max(0.0, deadline - time.monotonic()) if deadline is not None else 680
            return json.dumps(future.result(timeout=timeout))
        except concurrent.futures.TimeoutError:
            future.cancel()
            return json.dumps({"error": "Tool wait timed out; check helper_status before launching again"})

    def _launch_helpers(self, tasks, briefing="", messages=None, call_deadline_monotonic_s=None):
        async def launch():
            parsed = json.loads(tasks) if isinstance(tasks, str) else tasks
            turn = self._async_turn
            snapshot = None
            if isinstance(parsed, list) and any(isinstance(t, dict) and t.get("include_workspace", True) for t in parsed):
                snapshot = await launch_snapshot(self, find_container_id(messages or []), call_deadline_monotonic_s)
            if (not turn.get("open") or turn is not self._async_turn
                    or (call_deadline_monotonic_s is not None and time.monotonic() >= call_deadline_monotonic_s)):
                return {"error": "Author invocation closed during launch snapshot"}
            return await self._async_session.launch(self, parsed, briefing, call_deadline_monotonic_s, snapshot)
        return self._async_tool(launch, messages, call_deadline_monotonic_s)

    def _helper_status(self, agent_ids=None, messages=None, call_deadline_monotonic_s=None):
        async def status():
            return self._async_session.status(agent_ids)
        return self._async_tool(status, messages, call_deadline_monotonic_s)

    def _wait_helpers(self, agent_ids, timeout_s=600, messages=None, call_deadline_monotonic_s=None):
        async def wait():
            if not isinstance(agent_ids, list) or not agent_ids:
                raise ValueError("wait_helpers requires an explicit nonempty list of helper IDs")
            return await self._async_session.wait(agent_ids, timeout_s)
        return self._async_tool(wait, messages, call_deadline_monotonic_s)

    def _cancel_helpers(self, agent_ids, messages=None, call_deadline_monotonic_s=None):
        async def cancel():
            if not isinstance(agent_ids, list) or not agent_ids:
                raise ValueError("cancel_helpers requires an explicit nonempty list of helper IDs")
            return await self._async_session.cancel(agent_ids)
        return self._async_tool(cancel, messages, call_deadline_monotonic_s)

    def _build_api_client_with_file_ids(self, file_ids: list[str]):
        if not self.delegation_enabled():
            return super()._build_api_client_with_file_ids(file_ids)
        cfg = self._container_model_config()
        self._pro_mode = str((cfg.get("reasoning") or {}).get("mode", "")).lower() == "pro"
        cfg["tools"] = [
            (None, {"type": "code_interpreter", "container": {"type": "auto", "file_ids": list(file_ids)}}),
            (None, {"type": "web_search_preview"}),
            *self._context_tools(),
            *self._helper_tools(),
        ]
        cfg["max_tool_calls"] = self._lead_tool_limits()
        client = self.ctx.api_client_factory(self._limit_client_deadline(cfg))
        # The factory deep-copies cfg (load_solver_config), so the descriptor
        # to mutate for carry-over is the one the built client actually sends:
        # APIClient copies it only shallowly per request.
        self._ci_container = self._client_container_descriptor(client) or {"type": "auto", "file_ids": list(file_ids)}
        state = getattr(self, "_sandbox_carry", None)
        if state is not None:
            state.bind_container(self._ci_container)
        return client

    @staticmethod
    def _client_container_descriptor(client: Any) -> dict[str, Any] | None:
        for desc in getattr(client, "tool_descriptions", None) or []:
            if isinstance(desc, dict) and desc.get("type") == "code_interpreter":
                container = desc.get("container")
                if isinstance(container, dict) and isinstance(container.get("file_ids"), list):
                    return container
        return None

    def _build_anthropic_api_client_with_files(self):
        if not self.delegation_enabled():
            return super()._build_anthropic_api_client_with_files()
        factory = self.ctx.api_client_factory

        def _with_delegate(cfg):
            cfg = dict(cfg)
            cfg["tools"] = list(cfg.get("tools") or []) + self._context_tools() + self._helper_tools()
            cfg["max_tool_calls"] = self._lead_tool_limits()
            return factory(cfg)

        self.ctx.api_client_factory = _with_delegate
        try:
            return super()._build_anthropic_api_client_with_files()
        finally:
            self.ctx.api_client_factory = factory

    async def run(self, inp: Author.Inputs) -> Author.Outputs:
        try:
            return await super().run(inp)
        finally:
            if hasattr(self, "_async_turn"):
                self._async_turn["open"] = False
            session = getattr(self, "_async_session", None)
            try:
                if session is not None and not workflow_owns_helpers():
                    await session.close()
            finally:
                await self._close_sandbox_carry()

    async def _query(self, client, messages, query, *, call_id=None):
        state = getattr(self, "_sandbox_carry", None)
        try:
            result = await super()._query(client, messages, query, call_id=call_id)
        except BaseException:
            wave = getattr(self, "_active_wave", None)
            if wave is not None and not wave.done():
                wave.cancel()
                await asyncio.gather(wave, return_exceptions=True)
            # Attachment fallback can start another turn before run() exits.
            await self._close_sandbox_carry(state)
            raise
        if self.delegation_enabled() and result is not None:
            # The full conversation (with code_interpreter_call items and
            # their container ids) is the only record of what happened
            # between waves; the base class keeps just the final text.
            try:
                (self.workdir / "conversation.json").write_text(
                    json.dumps(result[1], default=str, indent=1), encoding="utf-8"
                )
            except (OSError, TypeError, IndexError):
                pass
        return result

    # ---- the tool ---------------------------------------------------------

    def _delegate(
        self, tasks: Any, briefing: str = "", messages: Any = None,
        call_deadline_monotonic_s: float | None = None,
    ) -> str:
        """Runs in the APIClient tool loop's worker thread; blocks until the wave is done."""
        cfg = self._delegation_cfg()
        if not cfg["enabled"]:
            return "Author delegation is disabled. Continue with your own tools."
        if isinstance(tasks, str):
            try:
                tasks = json.loads(tasks)
            except json.JSONDecodeError:
                return "Error: `tasks` must be a JSON array of {role, task} objects."
        if not isinstance(tasks, list) or not tasks:
            return "Error: `tasks` must be a non-empty array of {role, task} objects."
        if self._budget_stop is not None:
            return "The run budget is exhausted. No further delegation; save your files and end the turn now."
        if self._waves_done >= cfg["max_waves"]:
            return (
                f"No delegation waves left this turn ({cfg['max_waves']} used). "
                "Finish the turn with your own tools."
            )
        if len(tasks) > cfg["max_tasks_per_wave"]:
            return f"Error: at most {cfg['max_tasks_per_wave']} tasks per wave (got {len(tasks)}). Merge or drop tasks and call again."
        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for i, t in enumerate(tasks):
            if not isinstance(t, dict) or not str(t.get("task", "")).strip():
                return f"Error: task #{i + 1} has no `task` text."
            role = str(t.get("role", "")).strip()
            agent_id = t.get("agent_id") or None
            if type(t.get("include_workspace", True)) is not bool:
                return "Error: include_workspace must be a boolean."
            if agent_id is not None and not isinstance(agent_id, str):
                return "Error: agent_id must name an earlier completed agent."
            if agent_id:
                if agent_id not in self._seats:
                    return f"Error: unknown agent_id {agent_id!r}; known: {sorted(self._seats)}."
                if agent_id in seen_ids:
                    return f"Error: agent_id {agent_id!r} appears twice in this wave; one follow-up per subagent per wave."
                seen_ids.add(agent_id)
                role = self._seats[agent_id]["role"]
            if role not in cfg["roles"]:
                return f"Error: task #{i + 1} has unknown role {role!r}; allowed: {cfg['roles']}."
            dependencies = t.get("depends_on", [])
            if not isinstance(dependencies, list) or any(not isinstance(d, str) or d not in self._seats for d in dependencies):
                return "Error: depends_on must name completed agents from an earlier wave."
            if any(not self._seats[d].get("report_path") for d in dependencies):
                return "Error: a dependency's full report was not transferred; restate its complete findings explicitly."
            if agent_id and not bool(t.get("include_workspace", True)) and self._seats[agent_id].get("shared_context"):
                return "Error: a continued agent cannot forget shared context; start a fresh blind agent."
            normalized.append(
                {
                    "role": role,
                    "task": str(t["task"]),
                    "agent_id": agent_id,
                    "include_workspace": bool(t.get("include_workspace", True)),
                    "depends_on": dependencies,
                }
            )
        loop = getattr(self, "_loop", None)
        if loop is None or loop.is_closed():
            return "Error: delegation unavailable (no event loop)."

        deadline_s = float(cfg["job_timeout_s"])
        remaining = self.tracker.remaining_wallclock_s()
        if call_deadline_monotonic_s is not None:
            call_remaining = max(0.0, call_deadline_monotonic_s - time.monotonic())
            remaining = min(remaining, call_remaining) if remaining is not None else call_remaining
        if remaining is not None:
            available = remaining - float(cfg["synthesis_reserve_s"])
            if available < min(300.0, deadline_s):
                return (
                    "Not enough wallclock budget left for a delegation wave. "
                    "Finish the turn with your own tools now."
                )
            deadline_s = min(deadline_s, available)

        container_id = find_container_id(messages or []) if isinstance(messages, list) else None
        self._waves_done += 1
        wave = self._waves_done
        fut = self._submit_in_lead_context(
            loop, self._run_wave(wave, normalized, str(briefing or ""), deadline_s, container_id)
        )
        try:
            return fut.result(timeout=deadline_s + _WAVE_GRACE_S)
        except (Exception, asyncio.CancelledError) as e:  # noqa: BLE001 — must not crash the lead's tool loop
            if not fut.done():
                loop.call_soon_threadsafe(self._cancel_active_wave)
            return f"Error: delegation wave {wave} failed: {type(e).__name__}: {e}"

    def _cancel_active_wave(self) -> None:
        wave = getattr(self, "_active_wave", None)
        if wave is not None and not wave.done():
            wave.cancel()

    def _submit_in_lead_context(self, loop: asyncio.AbstractEventLoop, coro: Any, *, track_wave=True) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()

        def _start() -> None:
            if fut.cancelled():
                coro.close()
                return
            task = loop.create_task(coro)
            if track_wave:
                self._active_wave = task
            def cancel_task(f):
                if f.cancelled() and not loop.is_closed():
                    loop.call_soon_threadsafe(task.cancel)
            fut.add_done_callback(cancel_task)

            def _done(t: asyncio.Task) -> None:
                if fut.done():
                    if not t.cancelled():
                        t.exception()
                    return
                if t.cancelled():
                    fut.cancel()
                elif t.exception() is not None:
                    fut.set_exception(t.exception())
                else:
                    fut.set_result(t.result())

            task.add_done_callback(_done)

        loop.call_soon_threadsafe(_start, context=self._ctxvars)
        return fut

    def _openai(self, *, timeout: float = _FILE_API_TIMEOUT_S):
        from openai import OpenAI  # lazy: tests stub the module

        return OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout, max_retries=0)

    def _delete_platform_files(self, ids: list[str]) -> None:
        try:
            client = self._openai(timeout=5.0)
        except Exception:  # noqa: BLE001
            return
        for fid in ids:
            try:
                client.files.delete(fid)
            except Exception:  # noqa: BLE001
                pass

    async def _close_sandbox_carry(self, state: SandboxCarryState | None = None) -> None:
        state = state if state is not None else getattr(self, "_sandbox_carry", None)
        if state is None:
            return
        ids = state.close()
        if ids:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._delete_platform_files, ids), timeout=_CLEANUP_WAIT_S
                )
            except Exception:  # noqa: BLE001 — cleanup must not replace the turn result
                pass

    def _carry_over_enabled(self) -> bool:
        flag = self._delegation_cfg().get("sandbox_carry_over")
        if flag is None:
            return bool(getattr(self, "_pro_mode", False))
        return bool(flag)

    def _snapshot_sandbox_sync(
        self, container_id: str, *, state: SandboxCarryState,
        snapshot: SandboxSnapshot, deadline: float,
    ) -> list[tuple[str, str]]:
        """Upload a snapshot; only its owning coroutine can attach it."""
        cfg = self._delegation_cfg()

        def check_active() -> None:
            if time.monotonic() >= deadline or not state.active(snapshot):
                raise TimeoutError("Sandbox snapshot deadline reached or turn closed")

        check_active()
        client = self._openai(timeout=min(_FILE_API_TIMEOUT_S, max(0.001, deadline - time.monotonic())))
        carried: list[tuple[str, str]] = []
        for cf in client.containers.files.list(container_id):
            check_active()
            path = str(getattr(cf, "path", "") or "")
            if not path.startswith(CONTAINER_DATA_ROOT + "/"):
                continue
            name = path[len(CONTAINER_DATA_ROOT) + 1 :]
            if "/" in name or name.startswith("file-") or not name.lower().endswith(_CARRY_OVER_SUFFIXES):
                continue
            size = getattr(cf, "bytes", None)
            if size is not None and size > int(cfg["carry_over_max_bytes"]):
                continue
            body = _read_container_file(client, container_id, cf.id)
            check_active()
            if body == "":
                # The Files API rejects zero bytes. Preserve emptiness in the
                # snapshot instead of losing every file or inserting text.
                snapshot.empty_files.append(name)
                continue
            up = client.files.create(file=(name, body.encode("utf-8")), purpose="user_data")
            new_path = f"{CONTAINER_DATA_ROOT}/{up.id}-{name}"
            if not state.register_upload(snapshot, name, up.id, new_path):
                self._delete_platform_files([up.id])
                raise TimeoutError("Sandbox snapshot was abandoned during upload")
            check_active()
            carried.append((name, new_path))
        return carried

    async def _snapshot_sandbox(
        self, container_id: str, *, deadline: float | None = None,
        state: SandboxCarryState | None = None,
    ) -> dict[str, Any]:
        state = state if state is not None else self._sandbox_carry
        if deadline is None:
            deadline = time.monotonic() + _SNAPSHOT_ATTEMPTS * 300 + (_SNAPSHOT_ATTEMPTS - 1) * _SNAPSHOT_RETRY_S
        rec: dict[str, Any] = {"container_id": container_id, "files": [], "error": "Sandbox snapshot deadline reached"}
        for attempt in range(_SNAPSHOT_ATTEMPTS):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            snapshot = SandboxSnapshot()
            worker = asyncio.create_task(asyncio.to_thread(
                self._snapshot_sandbox_sync, container_id, state=state,
                snapshot=snapshot, deadline=min(deadline, time.monotonic() + 300),
            ))
            # A timed-out thread can still finish an upload. Keep its result
            # observed, but never let it commit to this or a later turn.
            worker.add_done_callback(lambda task: None if task.cancelled() else task.exception())
            try:
                files = await asyncio.wait_for(asyncio.shield(worker), timeout=min(300, remaining))
                if time.monotonic() >= deadline or not state.commit(snapshot):
                    raise TimeoutError("Sandbox snapshot deadline reached or turn closed")
                rec = {"container_id": container_id, "files": files,
                       "empty_files": list(snapshot.empty_files), "error": None}
                break
            except asyncio.CancelledError:
                state.abandon(snapshot)
                raise
            except Exception as e:  # noqa: BLE001
                state.abandon(snapshot)
                rec = {"container_id": container_id, "files": [], "error": f"{type(e).__name__}: {e}"}
                if attempt + 1 < _SNAPSHOT_ATTEMPTS:
                    await asyncio.sleep(min(_SNAPSHOT_RETRY_S, max(0.0, deadline - time.monotonic())))
        await self.events.emit("ac.author.sandbox_carry_over", rec)
        return rec

    async def _keep_container_alive(self, container_id: str, every_s: float) -> None:
        client = self._openai()
        while True:
            await asyncio.sleep(every_s)
            try:
                c = await asyncio.to_thread(client.containers.retrieve, container_id)
                status = getattr(c, "status", None)
            except Exception as e:  # noqa: BLE001
                status = f"error: {type(e).__name__}: {e}"
            await self.events.emit(
                "ac.author.container_keepalive", {"container_id": container_id, "status": str(status)}
            )

    async def _run_wave(
        self,
        wave: int,
        tasks: list[dict[str, Any]],
        briefing: str,
        deadline_s: float,
        container_id: str | None,
    ) -> str:
        cfg = self._delegation_cfg()
        inp = self._current_inp
        lead_model = self.ctx.model_for(self, self.MODEL)
        sem = asyncio.Semaphore(int(cfg["max_threads"]))
        started = time.monotonic()
        deadline_unix = time.time() + deadline_s
        prior_files = set(self._context.files)
        prior_seats = dict(self._seats)
        await self.events.emit(
            "ac.author.delegate.wave_start",
            {
                "wave": wave, "n_tasks": len(tasks), "roles": [t["role"] for t in tasks],
                "deadline_s": deadline_s, "container_id": container_id, "briefing_chars": len(briefing),
            },
        )

        recs: list[dict[str, Any]] = []
        model_refs: dict[str, ModelSpec] = {}
        for t in tasks:
            agent_id = t["agent_id"]
            if not agent_id:
                self._agent_counter += 1
                agent_id = f"{t['role']}{self._agent_counter}"
            model_ref = (cfg["role_models"] or {}).get(t["role"]) or cfg["subagent_model"] or lead_model
            model_refs[agent_id] = model_ref
            recs.append(
                {
                    "wave": wave, "agent_id": agent_id, "role": t["role"], "model": str(model_ref),
                    "task": t["task"], "followup": bool(t["agent_id"]), "include_workspace": t["include_workspace"],
                    "report": "", "error": None, "duration_s": 0.0, "usd": 0.0,
                    "depends_on": t.get("depends_on", []), "artifacts": [],
                }
            )

        from .delegation_recovery import save_checkpoint
        self._delegation_log.extend(recs)
        # Commit consumed allowances before launching any paid helper work.
        save_checkpoint(self)

        async def _one(rec: dict[str, Any]) -> None:
            role = rec["role"]
            previous = prior_seats.get(rec["agent_id"], {})
            prior = list(previous.get("messages_after") or []) if rec["followup"] else []
            seat = SubAuthorSeat(
                self.ctx,
                model_ref=model_refs[rec["agent_id"]],
                name=f"SubAuthor.{rec['agent_id']}",
                parent_budget_scope=self.tracker.scope,
            )
            seat.MAX_TOOL_CALLS = cfg["seat_max_tool_calls"]
            allowed = set(prior_files) if rec["include_workspace"] else set()
            for dependency in rec["depends_on"]:
                allowed.update(prior_seats[dependency].get("artifacts", []))
            prior_artifacts = []
            if rec["followup"]:
                prior_artifacts = list(previous.get("artifacts", []))
                allowed.update(prior_artifacts)
            view = self._context.view(allowed, publisher=f"helpers/wave{wave}-{rec['agent_id']}",
                                      include_omissions=rec["include_workspace"])
            seat.context_view = view

            def retain(out):
                rec["report"] = (out.report or "").strip() or "(empty response from provider)"
                report_path = None
                try:
                    report_path = view.retain_report(rec["report"])
                except (ValueError, OSError) as exc:
                    rec["artifact_error"] = str(exc) if isinstance(exc, ValueError) else "report persistence failed"
                rec["artifacts"] = prior_artifacts + list(view.published)
                self._seats[rec["agent_id"]] = {
                    "role": role, "messages_after": out.messages_after, "artifacts": rec["artifacts"],
                    "report_path": report_path,
                    "shared_context": rec["include_workspace"] or bool(previous.get("shared_context")),
                }
            t0 = time.monotonic()
            try:
                async with sem:
                    ws = rec["include_workspace"]
                    remaining = max(0.0, started + deadline_s - time.monotonic())
                    if remaining <= 0:
                        raise TimeoutError("wave expired while queued")
                    out = await seat(
                        role=role,
                        task=rec["task"],
                        problem=inp.problem,
                        round=inp.round,
                        briefing=briefing,
                        answer_tex=inp.answer_tex if ws else "",
                        research_notes_tex=inp.research_notes_tex if ws else "",
                        references_bib=inp.references_bib if ws else "",
                        include_workspace=ws,
                        prior_messages=prior,
                        started_at_utc=datetime.now(timezone.utc).isoformat(),
                        deadline_utc=datetime.fromtimestamp(deadline_unix, timezone.utc).isoformat(),
                        remaining_seconds=remaining,
                        wrapup_seconds=cfg["wrapup_reserve_s"],
                        context_notice=("Use read_context('manifest.json') for the exact available files and omissions. "
                                        + ("Shared round context and completed earlier-wave artifacts are available. " if ws else "Blind context: only explicitly selected dependencies and your own published artifacts. ")
                                        + "Dependencies: " + ", ".join(rec["depends_on"])),
                    )
                retain(out)
            except BudgetExhausted as e:
                completed = getattr(e, "completed_output", None)
                if isinstance(completed, SubAuthorSeat.Outputs):
                    retain(completed)
                rec["error"] = f"BudgetExhausted({e.scope}): {e}"
                if e.scope == "run":
                    self._budget_stop = e
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
            finally:
                if seat.interrupted_report and not rec["report"]:
                    retain(SubAuthorSeat.Outputs(report=seat.interrupted_report))
                if not rec["report"]:
                    partials = [p for p in view.published if p.endswith(".md")]
                    if partials:
                        retain(SubAuthorSeat.Outputs(report=(
                            "Interrupted/unfinished helper. Last published progress (unreviewed):\n\n"
                            + self._context.files[partials[-1]])))
                rec["artifacts"] = prior_artifacts + list(view.published)
                rec["checkpoint_errors"] = list(seat._helper_sandbox.failures) if seat._helper_sandbox else []
                view.close()
                rec["duration_s"] = time.monotonic() - t0
                rec["usd"] = float(seat.tracker.counters.usd)
                rec["finished"] = True
                save_checkpoint(self)
                await self.events.emit(
                    "ac.author.subagent_done",
                    {k: rec.get(k) for k in ("wave", "agent_id", "role", "model", "duration_s", "usd", "error", "followup", "checkpoint_errors", "artifact_error")},
                )

        pending = {asyncio.create_task(_one(rec), name=f"SubAuthor-{rec['agent_id']}"): rec for rec in recs}
        # The lead is blocked in the tool call, so its sandbox is frozen:
        # snapshot it now, in parallel with the seats.
        snapshot = None
        if container_id and self._carry_over_enabled():
            snapshot = asyncio.create_task(self._snapshot_sandbox(
                container_id, deadline=started + deadline_s, state=self._sandbox_carry
            ))
        keepalive = None
        if container_id and float(cfg["container_keepalive_s"]) > 0:
            keepalive = asyncio.create_task(
                self._keep_container_alive(container_id, float(cfg["container_keepalive_s"]))
            )
        timed_out = False
        interrupted = False
        try:
            while pending:
                left = deadline_s - (time.monotonic() - started)
                if left <= 0:
                    timed_out = True
                    break
                done, _ = await asyncio.wait(pending, timeout=left, return_when=asyncio.FIRST_COMPLETED)
                for d in done:
                    pending.pop(d)
                    d.result()  # A failed durable checkpoint must not be silently ignored.
                if self._budget_stop is not None:
                    break
        except BaseException:
            interrupted = True
            raise
        finally:
            for task, rec in pending.items():
                task.cancel()
                if rec["error"] is None:
                    rec["error"] = (
                        f"cancelled: wave deadline {deadline_s:g} s reached" if timed_out
                        else "cancelled" + (": run budget exhausted" if self._budget_stop else "")
                    )
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            self._write_wave_artifacts(wave, recs)
            save_checkpoint(self)
            if interrupted:
                helpers = [task for task in (snapshot, keepalive) if task is not None]
                for task in helpers:
                    task.cancel()
                await asyncio.gather(*helpers, return_exceptions=True)

        carry = None
        try:
            if snapshot is not None:
                if snapshot.done():
                    carry = snapshot.result()
                else:
                    carry = await asyncio.wait_for(snapshot, timeout=max(0.0, started + deadline_s - time.monotonic()))
        except Exception as e:  # noqa: BLE001 — optional carry-over must not discard seat reports
            carry = {"container_id": container_id, "files": [], "error": f"{type(e).__name__}: {e}"}
        finally:
            helpers = [task for task in (snapshot, keepalive) if task is not None]
            for task in helpers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*helpers, return_exceptions=True)
        total_usd = sum(r["usd"] for r in recs)
        elapsed = time.monotonic() - started
        await self.events.emit(
            "ac.author.delegate.wave_done",
            {
                "wave": wave,
                "duration_s": elapsed,
                "cost_usd": total_usd,
                "agent_ids": [r["agent_id"] for r in recs],
                "errors": sum(1 for r in recs if r["error"]),
                "artifact_degraded": sum(bool(r.get("checkpoint_errors") or r.get("artifact_error")) for r in recs),
                "timed_out": timed_out,
                "budget_stop": self._budget_stop is not None,
            },
        )
        return self._render_wave_for_lead(wave, recs, elapsed, carry)

    def _truncate_report(self, body: str, limit: int, artifact: str) -> str:
        if len(body) <= limit:
            return body
        note = f"\n\n[... Findings truncated by the harness at {limit} characters; read the FULL report using read_context(path={artifact!r}), following next_offset ...]\n\n"
        # Reports put Caveats before Findings, so cutting the tail loses only
        # findings; if a seat ignored the order, keep its Caveats anyway.
        cav = body.find("## Caveats")
        find = body.find("## Findings")
        if 0 <= find < cav:
            caveats = body[cav:]
            head = body[: max(0, limit - len(caveats) - len(note))]
            return head + note + caveats
        return body[:limit] + note

    def _render_sandbox_note(self, carry: dict[str, Any] | None) -> str:
        if carry is None:
            return ""
        head = (
            "Sandbox note: the provider may have started a fresh container for this request, "
            "in which case /mnt/data may not hold the files you saved before this tool call. "
        )
        if carry["error"]:
            return head + (
                f"The harness could not copy them over ({carry['error']}); "
                "run `ls /mnt/data` and re-create anything missing from the original attachments and your notes."
            )
        empty_files = carry.get("empty_files", [])
        if not carry["files"] and not empty_files:
            return head + (
                "No files written by you were found in the previous sandbox, so nothing was carried over; "
                "the original attachments are still available."
            )
        parts = [head]
        if carry["files"]:
            lines = "\n".join(f"- {name}: `{path}`" for name, path in carry["files"])
            parts.append(
                "The harness attached read-only copies of the files saved before this tool call:\n"
                f"{lines}\n"
                "Run `ls /mnt/data` first. If the canonical files (e.g. /mnt/data/answer.tex) are still present, "
                "the sandbox survived and you can ignore the copies; otherwise copy each attachment back to its "
                "canonical path before editing."
            )
        if empty_files:
            paths = ", ".join(f"`{CONTAINER_DATA_ROOT}/{name}`" for name in empty_files)
            parts.append(
                f"These files were empty before this tool call: {paths}. No attachments were uploaded for them. "
                "After a sandbox reset, recreate each as a zero-byte file at its canonical path; "
                "do not restore its contents from older attachments. If the sandbox survived, leave the existing files intact."
            )
        return "\n".join(parts)

    def _render_wave_for_lead(
        self, wave: int, results: list[dict[str, Any]], elapsed: float, carry: dict[str, Any] | None = None
    ) -> str:
        cfg = self._delegation_cfg()
        limit = int(cfg["max_report_chars"])
        parts = [f"## Delegation wave {wave} of {cfg['max_waves']} — {len(results)} subagent(s), {elapsed / 60:.1f} min"]
        for r in results:
            head = f"### {r['agent_id']} (role={r['role']}, model={_short_label(r['model'])}, {r['duration_s'] / 60:.1f} min) ###"
            if r["error"]:
                parts.append(f"{head}\n(error: {r['error']})")
            artifact = f"helpers/wave{wave}-{r['agent_id']}/final-response.md"
            if r["report"]:
                if artifact in r.get("artifacts", []):
                    parts.append(f"{head}\n{self._truncate_report(r['report'], limit, artifact)}")
                else:
                    parts.append(f"{head}\nReport transfer failed; do not rely on omitted findings.\n" + r["report"][:limit])
            if r.get("artifacts"):
                parts.append("Readable with read_context: " + ", ".join(r["artifacts"]))
            if r.get("artifact_error"):
                parts.append("Artifact warning: " + r["artifact_error"])
            if r.get("checkpoint_errors"):
                parts.append("Harness checkpoint warnings (evidence may be missing): " + "; ".join(r["checkpoint_errors"]))
        if self._budget_stop is not None:
            parts.append(
                "RUN BUDGET EXHAUSTED. Do not call any more tools except to save your files: "
                "write the best current versions of answer.tex, research_notes.tex and references.bib "
                "to /mnt/data now and end your turn."
            )
        else:
            left = cfg["max_waves"] - self._waves_done
            parts.append(f"Waves remaining this turn: {left}. To continue a subagent, pass its agent_id.")
        note = self._render_sandbox_note(carry)
        if note:
            parts.append(note)
        elif self._budget_stop is None:
            parts.append("Check `ls /mnt/data`: if files you wrote before this wave are missing, re-create them now.")
        return "\n\n".join(parts)

    def _write_wave_artifacts(self, wave: int, results: list[dict[str, Any]]) -> None:
        try:
            d = self.workdir / "subagents"
            d.mkdir(parents=True, exist_ok=True)
            for r in results:
                (d / f"wave{wave}-{r['agent_id']}.md").write_text(
                    f"# {r['agent_id']} — wave {wave}, role {r['role']}, model {r['model']}\n\n"
                    f"Duration: {r['duration_s']:.0f} s. Cost: ${r['usd']:.2f}. "
                    f"Follow-up: {r['followup']}. Error: {r['error'] or '(none)'}\n\n"
                    f"## Task\n\n{r['task']}\n\n## Report\n\n{r['report']}\n",
                    encoding="utf-8",
                )
            (d / "delegation_log.json").write_text(
                json.dumps(self._delegation_log, indent=1, default=str), encoding="utf-8"
            )
        except OSError:
            pass

    def delegation_summary(self) -> str:
        session = getattr(self, "_async_session", None)
        if self._delegation_cfg()["asynchronous"] and session is not None:
            return json.dumps(session.status(), indent=2)
        log = getattr(self, "_delegation_log", None) or []
        if not log:
            return ""
        lines = [f"{len(log)} subagent call(s) in {getattr(self, '_waves_done', 0)} wave(s):"]
        for r in log:
            status = f"error: {r['error']}" if r["error"] else "ok"
            lines.append(
                f"- wave {r['wave']} {r['agent_id']} ({r['role']}, {_short_label(r['model'])}, "
                f"{r['duration_s'] / 60:.1f} min, ${r['usd']:.2f}) {status}"
            )
        return "\n".join(lines)

    # ---- outputs ----------------------------------------------------------

    def _build_outputs_from_container(
        self, inp, raw_text, modified, container_id, *, via="container_files", execution_failed=False,
    ):
        out = super()._build_outputs_from_container(
            inp, raw_text, modified, container_id, via=via, execution_failed=execution_failed,
        )
        out.delegation_summary = self.delegation_summary()
        return out

    def parse_output(self, raw_text: str, inp: Author.Inputs) -> Author.Outputs:
        out = super().parse_output(raw_text, inp)
        out.delegation_summary = self.delegation_summary()
        return out


__all__ = ["MultiAuthor", "SubAuthorSeat", "ROLE_DESCRIPTIONS", "DELEGATION_DEFAULTS"]
