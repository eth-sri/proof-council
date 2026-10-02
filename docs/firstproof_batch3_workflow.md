# Batch 3 workflow

The submission image selects `firstproof_batch3` automatically. Preparing the
workflow does not authorize or launch paid calls. The standalone Python adapter
retains its legacy default unless `FIRSTPROOF_WORKFLOW=firstproof_batch3` is set.

`author_parallelism` controls Author delegation within this workflow. Its
baseline default is `1`: the ordinary Author has no delegation tool or guide.
At `N > 1`, the lead may use up to `N` concurrent helpers per problem and waits
for their reports before continuing. The Critic, advisory council and Compute
retain their normal behavior. `firstproof_batch3_multiauthor` is the same
configuration with `author_parallelism=4`; changing this one input is sufficient
to switch between the two modes. See `docs/multi_agent_author.md` for delegation
details and advanced settings, including the separate two-wave limit per turn.

Use `--input author_parallelism=N` with `scripts/run_workflow.py`, or
`FIRSTPROOF_AUTHOR_PARALLELISM=N` in the competition adapter. Both accept a
positive integer; invalid or nonpositive environment overrides warn and retain
the selected preset's value. This is a per-problem maximum, not a batch-wide
API limit: ten problems with a value of `4` may run forty remote helpers.

## Candidate-solution path

1. **Research:** the existing Author/Critic DAG, with Astra Pro/max as Author
   and Critic, fresh critic resets and forced-fresh acceptance checks. Council
   remains Astra Pro, Fable 5.1 and Gemini 3.1 Pro. Compute remains an
   Author-requested Astra/xhigh Codex worker, not the Pro driver.
2. **One rewrite:** a compiling answer and Author/Critic agreement start
   editorial review, not automatic submission. David's RewriteSeat and template
   load the improved `src/proofstack/agents/writeup_prompts/WRITING_GUIDANCE.md`.
   The writer receives the manuscript, original problem, research notes and
   16-page contract. No mandatory four passes or unconditional final polish.
3. **Stateful regression critic:** Batch3CleanupCritic gets the original problem,
   pre-rewrite baseline, full edited manuscript, mechanical checks and research
   critic conversation. Each review retains that history and reads the entire
   current manuscript. Prior acceptance is explicitly not evidence of correctness.
4. **Routing:** the critic returns exactly one of the following dispositions.
   - `accept`: the exact manuscript completely solves the original problem.
   - `repair`: only local editorial, notation, reference or formatting work.
   - `restore`: editing introduced a mathematical regression, and the critic
     identifies correct baseline passages that repair it without new mathematics.
   - `research`: a substantive flaw also affects the baseline, its origin is
     uncertain, or repair requires new mathematics. This is a rejection; the
     existing routing name is retained for checkpoint compatibility.
5. **Targeted repairs:** RepairSeat addresses the findings; the stateful critic
   reviews the result again. For `restore`, the editor receives the baseline and
   must restore only the identified proof passages, preserving unaffected edits.
   Restoration is never automatic acceptance. If it cannot fix the regression,
   return to research. There is no full rewrite between these reviews.
6. **Return to research:** a mathematical rejection revokes prior acceptance.
   The edited manuscript, baseline and findings go to the existing research
   checkpoint. Round numbering, research/Compute context, spending and deadlines
   carry on. A newly accepted research candidate starts a new rewrite episode.
7. **Exact-document gate:** normalization reaches a fixed point before review.
   Publication requires a self-contained compiling manuscript of 1--16 pages and an `accept` for
   those exact bytes. No model edit or normalization follows acceptance.

All editorial seats and the regression critic use **Astra Pro/max API**.
The shared Astra Pro preset switches a failing logical call to **standard Astra**
after two retryable failed Pro requests. This applies to Author, critics, the
Astra council seat and editorial seats. The fallback keeps the same model ID,
conversation, tools and remaining deadline; it changes `reasoning.mode` to
`standard`, retaining the effective effort (including the existing max-to-high
timeout downgrade). It remains standard for that call's tool/wrap-up turns and
outer retries. A later independent call starts in Pro again; parallel calls do
not share failure counts. Polling transport errors retry the existing response
ID, not a new generation. Salvaged usable output does not trigger fallback;
terminal credential, quota or configuration errors still fail fast. The switch
is logged, and each request log records the actual reasoning mode. This does
not extend retry, dollar or time limits or guarantee a fallback if time runs out.
Accepted documents go to `submissions/<problem_id>/<sha256>.tex`, with
`submission_approved=true` and a SHA-256 hash. The adapter validates that hash
and refuses further normalization. A model's approval is an automated gate,
not independent verification of a proof.

## Research critic context recovery

Normal research reviews retain their full conversation and the configured
periodic fresh-review schedule. Research notes are now provided as a file by
default; the complete problem, manuscript and bibliography remain inline.
Before each API call, the research critic checks
a conservative text-token estimate, reserving the configured model output
limit and room for tool activity. `ACCritic.context_window_tokens` defaults to
1,050,000 for the default Astra model; `context_tool_reserve_tokens` defaults to
64,000. Set the component's window explicitly when using a different capacity.

Only when the estimate exceeds that input budget does the critic remove earlier
manuscript snapshots. It retains every prior referee report and restores the
complete original question, referee instructions, current manuscript, notes,
and bibliography. If that packet still does not fit, it uses a fresh review
without history or the Author's thinking summary. Current proof files are
never truncated. A provider context rejection of a continuing review also
permits fresh-review recovery, within the existing budget and deadline. The
estimate uses 2.4 ASCII bytes per token and one token per non-ASCII byte, plus
framing overhead. It is not an exact model tokenizer: a large fresh packet is
still submitted intact and only an actual provider context rejection blocks it.

Recovery is local to the critic, so successful recovery does not restart the
Author or cancel concurrent council/Compute work. The returned conversation
and fresh-review turn count are saved in the ordinary research checkpoint.
A packet-specific journal under `critic_context/` also records a started
provider recovery and its completed result. On resume, completed provider receipts
are reused, unresolved provider work blocks a duplicate call, and interrupted
or transiently failed recoveries allow at most two fresh attempts total across
restarts. Cancellation still propagates; it never starts a retry by itself.
An error while polling a known response ID is not a terminal provider status.
Resume retrieves saved response IDs and, if necessary, cancels them within a
30-second reconciliation window. Every outstanding attempt is checked before
a completed report is reused. Missing terminal status still blocks another paid
call; reconciliation never creates a new response. If a retrieved terminal
response omits usage, recovery records a conservative model-token charge priced
at the configured full context window and maximum output allowance. The receipt
keeps `usage_unavailable` and adds `cost_estimated`, the estimated amount and its
basis; this is not measured usage or a bound on separate tool charges. The debit
is not automatically refunded and can exhaust the budget before a retry. Durable
receipts settle only that critic call's previously uncharged costs, not concurrent
Council or Compute spending. Outer resume records restored receipt deltas and
restores dollar and token counters from the same frozen snapshot and pre-await
counter baseline, so concurrent charges are not subtracted from the restoration.
Later receipt updates are charged only when their additional usage is settled.
Ordinary API-call logging and recovery share the same settlement lock across
child contexts. Both subtract usage already logged for each call ID, so either
can finish first without charging twice. Completion metadata is still logged
when the remaining charge is zero. Known paid usage remains charged even if an
ordinary event append fails; an in-process per-call usage floor prevents local
recovery from charging it again, and provider receipts survive process restarts.
New call IDs contain 128 random bits; existing shorter IDs remain readable.
Historical numeric usage strings are normalized before accounting. If an ordinary
call cannot read event history, its completed result and known charge are kept,
with an `accounting.events_unreadable` warning (or process-log fallback). No
`model.call` billing delta is appended without history: provider receipts settle
the missing event later. The in-process floor includes charges restored on resume,
so repeated read failures do not charge those calls twice. Resume itself still
requires readable history and receipts before starting new paid work.
Settlement finishes the event append and
counter update before propagating cancellation, including cancellation during
the event writer thread. If that write fails, cancellation remains the primary
exception (with the write failure chained), preventing a cancelled call from
becoming a retryable logging error.
Damaged supplementary event-log rows are skipped with an
`accounting.events_corrupt` warning listing their line numbers; resumed events
are appended on a new line after any crash fragment. Durable provider receipts
remain strict: corrupt interior receipt rows still block recovery. This cannot
reconstruct usage missing from both the event log and the provider receipts.
Pending reviews preserve their exact mode, conversation and omit-thinking
setting, so extending the round limit cannot bypass a recovery receipt. Omitted
author thinking is excluded from its identity. Unrelated recovery journals do
not invalidate other reviews' cached outputs.
Awaiting-review checkpoints preserve the exact normalized manuscript and notes.
A fresh packet rejected by the provider is a non-retryable context failure; the outer
workflow preserves its retained answer without repeating paid research or
cleanup. Events distinguish preflight, reduction, recovery, and blocked states.
The cleanup critic retains its separate lossless baseline/findings policy.

### File-backed research notes

Both competition presets and the generic `ACCritic` use
`research_notes_transport: file`. This enables the implementation; it does not
claim that the separately authorized live Astra Pro file-review test has passed.
Explicit `inline` mode remains available and retains the context guards and the
instruction to skim for fatal errors while concentrating on `answer.tex`.

In file mode, every nonempty notes
snapshot is saved locally with a round-specific filename and SHA-256, then
uploaded through the configured APIClient to the OpenAI Responses Code
Interpreter tool. The prompt includes its name, byte count and hash, not its
contents. This is a container attachment, **not** a text `input_file`, which
would put the full notes back into the prompt.

The critic must use the Python tool to verify the current file's bytes/hash,
then consult only relevant sections through targeted searches and small
excerpts. It treats the notes as read-only background, not as proof belonging
to the manuscript. Only the current round's file is attached; older filenames
in the conversation are explicitly obsolete. No critic-written files are
downloaded or promoted into the Author workspace.

The default and both competition presets use `research_notes_container: auto`
with the Pro critic. Pro reasoning mode rejects explicit container IDs, so an
explicit-container/Pro configuration is rejected locally before uploading or
calling the model, not repaired by silently selecting a non-Pro critic. The
existing provider-failure retry/fallback policy is unchanged.
Auto mode attaches uploaded files with a 48-hour lifetime; this does not extend
the container's separate 20-minute idle lifetime. The checker-last instruction
mitigates receipt expiry, but cannot guarantee availability if reasoning after
the final checker lasts more than 20 minutes. Live Pro probes have withheld tool
items until completion under both background polling and streaming, so neither
is relied on to discover an auto container early enough for keepalive.

For a deliberately selected non-Pro critic, `research_notes_container: explicit`
remains available: the harness creates a container and uploads notes directly
into it before the model call, so the container ID is known even when background
polling hides tools.
`research-notes-attachment.json` and `ac.critic.notes.attached` record container
and file IDs, round, size and hash. Setup is capped at 120 seconds or the remaining
run deadline, whichever is shorter. Containers expire after 20 idle minutes;
the harness retrieves the container every five minutes to refresh its activity,
through the model call and final receipt retrieval, within the run deadline.
Keepalive failures/status are logged. No model call is made by the keepalive.
Keepalive uses its own single-worker executor; its 15-second limit (five seconds
for a resume touch) includes queue time and is capped by the remaining deadline.
Workers recheck cancellation and the deadline before touching the container.
Setup also uses a dedicated single-worker executor, with queue time included
in its upload deadline. Cancellation/timeout terminates the client and cancels
queued work without waiting for an in-flight worker. A worker rechecks the
deadline and cancellation before uploading; an explicit container returned
after abandonment is discarded rather than orphaned. Keepalive shutdown also
cancels queued work without waiting for an in-flight worker. Unused containers are
deleted best-effort; submitted containers are retained for recovery and expire
naturally after keepalive stops. Resume touches the saved container before
bounded response reconciliation; it cannot revive an already expired container.
Each fresh recovery uploads the same immutable bytes under a new container/file
ID. Completed recovery replay uses its saved report without uploading again.
Tool reads and reasoning still consume the ordinary model budget/context.

Upload errors propagate without silently dropping notes or inlining them,
under either verification policy. Empty notes require no upload or status. Providers without
Responses/Code Interpreter must explicitly select `research_notes_transport:
inline`; this compatibility option retains the same context guards. This
setting does not alter the cleanup critic's baseline/findings protocol.

`ACCritic.research_notes_verification` selects the acceptance policy:

- `required` (generic default): the critic must
  return exactly one `<research_notes_status>verified</research_notes_status>`
  tag, backed by harness-verified evidence. Missing, duplicated or unavailable
  status blocks acceptance.
- `advisory` (selected by both competition presets): still upload the immutable
  snapshot, request the checker, and attempt the same bounded receipt retrievals. Missing evidence,
  including an expired container, does not veto an otherwise positive
  mathematical verdict. A negative or unparseable mathematical verdict is
  never promoted. A detected filename, byte-count or hash mismatch still blocks
  acceptance, including a mismatch explicitly reported by the critic. Upload
  failure still stops the review before a model call.

Both competition presets explicitly select this policy for **new runs**:

```yaml
components:
  ACCritic:
    research_notes_verification: advisory
```

Do not change the policy in a running or resumed experiment: configuration is
part of both cache and recovery identity. Previously started runs keep their
original policy. Inline reviews and cleanup reviews retain their existing
behavior. Other presets remain on the generic `required` default unless they
explicitly select advisory verification.

Advisory mode is a deliberate loss of assurance about background-note access,
not a relaxation of mathematical review. Missing essential proof steps or
computational certificates and LaTeX-contract violations remain grounds for
rejection. Notes-only arguments never count as submitted proof. A receipt
proves access to the matching bytes, not that the reviewer read them carefully.
`not_checked`/`unavailable` remain unverified: outputs keep
`research_notes_execution_verified=false` and verification version zero.
Outputs also record `research_notes_verification_policy`,
`research_notes_integrity_failed`, and `research_notes_advisory_accepted`.
The report warns when access is unconfirmed, and `ac.critic.notes.verification`
records the policy, evidence status, integrity flag, and whether acceptance
used the advisory allowance. A `NotFoundError` alone is not classified as proof
of expiry; advisory mode does not need to infer why evidence is unavailable.

To record **verified** access under either policy, the harness
requires a completed Code Interpreter call executing the supplied hash checker
and a matching filename, byte count and SHA-256 in its tool output. Syntax-tree
comparison tolerates whitespace, quote choices and split imports, but not
different logic or a fabricated printout of the expected metadata.
Diagnostic text around the checker's JSON output is allowed; the metadata must
still match exactly and come from the unchanged, completed checker call.
The checker removes any stale receipt before checking the file, then writes a
small JSON receipt under `/mnt/data`. A failed assertion/read cannot leave old
evidence at that path. When stdout is
missing or unusable, the harness downloads that exact path from the container
of the completed checker call via the [container-files API](https://developers.openai.com/api/docs/guides/tools-code-interpreter).
The receipt must contain the current filename, byte count, SHA-256, and
`receipt_sha256 = sha256(b"receipt:" + data)`. The last digest's expected value
is computed independently by the harness and never included in the prompt;
copying the advertised filename/size/hash into a receipt is not verification.
The receipt is shared by checker calls in the same container, so it establishes
access to the matching bytes there, not which particular call wrote it. A file
without matching executed checker code is not evidence. Polling captures the
receipt as soon as a completed checker becomes visible, before later reasoning
can outlast the container's inactivity expiry. Capture permits at most three
five-second retrievals per invocation, within the run deadline. The prompt also
requests a final checker call immediately before the report. This is additional
protection, not a guarantee that a provider exposes tool items before completion.
If background polling withholds tool items, early capture cannot help. Only the
non-Pro explicit-container option can protect the notes and receipt with
keepalive during this gap; auto-mode Pro reviews retain the expiry risk above.
Post-call retrieval has its own 15-second bound within the remaining deadline;
all downloads require a complete bounded listing and at most 4 KiB of content.
Provider-created files may report `bytes=null`; the streamed-content bound is
always enforced, independently of whether a metadata size is available.
Missing, expired or ambiguous receipts block acceptance in required mode.
Confirmed metadata mismatches block both modes and are retained for replay,
even if another checker produced matching metadata. Replay restores all cached
receipts before attempting downloads or deciding on a verdict, including when
the network deadline has passed. Successful downloads do not skip later items
within the existing three-download/15-second bound. No model call or
inline-notes fallback is made to obtain evidence.
An expired-container 404 cannot establish verified access: even an AST-exact checker
item marked `completed` may have raised an assertion or read error. Its code and
status alone cannot prove successful file access. The paid report is retained
for diagnosis/recovery with `research_notes_status=not_checked`; required mode
sets `answer_ready=false`, while advisory mode retains the mathematical verdict.
Downloaded receipts are saved locally with the tool-call and container IDs;
recovery can reuse them after the remote container expires. Retained stdout from
both the legacy no-receipt checker and the previous three-field receipt checker
remains valid on replay: matching metadata verifies access, and mismatches still
block both policies. Those older checker variants cannot use the file-receipt
fallback. Older receipt-only approvals are not trusted under the revised
verification contract.
Verified outputs now carry verification version 3; older cached approvals must
be revalidated from their retained evidence, not reused solely on their flags.
Large diagnostic logs are bounded without discarding the checker's identity and
code; oversized code is explicitly unverifiable. Responses continuations retain
the supported tool `outputs` field, including explicit null when absent.
A model's `verified` tag alone is insufficient. Evidence is retained in durable
provider receipts for recovery. Cancellation/recovery retrieves missing Code
Interpreter outputs from the saved response ID without starting another model
call. Uploads have three bounded transport attempts;
upload-only failures do not consume either of the two fresh model attempts.

Ordinary fresh/stateful reviews also persist their exact input, report and
provider receipt location before post-call evidence retrieval. Cancellation
during retrieval resumes that paid review, not another model call. A completed
call crossing its monetary budget still retrieves evidence within the remaining
wall-clock time, refreshes its retained verdict, and re-raises the budget stop.
Ordinary reviews remain governed by the workflow's retry and budget limits;
they do not inherit the separate two-attempt fresh context-recovery cap.
A resumed ordinary stateful review rejected for context overflow takes the
same fresh-recovery path as a first invocation, using its saved review packet.
A context rejection of a complete fresh packet remains terminal.
In-place restart and `--restart-copy` preserve these review checkpoints.
`--resume-from` imports only the prior run's verdict cache, not this checkpoint
directory, so it can pay again for a completed research-critic review.
In required mode, unverified research verdicts are not reusable cache hits;
durable reports are rechecked without repaying. Advisory outputs are reusable
only under the same policy and without integrity failures. Positive unverified
outputs must explicitly record advisory acceptance; legacy outputs cannot
silently become advisory approvals. Recovery reparses the saved raw report
with its retained evidence, without starting another model call. A legacy
unverified output with no raw verdict is never promoted. Cleanup critics use
their own verdict cache policy.
Completed inline Anthropic/Google reviews use their provider's terminal stop
reason and settled usage on resume, never OpenAI response reconciliation.
Unknown terminal state or missing usage remains blocked rather than paid again.
The research critic's `max_hosted_tool_calls` defaults to 30 (also explicit in both
competition presets), separately from its 12-call local-function allowance.
The hosted cap reaches the Responses API and is shared across retries/continuations.
It leaves more room for research and the requested final checker, but cannot
guarantee that a model reserves a tool call for it. An ambiguous failed attempt does
not refund its allowance. If required verification has no remaining Python tool
and no validated notes evidence, the client stops instead of paying for a tool-less
retry. Advisory mode allows the ordinary tools-exhausted continuation without
requiring receipt evidence; the resulting report still passes the same verdict
and integrity checks. A completed tool item alone never establishes verified access.

## Unresolved-problem path

At the shared **research cutoff**, `partial_cleanup_seconds=7200` (two hours)
before the run deadline, stop research, including in-flight
calls, and give each unresolved problem's latest saved attempt **one rewrite**.
It also receives the original problem, notes, bibliography data and latest
critic findings. It must preserve gaps and conditional claims and must not
invent a complete solution. **No mathematical critic follows this rewrite.**
The harness normalizes and compiles it, then permits at most two targeted
mechanical repairs for compiler errors or page overflow. These are not new
rewrites or research: they may fix syntax and shorten redundant exposition,
but must preserve essential arguments, caveats and gaps. Every repair is
compiled and measured again before it can replace a fallback.
An initial Claude Code timeout or incomplete edit can consume one repair slot
for a single finishing-only continuation of the same saved session. Workers
must be stopped and usage settled first; external cancellation and uncertain
accounting never trigger continuation. Unfinished edits cannot replace the
compiled fallback until the completion protocol and export gates pass.
When a completed Author draft was saved but is still awaiting review, the
partial writer is explicitly told that the critic findings refer to an earlier
revision. It must not reintroduce obsolete caveats or treat claimed repairs as
verified.

These documents carry a visible partial/unreviewed notice and go to
`partials/<problem_id>/<sha256>.tex`, with `partial_ready=true`, `partial_sha256`,
`output_kind=partial_unreviewed` and `submission_approved=false`. The adapter
exports them but never marks them solved. After each completed Author turn,
the workflow also checks the standalone export form with inline references and
a compact notice placed after `\maketitle`, if present, rather than on a
separate pre-title page. Simple leading grouped titles are supported too;
uncertain conditional/group layouts use the document start so the notice is
not hidden inside an inactive branch. Missing active document markers are
reported to the editor as repairable structure errors. A successful check
atomically checkpoints those exact bytes in the publication manifest without
changing the research manuscript.
Measured final-format feedback goes to the next Author. Later invalid drafts,
failed rewrites/repairs and resumes cannot erase that last exportable partial.
If no valid checkpoint or cleanup result exists, the adapter abstains.
An interim partial does not disable the adapter's bounded checkpoint retry
after a crash; only an exportable approved submission does. Existing budget,
deadline, parking and non-retryable-error guards still apply.

API editors have web search but no compiler; their prompts explicitly delegate
compilation to the harness. They receive measured page counts and a target one
page below the hard cap (at least one page), covering the title, references and
notice. The target is headroom, not a stricter acceptance limit. A mathematical
`UNABLE` still blocks the edit; inability to execute `pdflatex` does not require
such a declaration.

Recoverable research failures get at most `max_recovery_attempts=2` checkpoint
retries while research time and money remain. Failed editorial cleanup of an
accepted draft returns to research with its technical findings, without
claiming a mathematical flaw, at most `max_cleanup_handoffs=4` times; these do
not use research recoveries. When those handoffs, the research time or the
money run out, or the failure cannot be retried, the retained critic-accepted
baseline is submitted unpolished as an approved solution, if it still compiles
within the page limit before the run deadline. Its exact text and accepting
review are saved in `batch3-accepted-baseline.json`, bound to the problem and
exported content hash, separately from the evolving research draft and output
manifest. The record also stores separate hashes of the reviewed `answer.tex`
and `references.bib` from the accepted round's snapshot, before final
compilation, normalization, and bibliography embedding can change its bytes.
Editorial handoffs, failed restoration of rewrite-only errors, and later
research episodes without a new acceptance do not discard it. Explicit resumes
retain it too; a fresh run does not inherit it. New research acceptance replaces
the retained baseline. A cleanup `research` verdict (a baseline flaw, uncertain
origin, or a need for new mathematics) durably revokes acceptance; the partial
path then applies unless fresh research earns acceptance. A later research
critic's valid rejection of the exact retained manuscript also revokes it;
rejection of a different draft does not. The workflow scans all snapshots after
the accepted round, checking both stateful and forced-fresh verdicts against
the reviewed source-and-bibliography hashes or exported-content hash. This
happens after research even on error, timeout, or cancellation, on resume, and
before publishing a retained fallback. It is not limited to the final draft or
the last stateful verdict.
On normal research completion, the exported manuscript is also compared against
the retained export for that round's verdicts, catching source differences that
disappear during export. Error and last-gasp drafts are excluded from this
additional check because they may contain unreviewed edits. A changed external
bibliography alone no longer makes a rejection apply to the older references.
Revocation records the rejecting round, so a genuinely later acceptance can
establish a new baseline. Legacy records recover missing source/bibliography
hashes from their accepted round, never the live workspace; if the source is
missing, unreadable, or inconsistent with its saved hash, they are not eligible
as automatic fallbacks. Missing or malformed verdicts are not mathematical
rejections. Neither retention nor publication approves the failed rewrite or a
subsequent unreviewed draft.
A non-retryable failure still reports its error for operator action.
Exhausted research money or exhausted recovery attempts can trigger the partial
path earlier. A
mathematical flaw found after the cutoff also takes this path instead of
starting more research. External cancellation does not start new calls.
Programming errors (including type/attribute/signature errors) are not automatic
recovery candidates. Their classification survives the AC salvage boundary as
`error_retryable=false`. The affected problem stops without further paid research
or cleanup, preserves a compiling labelled draft when possible, and records
`batch3.recovery_blocked`. The outer adapter respects that flag too. An explicit
operator resume after a code fix remains possible within the retained limits.
`run_workflow.py` exits with code 1 for error-bearing outputs, so the resume
controller reports the failure even when a draft was preserved. Successful
outputs, including non-error partial results, exit 0; subscription parking exits 2.
Publication writes a content-addressed manuscript, then atomically replaces
`batch3-output.json` to select it (`publication_version=1`). Prior revisions
remain intact if publication is interrupted. The adapter prefers this checkpoint
over terminal metadata from an earlier run segment. Legacy checkpoints using
`partials/<problem_id>.tex` or `submissions/<problem_id>.tex` remain readable.

David's generic `writeup_loop_chain4` and other writing presets retain their
orchestration. They also load the shared updated writing guidance, so their
prompts and cache keys have changed. Only Batch 3 uses the routing policy above.

## Budgets and deadlines

Competition defaults: **10 simultaneous problems**, **$1,050 per problem**
($10,500 planned total for ten problems), 23 hours of research and a
**23h55m soft batch deadline** inside the 24-hour competition envelope.
These are configurable defaults, not authorization to run.
The per-problem allowance includes Author/Critic, council, Compute and all
editorial work; cleanup is not a separate additional budget.

- `partial_cleanup_reserve_fraction=0.15` reserves 15% of the initial dollar
  allowance for partial cleanup. Research and candidate cleanup share the
  other 85%: $157.50 reserved and $892.50 shared at the default $1,050 budget.
  Every phase also charges the same outer run tracker.
- `max_partial_cleanup_repairs=2` bounds additional calls after the partial
  rewrite, including at most one continuation; zero retains one-call behavior.
  The two-hour window gives the initial rewrite 90 minutes, subsequent
  finishing/repairs 25 minutes, and final compilation/export five minutes.
  A continuation normally gets 15 of those 25 minutes, retaining 10 minutes
  for another repair. It requires at least `min_partial_continuation_seconds=300`
  seconds of its own and a subsequent repair slot; otherwise the saved fallback
  is retained. Short tests must explicitly lower that minimum.
  Short runs scale these windows down. Money retains a 75/15/10 allocation
  for rewrite and two repairs; a continuation uses its slot without consuming
  the final repair's time or allocation. A resumed mechanical repair remains
  mechanical-only. Unspent allowances carry forward, without
  extending the run deadline or the per-problem/cleanup-episode spending caps.
- `cleanup_page_headroom=1` sets the editorial page target below the hard cap.
- Candidate cleanup first publishes a checked, labelled baseline and leaves
  the larger of `partial_cleanup_seconds` and an hour (10% of remaining time
  for shorter runs) for partial cleanup.
  Bibliographies are embedded before the fallback is checked. Timeout or
  interruption cannot erase an already published valid fallback.
- `n_rounds=50` is a research continuation chunk, not a new run or budget.
  Unresolved research extends its checkpoint until cutoff, budget exhaustion,
  or the Author/Critic implementation's 500-round ceiling.
- `max_cleanup_repairs=8` prevents an endless editorial loop. Reaching it without
  acceptance returns to research if cleanup handoffs remain; otherwise it
  submits the accepted baseline; the rejected rewrite is never approved.
- `research_seconds=82800` is the standalone relative research cutoff.
  The adapter also passes absolute shared research and run deadlines.
  The two-hour reservation deliberately trades roughly an additional hour of
  research for completing partial manuscripts. It is not extra runtime: the
  earlier of the research cap and deadline-minus-reserve controls the cutoff.
  The adapter reads this reservation from the selected preset, rather than
  maintaining an independent two-hour constant.
- Candidate editorial review retains the partial-cleanup window described
  above. Rejection cannot restart research past the cutoff.
- Costs are cooperative limits: an in-flight call can overshoot a cap or consume
  the planned reserve. Cancellation requests terminate API polling and stop
  compiler/Compute work. Client timeouts are the smaller of their configured
  caps and remaining phase time, never an extension of the configured caps.
  A completed accepting review is retained if its charge exhausts the USD cap;
  publication is deterministic and makes no additional model call.
  A completed partial rewrite is also retained on USD exhaustion, subject to
  the usual parsing, compilation and page checks, without another paid call.
  Cancellation is best effort, not a promise to erase provider charges; missing
  usage is logged as unavailable, not assumed free.

The **24 hours cover the entire batch**. Setup time counts. The adapter passes
the time remaining until its shared deadline, minus one minute for finalization.
This preset defaults to 10 parallel research problems; other presets retain
their defaults. `FIRSTPROOF_MAX_PARALLEL` overrides research concurrency.
At the research cutoff, semaphore slots are released so partial cleanups are
not serialized behind ongoing research or candidate review. Queued problems do
not reset either clock; a problem without a saved attempt must abstain.

This composite runs once per problem, without the adapter's legacy staged
continuation. If its subprocess exits nonzero before an approved submission is
available to export, the adapter attempts **one** `--restart-from` retry, only
with a valid matching schedule, recoverable manuscript or helper-research
checkpoint, and time remaining. A schedule alone is not resumable research.
Subscription parking, a
reported exhausted budget and external cancellation do not trigger this retry.
A second failure is final; an interim partial is preserved during the retry and
can still be exported if recovery fails. Approved submissions take the normal
export path without a retry. This does not recover from termination of the adapter or
container itself, and cannot reconstruct provider usage lost in a crash.

Explicit `--restart-from` resumes require `batch3-schedule.json`
from this implementation, with a matching problem. Resume retains the original
absolute deadlines and dollar ceiling and restores recorded spending; it does
not grant a new 24 hours or a new budget. Older runs without this schedule cannot
be resumed safely. A retryable crash during the first Author call may use the
adapter's one crash retry even before a manuscript or helper report was saved;
the valid schedule still bounds its time and cumulative spending. Research resumes its checkpoint; interrupted editorial
work can start a new cleanup episode. Internal handoffs do not re-charge calls
already accounted for. Checkpoints, editor replies,
candidate versions, reviewer conversations, phase events and hashes are kept.

### Stopping and retrieving artifacts

Send SIGTERM or SIGINT to the adapter, not just an individual workflow child.
The adapter first closes admission/retry, then cancels the child process groups
and allows provider/worker cleanup. Signal exit codes 130/143 and -SIGINT/-SIGTERM
are never automatic retry candidates. An operator stop does not launch a
cleanup model call or a fresh submission compile. It exports the committed
accepted or partial publication if its hash, compile metadata and page limit
validate; otherwise the last adapter export (possibly an abstention stub)
remains. Raw research is preserved for explicit recovery, and partials never
become accepted solutions just because of a stop. The internal batch deadline
uses the same publication-only export policy. Final submission validation shares
one timeout across both LaTeX passes and page counting, bounded by the batch
deadline; cancellation kills the active compiler/page-counter process group.
Per-problem token and cost totals remain in the summary after completion,
including for partial and failed problems. Startup authentication and
health checks are also cancelled on operator stop. Summary rows show
`operator_stopped` or `operator_stopped_with_solution`, independently of
deadline exhaustion. The adapter returns 143 on operator stop.

`run_summary.json`, `solutions.json` and `token_usage.jsonl` refresh every
60 seconds as well as at normal stage boundaries. The summary includes an
update time, last activity, finished/completed helpers, timeouts and degraded
artifact handoffs. Helper/wave cost summaries are not counted again on top of
per-model usage. Usage and health share an incremental event-log cursor;
incomplete trailing records are retried and rotation/truncation resets it.
Provider receipts reconcile missing event charges without
double counting; unavailable provider usage remains marked as such.

The normal adapter exit path, including graceful interruption, sanitizes the
output tree and makes permitted artifacts readable by the host collector.
SIGKILL or host loss cannot execute that finalizer. After all writers have
stopped, an operator with access to the original output tree can run:

```bash
python scripts/export_firstproof_outputs.py /data/output
```

This applies the same secret/symlink filtering and collector permissions,
without model calls or changes to research results. Inspect any warnings;
it does not reconstruct files that were never saved, terminate a remote
provider response, or grant a new run budget/deadline. Never run it against
an active output tree.

## Testing and deployment

The competition submission's root `hardware.json` requests `r7i.2xlarge`
(8 vCPU, 64 GB RAM), a 250 GB gp3 root disk and a 24-hour timeout. Use the
official AWS harness and fresh Docker build for a competition-faithful rehearsal;
development runs on other machines do not establish safety on this 64 GB host.
Ten problem pipelines remain parallel, but local Compute work shares the EC2
machine's resources. Both `firstproof_batch3` and `firstproof_batch3_multiauthor`
start with at most **four Compute workers across the batch**, using a shared
registry outside the worker workspaces. Four is a conservative configurable
starting point for the 64 GB / 8 vCPU deployment, not a measured optimum. Extra Compute
requests wait for a slot; Author, critic, and council concurrency is unchanged.
Queue waiting has a separate bound equal to the effective worker hard timeout
(normally 9,000 seconds). The execution clock starts at launch, not queue entry;
both stages remain capped by the run's absolute deadline. After admission, the
soft wrap-up timeout is shortened if the remaining run time requires it.
Each worker retains its 8 GiB allowance. Admission also preserves 16 GiB of RAM
headroom and may therefore run fewer than four workers under memory pressure.
The four worker allowances total 32 GiB; admission separately reserves 16 GiB
of available memory for other work and allocation bursts. It uses measured host
availability and visible cgroup headroom, not the advertised 64 GB capacity.
Actual concurrency depends on other memory use. Each worker
samples once per second, grouping all registered workers in one process-table
pass off the workflow event loop. Admission scans also run off the event loop.
This is a cooperative, polling-based guard, not a kernel-enforced aggregate
memory quota. Offline concurrency tests do not establish peak-memory
safety or provider capacity under ten simultaneous research workloads.

`FIRSTPROOF_COMPUTE_MAX_PARALLEL_WORKERS` optionally overrides the selected
preset's positive Compute cap in the competition adapter. It is independent of
`FIRSTPROOF_MAX_PARALLEL`, which controls problem pipelines, and
`author_parallelism` / `FIRSTPROOF_AUTHOR_PARALLELISM`, which controls remote
Author helpers per problem. Increasing Author parallelism does not increase
local Compute capacity or the fixed 64 GB hardware target.
Invalid or zero Compute environment overrides produce a warning and retain
the preset's cap. The direct CLI accepts `--input compute_max_parallel_workers=N`;
Batch 3 rejects zero because zero disables both admission and aggregate-memory
monitoring in the generic workflow. Changing the positive cap does not change
the 8 GiB worker allowance or 16 GiB reserve.

All local runs sharing resources must resolve `compute_memory_registry_dir` to
the same host-local directory (default: `.proofcouncil-compute-memory` under the
outputs root). Its `control.json` retains the worker-count, memory and reserve
policy, and rejects a different policy even after workers exit. Before changing
that policy in an existing output tree, stop all workers and surviving
descendants that used it, then use a fresh registry shared by every new run.
Do not split overlapping runs across registries to bypass the mismatch.

Memory registry format 2 separates read-only, inherited `slot-N.lock` handles
from atomically replaced `slot-N.json` process markers. This also applies to
the shared cleanup-editor and reviewer registries. A worker's inherited handle
cannot write the accounting metadata; its lifetime lock and orphan-process
tracking still prevent premature slot reuse. An unreadable slot stays occupied
and reserves its full allowance for admission, even if unlocked, since its
descendants may still be alive. Each owner repairs only its own marker under
the control lock; peers do not guess its identity or erase the slot. Memory
samples report `unreadable_slots` and cumulative `marker_repairs`. Admission and
running-worker monitors emit `cli.memory_registry_quarantine` once per newly
observed slot per worker, including the registry path and reserved allowance;
this also diagnoses a fully quarantined pool whose workers never start. Global RAM
pressure still stops a known worker, and unreadable control metadata or
incompatible registry versions fail closed. Older registries are not migrated
automatically: stop all their workers and surviving descendants
before selecting a fresh shared registry for the upgraded code. Do not upgrade
overlapping workers in place. This remains a polling safety guard, not a
filesystem security boundary against a same-user process.

Lock-open and lock-operation failures also quarantine the affected slot for
peers. An owner whose own slot is unreadable raises an accounting error instead
of treating its reserved allowance as measured usage; its watchdog stops the
worker if the error persists through the accounting-gap grace period. Healthy
peers continue sampling with that slot's full allowance reserved. Owners
check the held lock's device/inode before repairing their marker, so deleting or
replacing a live lock is detected rather than overwriting another owner's marker.
Fresh unused markers are published before their lock files to avoid quarantining
a never-used slot after an initialization crash. These checks do not make it safe
to remove an active registry. The existing pressure-victim policy can still elect
a known orphan whose controller no longer samples; reconcile surviving descendants
before resetting registries or launching recovery.

For an explicitly authorized single-problem test, use
`scripts/run_workflow.py --workflow firstproof_batch3` with the problem path/ID
and an approved `--budget-usd`. `--input max_wallclock_s=...` sets total duration.
For a short cutoff test also set `--input research_seconds=...` below that
duration, leaving time for partial cleanup. Use `--restart-from` only to resume
an existing run with its original schedule and remaining budget.

For container tests, configure approved budgets and tested parallelism:

```text
FIRSTPROOF_WORKFLOW=firstproof_batch3
FIRSTPROOF_BUDGET_USD_PER_QUESTION=<approved per-problem budget>
FIRSTPROOF_MAX_PARALLEL=<tested research parallelism>
FIRSTPROOF_AUTHOR_PARALLELISM=<positive per-problem Author parallelism; baseline default 1>
FIRSTPROOF_COMPUTE_MAX_PARALLEL_WORKERS=<tested positive local Compute cap; preset default 4>
```

This selects 16 pages, 50-round research chunks and one pipeline per problem.
Incompatible `FIRSTPROOF_ADAPTIVE_CONTINUATION=true` is disabled with a warning;
out-of-range page/round limits are corrected with warnings rather than crashing
before writing harness output. The preset's workflow class selects Batch 3
handling even if its YAML file is renamed. The standalone adapter
needs the workflow environment variable; the submission image sets it already.

Compute defaults are 6 GiB soft / 8 GiB hard per workspace, an 8 GiB cooperative
admission reservation and a shared 10 GiB free-space floor. Reservations are not
filesystem quotas and do not cover all artifacts. Set a common
`compute_filesystem_reservation_dir` when independent output roots share a disk.
Load-test the actual disk and concurrency before submission.

Before competition deployment:
- Approve actual dollar budgets, model availability and hardware/parallelism.
- Run the official container input/output and deadline smoke tests, then an
  explicitly budgeted end-to-end proof test on Ada.
- Confirm approval of the updated writing guidance and review a live manuscript
  smoke test. Ten concurrent Pro rewrites may not all finish in the final window;
  published labelled fallbacks protect output, not completion of every rewrite.
- In that explicitly budgeted smoke test, exercise `restore`: introduce a known
  rewrite-only regression, check that the critic identifies the correct baseline
  passage, and verify targeted restoration and a fresh review before acceptance.
  Also check that a baseline mathematical flaw routes back to research instead.
- Exercise the per-round export feedback in a subsequent Author turn and an
  oversized or non-compiling partial rewrite through `partial-repair.txt`.
  Check prompt clarity, measured page feedback, actual rewrite duration and
  fallback export after an interrupted cleanup. Offline tests cover routing and
  prompt contents, not live model behavior or provider latency.
- Finish the interrupted-provider billing audit and competition progress heartbeat.
- Confirm that explicitly labelled partial results are an acceptable answer
  format; they are exported without a solved claim.

No paid API call, Ada job, Docker launch or submission is performed by preparing
this workflow.
