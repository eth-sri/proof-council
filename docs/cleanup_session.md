# Persistent cleanup editor

`CleanupSession` implements the editorial process from the supplied `prompt.txt`:
a persistent Claude Code session edits the manuscript autonomously, requests
independent feedback when useful, compiles, and retains `feedback.md`. It uses
the repository's v3-derived `WRITING_GUIDANCE.md`. The lead is Fable 5.1 at
`xhigh`; native reviewers use the same model. The managed Codex reviewer uses
GPT-6 Astra at `xhigh`, **not Pro**. The ambiguous model name in the source
prompt is replaced with these explicit configurable choices.

CleanupSession is the default write-up node. `WriteupLoop` and `proof_cleanup` are
fallbacks; see "Write-up / cleanup nodes" in `configs/workflows/instructions.md`.

## Configuration

The workflow input default and both production presets, `firstproof_batch3` and
`firstproof_batch3_multiauthor`, select `cleanup_backend: claude_code`. Override the run
input with `cleanup_backend: api` to use the previous API editor instead.
The default configuration is equivalent to:

```yaml
inputs:
  cleanup_backend: claude_code
components:
  CleanupSession:
    cleanup:
      model: claude-fable-5-1
      effort: xhigh
      cost_config: models/anthropic/fable_51
      codex_model: gpt-6-astra
      codex_effort: xhigh
      codex_cost_config: models/openai/gpt-6-astra
      codex_budget_fraction: 0.3
      linear_read_model: models/anthropic/sonnet_5
      linear_read_budget_fraction: 0.1
      max_invocations: 3
      max_invocation_usd: 150
      max_episode_usd: 500
      memory_gb: 1
      memory_reserve_gb: 4
      max_parallel_editors: 10
      max_parallel_codex_reviews: 10
      workspace_bytes: 2147483648
```

`cleanup_session.yaml` is a standalone test preset with planning limits of
$150 and two hours. Supply `problem`, the manuscript text as `document`,
`partial`, and `page_limit`. It does **not** independently certify correctness
or publish a competition answer. Defaults and example budgets are not spending
authorization.

## Automatic workflow

- A solved candidate starts one editing episode. The existing stateful critic
  then accepts it, requests targeted edits in the **same Claude session**, asks
  for targeted restoration from the baseline, or returns it to research.
- Partial cleanup at the research cutoff uses its own episode. It preserves
  gaps and limitations; no mathematical critic follows. Subsequent mechanical
  page/compilation repairs reuse that session.
- Exact-byte publication checks, page limits, partial labels, and previously
  saved fallback manuscripts remain owned by the Batch 3 workflow. A CLI
  failure does not become acceptance or replace the fallback with unchecked text.
- The exact research-accepted baseline remains a fallback across editorial
  handoffs, subsequent unaccepted research drafts, and explicit resumes. A
  failed restoration of a rewrite-only error does not revoke it. A substantive
  cleanup rejection of the baseline does; fresh research acceptance is then
  required. Subsequent round snapshots are checked for both stateful and
  forced-fresh rejections, including when research errors or is interrupted.
  Source matching includes the reviewed bibliography, so changed references
  cannot revoke an older baseline solely because its TeX is unchanged. On
  normal research completion, an identical exported manuscript also matches
  that round's rejection even when its source differs trivially. Final
  publication still checks compilation and the page limit.
- If Claude stops without `completion.json`, or writes one that is not a JSON
  object with `status` `ready`/`unable`, it receives a continue prompt in
  the same session, at most `max_invocations` times per requested revision.
- An editor that stops early (non-zero exit, its own budget stop, a deadline,
  cancellation, the live monitor or a memory kill) raises `CleanupIncomplete`,
  a retryable `RuntimeError` whose `output` carries the last `answer.tex` with
  `status: incomplete`. It is never returned as a finished revision; Batch 3
  currently keeps its fallback and does not publish it.
  The existing repair-loop limit still bounds external critic revisions.

### External critic context and failures

Each cleanup review sends the complete original problem, one pre-rewrite
baseline, one current manuscript, mechanical checks, and the editor's latest
feedback. The critic's earlier reports remain verbatim and chronological,
including research-stage reports and every unresolved objection. Obsolete
user packets containing earlier full drafts are removed, including when loading
legacy saved conversations. The critic must re-check earlier findings against
the current draft; neither previous acceptance nor the editor's claims certify
the repair. Reports are not automatically summarized or truncated.

Before a paid review, `cleanup.context.preflight` records a conservative text
token upper bound (UTF-8 bytes plus message overhead), not a provider token
measurement. The default [Astra window](https://developers.openai.com/api/docs/models/gpt-6-astra)
is 1,050,000 tokens, reserving the model
configuration's output allowance (normally 128,000) plus 64,000 tokens for tools
and framing. This bound can reject inputs that an exact tokenizer would admit.
`Batch3CleanupCritic` component settings `context_window_tokens` and
`context_tool_reserve_tokens` override these values; when selecting a different
model, set its verified window explicitly. Hosted-tool growth is not strictly
bounded by this initial preflight.

If the complete packet still does not fit, no review is sent and no mathematical
content is silently removed. A provider `context_length_exceeded` failure is
also terminal for the unchanged request: no retry and no Pro-to-standard
fallback. The Batch 3 workflow preserves saved manuscripts and stops for
operator action instead of restarting research or another paid rewrite. A
subsequent explicitly requested resume can use a corrected packet/configuration.

Responses API attempt failures are persisted immediately in provider receipts
and emitted as `model.attempt.failed` while the call is still in progress,
with the error and retry/stop decision. These diagnostic events are not charges;
normal usage accounting remains authoritative, and absent provider usage is
still marked unavailable rather than assumed to be free.

The Batch 3 adapter checks the selected cleanup model/rate configuration,
required keys, and CLI versions/options **at startup, before paid research**,
even when `FIRSTPROOF_HEALTHCHECK=off`. A missing key or incompatible/missing
CLI selects `cleanup_backend=api` for all problems and retries in that launch.
The adapter emits a warning and saves `cleanup-healthcheck.json` plus
`cleanup_disabled_reason` in `run_summary.json`. Invalid pricing/configuration
and report-write failures remain fatal; they are not silently downgraded.
`FIRSTPROOF_HEALTHCHECK=strict` also makes runtime/key failures fatal.
An explicitly selected API backend skips the CLI check. Explicit batch resumes
run the same check before modifying any checkpoint, but fail on unavailable CLI
cleanup rather than silently changing the reviewed resume recipe.

## Tools and isolation

Claude uses managed file read/write/edit tools rather than arbitrary shell or
filesystem access. Only `answer.tex`, `feedback.md`, and `completion.json` are
editable. Native reviewers have read-only tools. Codex receives the current
manuscript, baseline, problem, findings, constraints and feedback as a textual
snapshot; its shell and delegation tools are disabled. Codex calls are
serialized per editor and owned by the editorial invocation, not an HTTP
request. `mcp__cleanup__review` returns a durable review ID immediately;
`mcp__cleanup__review_status` lists jobs or waits at most 240 seconds for one.
A client timeout does not cancel the worker or authorize a duplicate call.
Identical requests reuse running/completed work; settled failed or cancelled
jobs may be retried subject to the existing budget and accounting guards.
New requests first return any
running or completed-but-unretrieved review. The full report is retained in
`reviews/`; status returns it in chunks using the normal managed reader.
Retrieval does not require remaining inference budget or charge again.

Job records under the private `review_jobs/` directory retain task, input
hashes, state, completed cost, report location and retrieval progress across
editor invocations. An interrupted supervisor's running jobs are not silently
restarted. Existing session accounting guards still require reconciliation
before resuming interrupted work. Normal invocation shutdown cancels and joins
all its outstanding jobs before settling costs. Codex cancellation is requested
before waiting for HTTP helpers such as the linear reader to drain, so Codex
does not keep spending during that wait.

The required first review uses `purpose: attribution`. The editor must retrieve
the complete attribution report (including remaining chunks) before rewriting;
the harness refuses a `ready` completion without a completed, retrieved
attribution review. On targeted repairs it can reuse that episode's report.
This requirement applies to both solved and partial drafts. Codex web search
remains enabled for citation verification; search queries may be sent to
external search services, so these runs are not an offline confidentiality mode.
Missing/failed reviews must be disclosed, not treated as successful checks.
Setting the Codex budget fraction to zero explicitly disables this requirement.
Native correctness/exposition reviews remain discretionary, and the external
stateful critic still decides mathematical acceptance. This external critic
defaults to Astra **Pro/max** (distinct from the Codex attribution reviewer).
Its context-preflight event records the requested model, reasoning mode and
effort; provider receipts record the actual settings, including any fallback.

Each targeted repair receives the current critic findings inline in the new
editor prompt as well as in the refreshed `findings.md`. It must reread updated
inputs even when resuming a Claude session. An unchanged repair is stopped
before another paid critic call and follows the existing editorial-failure
handoff/fallback policy; it is not treated as a newly accepted revision.

`mcp__cleanup__linear_read` addresses define-before-use directly: a cheap API
model (Sonnet 5 by default, `linear_read_model`) reads the current `answer.tex`
strictly from the start, one passage at a time, seeing only the preceding text,
and reports each symbol, term or result used before the manuscript introduces
it. Passages are queried in batches of at most four through `APIClient`, checking
the remaining time and budget before admitting each batch. It draws on its
own share of the invocation allowance (`linear_read_budget_fraction`, default
10%, taken from the editor's share); the report is returned and saved under
`reviews/linear-*.md`. Set `linear_read_model: ""` or a zero reader fraction to
remove the tool. Every batch requires a positive remaining reader allowance.
Passages have unique query indices across all batches, so provider receipts
are charged once rather than recounting earlier batches.
Responses are charged incrementally to the normal `model.call` ledger, with
durable provider receipts for failed or interrupted calls. Shutdown requests API
cancellation and joins the reader before settling the editorial invocation;
unresolved provider usage blocks reuse rather than silently disappearing.

Claude uses the native `Task` tool (`Agent` is a CLI alias), with empty user/project
setting sources, disabled hooks and skills, and a strict explicit MCP configuration.
It does not use `--bare`, which suppresses native subagents. Each Codex invocation
gets an existing, private `CODEX_HOME`, with no copied subscription credentials.
Its child environment maps the admitted `OPENAI_API_KEY` to the noninteractive
CLI's `CODEX_API_KEY`; the key is not stored in workflow configuration or an
auth file. Native Claude reviewers do not receive either OpenAI variable.

Compilation uses the existing deterministic compiler in a temporary directory,
with shell escape disabled, a minimal environment without provider keys, and
restricted TeX file access. The final publication gate remains separate.
The MCP server binds only to loopback with an invocation-scoped bearer token;
the transient connection configuration is removed after child shutdown.

This runs inside the trusted competition container using the existing
subprocess lifecycle, not nested Docker. Each CLI has resident-memory and
workspace guards. Cross-process admission defaults to two editors and two
Codex reviewers, in separate pools to avoid nested-slot deadlock, with a 16 GiB
host emergency floor. These are polling/admission guards, not kernel quotas.
The compiler retains its existing per-pass timeout and cancellation mechanism.

## Accounting and recovery

Both provider keys are required unless the Codex fraction is zero. Host
subscription credentials/settings are not inherited. Claude Code >=2.1.251
is required by Fable 5.1 and includes native-subagent budget enforcement; the submission image
pins that version. Local installations must be upgraded separately.

The image installs Claude's native Linux binary directly, with pinned SHA-256
checksums for x86-64 and ARM64, and disables auto-updates. It does not install
the Claude npm package or depend on Node for Claude. Codex retains its pinned
npm installation. Image builds check both CLI versions and every cleanup option;
cleanup admission repeats the compatibility checks before any paid invocation.
Version/help probes use a temporary home without API keys or user settings.

Each invocation receives the remaining shared deadline. Its allowance is the
minimum of the remaining enclosing budget, `max_invocation_usd` (default $150),
and the remaining `max_episode_usd` (default $500). An episode is the initial
rewrite plus its continuations and targeted repairs. Recorded episode spend is
persisted in `session.json`, including billed failures and joined Codex reviews;
recreating the agent or restarting the workflow does not reset this ledger.
These allocations cover Claude, native reviewers, and managed Codex reviews. The
external stateful Astra Pro critic is charged separately to the enclosing phase and
problem budgets, not this editor ledger.
The episode cap does not raise an enclosing budget or release the Batch 3
partial-cleanup reserve. The standalone preset still has a $400 overall default;
using the full $500 episode allowance requires a sufficiently large enclosing
budget.

After the retry holdback described below, 30% of the allocated funds go to managed Codex reviews, 10% to the linear reader and 60% to Claude plus
native reviewers. Claude's native dollar cap covers its native subagents but
not external Codex. All recorded costs also propagate to the enclosing phase
and problem budgets. A continuation gets a newly calculated remaining cap,
not the original allowance. Unused helper allowance is not billed.

The initial invocation cap is not a rigid reviewer sub-budget. After a failed
mandatory attribution review, if at most 25% of its initial Codex allocation
remains, one top-up of at most `max_review_topup_usd=75` may increase that
invocation and reviewer allowance. This borrows only unallocated enclosing
phase/problem funds, within the unchanged $500 episode ceiling. It reserves
all money already promised to the active editor and linear reader, including
unreported usage. General reviews, successful attribution reviews, and
unsettled helpers cannot trigger a top-up. A denied retry retains the fallback.

Managed tool results include `cleanup_control` with the remaining time and
current stage. In the last `finishing_seconds=900` seconds (20% for short
invocations), the editor must stop optional polishing and native reviewer
tasks and finish the outstanding required checks. The harness rejects new
optional Codex reviews in this stage; existing reports remain retrievable.
The finishing deadline and stage are persisted per session; later invocations
cannot return to editing or reopen optional reviews. Every invocation starting
in finishing disables native reviewer tools entirely. Its prompt states that
finishing is already active and distinguishes an unchanged saved draft from a
manuscript replaced by the harness. Mechanical-repair constraints still apply;
finishing does not authorize a substantive rewrite. When a mandatory attribution report is already
completed and retrieved, its new-invocation budget share goes to the editor
instead of being reserved a second time; the linear reader keeps its share.
No new Codex reviews are then admitted, but paid reports remain readable, and
the prompt no longer advertises a review top-up for that zero-budget pool.
Missing attribution still keeps its review allowance and completion gate.
This reallocates remaining funds only: it does not refund earlier timeout
estimates or enlarge any invocation, episode, or problem cap.
An invocation initially holds back `review_retry_reserve_fraction=0.2` of its
allowance (at most the $75 top-up cap) before dividing the rest among workers.
This gives a failed mandatory attribution review retry headroom even when the
parent contains only the reserved cleanup funds. The holdback is part of the
existing allowance, not extra spending. Once attribution is fulfilled, a
finishing continuation releases this holdback to the editor too.
A successful compile does not waive mandatory
attribution review or turn an unreviewed partial into an accepted solution.

Available live usage is monitored; usage is charged once after exit, including
interrupted streams when measurable and tool/review shutdown. Settlement is joined under cancellation shielding before
the deadline cancellation propagates; an actually unresolved worker still
retains the fail-closed in-flight marker. This does not make SIGKILL recoverable.
Final Claude USD is used when its pricing is known.
For explicit `modelUsage.costBasis=unknown`, the CLI's fallback-priced portion
is replaced using configured rates and per-model tokens (including native
reviewers), while preserving the original reported USD in the event. This is
marked estimated; an unknown model without matching rates fails closed.
Interrupted usage is also marked estimated. In-flight calls and delayed provider usage can exceed
recorded caps, which are **not invoice guarantees**. In particular, Codex CLI
normally reports usage only at `turn.completed`: a single unfinished review can
overshoot its allowance before its cost becomes observable. Admission reserves
its recorded allowance, a completed over-budget review stops further reviews,
and the lead's live check includes the helper charges once reported. Logs/events keep the known
charges even when collection fails.
A managed Codex review with confirmed shutdown but unresolved usage (for example, one cancelled at MCP shutdown
after the editor exits) is instead charged its full admitted allowance to the helper pool, less anything already
recorded. This estimate is persisted as a `model.call` with `cost_estimated` and `usage_unavailable`, so resume
accounting and usage exports retain it, and reported as `cleanup.helper_usage_estimated`. An unconfirmed helper
shutdown or failure to persist the estimate still locks the session and prevents further paid invocations.
Likewise, when the harness stops the editor (deadline, cancellation, live
monitor, memory watchdog) Claude writes no terminal result. Once its process
group is confirmed stopped and helpers are joined, the editor is charged its
full `--max-budget-usd` lead allowance, less anything already recorded; this is
persisted as a `model.call` like the Codex estimate, reported as
`cleanup.editor_usage_estimated`, added to `recorded_usd` and to
`estimated_usd` in `session.json`, and the session stays usable. It is resumed
only if Claude saved a transcript. Claude checks its cap between turns, so it
can overshoot by part of one turn; that overshoot is not in the estimate.
`cleanup.editor_usage_estimated` and `session.json.editor_usage_estimates`
separately record the streamed debit, the saved per-invocation usage when
available (`saved_usage_usd`), and the remaining unreported allowance held
conservatively (`unreported_allowance_usd`). These diagnostic fields are not
additional charges. A saved cumulative total alone cannot establish the cost
of an interrupted provider call or native reviewer, so it does not authorize
refunding that allowance. No arbitrary one-turn dollar margin is used.
On `--resume`, Claude's `total_cost_usd` and `modelUsage` are cumulative over
the session, so `session.json` keeps the last cumulative total
(`claude_session_usd`, `claude_session_tokens`) and each invocation, and the
live monitor, is charged only the excess over it. After a clean result that
total becomes the baseline. After a harness stop, Claude writes its cumulative
total to the transcript on SIGTERM (and the next resume starts from it), so the
estimate is raised to that saved spend if higher and the baseline moves to it;
after SIGKILL nothing is saved, resume restores the old total, and the baseline
stays. A total below the baseline is charged in full. A live check that trips
(deadline, budget, unpriced usage) stops the worker rather than raising, so the
draft is kept.
Missing usage still fails closed while the worker may be running.
Claude's own cap still uses its internal pricing table. The external monitor
uses configured rates for available streamed usage, but native-reviewer costs
may not become observable until the terminal result. It stops the lead at
1.2x its allowance (or at the shared pool's remaining amount, if lower), so
Claude's own clean stop normally comes first. Unknown-price estimates
must not be described as invoice-confirmed charges.
An already recorded overrun does not discard the completed manuscript; the
normal publication/critic gates still apply, and no further over-budget calls
are admitted. A definitive pre-launch failure (for example, a missing CLI
executable, an admission error, or cancellation while queued for a slot) does
not require cost reconciliation, and a Codex review that never launched is not
charged.
Likewise, an unsuccessful exit or exception does not lock accounting when the
editor's final usage has been billed, its processes have stopped, and all managed
helpers have been joined and accounted for. The workflow can use its ordinary
failure recovery while retaining the previous fallback.

`cleanup_sessions/<session_key>/` holds supervisor metadata and the preserved
baseline; its `workspace/` holds the manuscript, context, reviews, Claude session
history, and feedback. Solved-candidate keys are derived from the baseline hash,
not a process-local episode counter: a new baseline after restart cannot collide
with an earlier episode, while repairs to the same baseline reuse its session.
Resume uses an explicit session UUID and checks the
problem/baseline/settings fingerprint. Reusing a key with different inputs is
an error. Supervisor crashes or unconfirmed worker stops leave an `in_flight` marker: inspect
workers and reconcile charges before any manual recovery. The implementation
does not silently reset an uncertain session or retry potentially unbilled work.
An unresolved invocation also writes `cleanup-accounting-uncertain.json` at the
problem run root. The admission check additionally scans session markers so a
hard crash cannot bypass it. It blocks new editorial episodes, and the automatic
workflow exports its existing fallback without resuming paid research.

A worker that cannot be confirmed stopped is still an accounting
interruption. A surviving manuscript is not evidence that all charges are
known. To recover manually:

1. Confirm the editor, native reviewers, and managed Codex processes have stopped.
2. Reconcile terminal usage, retained logs/events and provider billing, including
   interrupted calls, with the problem's cumulative spend.
3. Restore the session's `recorded_usd` ledger and problem-level accounting
   without double counting, then clear `in_flight` and the root uncertainty
   marker only for fully reconciled sessions. Preserve the evidence and original
   metadata. Old sessions without a cost ledger require this reconciliation too.
4. Prepare and inspect an explicit resume recipe before authorizing more calls.

There is no automatic invoice reconciliation or unsafe "unlock and retry" path.
Changing session keys or switching to the API editor does not bypass a saved
uncertainty marker.

## Validation

Offline tests use dummy CLI processes and a real local MCP client/server. Run a
separately authorized small paid fixture after changing pinned CLI versions or
tool/subagent flags; offline tests alone cannot validate provider compatibility.

The bounded integration fixture is `scripts/cleanup_smoke.py`. Without `--live`
it only checks the installed CLIs. An authorized live test requires
`--live --budget-usd 25 --output /data/output` and a fresh output directory.
It uses a short induction proof, two native reviewers, one Codex review, and a
finishing-only targeted repair in the same Claude session, under a shared 30-minute deadline.
The report checks actual CLI tool results, retained Codex output, compilation,
session continuity and accounting. The finishing repair checks the real CLI's
startup tool list and successful MCP compilation with `--tools ""`, without
native delegation. Offline tests cover interrupted-worker
cleanup; the live fixture does not deliberately cancel a paid response.

That short fixture does **not** establish that a real 16-page manuscript fits
the partial-cleanup first-pass window (90 minutes). Before a large CLI
cleanup run, separately authorize a realistic manuscript test using the
standalone `cleanup_session` preset, a fresh output directory, the actual
problem/document, `partial: true`, `page_limit: 16`, and a 5,400-second wallclock
budget. Retain the manuscript, feedback, CLI logs, compile/page results,
elapsed time and recorded spend, including failures. No independent mathematical
acceptance follows partial cleanup. Also validate the Batch 3 finishing-only
continuation and reviewer top-up under its two-hour overall cleanup window.
The new allocation/recovery policy has offline coverage; it still requires
separately authorized realistic-size live validation.
