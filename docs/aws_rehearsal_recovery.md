# AWS recovery safeguards

This guide describes failure handling and explicitly authorized recovery of an
interrupted run. It does not authorize a restart, additional spending, or a new
clock. Examples are schematic; actual research inputs and recovery evidence
belong outside the repository.

## Failure-to-fix map

| Observed failure | Change | Offline coverage |
| --- | --- | --- |
| Codex's V8 code-mode host could not reserve virtual address space under an 8 GiB `RLIMIT_AS`. | Compute disables only the address-space limit. CPU/core limits, storage limits, RSS accounting and shared worker admission remain. A credential-free native `text(1)` protocol probe runs before paid work. | Preflight success/failure; RLIMIT isolation; shared and standalone RSS enforcement. |
| A successful model turn that executed nothing was presented as useful partial computation. | Recognized host failures without a successful finish become infrastructure errors. Failed/legacy host-crash cache entries are not reused; a new invocation must pass the preflight. Genuine scientific partial results remain reusable. | Error-vs-partial cache regression and existing finish/salvage tests. |
| Gemini exhausted hosted tools without a final answer. | Preserve finish reason and usage, then make at most one tools-disabled wrap-up using bounded retained findings. Failure or emptiness in that wrap-up is terminal for the invocation; it cannot restart the tool loop. | Exhaustion, retained findings, billing, failed wrap-up with retries configured. |
| Claude continued a truncated `server_tool_use` with no matching result. | Remove the incomplete tail before tools-disabled salvage; do not fabricate a result. HTTP 400 invalid requests are terminal. Valid `pause_turn` behavior remains. | Unpaired-tool continuation fixture; terminal-error regression. |
| A blocked Claude stream exceeded its deadline and delayed shutdown. | Bound SDK I/O independently of incoming events. Close the stream on cancellation/deadline; retain known usage and a bounded partial-text audit artifact, not a successful answer. | Blocked-stream deadline and usage/close tests. |
| Earlier successful provider segments disappeared from accounting when a later segment failed. | Append-only per-attempt receipts persist model/mode, response IDs, usage and cancellation state. Failed invocations charge known usage; resume and reporting reconcile late receipts without charging them twice. | Repeated/late receipts; failed invocation charged once. |
| Cancelled OpenAI background jobs had no durable acknowledgement, and a late response could escape local cancellation. | Record IDs at creation, request provider cancellation, persist its acknowledgement/usage, and cancel late-created responses. Polls have bounded transport/deadline handling and persist late receipts. Unknown usage stays explicitly unknown. | Cancellation receipt and abandoned-response race tests; existing poll/fallback tests. |
| An Author reports failed hosted edits while returning an unchanged manuscript. | Mark the artifact outcome explicitly, prevent `ready` on a failed unchanged edit, preserve the previous files, retain diagnostics and feed failure feedback into the next round. Downloads that fail are not treated as successful edits. Prompt guidance requires saving edits before heavy calculations and delegating bounded jobs. | Failed edit cannot become ready; existing container-file and workflow tests. |
| Live spend included only completed problems. | Aggregate all children's events and receipt adjustments. Distinguish running from pending; parse Anthropic `thinking_tokens`. | Unfinished-problem aggregate, token parsing, reconciliation. |
| SIGTERM bypassed asyncio cleanup; direct checkpoint writes could truncate good state. | SIGTERM cancels the workflow task once, allowing cleanup before the outer supervisor's kill grace. Cache and AC checkpoint publication use atomic replacement. | Actual subprocess SIGTERM; interrupted replace preserves previous checkpoint. |
| MultiAuthor rejected the new `execution_failed` keyword after a paid Author response completed. | Forward the keyword to Author's result builder, retaining readiness checks and delegation summaries. Programming errors stop automatic recovery and paid cleanup; the draft and error classification are preserved. | Full OpenAI/Anthropic MultiAuthor container paths, delegation on/off, changed/unchanged/failed downloads; serialized error and adapter retry guards. |
| A restored six-slot memory registry passed native preflight but rejected four-slot Compute admission. | Preflight now also streams a credential-free `codex --version` through real admission and verifies worker cleanup. Mismatched registries fail the gate without automatic policy migration. | Stale-policy rejection with unchanged registry; failed/timed-out/cancelled probe cleanup. |

Hosted execution outputs are requested where supported. Their debug-log copies
are bounded and redact known environment credentials before truncation. Provider
receipts contain accounting/identifiers, not prompts or tool output. Logs are
correlated to run, agent, invocation and attempt rather than `uninitialized`.

## Limits of these fixes

- The subprocess memory guard measures resident process-tree memory. It is a
  polling guard, not a kernel-enforced per-worker cgroup quota; an allocation
  spike can overshoot between samples. Do not remove the host reserve. Docker's
  separate hard container limit is unchanged. The current default is four
  Compute slots. Recovery must select this deliberately, not accidentally
  inherit a different default from a saved run.
- The native-host preflight executes no model request. A passing probe proves
  the actual runtime can start and evaluate its trivial program, not that every
  later CAS computation fits in memory. Unsupported native protocol/layout
  changes fail before spending, with an actionable setup error.
- A cheap-model smoke using direct execution does not exercise Astra's native
  V8 code-mode path. An 8 GiB address-space ceiling can prevent that host from
  starting even with one worker. Concurrent RSS pressure is a separate risk.
- Providers may continue remote work after a broken connection. A cancellation
  request without an acknowledgement is not evidence of zero subsequent cost.
  Daemonized, deadline-bounded client I/O prevents local shutdown hangs; it does
  not claim that every remote provider offers cancellation.
- Gemini's provider-internal tool budget is not controlled by the local function
  call counter. One tools-disabled recovery handles exhaustion; it does not
  eliminate the cost of the preceding tool loop or guarantee time remains.
- Failed provider-hosted calculations cannot be made reliable by merely
  replacing the container. The workload must change. Diagnostics/provenance
  prevent false progress but cannot prove the mathematics or guarantee success.

## Read-only recovery inventory

Use the verified output archive, not the incomplete file-by-file SCP copy. Keep
the original archives immutable. Inspect a separate extracted copy:

```sh
python scripts/check_batch3_recovery.py /path/to/output/workflow_runs --page-limit 40
```

This checks problem/schedule hashes, checkpoint and cache JSON readability,
approved-export hashes and formatting/page metadata, cumulative known spend,
unresolved cancellation records and the persisted deadlines. It prints JSON and
never modifies artifacts, makes model calls, changes the clock, or authorizes a
launch. JSON readability is not a promise that every cached artifact path is
portable or every cached output semantically reusable.

Classify each checkpoint before recovery (round indices are zero-based):

| Checkpoint state | Recovery treatment |
| --- | --- |
| Approved export | Preserve the approved export; no research or cleanup calls. Workflow approval is not independent mathematical validation. |
| Accepted research, interrupted editorial cleanup | Resume cleanup, not a fresh research attempt. |
| Interrupted Author invocation | Retry from the retained checkpoint. |
| Incomplete reviews | Complete missing reviews of the retained draft; reuse compatible completed seats. |
| Failed unchanged edit | Retain mathematical work and useful reviews, flag the failed edit and avoid repeating the same large calculation. Do not treat empty council or failed Compute records as successes. |

## Required before a paid restart

1. Reconcile legacy usage. New receipts cannot reconstruct information the old
   code never saved. Known omissions must be posted once to a recovery copy
   with provenance; provider-invoice reconciliation or an explicit
   conservative liability reserve is still needed for unknown charges. Preserve
   the original cumulative per-problem cap, not a new allowance per segment.
2. Keep `batch3-schedule.json` and the original absolute deadlines. An extension
   for an interrupted rehearsal requires explicit approval and a recorded old/new
   schedule; it is not competition-equivalent automatic resume behavior.
3. Restore the original container paths, preferably `/data/output`, before
   reusing path-bearing caches. Verify artifacts exist and hashes agree. Never
   reinstate old process IDs, slot ownership or reservation locks as live state.
4. Preserve the original problem inputs, review histories, and page limit.
   Passing `--page-limit 40` to the inventory does not alter the competition
   preset or authorize a page-limit change.
5. Run a network-disabled migrated replay with fail-on-unexpected-call clients:
   approved problems must make zero calls, accepted drafts must enter cleanup,
   and interrupted review joins must reuse compatible completed seats. Merely
   launching the official fresh-run harness does not perform this migration.
6. Build the intended Linux image and run the native-host preflight under its
   real child limits. Exercise the physical-memory and shared-slot guards there.
   Then, with explicit spend authorization, run a small production Astra
   code-mode smoke including an executable, artifact, finish record and Author
   handoff. A cheap model's different tool path is not equivalent coverage.

These gates deliberately separate code regression coverage from deployment and
account reconciliation. No new full rehearsal should be launched on the strength
of unit tests alone.

## Startup Compute check

The submission adapter now runs the credential-free CLI-version/native-host
probe before starting any Batch 3 problem. It uses the same sandbox builder,
backend, 8 GiB RSS policy, shared worker registry and host reserve as Compute.
After checking the native execution host, it launches `codex --version` through
the streamed-worker path, including shared-slot admission and confirmed cleanup.
The admission probe allows 30 seconds to queue and 30 seconds to execute; a busy
or mismatched registry fails the gate rather than being silently bypassed. A
restored registry is never automatically migrated by this check.
Batch 3 always runs this check, including with `FIRSTPROOF_HEALTHCHECK=off`
(the default) or `warn`. If the probe fails in either of those modes, the
adapter logs a warning and passes `enable_compute=false` to every problem,
including automatic retries. Author, critic, council and cleanup keep their
configured behavior. The failure remains in `healthcheck.json` with
`paid_calls: 0`, and `run_summary.json` records `compute_disabled_reason` and
the warning before any workflow starts. A null reason means no startup override;
the selected preset still controls whether Compute is enabled.

Explicit `FIRSTPROOF_HEALTHCHECK=strict` exits with code 2 on probe failure.
The adapter writes fallback `.tex` files and an initial `solutions.json`
before authentication setup or the check, so even a strict failure or an
interrupted startup leaves a fallback submission bundle. Invalid healthcheck
settings and failures writing the healthcheck report still abort rather than
degrading. Other workflows retain
their opt-in `warn`/`strict` behavior. The controlled resume launcher remains
fail-closed: all required checks must pass before it mutates or launches a run.
Compute also retains its per-invocation probe when enabled.

Native cleanup startup also validates the control metadata of the editor and
Codex-reviewer memory registries using the preset's exact role policies (the
reviewer pool is skipped when its budget fraction is zero). This is read-only,
does not claim slots or call models, and runs even with Compute disabled.
Missing registries are left fresh. Each role's control lock is retried every
50 ms for up to three seconds, allowing active workers' brief memory samples
to finish. A lock still busy at that deadline, or legacy, malformed, or
policy-mismatched control metadata, blocks launch in every healthcheck mode.
Only lock contention is retried; validation and other I/O errors fail immediately.
Failures are recorded in
`cleanup-healthcheck.json`. Only unavailable CLI tooling or credentials can
select the existing API-cleanup fallback. The controlled resume launcher checks
the same registries before changing checkpoints.

Before the first format-2 launch on a reused host: **stop all workers and surviving
descendants, then move aside all three registry directories** under the shared
`output/workflow_runs` root: `.proofcouncil-compute-memory`,
`.proofcouncil-cleanup-memory/editors`, and `.proofcouncil-cleanup-memory/codex`.
Use the configured path instead if Compute has a custom registry directory.
Do not touch these directories while an old-code run or any descendants are
still active; restarting a controller alone does not establish that they stopped.

This does not validate provider credentials, model access, or extended hosted
tool behavior. It is not a paid one-token provider test. The production-path
Linux smoke in the checklist above remains a separate deployment gate.

## Controlled batch resume

`scripts/resume_batch3.py` defaults to read-only planning. It is a rehearsal
recovery launcher, not an AWS provisioner or a replacement competition entrypoint.
It resumes existing per-problem workflows and their publication artifacts; it
does not regenerate the official adapter's top-level `solutions.json` bundle.
The fresh submission adapter refuses an output tree with existing Batch 3
schedules instead of silently granting new clocks.

Start with a separate extracted checkpoint copy:

```sh
python scripts/resume_batch3.py /path/to/output/workflow_runs \
  --replay --report /path/outside-checkpoints/recovery-plan.json
```

The replay executes the actual workflow on a temporary copy, reuses compatible
cached leaves, and stops at the first uncached model-call wave. Agent execution
and Python socket connections are blocked. It reports exact `next_calls` and
`cache_hits`, including keys. It checks referenced cached artifacts, refuses
symlinks, and leaves the source untouched. This is not a proof of all future
rounds or an OS-level network sandbox for arbitrary external tools.

For execution, restore the recorded absolute layout (normally
`/data/output/workflow_runs/<run-id>`). Restore artifacts, not active processes
or shared admission leases. A still-live recorded `run.pid` blocks execution;
investigate PID reuse on a different host rather than killing an unrelated PID.

A JSON manifest has a `runs` object keyed by full run ID. Each unfinished run
requires explicit reconciliation and the reviewed replay. For example:

```json
{
  "runs": {
    "<run-id>": {
      "reason": "Operator-approved interrupted rehearsal recovery",
      "billing_reconciled": true,
      "extend_seconds": 0,
      "liability_reserve_usd": 0,
      "compute_max_parallel_workers": 4,
      "debits": {},
      "code_sha256": "<top-level code hash from final dry run>",
      "schedule_sha256": "<this run's original schedule hash>",
      "checkpoint_sha256": "<this run's checkpoint hash>",
      "replay": {"next_calls": [], "cache_hits": []}
    }
  }
}
```

The example is a schema illustration, not a billing reconciliation or approval.
Fill `replay` with the actual dry-run result, not the empty example. Post known
unrecorded charges with stable IDs and provenance, e.g.
`"legacy-claude-<call-id>": {"cost_usd": 0.25, "reason": "<usage evidence>"}`
inside the affected run's `debits`. Do not post this on every problem. Resolve
unknown liability against invoices or explicitly reserve a conservative amount.
Debit entries are immutable and included once in cumulative accounting;
liability reserves reduce the spendable cap and persist across further resumes.
A reserve is not reported as incurred cost. An explicit later manifest can
release it after reconciliation.

Re-run planning with that manifest before approving its hashes/replay:

```sh
python scripts/resume_batch3.py /data/output/workflow_runs \
  --manifest /path/recovery-manifest.json --replay \
  --report /path/outside-checkpoints/final-plan.json
```

Review and transfer the resulting hashes/replay to the manifest. Only after
the billing and Linux deployment gates are satisfied, and spending is authorized:

```sh
python scripts/resume_batch3.py /data/output/workflow_runs \
  --manifest /path/recovery-manifest.json --execute --parallel 10 \
  --report /path/outside-checkpoints/resume-results.json
```

Execution repeats replay and validates its result, code hash, checkpoint hash,
absolute paths, remaining clock and cumulative budget. All native-host checks
pass before checkpoint publication or paid continuation. A batch lock excludes
another recovery launcher; SIGTERM cancels children with a bounded shutdown
grace. Do not use other launchers concurrently.

Approved exports are preserved without launching research or cleanup. Other
children get a journaled per-run recipe using `FirstProofBatch3RehearsalWorkflow`,
which preserves the saved page limit and cumulative cap without changing the
competition class's 16-page limit. Saved concurrency is
preserved unless the manifest deliberately selects four. Relative launch caps
cannot override the original absolute deadlines. A nonzero extension moves both
deadlines by exactly the approved seconds and records the old/new schedules.
After an interrupted migration, regenerate the plan from disk; do not blindly
reapply an old extension manifest.

## Provider references

- [Anthropic stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)
- [Gemini finish reasons](https://ai.google.dev/api/generate-content#FinishReason)
- [OpenAI background responses](https://developers.openai.com/api/docs/guides/background)
