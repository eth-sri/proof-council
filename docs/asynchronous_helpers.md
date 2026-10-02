# Asynchronous Author helpers

Both Batch 3 presets enable `Author.delegation.asynchronous: true`. Delegation
still requires `author_parallelism > 1`; the ordinary single-Author workflow is
unchanged. Direct component users can retain the legacy blocking implementation
with `asynchronous: false`.

## Model-controlled coordination

- `delegate(tasks, briefing)` starts independent helpers and returns task IDs.
  It does not wait for their research. A bounded sandbox snapshot may run first.
- `helper_status(agent_ids)` returns states, source versions, deadlines, cost
  receipts and readable artifact paths. Omit IDs to list all jobs for the problem.
- `wait_helpers(agent_ids, timeout_s)` waits until any selected job finishes,
  or at most 600 seconds elapse. An expired wait does not cancel a helper.
  The Author can wait again or work on something else. Already finished selected
  jobs return immediately; omit them when waiting for remaining work.
- `cancel_helpers(agent_ids)` requests cancellation and saves partial results.
- `read_context(path, offset, max_chars)` reads complete reports, code, proofs
  and data in chunks. Follow `next_offset` until null. For manifest pages, also
  pass the returned `revision`; a missing or expired revision requires restarting
  at offset zero. Each reader retains at most four snapshots within 8 MB.

There is no prescribed two-wave schedule, fixed task decomposition, mandatory
checkpoint cadence, automatic restart, or default 50-minute helper cutoff.
Continuations are explicit: pass a finished job's `agent_id` to `delegate`.
The continuation receives a new ID and the earlier published work, not the
earlier sandbox or private reasoning. Empty model replies are failures, not
successful helper completion.

## Lifetimes and versioning

The AC research workflow owns the helper session. Helpers can continue while
the Author edits and while the critic reviews a frozen manuscript. A later
Author turn can read their results or continue/cancel them. Only the Author
edits canonical manuscript files; a helper never changes a document under
review. Results carry the source round and SHA-256 of the Author input snapshot.
Each Author turn is told about running and recent helpers; the status tool lists
the full history. Notices include short assignments, full-task paths, deadlines,
source rounds and snapshot digests. The Author decides which results remain
relevant; reports are not inserted into a manuscript automatically.
Assignments use `helper-inputs/<id>/task.txt`, separate from the helper's
published `helpers/<id>/...` namespace.

Each helper gets a bounded, immutable context snapshot: the problem and explicit
assignment, optionally the saved manuscript at launch and latest critic/council/Compute
context, and previously published helper artifacts. Concurrent live edits and
private reasoning are not shared. `include_workspace: false` gives a blind
assignment except for explicit `depends_on` artifacts. Dependencies must have
finished. A continued non-blind helper cannot become blind by changing a flag.
If required dependency artifacts cannot fit, the launch is refused rather than
starting a checker or continuation without that evidence.

For shared tasks the harness reads only the three canonical files from the
current hosted sandbox, with a 60-second transfer bound, a 256-entry scan limit,
strict UTF-8 decoding and a 2 MB per-file ceiling. Each unavailable file falls
back explicitly to its turn-start input. The launch reply and helper prompt
identify these fallbacks and their warnings. Blind tasks do not fetch this
snapshot. Snapshot digests are distinct from the turn identity used for launch
allowances, so editing a file does not reset a launch cap. The snapshot does not
change the manuscript being reviewed or the lead's turn-start context.

Inputs are limited to 24 MB and 448 files, leaving at least 8 MB and 64 slots for
output. Required dependencies take precedence over optional helper history and
Compute files; optional omissions are listed, while an oversized required input
refuses the launch. Helper publication also leaves one slot and 2 MB for its
final textual report.
Omission details live in each helper's manifest, not the shared ledger. The
ledger and status retain the exact count; `context_omissions_path` reads up to
128 examples (1,024 characters each) and an explicit overflow summary. Legacy
embedded omission lists migrate on recovery without resetting launch counts.

When research exits for any reason, all remaining helpers are cancelled and
joined before the phase's final accounting and the rewrite phase. Standalone
`MultiAuthor` calls own their helpers and join them when the call ends. Nested
helpers are not allowed.
Repeated cancellation cannot interrupt normal finalization. A join has a
60-second grace period, including at research shutdown. If a helper cannot stop,
available artifacts and any latched report are retained, its store is sealed
against late publications, and the ledger marks `shutdown_incomplete` and
`usage_unresolved`. Only then is a stuck local cancellation handler interrupted
again. Further delegation is disabled, including on resume, pending
provider reconciliation. Late recorded usage still propagates to shared counters
and gets a separate cumulative receipt in `late-usage/`; it is not an additional
charge and does not overwrite the final ledger. This does not guarantee provider
termination or complete invoice accounting, and a stuck synchronous filesystem
operation or event loop cannot be preempted by an asyncio deadline.
An external stop still propagates after finalization or this bounded fallback. Auxiliary bookkeeping
failures are logged as `ac.author.helper_warning` events (with logging fallback)
without replacing a successful research result or its original failure.

## Limits and cancellation

The concurrency limit is per problem across all currently running jobs,
including earlier rounds. Busy slots cause a launch to be refused, not queued
with a misleading fresh deadline. The Author decides whether to wait or cancel.
Its prompt explains that substantive work can take hours: a quiet interval, an
expired wait, or the end of an Author turn is not a cancellation criterion.
Waiting is encouraged for results needed by the next submission when the
remaining allowance permits; useful independent work and cross-turn helpers
remain available without a mandatory wait-for-all barrier.

Helpers use the remaining research allowance minus `synthesis_reserve_s`
(default 1500 seconds). A finite workflow deadline is required; standalone
tools can fall back to their finite enclosing API-call deadline. An optional
`helper_timeout_s` can impose a smaller operator-selected limit. Provider/model
transport and invocation caps still apply and can fail earlier. The task prompt
reports its start and hard research deadline. The existing final-delivery
reserve is retained, but cannot interrupt opaque in-flight provider reasoning.

Costs and tokens propagate through the same research budget tree. Budget checks
before launch and during execution stop helpers when recorded shared resources
are exhausted. This is cooperative accounting, not an invoice guarantee:
concurrent requests and unreported interrupted usage can exceed a recorded cap.
Cancellation requests provider termination through the existing APIClient path;
provider-side completion/cancellation remains subject to transport availability.

Defaults leave `max_tasks_per_turn` unset. Operators may set a launch cap for a
bounded smoke test. Continuations and restored launches count against that cap.
The durable ledger is bounded to 1000 jobs and 2 MB; context bundles retain the
existing per-job file/byte and credential-filtering limits. Capacity failures
are explicit, not silently accepted complete handoffs.

## Durable partial results

Jobs live under `async-helpers/<problem-hash>/` in the run directory. `state.json`
records admitted jobs before any model invocation starts. Each job owns a
context manifest with byte sizes and content digests. Successful publications
are flushed as they happen; the ledger does not have to be rewritten on every
publication for those files to survive a process crash.
Finished job contents are released from RAM and read back from disk with digest
verification; a full day's context snapshots do not remain resident.

Interrupted jobs get a deterministic `helpers/helperN/final-response.md`
handoff listing all published artifact types, even if no Markdown progress
report exists. A completed response latched before interruption is retained.
Unpublished sandbox files and private reasoning cannot be recovered.

On restart, manifests and file hashes are checked. Previously running jobs
become `interrupted`; remote work is not automatically restarted or attached.
Published results and original input identities remain available, and consumed
launch allowances do not reset. A corrupted ledger/artifact fails closed.
Corrupt state is quarantined, not used as mathematical context, and does not
prevent the Author from rendering its next turn. If the launch ledger is valid
but its artifacts are damaged, known launch counts remain consumed and new jobs
receive fresh IDs. If launch history cannot be trusted, or quarantine/persistence
fails, further delegation is disabled while the Author/Critic flow continues.
Recovery warnings are included in the Author prompt and helper status.

Pro-mode local tool calls may replace the lead sandbox. Launch, status, wait,
cancel and context reads therefore snapshot transferable lead files before
returning; their replies describe read-only attachments to restore if needed.
This is real overhead: use waits rather than rapid polling. The model chooses
the polling/working strategy; the harness does not prescribe one.

## Verification

`tests/test_async_helpers.py` exercises launch/wait/cancel, concurrent limits,
cross-turn versioned results, partial `.tex`/code/data handoffs, crash recovery,
budget and deadline stops, and phase cleanup without paid model calls.
`scripts/pro_helper_smoke.py` supports the asynchronous path, but its paid mode
still requires explicit operator authorization and a budget.
