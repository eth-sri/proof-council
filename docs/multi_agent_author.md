# Multi-agent Author (`MultiAuthor`)

Target: the First Proof Third Batch Author/Critic workflow
(`configs/workflows/firstproof_batch3.yaml`).

**Current Batch 3 behavior is documented in [Asynchronous helpers](asynchronous_helpers.md).**
Both Batch 3 presets now use nonblocking delegation. The implementation notes
and measurements below describe the retained legacy `asynchronous: false`
mode, including its two-wave schedule and 50-minute cutoff; those are no
longer the Batch 3 defaults.

`author_parallelism` selects the Author's behavior in the same workflow.
At `1`, it uses the ordinary Author with no delegation tool or instructions.
At `N > 1`, the lead may delegate to at most `N` helpers concurrently for one
problem. It may choose fewer helpers or none, and chooses when to wait.
This setting does not disable the Critic, advisory council, or Compute.

## Legacy Design

The single Author is one long Pro-mode Responses call per round. The
observed failure mode is a local minimum: the same argument gets patched
round after round because one context holds one plan. Codex "Ultra" mode
attacks this with the multiagent-v2 architecture: a root agent spawns
role-specialised subagents, waits for their summaries, and keeps the
integration for itself. `MultiAuthor` is that architecture translated to
plain API calls (no CLI, no Codex).

## Mapping to Codex multiagent-v2

| multiagent-v2 | MultiAuthor |
|---|---|
| root agent with tools | the lead `Author` call, unchanged (container files, code_interpreter, web search) |
| roles (`developer_instructions`, `model`, effort) | `ROLE_INSTRUCTIONS` for explorer / prover / checker (drafter available, off by default); `role_models` / `subagent_model` |
| `spawn_agent` + `wait_agent` | one blocking `delegate(tasks=[...], briefing=...)` call per wave |
| `followup_task` / `send_message` | a task with `agent_id`; the seat's conversation is replayed (`messages_after`, same trick as the stateful Critic) |
| `close_agent`, `report_agent_job_result` | implicit: a seat ends when its call returns; the report is the tool result |
| `max_threads`, `job_max_runtime_seconds`, depth 1 | `author_parallelism`, `job_timeout_s`; seats have no `delegate` |
| subagents return summaries | `## Summary / ## Caveats / ## Findings` report; findings cut at `max_report_chars`, full text accessible through `read_context` |
| fork_turns (share the root's context) | explicit read-only context snapshot, not private reasoning or a live sandbox fork |

Why one blocking call per wave instead of spawn/wait pairs: every local
function-tool call ends the current Responses request and starts a new
one that replays the whole conversation and re-runs the lead's (Pro)
reasoning. Cached input is cheap, but each round trip costs one more
full lead reasoning pass and adds latency. Batching a wave into one
call keeps that to two or three lead requests per turn.

## What happens in a turn

The following applies when `author_parallelism > 1`:

1. The workflow calls the Author exactly as before. `MultiAuthor` renders
   the same developer prompt plus a "Delegation (multi-agent mode)"
   guide, and builds the same client plus local `delegate` and `read_context` tools.
2. The lead reads the Critic review and the files, decides what the
   round needs, and (optionally) calls `delegate` with up to
   `author_parallelism` self-contained tasks by default. The APIClient runs the
   tool in its worker thread; `MultiAuthor._delegate` hands the wave to the
   asyncio loop and blocks until all seats report.
3. Each `SubAuthorSeat` is a fresh `APICallAgent` call (its own budget
   child under the Author, its own events and workdir, nested under the
   Author's agent path), with code_interpreter (own container) and web
   search. Concurrency is a semaphore of `author_parallelism`. The whole wave
   has one absolute deadline: `min(job_timeout_s, min(remaining workflow
   wallclock, remaining lead API call time) - synthesis_reserve_s)`;
   a wave is refused if that leaves under 5 min. The lead call deadline is
   supplied by APIClient and retains the original start across retries.
   Each helper is told its actual UTC start, absolute deadline and remaining
   seconds after admission. Queued tasks do not receive a fresh 50 minutes.
   `wrapup_reserve_s` defaults to 300 seconds, capped at 10% of the admitted
   allowance. It is inside, not added to, the hard deadline. During this final
   interval, Responses helpers can read shared context and publish existing
   artifacts, but subsequent requests no longer offer research tools. A long
   already-running provider request can still consume this interval; the
   reserve is not a guarantee that a final report arrives.
   At the deadline pending seats are cancelled and reported as
   `(error: cancelled: wave deadline ...)`; finished reports are kept.
   A run-scope `BudgetExhausted` in any seat cancels its siblings, and
   the tool result ends with an instruction to save the files and stop;
   further `delegate` calls are refused, and the Author's own post-call
   budget check then raises as usual. If the lead itself is cancelled
   while blocked in `delegate`, the active wave is cancelled too.
   The sandbox snapshot shares the wave deadline. A failed or timed-out
   snapshot adds a warning without dropping completed reports; reports
   are persisted before waiting for the snapshot. During a wave the lead's
   container is refreshed every
   `container_keepalive_s` (event `ac.author.container_keepalive`).
4. The tool result is all reports concatenated (`## Delegation wave k of
   n`, one `### agent_id (role, model, minutes)` section each). The lead
   continues in the same conversation: verifies, integrates, edits the
   files in its container, and finishes as usual.
5. Reports are written to `<author workdir>/subagents/wave{k}-{id}.md`
   and `delegation_log.json`; a one-line-per-seat summary lands in
   `Author.Outputs.delegation_summary` and in `.ac/author-round-k.md`.
   Events: `ac.author.delegate.wave_start`, `ac.author.subagent_done`,
   `ac.author.delegate.wave_done`.

## Shared evidence and artifact handoffs

Each Author turn builds a bounded read-only context bundle. Its manifest lists
the round number, source, byte size and SHA-256 for each file. The snapshot
contains the problem, three canonical files, latest critic/council/Compute
replies, workflow feedback, and safe text members of the Compute archive.
The lead and helpers use `read_context(path="manifest.json")`, then read files
in chunks using `next_offset`. These are tool paths, not hosted sandbox paths.
Context reads have their own local-tool allowance, separate from hosted tools.

Helpers normally receive all completed earlier-wave reports and published
artifacts as well. `depends_on: [prover1]` explicitly requires a completed,
transferred report before a checker is launched; unknown or unavailable
dependencies fail before consuming a wave. Same-wave dependencies are not
supported. A fresh task with `include_workspace: false` is blind: only the
problem, explicit task/briefing and selected dependencies are shared. A continued
seat cannot forget previous context; use a fresh seat for a blind check.

Helpers publish selected UTF-8 proofs, scripts or certificates using
`publish_artifact(name=..., content=...)`. Successful publication returns a
readable path. Revisions require new names. The complete final reply is saved as
`helpers/waveK-ID/final-response.md`, even when charging the completed call
exhausts the budget. Long summaries point to this accessible file, not merely a
local workflow path. Publication closes when the seat ends, so late tool calls
cannot mutate the completed handoff. Apart from the explicit checkpoints described
below, live sandbox files and private reasoning are not transferred; the prompts require publication of essential
evidence and warn that sandbox links alone are insufficient.

Transfers allow 2 MB per file, 32 MB and 512 files per turn. Compute imports may
only fill the first 16 MB and 256 slots, counting the round context already
present, so at least half remains for helper outputs. A Compute member that
cannot be read completely within the scan limit
is omitted rather than published as a truncated file. Binary files,
symlinks, unsafe paths, known credential paths/content and oversized files are
omitted, never silently claimed as delivered. The lead sees omissions; an
untransferred full report cannot serve as a declared dependency. Compute
archives are read without extraction, subject to their existing preflight caps
and a bounded text scan. This does not replace the local Compute worker or
give hosted helpers its resources.

When an already-completed Author call exceeds a budget, bounded manuscript
retrieval and cleanup precede propagation of `BudgetExhausted`. The workflow
checkpoints the recovered manuscript as **unreviewed**, awaiting critic review
on an authorized resume, rather than falling back to the preceding draft.
The Author's readiness vote is retained, but approval still requires the normal
critic and submission gates after resuming.
Retrieval does not make an additional inference call or approve the manuscript.
Both provider download paths allow up to three attempts, with five-second
backoffs inside one 90-second retrieval deadline and 15-second SDK request
timeouts. An outer timeout or cancellation does not start another download
while the previous thread may still be running. Exhausted retrieval attempts
retain the existing failed-artifact behavior.
Interrupted calls or failed downloads still cannot supply a verified artifact.

## Container persistence

Probed live (gpt-5.4-mini, `scratch/probe_container_persist.py`): the
code_interpreter container survives the extra request forced by a local
tool call: same container id, files written before the tool call are
still there afterwards. OpenAI expires a container after 20 minutes
without activity, and "any container operation, like retrieving the
container", refreshes `last_active_at`; a request that names an expired
container fails. A wave can take 45 minutes, so three defences:

- The wave driver retrieves the lead's container (id taken from the
  conversation passed to the tool) every `container_keepalive_s`
  (default 300 s) while seats run.
- `find_container_id` returns the *last* container of the turn, so the
  download reads whichever container holds the final writes.
- The guide tells the lead to edit files after the final wave and to
  re-create anything missing from `/mnt/data`.

Measured (gpt-5.4-mini, `scratch/probe_container_expiry.py`, a tool call
that idles 22 minutes, logs `scratch/probe_expiry_{nokeep,keep}.log`):

- without keepalive the request after the tool call did **not** fail;
  OpenAI silently created a new container holding only the originally
  uploaded files, so everything the lead had written in the sandbox was
  gone (the old container answers 404 "Container is expired");
- with `containers.retrieve` every 5 minutes the same container was used
  throughout and the written file was intact.

So the keepalive is load-bearing, and "last container id" is the right
one to download from when it does fail.

**Pro mode resets the sandbox on every request.** Measured with the same
probe without the idle wait (`scratch/probe_container_persist_{astra,pro}.py`):
`gpt-6-astra-max` in background mode reuses the container across the
request forced by a local tool call; `gpt-6-astra-pro` gets a *new*
container for the next request, one minute later, with only the original
attachments in it. Files can remain in the previous container without being
visible in the next request, so original attachments alone are not a checkpoint
of edits made during the turn.

Defence (`sandbox_carry_over`, auto-on for Pro): while a wave runs, the
harness lists the frozen pre-wave container, copies every text file the
lead wrote under `/mnt/data` (not the `file-*` attachments, not PDFs, up
to `carry_over_max_bytes`) to fresh platform files, and appends their ids
to the code_interpreter tool's `file_ids`, which the APIClient re-reads
per request. The wave result then lists them
(`/mnt/data/file-<id>-answer.tex`) and tells the lead to copy each back
to its canonical path if the canonical files are gone. Copies from an
earlier wave are replaced. Each turn owns its uploads and container descriptor;
only a completed snapshot of an active turn can change its attachments.
Zero-byte files are recorded in the snapshot's `empty_files` list instead of
uploaded, because the Files API rejects empty uploads. The tool reply tells
the lead to recreate these as zero-byte files after a sandbox reset, without
restoring older contents. A successful empty snapshot entry detaches any
earlier carried copy of that file; it does not add placeholder text.
Cancelling an asynchronous wait cannot stop an upload already running in a
thread, so late uploads are deleted and cannot attach to a later turn.
Carried platform files are deleted after the final manuscript download,
including failed and cancelled turns. File operations and cleanup waits are
bounded; deletion remains best effort when the provider is unavailable.
Event: `ac.author.sandbox_carry_over`.

The config handed to `ctx.api_client_factory` is deep-copied by
`load_solver_config`. The harness therefore updates the built client's
`client.tool_descriptions`, not the original configuration dictionary. Tests
must reproduce that copying boundary so detached-descriptor regressions do not
pass unnoticed.

### Helper checkpoints and direct publication

OpenAI Responses helpers now have a separate, bounded checkpoint mechanism.
Before `read_context`, `publish_artifact`, or `publish_sandbox_artifact`, the
harness reads flat UTF-8 files under `/mnt/data/checkpoints` from the container
identified by the helper's actual execution history. The model cannot supply
another helper's container ID or replace the injected execution history.
No other directory is automatically copied.

`publish_sandbox_artifact(path="/mnt/data/checkpoints/result.json")` publishes
the downloaded bytes, with their SHA-256, container ID, file ID, execution-item
ID and sandbox path in the shared context manifest. Ordinary
`publish_artifact(name=..., content=...)` remains available for prose but does
not receive this provenance. Published artifacts remain immutable; use a new
filename for revisions. Provenance certifies where bytes were obtained, not
whether a mathematical claim in them is true.

Checkpoint snapshots are durably saved in the helper's context namespace and
uploaded as read-only attachments for subsequent requests. The harness changes
the built API client's container descriptor, not the pre-copy configuration.
The tool reply's `sandbox_checkpoint.files` lists `attachment_path` and
`sandbox_path`; after a reset the helper copies the former to the latter.
Restoration is explicit in the prompt, not a claim that attachments already
occupy their original paths. Repeated local calls without new hosted execution
reuse a snapshot; unchanged contents reuse an upload, while later executions
receive fresh download provenance.

Limits per helper invocation: 16 retained files, 2 MB per file, 4 MB retained
total, 64 distinct `(filename, content hash)` versions, 256 listing entries per boundary, and a 60-second
transfer deadline clamped to the remaining API-call deadline. Individual SDK
operations have at most a 10-second timeout and no retries; an in-flight
operation can finish after cancellation, but cannot commit to a closed helper.
Downloads enforce the actual streamed-byte ceiling, not only reported size.
The turn's shared 32 MB / 512-file context limits also apply to retained
versions and published aliases. These are checkpoints, not whole-sandbox backups.

Unsupported files and nested caches are recorded as omissions and skipped.
C/C++ source and header files are supported. Each eligible file transfers
independently: a malformed or oversized neighbor cannot suppress a valid
checkpoint or its explicit publication. Duplicate paths are not published.
An incomplete/over-limit container listing is still a scan failure. Repeated
unchanged files reuse their immutable artifact and upload, without consuming
another version; the current boundary still downloads them and gets a fresh
capture receipt. Direct publication requires that fresh receipt, not a stale
entry retained from an earlier boundary.

Failures are returned explicitly in tool replies, copied into the final report
and follow-up history, and included separately in the lead's wave result so
report truncation cannot hide them. `helper-checkpoints.json` records the latest
completed boundary; `ac.author.subagent_done` and `delegation_log.json` include
`checkpoint_errors`. Successful downloads remain available even if uploading
the restoration attachment fails. Upload cleanup is best effort and bounded;
late uploads cannot attach to another invocation. Each helper owns its own
attachments, even when filenames are identical.

`scripts/pro_helper_smoke.py` now classifies recovery as `passed`, `failed`, or
`unverified`. It requires independent original and recovered marker downloads
from different containers, rejects explicit protocol failures and transfer
warnings, and never converts matching model-authored fields into verified
execution evidence. It does not require stdout to establish file continuity.
Offline tests alone do not validate provider behavior; a live test requires
explicit spending authorization.

### Interrupted-turn recovery

Before paid helpers launch, the lead atomically records consumed wave/seat
allowances and the context location under
`helper-recovery/<canonical-problem-sha256>/round-<n>.json`. Published helper
artifacts update the bounded context manifest; completed or cancelled helpers
also update the recovery record. On a same-round retry, matching problem,
baseline manuscript, critic/council/Compute inputs and artifact hashes are
required. Symlinks, nonregular files and invalid records are rejected before
further model work. Recorded spend and original run deadlines still apply.
If the same round now has different Author inputs (for example, a manuscript
was saved before the resume-state transition), its old checkpoint is not
replayed. The new invocation starts with no recovered research or wave counters
from those old inputs; its first checkpoint replaces the stale manifest.

The lead receives the recovered reports and scripts through `read_context`
and a recovery notice; consumed delegation waves are not reset. The reports
remain unreviewed evidence. Remote conversations, attachment IDs and sandbox
state are not replayed. A helper continuing a recovered task must read its
saved artifacts and recreate files in a fresh sandbox. Nothing can recover
unpublished reasoning or files that never reached the local context.

On a deadline/cancellation race, the client attempts bounded cancellation and
a same-response-ID retrieval. If the provider already completed, the final
text and usage are retained without executing additional tools or issuing a
new generation. Usage receipts are persisted before recovered report export.
Unresolved IDs and unavailable usage remain explicit, not silently counted as
zero or treated as a successful helper. Completed text is also saved as
`provider-completed-<invocation-id>.json` beside provider receipts. Failure to
write this supplementary copy is a warning, not a reason to discard the
in-memory result or retry the paid call; failure to persist the accounting
receipt remains terminal. Helpers retain completed reports before cancellable
attachment cleanup, so a deadline during cleanup does not erase their result.

Runs created before these recovery manifests existed are not automatically
migrated. Their saved reports need
an explicitly prepared, validated recovery packet before a resumed launch.

## Provider polling failures

Transient polling errors are not evidence that a background response failed.
Retry retrieval of the same response ID within the attempt deadline instead
of immediately issuing another paid generation. Preserve response IDs and
accounting receipts so cancellation and unresolved usage can be reconciled.
Choose polling intervals appropriate to the model's expected response time;
a long interval adds unnecessary latency to cheap-model smoke tests.

## Configuration

Set the public workflow input to choose the Author's maximum parallelism:

| `author_parallelism` | Author behavior |
|---|---|
| `1` | Ordinary Author; no delegation tool or delegation instructions |
| `2` | Optional delegation to up to two concurrent helpers per problem |
| `4` | Optional delegation to up to four concurrent helpers per problem |

The value must be a positive integer. The lead waits for each wave, so it
does not count as an additional active helper. `max_waves` separately limits
successive waves within one Author turn. The default tasks-per-wave limit
follows `author_parallelism`; an explicit advanced `max_tasks_per_wave` can
change the amount of work in a wave without increasing concurrent helpers.

The presets share all component settings: `firstproof_batch3.yaml` defaults to
`author_parallelism: 1`, and `firstproof_batch3_multiauthor.yaml` to `4`.
Changing only this input on either preset selects the corresponding behavior.
Both presets use Astra Pro helpers at max effort, with the existing Pro fallback
policy. Their charges share the problem budget; this does not change the
standard Astra model used by local Compute workers.
For an explicitly budgeted run, the direct CLI override is:

```bash
uv run python scripts/run_workflow.py --workflow firstproof_batch3 ... \
  --input author_parallelism=4
```

The competition adapter accepts `FIRSTPROOF_AUTHOR_PARALLELISM=4` and otherwise
uses the selected preset's value. Set it to `1` to use the ordinary Author even
with the multiauthor preset. Invalid or nonpositive environment overrides warn
and retain the selected preset's default.

The workflow input takes precedence over legacy `delegation.enabled` and
`delegation.max_threads` component settings; migrate those overrides to
`author_parallelism` when launching the workflow.

Advanced delegation settings remain on the `Author` component:

```yaml
inputs:
  author_parallelism: 4
components:
  Author:
    model: models/openai/gpt-6-astra-pro
    delegation:
      max_waves: 2               # launched helper waves per Author turn
      # max_tasks_per_wave: 4    # optional; defaults to author_parallelism
      job_timeout_s: 3000        # absolute deadline for the whole wave
      wrapup_reserve_s: 300      # inside that deadline, capped at 10% of admitted time
      synthesis_reserve_s: 1500  # wallclock the lead keeps after a wave
      subagent_model: models/openai/gpt-6-astra-pro  # null: lead's model
      role_models: {}            # optional per-role model overrides
      roles: [explorer, prover, checker]   # drafter available
      seat_max_tool_calls: null  # no extra hosted-tool cap; 0 disables hosted tools
      max_report_chars: 14000    # findings cut here; caveats kept
      container_keepalive_s: 300
      sandbox_carry_over: null   # null = auto (Pro mode); true/false to force
      carry_over_max_bytes: 2000000
```

`seat_max_tool_calls` defaults to `null` (no additional hosted-tool cap).
The wave deadline and shared dollar budget still apply. A nonnegative integer
is enforced for OpenAI Responses seats through the
separate `APIClient.max_hosted_tool_calls` allowance. The provider receives
the remaining allowance on every request; continuations and retries do not
reset it. An ambiguous failed request consumes its granted allowance because
its hosted tools may already have run. Zero removes hosted tools. Other seat
providers retain their existing behavior and do not receive this Responses-only
limit. Local function limits such as the lead's `delegate` counter are separate.
The lead's tool-call allowance includes three additional attempts for correcting
invalid delegation requests. `max_waves` still caps actual launched waves;
rejected requests cannot increase it, and repeated invalid requests eventually
exhaust the bounded attempt allowance. Setting `max_waves: 0` disables the tool.

Model settings accept both config references and inline config dictionaries,
including `subagent_model`, entries in `role_models`, and the inherited lead
model. Continued seats receive the new wave's briefing as well as their prior
task/report conversation.

## Competition resources: 64 GB only

The competition deployment remains `r7i.2xlarge`, with 8 vCPU and 64 GB RAM.
Both Batch 3 presets start with a configurable ceiling of four local Compute
workers, each with an 8 GiB allowance, and a shared 16 GiB free-memory reserve.
Actual available RAM and cgroup headroom can reduce admission below that ceiling.
Four is a conservative starting policy, not a measured throughput optimum.
The policy does not depend on obtaining a larger machine. Batch 3 rejects a
zero Compute cap because that would disable the shared memory guard.

The three concurrency controls are independent:

| Control | Scope | Initial setting |
|---|---|---|
| `FIRSTPROOF_MAX_PARALLEL` | Problem pipelines in the competition adapter | 10 |
| `author_parallelism` / `FIRSTPROOF_AUTHOR_PARALLELISM` | Author mode and maximum concurrent remote helpers per problem | 1 (baseline), 4 (multiauthor preset) |
| `compute_max_parallel_workers` | Local Compute workers sharing the registry | 4 |

With `author_parallelism=4`, ten Authors can have forty delegated API seats
running; hosted seat sandboxes do not occupy local Compute slots. There is no batch-wide API
concurrency limit in this change. Increasing `author_parallelism` does not
increase local Compute concurrency or require a larger machine. Setting an
advanced `max_tasks_per_wave` above parallelism queues some seats inside the
same wave deadline, so adjust work and timeouts together. Seat charges propagate
through the existing research and problem budgets; already-running or cancelled
calls can still incur charges.

Use `FIRSTPROOF_COMPUTE_MAX_PARALLEL_WORKERS` to override the selected preset's
positive ceiling in the competition adapter, or the direct CLI's
`--input compute_max_parallel_workers=N`. See
`docs/firstproof_batch3_workflow.md` for shared-registry requirements and
changing an existing registry's policy safely. The memory guard is cooperative
polling, not a kernel aggregate quota. Profile queue wait, process-tree peak
RSS, CPU saturation, completed computations, and time to the next Author turn
on the actual 64 GB deployment before increasing concurrency.

## Cost and wall-clock

- A wave of 4 seats on the same Pro model costs roughly 4 extra
  Author-sized calls; the lead's extra request after the wave is mostly
  cached input plus one more reasoning pass. Expect a delegating round
  to cost 3 to 6 times a plain round and take 1.5 to 2.5 times as long.
  With the Batch 3 budget ($1000, 23 h) that means fewer, heavier rounds.
- `job_timeout_s` 3000 bounds a wave at 50 min (plus a 2 min cancellation
  and bookkeeping grace for the tool thread). The APIClient's
  `max_wallclock_per_call_s` (14000 s for astra) spans the whole tool loop.
  The wave is shortened or refused when either this call deadline or the
  workflow deadline cannot leave `synthesis_reserve_s` for the lead. A
  3000 s seat call is also shorter than the 3300 s
  background-server kill, so a slow seat times out cleanly rather than
  being downgraded mid-flight.
- The lead's request that issues `delegate` ends at the tool call, so
  the ~60 min server-side kill of background Responses requests is not
  made worse by the wave: the wave runs between requests.

## Tests

`tests/test_ac_multi_author.py`: disabled == plain Author (tools,
prompt, client config); enabled adds the tool and guide; a wave respects
the concurrency limit, formats reports, writes artifacts, nests seat events under
the Author, follow-ups replay the seat conversation, wave and task limits
are enforced; wave deadline cancels pending seats; run-scope budget
exhaustion cancels siblings and refuses further waves; wallclock reserve
refuses a wave; duplicate `agent_id` rejected; briefing reaches every
seat; truncation keeps caveats; container keepalive events; sandbox
carry-over re-attaches written files and cleans up; role model
selection; `find_container_id` returns the last container.

`tests/test_ac_multi_author_context.py` covers updated follow-up briefings,
inline model specifications, lead-call deadline refusal and clamping, and
provider-specific hosted limits. `tests/test_ac_multi_author_snapshot.py`
covers interrupted uploads, bounded snapshots, retained reports, and cleanup
after download. `tests/test_api_client_delegation_limits.py` checks outgoing
provider limits, shared retry/continuation allowances, and trusted call-deadline
injection. Deployment tests enforce preset parity apart from Author parallelism,
positive Batch 3 Compute limits, and competition-adapter override forwarding.

## Open questions

- Which `author_parallelism` improves results within the same budget? The
  baseline stays at `1` until paired runs establish a benefit.
- Pro helpers are the preset default. Their quality, latency and total cost
  relative to standard Astra/max helpers have not been measured on these
  problems; no fixed cost multiplier is established.
- Resume restores published research, not unpublished reasoning or a provider
  conversation. The lead still makes a new, budgeted integration call.
- The Anthropic container-files path gets the tool too but is untested
  live.
- No `fork_turns`: a seat never sees the lead's reasoning. Cheap to add
  as "include the Critic review" if the lead asks for it.
