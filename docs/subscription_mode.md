# Subscription-only execution

`execution_backend: api` remains the default. For `firstproof_batch3` or
`author_critic`, opt into the subscription backend with a workflow input:

```bash
uv run python scripts/run_workflow.py --workflow firstproof_batch3 \
  --problem problems/example.txt \
  --input execution_backend=subscription
```

The backend replaces all model-bearing roles, not just the main Author:

| Role | Subscription runner |
| --- | --- |
| Author, research critic, simplification Author/critic | Codex, `gpt-6-astra`, `xhigh` |
| Cleanup regression critic and final stateless critic | Codex, `gpt-6-astra`, `xhigh` |
| Council | Astra/Codex plus `claude-fable-5-1`/Claude Code (`xhigh`); no Gemini seat |
| Compute | Codex, `gpt-6-astra`, `xhigh` |
| Cleanup editor and native correctness/exposition reviewers | Claude Code, `claude-fable-5-1`, `xhigh` |
| Cleanup attribution reviewer | Codex, `gpt-6-astra`, `xhigh` |
| Optional cleanup linear reader | Fresh tool-free Claude Code calls, one supplied prefix per call |

Model selection is pinned by this backend, overriding API model references in
components. Pro is not requested. Existing mathematical prompts, verdict parsers,
compile/page checks, simplification policy, partial-cleanup policy and final
stateless gate remain in force. This toggle does not enable optional stages that
were disabled. In particular, use `simplification_rounds=3` to enable the three
optional simplification attempts and their mandatory final stateless gate.

## Authentication and prerequisites

Both CLIs must be installed and logged into subscriptions on the execution host.
Codex reads `CODEX_HOME/auth.json` (default `~/.codex/auth.json`); Claude Code reads
`CLAUDE_CONFIG_DIR/.credentials.json` (default `~/.claude/.credentials.json`).
Only Claude's file-based OAuth subscription login is supported here; a
keychain-only login must first be made available using the CLI's supported
authentication flow on the execution host. Do not put credentials in YAML,
manuscripts, Git, or run artifacts.

Each worker receives a private temporary login directory outside the run tree.
The harness copies only subscription authentication, not user settings, API key
helpers, or gateway configuration. Temporary logins are removed at shutdown.
Provider API keys and cloud-provider authentication are not passed to children.
The API client factory and API client entry points reject subscription-mode
calls. The competition launcher skips API-key login bootstrapping and never
selects API cleanup when subscription preflight fails.

Authentication checks are offline. They cannot establish current subscription
quota, entitlement to a particular model, or CLI compatibility with the account.
Missing/expired credentials, quota failures, unsupported models, malformed
outputs or missing usage do **not** trigger paid API fallback. They produce a
failure or a reconciliation requirement. Account-level extra-usage/credits
settings are outside the harness: turn those off at the provider if required.
"No API fallback" does not mean the subscription itself has no charges.

This first version supports one Author (`author_parallelism=1`, the normal
single-author preset). Native API multi-author delegation is rejected, not
silently replaced. The separate API-backed LLM dashboard monitor (`--monitor`)
is also rejected; ordinary event logs still work. Use the native subprocess
host with the normal LaTeX/runtime dependencies. No live subscription validation
is implied by the offline tests.

## Context and delivery

Every research/review call starts a fresh CLI worker. Its complete role packet
is saved as `request.md`; any stateful history is explicitly included there.
Authors edit canonical files in their workspace. Critics receive the full
manuscript and notes in the packet, not expiring provider-hosted attachments.
The notes are still background, not a substitute for proofs in the manuscript.
The final stateless gate receives its ordinary fresh packet without prior
review conversations. There is no automatic switch back to Pro, Gemini or an
API editor. CLI tools replace hosted API tools; their individual tool limits
and context-management behavior are not identical to the API models'.

Fresh packets are conversation-isolated, not a guarantee of filesystem read
isolation. Native CLI tools can have access beyond the invocation directory;
use a dedicated execution account/container with no unrelated private files.
Compute uses `codex_sandbox: auto`, which selects `workspace-write` on a native
host and only bypasses the inner sandbox inside the Docker backend. Subscription
mode overrides API presets' Compute bypass setting.

## Limits, accounting and resume

- `subscription_planning_usd` defaults to 1000. This is a nominal API-equivalent
  allowance, **not** a dollar bill or a provider-quota measurement. It keeps
  research, cleanup and final-review reserves meaningful without spending API
  money. Existing phase dollar limits operate in these planning units.
- Tokens and `api_equivalent_usd` are recorded; actual `cost_usd` stays zero.
  Reasoning tokens are part of output tokens, not charged twice. Missing usage
  fails closed. Existing conservative cleanup allowance estimates remain
  conservative planning charges, visibly marked as estimates.
- `subscription_call_seconds` defaults to 5400 and is capped by remaining phase
  time, including time queued for a subscription worker.
- `subscription_max_parallel` defaults to 2 for research/review packet workers.
  Compute and cleanup retain their separate existing memory/worker admission
  controls; this is not a host-wide account-quota limit. Cleanup's lead does not
  hold a packet-worker slot while it waits for a helper.
- The normal wallclock deadline, cleanup/final-review time reservations, token
  limits and workspace limits still apply. A subscription quota reset does not
  extend the saved deadline automatically.
- Backend, pinned models and subscription settings are persisted in
  `execution-backend.json` and cannot change on resume. Start in a fresh run
  directory; an API checkpoint cannot be resumed as subscription work.
  Use `--restart-from` to continue the full run directory (or `--restart-copy`
  for a complete copy). External-cache-only resume is rejected because it
  would omit the cumulative ledger and deadline state.
- Completed invocations have content-keyed receipts in `subscription_calls/`
  and can be reused without another model call. An interrupted invocation is
  blocked on replay or resume, even with changed inputs, until its
  worker/output/usage is reconciled. Confirmed no-spawn failures and terminal
  workers with settled usage become `failed`, with at most one retry of the
  same packet. Cancellation with only partial usage remains unresolved.
  Council failures that need reconciliation propagate instead of being hidden
  in a textual reply. The guard is checked before subsequent packet calls too,
  not only when entering a workflow.
- Atomic metering receipts in `subscription_usage/` back the event ledger.
  Resume deduplicates the two sources and can recover charges even if a crash
  truncated `events.jsonl`. Legacy runs without full receipt coverage still
  fail closed on a damaged usage log. The original allowance and absolute
  Batch 3 deadline are not replenished.

Do not delete a pending receipt or an accounting guard merely to force a retry.
Preserve the CLI logs, inspect worker termination and reconcile usage first.

For a crash that left no terminal worker receipt, stop every controller,
worker and descendant first. Then use the explicit operator command:

```bash
uv run python scripts/reconcile_subscription.py /path/to/run RECEIPT_ID \
  --confirm-workers-stopped --unreported-allowance-usd 100
```

The amount is an **additional conservative API-equivalent debit** for any
unreported work, not a new budget. Choose it from the interrupted call's saved
allowance; use zero only when inspection establishes that no unreported work
occurred. The command refuses an active controller lock, records the debit
idempotently per attempt, marks the invocation failed and never launches anything.
Repeating reconciliation for the same attempt does not charge again; if its
bounded retry is also interrupted, that attempt needs its own additional debit.
The command does not infer from the absence of a controller that its children stopped.
Normal resume remains an explicit, separate action, within the original limits.

Temporary credential copies are not written back to the source account on
shutdown. Long-running OAuth refresh/rotation compatibility therefore still
needs live validation; expired source credentials require re-authentication,
never a paid-key fallback.
