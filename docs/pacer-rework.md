# Subscription pacer — rework plan

**Status: EXPERIMENTAL, disabled by default, carved out of the council-infra PR.**

The subscription **pacer** (admission control that gates CLI nodes against
provider usage windows) is `enabled: False` by default and must not be relied on
in production until the rework below lands. The usage **ledger** in the same
module (`SubscriptionStore.append_usage` / `window_entries`, used by metering)
is stable and stays in the PR — only the pacing/admission logic is deferred.

## Why it was carved out

The pacer went through four consecutive focused review passes. Each one found
real P1/P2 correctness defects in the same subsystem, several in the unsafe
*over-admit* (overspend) direction, while the rest of the branch (batch status,
human-input coercion, metering-on-cancel, dashboard) stayed stable. The defects
are structural, not typos, so incremental patch-and-re-review was not
converging. It is disabled by default, so none of these affect a shipped run;
this document is the backlog for a dedicated rework branch.

## Open findings (pacer-only review pass at `f112f00`)

Severity as reported by the reviewer; ✓ = reproduced locally in this session.

Over-admit / overspend (the ones that cost money):
- **P1 — write-side legacy shadow** ✓ `record_rate_limit` reads/writes only the
  provider-scoped calibration entry, so a reset-less hit overwrites (per field)
  the legacy top-level block that `_provider_calibration` merges only on *read*
  → admits into an exhausted window. (The read-side merge, F2, is fixed; the
  write side is not.)
- **P1 — zero manual ceiling falls through to seed** ✓ `if manual:` treats a
  manual ceiling of `0` as absent and uses the plan seed instead of hard-blocking.
- **P1 — stale codex re-harvest admits an unbounded batch** once a probe-only
  codex sample expires the window is dropped from the status list, so N nodes
  admit where only one should run and re-harvest.
- **P1 — probe-delta calibration across a reset** (pre-existing) a positive pct
  delta spanning two different reset epochs inflates the durable ceiling
  (`own_tokens` uses a rolling filter, not the discrete block).
- **P1 — `try_claim` captures `now` before a slow probe** (pre-existing) ledger
  entries committed during a ≤60 s probe are excluded as "future", so admission
  decides on a stale timestamp under concurrency.

Over-block / self-healing (errs safe — pauses, does not overspend):
- **P2 — durable boundary not advanced beyond one window** ✓ once a boundary is
  >1 window in the past, `block_start` stays that old epoch, so a completed
  block's spend is still counted until the next probe rewrites the boundary.

Incomplete versions of the Round-4 async fixes:
- **P1 — partial-metering double-cancel** the B6 drain fixed the primary meter
  task but not the partial-usage fallback, which still shields an anonymous
  coroutine; a second cancel orphans it.
- **P2 — claim leak** the B7 fix covered cancel-during-admit-emit but not
  cancel-during-`try_claim` (claim registered in the thread, result lost) nor a
  non-`CancelledError` exception from the emit.

Cosmetic:
- **P2 — window-label misclassification** an unlabeled `usage limit reached`
  match with a `5-hour` prefix *and* unrelated `weekly` prose elsewhere still
  classifies as weekly. Very low real-world likelihood.

## Root-cause themes for the rework

1. **Two competing calibration stores.** Legacy pre-namespacing (top-level) vs
   provider-scoped, reconciled only by a read-time merge while writers touch the
   scoped copy → writes clobber merged legacy fields. Fix: a one-time migration
   at load to a single source of truth; drop the read-time merge.
2. **Single-epoch block boundary.** The boundary is a frozen reset epoch, not
   advanced by whole window durations, so it goes stale after one full window.
   Fix: derive the current block by advancing the reset by `⌈(now-reset)/window⌉`
   windows; compute `block_start` / next-reset from that.
3. **`None` / `0` / absent ceiling conflation.** `allowed` and the manual-ceiling
   check use truthiness. Fix: a typed ceiling (absent vs explicit-0 vs N) so a
   0% cap hard-blocks and probe-only stays uncapped.
4. **Incomplete async cleanup.** Claim ownership and partial metering need the
   same retained-task + drain-loop discipline on every cancel/except edge, not
   just the happy path.
5. **Discrete-block-aware calibration.** `own_tokens` for probe-delta ceiling
   estimation must be scoped to the block, and a changed reset epoch must reset
   the delta baseline.

## What stays in the PR

- The usage **ledger** / metering (`append_usage`, `window_entries`,
  `record_probe`/`record_rate_limit` as write-only sinks) — the dashboard panel
  reads it for display only.
- The pacer code, `enabled: False` by default, with a runtime `RuntimeWarning`
  if anyone turns it on (`SubscriptionPacer.gate_config`).
