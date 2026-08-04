# Playbook index

General strategy for the CLI lead-researcher mode. Everything here is
meant to hold for *any* user on *any* problem; anything tied to a
machine, an institution or a person belongs in `local/` instead. Each
entry cites where it was learned so it can be revised or retired.

Read this index every session; read the individual files when the
situation calls for them.

| File | One-line summary |
|---|---|
| `packets.md` | A consultation packet is a self-contained folder with a frozen copy of what the consultant saw. Carry a cumulative dead-end ledger. |
| `delegation.md` | Match the job to the channel: browser packet for deep single-shot grinding, subagents for parallel breadth, a second model family for code review, local compute for anything finite. |
| `verification.md` | "Verified" means exact and scoped. Print invariants next to every result; an impossible number is a free bug detector. |
| `adversarial.md` | Commission red teams against your *strongest* result. Objections that are cheap to test are worth more than objections that are deep. |
| `continuity.md` | Compaction drops procedure, not results. Procedure needs an auto-loaded home and must be written as a checklist. |

## The five rules that matter most

If you read nothing else:

1. **Verify every delegated claim independently before adopting it.**
   Consultants and subagents are wrong in confident prose. This is the
   single highest-value discipline in the mode.
2. **Carry a cumulative dead-end ledger into every packet.** Every
   refuted claim, stated precisely with its counterexample, never
   pruned. It is what stops rounds re-deriving known-false things.
3. **Freeze what you sent.** A packet folder contains the brief *and*
   a copy of the artifact as of dispatch. Reports change between
   rounds; without the freeze the round is not reproducible.
4. **An "n out of n, no difference" result is a theorem in disguise.**
   Do not file it as robustness — conjecture the structural reason and
   test that instead.
5. **Report scope, not vibes.** "Each of 53 representatives is smooth"
   is not "the fan is smooth". State limitations as limitations.

## Provenance

Entries were learned running this mode on an open problem in the
geometry of toroidal compactifications (2026), and are written to
generalise. Where an entry has only ever been observed once, it says
so.
