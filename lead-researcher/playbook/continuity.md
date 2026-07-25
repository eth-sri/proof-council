# Continuity: surviving compaction and crashes

Long research sessions get compacted. **Compaction preserves results
and decisions; it drops procedure.** That asymmetry is the whole
problem, and it has a specific cause: results live in a ledger file
that gets re-read, while procedure tends to live only in context and
in prose scattered across documents.

Narrative learnings are not a substitute for a checklist. A lesson
written as "we learned that X matters" will be followed until the
first compaction and then silently dropped. The same lesson written as
a checkbox in a file that gets read before the relevant action will
not.

## What actually survives, in descending reliability

1. **Auto-loaded memory** — if the harness loads a memory index into
   every session, that is the only mechanism verified to survive.
   Keep entries there short: pointers, not content.
2. **A `CLAUDE.md` in the working directory** — loaded at session
   start and re-injected after compaction. The right place for a
   project entry point: what this is, which files to read, in what
   order.
3. **Skills** — loaded on demand, so they survive only if you think to
   invoke them. Right home for *cross-project* procedures.
4. **Ordinary files** — survive only if something above points at
   them.

Use several layers. They are cheap.

## The four files a long project wants

| File | Holds | Read when |
|---|---|---|
| `CLAUDE.md` | entry point: what this is, what to read | automatically |
| `SESSION_STATE.md` | where we are right now, what is in flight | first, after any break |
| results ledger | every result in order, numbered | tail, every session |
| procedures/checklists | what to do, as checkboxes | before the action |

Keep the *narrative* of workflow lessons separate from the
*checklists*. They serve different moments: one is read for insight,
the other is read before pressing the button.

## After a compaction

- [ ] Read the session-state file, then the tail of the results
      ledger.
- [ ] Skim the checklists before the first delegation or heavy job.
- [ ] **Re-check which in-flight jobs still exist** rather than
      assuming they all do. Verify, do not infer.

## Durability of in-flight work

Anything whose result matters should write to a **durable file on a
machine that is not the human's laptop**.

*Instance:* a crash on the human's machine destroyed the job registry
of a locally-orchestrated background task — the tool reported "no jobs
recorded" and the result was unrecoverable. Remote jobs launched in
their own session group, writing to files on a server, survived
untouched and were collected hours later.

Corollary: prefer remote-plus-output-file over any tool whose state is
held by the local session, for anything you would be annoyed to lose.

## Recording as you go

Write the result down when you get it, not when the session ends. A
session that ends unexpectedly loses exactly the material that was
still "about to be written up" — which is always the newest and most
valuable.
