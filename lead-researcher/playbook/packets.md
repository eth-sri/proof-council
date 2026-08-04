# Consultation packets

Deep single-shot reasoning is delegated to a browser-run frontier model
via **self-contained prompt packets**. This is the core technology of
the mode; treat packet quality as you would treat proof quality.

## A packet is a folder, not a file

```
prompts/<ID>_<topic>/
    instructions.txt    the brief
    <artifact>.pdf      the report/data AS OF DISPATCH — frozen copy
    answer.md           the returned response (added on return)
```

**Freezing the artifact is not optional.** The report changes between
rounds, so a path to "the current report" is a moving target: six
weeks later nobody can tell what the consultant actually read, and a
disputed claim becomes unresolvable. Copy it into the packet folder at
dispatch.

*(Learned the hard way: a packet went out referencing a report that
had been rewritten twice since the previous round.)*

## Dispatch checklist

- [ ] `instructions.txt` is **fully self-contained** — role, notation,
      definitions, established results, questions. The consultant has
      no memory and no other context.
- [ ] The **dead-end ledger** is included verbatim: every claim this
      project has refuted, stated precisely, one line of counter-
      example each. Cumulative across all rounds, never pruned.
- [ ] An **honesty protocol**: label every claim PROVED / SKETCH /
      COMPUTED / SPECULATION; refuting the client is a first-class
      deliverable; print verification code inline.
- [ ] A **goal ladder**, easiest first, with "a fully proved weak
      statement beats an ambitious sketch" stated explicitly.
- [ ] **Planted ideas**: your own best conjecture, marked "verify
      first, then use or refute". Expect to be wrong; refutations come
      back with proofs and correct replacements, which is cheap.
- [ ] Tell the consultant what compute you can run, so it can specify
      finite subtasks precisely instead of skipping them.
- [ ] Copy the artifact into the folder.
- [ ] Commit before dispatch.

## Return checklist

- [ ] Save the response as `answer.md` in the packet folder.
- [ ] **Verify every mathematical claim independently before adopting
      anything.** Machine-check what is finite, hand-check what is not.
- [ ] Record the round in the results ledger: what was adopted, what
      was refuted, what was "gifted" (things you had not asked about).
- [ ] Any claim of *yours* the round refuted → new entry in the
      dead-end ledger.
- [ ] Update the session state file.

## What makes a packet worth sending

Send what needs deep focused reasoning **and** what you can check. Do
not send what you can do faster yourself, and do not send what you
could not evaluate if it came back wrong.

Ask for the reason, not just the fact. A round that returns "yes, 26
of 26" is worth less than one that returns why — and the second is
usually obtainable by rewriting the question.

## Retargeting a packet mid-flight

If your own experiments settle a packet question before you dispatch
it, **rewrite the question rather than deleting it**: change "test
this" to "explain this". A consultant asked to explain a verified
phenomenon does better work than one asked to test a conjecture,
because it cannot spend the round doubting the premise.
