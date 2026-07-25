# Adversarial rounds

## Attack your strongest result, not your weakest

The instinct is to red-team the shaky thing. It is the wrong instinct.
A round aimed at a result you are already confident in is where
refutation is most informative, because you will act on it — and
because a construction that survives a determined attack becomes
quotable in a way that an unattacked one never is.

*Instance:* the most valuable round of one project was commissioned
against a construction the team believed in. It refuted two of their
conjectures, refuted their proposed termination invariant, supplied a
counterexample showing that a well-definedness property they had been
treating as automatic was not, **and** handed back two theorems they
had not asked for — while confirming the main results.

Give such a round the strongest version of the claim plus all
supporting data. A red team working from a weak statement finds weak
objections.

## Read the objections by testability, not by depth

Sort what comes back by **how cheaply it can be settled**, not by how
profound it sounds. A shallow objection you can answer today with a
computation is worth more than a deep one you cannot act on, because
it converts into a stated theorem or a corrected definition
immediately.

Typical yield from one round, in descending value:

1. **Objections that are decisive and cheap.** Run them the same day.
2. **Gifts** — things the reviewer proves in passing that you had not
   asked about. Adopt them, with attribution in your notes.
3. **A strategic critique** of whether the work matters at all. Record
   it; do not argue with it in your head. It is the thing a referee
   will say.
4. **Deep objections you cannot yet test.** Convert into the next
   packet's questions.

## Watch for the non-discriminating confirmation

A red team's most useful move is often to point out that one of your
confirmations had no power to fail. If a symmetry forces a unique
answer in the case you checked, then *every* rule in your family
returns that answer there, and your agreement with a known object is a
consistency check, not evidence.

Always ask, of a confirming case: **could this have come out
differently?** If not, find the first case that could, and report how
often the rule is actually exercised. That number belongs in the
paper — it is the honest measure of what the definition is claiming.

## Commission the reviews you would not enjoy

- Certificate review: is the test *sufficient* for the conclusion?
- Prior-art review: has someone done this, under another name?
- Strategic review: why would anyone in this field care?

The third is the one most often skipped and the one most likely to
change what you do next.

## Point the review at the certificate, not the code

The highest-yield review question is not "is this code correct" but
**"is this test sufficient for the claim it is being used to
support"**. Those are different questions and they fail differently.

Reviews framed that way, run by a model family different from the one
that wrote the code, produced the two most valuable findings of a
multi-week investigation:

- a verification that rejected a case only when a quantity was
  *negative*, silently accepting the boundary value zero — which was
  exactly the case that falsified the theorem. The code was correct;
  the certificate was too weak by one boundary case, and had been
  reporting a clean pass for weeks.
- a bijection whose count was right and whose indexing set was wrong
  (see `verification.md`).

Neither is the kind of thing self-review finds, because both are
invisible from inside the framing that produced them. Give the
reviewer the exact claims to invalidate, in the client's own words,
and ask for a minimal input that would expose each defect.
