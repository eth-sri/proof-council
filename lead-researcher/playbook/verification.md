# Verification discipline

What "verified" is allowed to mean, and how to keep it honest.

## Exactness

Anything that enters the written record is computed in **exact
arithmetic** — rationals, integers, exact LP/ILP solvers — never
floating point. A floating-point coincidence is not evidence, and the
cost of exactness is almost always affordable on the sizes that matter.

## Print the invariants

Next to every result, print the quantities that *cannot* be wrong:
group orders, orbit–stabilizer products, row and column totals,
parities, dimension counts. Then treat any violation as blocking.

This is the cheapest bug detector available. Two real bugs found this
way in one project: an automorphism group computed as order 0 (the
identity is always an automorphism — the search had assumed the basis
vectors were minimal, valid only for root lattices), and a census
table whose rows did not sum to a known total. Both would have reached
the paper.

## Scope your claims

State exactly what was checked.

- "Each of 53 orbit representatives is smooth" ≠ "the fan is smooth".
- "Verified at g ≤ 5" ≠ "true".
- "0 counterexamples found under a cap of N" ≠ "no counterexamples",
  and the cap belongs in the sentence.

When a verification is skipped or capped, **say so in the output**, in
the notes, and in the paper. Silent truncation reads as full coverage.

## Limitations are results

Record what you could not check and why, as a numbered entry alongside
the successes. An honest "this is infeasible with our method, and no
conclusion depends on it" is worth more than silence, and it stops the
question being re-opened every session.

## The "no difference" trap

When an experiment compares two rules and finds them identical across
n independent trials, the weak reading is *robustness*. Resist it. Ask
**why** they agree; the answer is usually a stronger statement that
also explains the cases where other rules differ.

*Instance:* two candidate objective functions selected the same point
in 26 of 26 cases. Filed as robustness, that is a footnote. The
structural reason turned out to be that the point is the minimum of
the relevant lattice set in the cone partial order — hence minimal for
*every* positive objective, which removed the objective from the
definition entirely and identified the underlying object as
Gorenstein. Same data, different question.

## Independent replication beats another self-check

A subagent or a second tool re-deriving a central computation in
architecturally different software is stronger evidence than another
script of your own. This only becomes possible if you **export clean,
documented data early**, in a standard format. Do that before you need
it.

Scope the replication claim precisely when you report it.

## Reviewing the certificate, not just the result

When a claim rests on one script's certificate logic, get that logic
attacked specifically — is the test *sufficient* for the conclusion?
Does the feasible region match the mathematical set? Is the lattice the
right one, or a finite-index sublattice? Are exceptions silently
converted into passes? These are different questions from "is the code
buggy", and they are where certificates actually fail.

## A count cannot validate an indexing set

If you claim "the objects are in bijection with such-and-such data",
counting both sides is *not* enough, and the failure mode is subtle:
you can count the right number while counting the wrong thing.

*Instance:* a classification asserted a bijection between 32,320 cones
and triples $(M, D, \sigma)$, and "verified" it by a table of seven
products summing to exactly 32,320. The products were right; the
indexing set was wrong. $D$ did not determine one of the components,
so there were only 22,080 distinct triples — the count had implicitly
enumerated a *finer* datum than the one named in the statement. No
amount of recounting would have caught it.

The guard is to **exhibit the inverse map explicitly**: say, for each
object, how to read the datum back off it. If you can do that,
injectivity is immediate and the count then genuinely finishes the
argument. If you cannot, you do not have a bijection yet.

## Distinguish evidence from free lunches

Some properties follow automatically from the framework you are
working in, and confirming them tells you nothing about the specific
construction. Before citing a property as evidence of naturality, ask
whether *any* candidate in the same class would have it.

*Instance:* a construction was supported by pointing at its good
behaviour under restriction to boundary strata and under products.
Both turned out to follow formally from the construction being
equivariant — every candidate rule had them. Meanwhile the one
genuinely discriminating case had been checked in a setting where a
symmetry forced a unique answer, so every rule agreed there too.

Failing a free lunch is informative; passing one is not. Keep the two
lists separate in your own notes, and state which is which when you
write up.

## Small cases can be coincidences — pick the test case that breaks them

A clean combinatorial picture in the smallest interesting case is
weak evidence, because small parameters satisfy accidental identities.
Before generalising, find the arithmetic coincidence the picture might
be resting on, and test at the first parameter that breaks it — which
is often *not* the next parameter.

*Instance:* a beautiful classification at $g = 5$ rested on $g - 1 =
4$, an identity true only there. Testing at $g = 6$ would have been
misleading, since $g = 6$ fails for an unrelated parity reason; the
informative test was $g = 7$, where parity is favourable and the
coincidence is exactly what breaks. Identify the coincidence first,
then choose the test.
