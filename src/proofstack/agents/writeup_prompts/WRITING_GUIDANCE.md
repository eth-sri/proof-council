# Writing Guidance for Research Mathematics

## 1. Accuracy and scope

Improve the exposition of the supplied manuscript while respecting the task for
this pass. These are writing principles, not a request to expand the research.

- **In a rewrite**, preserve the existing argument, hypotheses, conclusions,
  quantifier order, implication direction, and dependencies. Do not substitute a
  different proof or silently repair a suspected error. Retain precise wording
  when a smoother version risks changing its meaning.
- **In a referee-driven repair**, make the minimal targeted corrections requested
  by the repair task, not another general rewrite. Explain any necessary change
  to a claim's scope; do not present a weaker or conditional result as a proof of
  the original claim. Follow the task's reporting instructions for errors you
  cannot fix.

Transitions and explanatory prose carry mathematical claims too. Check them as
carefully as formulas. Do not assume the input is correct merely because this is
an editing pass, or describe an unchecked claim as verified.

There is no human editor in this workflow. Record unresolved concerns, missing
justifications, and unverified information as LaTeX comments beginning
`%% FLAG:` at the relevant point. State what needs checking rather than inventing
an explanation. Preserve existing flags unless the issue has actually been
resolved; a flag does not turn a gap into a proof. Keep assumptions and
limitations visible in the mathematical text, not only in comments.

Read each paragraph as an expert who does not yet know where the argument is
going: what are we doing, why is this step needed, and what do its terms mean?

## 2. Reader and organization

Write for an expert in the broad area who is unfamiliar with this problem.
Assume standard background, while allowing for forgotten details of cited
results. Adjust the level of explanation to the manuscript's audience and genre.

### Introduction and abstract

Open with the concrete problem. State the main result and its significance
early, with the definitions it needs. Explicitly label any informal statement
and keep it faithful to the precise theorem. Sketch the proof's strategy, main
difficulty, and central construction or mechanism, and explain how the sections
contribute. For example:

> The proof proceeds by [strategy]. The main difficulty is [obstruction], which
> we overcome by [key construction]. Sections 3-4 establish [the needed
> properties], and Section 5 uses them to [finish the argument].

Keep the abstract brief. Describe what was open, what is now established, why
it matters, and how it is checked, insofar as the manuscript supports these
claims. Use familiar concepts; reserve technical definitions, precise
statements, and displayed objects for the introduction. Avoid undefined
notation and a summary of every part of the project.

### Introducing concepts

Introduce objects, symbols, terms, and results before using them. Gloss
specialized terms on first use, with a citation or section reference where
appropriate. Standard textbook terms need no explanation for readers who know
them. For an algebraic geometry audience, for instance, "vector bundle" may
need none, while "logarithmic Chow ring" may need a brief description.

Introduce an abstraction through an example when that helps; otherwise define
it first and then give examples. Use non-examples to clarify boundaries and
worked examples to explain methods. Defer details only when the description
already given supports the argument at that point. Supply any internal details
needed to follow it.

### Sections and dependencies

Explain the purpose of sections and constructions before developing them. At
their conclusion, make clear what can now be used and which details can be set
aside. A short introduction can explain why terminology is needed: "To state
our main result, we need the following notion ..."

Keep dependencies local. Place supporting lemmas near their use, preferably in
the order needed, and finish one line of argument before opening several others.
Separate lengthy routine calculations and technical checks into lemmas,
remarks, or appendices. Keep essential assumptions and unresolved gaps explicit
where the proof depends on them.

Distinguish the main result from supporting tools, routine checks, and optional
remarks. Number statements cited later. Use separate lemmas for reusable
results, distinct ideas, or lengthy verifications whose conclusions can be used
independently. Combine short observations serving one subgoal.

## 3. Proof exposition

### Strategy and choices

Begin substantial proofs with their strategy: "by induction on $n$," "by
contradiction," or "we construct the map explicitly." Identify cases as they
arise. After a long calculation, state what it establishes and how it advances
the argument.

Explain non-obvious choices and give the conceptual step before its verification:

> We compare both solutions to the same frozen-coefficient model; the triangle
> inequality then gives ...

For an auxiliary construction, state what it must achieve:

> We want a quantity bounded below by $A$ and controlled by $B$, which leads us
> to consider ...

Present the proof in a forward sequence, preserving the insight behind choices
discovered by working backward. Omit discovery history unless it helps explain
the proof. If the argument chooses $\varepsilon = A^{-3/7}$, explain which terms
this balances. Give the reason supported by the argument or supplied context,
not a plausible but invented account of the author's intentions.

Make nontrivial reductions explicit. For steps such as "we may assume,"
"choose suitable coordinates," or "perturb generically," explain why the
construction exists, which hypotheses, bounds, and other relevant properties
it preserves, and how the conclusion transfers back to the original problem.
Include any limiting step on which that transfer depends. Expand justifications
supported by the argument or supplied sources; if a required justification is
missing, flag the gap rather than inventing one during the rewrite.

### Hypotheses and equations

Identify where important or surprising hypotheses are used, so readers can
assess sharpness and possible generalizations. State dependence on dimension,
regularity, constants, and genericity, together with known limitations and
obstructions.

Display formulas when their role or complexity warrants it. Explain the purpose
and consequence of important displays. Annotate nontrivial steps with the
relevant hypothesis, result, or property, for example "by the induction
hypothesis" or "because the supports are disjoint." Routine rearrangements need
little commentary.

### Detail and economy

Allocate detail according to difficulty and novelty, keeping the reader's
reasoning load reasonably steady. Expand delicate steps and unfamiliar ideas.
Include intermediate steps and motivation when their omission could leave a
reader stuck.

Make compressed work reconstructible through a precise citation, a specified
calculation, or an explanation of how an earlier argument changes:

> Repeat the proof of Lemma 3.2 with $L^2$ replaced by $L^p$, Hölder's inequality
> being the only changed step.

Replace unsupported "clearly" or "standard" with a reason or reference. For
analogous cases with substantive bookkeeping, do not rely on "similarly" or
"by symmetry" alone. State the transformation or relabelling and explain why
it preserves the relevant hypotheses, constructions, and claimed conclusion,
including signs and endpoints. If no such symmetry applies, explain the changed
steps or write out the additional cases. Do not replace a missing argument with
an unsupported symmetry claim.

Cut ceremonial transitions, repeated summaries, generic praise, and decorative
generality. A signpost should explain a purpose or consequence, not merely
announce that the text continues. Concision must not remove substantive steps,
necessary motivation, or qualifications.

## 4. Notation and terminology

### Symbols and scope

Follow conventions in the relevant literature. Use symbols consistently: one
symbol per object, distinguishable notation for unrelated objects, and
recognizable relationships between related ones. Prefer mnemonic choices
without conflicts with established meanings.

Minimize indices, subscripts, names, and acronyms. Describe a short-lived object
in words or write out a constant when a symbol adds little. Keep global notation
stable and temporary notation local, with a clear scope. When returning to an
object or result after a long gap, recall its role and any formula needed:

> Recall that $K$ is the compact core from Proposition 3.1.

### Names and conventions

Check for an existing term before coining one. Loci where cohomology ranks
change, for example, are called *cohomology-jumping loci*. For a new notion,
choose a descriptive name that avoids misleading associations with neighboring
concepts. Words such as "chamber" and "wall" already have specific meanings
near moduli and wall-crossing theory; reusing them for different auxiliary loci
can mislead readers. Check symbols and letters for similar conflicts.

Keep terminology stable. Cycling through "map," "operator," and "morphism"
can suggest unintended distinctions.

## 5. Sentences and statements

### Sentence structure and references

Match connectives to the logic: "since" gives a reason, "therefore" a consequence,
"however" a contrast, and "in particular" a specialization. Reserve "equivalent"
for genuine equivalence.

Use active, concrete phrasing and complete sentences, with displayed equations
punctuated as part of the prose. Prefer a descriptive noun before a symbol at
the start of a sentence. Keep subjects and verbs close together; move lengthy
hypotheses to the beginning or end. Give pronouns clear antecedents: "this
estimate" or "the first inclusion" identifies what "this" refers to. Use
numbered references that survive reorganization; "the preceding theorem" can
become incorrect after a section moves.

### Definitions and theorem statements

Identify each assertion's status: assumption, definition, recalled fact,
earlier result, conjecture, heuristic, or claim being proved.

A **definition** should introduce one concept, state any hypotheses explicitly,
and leave additional results to separate statements. Italicize the defined term
and explain dense wording in plain language.

Give definitions an operational meaning: specify what is counted, compared, or
constructed, where or at which stage it is evaluated, what varies, and which
objects remain fixed. Replace vague claims of "correspondence" with an explicit
map or equivalence. State its domain and codomain, or the two conditions being
compared, and justify the properties the argument uses. Do not turn a one-way
implication or a non-bijective map into an equivalence or bijection by rewording.

Make **theorem statements** independently usable. Give hypotheses before
conclusions, specify quantifier order and constant dependencies, and ensure all
symbols have been defined. Replace vague references such as "where everything
is as above" with the information needed to apply the theorem. Put history and
commentary outside the statement.

Address relevant **boundary cases** explicitly, even if a clause suffices: a
zero index, empty set, zero map, or other degenerate case.

## 6. Sources and credit

### Citing results and ideas

Identify imported theorems precisely, including a number or page where feasible.
State the part used, reconcile conventions, and check the hypotheses. Credit
borrowed constructions, ideas, strategies, and proof structures, including
adaptations from other settings, even when no external theorem is invoked.
Preserve attribution, bibliography entries, and provenance needed to trace the
argument when reorganizing or repairing it. An unverifiable source calls for a
flag, not the silent deletion of its credit or replacement by invented details.

### Novelty and unverified claims

Describe novelty through checkable comparisons: a removed assumption, improved
exponent, uniform estimate, or new construction. Acknowledge limitations and
incomparable regimes. Replace praise with concrete consequences: "This removes
the compactness hypothesis of [X]" or "The estimate is uniform in dimension"
explains the advance more precisely than "Our approach is substantially more
general."

Do not invent citations, theorem numbers, priority claims, history, motivation,
or interpretations. Distinguish what has been checked against a source from
what the manuscript merely reports. Flag unverified information and missing
explanations; do not fabricate them to make the exposition read smoothly.

## 7. Computer-assisted and formalized arguments

Lead with mathematical statements and explanations. File paths, tactic names,
and certificate labels are pointers, not substitutes for the argument. Put them
in a parenthetical, footnote, or paper-to-code correspondence table when useful.

Keep reproduced code faithful to the supplied source version; check listings
against that source when available and label readability edits. If the source
is unavailable, flag the limitation rather than claiming to have checked it.
Do not invent code executions, outputs, certificates, or successful checks.

State precisely what is machine-checked, what is assumed, and what is cited but
not formalized. Distinguish finite computations, numerical or heuristic
evidence, and proofs covering all cases. Explain the connection between a
computation or formal statement and the theorem it supports; do not extend its
scope through an informal use of "verified."

Keep a concise account of methods and provenance in the body. Move lengthy
prompts, timings, and tool logs to an appendix or supplement when appropriate,
without losing information needed to assess or reproduce the result.

## 8. Revision checks

For a rewrite, work in passes on structure, explanation, notation, and prose,
revisiting earlier sections as needed. For a targeted repair, apply these checks
to the changes and their dependencies without reopening unrelated exposition.
Check mathematical fidelity throughout and at the end. Resolve cross-references
after reorganization.

- **Overview:** Can a reader summarize the problem, result, proof idea, and
  significance after the abstract and introduction, and identify the prior
  state of knowledge and the main limitation?
- **Structure:** Do headings, statements, and paragraph openings give a coherent
  outline? Does the prose explain the argument's purpose when displays are skipped?
- **Statements:** Are major statements unambiguous without their proofs, with
  hypotheses, quantifiers, and dependencies intact? Do definitions identify what
  varies, what stays fixed, and what is evaluated or counted?
- **Proof strategy:** Can each substantial proof be summarized in a few steps,
  with its key choices explained and compressed work reconstructible?
- **Reductions and symmetry:** Are the transformations, preserved properties,
  and transfer back to the original problem justified rather than assumed?
- **Notation and prose:** Are later uses of symbols clear? Could a pronoun,
  "respectively," "similarly," or an overloaded sentence admit another
  mathematical interpretation? Does reading aloud reveal missing connections
  or repetitive phrasing?
- **Integrity:** Are credit, limitations, computational scope, and unresolved
  flags preserved? Does any new explanation assert more than the sources support?
