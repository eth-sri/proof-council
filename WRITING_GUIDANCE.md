# Writing Guidance for Research Mathematics

*Consensus draft v3 — 4 July 2026.* A guide for turning a correct-but-dense mathematical write-up — often one produced by an AI system — into a paper that an expert will understand, trust, and enjoy. It synthesizes three independent proposals (two AI drafts and the core tenets of a working mathematician) together with the standard literature they draw on. The principles below are defaults, not laws: a deliberate exception is often good writing; an accidental one rarely is. Version v2 added, from experience applying the guide to a formally verified paper, targeted guidance on writing the abstract (§2) and on computer-assisted and formalized write-ups (§6). This version (v3) adds, from a naming collision observed in practice (auxiliary loci called "chambers" and "walls" in a field where those words mean something else), rules for coining terms and choosing notation against the literature (§6, §8, §9).

---

## 0. What this pass is — and the one rule above all others

You are given a write-up whose **correctness has already been checked**: the lemmas hold, the constants work out, the implications run the right way. Your job is not to re-verify the mathematics or to find new results. It is to make the manuscript *a pleasure for a human expert to read* — to remove friction, supply motivation, reveal structure, and let the ideas breathe.

One rule governs everything below.

**Never let presentation change truth.** Every edit must preserve the exact logical content: hypotheses, the order of quantifiers, the direction of each implication, the dependency structure, the precise statement of each result. A smoother phrasing that quietly weakens a hypothesis or strengthens a conclusion is wrong, however elegant. This failure has a name — *semantic smoothing* — and it is the one mistake a writing pass must never make. When you cannot improve the prose without risking the mathematics, leave it and flag it.

Two corollaries:

- **You are an editor, not a co-author.** Respect the existing argument. Do not substitute your own proof because you find it cleaner, and do not silently "fix" a step you suspect is wrong — flag it instead. Correctness was someone else's job, and your suspicion may itself be the error.
- **Connective prose carries logic too.** Do not assume the mathematics is untouched merely because no theorem environment was edited. A rewritten transition can change what is asserted. Treat every claim-bearing sentence with the same care as a formula.

A test for the whole pass: **read each paragraph as a skeptical, intelligent expert who does not yet know where the argument is going.** At every sentence ask — *Do I know why we are doing this? What role does this step play? Could I have guessed the next move?* Friction lives wherever the answer is "no."

---

## 1. Write for one reader — and never lose them

Halmos's first principle: **decide whom you are writing for, and write for a single specific person** — someone mathematically strong but outside your immediate context. For a research paper that reader is usually an expert in the broad area but *not* in this sub-problem. They know the standard tools; they have not spent the last six months inside this argument. So they will not recall the precise form of every cited result, will not see why a definition is shaped the way it is until told, can fill in routine steps once they know the goal, and will lose patience with unmotivated formalism. (Adjust the dial by genre: a survey assumes less and motivates more; a specialist note can assume more. But always write *to* a person, never *at* a record.)

From that reader follows the floor on which the rest of this document is built — the single most important requirement, and the one most often violated:

> **The orientation invariant.** At *every* point in the paper, the reader should be able to answer three questions: **what are we doing, why are we doing it, and what does every symbol and term in front of me mean?** Nothing — object, notation, or result — should be used before it has been introduced.

This is not a stylistic nicety; it is the contract that makes a paper readable at all. It has direct consequences:

- **Introduce before you use.** A reader who meets a symbol, term, or cited result before its introduction has already been thrown. Order the text so the reader is always equipped for the sentence in front of them.
- **Gloss every non-textbook term on first use.** Notions that are standard for the field need nothing — in algebraic geometry, "vector bundle," "fat point," or "Riemann–Roch" can be used freely. More specialized objects — a "logarithmic Chow ring," a "tropical curve" — need at least a half-sentence of explanation and, where appropriate, a citation or a pointer to the section where they are treated properly.
- **Forward-reference only with a working black box.** It is fine to defer the full treatment of a notion, *provided* the brief description you give is enough to follow the logic where it appears. If the reader needs the internals to follow the argument, supply them first, not later.
- **Recall what you rely on.** When you invoke notation or a result from many pages back, remind the reader what it was. The author has the whole paper loaded in memory; the reader has only what you have recently put there.

---

## 2. State the destination before you demand the journey

Readers tolerate technicality when they know what it buys. So tell them where they are going before you make them walk.

- **Front-load the result.** Say what you prove and why it matters before how you prove it. State the main theorem precisely — or in a faithful informal version, explicitly labelled as such — early, ideally within the first page or two, together with the nonstandard definitions it needs. A reader who knows the destination reads the journey as purposeful; one who does not is following instructions in the dark.
- **Give the idea of the proof.** This is the single highest-value addition to a dense draft. In a few sentences the reader can hold in mind: *the proof goes by [strategy]; the main difficulty is [X]; we overcome it by [key idea]; Sections 3–4 build the tools and Section 5 assembles them.* A real proof overview names the mechanism — the new object, the decomposition, the delicate step. "Section 2 contains preliminaries; Section 3 proves the theorem" tells the reader where the pages are, not how the mathematics works.
- **Preview before you build.** A recurring failure mode: a paragraph ends, and the next opens straight into a block of new notation and constructions, with the theorem they serve revealed only at the end. Put the orientation first — *"to state our main result we need the following notion ..."* — so the reader knows what the machinery is for *while* reading it, not afterward.
- **Pitch the abstract above the apparatus.** The abstract is the one place where the orientation invariant cannot be met by introduction — you have no room to define everything you would name. Resolve this not by cramming in definitions but by raising the altitude: state the *shape* of the contribution (what was open, what is now known, why it matters, how it is checked) in terms the reader already commands, and leave the precise statement, the central definitions, and every displayed object to the introduction. A reader should finish the abstract knowing what changed, not holding three undefined symbols — and an abstract that reads like an executive summary of the whole project is too long.

The same move repeats at every scale. Before a section, say what it accomplishes and why it is needed. Before a hard definition, say what phenomenon it captures. Before a long proof, give its plan. At each scale the reader should be able to ask not only *where are we* and *what is happening*, but *why are we here* and, on leaving, *what may I now use or forget*.

---

## 3. Design the route, and make its structure visible

Logical validity does not fix a unique order, and the order in which a result was discovered or verified is rarely the order that is easiest to read. That freedom is where exposition is won or lost.

**Order for the reader's understanding.**
- *Motivate, then formalize.* Introduce a concrete or familiar instance before an abstraction that would otherwise look arbitrary; conversely, when examples-without-the-definition would be a grab-bag, define first and then give a rich spread of examples. Choose whichever earns understanding faster.
- *Question before answer.* Let the reader feel the need that a definition or lemma fills before it arrives, so it reads as inevitable rather than as a rabbit from a hat.
- *Keep dependencies local.* A reader should not have to hold five open threads to follow the sixth. State the lemmas a proof needs just before it, in the order used; close loops before opening new ones.
- *Defer the routine.* Move calculations, routine verifications, and side conditions into clearly marked lemmas, remarks, or appendices so the spine of the argument stays visible — but never hide a load-bearing assumption or a genuine gap in an appendix.

**Make the architecture visible.** A human reader, unlike a proof assistant, has no display of the current proof state; you must supply it.
- *Name and label* the objects and results you will refer to. A result invoked later should be a numbered statement, not a sentence buried mid-paragraph.
- *Use lemmas as interfaces, not as confetti.* Promote a step to a lemma when it is reused, isolates a real idea, hides a long verification behind a usable statement, or lets the reader discharge details from memory. Do not atomize a single thought into a dozen one-line lemmas; that inflates apparent complexity and forces the reader to reassemble the argument from fragments.
- *Signpost relentlessly.* Open each proof by announcing its strategy ("by induction on $n$," "by contradiction," "we construct the map explicitly"). At each branch, say which case you are in. After a long computation, say what was achieved and how it feeds the next step. These markers cost a sentence each and save the reader minutes.
- *Connect the blocks.* Theorems, definitions, and proofs should be joined by prose, not stacked like bricks: a sentence on why a definition was made, on how one lemma sets up the next, on what a theorem buys. A manuscript of flawless displayed environments with no connective tissue reads as a list, not an argument. The transitions are where the story lives.

**Reveal the hierarchy.** Not every statement deserves equal weight. Make plain which result is the main theorem, which lemmas carry reusable ideas, which estimates are routine, which examples are diagnostic, which remarks are optional. A paper that gives every fact equal prominence reads like a database, not an argument. Use the visual apparatus — sectioning, naming, the occasional figure or diagram — to carry this hierarchy, not just to break up the page.

---

## 4. Explain ideas, not just valid steps

A proof can be formally complete and still opaque. The most damaging defect of correct-but-dense writing is the **unmotivated move**: the clever substitution, the auxiliary quantity pulled from nowhere, the inequality applied with a foresight the reader does not share. The reader asks "where did *that* come from?" and gets no answer.

A motivated proof lets the reader see, at each step, *what is being attempted and why this is a reasonable thing to try.* You need not narrate the history of discovery, and you must not pad. A short phrase usually suffices: *"to exploit the symmetry of ...," "the obstruction in the previous attempt suggests ...," "we want a quantity bounded below by $A$ and controlled by $B$, which leads us to consider ...."* The target is that the reader thinks **"I could have come up with that"** — the signature of excellent exposition.

Concretely:

- **Distinguish the idea from the verification.** When a calculation merely implements a simple conceptual move, state the move first: *"we compare both solutions to the same frozen-coefficient model; the triangle inequality then gives ...."* Now the reader understands the computation instead of merely auditing it.
- **Present the argument forward.** Discovery often runs backward — guess the bound, ask what would imply it, tune a parameter until the algebra closes. The clean version usually runs forward, while still explaining the insight behind a non-obvious choice. Never write "Choose $\varepsilon = A^{-3/7}$" without saying what is being balanced; a magic constant is not made less magic by being correct.
- **Show where each hypothesis enters.** For an important or surprising assumption, say where it is used. Readers understand a theorem largely through the roles of its hypotheses, and this lets them judge sharpness and possible generalizations.
- **Explain the equations that matter.** A displayed formula should earn its display. Say what it will establish before it and what it means after, and annotate the nontrivial steps of a chain ("by the induction hypothesis," "because the supports are disjoint") — but not every routine rearrangement.

This is exactly the layer that a draft optimized for *finding or checking* a proof tends to omit. Such writing is typically complete but motivationally empty: every step is justified as *valid*, none as *natural*. Supplying the missing "why" — without altering a single "what" — is the heart of the pass.

---

## 5. Pace the difficulty evenly; spend words where the ideas are

Aim for roughly even **inference distance per paragraph** for the intended reader — a steady thinking load, not a flat word count. A common and jarring failure is uneven pacing in the wrong direction: standard constructions everyone knows are expanded in loving detail, while the genuinely novel and intricate steps are dispatched in a sentence. Reverse that. Spend words in proportion to difficulty and novelty; let the one clever step breathe and compress the routine $\varepsilon$–$\delta$ check to a line. Uniform density is itself a defect, because it tells the reader nothing about where to concentrate.

This raises the one genuine tension between concision and completeness. Resolve it by distinguishing two things:

- **Be generous with mathematical content and motivation.** When in doubt about whether to include a *substantive* step, a motivating sentence, or a clarifying intermediate, include it. For the reader, skimming past detail they do not need is cheap; getting stuck on a step that was silently skipped is expensive. More detail, here, is the safer error.
- **Be ruthless with filler.** Ceremonial prose ("we now proceed to prove the following result"), restatement of the obvious, ritual generality with no payoff, and decorative formalism add length without information. Cut them without mercy.

Concision, in other words, is not fewer words — it is no *wasted* words. When you do compress real mathematics, **compress by delegation, not by silence.** "A routine computation gives ...," "repeat the proof of Lemma 3.2 with $L^2$ replaced by $L^p$, Hölder's inequality being the only changed step," or a precise citation lets a qualified reader reconstruct what was omitted. A bare "clearly" or "it is standard" that names no reason and no result is a gap wearing the costume of confidence; replace it with the reason or the reference. And give examples their space when they make an abstraction concrete, mark a definition's boundary (an example and a non-example), or show a method before its general statement.

---

## 6. Notation, prose, and logical status

**Notation is infrastructure — design it, don't accrete it.**

- Choose symbols deliberately and use them consistently; reusing a letter for two things, or two letters for one thing, reliably frustrates. Pursue *alphabetical harmony*: related objects look related, unrelated objects look different, and the choices are mnemonic and free of collisions with entrenched conventions.
- Keep a notation budget. Every symbol is a tax on memory; prefer fewer indices and subscripts, inline a constant used once, and resist a symbol you will use only twice — often words are clearer ("the best notation is no notation").
- Make scope visible: global notation stays stable; temporary notation is introduced locally and allowed to expire. After a long gap, recall an object's *role*, not merely its formula ("recall that $K$ is the compact core from Proposition 3.1").

**Name into the literature, not against it.** Terminology and notation are decoded through the reader's prior associations; check those associations before coining anything.

- Do not mint a symbol or a name for an object that lives only in a few adjacent equations or sentences; local objects deserve plain words, not designations. (The notation-budget rule, extended from symbols to *terms*.)
- Choose letters and symbols that follow the standard conventions of the relevant literature; a correct but nonstandard letter costs every expert reader a translation step, and the cost recurs at every use.
- Before introducing a new term, check whether the notion already has a name in the literature — if it does, use that name (loci where cohomology ranks change are *cohomology-jumping loci*; no coinage needed).
- If the notion is genuinely new, pick a name free of conflicting associations with neighboring-but-different concepts. A term that pattern-matches an established notion in the same area — *chamber* and *wall* near moduli and wall-crossing theory, say — will be parsed as that notion by precisely the readers who know the field best. Test each candidate by asking what it already means to the intended reader, and prefer a plain descriptive phrase over an evocative loaded one.

**Write real sentences, even with symbols in them.**

- Every sentence has a verb and ends with punctuation; a displayed equation is part of a sentence and is punctuated as one. Never open a sentence with a symbol ("$x$ is even" → "the integer $x$ is even"), and never abut two formulas with only punctuation between them.
- Match the connective to the logic: "since" for a reason, "hence" or "therefore" for a consequence, "however" for a genuine contrast, "in particular" for a specialization. These words make promises; keep them. Reserve "equivalent" for genuine equivalence.
- Keep terminology stable. Variation that is elegant in literary prose creates phantom distinctions in mathematics: do not cycle through "map," "operator," "morphism" for one object merely to avoid repetition. Vary rhythm and sentence shape, not the names of things.
- Anchor every reference. "This," "the above," "the former" often have several candidate antecedents; attach a noun ("this estimate," "the first inclusion") and prefer stable numbered references over "the preceding theorem," which becomes false after reorganization.
- Prefer the active and the concrete, and keep subject and verb close enough to parse — move a heavy hypothesis to the start or end of the sentence rather than wedging it between them.

**State the logical status of every assertion.** The reader should always know whether a claim is an assumption, a definition, something standard being recalled, a previously proved result, a conjecture, a heuristic, or the thing now being proved. Ambiguity here is one of the most common and most avoidable confusions. The same honesty applies to limitations: state the regime of validity and the known obstructions plainly — dependence on dimension, regularity, constants, genericity. A limitation stated openly usually raises confidence in the result.

Finally, handle the set pieces cleanly. A **definition** defines one thing and smuggles in no hypotheses or results; set the term in italics and, when the formal wording is dense, add a plain-language paraphrase. A **theorem statement** should be usable on its own — hypotheses before conclusion, quantifier order unambiguous, constant dependence stated, all symbols already defined — with no "where everything is as above" that forces a hunt; keep history and commentary out of the statement and place them after it. And treat **boundary cases** honestly: dispatch the degenerate case ($n = 0$, the empty set, the zero map) in at least a clause, since silence there is both poor exposition and an occasional hiding place for a real gap.

**Computer-assisted and formalized write-ups.** A growing share of correct-but-dense drafts — especially AI-generated ones — come with a formalization, a body of code, or a machine-checked certificate behind them. That backing is a strength, but it leaks into the prose in characteristic ways.

- **Keep implementation identifiers out of the argument.** Lemma names, file paths, tactic names, and certificate labels are pointers, not mathematics. Lead each step with the mathematical statement and relegate the code identifier to a parenthetical, a footnote, or a single paper-to-code correspondence table. "Proved in `Foo.bar_baz`" explains nothing; if it stands in for the idea, the idea is missing.
- **Keep reproduced code faithful and current.** A listing copied into the paper is a claim about the source, and it rots silently when the source moves on. Re-derive it from the current source on each pass, mark it if it is lightly edited for readability, and keep its lines short enough not to wrap.
- **Document process in proportion to its interest, and quarantine it.** Provenance, prompts, run times, and tool logs are legitimately interesting for a computer-assisted result, but a reader who came for the theorem should not wade through more pages on *how* it was obtained than on *what* it says. Keep a short, candid account in the body and move the full record to an appendix or supplement.
- **State the trust boundary once, precisely.** Say exactly what is machine-checked, what is assumed, and what is cited but not formalized, and do not let an informal "verified" blur the three. A precisely drawn boundary raises confidence; a vague one invites the reader to suspect that more is claimed than is shown.

---

## 7. Integrity and credit

Trust is part of readability, and it has to be earned precisely.

- **Cite imported results like dependencies.** When you use a theorem from the literature, identify it exactly (statement, and page or number where feasible), state the part you use, reconcile any difference in conventions, and confirm that its hypotheses hold. A citation should shorten verification, not start a scavenger hunt.
- **Credit ideas, not only results.** Often a proof imports no external statement, yet its global structure or a central idea is a variation on something already in the literature, perhaps from another context. Crediting that lineage is not logically necessary, but it is right, and it keeps the web of citations connected so that ideas can be traced and followed. Give the credit even when you borrow only a strategy or a template.
- **State novelty in checkable terms.** Prefer "this removes the compactness hypothesis of [X]" to "our approach is substantially more general," and acknowledge incomparable regimes and limitations honestly. Overclaiming costs trust and may misrepresent the literature.
- **Replace praise with information.** "Interesting," "powerful," "natural," "important" carry weight only when the surrounding text says why. Name the concrete advance instead: the exponent is optimal, the construction is functorial, the estimate is uniform in dimension.
- **Never invent.** An AI editor must not manufacture a citation, theorem number, priority claim, or historical narrative, and must not invent the motivation or interpretation behind a step. If the reason for a construction is not present in the source material, say so and ask — do not supply a plausible story as though it were the author's. Mark anything unverified.

---

## 8. Characteristic failure modes of AI-generated mathematical prose

This pass exists largely because drafts optimized for finding or checking proofs share a recognizable set of weaknesses. Treat the table as a diagnostic, not as a claim that every draft has every fault.

| Symptom | Why it frustrates an expert | Repair |
|---|---|---|
| **Generic opening** — broad remarks on the importance of the field | delays the actual problem; interchangeable with any paper | open on the concrete object, tension, or limitation |
| **Use before introduction** — a symbol, term, or result appears before it is defined | breaks the orientation invariant; forces a backward hunt | reorder so the reader is always equipped; gloss on first use |
| **Proof transcription** — a valid sequence of steps with no visible strategy | reveals no idea; readable only by re-deriving it | add a proof map; name the key choice; separate idea from verification |
| **Motivational emptiness** — every step valid, none shown to be natural | the reader cannot see why any move was made | add a short "why this, why now" at each non-obvious step |
| **Fake or late motivation** — a generic reason attached after the machinery | explains nothing, and arrives too late | state the precise obstruction or use *before* the construction |
| **Uniform pacing** — routine and novel steps given equal detail | hides where the difficulty actually lies | compress the routine; expand the new and delicate |
| **Lemma atomization** — every observation is its own numbered lemma | inflates complexity; scatters the main line | merge fragments that serve one subgoal; keep lemmas that are real interfaces |
| **Notation inflation** — a symbol or acronym for every passing quantity | needless tax on memory | delete one-use symbols; keep notation local |
| **Colliding coinage** — a new term or symbol that already means something else nearby | the expert reads the established meaning and is silently misled | search the literature first; reuse the existing name, or choose an unloaded one |
| **Synonym churn** — one concept under rotating names | suggests distinctions that do not exist | fix one term per concept and repeat it |
| **Signposting spam** — "we now turn to ..." with no content | occupies space, conveys no reason | replace navigation with purpose and output |
| **Unearned "clearly" / "standard"** — a gap dressed as confidence | transfers work to the reader with no handle | give the reason, the theorem, or a recoverable reference |
| **Fake symmetry** — "the other case is analogous" when it is not | the omitted case may hide different signs or endpoints | name the symmetry, or write the changed step |
| **Overclaiming** — novelty or significance beyond what is shown | erodes trust; risks misrepresenting prior work | state a checkable comparison; acknowledge limits |
| **Semantic smoothing** — an awkward-but-precise hypothesis polished into an elegant falsehood | silently changes the mathematics | preserve exact content; recheck every claim-bearing edit |

---

## 9. Tests and checklist

**Read-back tests.** A pass succeeds only if it improves how a real reader can use the paper. Apply a few:

- *One-sentence test* — after the abstract and introduction, can a reader state the problem, the result, the main idea, and the significance in a sentence or two?
- *Five-minute test* — after the introduction, can they name the main theorem, the prior frontier, the new contribution, the central mechanism, and the main limitation?
- *Skim test* — reading only headings, statements, and first sentences of paragraphs, does a coherent route remain?
- *Theorem-only test* — read each major statement without its proof: are all hypotheses, quantifiers, and dependencies unambiguous?
- *Formula-blind test* — skip the displays: does the prose still convey the argument's purpose and progression?
- *Proof-compression test* — after a proof, can the reader summarize it in three to seven steps? If not, its architecture is not visible.
- *Notation-recall test* — at each late use of a symbol, could the intended reader still remember its role? If not, recall, rename, or localize it.
- *Adversarial-ambiguity test* — for every pronoun, "respectively," "similarly," and long sentence, is there a second plausible mathematical parse? Remove it.
- *Read-aloud test* — dense prose read aloud exposes overload, missing connectives, and monotonous cadence; AI passes especially tend to produce locally polished but uniformly shaped sentences.

**Checklist for the pass.** Work in passes rather than fixing everything at once; later context changes what earlier sections should say, so revisit.

*Global*
- [ ] Is the main result stated precisely and early, with its significance clear, and the proof idea sketched before the technical work?
- [ ] Is the order driven by the reader's understanding rather than by the order of discovery or verification?
- [ ] Does every section have a clear role, and are the major objects and results named so they can be referred to?
- [ ] Is the hierarchy visible — main result vs. tool vs. routine?

*Local*
- [ ] At every point, can the reader answer what / why / what-do-the-symbols-mean? Is anything used before it is introduced?
- [ ] Is every non-textbook term glossed on first use, with a citation or pointer where needed?
- [ ] At each clever step, is there a phrase saying what is attempted and why it is natural?
- [ ] Is notation consistent, harmonious, scoped, and minimal — and recalled when re-invoked after a gap?
- [ ] Is every coined term and chosen symbol checked against the literature — existing names reused, collisions with neighboring notions avoided, standard letter conventions respected?
- [ ] Does every sentence parse (verb, punctuation, no leading symbol), with connectives that match the logic and stable terminology?
- [ ] Is the status of every assertion (assumption / definition / known / conjecture / to prove) clear?
- [ ] Are definitions clean, theorems self-contained, and boundary cases handled?

*Pace and economy*
- [ ] Is detail proportional to difficulty — generous on the new, compressed on the routine?
- [ ] Is every compression reversible (delegated to a named result or argument), never a silent gap?
- [ ] Is filler — ceremony, restatement, decorative generality — cut?

*Integrity (check continuously, and last)*
- [ ] Does every edit preserve exact hypotheses, conclusions, and quantifier order?
- [ ] Are imported results cited precisely, and are borrowed ideas and templates credited?
- [ ] Is novelty stated checkably, with limitations acknowledged and no invented citations or history?
- [ ] Do all cross-references still resolve after reorganization, and are mathematics-sensitive edits flagged for recheck?

---

## 10. In one paragraph

Take a proof that is correct but reads like a verification transcript, and turn it into something a strong mathematician would enjoy. Write for one real reader, and never let them reach a sentence they cannot decode: at every point they should know what you are doing, why, and what each symbol means. Tell them the destination, and the idea of the proof, before the march begins. Order the material so that each step feels needed; signpost strategy, cases, and what was just achieved, because the reader has no proof-state display. At every clever move, spend a sentence on *why this, why now*, so the reader feels they could have thought of it. Pace the difficulty evenly — generous where the ideas are new, brisk where they are routine — cutting filler but never content. Design notation as a system, keep prose clean and terminology stable, and always mark what is assumed, known, or being proved. Credit the ideas you build on, and claim only what you show. And through all of it, change not one thing that is true: when elegance and correctness collide, correctness wins, every time.

---

## Further reading

The three proposals behind this draft converge on a small canon. Their shared message is that clarity comes not from simplifying the mathematics but from designing the order, hierarchy, notation, prose, and level of detail around the cognitive work a real reader must do.

- **Paul R. Halmos, "How to Write Mathematics" (1970)** — audience, organization, notation, honesty; write for one person.
- **Knuth, Larrabee & Roberts, *Mathematical Writing*** — the reader's state of mind, motivation, stable notation, proofs in memorable units.
- **Terence Tao, "On Writing"** — motivation, allocating detail, modular lemmas, separating the discovery draft from the polished one.
- **Francis E. Su, "Guidelines for Good Mathematical Writing"** — audience, complete prose, examples, iteration.
- **Bjorn Poonen, "Practical Suggestions for Mathematical Writing"** — ambiguity, quantifiers, theorem statements, precise citation.
- **Nicholas J. Higham, *Handbook of Writing for the Mathematical Sciences*** — a broad reference on style, notation, and revision.
- **George D. Gopen & Judith A. Swan, "The Science of Scientific Writing"** — reader expectations: context before new information, topic and stress positions.
- **Brett Mensh & Konrad Kording, "Ten Simple Rules for Structuring Papers"** — one central story at multiple resolutions; context–content–conclusion.
- **AMS Style Guide** — editorial conventions and mathematical typography.
