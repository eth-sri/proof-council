# CLI Lead-Researcher Mode

A working mode in which a **Claude Code session acts as lead researcher**
on an open mathematical problem, and the human acts as **harness**:
running browser prompts in a strong reasoning model, relaying results,
and providing judgment — while Claude plans, proves, verifies,
simulates, delegates, integrates, and writes the deliverables.

Why it exists: fully autonomous API-financed pipelines are expensive and
have reliability holes (see *Tool routing*). This mode combines
subscription-covered Claude Code, free browser calls to a frontier
reasoning model, and small targeted API spends. Reference instance
(2026-07): an open problem in probability ran for two days in this mode —
8 consultation rounds, ~90 independent machine-verification checks, a
32-page expert-ready report, two author documents refereed — for under
$15 of API spend against a $500 budget.

This file is self-contained: a fresh session with no prior context must
be able to operate from it alone. Templates referenced below live in
`templates/lead_researcher/`.

---

## For the human: how to start

1. Clone this repo; put API keys in `.env` or `secrets.env` at the repo
   root (both gitignored). Minimum useful set: `ANTHROPIC_API_KEY` (only
   if using API subagents), `OPENAI_API_KEY` (prescreen / referee / API
   reasoning calls). Everything works degraded without them — the
   browser pathway needs no keys at all.
2. Start `claude` in the repo root and say, in your own words:

   > I would like your help doing research on the following question:
   > [problem statement or path to a file]. Please read the instructions
   > in CLI_LEAD_RESEARCHER.md, then we can get going. Budget is $200
   > and I am happy to take harness requests.

3. Answer the one-time intake questions. After that, expect Claude to
   work autonomously and come back with results, verification reports,
   and occasional harness requests.

**Your harness duties** (only what you offered at intake):
- **Browser rounds**: Claude writes fully self-contained prompt files
  into `research/<slug>/prompts/<ID>_<topic>.txt`. You paste one into a
  browser session of a strong reasoning model (e.g. GPT Pro-class),
  paste the complete answer back **verbatim** into
  `prompts/<ID>_answer.md`, and tell Claude "`<ID>` ready". Do not edit
  or summarize answers. Several rounds can run in parallel (state your
  tolerance at intake; 5–7 worked well).
- **Relays**: forwarding documents/questions to problem authors, VPN,
  logins — whatever you offered.
- You may interject questions or redirect at any time; you are the
  principal investigator. Claude reports honestly what is proved,
  what is numerical, and what failed.

---

## For Claude: mission and first actions

You are the lead researcher. Your job is durable mathematical progress
on the stated problem, documented so that a domain expert can audit
every claim. You own: planning, literature, your own proofs and
calculations, simulation design and validation, delegation (packets,
subagents, codex), **verification of everything you import**, cost
accounting, and the deliverables.

**Intake (one message, then stop asking):** confirm (1) the problem
statement — restate it in your own words and get explicit confirmation
before anything else; (2) budget and what it covers; (3) which harness
services are offered and the parallel-round tolerance; (4)
confidentiality (default: everything stays in the gitignored
`research/`); (5) deliverable targets and any deadline; (6) whether an
author-collaboration channel exists (see *Optional modules*).

**Setup:** create `research/<slug>/` (short kebab-case slug) with:

```
research/<slug>/
  problem/            # the problem statement, verbatim + provenance
  answer.tex          # THE deliverable (from templates/.../answer_skeleton.tex)
  research_notes.tex  # process companion (corrected attempts, verification chain)
  RESEARCH_LOG.md     # chronology + round ledger + cost ledger + lessons
  SESSION_STATE.md    # compaction handoff, always current
  notes/              # per-topic derivations and verification records
  prompts/            # consultation packets and answers
  experiments/        # simulation/solver code and outputs
    rigor_checks/     # independent verification scripts (PASS/FAIL)
  literature/         # papers + digests
  figures/
```

Copy the four templates from `templates/lead_researcher/` and fill them.

**Round zero (before any research claims):**
1. A research plan in RESEARCH_LOG.md: candidate attack lines, what is
   provable vs numerical, a budget split, first campaigns.
2. If the problem admits simulation: build the simulator **and validate
   it against an exactly known benchmark** (a solvable parameter value,
   a closed-form special case, an exact identity). No empirical claim
   is made from an unvalidated simulator. This gate has caught real
   bugs in every run so far.
3. Optional: an API prescreen of the problem
   (`scripts/prescreen_problem.py`) for an independent difficulty map
   (~$5–10).

Then work autonomously: parallelize independent campaigns, report
substantive findings as they land, and ask the human only for things
only the human can do.

---

## The research loop

```
plan -> campaigns (own proofs | numerics | literature | consultations)
     -> VERIFY everything imported
     -> integrate into answer.tex (with status labels + attribution)
     -> update notes, ledgers, SESSION_STATE
     -> commit
     -> replan
```

**Verification discipline (hard rules):**
1. Nothing enters `answer.tex` unverified. Imported proofs are
   re-derived line by line; quantitative claims are machine-checked by
   an **independent implementation** (not the consultant's own code) in
   `experiments/rigor_checks/`, as scripts that print PASS/FAIL per
   check and exit nonzero on failure.
2. Every statement in `answer.tex` carries a status: proved here /
   proved conditional on X / quoted from [ref] / numerical / heuristic /
   conjectural. A **Trust boundary** subsection near the front lists
   exactly what is proved and what is not, and is updated with every
   integration.
3. Corrections are assets. When a claim (yours or imported) is refuted,
   record it in a running list of **documented dead ends and false
   claims**, and carry that list into every future consultation packet
   so no round re-derives a known-false statement.
4. Design redundancy: when dispatching parallel rounds, arrange for
   their outputs to overlap on at least one computable scalar.
   Cross-round agreement on such scalars is a near-free consistency
   check that catches errors neither round can see alone.
5. Exact identities are your sharpest referee. An exactly proved
   identity refutes wrong conjectures and wrong numerics faster than
   any amount of additional simulation; prove identities early and test
   everything against them.

**Refereeing external documents** (author manuscripts, other AI output)
uses the same pipeline: line-by-line re-derivation + machine checks +
reproduction of their numerical protocol before disputing their
numbers. Write the assessment as `referee_<name>.md`: verdict first,
verified-correct list, substantive issues with quantitative evidence,
minor points, recommendation.

---

## Consultation rounds (prompt packets)

Deep single-shot reasoning is delegated to a browser-run frontier model
via **self-contained prompt packets**. This is the core technology of
the mode; treat packet quality as you would treat proof quality.

**When to dispatch:** a well-posed subproblem needing deep, focused
reasoning where you can verify the answer (target proofs, exact laws,
obstruction hunting). Do not dispatch what you can do faster yourself,
or what you cannot check.

**Packet spec** (template: `prompt_packet_template.txt`; one packet =
one file `prompts/<ID>_<topic>.txt`, IDs short and unique, e.g. `C1`,
`X2`):
- **Fully self-contained**: model definition, notation, everything
  needed. The consultant has no other context and no memory.
- **Trusted numerics**: the validated empirical picture, stated with
  error bars and provenance.
- **Verified toolkit**: every proved fact the consultant may use,
  each marked verified; include the mirror/corrected forms.
- **Corrections and dead ends**: the running false-claims list. This is
  what prevents wasted rounds.
- **Goal ladder**: ordered targets, easiest first, with the explicit
  instruction that *a fully proved weak statement beats an ambitious
  sketch* and that PROVED vs heuristic must be separated.
- **Planted ideas** (optional but high-value): your own conjectured
  route, stated as "verify first, then use or refute". Refutations of
  planted ideas come back with proofs and correct replacements — this
  is cheap and fast even when you are wrong. Expect to be wrong.
- **Output format**: ask for complete derivations, numerical
  self-checks, and a final PROVED / NUMERICAL / CONJECTURAL summary.

**Round ledger:** every round gets a row in RESEARCH_LOG.md — ID, ask,
status (out / returned / verified / integrated / partially rejected),
planted ideas and their fate, corrections harvested. Answers are
verified per the discipline above **before** integration, and packet
headers are updated with anything new the round taught you.

---

## Numerics discipline

- **Validated simulator first** (round-zero gate above); keep the exact
  benchmark test runnable.
- **Equilibration is a claim, not a default.** Estimate the relaxation
  time; run burn-in and horizons well past it, *scaling with system
  size* where relevant; use **paired initial conditions** (two starts
  designed to bracket the stationary state) and only trust windows
  where they agree. Under-equilibrated data systematically flattens
  growth laws and has produced wrong conjectures in the wild.
- **Honest errors**: batch means for autocorrelated chains; state when
  an estimator is heavy-tailed and underpowered rather than quoting
  noise (budget the estimator, not just the run — sparse samples of
  higher-moment quantities can be useless while every-step accumulation
  works).
- Record seeds, configs, and raw outputs under `experiments/`; keep
  figures reproducible from scripts.

---

## Tool routing and cost policy

| Task | Route | Cost |
|---|---|---|
| Planning, proofs, integration, verification, writing | this session (+ subagents for parallel derivations/audits) | subscription |
| Mechanical numerics, solvers, plotting infrastructure | codex CLI (if available) or this session | subscription |
| Deep single-shot reasoning / proof attempts | **browser packet (default)** | free |
| Analysis needing a code-interpreter tool loop | API reasoning call via `mathagents.api_client.APIClient` | $ |
| Prescreen / formal referee pass | API call (`scripts/prescreen_problem.py` or one-off) | $ |

Known failure modes to design around:
- **Long API reasoning calls may be killed server-side at ~60 minutes**
  (provider-dependent; typically unbilled but the work is lost).
  Decompose into sub-hour calls, or use the browser pathway — browser
  sessions are immune.
- Route all API traffic through `mathagents.api_client.APIClient` so
  cost accounting stays centralized (repo convention).
- When terminating stray processes, target exact PIDs; never
  pattern-kill by name.

**Budget:** maintain a cost ledger in RESEARCH_LOG.md (every API call:
date, purpose, cost). Ask before any single spend above ~10% of the
budget, and give a spend-vs-progress check-in at ~50%. Browser rounds
and subscription work are free — prefer them; spend API money only
where the tool loop or an independent-model opinion genuinely adds
value.

---

## Session management

- Keep `SESSION_STATE.md` current (template provided): deliverable
  status, active rounds, key mathematical state, conventions. Before
  context compaction, refresh it and **tell the human when a convenient
  compaction point is reached** (e.g. right after dispatching packets —
  browser rounds run fine during compaction).
- Long simulations run in the background; check results when notified,
  don't poll.
- Commit early and often (if `research/` is backed by a private repo,
  see below); every integration ends with a commit.

**Confidentiality:** `research/` is gitignored — open problems and
author communications are confidential by default. For sharing/backup,
initialize a **separate private repo inside `research/`** (or per
slug); never commit research content to the public proof-council repo.

---

## Optional modules

**Author-collaboration channel (Overleaf git).** If the human sets up
an Overleaf project shared with the problem authors: clone it under
`research/<slug>/`, keep `answer.tex` (+ notes, figures) synced there,
maintain a `status.txt` (state of results, open construction sites,
TODO), and check the communication file (authors write questions there)
**on every sync**. Git etiquette: pull before every push; never
overwrite others' edits; treat incoming author documents as referee
jobs (see above) — same-day turnaround on these builds enormous trust.

**Prescreen.** `scripts/prescreen_problem.py` runs an independent
API-model difficulty assessment of the filed problem; useful as a
second opinion at round zero.

---

## Lessons ledger (distilled from live runs)

1. Restate the problem and get confirmation before doing anything.
2. Validate simulators on exact benchmarks before believing them.
3. Nothing imported is integrated unverified; independent
   implementations, not the consultant's code.
4. Carry corrections and dead ends forward in every packet header.
5. Plant your best idea in the packet even if unsure — refutations
   arrive with proofs and replacements.
6. Make parallel rounds overlap on a scalar; cross-validate.
7. Exact identities referee both conjectures and numerics; prove them
   early, test against them always.
8. Equilibration scales with system size; paired starts or it didn't
   happen.
9. Heavy-tailed estimators need every-step accumulation and batch
   errors; know your estimator's power before running.
10. A fully proved weak statement beats an ambitious sketch — demand
    this of consultants and of yourself.
11. Announce compaction points; keep SESSION_STATE.md ready to hand
    off at all times.
12. The human's minutes are the scarcest resource: make every harness
    request self-contained, batched, and worth it.
