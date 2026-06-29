"""Generate attn_exp_* prompt variants from the baseline harness.

Only the `prompt:` field of specialist/synthesize/author/critic components is
varied. DAG topology, roles, node count, budget, and models are untouched.
Run: uv run python scripts/make_attn_variants.py
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
# These experiment variants are personal scaffolding, not shipped examples;
# they live under the gitignored configs/workflows/local/ tree.
WF = ROOT / "configs" / "workflows" / "local"
BASE = WF / "attention_harness_auto.yaml"

base = yaml.safe_load(BASE.read_text())

# ---------------------------------------------------------------------------
# Baseline prompts (verbatim from attention_harness_auto.yaml), kept here so we
# can selectively override individual ones per variant.
# ---------------------------------------------------------------------------
P_PREMISE = base["components"]["claude_premise"]["prompt"]
P_ANALOGY = base["components"]["claude_analogy"]["prompt"]
P_LEMMA = base["components"]["claude_lemma"]["prompt"]
P_SYNTH = base["components"]["claude_synthesize"]["prompt"]
P_AUTHOR = base["components"]["claude_author"]["prompt"]
P_CRITIC = base["components"]["claude_critic"]["prompt"]

# ---------------------------------------------------------------------------
# Variant prompt overrides. Each keeps the same I/O contract (write file X,
# then run finish '{...}'), changing only the substantive guidance.
# ---------------------------------------------------------------------------

# --- specialists variant: sharper, domain-pointed specialist briefs ---------
SPEC_PREMISE = """You are the PREMISE specialist in a proof-writing council. This is a
COUNTING problem (count isomorphism classes of combinatorial structures). In a
few bullet points, name the key definitions and classical results most relevant.
Be brief and concrete; this is a hint for the author, not a proof.

Make sure to consider, and say whether each is actually relevant:
- The precise definition of a stable graph of genus g (vertices carry genus
  weights g_v >= 0, the genus formula g = sum(g_v) + first Betti number =
  sum(g_v) + (#edges - #vertices + 1), and the stability condition
  2 g_v - 2 + (valence at v) > 0 for every vertex).
- Counting isomorphism classes = counting orbits, so Burnside / orbit-counting
  and the structure of graph automorphism groups are likely central.
- That with e fixed and g large, the count is a value of a combinatorial
  counting function in g; quasi-polynomiality of such counts (Ehrhart-style /
  lattice-point / partition-with-bounded-parts behavior) is the expected shape.

Flag any item you are unsure is actually relevant.

Problem:
{problem}

Current proof attempt (may be empty on round 1):
{proof}

Critic's feedback from last round (may be empty):
{memory}

Write your brief to a file named  hint.txt , then run:
  finish '{"status":"done","summary":"premise brief"}'
Work autonomously; do not ask questions.
"""

SPEC_ANALOGY = """You are the ANALOGY specialist. In a few bullet points, suggest analogous
problems, standard proof templates, or strategies that have worked on similar
problems. Be brief; this is a hint for the author. Flag how promising vs.
speculative each suggestion is.

This is a fixed-e, varying-g enumeration. Strongly consider:
- Reducing to a finite list of "topological types" (the underlying graph shape
  with e edges, ignoring genus weights) and then, for each shape, counting the
  ways to distribute the leftover genus across vertices subject to stability.
- The genus-distribution step as counting integer points / partitions into a
  bounded number of parts -> these counts are quasi-polynomial in g with period
  dividing lcm of the relevant denominators, which explains the periodicity.
- Doing e=0,1,2,3 by hand FIRST as the source of truth, then matching the
  general formula to those data points.

Flag how promising vs. speculative each suggestion is.

Problem:
{problem}

Current proof attempt (may be empty on round 1):
{proof}

Critic's feedback from last round (may be empty):
{memory}

Write your brief to a file named  hint.txt , then run:
  finish '{"status":"done","summary":"analogy brief"}'
Work autonomously; do not ask questions.
"""

SPEC_LEMMA = """You are the LEMMA specialist. In a few bullet points, list the intermediate
claims or sub-lemmas a correct proof would likely need, and flag which look
hardest and how confident you are each is needed and true. Be brief.

REQUIREMENT: explicitly demand small-case verification. The author must
enumerate M_e(g) by hand for the small cases (e = 0, 1, 2, 3, and several
genera each, including computing M_3(2)) and CHECK any proposed general formula
reproduces those numbers before claiming it. List, as sub-lemmas:
- finiteness of underlying topological types for fixed e;
- for each type, the genus-weight distribution count as a function of g
  (quasi-polynomial, with its period);
- correct handling of automorphisms / overcounting when summing over types;
- the boundary between "small g" exceptional values and the eventual
  quasi-polynomial regime.

Flag which sub-lemmas are hardest and your confidence in each.

Problem:
{problem}

Current proof attempt (may be empty on round 1):
{proof}

Critic's feedback from last round (may be empty):
{memory}

Write your brief to a file named  hint.txt , then run:
  finish '{"status":"done","summary":"lemma brief"}'
Work autonomously; do not ask questions.
"""

# --- critic variant: forces small-case checking before voting ready ---------
CRITIC_STRICT = """You are a strict referee. Check the proof rigorously for gaps, unjustified
steps, false claims, and missing cases.

CRITICAL GATING RULE: before you may write the verdict "ready", you must
verify the proof against concrete small cases by independent computation. In
particular:
- Independently confirm M_3(2) = 2 and that the proof's claimed genus-3
  formula reproduces it.
- Spot-check at least two more values (e.g. M_3(g) for a small g >= 3, and an
  e in {0,1,2}) against the proof's general formula.
- Confirm the proof actually states the eventual quasi-polynomial structure
  result AND gives complete descriptions for e = 0, 1, 2, 3.
If ANY of these checks fails, is missing, or the proof is internally
contradictory, the verdict MUST be "continue".

Problem:
{problem}

Proof to check:
{proof}

Then do BOTH of these:
1. Write specific, actionable feedback (what to fix next), including the
   results of your small-case checks, to a file named  feedback.md .
2. Write a file named  verdict.txt  containing EXACTLY one word, no
   punctuation: "ready" if and only if every check above passed and the proof
   is complete and correct, otherwise "continue".
Finally run:
  finish '{"status":"done","summary":"critique"}'
Work autonomously; do not ask questions.
"""

# --- author variant: demands worked enumeration before general claims -------
AUTHOR_ENUM = """You are the author/documenter for a mathematical research workflow. Your job
is to turn the focused brief below into the clearest possible LaTeX progress
document.

MANDATORY METHOD: before stating ANY general structural claim, you must first
work out the small cases explicitly. Enumerate the isomorphism classes of
stable graphs of genus g with 0 legs and e edges by hand for e = 0, 1, 2, 3,
deriving closed-form expressions for M_0(g), M_1(g), M_2(g), M_3(g), and
explicitly computing M_3(2). Only after these worked enumerations may you state
and justify the general structure result (eventual quasi-polynomiality in g for
fixed e). Every general formula you assert must be cross-checked against your
own enumerated small-case values; show that check.

If the ingredients are enough for a complete proof, write a complete rigorous
proof. If they are not enough, do not pretend they are: write a clean partial
proof, clearly separate established steps from conjectural or unfinished steps,
and state exactly what remains to be proved.

Problem:
{problem}

Focused brief (synthesized from the specialists):
{brief}

Current proof attempt (may be empty on round 1):
{proof}

Critic's feedback from last round (may be empty):
{memory}

Write a self-contained LaTeX document to a file named answer.tex. It should
include:
1. a concise status paragraph saying whether this is a complete proof or a
   progress report;
2. the explicit small-case enumerations (e = 0,1,2,3) with M_3(2) computed;
3. the cleanest rigorous general argument currently justified, cross-checked
   against the small cases;
4. a clearly marked section listing the remaining gaps, missing lemmas, or
   checks still needed.
Then run:
  finish '{"status":"done","summary":"author progress draft"}'
Work autonomously; do not ask questions.
"""

# --- synth variant: forces explicit prioritization of author's first move ---
SYNTH_PRIORITIZED = """You are the council editor. Three specialists each wrote a brief about this
problem. Merge them into ONE short, focused brief for the author. Do NOT just
pick one - integrate the strongest, most useful ideas from each. Keep it
actionable and concise. Note which ideas look solid vs. speculative.

Your brief MUST open with an explicit, ordered "DO THIS FIRST" action list (3-5
numbered steps) telling the author the exact sequence to follow - e.g. which
small cases to enumerate by hand before any general claim, then which general
structure to aim for, then what to cross-check. Put the highest-leverage,
most-certain step first.

Problem:
{problem}

Premise specialist:
{premise}

Analogy specialist:
{analogy}

Lemma specialist:
{lemma}

Write the merged brief to a file named  brief.txt , then run:
  finish '{"status":"done","summary":"merged brief"}'
Work autonomously; do not ask questions.
"""

VARIANTS = {
    "baseline": {},  # unmodified
    "specialists": {
        "claude_premise": SPEC_PREMISE,
        "claude_analogy": SPEC_ANALOGY,
        "claude_lemma": SPEC_LEMMA,
    },
    "critic": {"claude_critic": CRITIC_STRICT},
    "author": {"claude_author": AUTHOR_ENUM},
    "synth": {"claude_synthesize": SYNTH_PRIORITIZED},
    "combo": {
        "claude_premise": SPEC_PREMISE,
        "claude_analogy": SPEC_ANALOGY,
        "claude_lemma": SPEC_LEMMA,
        "claude_synthesize": SYNTH_PRIORITIZED,
        "claude_author": AUTHOR_ENUM,
        "claude_critic": CRITIC_STRICT,
    },
}

for name, overrides in VARIANTS.items():
    cfg = copy.deepcopy(base)
    cfg["description"] = f"Attention-harness experiment variant '{name}' (prompt-only change from baseline)."
    for comp, prompt in overrides.items():
        cfg["components"][comp]["prompt"] = prompt
    out = WF / f"attn_exp_{name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, width=4096, allow_unicode=True))
    print("wrote", out.name)
