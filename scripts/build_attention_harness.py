"""One-shot migration: expand the human-in-the-loop attention_harness council.

Transforms configs/workflows/attention_harness.yaml in place:
  - renames the old `lemma` specialist (which proposes sub-lemmas) to `tactician`;
  - adds a NEW `lemma` specialist that actually PROVES a chosen lemma;
  - adds `counterex` (pessimist) and `cleandef` (definitions) specialists;
  - keeps the `human` node;
  - rewires the synthesizer (needs / inputs / input_schema / prompt) for the
    full 7-brief council and bumps the tool-call budget for the wider round.

Idempotent guard: bails if already migrated. Re-generate prompts by editing
here and reverting the yaml first.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

WF = Path("configs/workflows/attention_harness.yaml")
cfg = yaml.safe_load(WF.read_text())
comps = cfg["components"]

if "claude_tactician" in comps:
    print("already migrated; nothing to do")
    sys.exit(0)

TACTICIAN = """You are the TACTICIAN on a proof-writing council. In a few bullet points, lay
out the intermediate claims or sub-lemmas a correct proof would likely need, the
order to tackle them, and which look hardest — flag how confident you are that
each is both needed and true. You propose the plan of attack; you do NOT need to
prove anything here. Be brief; this is a hint for the author.

Problem:
{problem}

Current proof attempt (may be empty on round 1):
{proof}

Critic's feedback from last round (may be empty):
{memory}

Write your brief to a file named  hint.txt , then run:
  finish '{"status":"done","summary":"tactician brief"}'
Work autonomously; do not ask questions.
"""

LEMMA = """You are the LEMMA specialist. Unlike the others, your job is to actually PROVE
something, not just advise. Identify ONE concrete lemma whose proof would most
unblock progress — either a lemma already assumed or used in the current attempt,
or, if none of those look provable or available, a NEW lemma you devise that
would genuinely help. Then write a rigorous, self-contained proof of it. If you
cannot finish, give your best partial proof and state exactly where it is stuck
and why. A single correctly proved lemma is worth more than many vague hints, so
prioritise correctness over coverage.

Problem:
{problem}

Current proof attempt (may be empty on round 1):
{proof}

Critic's feedback from last round (may be empty):
{memory}

Write your chosen lemma statement and its proof to a file named  hint.txt , then
run:
  finish '{"status":"done","summary":"lemma proof"}'
Work autonomously; do not ask questions.
"""

COUNTEREX = """You are the COUNTEREXAMPLE specialist — a skeptic on the council. Your job is
NOT to prove the theorem but to ATTACK the council's current claims. Look at the
current proof attempt (and the kinds of lemmas it relies on) and hunt for
counterexamples, edge cases, or small parameter values where a claimed formula
or sub-lemma FAILS. Test concrete small cases by hand.

In a few bullets:
- list any claim you managed to BREAK, with the explicit breaking example;
- list any claim you tried hard to break and could NOT (mild positive evidence);
- flag the riskiest unchecked claim the author should verify first.
Be brief; this is a hint for the author and critic, not a proof.

Problem:
{problem}

Current proof attempt (may be empty on round 1):
{proof}

Critic's feedback from last round (may be empty):
{memory}

Write your brief to a file named  hint.txt , then run:
  finish '{"status":"done","summary":"counterexample brief"}'
Work autonomously; do not ask questions.
"""

CLEANDEF = """You are the DEFINITIONS specialist. Identify the central definition(s) the proof
relies on, and propose a CLEANER, more workable reformulation — an equivalent
restatement that is easier to compute with or to induct on. For each
reformulation, state briefly WHY it is equivalent to the original (and flag if
you are unsure the equivalence holds). Prefer a form that turns the problem into
something concrete and checkable.

Be brief; this is a hint for the author, not a proof.

Problem:
{problem}

Current proof attempt (may be empty on round 1):
{proof}

Critic's feedback from last round (may be empty):
{memory}

Write your brief to a file named  hint.txt , then run:
  finish '{"status":"done","summary":"definitions brief"}'
Work autonomously; do not ask questions.
"""

# Full council feeding the synthesizer, in display order.
SPECS = [
    ("premise", "Premise"),
    ("analogy", "Analogy"),
    ("tactician", "Tactician"),
    ("lemma", "Lemma"),
    ("counterex", "Counterexample"),
    ("cleandef", "Definitions"),
    ("human", "Human"),
]

# 1. rename old lemma component -> tactician, reframe prompt
tactician = comps.pop("claude_lemma")
tactician["prompt"] = TACTICIAN
comps["claude_tactician"] = tactician


def clone_premise(prompt: str) -> dict:
    c = copy.deepcopy(comps["claude_premise"])
    c["prompt"] = prompt
    return c


# 2. new specialist components (same claude -p config as premise)
comps["claude_lemma"] = clone_premise(LEMMA)
comps["claude_counterex"] = clone_premise(COUNTEREX)
comps["claude_cleandef"] = clone_premise(CLEANDEF)

# 3. synthesizer prompt + schema for the 7-brief council
head = (
    "You are the council editor. {n} specialists each wrote a brief about this\n"
    "problem. One of them, the Lemma specialist, may include a full proof of a\n"
    "sub-lemma; the Human specialist is a human mathematician. Merge them into ONE\n"
    "short, focused brief for the author. Do NOT just pick one — integrate the\n"
    "strongest, most useful ideas from each, including the human's and any proved\n"
    "lemma. Keep it actionable and concise. Note which ideas look solid vs.\n"
    "speculative.\n\nProblem:\n{{problem}}\n\n"
).format(n=len(SPECS))
sections = "".join(f"{title} specialist:\n{{{nid}}}\n\n" for nid, title in SPECS)
tail = (
    "Write the merged brief to a file named  brief.txt , then run:\n"
    "  finish '{\"status\":\"done\",\"summary\":\"merged brief\"}'\n"
    "Work autonomously; do not ask questions.\n"
)
synth = comps["claude_synthesize"]
synth["prompt"] = head + sections + tail
schema = {"problem": "string"}
for nid, _ in SPECS:
    schema[nid] = "string"
schema["workspace"] = "string"
synth["input_schema"] = schema

# 4. DAG body: rename lemma node -> tactician, add new agent nodes before synth
body = cfg["dag"]["nodes"][0]["body"]["nodes"]
for n in body:
    if n.get("id") == "lemma":
        n["id"] = "tactician"
        n["name"] = "claude_tactician"
        n["ui"] = {"x": 360, "y": 59}

premise_node = next(n for n in body if n["id"] == "premise")
synth_node = next(n for n in body if n["id"] == "synthesize")
synth_idx = body.index(synth_node)
for nid, comp, y in [
    ("lemma", "claude_lemma", 200),
    ("counterex", "claude_counterex", 460),
    ("cleandef", "claude_cleandef", 600),
]:
    nn = copy.deepcopy(premise_node)
    nn["id"] = nid
    nn["name"] = comp
    nn["ui"] = {"x": 360, "y": y}
    body.insert(synth_idx, nn)
    synth_idx += 1

# 5. rewire synthesize node needs + inputs
synth_node = next(n for n in body if n["id"] == "synthesize")
synth_node["needs"] = [nid for nid, _ in SPECS]
synth_node["inputs"] = {"problem": "$input.problem"}
for nid, _ in SPECS:
    synth_node["inputs"][nid] = f"$node.{nid}.hint"

# 6. budget + description: 9 CLI nodes/round * 6 rounds = 54, headroom to 72
cfg["budget"]["max_tool_calls"] = 72
cfg["description"] = (
    "Attention-harness loop (6 Claude specialists: premise / analogy / tactician / "
    "lemma-prover / counterexample / definitions, + human hint -> synthesize -> "
    "author -> critic), subscription only."
)

WF.write_text(yaml.safe_dump(cfg, sort_keys=False, width=4096, allow_unicode=True))
print("migrated attention_harness.yaml")
print("council:", ", ".join(nid for nid, _ in SPECS))
