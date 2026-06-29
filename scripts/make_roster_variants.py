"""Generate attn_roster_* variants: the human-free baseline council with one (or
several) EXTRA specialist nodes added to the parallel set.

Unlike make_attn_variants.py (which only rewords existing prompts), this adds new
nodes to the DAG and rewires the synthesizer to consume them. Each new specialist
is a clone of claude_premise (same claude -p config / sandbox / I/O contract:
write hint.txt, then finish), differing only in its prompt. The synthesize node's
needs / inputs / input_schema and its prompt are regenerated from the full
specialist list so the editor stays consistent.

Run: uv run python scripts/make_roster_variants.py
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT / "configs" / "workflows"
BASE = WF / "attention_harness_auto.yaml"

base = yaml.safe_load(BASE.read_text())

# Base specialists already in the council: (node_id, Title for the synth prompt).
BASE_SPECS = [("premise", "Premise"), ("analogy", "Analogy"), ("lemma", "Lemma")]

# ---------------------------------------------------------------------------
# New experimental specialist prompts. Each keeps the premise node's I/O
# contract: inputs {problem, proof, memory}, writes hint.txt, finishes.
# ---------------------------------------------------------------------------

COUNTEREX = """You are the COUNTEREXAMPLE specialist — a skeptic on the council. Your job is
NOT to prove the theorem but to ATTACK the council's current claims. Look at the
current proof attempt (and the kinds of lemmas this proof needs) and hunt for
counterexamples, edge cases, or small parameter values where a claimed formula
or sub-lemma FAILS. Test concrete small cases by hand and compute.

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

CLEANDEF = """You are the DEFINITIONS specialist. Identify the central definition(s) the
proof relies on, and propose a CLEANER, more workable reformulation — an
equivalent restatement that is easier to compute with or to induct on. For each
reformulation, state briefly WHY it is equivalent to the original (and flag if
you are unsure the equivalence holds). Prefer a form that turns the problem into
something concrete and countable.

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

STRATEGY = """You are the STRATEGY specialist. Before the author writes anything, critique the
overall PROOF STRATEGY at a high level. Given the problem and the current
attempt, say:
- which high-level approach is most likely to actually work here;
- what the current approach (if any) is getting wrong or wasting effort on;
- the single most important thing the author should do FIRST.
Be brief and decisive — this is a steer for the author, not a proof. Do not
write out the proof yourself.

Problem:
{problem}

Current proof attempt (may be empty on round 1):
{proof}

Critic's feedback from last round (may be empty):
{memory}

Write your brief to a file named  hint.txt , then run:
  finish '{"status":"done","summary":"strategy brief"}'
Work autonomously; do not ask questions.
"""

REDUCE = """You are the REDUCTION specialist. Suggest a change of target that makes the
problem easier. Pick whichever of these is more promising and explain why:
(a) a SIMPLER sub-problem to solve first that would unlock the full result
    (a special case, a smallest nontrivial parameter, a stripped-down version);
or
(b) a MORE GENERAL statement that is actually EASIER to prove (e.g. a stronger
    induction hypothesis that carries more information through the induction).
Be brief; this is a hint for the author, not a proof.

Problem:
{problem}

Current proof attempt (may be empty on round 1):
{proof}

Critic's feedback from last round (may be empty):
{memory}

Write your brief to a file named  hint.txt , then run:
  finish '{"status":"done","summary":"reduction brief"}'
Work autonomously; do not ask questions.
"""

# new_id -> (Title, prompt, ui_y)
NEW_NODES = {
    "counterex": ("Counterexample", COUNTEREX, 480),
    "cleandef": ("Definitions", CLEANDEF, 620),
    "strategy": ("Strategy", STRATEGY, 760),
    "reduce": ("Reduction", REDUCE, 900),
}

VARIANTS = {
    "counterex": ["counterex"],
    "cleandef": ["cleandef"],
    "strategy": ["strategy"],
    "reduce": ["reduce"],
    "combo": ["counterex", "cleandef", "strategy", "reduce"],
}

SYNTH_HEAD = (
    "You are the council editor. {n} specialists each wrote a brief about this\n"
    "problem. Merge them into ONE short, focused brief for the author. Do NOT\n"
    "just pick one — integrate the strongest, most useful ideas from each. Keep it\n"
    "actionable and concise. Note which ideas look solid vs. speculative.\n\n"
    "Problem:\n{{problem}}\n\n"
)
SYNTH_TAIL = (
    "Write the merged brief to a file named  brief.txt , then run:\n"
    "  finish '{\"status\":\"done\",\"summary\":\"merged brief\"}'\n"
    "Work autonomously; do not ask questions.\n"
)


def build_synth_prompt(full_specs: list[tuple[str, str]]) -> str:
    sections = "".join(f"{title} specialist:\n{{{nid}}}\n\n" for nid, title in full_specs)
    return SYNTH_HEAD.format(n=len(full_specs)) + sections + SYNTH_TAIL


def body_nodes(cfg: dict) -> list:
    return cfg["dag"]["nodes"][0]["body"]["nodes"]


def find_node(nodes: list, node_id: str) -> dict:
    return next(n for n in nodes if n.get("id") == node_id)


for name, added in VARIANTS.items():
    cfg = copy.deepcopy(base)
    cfg["description"] = (
        f"Attention-harness ROSTER variant '{name}': baseline 3 specialists "
        f"plus {', '.join(added)} (extra node(s) added to the parallel set)."
    )

    nodes = body_nodes(cfg)
    synth_node = find_node(nodes, "synthesize")
    synth_idx = nodes.index(synth_node)
    premise_node = find_node(nodes, "premise")

    full_specs = list(BASE_SPECS)
    for nid in added:
        title, prompt, ui_y = NEW_NODES[nid]
        comp_key = f"claude_{nid}"
        # component: clone premise's claude -p config, swap the prompt
        comp = copy.deepcopy(cfg["components"]["claude_premise"])
        comp["prompt"] = prompt
        cfg["components"][comp_key] = comp
        # DAG node: clone premise's body node, before synthesize
        dag_node = copy.deepcopy(premise_node)
        dag_node["id"] = nid
        dag_node["name"] = comp_key
        dag_node["ui"] = {"x": 360, "y": ui_y}
        nodes.insert(synth_idx, dag_node)
        synth_idx += 1
        synth_node = find_node(nodes, "synthesize")
        full_specs.append((nid, title))

    # rewire synthesize node + component for the full specialist list
    synth_node["needs"] = [nid for nid, _ in full_specs]
    synth_node["inputs"] = {"problem": "$input.problem"}
    for nid, _ in full_specs:
        synth_node["inputs"][nid] = f"$node.{nid}.hint"

    synth_comp = cfg["components"]["claude_synthesize"]
    synth_comp["prompt"] = build_synth_prompt(full_specs)
    schema = {"problem": "string"}
    for nid, _ in full_specs:
        schema[nid] = "string"
    schema["workspace"] = "string"
    synth_comp["input_schema"] = schema

    out = WF / f"attn_roster_{name}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, width=4096, allow_unicode=True))
    print("wrote", out.name)
