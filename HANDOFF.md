# HANDOFF — ProofCouncil human-in-the-loop session

**If you are a fresh Claude Code session: read "CURRENT STATE" first — it is
where the work actually is. The rest of this file is the detailed log of how
we got here. David is a mathematician, not a software engineer — explain
things plainly and do not assume he can debug.**

---

# CURRENT STATE (2026-06-25, end-of-session handoff)

## What we're doing

David is running ProofCouncil **with himself in the loop** — he is the human
node in the council. The goal is to feel out the system on real problems and
own the human-in-the-loop angle. Automated harness evolution (sweeps, prompt
tuning) is a separate dev's problem; don't conflate them.

## Branch / git state

Branch: `feat/human-in-the-loop`  
Last pushed commit: `be6e8ee` (Fix attention_harness budget mismatch)  
**This session's work is uncommitted but staged and ready to commit/push.**

Files changed (our work, committed in the shutdown commit):
- `AGENTS.md` — added the "Local dev helpers" section (helper scripts)
- `configs/workflows/attention_harness.yaml` — major rewrite: 9-specialist council
- `HANDOFF.md` — this file
- `configs/workflows/attention_harness_auto.yaml` — human-free baseline (for automated experiments)
- `scripts/build_attention_harness.py` — one-shot migration script (already ran; idempotent)
- `scripts/validate_preset.sh`, `run_dashboard.sh`, `health.sh`, `run_status.sh`,
  `watch_run.sh`, `verify_resume.sh` — dev helper wrappers (keep commands
  auto-approvable per AGENTS.md)
- `scripts/judge_attn.py`, `run_attn_sweep.sh`, `run_roster_sweep.sh`,
  `make_attn_variants.py`, `make_roster_variants.py` — sweep infrastructure
  (built, ran, but grading failed due to org usage limit — see below)
- `configs/workflows/attn_exp_*.yaml`, `attn_roster_*.yaml` — sweep variants
- `problems/Cross_ratio_completeness.txt`, `problems/Monotonicity_of_the_cross-ratio_degree.txt`

NOT ours (leave unstaged):
- `configs/workflows/author_critic.yaml` — pre-existing change (managed_needs UI field)
- `configs/workflows/dapper_zigzag.yaml`, `plucky_spark.yaml`, `zesty_zigzag.yaml`
  — dashboard auto-generated empty templates
- `problems/GRH.txt` — pre-existing

## The attention harness (main artifact)

`configs/workflows/attention_harness.yaml` is the primary thing David uses.
It now has a **9-specialist council** (8 AI + 1 human) running in parallel
each round, feeding a synthesizer → author → critic loop.

**Council members** (all feed `synthesize`, which merges into a brief for `author`):
| node | role |
|------|------|
| `premise` | key definitions and classical results |
| `analogy` | analogous problems and proof templates |
| `tactician` | sub-lemma plan (what to prove and in what order) |
| `lemma` | actually PROVES one chosen lemma (rigorous self-contained) |
| `counterex` | attacks claims; finds counterexamples or edge cases |
| `cleandef` | proposes a cleaner/more workable reformulation of key definitions |
| `litsearch` | live web search for relevant papers (WebSearch + WebFetch) |
| `compute` | writes and runs Python to check small cases (Bash(*)) |
| `human` | David's hint (pauses the run for input) |

**Config knobs:**
- Default model: `sonnet` (override with `--input claude_model=haiku` for cheap)
- Max rounds: 6
- Budget: `max_tool_calls: 90` (11 CLI nodes × 6 rounds = 66 + headroom)
- Max wallclock: 2 h
- Preset validates: `ok: True`

**How to run:**  
Start dashboard: `scripts/run_dashboard.sh` → http://localhost:5005  
Then: Run Agent → attention_harness → paste problem → Launch.  
Each round pauses at the human node for your hint; submit in the dashboard UI.

**Alternatively (CLI):**
```bash
uv run python scripts/run_workflow.py \
  --workflow attention_harness \
  --problem problems/Cross_ratio_completeness.txt \
  --input claude_model=sonnet
```

## Org monthly usage limit — important

During automated sweep experiments this session, the Anthropic org usage limit
was hit. Any call returning "You've hit your org's monthly usage limit" means
the API/CLI is blocked until the limit resets (monthly). Check before starting
a run — if `claude -p "say hi"` fails with that message, wait.

David was testing the harness (possibly running a problem) when this session
ended. Check whether his run completed.

## Two new nodes — implementation notes

**Lit-search** (`claude_litsearch`): uses `--allowedTools WebSearch,WebFetch,Bash(finish:*)`
so the node can search the web live and fetch pages. Prompt instructs it to
cite precisely, state what result is applicable, and say so rather than invent
if nothing is found. Most valuable on research-frontier problems; less so on
self-contained combinatorics.

**Compute** (`claude_compute`): uses `--allowedTools Bash(*)` (unrestricted
bash in the subprocess sandbox). Can write and run Python (sympy, numpy, etc.)
to check formulas and find counterexamples. `HOME=/Users/davidholmes` pin is
still in place (needed for Claude subscription auth) — so this node has home
access. On a trusted machine this is fine; for untrusted problems, consider a
more restricted allowedTools or a separate sandboxed HOME.

## Pending / next session

1. **Check if David's test run completed** and how the new council felt.
2. **Pick a real problem** (not stable_graphs, which is too easy for this
   council) and run with David fully in the loop.
3. **Sweep results (deferred)**: if the usage limit has reset, `judge_attn.py`
   can grade the exp-* and roster-* runs that produced `best_tex` but weren't
   graded. Run IDs in `outputs/`.
4. **On-demand compute (deferred)**: David's preference is eventually for the
   compute node to be author-commissioned rather than fire every round. That
   needs new runtime plumbing (trigger node type); synchronous-every-round is
   the v1 built here.
5. **PR #1** on eth-sri/proof-council: if/when it merges to their main, post
   the follow-up Zulip comment pointing people at the upstream URL instead of
   David's fork.

---

# PREVIOUS STATE (2026-06-25, mid-session note)

## Zulip message posted

David posted a message to Zulip describing the human-in-the-loop harness. The
agreed message pointed at `attention_harness.yaml` (not `human_council.yaml`
which requires the `codex` CLI too). Key lines:

- "You are the node" (the human is a first-class council member)
- Workflow: 6 Claude specialists run in parallel each round, you submit a hint
  via the dashboard, synthesizer merges everything, author drafts the proof.
- `human_council.yaml` needs both `claude` AND `codex` CLIs logged in; use
  `attention_harness.yaml` if you only have `claude`.
- A comment was added to `human_council.yaml` noting the codex dependency.

## Sweep experiments (built this session, grading failed)

Built a full automated sweep pipeline (prompt variants and roster variants) on
the `stable_graphs_M_e.tex` problem using Haiku. Runs were produced (6 prompt
variants, 5 roster variants) but the independent judge (`judge_attn.py`) hit
the org usage limit on every call, leaving results ungraded. Infrastructure is
in `scripts/` and `configs/workflows/attn_*.yaml` — reuse when limit resets.

## David's pivot

After the sweeps ran, David decided: automated harness evolution is another
dev's problem. He wants to focus on human-in-the-loop runs with himself
participating. This is now the primary direction.

---

# PREVIOUS STATE (2026-06-24, COST-CONTROL session)

**Read this section only if you need the full history of cost controls and
durable resume.**

## UPDATE (2026-06-25, Codex handoff note)
**Files changed by Codex in this tag-in:** `configs/workflows/attention_harness.yaml`
and this `HANDOFF.md` note only.

- David wanted to try `configs/workflows/attention_harness.yaml` with Codex
  models instead of Claude. Edited that workflow directly: the AI nodes now use
  `codex exec` with `copy_codex_auth: true` and `codex_sandbox: auto`.
- Model split: premise / analogy / lemma / synthesize / author use
  `gpt-5.4-mini`; critic uses `gpt-5.5`. The human hint node is unchanged.
- Removed the `claude_model` workflow input and Claude-specific `usage:
  {type: claude_json}` / `HOME` settings from those attention-harness nodes.
  Validation passed with `scripts/validate_preset.sh
  configs/workflows/attention_harness.yaml`.
- After David tried the run, he noticed the old author prompt was too strong:
  it asked for a complete proof. Changed the author prompt so the node is a
  progress documenter: write a complete proof only if the ingredients justify
  it; otherwise write a clean partial proof/progress report with established
  steps separated from gaps and next lemmas/checks. Validation passed again.
- Debugged the active run on Monotonicity_of_the_cross-ratio_degree. The
  synthesizer output DID reach the author. Important runtime note:
  `cli_stdout.log` / `cli_stderr.log` are written only after the CLI process
  exits; while a node is live, heartbeat events show `stdout_chars` /
  `stderr_chars` instead.
- Cleanup for later: component names are still `claude_premise`, `claude_author`,
  etc. even though they now run Codex (before the Claude migration of this
  session restored them). Runtime is fine, but naming is messy.

## OPEN FOLLOW-UPS (from cost-control session)
- **LONGER-TERM (David's call, deferred): borrow the structured-output SHAPE from
  the First Proof Foundation OSS harness** `github.com/1stproof/math-solve-oss-FP`.
  The gem is its proof representation (a DAG of lemma/definition nodes with
  per-node suspicion probabilities `p_statement_false` / `p_argument_gap`). That
  DAG is a bigger architectural change for main devs. What David IS interested in
  (later, quick): copy the SHAPE only — a structured-output envelope
  (`{id, agent, notes}`) + a defined output schema per agent. LEGAL: their code +
  schemas are AGPL-3.0, prompts CC-BY-SA-4.0; ProofCouncil is MIT. Clean-room
  reimplement; paraphrase prompts. Cheap bonus: a self-assessed `p_argument_gap` /
  confidence field on solver/critic outputs.

## GIT STATE (cost-control session, 2026-06-24)
All cost-control work was committed and pushed:
- `b844af9` token accounting + guardrails
- `5fdb1e5` meter full throughput incl. cache + recalibrate
- `c161d1e` Stop button + phantom detection (detail)
- `d7db536` phantom badge on /runs list
- `ee7634b` stream-json metering (timeout fix)
- Plus: `9a7192a`, `542d630`, `0c62e08`, `34f94cd` (cache, UX, prune)

---

# PREVIOUS STATE (durable-resume phase, 2026-06-24 late session)

Everything below is committed and clean — the durable-resume / persistence
phase is complete (items 1–5 of the LEFT TO DO list).

## What this phase was about (ALL DONE)
Don't lose days of work when a laptop closes / crashes. The key features:
- **Resume**: replay completed nodes from disk (including answered human prompts)
  and continue from the first unfinished node.
- **Cache**: all CLI node outputs cached on disk; human nodes default on.
- **Prune**: manual button to free sandbox/log disk space for finished nodes.
- **Waiting pill**: global nav indicator shows which runs need human input.
- **Phantom detection**: status "running" but PID dead → shows Resume instead of Stop.

All verified end-to-end (321 tests pass, 1 pre-existing failure unrelated to us).

## Key caveats
- Resume keys nodes by `hash(config + inputs)`. Editing a preset between crash
  and resume re-runs changed nodes — and the cascade can re-ask the human.
- `cache_enabled` is OPT-IN per component. Do NOT blanket-enable for presets
  where CLI nodes share a live `workspace` (replayed JSON output won't
  reconstruct sandbox files).

---

## The goal (persistent)
David wants to run ProofCouncil mostly on his **Claude Max subscription** (with
a little GPT for diversity), using the `claude` / `codex` CLIs — NOT paid API
keys. He does not have API credits and does not want to buy them.

## Key facts about his machine
- **No Docker installed.** CLI worker nodes must use `sandbox.backend: subprocess`.
- `codex` and `claude` CLIs both installed and logged in.
- Package manager: `uv`. Run things with `uv run python ...`.
- Global commands: `proofcouncil` (CLI) and `proofcouncil-app` (web UI on :5002).

## How models are used
1. **`APIClient`** — Author/Critic/Council "thinking" blocks. Needs API keys.
   There is no subscription/CLI fallback for these blocks.
2. **`ConfigurableCLIAgent`** — drives `codex`/`claude` as agentic CLI workers.
   **This is the only key-free path.** Pipes the component `prompt` to stdin,
   installs a `finish` command on PATH, waits for `finish` (writes `done.json`).

The attention_harness is entirely CLI-based (no API keys needed).

## Claude subscription login in subprocess sandbox
The subprocess sandbox pins `HOME` to the sandbox root, so Claude reports
"Not logged in". Fix: `env: { HOME: /Users/davidholmes }` in the component
overrides HOME back to the real home directory. Applied to all claude nodes.
Safety: the subprocess sandbox's file-read tools are working-directory gated;
`--allowedTools Bash(finish:*)` prevents arbitrary shell access.
