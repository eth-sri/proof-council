# mathagents — project description

This repository will be the implementation of an autonomous math-research
agent system targeting the **First Proof Foundation, Second Batch**
benchmark (June 2026). Beyond the benchmark, the same system is intended
to grow into a human-in-the-loop research assistant for mathematicians.

`configs/workflows/instructions.md` is the current source of truth for
workflow syntax and reusable YAML components.

---

## What this repo currently is

A ProofStack workflow framework built on the MathArena provider/tool
layer. The kept pieces are:

- `src/mathagents/api_client.py` — robust multi-provider client (OpenAI,
  Anthropic, Google, xAI, DeepSeek, GLM, Moonshot, Together, vLLM, …)
  with retries, batch processing, tool-call loops, and cost accounting.
- `src/mathagents/tools/` — `code_execution` (Modal + Docker fallback),
  `paper_search` (Semantic Scholar + GLM OCR), `query_knowledge` (OEIS,
  Wikipedia, Wolfram).
- `configs/models/` — layered YAML model definitions with `base:`
  inheritance.
- `configs/workflows/` — DAG workflow presets. Read
  `configs/workflows/instructions.md` before creating or editing these.
- `app/` — Flask viewer for `outputs/<run>/*.json` artifacts.
- `scripts/run_workflow.py` — CLI entry point for workflow presets.

## What it is becoming

A modular framework where every "agent" — single API call, CLI subprocess
(Codex / Claude Code), or multi-turn tool-using loop — is a free-standing
unit with typed input / output channels, history, and cost tracking.
Workflows are composed by passing agents to other agents (tools may
themselves be agents). A non-technical mathematician should be able to
describe a new sub-agent conceptually and get something working with a
prompt template plus a small config.

The autopilot benchmark mode and a future human-in-the-loop UI are two
front-ends on the same agent layer.

For new workflow work, follow `configs/workflows/instructions.md`.

---

## Repo layout

```
src/mathagents/      # API client, config loader, and reusable tools
src/proofstack/      # Workflow/agent runtime
configs/             # YAML configs (models/, tools/, workflows/)
app/                 # Flask output viewer
scripts/             # CLI entry points
problems/            # Plain-text problem files
outputs/             # Run artifacts (JSON; gitignored)
solutions/           # Plain-text final answers (gitignored)
```

The new agent layer lives at `src/proofstack/`. The supported authoring
path is YAML workflow presets using `ConfigurablePromptAgent`,
`ConfigurableCLIAgent`, `DAGWorkflow`, and small deterministic helpers.

---

## How to run things today

We use [`uv`](https://github.com/astral-sh/uv).

Set provider keys via env vars (see `README.md` for the full table —
typically `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, …).

Run a workflow preset:

```bash
uv run python scripts/run_workflow.py \
  --workflow configs/workflows/nimble_proof.yaml \
  --problem "Prove that there are infinitely many primes."
```

Browse outputs:

```bash
uv run python app/app.py --output-folder demo-run
```

---

## Conventions

- **Minimalistic code.** No premature abstraction. A bug fix is a bug
  fix; don't slip in a refactor.
- **No comments unless they explain a non-obvious *why*.** Names should
  carry the *what*.
- **Python ≥ 3.12** (matches `pyproject.toml`). Type hints welcome where
  they help; not required everywhere.
- **Configs are YAML.** Use `base:` to inherit from another config; don't
  copy-paste prompts. Place model configs under `configs/models/<provider>/`.
- **Workflow presets have their own syntax guide.** Before adding or
  editing `configs/workflows/*.yaml`, read
  `configs/workflows/instructions.md`. It covers DAG node syntax,
  `ConfigurablePromptAgent`, tool refs, repeat loops, compile nodes, and
  validation commands.
- **All API traffic goes through `mathagents.api_client.APIClient`** so
  cost / token / retry logic stays consistent. Don't spin up a raw
  `openai.OpenAI()` somewhere.
- **Cost is real.** Workflow agents should route model calls through
  `mathagents.api_client.APIClient` so token and cost accounting stays
  centralized.
- **Checkpointing matters.** Long runs should persist enough state under
  their run directory to inspect and resume them.

### When editing workflow presets

Prefer config-only workflows in `configs/workflows/*.yaml` when the
requested agent is just a DAG of prompt nodes, repeat loops, and existing
tools. Read `configs/workflows/instructions.md` first; it is the compact
reference for the workflow runtime and should prevent spelunking through
`src/proofstack/agents/dag_workflow.py` for common syntax.

---

## Reading order for a new session

1. **This file** — orientation.
2. **`configs/workflows/instructions.md`** — concrete workflow syntax and
   reusable component guidance; required before creating or editing
   workflow preset YAML.
3. **`README.md`** — how to install + run today.

## Out of scope (for now)

- A web UI for the human-in-the-loop mode. Capture the *requirements*
  for it (event log shape, agent introspection) before building, but do
  not build it before the benchmark.
- Re-implementing what `APIClient` already handles (provider quirks,
  retries, batch processing, response normalization).
- Generic "AI productivity" features unrelated to mathematical proof
  workflows.
