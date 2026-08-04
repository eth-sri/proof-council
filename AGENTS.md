# ProofCouncil — project description

This repository will be the implementation of ProofCouncil, an autonomous
math-research agent system targeting the **First Proof Foundation, Second Batch**
benchmark (June 2026). Beyond the benchmark, the same system is intended
to grow into a human-in-the-loop research assistant for mathematicians.

`configs/workflows/instructions.md` is the current source of truth for
workflow syntax and reusable YAML components.

---

## What this repo currently is

A ProofCouncil workflow framework built on the MathArena provider/tool
layer. The kept pieces are:

- `src/mathagents/api_client.py` — robust multi-provider client (OpenAI,
  Anthropic, Google, xAI, DeepSeek, GLM, Moonshot, Together, vLLM, …)
  with retries, batch processing, tool-call loops, and cost accounting.
- `src/proofstack/tools/` — local workflow tools such as code execution and
  persisted files.
- `configs/models/` — layered YAML model definitions with `base:`
  inheritance.
- `configs/workflows/` — DAG workflow presets. Read
  `configs/workflows/instructions.md` before creating or editing these.
- `app/` — Flask developer dashboard for workflow runs, presets, and traces.
- `scripts/run_workflow.py` — CLI entry point for workflow presets.

---

## Repo layout

```
src/mathagents/      # API client, config loader, and provider-side tools
src/proofstack/      # Workflow/agent runtime
configs/             # YAML configs (models/, tools/, workflows/)
app/                 # Flask developer dashboard
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
  --workflow author_critic \
  --problem "Prove that there are infinitely many primes."
```

Browse outputs and presets:

```bash
uv run python app/dev.py
```

### Local dev helpers (use these instead of ad-hoc shell)

To keep commands auto-approvable (raw `curl`/compound shell lines trigger
permission prompts every time), prefer the wrapper scripts:

- **Start the dashboard:** `scripts/run_dashboard.sh [PORT]` (default 5005),
  launched in **background mode** — not a raw
  `PYTHONPATH=… uv run python app/dev.py … > /tmp/… 2>&1 &` one-liner.
- **Dashboard health check:** `scripts/health.sh [PORT] [PATH]`
  (e.g. `scripts/health.sh 5005 /runs`) instead of a raw
  `curl … 127.0.0.1:… ; echo …` one-liner.
- **Run status snapshot:** `scripts/run_status.sh RUN [PORT]` — cache entries,
  finished flag, latest batch log tail, and live python procs in one shot.
- **Watch a run:** `scripts/watch_run.sh RUN [PORT] [MAX_POLLS]` — polls until
  the first node is cached or the run reports finished.
- **Verify resume state:** `scripts/verify_resume.sh RUN [PORT]` — orphaned CLI
  workers, finished flag, `resume.json` presence, run-detail HTTP, and whether the
  "Resume run" button renders.
- **Validate a preset:** `scripts/validate_preset.sh PRESET_YAML` — prints
  ok/errors/warnings. Use this instead of `uv run python -c "…validate…"`; inline
  `python -c` always requires manual approval (anti-bypass guard), a wrapper script
  does not.
- **Review the dashboard UI:** `scripts/ui_review.sh [PORT] [--probe]` — drives the
  running dashboard headlessly (Playwright) and screenshots the key states to
  `outputs/ui_review/`; Read the PNGs and fix what you see BEFORE handing UI
  changes to the user. The dashboard caches templates (no --debug), so restart
  it first when templates changed, or the screenshots show stale copy. On a
  fresh checkout, install the browser once with
  `uv run playwright install chromium`.

Do **not** hand-write monitoring loops (`for i in $(seq …); do … curl … done`)
or multi-tool status bundles inline — those control-flow shell blobs cannot be
allowlisted and prompt on every run. Add a flag/argument to one of these scripts
(or a new `scripts/*.sh`) instead.

When searching the codebase, prefer the built-in **Grep**/**Glob** tools over
shell `grep`/`find`, and avoid `cd`-prefixed or `;`/`&&`-bundled commands — a
single plain command auto-approves; a bundled one does not.

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
