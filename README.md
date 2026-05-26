# mathagents

ProofStack math-research workflow framework, built on the provider and tool
layer originally extracted from MathArena.

## Included

- `mathagents.APIClient` for normalized multi-provider model calls
- Config helpers:
  - `mathagents.load_yaml_config(...)`
  - `mathagents.load_solver_config(...)`
- Workflow agents under `src/proofstack/`
- Workflow presets under `configs/workflows/`
- Curated high-end model configs under `configs/models/`
- Kept tool modules under `src/mathagents/tools/`

The old MathArena solver stack has been removed. New agents should be written
as workflow YAML using `ConfigurablePromptAgent`, `ConfigurableCLIAgent`,
`DAGWorkflow`, and reusable deterministic helpers.

## Install

We use[ `uv`](https://github.com/astral-sh/uv) for environment and dependency management.

## API Keys And Environment Variables

`APIClient` reads provider credentials from environment variables based on the `api:` field in a model config.

### Provider keys

| `api` value | Environment variable |
| --- | --- |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `google` | `GOOGLE_API_KEY` |
| `xai` | `XAI_API_KEY` |
| `glm` | `GLM_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `deepseek_special` | `DEEPSEEK_API_KEY` |
| `moonshot` | `MOONSHOT_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `together` | `TOGETHER_API_KEY` |
| `stepfun` | `STEPFUN_API_KEY` |
| `tiiuae` | `TIIUAE_API_KEY` |
| `sri` | `SRI_API_KEY` |
| `custom` | whatever is named by `api_key_env` in the config |
| `vllm` | no API key |

### Tool-specific environment variables

These are only needed if you use the corresponding kept tools.

| Tool/module | Environment variable | Purpose |
| --- | --- | --- |
| `paper_search.py` | `S2_API_KEY` | Semantic Scholar API access |

### Minimal setup example

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export XAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export OPENROUTER_API_KEY=...
```

You only need to set the variables for the providers you actually use.

## Config Layout

Model configs live under `configs/models/...` and map directly to
`APIClient` kwargs.

Example refs that currently exist:

- `models/openai/gpt-54`
- `models/openai/gpt-54-pro`
- `models/openai/gpt-54-mini`
- `models/anthropic/opus_47_max`
- `models/gemini/gemini-31-pro`
- `models/xai/grok-41-fast-reasoning`

Workflow presets live under `configs/workflows/...`. Read
`configs/workflows/instructions.md` before creating or editing workflow YAML.

## Running Workflows

```bash
uv run python scripts/run_workflow.py \
  --workflow configs/workflows/nimble_proof.yaml \
  --problem "Prove that there are infinitely many primes."
```

Continue an existing run in place:

```bash
uv run python scripts/run_workflow.py \
  --workflow configs/workflows/author_critic.yaml \
  --restart-from 20260523-120000 \
  --input n_rounds=8
```

Add `--restart-copy --run-id <new-run-id>` to continue from a copied run
directory while leaving the original artifacts untouched.

## First Proof AWS Harness

The First Proof harness should build from the repository root:

```bash
docker build -t mathagents-firstproof .
```

The root `Dockerfile` starts `scripts/firstproof_entrypoint.py`
automatically. At runtime it reads `/data/input/input.json`, writes the
required aggregate files to `/data/output`, and preserves detailed per-problem
workflow outputs under `/data/output/workflow_runs/`.

The image sets `PROOFSTACK_SANDBOX_BACKEND=subprocess` because the First Proof
harness does not mount a Docker socket into the submission container. The
default First Proof workflow is `author_critic_long`, the AC workflow configured
for subprocess compute execution inside the already-isolated submission
container. The adapter also defaults the Compute Worker's Codex sandbox mode to
`docker-bypass`; Codex's own bubblewrap sandbox cannot create nested namespaces
inside the First Proof container.

The Compute Worker has the open-source CAS stack on PATH inside the submission
container:

- `sage` — SageMath.
- `gap` — GAP standalone.
- `singular` and `gp` — standalone CAS backends.

The Author and Compute prompts list these explicitly so the agent reaches for
them rather than fighting with `sympy` on symbolic algebra it can't do.

Required provider secrets are passed through Docker environment variables, for
example via `docker run --env-file ...`. The current model configs may require:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY` for Gemini/Google configs, or `GEMINI_API_KEY` if using a
  custom config that expects it
- `ANTHROPIC_API_KEY`

Optional First Proof runtime overrides:

- `FIRSTPROOF_MAX_PARALLEL` (default `6`)
- `FIRSTPROOF_PAGE_LIMIT` (default `12`)
- `FIRSTPROOF_BUDGET_USD_PER_QUESTION` (default `1000`)
- `FIRSTPROOF_N_ROUNDS` (default `10`, initial staged target per problem)
- `FIRSTPROOF_ROUND_BATCH_SIZE` (default `5`; unsolved problems are resumed
  in cumulative batches, e.g. 5 rounds then 10 rounds)
- `FIRSTPROOF_ADAPTIVE_CONTINUATION` (default `true`; after the initial
  `FIRSTPROOF_N_ROUNDS`, keep resuming unsolved problems in additional
  `FIRSTPROOF_ROUND_BATCH_SIZE` batches while time and budget remain)
- `FIRSTPROOF_ADAPTIVE_MAX_ROUNDS` (default `200`; safety cap for adaptive
  continuation, normally reached only if time and budget do not stop first)
- `FIRSTPROOF_WORKFLOW` (default `author_critic_long`)
- `FIRSTPROOF_COMPUTE_CODEX_SANDBOX` (default `docker-bypass`; use
  `workspace-write` only on hosts where Codex/bubblewrap sandboxing is known to
  work)

The Author/Critic workflow also surfaces the LaTeX output contract to the
models: at most 12 pages by default, `\documentclass[12pt]{article}`,
plain `fullpage` permitted, and no other margin/layout, line-spacing, or
font-size changes.

The First Proof adapter runs problems in staged round batches by default:
each problem first gets 5 rounds, its raw workflow answer is preserved under
`/data/output/staged_solutions/rounds-005/`, and only problems that did not
reach Author/Critic agreement (`early_stopped` is not `true`) are resumed for
the next batch. The initial target is `FIRSTPROOF_N_ROUNDS` (default 10), after
which adaptive continuation keeps giving unsolved problems more cumulative
batches until Author/Critic agreement, the per-problem budget, the internal
deadline, or `FIRSTPROOF_ADAPTIVE_MAX_ROUNDS` stops them. For AC workflows,
non-final batch boundaries stop after Critic review rather than suppressing
terminal Council/Compute requests. Each compiling batch answer is promoted into
the top-level `.tex` and aggregate JSON snapshots, and finalization keeps the
best verified batch answer if a later continuation regresses. Slow first-stage
runs do not hold a global stage barrier; free worker slots can keep other
unsolved problems moving through later batches.

Local adapter smoke test without API calls:

```bash
python scripts/smoke_firstproof_adapter.py
```

## Output Viewer App

Use the local developer dashboard for workflow runs and presets:

```bash
uv run python app/dev.py
```

## Using The API Client

```python
from mathagents import APIClient, load_solver_config

cfg = load_solver_config("models/openai/gpt-54")
client = APIClient(**cfg)
```

## Tools Package

Reusable tools live under `src/mathagents/tools/` and are wired into
ProofStack through workflow/tool configuration.

## Code Execution Setup

The `execute_code` / `execute_code_long` tool in [src/mathagents/tools/code_execution.py](src/mathagents/tools/code_execution.py) needs a sandbox backend.

Preferred setup:

- use Modal for remote sandboxed code execution
- follow the Modal quickstart: <https://modal.com/docs/guide>

Local fallback:

- the tool can also fall back to Docker
- if you want that path, you need a local image tagged `mathagents-docker`

Example:

```bash
docker build -t mathagents-docker docker/
```

The current code expects:

- Modal app name: `project-euler-mathagents`
- Docker image name: `mathagents-docker`

If you do not configure either Modal or Docker, code-execution-based agents/tools will fail at runtime.
