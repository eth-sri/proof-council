<p align="center">
  <img src="app/static/favicon.png" alt="ProofCouncil" width="96">
</p>

<h1 align="center">ProofCouncil</h1>

<p align="center">
  A local app for running, editing, and inspecting math-research agents.
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.12+" src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="https://github.com/astral-sh/uv"><img alt="uv" src="https://img.shields.io/badge/Package%20manager-uv-DE5FE9"></a>
  <a href="LICENSE"><img alt="License MIT" src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

ProofCouncil helps you run configurable proof agents, inspect execution traces, review costs, and iterate on workflow presets locally.

## Quick Start

This repository contains the workflow implementation and generic examples, not
research inputs or results. See [Public Release](docs/public_release.md) before
publishing source or building a distribution.

Install dependencies with [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

Set the provider keys needed by the agent you plan to run. The app and CLI both read your shell environment and a local `.env` file.

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

Other supported key names include `XAI_API_KEY`, `GLM_API_KEY`, `DEEPSEEK_API_KEY`, `MOONSHOT_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`, `STEPFUN_API_KEY`, and `TIIUAE_API_KEY`. You only need keys for the models used by your selected workflow.

## Use The App

The app is the main way to use ProofCouncil.

```bash
uv run python app/dev.py
```

Open <http://127.0.0.1:5002>.

From the home screen:

- **Run Agent**: choose a workflow preset, add or select problems, provide any missing API keys, and launch one or more runs.
- **View Runs**: inspect status, cost, wallclock time, execution graphs, messages, files, and final outputs from `outputs/<run-id>/`.
- **Edit Agent**: edit workflow presets from `configs/workflows/` in the local visual/YAML editor.
- **Create New Agent**: start a new workflow preset from the app.

The agent editor has many features, including:
- Visual DAG editor with drag-and-drop nodes, customizable prompts, inputs, outputs, models, and more. The interface is created to work as smoothly as possible, with common actions like copying (Ctrl+C) and pasting (Ctrl+V) nodes, and undo/redo (Ctrl+Z / Ctrl+Y).
- Agents can directly be used to edit the underlying YAML of an agent, allowing you to instruct your personal agent to edit the workflow you are working on. The DAG editor will automatically update to reflect any changes made to the YAML, and vice versa.

Saved problems live in `problems/`. Only the generic `example.txt` is tracked;
other local problem files are ignored. Run artifacts are written under `outputs/`
by default and are also ignored. Keep private research archives outside the
repository rather than relying on ignore rules when sharing a folder.

## Run From The CLI

Use the CLI when you want a scriptable single-problem run.

```bash
uv run python scripts/run_workflow.py \
  --workflow author_critic \
  --problem problems/example.txt
```

You can also pass an inline problem:

```bash
uv run python scripts/run_workflow.py \
  --workflow author_critic \
  --problem-text "Prove that there are infinitely many primes." \
  --problem-id infinitely_many_primes
```

Run output goes to `outputs/<run-id>/`. Start the app afterward and open **View Runs** to inspect the trace.

To continue an existing run:

```bash
uv run python scripts/run_workflow.py \
  --workflow author_critic \
  --restart-from <run-id>
```

Workflow presets are in `configs/workflows/`. Pass either a preset name such as `author_critic` or a YAML path such as `configs/workflows/author_critic.yaml`.

### Research Model Defaults

The Author/Critic research presets and prescreen use
`models/openai/gpt-6-astra-pro` for their OpenAI research roles. This sends
`model: gpt-6-astra` with `reasoning.mode: pro`, `reasoning.effort: max`, and
automatic reasoning summaries through the Responses API in background mode.
The Fable Author preset uses `models/anthropic/fable_51_max` (Claude Fable 5.1).
The default Council is GPT-6 Astra Pro, Claude Fable 5.1
(`models/anthropic/fable_51`) and Gemini 3.1 Pro; every council seat has the
provider's code sandbox and web search (OpenAI `code_interpreter` +
`web_search_preview`, Anthropic `code_execution` + `web_search`, Gemini
`codeExecution` + `googleSearch`) for independent experiments and literature
checks. Compute runs `gpt-6-astra` at `xhigh` reasoning effort through the
Codex CLI, billed with the `models/openai/gpt-6-astra` rates.

`models/openai/gpt-6-astra` and `models/openai/gpt-6-astra-max` select standard
mode; the `-max` suffix sets maximum reasoning effort without enabling Pro.
The previous Sol configs remain available for explicitly selecting Sol.
The Astra Max and Pro configs retain the existing timeout policy: 11,400
seconds per background attempt, a 14,000-second tool-loop wallclock cap,
and a retry at `high` effort after a background timeout (still Pro for the
Pro config). Preset and run budgets remain separate limits.

Two failure modes of long Astra/Sol calls are handled in `APIClient`:

- OpenAI currently kills sol/astra background responses server-side at about
  60 minutes (`status: failed`, `error.code: server_error`, empty output,
  `usage: null`). With `background_server_kill_after_s: 3300` such a failure
  counts as a background timeout, so the retry runs at the downgraded
  `background_timeout_reasoning_efforts` (`high`) instead of dying again.
- A response that ends with `status: incomplete` and
  `incomplete_details.reason: max_output_tokens` is followed by one wrap-up
  turn (`openai_continue_on_max_output_tokens`, at
  `openai_max_output_token_continuation_effort: high`): the trailing reasoning
  items are dropped, the conversation is replayed with a "wrap up now, finish
  file edits, give the final visible output" user message, and both responses
  are billed.

Astra token accounting uses $10 input, $1 cached input, $12.50 cache writes,
and $50 output per million tokens. Above 272,000 input tokens, input rates
double and output rates increase by 1.5 for the full request. Pro mode bills
aggregated model work at these token rates; reasoning tokens are included in
output usage, not charged again separately. See the
[Astra model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra)
and [reasoning-mode guide](https://developers.openai.com/api/docs/guides/reasoning#reasoning-mode).

## Run First Proof

ProofCouncil includes a Docker setup for the First Proof harness. The harness expects one JSON file at `/data/input/input.json` and writes all results to `/data/output`.

### Input File

`input.json` can be either a list of problems or an object with a `problems` list. Each problem should include:

- `id`: a stable problem identifier used in output filenames.
- `latex`: the full problem statement. This can be a complete LaTeX document or just the problem text. This field is required.

Minimal example:

```json
{
  "problems": [
    {
      "id": "sqrt2",
      "latex": "Prove that \\sqrt{2} is irrational."
    },
    {
      "id": "infinitely-many-primes",
      "latex": "\\documentclass{article}\n\\begin{document}\nProve that there are infinitely many primes.\n\\end{document}"
    }
  ]
}
```

For the included smoke run, this file is already provided at
`smoke/input.json`.

### Secrets File

`smoke/secrets.env` is a local environment file for API keys. It is not an input problem file and should not be committed. The smoke workflow currently needs:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

Start from the template:

```bash
cp smoke/secrets.env.example smoke/secrets.env
# Fill in the keys in smoke/secrets.env.
```

### Smoke Run

The included end-to-end smoke check is:

```bash
./smoke/run_container.sh
```

This builds a Docker image, mounts `smoke/input.json` at `/data/input/input.json`, mounts `smoke/output_container/` at `/data/output`, passes `smoke/secrets.env` into the container, and runs `configs/workflows/firstproof_smoke_fast.yaml`.

Use the local non-Docker path when you only want to exercise the adapter:

```bash
./smoke/run_local.sh
```

### Harness Image

To build the image that the First Proof-style harness runs:

```bash
docker build -t proofcouncil-firstproof .
```

To run it manually with your own input and output directories:

```bash
docker run --rm \
  -v "$PWD/smoke/input.json":/data/input/input.json:ro \
  -v "$PWD/smoke/output_container":/data/output \
  --env-file smoke/secrets.env \
  proofcouncil-firstproof
```

A raw container run uses the default First Proof workflow. For smoke testing, prefer `./smoke/run_container.sh` because it selects the cheap workflow and sets the smoke-sized budget, page limit, and round count.

The submission image sets `FIRSTPROOF_WORKFLOW=firstproof_batch3` by default.
Its competition defaults are **10 simultaneous problems** and **$1,050 per
problem**, including research, council, Compute and cleanup ($10,500 planned
total for ten problems). These are cooperative spending limits, not a provider
billing guarantee. `FIRSTPROOF_MAX_PARALLEL` and
`FIRSTPROOF_BUDGET_USD_PER_QUESTION` override them for smaller tests.
It selects Astra Pro Author/Critic, one guide-driven rewrite followed by stateful
criticism and targeted repairs, and an exact-document 16-page submission gate.
Mathematical flaws return to research; unresolved attempts receive one unreviewed
partial-results rewrite at the shared 23-hour cutoff, with at most two mechanical
repairs and a preserved, exportable partial checkpoint. Explicit runtime environment
variables can select another preset, including `firstproof_smoke_fast` or the
legacy `firstproof_submission` workflow. The direct Python adapter retains its
legacy default when `FIRSTPROOF_WORKFLOW` is unset.

See
[the Batch 3 workflow notes](docs/firstproof_batch3_workflow.md) for phase
budgets, partial-result handling and remaining pre-submission tests. A container
run with credentials can incur API charges; use explicitly approved budgets.

Example:

```bash
docker run --rm \
  -v "$PWD/smoke/input.json":/data/input/input.json:ro \
  -v "$PWD/smoke/output_container":/data/output \
  --env-file smoke/secrets.env \
  proofcouncil-firstproof
```

Explicit `FIRSTPROOF_*` environment variables still override defaults.

First Proof outputs include per-problem `.tex` files, `solutions.json`, `run_summary.json`, `token_usage.jsonl`, and detailed workflow traces under `/data/output/workflow_runs/`.
