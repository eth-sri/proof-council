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

ProofCouncil helps you run configurable proof agents, inspect execution traces, review costs, and iterate on workflow presets locally. It was used to create an agent for the Second Batch of First Proof. You may find more information about ProofCouncil in the accompanying paper ProofCouncil.pdf, which can be found in the top folder of this repository.

## Quick Start

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

Saved problems live in `problems/`. Run artifacts are written under `outputs/` by default.

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

The default submission run uses `configs/workflows/firstproof_submission.yaml` with adaptive continuation stages up to the submission cap. Override `FIRSTPROOF_WORKFLOW` only when you want a different preset.

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
