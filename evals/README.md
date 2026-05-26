# Agent regression evals

Per SPEC §13. One YAML per regression case, runnable via
`scripts/run_evals.py`. The `EvalJudge` agent
(`src/proofstack/agents/eval_judge.py`) scores each case's output
against a free-text success criterion.

## Run

```bash
# All cases
uv run python scripts/run_evals.py

# One case
uv run python scripts/run_evals.py --case validator/stable_graphs_g8_latex_escapes_in_findings

# All cases under one agent
uv run python scripts/run_evals.py --agent validator

# Tag filter (repeatable)
uv run python scripts/run_evals.py --tag regression --tag parser

# CI: JSONL output, non-zero exit on any failure
uv run python scripts/run_evals.py --ci
```

Each case run produces an `outputs/evals/eval-<id>/` dir with the
standard JSONL event-stream layout (including `run.start` / `run.end`
events and a `run-metadata.json` with `mode: "eval"`), so the dev UI
inspects eval runs unchanged. Launch the UI with
`--runs-root outputs/evals` to see them in the Runs tab.

## Add a case

1. Pick the agent. Files live under `evals/agents/<agent_name>/`.
2. Drop fixtures under `evals/agents/<agent_name>/assets/`.
3. Write the YAML — minimum:

```yaml
agent: proofstack.agents.<module>.<ClassName>
case_id: short_descriptive_id
description: |
  One paragraph. Why does this case exist? What real failure was it
  capturing? Link the source run / call_id when capturing from a real
  failure.
tags: [regression, ...]

inputs:
  problem_path: assets/problem.txt    # _path suffix → file content substituted in
  solution: "(inline string is also fine)"

# Optional: skip the model call entirely. The runner injects the file's
# contents into <agent>.parse_output(raw_text, inp). Use this for any
# parser-only test where you've captured a known-tricky model response.
fixture_raw_response: assets/raw_response.txt

# Optional: structural assertions checked BEFORE the judge. If any
# fails, the case fails immediately (no judge call → faster, cheaper,
# clearer signal). See run_evals.py:HardCheck.
hard_checks:
  - field: findings
    min_length: 8
  - field: findings
    where: {verdict: wrong}
    min_count: 2

success_criteria: |
  Free-text description that the EvalJudge will score the agent
  output against. Be concrete: name specific values, specific
  required fields, specific failure modes. Empty / "I don't know"
  outputs should be FAIL, not inconclusive — say so explicitly.

# Optional. Default 0.05 USD per case (judge model only in fixture mode).
budget_usd: 0.05

# Optional. Default is whatever the EvalJudge's MODEL is.
judge_model: models/openai/gpt-54-mini
```

## Two evaluation modes

- **fixture** (`fixture_raw_response: ...`) — runner reads the file
  and calls `agent.parse_output(raw_text, inp)` directly. No model
  call, no API cost beyond the judge. Canonical for parser
  regressions.
- **live** (no fixture) — runner constructs the agent and invokes
  `await agent(**inputs)` normally. Tests the agent and the model
  together. Slower, costlier, flakier.

Use fixture by default when you have a captured raw response.

## Verdict semantics

The judge returns one of:

- **pass** — output clearly satisfies the criterion.
- **fail** — output clearly violates the criterion, OR is empty/
  malformed in a way the criterion says should not happen.
- **inconclusive** — judge cannot decide. Treated as a non-failure
  for CI exit code, but flagged in the summary. The judge is
  instructed to prefer `inconclusive` over a guess; if you see this
  often on the same case, sharpen the criterion.

Hard-check failures short-circuit to **fail** without invoking the
judge.

## Capturing from a real run

Manual procedure for now (`capture_eval_case.py` is on the SPEC §13
todo list):

1. Find the bad call in the dev UI:
   `http://127.0.0.1:5002/run/<run-id>/call/<call-id>`.
2. Copy the recorded files:
   ```bash
   mkdir -p evals/agents/<agent>/assets
   cp outputs/<run-id>/agents/<dir>/input.json     evals/agents/<agent>/assets/<id>_input.json
   cp outputs/<run-id>/agents/<dir>/raw_response.txt evals/agents/<agent>/assets/<id>_raw_response.txt
   ```
3. Write the YAML pointing at those assets.
4. Commit.

## See also

- SPEC §13 — design rationale and roadmap.
- `src/proofstack/agents/eval_judge.py` — judge prompt + schema.
- `scripts/run_evals.py` — runner internals (hard checks, fixture
  mode, judge invocation).
