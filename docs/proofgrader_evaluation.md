# Evaluate saved proofs with ProofGrader-style grading

To run a workflow YAML on benchmark problems and grade its outputs end to end,
see [Benchmark one harness YAML](harness_benchmark.md).

The evaluator implements the central procedure from
[ProofGrader](https://arxiv.org/html/2510.13888v2): prepare a problem-specific
marking scheme from the problem and reference solutions, then take the median of
five independent grades on an integer 0–7 scale. Valid alternative proofs should
receive full credit even when they differ from the references.

This is an independently written adaptation, with configurable models and prompts.
It does not invoke the upstream package or reproduce the paper's reported
accuracy. The published configuration used Gemini-2.5-Pro for marking schemes
and o3 for grading. The [upstream project](https://github.com/euclidgame/proofgrader)
provides a broader set of generation and evaluation scripts.

## Input

The fixed dataset directory, `opc_problems/` under the repository root, contains
**874 problems with reference solutions**. The 102
source problems without references have been removed from both tables. Keep
reference material on the evaluator side; the harness should receive only the
problem ID and statement.

Save candidate proofs as a JSON list or JSONL records:

```json
[
  {
    "problem_id": "an-id-from-problems.parquet",
    "candidate_id": "baseline",
    "solution": "The complete candidate proof, including its mathematical reasoning."
  }
]
```

`candidate_id` defaults to `default`. Distinct candidates for one problem must
have distinct candidate IDs. First Proof exports in the form
`{"solutions": [{"id": "...", "latex": "..."}]}` are also accepted.
Text is read from `solution` or `latex`; an `answer_tex` filesystem path is not
automatically dereferenced. Unknown IDs, duplicate candidate identities, blank
proofs, inconsistent tables, and missing references fail validation before any
grading call.

The evaluator obtains the problem and references from the local Parquet tables.
Caller-supplied replacement problem/reference fields are ignored. Candidate IDs,
harness names, and solver self-assessments are excluded from grading prompts.

## Dry run

From the repository root, install the evaluation dependency group with
`uv sync --group evaluation`. Then validate your candidate file without calling a
model:

```bash
uv run --group evaluation python scripts/evaluate_proofs.py \
  --solutions candidate_proofs.jsonl \
  --dry-run
```

The plan records dataset hashes, candidate/prompt hashes, eligible counts, and
the number of planned marking-scheme and grading calls. It requires no provider
credentials. Every invocation creates a unique directory under
`outputs/evaluations/` and prints its path, including dry runs. When called by
the harness benchmark, evaluation artifacts are stored in that benchmark run's
`grading/` subdirectory instead.

## Run an evaluation

Supply an existing ProofCouncil model configuration and an explicit spending
limit. Provider keys are read from the environment or the repository `.env` in
the same way as the normal workflow runner.

```bash
uv run --group evaluation python scripts/evaluate_proofs.py \
  --solutions candidate_proofs.jsonl \
  --model models/openai/gpt-6-astra \
  --samples 5 \
  --budget-usd 10 \
  --timeout-seconds 3600
```

The model and budget above are example settings. Grading quality with any chosen
model needs separate validation. `--scheme-model MODEL_CONFIG` selects a
different model for marking schemes. Model calls are sequential and go through
the existing `APICallAgent`/`APIClient`, retaining provider logs, token accounting,
and cooperative dollar/time limits. A single in-flight call can exceed the
remaining budget; a completed grade is retained when this happens.

The default ensemble size is five. `--samples 1` can be useful for a cheap pilot;
other positive odd sizes are accepted. Each grade uses a fresh conversation and
bypasses the ordinary response cache, so identical requests are not accidentally
counted as independent cached votes. Evaluator tools are disabled.

## Reuse marking schemes for harness comparisons

Marking-scheme generation sees the statement, references, and reference-quality
notes, **never the candidate proof**. One usable scheme is shared across all
candidates for a problem. If generation cannot produce a usable scheme, all
that problem's candidates receive the same ungraded outcome within the run.

To compare another harness under the same schemes, reuse the output file:

```bash
uv run --group evaluation python scripts/evaluate_proofs.py \
  --solutions challenger_proofs.jsonl \
  --marking-schemes outputs/evaluations/BASELINE_RUN/marking_schemes.json \
  --model models/openai/gpt-6-astra \
  --budget-usd 10
```

Replace `BASELINE_RUN` with the directory printed by the baseline run.
The frozen file must cover every eligible problem being evaluated. Each scheme
is checked against a hash of the exact problem, references, and quality notes.
Keep the grading model, prompts, and sample count fixed when comparing harnesses.
Inspect generated schemes before using scores as an optimization objective.

## Reference quality

All 874 problems remain in the dataset. **130 have reference-quality notes or
multiple reference versions** and are always skipped by the evaluator;
the other 744 are eligible. Flagged problems cannot be enabled through the CLI.
An unflagged reference is not an independently verified reference.

The dataset retains the two supplied reference variants for one flagged problem.
Its README also identifies duplicate statements under different IDs; keep
them in the same partition when constructing training/development/test splits.

## Results and failures

The output directory contains:

- `plan.json`: input fingerprints, settings, and planned calls.
- `run-metadata.json`: resolved model configurations and prompts.
- `marking_schemes.json`: reusable schemes and their provenance.
- `results.json`: one row per requested candidate, with status, median score,
  individual scores, reference warnings, and a detailed-result path.
- `grades/*.json`: checkpointed raw responses and per-sample explanations.
- `summary.json`: status counts, mean score over graded candidates, full-credit
  count, recorded spending, tokens, and accounting-completeness flags.
- Standard ProofCouncil events, agent artifacts, and provider receipts.

A score is emitted only when every requested sample has a valid integer grade
from 0 to 7. A Boolean, fractional, out-of-range, malformed, or missing grade
makes the ensemble incomplete with a null aggregate score. Failed samples are
not silently discarded or converted to mathematical score zero.

Provider errors stop the run, retaining completed results. Budget stops retain
the partially completed ensemble and mark later candidates `not_run`. Schemes
that cannot be used and flagged references have distinct statuses. The summary
reports these counts separately: compare coverage as well as the mean, because
the mean includes only graded candidates.

Recorded usage is reconciled with provider receipts. Missing usage and damaged
logs make `accounting_complete` false; known totals are retained even if a log
ends with a partial UTF-8 character. Zero recorded spending is not evidence of
zero cost when accounting is incomplete.

Ctrl-C and SIGTERM cancel the active evaluation, request provider cancellation,
and checkpoint the interrupted candidate. Remote cancellation remains best effort.

Automatically generated output directories prevent accidental overwrites or
budget resets inside an existing run. This first version checkpoints results but
does not automatically resume an interrupted batch. To continue, prepare a file
of the remaining candidates and invoke the evaluator with an explicit budget;
reuse available marking schemes where all eligible IDs are covered.

Exit status is 0 for a successful dry run or completed grading (including
explicitly skipped reference cases), 1 for incomplete grading, and 2 for invalid
inputs/configuration. Interruption returns 130. Full credit means the model assigned 7/7, not that a proof
has been formally verified. Calibration against expert-graded examples, such as
the authors' ProofBench dataset, remains necessary before treating the score as a
reliable optimization target.

## Code and tests

The implementation separates input validation (`proof_dataset.py`), model
responses (`proof_grader.py`), batch orchestration (`proof_grader_runner.py`),
evaluation setup and the benchmark subprocess (`proof_evaluation.py`),
the CLI (`scripts/evaluate_proofs.py`), and YAML prompts
(`configs/evaluators/proofgrader.yaml`). Shared configuration loading lives in
`configuration.py`; process cleanup, checkpoint writing, and usage accounting
live in `run_support.py`. Tests use synthetic proofs and fake
provider clients, in separate directories:

```bash
uv run --group evaluation --group dev pytest tests/open_proof_corpus tests/evaluation
```

These tests make no paid calls. They cover dataset filtering, reference isolation,
strict score parsing, independent samples, rubric reuse, budget stops, output
checkpoints, and the dry-run path.
