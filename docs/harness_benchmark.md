# Benchmark one harness YAML

`scripts/benchmark_harness.py` runs one workflow YAML on selected OPC problems,
collects its proof outputs, and grades them with the existing
[ProofGrader-style evaluator](proofgrader_evaluation.md). The solver receives
the statement; reference solutions are added only when preparing evaluator
inputs. This evaluates a harness configuration; it does not optimize the YAML.
The dataset is fixed at `opc_problems/` under the repository root. Every run
creates a unique directory under `outputs/benchmarks/` and prints its path.

## Validate without model calls

Run from the repository root:

```bash
uv run --group evaluation python scripts/benchmark_harness.py \
  --workflow configs/workflows/author_critic.yaml \
  --limit 10 --seed 0 \
  --dry-run
```

This validates the dataset, selection, workflow structure, and any frozen
marking schemes, then writes the plan and selected statements. It needs no
provider credentials and makes no solver or evaluator calls. Each invocation
automatically creates a fresh output directory, including dry runs.
Dry-run validation does not fully resolve nested workflow/model dependencies or
check provider availability; those can still fail during execution.

## Run the harness and evaluator

Live execution requires explicit solver and grader budgets plus a grading model:

```bash
uv run --group evaluation python scripts/benchmark_harness.py \
  --workflow configs/workflows/author_critic.yaml \
  --limit 10 --seed 0 \
  --solver-budget-usd-per-problem 1 \
  --grader-model models/openai/gpt-6-astra \
  --grader-budget-usd 5 \
  --grader-timeout-seconds 3600 \
  --samples 5
```

These amounts are illustrative, not an estimate of what completing ten proofs
will cost. The nominal allocation is ten separate $1 solver budgets and one $5
budget for all marking schemes and grading. Solver models come from the YAML;
`--grader-model` only selects the evaluator. `--scheme-model` can override the
marking-scheme model. Provider credentials follow the normal workflow setup.
All automated tests use synthetic data or fake providers and make no paid calls.
The per-problem solver timeout is configured by `solver_timeout_seconds` in
`configs/evaluators/benchmark.yaml`, initially 600 seconds.

Problems run sequentially in separate subprocesses. Completed proofs are then
graded together. By default, the runner looks for a nonempty workflow output
named `solution`, `answer_tex`, or `solution_tex`. Use `--answer-field FIELD` to
select another output; dotted paths such as `final.proof` are supported.

Proof extraction automatically accepts inline text or a proof-file path. Path-like values
must resolve to an existing, nonempty file inside that problem's run directory.
Missing or unreadable proofs receive `missing_proof`, not a grade.

## Select comparable problems

The supplied snapshot has 874 reference-bearing IDs. Selection always excludes
130 flagged references, leaving 744 IDs and **739 unique statement texts**.
Exact duplicate statements are removed before sampling, retaining the first ID
in sorted order. `--limit 10 --seed 0` therefore selects the same ten statements
for the same dataset. Requesting more than the eligible count fails validation.

- `--all` selects all 739 eligible unique statements.
- Repeat `--problem-id ID` to select particular IDs instead of sampling.

`--limit`, `--all`, and `--problem-id` are mutually exclusive. Explicit IDs must
exist, be distinct, and have distinct statements. Flagged IDs cannot be selected.
Counts describe the supplied snapshot; `plan.json` records the
actual selection and dataset hashes for every run.

For harness comparisons, keep the selected IDs, dataset revision, grading model,
prompts, and sample count fixed. Reuse a baseline's marking schemes by adding:

```bash
--marking-schemes outputs/benchmarks/BASELINE_RUN/grading/marking_schemes.json
```

Replace `BASELINE_RUN` with the directory printed by the baseline run and supply
the comparison YAML. The frozen file must cover
every selected problem and match its statement, references, and quality notes.
A partially completed baseline may not have generated every required scheme.
Keep optimization and final assessment sets separate, including identical
statements under different IDs in the same partition.

## Inspect results

| Path under the output directory | Contents |
| --- | --- |
| `plan.json`, `workflow.yaml` | Settings, input hashes, planned calls, and a copy of the supplied YAML. |
| `problems.json` | Selected IDs and statements, without references. |
| `solver-runs/problem-*/` | Per-problem statement, workflow log, metadata, provider receipts, and artifacts. |
| `candidates.json` | Successfully extracted proof texts for evaluation. |
| `grading/` | Marking schemes, individual grades, evaluator logs, and accounting. |
| `grading.log`, `grading_process.json` | Evaluator subprocess output and execution status. |
| `results.json`, `summary.json` | Joined solver/grade outcomes and aggregate metrics. |

The summary's `full_credit_rate` is the number of 7/7 proofs divided by **all
selected problems**, including failures and ungraded problems. It is a possible
optimization objective. `mean_score_of_graded` excludes missing grades; compare
it alongside `grading_coverage` and the solver/grading status counts. Missing
proofs or grades are never assigned an artificial score of zero. Recorded costs,
tokens, and runtimes are reported separately, with an accounting-completeness
flag when usage may be missing.

The benchmark checkpoints completed outputs but does not automatically resume
or retry failed problems. Normal provider-level retry behavior still applies.
Dollar limits are cooperative: an in-flight call can overshoot. Wall-clock
timeouts terminate the subprocess group, with best-effort cancellation of work
already submitted to remote providers. Ctrl-C and SIGTERM use the same cleanup
and checkpoint path. Interrupted usage can remain unknown.

Statement-only solver inputs are not a filesystem security boundary. A workflow
with local file or shell tools can still access the repository's reference data.
The YAML snapshot also retains references to repository configs and other
dependencies; it is not a complete portable snapshot of the execution environment.

Full credit is a stochastic model judgment, not formal proof certification.
Use expert-graded examples to calibrate the evaluator before relying on small
score differences. Exit status is 0 for a successful dry run or a fully graded
batch, 1 for incomplete execution/grading, 2 for invalid inputs, and 130 for an
interruption.

Tests live separately under `tests/benchmark/`:

```bash
uv run --group dev pytest tests/benchmark tests/evaluation
```
