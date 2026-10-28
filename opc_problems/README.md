# Open Proof Corpus problems and reference solutions

For grading saved harness proofs against these references, see
[ProofGrader-style evaluation](../docs/proofgrader_evaluation.md).

All **874 problems with a nonempty author reference solution** from all six splits of the public
[INSAIT-Institute/OPC dataset](https://huggingface.co/datasets/INSAIT-Institute/OPC),
extracted from revision `dcc3b4804e2d126ea34b13e3e0cd998c3302644b`.
The release has 4,934 rows of model attempts covering 976 distinct problem IDs.
This directory excludes the 102 IDs without references and keeps the remaining
problems, supplied reference solutions, provenance, and quality notes.

| File | Contents |
| --- | --- |
| `problems.parquet` | 874 rows: unique problem ID, statement, competition, category, level, year, source, URL, and split memberships. |
| `reference_solutions.parquet` | 874 rows keyed by problem ID: nonempty reference text lists, issue/note lists, availability, and review status. |
| `manifest.json` | Resolved upstream commit, retrieval time, split counts, license, and source/output SHA-256 hashes. |
| `update_dataset.py` | Downloads the current release, retains only IDs with author references, and rebuilds both tables and the manifest. |

Parquet preserves multiline mathematical text and loads directly into a pandas
DataFrame. It also avoids Python-specific pickle files. No generated attempts,
model reasoning, grades, judge annotations, or usage logs are stored here.

## Load for the harness and evaluator

Run these examples from the ProofCouncil repository root with pandas and pyarrow
installed:

```python
from pathlib import Path
import pandas as pd

root = Path("opc_problems")
problems = pd.read_parquet(root / "problems.parquet")

# The harness receives only the problem statement and its identifier.
harness_inputs = problems[["problem_id", "problem"]]

# The evaluator can additionally receive the supplied references and warnings.
references = pd.read_parquet(root / "reference_solutions.parquet")
evaluation_data = problems.merge(references, on="problem_id", validate="one_to_one")

row = evaluation_data.iloc[0]
statement = row["problem"]
reference_texts = list(row["reference_solutions"])
```

The reference table uses a list column because one problem,
`BMOSL_2017_18`, has two distinct supplied reference texts. Neither is discarded.
For an evaluator expecting one string, choose a reference explicitly or join the
variants with labels. Every retained problem has a nonempty reference list, and
`has_reference_solution` is always true. Missing years use pandas' nullable
integer dtype.

For the existing batch runner's JSON input format, an optional solver-only export
is:

```python
harness_inputs.rename(columns={"problem_id": "id"}).to_json(
    "opc_harness_inputs.json", orient="records", force_ascii=False, indent=2
)
```

Do not pass the merged evaluator table to a solver prompt. The updater itself
does not run a model, judge a proof, or incur inference costs.

## Reference quality and benchmark splitting

There are **874 problems with references**, containing **875 distinct reference
texts**. The **102 problems without references are excluded from both tables**.
References come exclusively from OPC's `ground_truth_solution` field. The
upstream `solution` field contains generated attempts and is excluded, regardless
of its score.

`reference_status` has two values:

- `available` (744): a reference is supplied without an aggregated issue/note.
- `needs_review` (130): a reference is supplied with an upstream problem issue,
  reference issue/note, or more than one reference version.

"Available" does not mean independently verified. All nonempty `problem_issue`,
`ground_truth_issue`, and `ground_truth_note` values are aggregated over every
occurrence of the problem. In particular, 114 problems carry the PutnamBench
warning that rephrasing may make the original solution only tangentially related.
The source's grades of generated attempts do not validate the reference texts.
The benchmark and proof-evaluation CLIs always exclude the 130 `needs_review`
problems. These remain in the dataset for inspection and future review.

Statements and reference texts are preserved verbatim; whitespace-only missing
values and exact duplicates are removed from list fields. Some upstream
statements include answer cues, including `IMOSL_2020_30` and `IMOSL_2017_16`.
Those remain part of the original proof task.

`source_splits` records physical Hugging Face split membership;
`annotated_splits` records labels in the upstream `split` column. These differ
for some rows. The six source splits are `best_of_n`, `generic`, `matharena`,
`pass_at_n`, `putnambench`, and `test`. They overlap at the problem level and are
not a ready-made disjoint training/development/test partition for harness tuning.

There are 869 distinct statement texts across 874 retained IDs. When creating your own
partition, keep each identical-statement group together, including these pairs:

- `RMM_Extralist_2021_1` / `RMM_Extralist_2021_5`
- `USAMO_2010_1` / `JMO_2010_3`
- `USAMO_2017_4` / `JMO_2017_6`
- `USAMO_2014_4` / `JMO_2014_5`
- `USAMO_2019_4` / `JMO_2019_5`

The updater examines the full public Hugging Face release and selects its
reference-bearing problems. The upstream code repository also contains auxiliary
evaluation catalogs; these are outside this release. The counts here describe
the actual downloadable files; the project website advertises larger corpus totals.

## Update

From the ProofCouncil repository root:

```bash
uv run --script opc_problems/update_dataset.py
```

The script declares its pandas/pyarrow dependencies inline, so `uv` can install
them in an isolated environment. Alternatively, install `pandas>=2.2,<3` and
`pyarrow>=18` in your own environment and run the script with Python 3.12+.

To reproduce this source snapshot:

```bash
uv run --script opc_problems/update_dataset.py \
  --revision dcc3b4804e2d126ea34b13e3e0cd998c3302644b
```

Use `--output-dir PATH` to write elsewhere. The default is the script's directory.
All downloads use one resolved commit, even when updating from `main`. The script
discovers all configured split files, checks their row counts against the release
metadata, aggregates every occurrence of each problem, and excludes IDs without
a nonempty `ground_truth_solution`. It checks retained statement/metadata
consistency before replacing the data files. It retains no raw upstream shards.
Updates replace the two Parquet files
and manifest; consult the manifest for current counts after an update. The counts
in this README describe the checked-in snapshot. The manifest records the source
problem count and how many were excluded for missing references.

Tests live separately under `tests/open_proof_corpus/`. With pandas and pyarrow
installed, run the dataset tests from the repository root:

```bash
python -m unittest discover -s tests/open_proof_corpus -p test_dataset.py -v
```

## Attribution and license

Source: **Open Proof Corpus (OPC)**, INSAIT Institute; Jasper Dekoninck,
Ivo Petrov, Kristian Minchev, Mislav Balunovic, Martin Vechev, and the OPC
contributors. See the [dataset card](https://huggingface.co/datasets/INSAIT-Institute/OPC/blob/dcc3b4804e2d126ea34b13e3e0cd998c3302644b/README.md)
and [project website](https://proofcorpus.ai/).

The extracted dataset is provided under the upstream
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
This data license applies to the Parquet contents independently of the
ProofCouncil software license. Retain the attribution and applicable upstream
notices when redistributing the data.

Changes made here: selection of problems with nonempty author references,
projection to problem/reference fields, deduplication by problem ID,
aggregation of all reference variants and issue notes, nullable
integer years, and separation of solver and evaluator tables. No proofs were
rewritten or generated.
