# test_run_problems/

Staging area for problems submitted to the May 20th, 2026 pressure-test run
(First Proof Round 2 calibration). **All content is gitignored** except this
README and the local `.gitignore`. Do not commit submitter material, model
outputs, or PDFs to the repo.

## Layout

```
test_run_problems/
├── README.md                # this file (tracked)
├── .gitignore               # ignore-all-but-README (tracked)
├── processing_bin/          # drop raw attachments here on receipt
└── <slug>/                  # one folder per submission
    ├── metadata.yaml        # author, dates, status flags, prescreen verdict
    ├── submission/          # original material as received
    │   ├── problem.tex      # primary problem statement (any format)
    │   ├── email.txt        # pasted email body for record
    │   └── ...              # additional attachments
    ├── cleaned/
    │   └── problem_clean.tex   # LaTeX extracted by the prescreen agent
    └── prescreen/
        ├── report.md        # rendered prescreen report
        ├── report.pdf       # pandoc-rendered version (if available)
        └── response.json    # raw model output for replay
```

## Slug convention

`<lastname>_<keyword>` (lowercase, ASCII, hyphens for spaces).
Example: `mueller_kthy`, `schmitt_modforms`. Tie-breakers get a numeric
suffix (`mueller_kthy_2`).

## Metadata schema

See any existing `<slug>/metadata.yaml`. Required keys:

- `slug`, `submitter.{name,email}`, `received_date`, `submission_files[]`
- `prescreen.{status,model,verdict,summary,flags[]}` (status: `pending|done|failed`)
- `cleaned_confirmation.{status,date,notes}` (status: `pending|confirmed|revisions_requested`; whether the author has signed off on the cleaned restatement before the live test)
- `live_test.{selected,date,result}`
- `author_feedback.{date,verdict,notes}` (verdict: `correct|partial|off-track`; refers to the live-test solution, not the restatement)

## Confidentiality

Submitter identity (name, email, affiliation) is **never** included in
prompts to model providers. The prescreen pipeline reads only
`submission/problem.tex` (and adjacent attachments), not `metadata.yaml`.

## Workflow

Drop raw attachments into `processing_bin/`, paste the email content into
the conversation. Claude will slug the problem, file the attachments,
populate `metadata.yaml`, run the prescreen workflow, and surface the
verdict. See `configs/workflows/prescreen.yaml` for the prescreen DAG.
