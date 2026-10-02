# Writing prompt provenance and maintenance

Model-facing templates are reviewed text. Changes require maintainer review;
comments and provenance belong here rather than in the templates. `{{...}}`
spans are substitution placeholders.

## Writing guidance

`WRITING_GUIDANCE.md` builds on Johannes Schmitt's writing guide and the
writing-guidance sections shared by `solve` and `solve_api` in Jasper
Dekoninck's
[`improofbench` commit 35bbb047](https://gitlab.com/jo314schmitt/improofbench/-/blob/35bbb04779d98a341cf748209f6d08dfd6d66e93/config/open_problems/prompts.yaml).
The adaptation concerns exposition, not the upstream solver's task or output
instructions. Preserve attribution when revising these materials.

The guidance preserves mathematical scope and attribution, distinguishes
expository changes from mathematical repairs, and covers computer-assisted
arguments. It does not assume that an input manuscript is correct.

## Legacy WriteupLoop templates

- `rewrite-wrapper.txt` applies the guide and requests explicit flags for
  unresolved issues.
- `cold-referee.txt` asks an independent model to identify errors and provides
  machine-readable verdict sentinels.
- `repair.txt` addresses the referee's findings and reports unresolved
  critical errors explicitly.
- `research-notes-block.txt` provides optional background; the manuscript
  remains the text being rewritten.

The legacy loop's compile gate, bounded repairs, and original-document fallback
are implemented in `writeup_loop.py`. Model declarations are not independent
mathematical verification. See `docs/cleanup_session.md` and
`docs/firstproof_batch3_workflow.md` for the current cleanup and submission
contracts. Presets, rather than historical test settings, select models and
page limits.
