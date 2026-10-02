# Public release checklist

This source tree contains implementation code, reusable documentation, synthetic
test fixtures, and one generic example problem. Private research inputs,
manuscripts, research archives, run logs, and infrastructure notes should be
stored outside the repository.

## History is separate from the current tree

A cleanup commit removes material from subsequent source snapshots, not from
earlier commits, tags, or branches. Do not make a private development repository
public merely because its latest tree is clean. Keep that history private and
prepare a separate public repository from a reviewed source snapshot with clean
history. Do not mirror development refs into it. Repository visibility changes
and public pushes require explicit authorization.

## Before publishing

1. Review the exact tracked snapshot for research content, personal paths,
   credentials, generated artifacts, and problem-specific run outcomes. Preserve
   licenses and upstream attribution.
2. Use a Git source export from the reviewed commit, not a ZIP of the working
   directory. Ignored and untracked local files are not safe to publish.
3. Scan credentials and review binary files separately. Pattern matching is not
   proof that a release is free of secrets. If a real credential has entered
   history, revoke or rotate it before addressing its removal.
4. Review Docker build inputs separately. The root Dockerfile copies selected
   runtime paths; `.dockerignore` also excludes private data from the context.
   Neither file removes anything from Git history.
5. Run the offline test suite and inspect the final export before publication.
   Source cleanup is not a deployment smoke test or a security certification.

New local files in `problems/` are ignored except for the generic `example.txt`.
Tests should construct synthetic research inputs in temporary directories.
Ignore rules are guardrails, not access controls: force-adding files bypasses
them, and a working-directory archive can still include ignored data.
