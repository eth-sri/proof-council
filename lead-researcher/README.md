# The lead-researcher knowledge pools

`CLI_LEAD_RESEARCHER.md` describes the *mode* — Claude as lead
researcher on an open problem, with the human running browser
consultations. This directory holds the **knowledge that mode
accumulates**, split into two pools with different lifetimes and
different audiences.

|  | **Playbook** (`playbook/`) | **Local pool** (`local/`) |
|---|---|---|
| Contains | strategies that work for *anyone* running this mode | this machine, this person, this institution |
| Examples | packet spec, adversarial timing, verification discipline | compute-server hostnames, a PI's presentation preferences, paths |
| Tracked? | **yes** — committed, reviewed, improved over time | **no** — gitignored |
| Audience | every user of the package | one user |

The **mechanism is public; the content of the local pool is not**. A
new user clones the repo, gets the whole playbook for free, and copies
`local.example/` to `local/` to describe their own resources.

## How Claude uses these

At the start of a lead-researcher session, and again after any
compaction:

1. Read `playbook/00_index.md` — general strategy, always applicable.
2. Read every file in `local/` if it exists — machine-specific
   resources and personal preferences. **Local overrides playbook**
   where they conflict, since the playbook cannot know your setup.
3. Read the project's own state files (`SESSION_STATE.md`, the
   results ledger). Project state is neither general nor local; it
   lives with the project.

A project may also keep a working-directory `CLAUDE.md` naming its
state files, so the entry point survives compaction without anyone
having to remember it.

## The curation loop

This is the part that makes the split worth having.

```
   a session learns something
             |
             v
   is it true for anyone running this mode?
        /                        \
      yes                         no
       |                           |
       v                           v
  playbook/  (commit,          local/  (gitignored)
  reviewed, shared)                |
       ^                           |
       |    it keeps recurring     |
       +---------------------------+
             across projects
```

Rules:

- **Default to local.** A learning from one project is not yet
  general. Write it down where it happened.
- **Promote on the second independent occurrence.** When the same
  lesson shows up in a second project or a second machine, it has
  earned a playbook entry. Rewrite it abstractly on promotion — strip
  hostnames, paths, and names.
- **Every playbook entry cites its provenance**: which project and
  roughly when it was learned. An entry nobody can trace is an entry
  nobody can revise.
- **Prune.** A playbook entry that turns out to be wrong or
  situational gets deleted or demoted, not softened. The value of the
  playbook is that it is short enough to read every session.

## Layout

```
lead-researcher/
  README.md              this file — the mechanism
  playbook/              GENERAL pool (tracked)
    00_index.md          read this first; one-line summary of each entry
    packets.md           consultation packets: spec, dispatch, return
    delegation.md        which channel for which job
    verification.md      what "verified" is allowed to mean
    adversarial.md       red-teaming: when, against what, how to read it
    continuity.md        surviving compaction and crashes
  local.example/         TEMPLATE for the local pool (tracked)
    README.md
    resources.md         describe your compute
    preferences.md       describe how the human wants to work
  local/                 YOUR local pool (gitignored — create from the example)
```

## Setting up the local pool

```bash
cp -r lead-researcher/local.example lead-researcher/local
$EDITOR lead-researcher/local/resources.md
$EDITOR lead-researcher/local/preferences.md
```

Nothing in `local/` is ever committed. Put credentials in your
environment or a secrets file, not here — `local/` should describe
*where things are and how to reach them*, not hold secrets.
