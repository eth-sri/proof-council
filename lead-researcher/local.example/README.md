# Your local knowledge pool (template)

Copy this directory to `lead-researcher/local/` and edit. `local/` is
gitignored; nothing in it is ever committed.

```bash
cp -r lead-researcher/local.example lead-researcher/local
```

## What belongs here

Anything true of **your** setup rather than of the mode:

- compute resources — hosts, how to reach them, what software is on
  them, and the three things you always forget
- how the human you work with wants to be worked with — writing
  preferences, what they consider interesting, how they want to be
  interrupted
- local conventions — commit trailers, where confidential work lives
- accounts, quotas, and which model you run in the browser

## What does not belong here

- **Secrets.** Describe *where things are and how to reach them*, not
  credentials. Keys go in the environment or a secrets file.
- **General strategy.** If it would help someone with entirely
  different hardware and a different collaborator, it belongs in
  `playbook/` instead — see the curation loop in `../README.md`.
- **Project state.** That lives with the project.

## Files

- `resources.md` — compute, tools, and their gotchas
- `preferences.md` — how the human wants to work

Add more if you need them; Claude reads every file in `local/`.
