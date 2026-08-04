# Compute resources (template — edit for your setup)

Describe what you can run, how to reach it, and the things that are
always forgotten. Be specific: this file exists so that a fresh
session does not have to rediscover your environment.

## This machine

- OS / shell:
- What counts as "heavy" here (RAM, cores):
- How many heavy jobs at once before it becomes a problem:
- Anything that has crashed it before:

## Remote compute

For each host:

- Name and how to reach it (jump hosts, VPN requirements, whether key
  auth is set up):
- Cores / RAM:
- Where large files go:
- What software is installed, and **the exact path to it if it is not
  on `PATH`** — this is the single most common time-waster:
- How to launch something that survives disconnection:
- How to kill it safely:

Example of the level of detail worth writing down:

> Sage is installed via mamba but is **not** on `PATH`. Use the full
> path `<...>/envs/sage/bin/sage -python script.py`; plain `python3`
> silently lacks the Sage globals and fails at the first use with a
> `NameError`, often an hour into a run.

## Delegated tooling

- Second-family CLI available? How invoked, and does its job state
  survive a crash?
- Which browser model the human runs, and at what setting:
- How consultation answers come back (share link? paste? a fetch
  script?):

## Gotchas

A running list. Each entry: the symptom, the cause, the fix. Anything
that cost you more than ten minutes once will cost the next session
the same unless it is written here.
