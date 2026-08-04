# Delegation: which channel for which job

The mode has several ways to spend effort. They differ in cost,
latency, and what they are actually good at. Matching the job to the
channel is most of the skill.

| Channel | Costs | Good at | Bad at |
|---|---|---|---|
| **Browser packet** (human-run frontier model) | human minutes, high latency | long single-shot grinding: case analysis, meticulous referee passes, proof pushes on a set target ladder | anything iterative; anything needing your files |
| **Subagents** (parallel) | tokens | breadth: ideation councils, literature sweeps, independent replication, packet red-teaming | deep serial reasoning; anything where you cannot check the answer |
| **A second model family** (e.g. a CLI from another vendor) | tokens | reading *code* adversarially; reviewing a certificate; an independent architecture for replication | mathematics you cannot check |
| **Local/remote compute** | wall-clock | anything finite and exact | anything you have not made finite |
| **Yourself** | context | judgement, synthesis, deciding what is worth doing | brute enumeration |

## Rules that generalise

- **Do not delegate what you cannot check.** This is the same rule as
  in `verification.md` and it applies to choosing the channel, not
  just to reading the answer.
- **Make it finite, then compute it.** Most "open" subquestions have a
  finite shadow — one genus, one dimension, one bounded search. The
  finite shadow is usually decisive and always cheap by comparison.
- **Separate "can I compute the whole object?" from "what do I
  actually need?"** The second is usually far cheaper. A full
  enumeration at the next parameter value may be infeasible while the
  specific structural fact you need is one targeted query away.
- **Ideate before dispatching.** Parallel subagents proposing routes,
  then the best of those routes planted in the packet, beats sending
  the packet cold.
- **Red-team the packet before it goes out.** One subagent whose only
  brief is to find what a hostile referee would attack.

## When a long job goes silent

Do not just poll it. Read the code at the point where it is stuck and
ask whether that step is finite *in practice* — face-lattice
traversals, unbounded enumerations and exact hull computations in high
dimension can be finite in principle and hopeless in fact. Kill it and
say why; a running job that cannot finish also hides the fact that the
evidence you needed may already be in hand from a cheaper computation.

## Cost of parallelism

Parallel work has a real cost on the human's machine. Over-
parallelising local jobs — several heavy processes, several monitors,
several subagents with children — can take down the human's session.
Heavy compute belongs on dedicated hardware, niced; local work should
be one heavy job at a time; monitors should be few and infrequent.

The specifics of *your* hardware belong in `local/resources.md`, not
here.
