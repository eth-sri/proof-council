# Claude Science Cohort — application draft (ProofCouncil)

Deadline: **July 15, 2026** · notifications by July 31 · project period Sept 1 – Dec 1, 2026.

Review criteria (from Anthropic's help center, for the broader AI for Science program): scientific merit (team credentials in subject area and with AI), potential impact, technical feasibility (value of the model for the use case), biosecurity screening.

Field contents below are plain text, ready to paste into the Google Form: no markdown syntax inside fields, no hard line-wraps within paragraphs, newlines only for basic list formatting. Word counts verified with wc: team 290/300, proposal 487/500, Claude-usage 193/300, acceleration 140/200, impact 71/200, applications 64/200.

---

## Research team — description (<300 words)

Our team combines research mathematics with hands-on LLM evaluation and agent engineering, and has collaborated for over a year on the MathArena and IMProofBench projects.

Johannes Schmitt (lead) is an algebraic geometer at ETH Zurich (SwissMAP Research Fellow; PhD 2019, ETH Zurich) working on moduli spaces of curves, and maintainer of the SageMath package admcycles. He leads IMProofBench, a benchmark of research-level proof problems authored and graded by expert mathematicians, and is a co-author of the Leiden Declaration on AI and Mathematics (June 2026, endorsed by the International Mathematical Union). He recently published an inequality on descendant integrals discovered and proved in collaboration with AI (arXiv:2512.14575).

Tim Gehrunger is a postdoctoral researcher in arithmetic geometry at ETH Zurich (PhD 2025, ETH Zurich; publications in the Journal of Number Theory and Mathematics of Computation), IMProofBench co-organizer, MathArena co-author, and contributor to Humanity's Last Exam.

Jasper Dekoninck is a PhD student in the Secure, Reliable, and Intelligent Systems Lab at ETH Zurich working on LLM evaluation and mathematical reasoning: co-creator of MathArena, first author of the Open Proof Corpus (5,000+ human-graded LLM proofs) and of "Proof or Bluff?", with publications at ICLR (Spotlight) and NeurIPS.

Together we built ProofCouncil: in the First Proof Foundation's Second Batch benchmark (June 2026), where autonomous AI systems were graded double-blind by expert referees on unpublished research-level problems, it received the most passing grades of the four evaluated systems (6 of 10 problems, two rated "essentially flawless").

We work with a broader circle of mathematician collaborators who beta-test our workflows, report issues, and contribute improvements, including First Proof teammates Uri Kreitner and Gergely Bérczi, and Liam Price, who obtained first solutions to several open Erdős problems via autonomous AI queries (arXiv:2605.00301, joint with Terence Tao et al.).

## Key team members using Claude Science

Johannes Schmitt, SwissMAP Research Fellow (Postdoc), ETH Zurich: project lead; workflow design, open-problem selection, mathematical refereeing of outputs, Lean formalization.

Tim Gehrunger, Postdoctoral researcher, ETH Zurich: problem curation, mathematical evaluation of generated proofs, Lean formalization experiments.

Jasper Dekoninck, PhD student, ETH Zurich: agent engineering, evaluation methodology, cost/performance benchmarking.

Liam Price, independent mathematician (one of several external collaborators in this role): beta-testing workflows on open Erdős problems, feedback and small code contributions.

## Profile links

Johannes Schmitt: https://scholar.google.com/citations?user=Kn9HIRMAAAAJ and https://johannesschmitt.gitlab.io/

Tim Gehrunger: https://people.math.ethz.ch/~timge/ and https://scholar.google.com/citations?user=6yv7gCIAAAAJ

Jasper Dekoninck: https://www.sri.inf.ethz.ch/people/jasper and https://scholar.google.com/citations?user=878R760AAAAJ and https://github.com/JasperDekoninck

Team projects: https://github.com/eth-sri/proof-council and https://matharena.ai and https://improofbench.math.ethz.ch

---

## Research proposal (<500 words)

Scientific question. Can autonomous multi-agent LLM systems solve open research-level mathematics problems and deliver the results in a form working mathematicians can verify, audit, and build on? Concretely:
(a) which agent workflows yield reliable proofs rather than plausible-looking failures,
(b) how can AI-generated proofs be aligned with academic writing standards to be readable and refereeable, and
(c) can the proofs be accompanied by partial formalizations in Lean, a proof assistant that verifies mathematical arguments by machine, giving checkable certificates for key steps?

Starting point. Our open-source system ProofCouncil orchestrates Author, Critic, Council, and Compute agents that draft, attack, and revise a manuscript over many rounds until the critics are satisfied; every run is checkpointed, resumable, and fully inspectable. In the First Proof Foundation's Second Batch benchmark (June 2026, 1stproof.org), where autonomous systems were graded double-blind by expert referees on unpublished research problems, ProofCouncil received the most passing grades of the four systems evaluated (6 of 10, two rated "essentially flawless"; official report: arXiv:2606.18119). Claude Opus 4.7 served as a council seat in our entry; the proposed work uses current Claude models, with Fable 5 as Author and Opus-class critics. Claude Science is a natural environment for this work: our tools are executable (Lean, SageMath, Python, LaTeX), and its provenance layer makes every proof, computation, and formalization attempt an auditable scientific object.

Methodology.
1. Open-problem campaigns: run Claude-centric ProofCouncil configurations on curated open problems (Erdős problems and questions from collaborators in algebraic and arithmetic geometry), with team mathematicians refereeing all outputs.
2. Lean agent: add a formalization role that produces mathlib-based Lean 4 proofs of central lemmas alongside the LaTeX manuscript via compile-and-repair loops. Our preliminary testing suggests Claude Fable 5 is unusually strong at Lean 4; a formalization of an Erdős-problem case study, produced with agentic AI workflows, serves as a template in our repository.
3. Proof communication: controlled prompting and steering experiments to align generated manuscripts with academic standards (structure, citation practice, honesty about gaps), graded by expert mathematicians with IMProofBench-style rubrics.
4. Human in the loop: extend our existing human-agent node and dashboard so mathematicians without technical background can participate as advisor or critic inside a live run.

Expected outcomes and deliverables.
(i) Solutions or substantial partial progress on open problems, published with full disclosure following the Leiden Declaration on AI and Mathematics;
(ii) an open-source (MIT) release of the Claude-integrated ProofCouncil including the Lean agent;
(iii) a systematic evaluation of Claude-based workflow configurations (solve rate, cost, referee-assessed readability);
(iv) participation in the First Proof Third Batch (Aug-Oct 2026), overlapping the program period and providing an external, double-blind refereed milestone;
(v) a best-practices report on generating and communicating AI-produced mathematics.

Timeline.
September: integrate Claude across all workflow roles; baseline campaigns on Erdős-style problems; begin Lean agent development.
October: Lean agent evaluation; proof-writing experiments; First Proof Third Batch participation.
November: human-in-the-loop studies with external mathematician collaborators; final evaluation, best-practices report, and public release by December 1.

---

## How will Claude's capabilities be used? (300 words max)

Claude models will operate as the agents inside every role of ProofCouncil's workflows: the Author (Claude Fable 5 drafting and revising a LaTeX manuscript across many critiqued rounds; its API profile and tool functions fit neatly into our existing code), stateful and fresh Critics hunting for gaps independently of the Author, Council seats cross-examining other models' reviews (Claude Opus 4.7 already served in this role in our First Proof entry), and an Advisor proposing strategies and literature. The central new development is a Lean agent: Claude, running through Claude Code as a sandboxed CLI worker (a mode our runtime already supports), writes and repairs mathlib-based Lean 4 formalizations of the manuscript's key lemmas in a compile-check loop, so successful runs ship machine-checkable certificates alongside prose. Claude will additionally drive computational exploration through our sandboxed code-execution tools (Python/SageMath CAS experiments, counterexample search) and web search for literature grounding. All calls run through our centralized API client with per-run cost accounting, checkpointing, and full trace inspection in our dashboard, keeping every Claude contribution auditable end to end. We will also evaluate Claude Science itself as the human-facing entry point for mathematician collaborators without technical backgrounds.

## How will Claude Science accelerate/enhance the research? (200 words max)

Claude Science is valuable to us because mathematics has unusually strict verification requirements: it can connect Claude to our existing executable research stack (Lean 4 with mathlib, SageMath, Python counterexample searches, LaTeX manuscript generation) while preserving provenance for the code, environment, and conversation behind each result, turning proof generation from an opaque chat transcript into a reproducible scientific workflow. The credits let us test the expensive part systematically: our "Proof or Bluff?" study showed that directly prompted frontier models produce confident but flawed proofs, and in the First Proof benchmark our author-critic-council workflow outscored direct use of a frontier model, so the open question is whether Claude-centered workflows plus machine checking can convert raw capability into mathematics that experts can audit. The project thus doubles as a product-relevant evaluation of Claude Science and Claude Code on long-horizon, tool-using, formal-science workloads.

---

## Potential scientific impact (200 words max)

If successful, the project is a proof of concept that open mathematical conjectures can be solved by AI agent systems and, equally important, delivered in a form (readable manuscripts plus machine-checkable Lean certificates for key steps) that other researchers can efficiently audit, trust, and build upon. Since verification by human referees is becoming the bottleneck as AI-generated mathematics multiplies, workflows producing auditable, partially formalized output address the community's most pressing need.

## Applications beyond pure discovery (200 words max)

Beyond individual discoveries, the project explores how mathematical knowledge will be generated and communicated in the AI era, a transition affecting mathematicians worldwide with best practices still unsettled. Our open-source workflows, human-in-the-loop interface, and proof-writing guidelines aim to become reference practices for the field, and the auditable author-critic-formalize pattern transfers to other domains that need verified reasoning, from theoretical computer science to software verification.

---

## Credits requested + justification

We request the full $30,000. Costs are anchored in measured usage: our First Proof Second Batch run cost about $3,200 in model credits for ten problems within a 24-hour limit, and individual multi-round ProofCouncil runs on hard problems cost roughly $50-300 depending on rounds, context length, and reasoning effort. The credits fund four workstreams:
(1) Claude-centric open-problem campaigns: 50-100 workflow runs across a curated pool of candidate problems and formalization tasks, with 10-20 deeply refereed case studies selected for full mathematical review and Lean formalization attempts;
(2) the new Lean formalization agent, whose compile-and-repair loops are token-intensive (long mathlib contexts, repeated lemma repair);
(3) controlled proof-writing and steering experiments, requiring repeated runs for reliable comparisons;
(4) participation in the First Proof Third Batch (Aug-Oct 2026), which overlaps the project period and provides an external refereed milestone.
Every successful run is a candidate for expert review and possible publication; all runs, including failures, feed a public evaluation of which Claude workflow configurations produce reliable, readable, and auditable mathematics.

## Compute

Does your project require compute? **Yes**

Amount: **$2,000**. Modal compute for sandboxed agent tool execution: CAS experiments (SageMath/admcycles, numerical counterexample search), Lean 4 + mathlib compilation loops (CPU-heavy), and parallel verification runs.

## Biosecurity assessment

**None of the above.** (Leave the explanation field empty.)

## Additional information

ProofCouncil is fully open source (MIT, https://github.com/eth-sri/proof-council), and all results will be published with complete disclosure of AI involvement and auditable run traces, following the Leiden Declaration on AI and Mathematics, which our team co-authored. Our First Proof Second Batch performance is independently documented in the Foundation's official report (arXiv:2606.18119). The program period overlaps with the First Proof Third Batch (Aug-Oct 2026), giving the project an external, double-blind refereed evaluation milestone.

All problem statements, manuscripts, traces, and formalization artifacts used with program credits will be public, owned by the team, or used with explicit permission for processing under the program terms; private benchmark materials will only be used where the organizers permit this. We will gladly share non-confidential feedback on Claude Science and Claude Code for formal-science workflows involving Lean, SageMath, long proof manuscripts, and reproducible provenance.

We selected "Other: Mathematics" as the scientific field since no dedicated option exists. We believe mathematics is the cleanest formal-science testbed for Claude Science: success and failure can be independently audited, with referees judging the manuscript, Lean checking the formal lemmas, and every artifact carrying provenance.

---