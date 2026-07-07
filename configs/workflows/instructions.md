# Editing workflow YAML

Keep workflow files small and explicit. The visual editor can maintain layout fields, but direct YAML edits should preserve the runtime wiring.

Reusable components are the default. Before adding a new prompt component, inline DAG fragment, or custom Python agent, first check whether an existing workflow preset, component, deterministic agent, or tool already covers the behavior. Prefer `workflow_ref`, component overrides, and shared tools over copy-pasting verifier/improver/compile/search logic. Only inline or create something new when the reusable piece cannot express the required behavior cleanly.

Use this file before searching runtime code. Most workflow-only agents can be built from:

- `proofstack.agents.dag_workflow.DAGWorkflow`
- `proofstack.agents.configurable_prompt.ConfigurablePromptAgent`
- `proofstack.agents.configurable_cli.ConfigurableCLIAgent`
- provider-managed tools declared through `tool_refs`

Good examples to copy:

- `configs/workflows/prescreen.yaml` for a compact prompt workflow with structured outputs.
- `configs/workflows/author_critic.yaml` for a repeat loop, council/compute branches, and compile gating.
- `configs/workflows/firstproof_smoke_fast.yaml` for a cheap harness smoke preset built from the same Author/Critic blocks.

## References

- Workflow inputs: `$input.problem`, `$input.solution`, etc.
- Node outputs: `$node.<node_id>.<field>`.
- Current output while building outputs or `best_tex`: `$output.<field>`.
- Repeat memory inside a repeat body: `$state.<field>`.
- Outer workflow from inside a repeat body: `$parent.node.<node_id>.<field>`.

If a node references another node with `$node.foo...`, it usually needs `foo` in `needs`. The validator enforces this.

Do not reference a node from inside its own `inputs`, `when`, `default`, or output mapping. For example, node `literature_search` cannot use `$node.literature_search...`.

## Workflow inputs

Workflow inputs are automatically available to agent nodes when the prompt asks for them. Do not wire `problem` through every node unless you are intentionally overriding it.

Keep budgets under top-level `budget`, not under `inputs`:

```yaml
budget:
  max_usd: 2.0
  max_wallclock_s: 600
```

Common top-level inputs:

```yaml
inputs:
  problem: ''
  max_iterations: 3
  page_limit: 12
```

## Prompt components

`components.<name>` defines the prompt/model/output parser. A DAG node uses it through `name: <name>`.

For most prompt-only nodes, use `ConfigurablePromptAgent` and define:

```yaml
components:
  cfg_solver:
    model: models/openai/gpt-54-mini
    system_prompt: |
      You are an expert research mathematician.
      Return only the proof inside <solution>...</solution>.
    user_prompt: |
      Problem:

      {problem}

      Write a complete proof inside <solution>...</solution>.
    input_schema:
      problem: string
    output:
      xml_tags:
      - solution
      default_field: solution
```

For prompt outputs, declare the actual fields:

```yaml
output:
  xml_tags: [verification, verdict]
  default_field: verification
```

Avoid adding a generic `text` output unless the prompt really emits one.

The placeholders in `user_prompt` come from the node input names after expression evaluation. If the prompt contains `{literature_search}`, the node must receive an input named `literature_search`.

## Prompt nodes

A prompt node normally looks like this:

```yaml
- id: solver
  kind: agent
  needs: [literature_search]
  agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
  name: cfg_solver
  inputs:
    problem: $input.problem
    literature_search: $node.literature_search.literature_search
  best_tex: $output.solution
```

`name` must match a component key under `components`. `best_tex` should point at proof-like output fields so last-gasp fallback has useful TeX.

## Custom Python Nodes

Most nodes should still be YAML-configured prompt or CLI components. Add a custom Python node when the interaction pattern itself is the component: nested loops, bounded parallelism, stateful orchestration, early stopping across branches, or any graph that would become unreadable as inline YAML.

A custom node is a normal `Agent` subclass in `src/proofstack/agents/`. Keep the public `Inputs` small and semantic. Do not add editor or graph-builder special cases for one node. The class should carry its own optional `PALETTE`, `default_component_config()`, `component_config_editor()`, and `HIDDEN_GRAPH_INPUTS` metadata so the generic editor can discover it and render its component settings.

For example, `ParallelSolveVerifyImprove` is available in the node palette and exposes only `problem` as a graph port; `n`, `m`, and the solver/verifier/improver/merger prompts stay in `components.<name>` config instead of becoming graph ports.

```yaml
- id: parallel_svi
  kind: agent
  agent: proofstack.agents.parallel_solve_verify_improve.ParallelSolveVerifyImprove
  name: cfg_parallel_svi
  inputs:
    problem: $input.problem
```

The corresponding component config can override prompts and models without changing the node shape:

```yaml
components:
  cfg_parallel_svi:
    model: models/openai/gpt-54-mini
    n: 4
    m: 3
    solver_system_prompt: |
      You are an expert research mathematician.
    verifier_system_prompt: |
      You are a strict mathematical verifier.
    improver_system_prompt: |
      You improve proofs according to verifier feedback.
    merger_system_prompt: |
      You merge candidate proofs into one rigorous proof.
```

If the only goal is to reuse existing graph structure, prefer `workflow_ref`. If the goal is to make a difficult orchestration easy to configure and easy to read, a custom Python node is appropriate.

## CLI components

Use `ConfigurableCLIAgent` for Codex/Claude/shell workers instead of adding a Python subclass such as `Writer` or `PWCWorker`. CLI agents run in Docker by default; set `sandbox.backend: subprocess` only for host-only debugging or tests. The default input and output channel is `workspace`; pass it between CLI nodes when the same files should persist.

Add normal prompt inputs with `input_schema`, refer to them from `prompt`, and expose useful files with `output_files`:
For Codex CLI components, set the raw Codex model with `model` and the Codex reasoning level with `model_reasoning_effort`; do not bake those into `cmd`.

```yaml
components:
  cfg_worker:
    cmd:
      - codex
      - exec
      - --ignore-user-config
      - --ephemeral
      - --skip-git-repo-check
      - --json
    model: gpt-5.4-mini
    model_reasoning_effort: low
    codex_sandbox: auto
    copy_codex_auth: true
    sandbox:
      backend: docker
      docker_image: proofstack-pwc-sandbox:latest
      docker_no_new_privileges: false
      timeout_s: 4500
    prompt: |
      Problem:
      {problem}

      Plan:
      {plan}

      Edit answer.tex, compile it, and call finish when finished.
    input_schema:
      problem: string
      plan: string
      workspace: string
    input_files:
      .pwc/plan.md:
        from_input: plan
    output_schema:
      workspace: string
      answer_tex: string
      answer_tex_path: string
      status: string
      open_questions:
        type: array
        items: {}
    output_files:
      answer_tex: answer.tex
      answer_tex_path:
        path: answer.tex
        type: path
    done_outputs:
      status: status
      open_questions: open_questions
    usage:
      type: codex_jsonl
      model: gpt-5.4-mini
      cost_config: models/openai/gpt-54-mini
```

A CLI node using that component is just a normal agent node:

```yaml
- id: worker
  kind: agent
  agent: proofstack.agents.configurable_cli.ConfigurableCLIAgent
  name: cfg_worker
  inputs:
    workspace: $node.init.workspace
    problem: $input.problem
    plan: $node.planner.plan
```

Prefer file outputs and `done_outputs` over custom Python collection code. Only add a specialized Python CLI agent if the behavior cannot be represented as prompt, setup files, output files, and `finish` metadata.

### Auto-generated output contract (`contract: auto`)

Set `contract: auto` on a CLI component to have the delivery boilerplate appended
to the prompt automatically: the agent generates a "HOW TO DELIVER YOUR OUTPUT"
tail listing the files to write (from `output_files`, skipping passive kinds
`path`/`exists`/`listing`) and the exact `finish '{"status":"done",...}'` command.
The prompt should then describe only the task — do not also hand-write
"write your brief to hint.txt, then run finish" instructions. This keeps prompts
free of executor mechanics, so the same component text stays portable across
backends (CLI, API, human). Components with unusual delivery semantics (e.g. a
file that must contain EXACTLY one word) should state those constraints in the
prompt body; the tail only covers the mechanics.

### Switching a component's executor

In the dashboard preset editor, every agent node has an **Executor** dropdown
(API model / Claude CLI / Codex CLI / Human). Switching keeps the component's
task identity (prompt, `input_schema`/`output_schema`, `output_files`,
`done_outputs`, tools) and replaces the executor-owned keys (`cmd`, `env`,
`usage`, `sandbox`, `model`, `contract`, ...) with defaults for the new
executor; the node's `agent:` class is updated everywhere the component is
used. CLI executors get `contract: auto`; the API executor maps a single text
output to `output.default_field` (multiple outputs become `xml_tags`).
CLI scaffolds bind their model to a declared workflow knob when one exists
(`{claude_model}`/`{base_model}` for Claude, `{codex_model}`/`{gpt_model}` for
Codex) and only fall back to a literal model when the preset declares none, so
tiered presets keep steering switched nodes.

## Tools

Put tools on the component, not the node. Use `tool_refs` for tools defined in `configs/tools`.

Provider-managed OpenAI Responses API tools:

```yaml
tool_refs:
- code_interpreter
- web_search_preview
```

Shared workflow scratch files:

```yaml
tool_refs:
- append_persisted_file
- edit_persisted_file
- list_persisted_files
- read_persisted_file
```

Local function tools are also referenced by name, for example `run_code`.

`max_tool_calls` is a component-level APIClient setting. It limits local function tools; provider-managed tools are forwarded to the provider. It is still fine to set it on a component using both local and provider tools:

```yaml
components:
  cfg_literature_search:
    tool_refs:
    - append_persisted_file
    - read_persisted_file
    - code_interpreter
    - web_search_preview
    max_tool_calls: 16
```

All API calls still go through `mathagents.api_client.APIClient`; do not create provider clients in workflow YAML or new helper code.

## Models

Model references are paths under `configs/models` without `.yaml`:

```yaml
model: models/openai/gpt-54-mini
```

For a new OpenAI Responses API model config, follow the existing `configs/models/openai/gpt-54.yaml` style:

```yaml
model: gpt-5.4-mini--low
api: openai
max_tokens: 128000
batch_processing: false
background: true
use_openai_responses_api: true
reasoning:
  summary: "auto"
```

The `--high` suffix is parsed by `APIClient` into `reasoning_effort`.

## If / else branches

Use Python expressions with `inputs.get(...)`; avoid mixing Python mode with `equals`, `min_len`, or `max_len`.

```yaml
- id: router
  kind: if_else
  needs: [verifier]
  inputs:
    verdict: $node.verifier.verdict
  condition:
    python: inputs.get("verdict") == "correct"
  then:
    'True': true
  else:
    'False': true
```

To make a downstream node run only on one branch, wire the branch output into `when.inputs`:

```yaml
when:
  inputs:
    'False': $node.router.False
  python: inputs.get("False")
```

If a branch output has no downstream consumer, that branch ends there.

## Repeat conditions

Repeat conditions also use `inputs.get(...)`. In a repeat, `inputs` is the current repeat memory, and `iteration` / `max_iterations` are available directly.

```yaml
condition:
  python: iteration == 0 or inputs.get("verdict") != "correct"
```

## Fallback output

`default` is used only when a node is skipped by `when`. Use it to return a simple fallback value with the same output field name:

```yaml
when:
  inputs:
    'False': $node.router.False
  python: inputs.get("False")
default:
  solution: $node.solver.solution
```

## Workflow outputs

For one source:

```yaml
outputs:
  solution: $node.final.solution
```

For mutually exclusive branch sources, use `coalesce`; it returns the first non-empty value:

```yaml
outputs:
  solution:
    coalesce:
      - $node.correct_polisher.solution
      - $node.incorrect_improver.solution
```

Do not add `coalesce: [$some.ref, ""]` just to avoid missing prompt text.
Fields declared as `string` in a configurable component's `input_schema`
default to the empty string when their wired value is missing or `null`.

## Repeat nodes

Inside a repeat body, `$node.*` means nodes from the current iteration. Use `$state.*` for values carried between iterations.

Use `$parent.node.*` for top-level nodes from inside the repeat body. This is the common way to pass a literature search into every verifier/improver iteration.

Prefer reusing the `verify_improve` preset through `workflow_ref` instead of inlining `Verifier -> Correct? -> Improver` every time:

```yaml
- id: proof_loop
  kind: repeat
  needs: [literature_search, solver]
  max_iterations: 3
  initial_state:
    solution: $node.solver.solution
    verification: ''
    verdict: pending
  condition:
    python: iteration == 0 or inputs.get("verdict") != "correct"
  body:
    nodes:
      - id: verify_improve
        kind: workflow_ref
        preset: verify_improve
        inputs:
          problem: $input.problem
          solution: $state.solution
          literature_search: $parent.node.literature_search.literature_search
        component_overrides:
          cfg_verifier:
            model: models/openai/gpt-54-mini
            tool_refs: [code_interpreter, web_search_preview]
          cfg_improver:
            model: models/openai/gpt-54-mini
            tool_refs: [code_interpreter, web_search_preview]
    state_updates:
      solution: $node.verify_improve.solution
      verification: $node.verify_improve.verification
      verdict: $node.verify_improve.verdict
  outputs:
    solution: $state.solution
    verification: $state.verification
    verdict: $state.verdict
```

Inline the router only when a workflow needs custom branch behavior not covered by `verify_improve`.

```yaml
- id: proof_loop
  kind: repeat
  needs: [literature_search, solver]
  max_iterations: 3
  initial_state:
    solution: $node.solver.solution
    verification: ''
    verdict: pending
  condition:
    python: iteration == 0 or inputs.get("verdict") != "correct"
  body:
    nodes:
      - id: verifier
        kind: agent
        agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
        name: cfg_verifier
        inputs:
          problem: $input.problem
          literature_search: $parent.node.literature_search.literature_search
          solution: $state.solution
      - id: improver
        kind: agent
        needs: [verifier]
        agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
        name: cfg_improver
        inputs:
          problem: $input.problem
          literature_search: $parent.node.literature_search.literature_search
          solution: $state.solution
          verification: $node.verifier.verification
    state_updates:
      solution: $node.improver.solution
      verification: $node.verifier.verification
      verdict: $node.verifier.verdict
  outputs:
    solution: $state.solution
    verification: $state.verification
    verdict: $state.verdict
```

If the improver should only run when the verifier says `incorrect`, add an `if_else` node in the repeat body and wire its branch output into `improver.when.inputs`; copy the pattern from `verify_improve.yaml`.

## Compile nodes

Use `ConfigurableCLIAgent` for LaTeX compilation. Keep it mechanical: write `main.tex`, run `pdflatex`/`bibtex`, and expose the generated files. Do not add a Python compile agent just to run shell commands.

```yaml
components:
  cfg_compile_latex:
    cmd:
    - sh
    - -c
    - |
      set +e
      pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex > compile.log 2>&1
      if [ -s references.bib ]; then
        bibtex main >> compile.log 2>&1 || true
        pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex >> compile.log 2>&1
        pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex >> compile.log 2>&1
      fi
      if [ -f main.pdf ]; then
        pdfinfo main.pdf 2>/dev/null | awk '/^Pages:/ {print $2; found=1} END {if (!found) print 0}' > pages.txt || printf '0\n' > pages.txt
        finish '{"status":"done","summary":"compiled"}'
      else
        printf '0\n' > pages.txt
        finish '{"status":"error","summary":"LaTeX compile failed; see compile.log"}'
      fi
    input_schema:
      tex_body: string
    input_files:
      main.tex:
        template: |
          \documentclass[11pt]{article}
          \usepackage{amsmath,amssymb,amsthm}
          \begin{document}
          {tex_body}
          \end{document}
    output_files:
      tex: main.tex
      tex_path: {path: main.tex, type: path}
      pdf_path: {path: main.pdf, type: path}
      compiled: {path: main.pdf, type: exists}
      pages: {path: pages.txt, type: int, default: 0}
      notes: {path: compile.log, default: ''}

dag:
  nodes:
  - id: compile_latex
    kind: agent
    needs: [proof_loop, bibtex_citations]
    agent: proofstack.agents.configurable_cli.ConfigurableCLIAgent
    name: cfg_compile_latex
    inputs:
      tex_body:
        format: |
          {solution}

          \bigskip

          \noindent\textbf{{References.}}
          \begin{{verbatim}}
          {bibtex}
          \end{{verbatim}}
        fields:
          solution: $node.proof_loop.solution
          bibtex: $node.bibtex_citations.bibtex
      page_limit: $input.page_limit
    best_tex: $output.tex
```

Avoid dumping long literature reports into `tex_body`; that can exceed page limits. Put only the proof and final references in `tex_body`.

To write the compiled/fixed TeX to the run's `solutions/` directory:

```yaml
outputs:
  latex: $node.compile_latex.tex
  solution_tex:
    stash_solution:
      problem_id: $input.problem_id
      tex: $node.compile_latex.tex
  compiled: $node.compile_latex.compiled
  pages: $node.compile_latex.pages
  compile_notes: $node.compile_latex.notes
```

## Validate after editing

Run:

```bash
PYTHONPATH=src uv run python -c "from app.dev_data import validate_preset_yaml; from pathlib import Path; r=validate_preset_yaml(Path('configs/workflows/<name>.yaml').read_text()); print(r['ok']); print(r.get('errors') or [])"
```

If `uv` or the project venv is unavailable, the same validation usually works with:

```bash
python -c "import sys; from pathlib import Path; sys.path[:0]=['src','.']; from app.dev_data import validate_preset_yaml; r=validate_preset_yaml(Path('configs/workflows/<name>.yaml').read_text()); print(r['ok']); print(r.get('errors') or []); print(r.get('warnings') or [])"
```

Useful focused tests after workflow YAML edits:

```bash
python -m pytest tests/test_configurable_prompt_outputs.py tests/test_dag_schema_report.py tests/test_workflow_graph_edges.py
```
