# ProofStack CLI sandbox image

Minimal Docker image used by `DockerSandbox` (SPEC §3.3.1) to wrap
Codex-driven `ConfigurableCLIAgent` invocations. Distinct from the
workflow runner image in `deploy/Dockerfile`.

## Build

```bash
docker build -t proofstack-sandbox:latest deploy/sandbox/
```

## What it contains

- `node:20-bookworm-slim` base
- `texlive-latex-{base,recommended,extra}` + `texlive-fonts-recommended` for pdflatex
- `@openai/codex` (latest at image build time) installed globally via npm
- Runs as the `node:20` base image's built-in non-root `node` user by default.
  At runtime `DockerSandbox` always passes `--user $(id -u):$(id -g)` so
  bind-mounted files stay owned by the host user regardless of image uid.

## What it does NOT contain

- No `proofstack` code, no `.venv`, no Python stack. The agent layer
  runs on the host; only the Codex CLI runs in the container.
- No provider API keys. They are forwarded per-invocation via `-e`.
- No shell completions, no user config, no `~/.codex/config.toml`.

## Why separate from `deploy/Dockerfile`?

The submission image hosts the orchestrator and must be large enough
to include Python + every agent. This image runs inside one CLI call,
should be tiny, and must not have any orchestrator code so a
compromised CLI cannot tamper with the run state.

## Auth

Codex auth is handled per invocation by `ConfigurableCLIAgent` when
`copy_codex_auth: true` is set:
`~/.codex/auth.json` is *copied* into a writable directory inside the
sandbox mount (`<workdir>/.codex-home/auth.json`), and
`CODEX_HOME=<workdir>/.codex-home` is passed to the container.

We copy rather than bind-mount because:
- A read-only bind mount breaks Codex's session init
  (it writes to `$CODEX_HOME/sessions/`, `$CODEX_HOME/cache/`).
- A read-write bind mount would let the agent tamper with your
  host credentials.
- `ConfigurableCLIAgent.teardown()` removes `.codex-home/` after the CLI exits so
  the token does not persist in the run workdir under `outputs/`
  (which is routinely shared for code review). The container is
  already `--rm`, but the bind-mounted host directory is not.

No other files from `~/.codex` are exposed (no config.toml, no
history, no skills, no logs).

## Codex internal sandbox

The container drops all Linux capabilities, which prevents Codex's
default `bwrap`-based workspace-write sandbox from creating user
namespaces. Configurable Codex CLI nodes therefore invoke Codex with
`--dangerously-bypass-approvals-and-sandbox` **only when running in
Docker** — the outer container is the security boundary. In
subprocess mode, Codex can use its own sandbox mode.
