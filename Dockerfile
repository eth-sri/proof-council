# Pin to bookworm: the bare ``python:3.12-slim`` tag currently resolves
# to Debian trixie, where ``sagemath`` is not in the apt index.
# Bookworm has it, and Python 3.12 is available via the
# python:3.12-slim-bookworm image.
FROM python:3.12-slim-bookworm

# System packages:
#   texlive-*    - pdflatex for configurable LaTeX compile nodes.
#   nodejs/npm   - host for the @openai/codex and @anthropic-ai/claude-code
#                  CLIs used by ConfigurableCLIAgent.
#   curl, ca-certs - provider HTTPS calls (provider SDKs already use these,
#                    but explicit install keeps the Dockerfile self-documenting).
#   git/file/time/column - common Compute Worker probes, profiling, retrieval.
#   bibtex-extra/biber - references.bib compile support.
#   poppler/ripgrep    - paper-extraction toolkit for CLI workers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        git \
        file \
        time \
        bsdextrautils \
        jq \
        ripgrep \
        gcc g++ \
        libgmp-dev libmpfr-dev libmpc-dev \
        nodejs npm \
        poppler-utils \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-science \
        texlive-bibtex-extra \
        biber \
    && rm -rf /var/lib/apt/lists/*

# Open-source CAS stack for the Compute Worker. Installed as a
# separate layer so iterative dev rebuilds don't repeat it.
#   sagemath    - umbrella system. Pulls ``gap``, ``singular``,
#                 ``pari-gp`` as deps, and exposes them via a Python API.
RUN apt-get update && apt-get install -y --no-install-recommends \
        sagemath \
    && rm -rf /var/lib/apt/lists/*

# Debian's ``singular-ui`` package (pulled in by sagemath) installs
# the binary as ``/usr/bin/Singular`` — capital S, no lowercase
# alias. Every other CAS binary on PATH here is lowercase, and the
# Compute / Author prompts probe with ``command -v singular``, so
# expose a lowercase alias under ``/usr/local/bin`` (which precedes
# ``/usr/bin`` on PATH) for symmetry.
RUN ln -s /usr/bin/Singular /usr/local/bin/singular

# Global fallback for ProofStack CLIAgent's finish handshake. The
# per-run shim is still installed under the workspace, but Codex may
# sanitize PATH in a way that bypasses it; /usr/local/bin is in the
# container's base PATH.
RUN set -eux; \
    { \
        printf '%s\n' '#!/bin/sh'; \
        printf '%s\n' 'set -eu'; \
        printf '%s\n' 'TARGET="${FINISH_DONE_PATH:-${PWD}/done.json}"'; \
        printf '%s\n' 'if [ "${1:-}" != "" ]; then'; \
        printf '%s\n' '    if [ -f "$1" ]; then'; \
        printf '%s\n' '        cp "$1" "$TARGET"'; \
        printf '%s\n' '    else'; \
        printf '%s\n' '        printf "%s" "$1" > "$TARGET"'; \
        printf '%s\n' '    fi'; \
        printf '%s\n' 'elif [ ! -t 0 ]; then'; \
        printf '%s\n' '    cat > "$TARGET"'; \
        printf '%s\n' 'else'; \
        printf '%s\n' '    printf '"'"'{"status": "done", "summary": "(no body supplied)"}'"'"' > "$TARGET"'; \
        printf '%s\n' 'fi'; \
        printf '%s\n' 'echo "finish: wrote $TARGET" >&2'; \
        printf '%s\n' 'exit 0'; \
    } > /usr/local/bin/finish; \
    chmod 0755 /usr/local/bin/finish

# Coding-CLI binaries used by ConfigurableCLIAgent.
# Pin Codex because workflow command flags are part of the runtime contract.
ARG OPENAI_CODEX_VERSION=0.132.0
RUN npm install -g @openai/codex@${OPENAI_CODEX_VERSION} @anthropic-ai/claude-code \
 && codex exec --help | grep -q -- '--output-last-message' \
 && codex exec --help | grep -q -- '--sandbox' \
 && codex exec --help | grep -q -- '--dangerously-bypass-approvals-and-sandbox'

WORKDIR /app

# Copy the package source AND README before installing.
# pyproject.toml references README.md, and hatchling builds from src/ -
# the install would fail if either were missing.
COPY README.md pyproject.toml ./
# hardware.json carries timeout_minutes; the entrypoint reads it to
# install an internal soft deadline ~5 min before the harness's outer
# `timeout` SIGKILL so partial aggregates get flushed cleanly.
COPY hardware.json ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Use uv for fast, reproducible installs.
RUN pip install --no-cache-dir uv \
    && uv pip install --system .

# Scientific stack used by CLI nodes and local experiments.
RUN uv pip install --system \
        sympy numpy scipy networkx mpmath pandas matplotlib \
        requests pymupdf pdfminer.six \
        gmpy2 python-flint z3-solver cvxpy

# Pre-warm Python bytecode so the first container call doesn't pay
# the .pyc compile cost. (Build time is excluded from timeout_minutes
# per the protocol, so this is pure latency reduction.)
RUN python -c "import compileall; compileall.compile_dir('/app/src', quiet=1)"

# Fail-fast smoke for the CAS stack. Verifies every binary the
# Compute / Author prompts mention is actually on PATH inside the
# submission image. We fail the build only on Sage/GAP startup, since
# those already passed in EC2 builds and are the primary entrypoints.
# Standalone backends can vary across Debian packages; the Compute
# Worker is instructed to probe before use.
RUN for bin in sage gap singular gp git file column time; do \
        command -v "$bin" >/dev/null \
            || { echo "missing CAS binary: $bin" >&2; exit 1; }; \
    done \
 && sage -c "print(1+1)" \
 && echo "Print(2+2); QUIT;" | gap -q

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=0 \
    MATHAGENTS_CONFIGS_ROOT=/app/configs \
    PROOFSTACK_SANDBOX_BACKEND=subprocess

RUN python -c "from proofstack.registry import load_preset; load_preset('author_critic_long')"

ENTRYPOINT ["python", "scripts/firstproof_entrypoint.py"]
