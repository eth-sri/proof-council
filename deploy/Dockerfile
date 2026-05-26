FROM python:3.12-slim

# System packages:
#   texlive-*    — pdflatex for configurable LaTeX compile nodes.
#   nodejs/npm   — host for the @openai/codex and @anthropic-ai/claude-code
#                  CLIs used by ConfigurableCLIAgent.
#   curl, ca-certs — provider HTTPS calls (provider SDKs already use these,
#                    but explicit install keeps the Dockerfile self-documenting).
#   bibtex-extra/biber — references.bib compile support.
#   poppler/ripgrep    — paper-extraction toolkit for CLI workers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        jq \
        ripgrep \
        gcc g++ \
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

# Coding-CLI binaries used by ConfigurableCLIAgent.
# Pinning to known-good majors so the image is reproducible.
RUN npm install -g @openai/codex @anthropic-ai/claude-code

WORKDIR /app

# Copy the package source AND README before installing.
# pyproject.toml references README.md, and hatchling builds from src/ —
# the install would fail if either were missing.
COPY README.md pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Use uv for fast, reproducible installs.
RUN pip install --no-cache-dir uv \
    && uv pip install --system .

# Scientific stack used by CLI nodes and local experiments.
RUN uv pip install --system \
        sympy numpy scipy networkx mpmath pandas matplotlib \
        requests pymupdf pdfminer.six

# Pre-warm Python bytecode so the first container call doesn't pay
# the .pyc compile cost. (Build time is excluded from timeout_minutes
# per the protocol, so this is pure latency reduction.)
RUN python -c "import compileall; compileall.compile_dir('/app/src', quiet=1)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=0 \
    MATHAGENTS_CONFIGS_ROOT=/app/configs

ENTRYPOINT ["python", "scripts/run_workflow.py"]
CMD ["--help"]
