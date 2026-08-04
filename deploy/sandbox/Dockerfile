# Minimal runtime image for proofstack CLI agents.
# Not the workflow runner image (see deploy/Dockerfile for that); this one
# is used *per invocation* during dev to wrap Codex in a container
# with dropped capabilities, locked-down filesystem, and a writable
# /work mount for main.tex + done.json.
#
# Build:
#
#     docker build -t proofstack-sandbox:latest deploy/sandbox/
#
# Used automatically when SandboxSpec.backend == "docker".

FROM node:20-bookworm-slim

# texlive for LaTeX compile nodes; ca-certificates for codex's
# HTTPS calls to OpenAI. Nothing else — this image is purely runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Codex CLI. GPT-5.6 Sol and max effort require a current runtime.
ARG OPENAI_CODEX_VERSION=0.144.0
RUN npm install -g @openai/codex@${OPENAI_CODEX_VERSION} \
    && codex --version | grep -q -- "${OPENAI_CODEX_VERSION}"

# The node:20 base already has a non-root `node` user at uid 1000;
# reuse it rather than creating a duplicate. At runtime DockerSandbox
# always passes --user <host_uid>:<host_gid>, so the Dockerfile USER
# is only a fallback for manual `docker run` inspections.
USER node

WORKDIR /work
ENTRYPOINT []
CMD ["/bin/bash"]
