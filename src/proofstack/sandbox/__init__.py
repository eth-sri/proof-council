"""Sandbox backends (SPEC §3.3.1)."""
from __future__ import annotations

import os
from pathlib import Path

from proofstack.sandbox.base import CommandResult, Sandbox, SandboxBackend, SandboxSpec
from proofstack.sandbox.docker import (
    DockerSandbox,
    DockerSandboxError,
    check_image_available,
)
from proofstack.sandbox.subprocess import SubprocessSandbox


def resolve_backend(spec: SandboxSpec) -> SandboxBackend:
    """Pick the backend, honoring the env override.

    ``PROOFSTACK_SANDBOX_BACKEND`` (values: ``subprocess`` / ``docker``)
    wins over the ``SandboxSpec`` default so the submission entrypoint
    can force subprocess mode without patching any agent code, and dev
    users can switch backends ad-hoc.
    """
    override = os.environ.get("PROOFSTACK_SANDBOX_BACKEND", "").strip().lower()
    if override in ("subprocess", "docker"):
        return override  # type: ignore[return-value]
    return spec.backend


def make_sandbox(spec: SandboxSpec, *, root: Path | None = None) -> Sandbox:
    """Construct the appropriate Sandbox backend.

    Raises ``DockerSandboxError`` early if docker is selected but the
    image is not built locally — far clearer than letting the CLI
    spawn fail with a cryptic docker error halfway through.
    """
    backend = resolve_backend(spec)
    if backend == "docker":
        if not check_image_available(spec.docker_image):
            # The PWC sandbox image needs the explicit -f Dockerfile.pwc
            # flag because it's the second stage on top of the base
            # proofstack-sandbox image. The base image follows the
            # default Dockerfile naming and doesn't.
            if "pwc" in spec.docker_image:
                build_cmd = (
                    "docker build -t proofstack-sandbox:latest deploy/sandbox/ && "
                    "docker build -t " + spec.docker_image
                    + " -f deploy/sandbox/Dockerfile.pwc deploy/sandbox/"
                )
            else:
                build_cmd = f"docker build -t {spec.docker_image} deploy/sandbox/"
            raise DockerSandboxError(
                f"docker image {spec.docker_image!r} is not built. Run:\n"
                f"    {build_cmd}\n"
                f"or set PROOFSTACK_SANDBOX_BACKEND=subprocess to skip the "
                f"container sandbox."
            )
        return DockerSandbox(spec, root=root)
    return SubprocessSandbox(spec, root=root)


__all__ = [
    "CommandResult",
    "DockerSandbox",
    "DockerSandboxError",
    "Sandbox",
    "SandboxBackend",
    "SandboxSpec",
    "SubprocessSandbox",
    "check_image_available",
    "make_sandbox",
    "resolve_backend",
]
