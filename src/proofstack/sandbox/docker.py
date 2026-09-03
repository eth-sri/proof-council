"""Docker-backed sandbox for configurable CLI agents.

Wraps each CLI invocation in ``docker run --rm ...`` with:

- capability drop (``--cap-drop ALL``; optionally
  ``--security-opt no-new-privileges``)
- resource limits (``--memory``, ``--cpus``, ``--pids-limit``)
- writable bind mount ONLY for the per-invocation workdir
- ``tmpfs`` for ``/tmp``
- non-root uid (matching the host's default 1000 so bind-mount file
  ownership stays sane)
- environment scrubbed to an explicit allowlist + provider keys

The image is expected to be built locally via ``deploy/sandbox/Dockerfile``;
we do not pull from a registry.

Use this backend when a CLI node should run inside the local sandbox
image from ``deploy/sandbox/Dockerfile``.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import stat
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Iterable, Mapping, TypeVar

from proofstack.sandbox.base import (
    CommandResult,
    Sandbox,
    SandboxSpawnError,
    SandboxSpec,
    WorkerStopState,
)
from proofstack.sandbox.subprocess import _StreamingProcess


class DockerSandboxError(RuntimeError):
    """Raised when docker is missing / image is not built / etc."""


class DockerSandboxSpawnError(DockerSandboxError, SandboxSpawnError):
    """The Docker client could not be spawned, so no container can exist."""


_DOCKER_CONTROL_TIMEOUT_S = 5.0
_DOCKER_OWNER_LABEL = "proofstack.workspace-owner"
_DOCKER_ABSENCE_PROBES = 3
_DOCKER_ABSENCE_PROBE_INTERVAL_S = 0.1
_DOCKER_LAUNCH_PROBE_INTERVAL_S = 0.1
_DOCKER_CID_MAX_BYTES = 128
_DOCKER_MANAGED_LIFECYCLE_FLAGS = {
    "--cidfile",
    "--detach",
    "--label-file",
    "--name",
    "--rm",
}
_T = TypeVar("_T")


def _validate_docker_extra_args(extra_args: Iterable[str]) -> None:
    """Reject options that can invalidate container lifecycle tracking."""
    args = tuple(extra_args)
    if any(not isinstance(value, str) for value in args):
        raise DockerSandboxError("docker_extra_args entries must be strings")
    index = 0
    while index < len(args):
        raw = args[index]
        option, separator, inline_value = raw.partition("=")
        if option in _DOCKER_MANAGED_LIFECYCLE_FLAGS:
            raise DockerSandboxError(
                f"docker_extra_args cannot override ProofStack-managed {option}"
            )
        if re.fullmatch(r"-[ditP]*d[ditP]*(?:=.*)?", raw):
            raise DockerSandboxError(
                "docker_extra_args cannot detach a ProofStack-managed container"
            )

        label_value: str | None = None
        if option in {"--label", "-l"}:
            if separator:
                label_value = inline_value
            elif index + 1 < len(args):
                index += 1
                label_value = args[index]
        elif option.startswith("-l") and not option.startswith("--"):
            label_value = option[2:].lstrip("=")
        if label_value is not None:
            label_key = label_value.split("=", maxsplit=1)[0]
            if label_key == _DOCKER_OWNER_LABEL:
                raise DockerSandboxError(
                    "docker_extra_args cannot override ProofStack's workspace-owner label"
                )
        index += 1


class _DockerLaunchReceipt:
    """Private host-side proof that dockerd created a container.

    ``docker run --cidfile`` writes the container ID only after creation. The
    directory is private and outside the model-mounted workspace, so a fast
    inner-command failure can be distinguished from a Docker client that lost
    contact with the daemon before receiving a create response.
    """

    def __init__(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="proofstack-docker-launch-"))
        self.path = self.directory / "container.cid"
        self._cleaned = False

    def container_id(self) -> str | None:
        try:
            metadata = self.path.lstat()
        except OSError:
            return None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > _DOCKER_CID_MAX_BYTES
        ):
            return None
        try:
            value = self.path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            return None
        if re.fullmatch(r"[0-9a-fA-F]{12,64}", value) is None:
            return None
        return value.lower()

    def cleanup(self) -> bool:
        if self._cleaned:
            return True
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            return False
        try:
            self.directory.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            return False
        self._cleaned = True
        return True


async def _drain_task_uninterruptibly(task: asyncio.Task[_T]) -> _T:
    """Await a safety cleanup even if the caller is cancelled repeatedly."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    return task.result()


async def _stop_control_process(proc: asyncio.subprocess.Process) -> None:
    try:
        proc.kill()
    except OSError:
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=1.0)
    except (asyncio.TimeoutError, ProcessLookupError, ValueError):
        pass


async def _docker_control(
    *args: str,
    timeout_s: float | None = None,
) -> tuple[int, str, str]:
    """Run a bounded Docker control command and retain its diagnostics."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, ValueError) as e:
        raise DockerSandboxError(f"could not start Docker control command: {e}") from e
    timeout = _DOCKER_CONTROL_TIMEOUT_S
    if timeout_s is not None:
        timeout = min(timeout, float(timeout_s))
        if timeout <= 0:
            await _stop_control_process(proc)
            raise DockerSandboxError(
                f"docker {' '.join(args)} exceeded its caller deadline"
            )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except (asyncio.TimeoutError, asyncio.CancelledError) as e:
        cleanup = asyncio.create_task(_stop_control_process(proc))
        try:
            await _drain_task_uninterruptibly(cleanup)
        except BaseException:
            pass
        if isinstance(e, asyncio.CancelledError):
            raise
        raise DockerSandboxError(
            f"docker {' '.join(args)} timed out"
        ) from e
    return (
        proc.returncode if proc.returncode is not None else -1,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )


async def _docker_container_identity(
    container_name: str,
    *,
    timeout_s: float | None = None,
) -> tuple[str, str | None, str] | None:
    """Return an exact container's status/owner/id, or ``None`` if absent."""
    returncode, stdout, stderr = await _docker_control(
        "container",
        "inspect",
        "--format",
        (
            "{{.Id}}\t{{.State.Status}}\t"
            f'{{{{index .Config.Labels "{_DOCKER_OWNER_LABEL}"}}}}'
        ),
        container_name,
        timeout_s=timeout_s,
    )
    if returncode == 0:
        fields = stdout.rstrip("\r\n").split("\t", maxsplit=2)
        if len(fields) != 3 or not fields[0].strip():
            raise DockerSandboxError(
                f"Docker returned malformed identity for {container_name!r}"
            )
        container_id, status, owner = fields
        normalized_owner: str | None = owner.strip()
        if normalized_owner in {"", "<no value>"}:
            normalized_owner = None
        return (
            status.strip() or "unknown",
            normalized_owner,
            container_id.strip(),
        )
    detail = stderr.strip()
    lowered = detail.lower()
    if "no such object" in lowered or "no such container" in lowered:
        return None
    failure_detail = detail or f"exit {returncode}"
    raise DockerSandboxError(
        f"could not inspect Docker container {container_name!r}: "
        f"{failure_detail}"
    )


async def _docker_container_status(container_name: str) -> str | None:
    identity = await _docker_container_identity_settled(container_name)
    return identity[0] if identity is not None else None


async def _docker_container_identity_settled(
    container_name: str,
) -> tuple[str, str | None, str] | None:
    """Require repeated absence before trusting that no container exists.

    Dockerd can still finish an in-flight create briefly after the ``docker run``
    client has died. A single ``No such container`` response is therefore not a
    sufficient terminal-state check for a persistent workspace.
    """
    for probe in range(_DOCKER_ABSENCE_PROBES):
        identity = await _docker_container_identity(container_name)
        if identity is not None:
            return identity
        if probe + 1 < _DOCKER_ABSENCE_PROBES:
            await asyncio.sleep(_DOCKER_ABSENCE_PROBE_INTERVAL_S)
    return None


async def _docker_kill(
    container_name: str,
    *,
    owner: str,
    absence_is_terminal: bool = True,
) -> WorkerStopState:
    """Stop an owned container and report the resulting control-plane state.

    Killing the ``docker run`` CLI client does NOT propagate to the
    container — dockerd sees a detached client but PID 1 in the
    container keeps running until it exits on its own, holding on to
    the per-container resource caps. Termination therefore addresses
    the named container and verifies that it no longer exists before
    the workspace is treated as idle.
    """
    try:
        identity = await _docker_container_identity_settled(container_name)
    except DockerSandboxError:
        return WorkerStopState.UNKNOWN
    if identity is None:
        return (
            WorkerStopState.STOPPED
            if absence_is_terminal
            else WorkerStopState.UNKNOWN
        )
    if identity[1] != owner:
        return WorkerStopState.SURVIVING
    container_id = identity[2]

    for command in (
        ("kill", container_id),
        ("rm", "--force", container_id),
    ):
        try:
            await _docker_control(*command)
        except DockerSandboxError:
            # The final inspect below is authoritative. A command can fail
            # simply because --rm already removed the container.
            continue
    try:
        identity = await _docker_container_identity_settled(container_name)
    except DockerSandboxError:
        return WorkerStopState.UNKNOWN
    return (
        WorkerStopState.STOPPED
        if identity is None
        else WorkerStopState.SURVIVING
    )


async def _terminate_docker_run(
    proc: asyncio.subprocess.Process,
    container_name: str,
    *,
    owner: str,
    launch_settled: bool,
    launch_receipt: _DockerLaunchReceipt | None = None,
) -> tuple[WorkerStopState, bool]:
    """Stop a Docker client and its owned container as one drained cleanup."""
    try:
        proc.kill()
    except OSError:
        pass
    try:
        await asyncio.wait_for(
            proc.communicate(),
            timeout=_DOCKER_CONTROL_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        await _stop_control_process(proc)
    except (ProcessLookupError, ValueError):
        pass
    if launch_receipt is not None and launch_receipt.container_id() is not None:
        launch_settled = True
    stop_state = await _docker_kill(
        container_name,
        owner=owner,
        absence_is_terminal=launch_settled,
    )
    return (
        stop_state,
        launch_settled or stop_state is WorkerStopState.STOPPED,
    )


CONTAINER_WORKDIR = "/work"
CONTAINER_TMPFS = "/tmp"
# Deliberate, minimal PATH inside the container. Does not inherit from
# the host (which may have random user-installed shims). Caller-supplied
# extra_path entries (typically the per-invocation shim dir under
# sandbox.root) are prepended to this by _build_docker_cmd after host-
# to-container translation.
CONTAINER_BASE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class DockerSandbox(Sandbox):
    """Run each command in a fresh ``docker run --rm`` container.

    Host <-> container path translation: anything the caller passes
    that contains ``str(self.root)`` (typically
    ``FINISH_DONE_PATH`` and ``extra_path``) is rewritten to
    ``/work`` before being handed to the container. The container sees
    ``/work``; the orchestrator on the host polls
    ``<self.root>/done.json`` via the same bind mount.
    """

    def __init__(
        self,
        spec: SandboxSpec,
        *,
        root: Path | None = None,
        inherited_fds: Iterable[int] = (),
    ) -> None:
        _validate_docker_extra_args(spec.docker_extra_args)
        super().__init__(spec, root=root, inherited_fds=inherited_fds)
        self.container_owner = uuid.uuid4().hex

    @property
    def container_name(self) -> str:
        return _container_name_for_root(self.root)

    async def ensure_workspace_available(self) -> None:
        status = await _docker_container_status(self.container_name)
        if status is not None:
            raise DockerSandboxError(
                "persistent workspace still has a Docker sandbox "
                f"({self.container_name}, status={status}); refusing to overlap "
                "a resumed invocation"
            )

    def _require_idle_for_launch(self) -> None:
        if (
            self.worker_stop_state is WorkerStopState.STOPPED
            and self.worker_launch_settled
        ):
            return
        raise DockerSandboxError(
            "cannot start another Docker command while a prior worker is "
            "active or its launch state is unresolved "
            f"(state={self.worker_stop_state.value}, "
            f"launch_settled={self.worker_launch_settled})"
        )

    async def run_command(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        timeout_s: int | None = None,
        env_extra: Mapping[str, str] | None = None,
        extra_path: Iterable[Path] = (),
        input_data: str | bytes | None = None,
    ) -> CommandResult:
        self._require_idle_for_launch()
        input_bytes = (
            input_data.encode("utf-8") if isinstance(input_data, str) else input_data
        )
        container_name = self.container_name
        try:
            launch_receipt = _DockerLaunchReceipt()
        except OSError as e:
            raise DockerSandboxSpawnError(
                f"could not create private Docker launch receipt: {e}"
            ) from e
        try:
            docker_cmd = self._build_docker_cmd(
                cmd,
                env_extra=env_extra,
                extra_path=list(extra_path),
                cwd=cwd,
                interactive=input_bytes is not None,
                container_name=container_name,
                cidfile=launch_receipt.path,
            )
        except BaseException:
            launch_receipt.cleanup()
            raise
        timeout = timeout_s if timeout_s is not None else self.spec.timeout_s
        start = time.monotonic()
        self._mark_worker_launch_pending()
        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdin=(asyncio.subprocess.PIPE if input_bytes is not None else None),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                pass_fds=self.inherited_fds,
            )
        except (OSError, ValueError) as e:
            self._set_worker_lifecycle(
                WorkerStopState.STOPPED,
                launch_settled=True,
            )
            launch_receipt.cleanup()
            raise DockerSandboxSpawnError(
                "could not start the Docker client; install Docker Desktop "
                "(Windows) or the Docker engine (Linux), or set "
                "PROOFSTACK_SANDBOX_BACKEND=subprocess to bypass: "
                f"{e}"
            ) from e
        except BaseException:
            launch_receipt.cleanup()
            raise
        try:
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(input=input_bytes), timeout=timeout
                )
                returncode = proc.returncode if proc.returncode is not None else -1
                # A cidfile proves that dockerd created the container even when
                # the foreground client propagates a nonzero inner-command
                # status. Without that receipt, a nonzero client exit remains
                # ambiguous because a daemon-side create may still materialize.
                launch_settled = (
                    returncode == 0
                    or launch_receipt.container_id() is not None
                )
                stop_state = WorkerStopState.STOPPED
                if returncode != 0:
                    stop_state = await _docker_kill(
                        container_name,
                        owner=self.container_owner,
                        absence_is_terminal=launch_settled,
                    )
                    if stop_state is not WorkerStopState.STOPPED:
                        suffix = (
                            "Docker container stop state is "
                            f"{stop_state.value}"
                        ).encode("utf-8")
                        stderr_b = stderr_b.rstrip()
                        stderr_b += (b"\n" if stderr_b else b"") + suffix
                self._set_worker_lifecycle(
                    stop_state,
                    launch_settled=(
                        launch_settled or stop_state is WorkerStopState.STOPPED
                    ),
                )
            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                launch_settled = launch_receipt.container_id() is not None
                cleanup = asyncio.create_task(
                    _terminate_docker_run(
                        proc,
                        container_name,
                        owner=self.container_owner,
                        launch_settled=launch_settled,
                        launch_receipt=launch_receipt,
                    )
                )
                try:
                    stop_state, launch_settled = await _drain_task_uninterruptibly(
                        cleanup
                    )
                except BaseException:
                    stop_state = WorkerStopState.UNKNOWN
                    launch_settled = (
                        launch_settled
                        or launch_receipt.container_id() is not None
                    )
                self._set_worker_lifecycle(
                    stop_state,
                    launch_settled=launch_settled,
                )
                if isinstance(e, asyncio.CancelledError):
                    raise
                returncode = -9
                stdout_b = b""
                detail = f"timeout after {timeout}s"
                if stop_state is not WorkerStopState.STOPPED:
                    detail += (
                        "; Docker container stop state is "
                        f"{stop_state.value}"
                    )
                stderr_b = detail.encode("utf-8")
            elapsed = time.monotonic() - start
            return CommandResult(
                cmd=cmd,
                returncode=returncode,
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr_b.decode("utf-8", errors="replace"),
                duration_s=elapsed,
            )
        finally:
            launch_receipt.cleanup()

    async def stream_command(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        timeout_s: int | None = None,
        env_extra: Mapping[str, str] | None = None,
        extra_path: Iterable[Path] = (),
    ) -> "_StreamingProcess":
        self._require_idle_for_launch()
        container_name = self.container_name
        try:
            launch_receipt = _DockerLaunchReceipt()
        except OSError as e:
            raise DockerSandboxSpawnError(
                f"could not create private Docker launch receipt: {e}"
            ) from e
        try:
            docker_cmd = self._build_docker_cmd(
                cmd,
                env_extra=env_extra,
                extra_path=list(extra_path),
                cwd=cwd,
                interactive=True,
                container_name=container_name,
                cidfile=launch_receipt.path,
            )
        except BaseException:
            launch_receipt.cleanup()
            raise
        self._mark_worker_launch_pending()
        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                pass_fds=self.inherited_fds,
            )
        except (OSError, ValueError) as e:
            self._set_worker_lifecycle(
                WorkerStopState.STOPPED,
                launch_settled=True,
            )
            launch_receipt.cleanup()
            raise DockerSandboxSpawnError(
                "could not start the Docker client; install Docker Desktop "
                "(Windows) or the Docker engine (Linux), or set "
                "PROOFSTACK_SANDBOX_BACKEND=subprocess to bypass: "
                f"{e}"
            ) from e
        except BaseException:
            launch_receipt.cleanup()
            raise
        started_at = time.monotonic()
        deadline = started_at + (
            timeout_s if timeout_s is not None else self.spec.timeout_s
        )
        launch_timeout_s = max(0.0, float(self.spec.docker_launch_timeout_s))
        launch_deadline = min(deadline, started_at + launch_timeout_s)
        stream = _DockerStreamingProcess(
            proc=proc,
            cmd=docker_cmd,
            deadline=deadline,
            launch_deadline=launch_deadline,
            container_name=container_name,
            container_owner=self.container_owner,
            launch_receipt=launch_receipt,
            lifecycle_callback=self._set_worker_lifecycle,
        )
        try:
            await stream.confirm_launch()
        except BaseException:
            # Never hand an unobserved Docker launch to the caller. Drain the
            # cleanup even under repeated cancellation so the inherited
            # workspace lock is not released while the client is still alive.
            cleanup = asyncio.ensure_future(stream.terminate())
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            try:
                cleanup.result()
            except (asyncio.CancelledError, Exception):
                # Preserve the launch/cancellation failure that triggered this
                # cleanup. Its caller retains the workspace guard.
                pass
            self._set_worker_lifecycle(
                stream.worker_stop_state,
                launch_settled=stream.launch_settled,
            )
            launch_receipt.cleanup()
            raise
        self._set_worker_lifecycle(
            stream.worker_stop_state,
            launch_settled=stream.launch_settled,
        )
        return stream

    # --- helpers ----------------------------------------------------------

    def _build_docker_cmd(
        self,
        inner_cmd: list[str],
        *,
        env_extra: Mapping[str, str] | None,
        extra_path: list[Path],
        cwd: str | None,
        interactive: bool,
        container_name: str,
        cidfile: Path | None = None,
    ) -> list[str]:
        _validate_docker_extra_args(self.spec.docker_extra_args)
        args: list[str] = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--label",
            f"{_DOCKER_OWNER_LABEL}={self.container_owner}",
        ]
        if cidfile is not None:
            args += ["--cidfile", str(cidfile)]
        if interactive:
            args += ["-i"]
        args += [
            "--memory", f"{self.spec.memory_gb}g",
            "--cpus", str(self.spec.cpu_limit),
            "--pids-limit", str(self.spec.docker_pids_limit),
            "--cap-drop", "ALL",
            "--network", self.spec.docker_network,
            "--user", f"{os.getuid()}:{os.getgid()}",
            # Docker treats relative source paths as named volumes, so
            # resolve to an absolute host path. RunContext.create can keep
            # the workdir relative (e.g. `--output outputs`), which would
            # otherwise yield `-v outputs/...:/work` and fail.
            "-v", f"{self.root.resolve()}:{CONTAINER_WORKDIR}",
            "-w",
            (CONTAINER_WORKDIR + "/" + cwd) if cwd else CONTAINER_WORKDIR,
            # tmpfs so /tmp is writable without escaping the mount
            "--tmpfs", f"{CONTAINER_TMPFS}:size=1g,mode=1777",
        ]
        if self.spec.docker_no_new_privileges:
            args += ["--security-opt", "no-new-privileges"]

        # --- Env forwarding -------------------------------------------
        # Pinned container env (HOME, TMPDIR, PATH) — the host's values
        # are NEVER leaked into the container. extra_path entries are
        # translated from host -> container paths and prepended to PATH
        # so the per-invocation shim dir (e.g. the CLIAgent finish
        # bin) resolves first. Mirrors SubprocessSandbox behaviour.
        translated_extra = [
            self._translate_path(str(p.resolve() if p.is_absolute() else p))
            for p in extra_path
        ]
        path_value = ":".join([*translated_extra, CONTAINER_BASE_PATH]) if translated_extra else CONTAINER_BASE_PATH
        fixed_env = {
            "HOME": CONTAINER_WORKDIR,
            "TMPDIR": CONTAINER_TMPFS,
            "PATH": path_value,
        }
        for k, v in fixed_env.items():
            args += ["-e", f"{k}={v}"]

        # Provider keys: pass names only; docker reads the value from
        # the orchestrator's env at spawn time and sets them in the
        # container. Values are never written to disk.
        for key in self.spec.provider_keys:
            if os.environ.get(key) is not None:
                args += ["-e", key]

        # Caller-supplied env_extra — translate host paths -> container paths.
        if env_extra:
            for k, v in env_extra.items():
                args += ["-e", f"{k}={self._translate_path(str(v))}"]

        # spec.extra_env last, so a SandboxSpec-level override wins.
        for k, v in self.spec.extra_env.items():
            args += ["-e", f"{k}={self._translate_path(str(v))}"]

        # --- Any extra user-supplied docker args ----------------------
        args += list(self.spec.docker_extra_args)

        # --- Image + inner command -----------------------------------
        args += [self.spec.docker_image]
        args += inner_cmd
        return args

    def _translate_path(self, value: str) -> str:
        """Rewrite any occurrence of ``str(self.root)`` with ``/work``.

        The bind mount aliases these two paths. Callers like CLIAgent
        pass host paths (``sandbox.root / "done.json"``) via env; we
        translate once here so every backend sees identical semantics.
        """
        root_str = str(self.root)
        if root_str in value:
            return value.replace(root_str, CONTAINER_WORKDIR)
        return value


def _container_name_for_root(root: Path) -> str:
    """Return the stable Docker identity protecting one workspace path."""
    resolved = Path(root).resolve()
    digest = hashlib.sha256(os.fsencode(str(resolved))).hexdigest()[:24]
    return f"proofstack-sbx-{digest}"


class _DockerStreamingProcess(_StreamingProcess):
    """Streaming handle that also knows how to signal its container.

    ``_StreamingProcess.terminate`` only kills the Docker CLI client,
    which can leave the container running. We then remove the exact
    owner-labelled container and verify that the workspace is idle.
    """

    def __init__(
        self,
        *,
        container_name: str,
        container_owner: str,
        launch_deadline: float | None = None,
        launch_receipt: _DockerLaunchReceipt | None = None,
        lifecycle_callback: Callable[..., None] | None = None,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.container_name = container_name
        self.container_owner = container_owner
        self.launch_deadline = (
            self.deadline if launch_deadline is None else float(launch_deadline)
        )
        self._container_stop_state = WorkerStopState.UNKNOWN
        self._launch_settled = False
        self._launch_receipt = launch_receipt
        self._lifecycle_callback = lifecycle_callback

    def _publish_lifecycle(self) -> None:
        if self._lifecycle_callback is not None:
            self._lifecycle_callback(
                self._container_stop_state,
                launch_settled=self._launch_settled,
            )

    def _cleanup_launch_receipt(self) -> None:
        if (
            self._launch_receipt is not None
            and self._launch_receipt.cleanup()
        ):
            self._launch_receipt = None

    def _receipt_confirms_creation(self) -> bool:
        return bool(
            self._launch_receipt is not None
            and self._launch_receipt.container_id() is not None
        )

    @property
    def launch_remaining_s(self) -> float:
        return max(0.0, self.launch_deadline - time.monotonic())

    async def confirm_launch(self) -> None:
        """Observe this invocation or a completed ``docker run`` request.

        A live stream is returned to callers only after the named container has
        been observed with this invocation's owner label. A successful Docker
        client exit also settles the request; a nonzero exit does not prove that
        a delayed daemon-side create cannot still materialize. An invisible or
        ambiguous launch is rejected instead of later treating a short run of
        absent ``inspect`` results as proof that no worker exists.
        """
        while self.launch_remaining_s > 0:
            returncode = getattr(self.proc, "returncode", None)
            if returncode is not None:
                if int(returncode) == 0 or self._receipt_confirms_creation():
                    self._launch_settled = True
                    self._publish_lifecycle()
                    return
                raise DockerSandboxError(
                    "Docker client failed before its container launch was "
                    "observed; daemon-side completion is unknown: "
                    f"{self.container_name} (exit={returncode})"
                )
            if self._receipt_confirms_creation():
                self._launch_settled = True
                self._container_stop_state = WorkerStopState.SURVIVING
                self._publish_lifecycle()
                return
            try:
                identity = await _docker_container_identity(
                    self.container_name,
                    timeout_s=min(self.remaining_s, self.launch_remaining_s),
                )
            except DockerSandboxError as e:
                if self.launch_remaining_s <= 0:
                    raise DockerSandboxError(
                        "Docker container launch was not observed before the "
                        f"launch deadline: {self.container_name}"
                    ) from e
                raise
            if identity is not None:
                if identity[1] != self.container_owner:
                    self._container_stop_state = WorkerStopState.SURVIVING
                    self._publish_lifecycle()
                    raise DockerSandboxError(
                        "Docker container name is owned by another invocation: "
                        f"{self.container_name}"
                    )
                self._launch_settled = True
                self._container_stop_state = WorkerStopState.SURVIVING
                self._publish_lifecycle()
                return
            delay = min(
                _DOCKER_LAUNCH_PROBE_INTERVAL_S,
                self.launch_remaining_s,
            )
            if delay > 0:
                await asyncio.sleep(delay)
        raise DockerSandboxError(
            "Docker container launch was not observed before the launch "
            f"deadline: {self.container_name}"
        )

    @property
    def worker_stop_state(self) -> WorkerStopState:
        return self._container_stop_state

    @property
    def worker_stopped(self) -> bool:
        return self.worker_stop_state is WorkerStopState.STOPPED

    @property
    def launch_settled(self) -> bool:
        return self._launch_settled

    async def terminate(self) -> None:
        if self.worker_stop_state is WorkerStopState.STOPPED:
            self._cleanup_launch_receipt()
            return
        try:
            # Stop the client first so no new container can appear after the
            # final inspect performed by _docker_kill.
            await super().terminate()
        finally:
            try:
                # The Docker client can write its cidfile while termination is
                # draining. Observe it only after the client has stopped so a
                # just-completed create is not misclassified as unresolved.
                if self._receipt_confirms_creation():
                    self._launch_settled = True
                self._container_stop_state = await _docker_kill(
                    self.container_name,
                    owner=self.container_owner,
                    absence_is_terminal=(
                        self._launch_settled
                        and self.process_group_stop_state
                        is WorkerStopState.STOPPED
                    ),
                )
                if (
                    self.process_group_stop_state is not WorkerStopState.STOPPED
                    and self._container_stop_state is WorkerStopState.STOPPED
                ):
                    self._container_stop_state = WorkerStopState.UNKNOWN
                if self._container_stop_state is WorkerStopState.STOPPED:
                    self._launch_settled = True
            except BaseException:
                self._container_stop_state = WorkerStopState.UNKNOWN
                raise
            finally:
                self._publish_lifecycle()
                self._cleanup_launch_receipt()


def check_image_available(image: str = "proofstack-sandbox:latest") -> bool:
    """Returns True if the given docker image is built locally.

    Non-async because it's called at startup / construction time.
    Uses ``docker image inspect`` which is cheap.
    """
    import subprocess as _sp
    try:
        res = _sp.run(
            ["docker", "image", "inspect", image],
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
            timeout=5,
        )
        return res.returncode == 0
    except (FileNotFoundError, _sp.TimeoutExpired):
        return False


__all__ = [
    "DockerSandbox",
    "DockerSandboxError",
    "DockerSandboxSpawnError",
    "check_image_available",
]
