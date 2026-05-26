"""First Proof preflight healthcheck.

Probes every model and tool path the configured workflow will exercise
*before* the entrypoint fans out the 10 problems. The point is to catch
broken API keys, mistyped model names, missing CAS binaries, and
unreachable codex CLI subprocesses inside the first ~60 seconds of a
run rather than discovering them halfway through hour 12.

Three modes via ``FIRSTPROOF_HEALTHCHECK`` env var:

  - ``off`` (default): skip entirely. **Use this for the official First
    Proof submission.** The First Proof harness counts files in
    ``/data/output`` to decide "success" vs "failed". A halted strict-mode
    run leaves ``healthcheck.json`` behind, which the harness would count
    as a successful submission — so do not enable strict on a live run.

  - ``warn``: run the probes, write the report, continue regardless of
    failures. Safe for the official harness because the entrypoint still
    proceeds to produce real ``solutions.json``. Useful when you want
    a token-cost-accounted preflight diagnostic without changing run
    behavior.

  - ``strict``: **manual dry-run only.** Every API probe (Author, Critic,
    every Council member, Compute) must succeed. On failure, the script
    writes a detailed report and *halts the container* — it does **not**
    exit non-zero — so the AWS instance is not torn down. The operator
    can SSH in, inspect ``/data/output/healthcheck.json``, fix the
    broken secret out-of-band, and either wait for the re-probe to
    recover or ``touch /data/output/healthcheck.proceed`` to proceed
    with the known failures. Not safe for the autonomous harness: if
    the harness's ``timeout_minutes`` fires while we are halted, the
    instance terminates and the partial output looks like a successful
    submission.

The script is also runnable standalone (``python
scripts/firstproof_healthcheck.py``) for ad-hoc credential checks
outside the docker entrypoint.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_OUTPUT_DIR = Path("/data/output")
DEFAULT_WORKFLOW = "author_critic_long"
DEFAULT_MODE = "off"

HEALTHCHECK_SENTINEL = "HEALTHCHECK_OK"
PROCEED_FILE = "healthcheck.proceed"
REPORT_FILE = "healthcheck.json"

WAIT_INTERVAL_S = 60
HEARTBEAT_INTERVAL_S = 300

API_PROBE_TIMEOUT_S = 30
COMPUTE_PROBE_TIMEOUT_S = 600

ROLE_AUTHOR = "author"
ROLE_CRITIC = "critic"
ROLE_COUNCIL_PREFIX = "council:"
ROLE_COMPUTE = "compute"


# Load .env BEFORE importing project modules.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from _env import load_dotenv_file  # noqa: E402

load_dotenv_file(REPO_ROOT / ".env")


@dataclass
class Settings:
    output_dir: Path
    workflow: str
    mode: str  # "strict" | "warn" | "off"
    wait_interval_s: int
    heartbeat_interval_s: int
    skip_compute: bool


@dataclass
class Probe:
    role: str
    model_ref: str
    tools: list[Any] = field(default_factory=list)
    # Only used by the compute probe.
    compute_cost_config: str | None = None
    compute_reasoning_effort: str | None = None
    compute_sandbox_backend: str | None = None
    compute_codex_sandbox: str | None = None
    compute_docker_image: str | None = None


@dataclass
class ProbeResult:
    role: str
    model_ref: str
    ok: bool
    duration_s: float
    started_at: str
    finished_at: str
    response_excerpt: str = ""
    error_type: str = ""
    error_message: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum)


def _settings_from_env(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workflow", default=None)
    parser.add_argument("--mode", choices=("strict", "warn", "off"), default=None)
    parser.add_argument("--skip-compute", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir or Path(
        os.environ.get("FIRSTPROOF_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR
    )
    workflow = args.workflow or os.environ.get("FIRSTPROOF_WORKFLOW") or DEFAULT_WORKFLOW
    mode = (args.mode or os.environ.get("FIRSTPROOF_HEALTHCHECK") or DEFAULT_MODE).lower()
    if mode not in ("strict", "warn", "off"):
        mode = DEFAULT_MODE
    skip_compute = args.skip_compute or os.environ.get(
        "FIRSTPROOF_HEALTHCHECK_SKIP_COMPUTE"
    ) in ("1", "true", "True")
    return Settings(
        output_dir=output_dir,
        workflow=workflow,
        mode=mode,
        wait_interval_s=_read_int_env("FIRSTPROOF_HEALTHCHECK_WAIT_INTERVAL_S", WAIT_INTERVAL_S),
        heartbeat_interval_s=_read_int_env(
            "FIRSTPROOF_HEALTHCHECK_HEARTBEAT_INTERVAL_S", HEARTBEAT_INTERVAL_S
        ),
        skip_compute=skip_compute,
    )


def _author_critic_tools() -> list[Any]:
    """Tools the Author and Critic use at runtime.

    Mirrors ``ACCritic.extra_client_kwargs`` so the probe call matches the
    real call's wiring (catches "tools field unsupported on this provider"
    errors, not just auth failures).
    """
    return [
        (None, {"type": "code_interpreter", "container": {"type": "auto"}}),
        (None, {"type": "web_search_preview"}),
    ]


def _resolve_model(
    preset: Any,
    *,
    component_name: str,
    component_default: str,
) -> str:
    """Mirror ``RunContext.model_for``: per-agent override > "*" wildcard
    > ``components[<name>].model`` > class default.

    Probe discovery used to read only ``component_configs[name].model``,
    which meant a preset with ``model_overrides: {Author: ...}`` or
    ``model_overrides: {"*": ...}`` would be preflighted against the
    wrong model. Read both layers here.
    """
    overrides = getattr(preset, "model_overrides", {}) or {}
    if component_name in overrides:
        return str(overrides[component_name])
    if "*" in overrides:
        return str(overrides["*"])
    component_configs = getattr(preset, "component_configs", {}) or {}
    component_cfg = component_configs.get(component_name) or {}
    return str(component_cfg.get("model") or component_default)


def discover_probes(preset: Any, *, skip_compute: bool = False) -> list[Probe]:
    """Build the probe list from a loaded WorkflowPreset.

    Author + Critic share a model in ``author_critic_long`` (both
    ``gpt-55-pro``); we still probe them as distinct roles because they
    map to distinct prompts/tool configs at runtime and might be split
    in future presets. Per-role models honour ``preset.model_overrides``
    so a preset that swaps Author to a different provider is preflighted
    against that provider, not the default.
    """
    inputs = getattr(preset, "inputs", {}) or {}

    author_model = _resolve_model(
        preset, component_name="Author", component_default="models/openai/gpt-55-pro"
    )
    critic_model = _resolve_model(
        preset, component_name="ACCritic", component_default="models/openai/gpt-55-pro"
    )

    probes: list[Probe] = [
        Probe(role=ROLE_AUTHOR, model_ref=author_model, tools=_author_critic_tools()),
        Probe(role=ROLE_CRITIC, model_ref=critic_model, tools=_author_critic_tools()),
    ]

    enable_council = bool(inputs.get("enable_council", True))
    council_models = inputs.get("council_models") or []
    if enable_council and isinstance(council_models, list):
        for declared_ref in council_models:
            if not isinstance(declared_ref, str) or not declared_ref:
                continue
            # Council members go through the same ``RunContext.model_for``
            # path at runtime: a global ``"*"`` override wins. A per-member
            # override doesn't apply here (the member's name varies), but
            # the wildcard does.
            overrides = getattr(preset, "model_overrides", {}) or {}
            effective_ref = str(overrides.get("*", declared_ref))
            probes.append(
                Probe(
                    role=f"{ROLE_COUNCIL_PREFIX}{_short_label(declared_ref)}",
                    model_ref=effective_ref,
                    tools=[],
                )
            )

    enable_compute = bool(inputs.get("enable_compute", True))
    if enable_compute and not skip_compute:
        # The Compute agent receives ``model`` as one of its inputs, so
        # ``preset.inputs.compute_model`` is the source of truth — a
        # ``model_overrides`` entry would not apply (Compute isn't routed
        # through ``ctx.model_for``). Mirror that here.
        compute_model = inputs.get("compute_model") or "gpt-5.5"
        probes.append(
            Probe(
                role=ROLE_COMPUTE,
                model_ref=str(compute_model),
                tools=[],
                compute_cost_config=str(
                    inputs.get("compute_cost_config") or "models/openai/gpt-55-high"
                ),
                compute_reasoning_effort=str(
                    inputs.get("compute_reasoning_effort") or "xhigh"
                ),
                compute_sandbox_backend=str(
                    inputs.get("compute_sandbox_backend") or "subprocess"
                ),
                compute_codex_sandbox=str(
                    inputs.get("compute_codex_sandbox") or "docker-bypass"
                ),
                compute_docker_image=(
                    str(inputs["compute_docker_image"])
                    if inputs.get("compute_docker_image")
                    else None
                ),
            )
        )

    return probes


def _short_label(model_ref: str) -> str:
    parts = model_ref.rsplit("/", 1)
    return parts[-1] if parts else model_ref


HEALTHCHECK_PROMPT = (
    "You are participating in a preflight healthcheck. Briefly (under 80 words) "
    "describe what tools, if any, you have access to in this call — name each "
    "one explicitly. Then, on a new line *by itself*, output exactly the "
    f"sentinel token {HEALTHCHECK_SENTINEL} and nothing after it."
)


def _excerpt(text: str, n: int = 800) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n] + f"... [+{len(text) - n} chars]"


_SIGNAL_PATTERNS = (
    "Got OpenAI error in responses api inner",
    "Got OpenAI CC non ratelimit error",
    "Got Anthropic error in",
    "Got Google error in",
    "API key not valid",
    "invalid x-api-key",
    "Incorrect API key",
    "Error code: 401",
    "Error code: 403",
    "Error code: 404",
    "API_KEY_INVALID",
    "model_not_found",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
)


_ERROR_PATTERNS = (
    "Error in outer retries",
    "Max inner retries reached",
    "Wallclock budget",
    "rate limit",
)


def _pick_signal_log(captured: list[str]) -> str:
    """Pick the log line most likely to explain the failure to the operator.

    Prefer lines matching known upstream-error markers (401, model-not-found,
    etc.) over the generic "Max inner retries reached" / "Traceback" lines
    that come from APIClient's retry supervisor.
    """
    for line in captured:
        if any(marker in line for marker in _SIGNAL_PATTERNS):
            return line
    return captured[-1] if captured else ""


async def probe_api_model(probe: Probe) -> ProbeResult:
    """Send a small message and assert the sentinel comes back.

    APIClient swallows the upstream provider exception (401, model-not-found,
    etc.) and returns an empty conversation on failure. To surface the real
    cause in the report, we attach a loguru sink during the call and capture
    any WARNING+ records — they typically contain the provider's HTTP response.

    Two oddities of APIClient's retry path that we work around here:

    1. ``max_retries=0`` *skips* the call entirely because the loop is
       ``while retry_idx < max_retries``. We use ``max_retries=1`` so
       one attempt actually happens.
    2. The OpenAI inner-retry handler does a hardcoded ``time.sleep(60)``
       after non-rate-limit errors. We monkey-patch ``time.sleep`` for
       the duration of the probe so the probe fails in ~1s instead of ~60s.
       Successful calls don't sleep, so this is a no-op on the happy path.
    """
    started_at = _utc_now()
    started = time.monotonic()
    response_text = ""
    error_type = ""
    error_message = ""
    ok = False
    detail: dict[str, Any] = {}
    captured_logs: list[str] = []
    usage: dict[str, Any] = {}
    try:
        from loguru import logger as _logger  # local import: optional dep
    except ImportError:
        _logger = None

    sink_id = None
    if _logger is not None:
        def _sink(message: Any) -> None:
            text = str(message).rstrip("\n")
            # Only keep the noisy upstream-error lines we know how to
            # interpret — drop generic INFO chatter like "Running N queries".
            if any(marker in text for marker in _SIGNAL_PATTERNS + _ERROR_PATTERNS):
                captured_logs.append(text)
        # INFO because the Google error in the OpenAI-CC path is logged
        # at INFO level (see ``Got OpenAI CC non ratelimit error``). We
        # filter for known error markers in the sink to avoid noise.
        sink_id = _logger.add(_sink, level="INFO", format="{message}")

    try:
        def _do_call() -> str:
            import time as _time

            from mathagents import APIClient, load_solver_config  # local import: heavy

            cfg = load_solver_config(probe.model_ref)
            cfg = {k: v for k, v in cfg.items() if not k.startswith("__")}
            if probe.tools:
                cfg["tools"] = probe.tools
                cfg["max_tool_calls"] = 0  # zero — we don't want the model to *use* tools, just see them
            # Fail fast: a 401 / model-not-found is not transient. Use
            # ``max_retries=1`` so we make ONE outer attempt and any backoff
            # exits immediately; ``max_retries=0`` would skip the call
            # entirely (the retry loop is ``while retry_idx < max_retries``).
            # Tight ``max_retries_inner`` keeps a single failing OpenAI /
            # Google call from looping for tens of seconds.
            cfg["max_retries"] = 1
            cfg["max_retries_inner"] = 1
            cfg["sleep_on_error"] = 0
            cfg["sleep_after_request"] = 0
            cfg["max_wallclock_per_call_s"] = float(API_PROBE_TIMEOUT_S)
            # Force foreground: gpt-55-pro.yaml sets ``background=true``
            # which uses 60s ``time.sleep`` poll loops to retrieve the
            # response. Those polls block the worker thread past any
            # asyncio cancellation, so a 401 probe could hang for minutes.
            cfg["background"] = False
            # H2: also bound the OpenAI / httpx layer timeout. APIClient
            # passes ``timeout`` through to the SDK; gpt-55-pro inherits
            # 4200s from its YAML config, which means a wedged provider
            # request keeps the worker thread alive long after
            # ``asyncio.wait_for`` cancels.
            cfg["timeout"] = float(API_PROBE_TIMEOUT_S)
            # Patch ``time.sleep`` to skip APIClient's hardcoded 60s
            # post-error backoff. Successful calls don't sleep, so this is
            # a no-op on the happy path.
            _orig_sleep = _time.sleep
            _time.sleep = lambda *_a, **_k: None
            try:
                client = APIClient(**cfg)
                messages = [[{"role": "user", "content": HEALTHCHECK_PROMPT}]]
                iterator = client.run_queries(messages, no_tqdm=True)
                _, conversation, detailed_cost = next(iter(iterator))
                usage.update(detailed_cost or {})
            finally:
                _time.sleep = _orig_sleep
            for msg in reversed(conversation):
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts: list[str] = []
                    for block in content:
                        if isinstance(block, dict):
                            if "text" in block:
                                parts.append(block["text"])
                            elif block.get("type") == "output_text":
                                parts.append(block.get("text", ""))
                    if parts:
                        return "\n".join(parts)
            return ""

        response_text = await asyncio.wait_for(
            asyncio.to_thread(_do_call), timeout=API_PROBE_TIMEOUT_S
        )
        if HEALTHCHECK_SENTINEL in response_text:
            ok = True
        else:
            ok = False
            if response_text.strip():
                error_type = "MissingSentinel"
                error_message = (
                    f"Response did not contain sentinel {HEALTHCHECK_SENTINEL!r}; "
                    "the model is reachable but its response shape is wrong."
                )
            else:
                error_type = "EmptyResponse"
                error_message = "Model returned an empty response."
    except asyncio.TimeoutError:
        ok = False
        error_type = "Timeout"
        error_message = f"Probe exceeded {API_PROBE_TIMEOUT_S}s."
    except Exception as exc:  # noqa: BLE001 — surface everything
        ok = False
        error_type = type(exc).__name__
        error_message = str(exc) or repr(exc)
    finally:
        if sink_id is not None and _logger is not None:
            _logger.remove(sink_id)

    if not ok and captured_logs:
        # APIClient logged the upstream provider exception but swallowed it.
        # Attach the captured stderr so the report is self-diagnosable.
        detail["upstream_errors"] = captured_logs
        signal = _pick_signal_log(captured_logs)
        if signal and "EmptyResponse" in error_type:
            error_message = (
                f"{error_message} Upstream provider error: {signal[:400]}"
            )

    duration = round(time.monotonic() - started, 3)
    return ProbeResult(
        role=probe.role,
        model_ref=probe.model_ref,
        ok=ok,
        duration_s=duration,
        started_at=started_at,
        finished_at=_utc_now(),
        response_excerpt=_excerpt(response_text),
        error_type=error_type,
        error_message=error_message,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cached_input_tokens=int(usage.get("cached_input_tokens") or 0),
        reasoning_tokens=int(usage.get("reasoning_tokens") or 0),
        cost_usd=float(usage.get("cost") or 0.0),
        detail=detail,
    )


COMPUTE_HEALTHCHECK_SENTINEL = "CAS_HEALTHCHECK_OK"


# Expressions the worker is allowed to pick — pin to a tiny known set so
# the validator can deterministically check that the recorded ``result``
# is actually the right answer (not just a plausible-looking string).
_COMPUTE_PROBE_ALLOWED_EXPRS = {
    "1+1": "2",
    "2+2": "4",
    "3+4": "7",
    "2*3": "6",
}


COMPUTE_PROBE_INSTRUCTIONS = f"""\
This is a *preflight healthcheck*, not a real task. Verify your
sandbox is operational, then finish quickly.

Concretely:

1. For each binary in [``sage``, ``gap``, ``singular``, ``gp``]:
   probe with ``command -v <bin>`` and record the resolved path
   (or the empty string ``""`` if the binary is not on PATH).
2. Pick one CAS that *is* on PATH (non-empty path in step 1) and call
   it to evaluate **exactly one** of the following expressions —
   pick whichever is easiest to express in the chosen CAS:
   ``1+1``, ``2+2``, ``3+4``, ``2*3``.
   For example, ``sage -c 'print(1+1)'`` or
   ``echo "Print(2+2); QUIT;" | gap -q``. Record the engine you used,
   the expression *string* you evaluated, and the numeric result the
   CAS produced.
3. Write a JSON report at ``responses/response_round_1.md``. The file
   must contain **valid JSON only** — no ``//`` comments, no prose,
   no ``` ``` ``` fences. Use exactly this shape::

       {{
         "sentinel": "{COMPUTE_HEALTHCHECK_SENTINEL}",
         "binaries": {{
           "sage": "/usr/bin/sage",
           "gap":  "/usr/bin/gap",
           "singular": "/usr/bin/Singular",
           "gp":   "/usr/bin/gp"
         }},
         "computation": {{
           "engine": "sage",
           "expr": "1+1",
           "result": "2"
         }}
       }}

   Field rules (do NOT include this list in the JSON file):

   - ``sentinel`` must equal ``{COMPUTE_HEALTHCHECK_SENTINEL}`` literally.
   - ``binaries[name]`` is the path printed by ``command -v <name>``,
     or the empty string ``""`` if the binary is not on PATH.
   - At least one ``binaries`` entry must be a non-empty path.
   - ``computation.engine`` must be one of the names in ``binaries``
     whose path is non-empty.
   - ``computation.expr`` must be exactly one of
     ``"1+1"``, ``"2+2"``, ``"3+4"``, ``"2*3"``.
   - ``computation.result`` must be the numeric answer your CAS
     actually printed, as a string.
4. Invoke ``finish '{{"status": "done", "summary": "healthcheck ok"}}'``.

Total wall time should be under 5 minutes. Do not attempt to
download papers, write code, or do anything beyond the steps above.
"""


async def probe_compute_worker(probe: Probe) -> ProbeResult:
    """Spin up the Compute agent on a throwaway workspace.

    Reuses the real ``proofstack.agents.ac.Compute`` class so the probe
    exercises the same codex CLI subprocess + sandbox backend + cost
    accounting hooks the runtime will use.
    """
    started_at = _utc_now()
    started = time.monotonic()
    response_text = ""
    error_type = ""
    error_message = ""
    ok = False
    detail: dict[str, Any] = {}
    workspace_dir = Path(tempfile.mkdtemp(prefix="firstproof-healthcheck-"))
    try:
        from proofstack.agents.ac import Compute
        from proofstack.context import RunContext

        workdir = workspace_dir / "agent"
        workdir.mkdir(parents=True, exist_ok=True)
        ctx = RunContext.create(
            run_id="firstproof-healthcheck",
            root_workdir=workdir,
            flat=True,
        )
        compute = Compute(ctx)
        compute_workspace = workspace_dir / "compute"
        compute_workspace.mkdir(parents=True, exist_ok=True)

        compute_kwargs: dict[str, Any] = dict(
            problem="(healthcheck — no problem)",
            problem_id="healthcheck",
            round=1,
            instructions=COMPUTE_PROBE_INSTRUCTIONS,
            compute_workspace=compute_workspace,
            model=probe.model_ref,
            reasoning_effort=probe.compute_reasoning_effort or "low",
            cost_config=probe.compute_cost_config or "models/openai/gpt-55-high",
            sandbox_backend=probe.compute_sandbox_backend or "subprocess",
            codex_sandbox=probe.compute_codex_sandbox or "docker-bypass",
        )
        # Forward the docker image only when the preset actually pinned one
        # AND we are about to use the docker backend; otherwise let Compute
        # use its built-in default. Probing with the wrong image is what
        # we are *trying* to avoid here.
        if (
            probe.compute_docker_image
            and (probe.compute_sandbox_backend or "subprocess") == "docker"
        ):
            compute_kwargs["docker_image"] = probe.compute_docker_image
            detail["docker_image"] = probe.compute_docker_image

        async def _run() -> Any:
            return await compute(**compute_kwargs)

        out = await asyncio.wait_for(_run(), timeout=COMPUTE_PROBE_TIMEOUT_S)
        status = getattr(out, "status", "") or ""
        summary = getattr(out, "summary", "") or ""
        response_md = getattr(out, "response_md", "") or ""
        error_attr = getattr(out, "error", None)
        detail["status"] = status
        detail["summary"] = summary
        # Compute.collect falls back to the finish summary when the
        # canonical responses/response_round_1.md is missing — that's
        # acceptable for real runs but masks a broken worker for the
        # healthcheck, so verify the file actually exists AND contains
        # the binary-check + sentinel we asked for.
        canonical_response = compute_workspace / "responses" / "response_round_1.md"
        report_exists = canonical_response.exists()
        detail["response_md_exists"] = report_exists
        usage_from_events = _collect_compute_usage(workdir)
        if usage_from_events:
            detail["usage"] = usage_from_events

        validation_error = None
        if not report_exists:
            validation_error = "canonical response file missing"
        else:
            try:
                report_text = canonical_response.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError as exc:
                validation_error = f"could not read response file: {exc}"
                report_text = ""
            else:
                validation_error = _validate_compute_report(report_text, detail)

        if status == "done" and not error_attr and validation_error is None:
            ok = True
            response_text = response_md or summary
        else:
            ok = False
            error_type = "ComputeWorkerError"
            parts = [f"status={status!r}", f"summary={summary!r}", f"error={error_attr!r}"]
            if validation_error:
                parts.append(f"validation={validation_error!r}")
            error_message = "; ".join(parts)
            response_text = response_md or summary
    except asyncio.TimeoutError:
        ok = False
        error_type = "Timeout"
        error_message = f"Compute probe exceeded {COMPUTE_PROBE_TIMEOUT_S}s."
    except Exception as exc:  # noqa: BLE001
        ok = False
        error_type = type(exc).__name__
        error_message = str(exc) or repr(exc)
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)

    duration = round(time.monotonic() - started, 3)
    usage = detail.get("usage") or {}
    return ProbeResult(
        role=probe.role,
        model_ref=probe.model_ref,
        ok=ok,
        duration_s=duration,
        started_at=started_at,
        finished_at=_utc_now(),
        response_excerpt=_excerpt(response_text),
        error_type=error_type,
        error_message=error_message,
        input_tokens=int(usage.get("in_tokens") or 0),
        output_tokens=int(usage.get("out_tokens") or 0),
        cached_input_tokens=int(usage.get("cached_in_tokens") or 0),
        reasoning_tokens=int(usage.get("reasoning_out_tokens") or 0),
        cost_usd=float(usage.get("cost_usd") or 0.0),
        detail=detail,
    )


_CAS_BINARY_NAMES = ("sage", "gap", "singular", "gp")


def _validate_compute_report(report_text: str, detail: dict[str, Any]) -> str | None:
    """Verify the worker's response_round_1.md actually exercises the CAS stack.

    Returns ``None`` on success, an error string otherwise. Updates
    ``detail`` with the parsed report so the operator can see what came
    back even on failure.
    """
    if not report_text.strip():
        return "response file is empty"
    parsed: Any = None
    try:
        parsed = json.loads(report_text)
    except json.JSONDecodeError:
        # Some models wrap JSON in ```json fences or prose; try to recover.
        snippet = report_text.strip()
        if snippet.startswith("```"):
            snippet = snippet.strip("`")
            if snippet.lower().startswith("json"):
                snippet = snippet[4:].strip()
            if "```" in snippet:
                snippet = snippet.split("```", 1)[0]
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            parsed = None

    if not isinstance(parsed, dict):
        # The prompt mandates a JSON object; refuse to fall back to
        # prose-grepping for the sentinel — that would let the worker
        # pass without producing a structured binary table or running
        # an actual CAS computation.
        detail["report_raw_excerpt"] = _excerpt(report_text, 400)
        return "response file is not a valid JSON object"

    detail["report_parsed"] = parsed
    sentinel = parsed.get("sentinel") or parsed.get("Sentinel")
    if sentinel != COMPUTE_HEALTHCHECK_SENTINEL:
        return (
            f"report JSON parsed but sentinel field was "
            f"{sentinel!r}, expected {COMPUTE_HEALTHCHECK_SENTINEL!r}"
        )
    binaries = parsed.get("binaries")
    if not isinstance(binaries, dict):
        return "report JSON parsed but the 'binaries' field is missing"
    # Require at least one CAS binary with a non-empty path. A report
    # claiming all four are empty would otherwise pass the "structural"
    # checks despite signalling that the entire CAS stack is
    # unavailable — exactly what the preflight is meant to catch.
    non_empty = {
        name: str(binaries.get(name, "")).strip()
        for name in _CAS_BINARY_NAMES
        if str(binaries.get(name, "")).strip()
    }
    if not non_empty:
        return (
            "report JSON parsed but every entry in 'binaries' is "
            f"empty — the worker found none of {_CAS_BINARY_NAMES} "
            "on PATH"
        )
    computation = parsed.get("computation")
    if not isinstance(computation, dict):
        return "report JSON parsed but 'computation' is missing"
    engine = str(computation.get("engine") or "").strip()
    if engine not in non_empty:
        return (
            f"report JSON parsed but 'computation.engine'={engine!r} "
            "does not point at a CAS binary with a non-empty path; "
            f"non-empty binaries are {sorted(non_empty)}"
        )
    expr = str(computation.get("expr") or "").strip()
    result = str(computation.get("result") or "").strip()
    if not result:
        return "report JSON parsed but 'computation.result' is empty"
    # Pin to a known answer table so a fabricated report can't pass.
    expected = _COMPUTE_PROBE_ALLOWED_EXPRS.get(expr.replace(" ", ""))
    if expected is None:
        return (
            f"report JSON parsed but 'computation.expr'={expr!r} is "
            f"not one of the allowed expressions "
            f"{sorted(_COMPUTE_PROBE_ALLOWED_EXPRS)}"
        )
    if result.replace(".0", "") != expected:
        return (
            f"report JSON parsed but 'computation.result'={result!r} "
            f"does not match the expected value {expected!r} for "
            f"expression {expr!r}"
        )
    return None


def _collect_compute_usage(workdir: Path) -> dict[str, Any]:
    """Pull the most recent ``model.call`` event from the run's events log."""
    events_path = workdir / "events.jsonl"
    if not events_path.exists():
        return {}
    try:
        with events_path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return {}
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict) and ev.get("kind") == "model.call":
            payload = ev.get("payload") or {}
            if isinstance(payload, dict):
                return payload
    return {}


async def run_probe(probe: Probe) -> ProbeResult:
    if probe.role == ROLE_COMPUTE:
        return await probe_compute_worker(probe)
    if probe.role == ROLE_AUTHOR and _is_openai_model_ref(probe.model_ref):
        return await probe_author_files_path(probe)
    return await probe_api_model(probe)


def _is_openai_model_ref(model_ref: str) -> bool:
    """Best-effort: the Files+code_interpreter round-trip is OpenAI-specific.

    Don't try to probe ``files.create`` against an Anthropic / Gemini ref.
    """
    return "openai" in (model_ref or "").lower()


AUTHOR_PROBE_PROMPT = (
    "You are participating in a preflight healthcheck of the file-attached "
    "code_interpreter path used by the Author.\n\n"
    "A small file has been uploaded and attached at the read-only input "
    "path shown in the workspace listing below. You **must**:\n\n"
    "1. Use code_interpreter to open the attached file (e.g. "
    "``with open(p) as f: print(f.read())``).\n"
    "2. Use code_interpreter to write a copy of that file's contents back "
    "to the canonical writable path ``/mnt/data/notes.tex``. The "
    "downloaded copy of this file is what the healthcheck verifies.\n"
    f"3. On a new line by itself, output the sentinel {HEALTHCHECK_SENTINEL}.\n"
)


async def probe_author_files_path(probe: Probe) -> ProbeResult:
    """Exercise the Author's runtime path: OpenAI Files.create +
    code_interpreter w/ ``container.file_ids`` + ``containers.files.list``
    + per-file download + ``files.delete`` cleanup.

    The generic sentinel probe (``probe_api_model``) only catches model-call
    failures. This probe catches Files-API permission issues, container-files
    download permission, and shape changes in the OpenAI Containers API.
    """
    started_at = _utc_now()
    started = time.monotonic()
    response_text = ""
    error_type = ""
    error_message = ""
    ok = False
    detail: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    captured_logs: list[str] = []
    try:
        from loguru import logger as _logger
    except ImportError:
        _logger = None

    sink_id = None
    if _logger is not None:
        def _sink(message: Any) -> None:
            text = str(message).rstrip("\n")
            if any(marker in text for marker in _SIGNAL_PATTERNS + _ERROR_PATTERNS):
                captured_logs.append(text)
        sink_id = _logger.add(_sink, level="INFO", format="{message}")

    workspace_dir: Path | None = None
    probe_token = f"preflight-{int(started * 1000)}"
    try:
        workspace_dir = Path(tempfile.mkdtemp(prefix="firstproof-author-probe-"))
        canonical = workspace_dir / "notes.tex"
        canonical.write_text(
            "% Author preflight probe — this file is uploaded as user_data,\n"
            "% attached to code_interpreter via container.file_ids, and\n"
            "% downloaded back to verify the round-trip.\n"
            f"PROBE_TOKEN={probe_token}\n",
            encoding="utf-8",
        )

        def _do_call() -> tuple[str, dict[str, Any], dict[str, Any]]:
            import time as _time

            from openai import OpenAI

            from mathagents import APIClient, load_solver_config
            from proofstack.agents.ac.container_files import (
                ContainerFileBridge,
                find_container_id,
            )

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set in the container env.")
            # ``asyncio.wait_for(asyncio.to_thread(...))`` at the call site
            # only cancels the awaiting coroutine on timeout — the worker
            # thread keeps running, and so does whatever HTTP request the
            # raw client started. Pin the SDK-level httpx timeout and
            # disable retries so files / containers calls actually fail
            # within the advertised ``API_PROBE_TIMEOUT_S`` budget rather
            # than blocking the entrypoint past the warn/strict gate.
            openai_client = OpenAI(
                api_key=api_key,
                timeout=float(API_PROBE_TIMEOUT_S),
                max_retries=0,
            )

            bridge = ContainerFileBridge(
                openai_client=openai_client,
                workspace=workspace_dir,  # type: ignore[arg-type]
                names=("notes.tex",),
            )

            _orig_sleep = _time.sleep
            _time.sleep = lambda *_a, **_k: None
            stage = "files.create"
            try:
                file_ids = bridge.upload()
                stage = "responses.create"
                cfg = load_solver_config(probe.model_ref)
                cfg = {k: v for k, v in cfg.items() if not k.startswith("__")}
                cfg["tools"] = [
                    (
                        None,
                        {
                            "type": "code_interpreter",
                            "container": {"type": "auto", "file_ids": file_ids},
                        },
                    ),
                    (None, {"type": "web_search_preview"}),
                ]
                cfg["max_tool_calls"] = 4  # cap CI usage so a chatty model doesn't blow the budget
                cfg["max_retries"] = 1
                cfg["max_retries_inner"] = 1
                cfg["sleep_on_error"] = 0
                cfg["sleep_after_request"] = 0
                cfg["max_wallclock_per_call_s"] = float(API_PROBE_TIMEOUT_S)
                cfg["background"] = False
                # H2: APIClient passes ``timeout`` through to the OpenAI /
                # httpx layer. Without overriding it here, gpt-55-pro
                # inherits the 4200s wallclock from its YAML config, which
                # means a wedged provider request keeps the worker thread
                # alive long after the ``asyncio.wait_for`` cancels.
                cfg["timeout"] = float(API_PROBE_TIMEOUT_S)
                listing = bridge.render_workspace_listing()
                prompt = f"{AUTHOR_PROBE_PROMPT}\n\n# Workspace files\n{listing}\n"
                client = APIClient(**cfg)
                messages = [[{"role": "user", "content": prompt}]]
                _, conversation, detailed_cost = next(
                    iter(client.run_queries(messages, no_tqdm=True))
                )
                text = ""
                for msg in reversed(conversation):
                    if msg.get("role") != "assistant":
                        continue
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        text = content
                        break
                    if isinstance(content, list):
                        parts: list[str] = []
                        for block in content:
                            if isinstance(block, dict):
                                if "text" in block:
                                    parts.append(block["text"])
                                elif block.get("type") == "output_text":
                                    parts.append(block.get("text", ""))
                        if parts:
                            text = "\n".join(parts)
                            break
                container_id = find_container_id(conversation)
                stage = "containers.files.list"
                if container_id is None:
                    # A model that returns the sentinel without invoking
                    # code_interpreter would otherwise let this probe pass
                    # while leaving the Containers API completely
                    # unexercised. Force a failure.
                    raise RuntimeError(
                        "model response had no code_interpreter_call — "
                        "the Containers API path was never exercised; "
                        "either the model didn't use the tool, or the "
                        "Responses API didn't return a container_id."
                    )
                try:
                    downloaded = bridge.download(container_id)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"containers.files round-trip failed for "
                        f"container_id={container_id!r}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                return text, detailed_cost or {}, {
                    "stage_reached": "ok",
                    "container_id": container_id,
                    "downloaded_files": sorted(downloaded.keys()),
                    "file_ids_uploaded": len(file_ids),
                    "downloaded_contents": downloaded,
                }
            except Exception as exc:
                # Attach the stage label so the operator knows whether
                # files.create / responses.create / containers.files.list
                # was the layer that broke.
                exc.__notes__ = getattr(exc, "__notes__", []) + [f"stage={stage}"]
                raise
            finally:
                _time.sleep = _orig_sleep
                try:
                    bridge.cleanup()
                except Exception:
                    pass

        response_text, detailed_cost, stage_info = await asyncio.wait_for(
            asyncio.to_thread(_do_call), timeout=API_PROBE_TIMEOUT_S
        )
        usage.update(detailed_cost)
        # Strip the raw download contents from ``detail`` before storing
        # to keep the JSON report small; the validation below still uses
        # it.
        downloaded_contents = stage_info.pop("downloaded_contents", {})
        detail.update(stage_info)
        downloaded_names = stage_info.get("downloaded_files") or []
        notes_body = downloaded_contents.get("notes.tex", "") if isinstance(
            downloaded_contents, dict
        ) else ""
        if not response_text.strip():
            ok = False
            error_type = "EmptyResponse"
            error_message = (
                "Author file-path round-trip returned an empty model response."
            )
        elif HEALTHCHECK_SENTINEL not in response_text:
            ok = False
            error_type = "MissingSentinel"
            error_message = (
                f"Author file-path round-trip succeeded but model response "
                f"did not contain sentinel {HEALTHCHECK_SENTINEL!r}."
            )
        elif "notes.tex" not in downloaded_names:
            # H1: the model returned the sentinel but never wrote the
            # canonical path back; the Containers API download path was
            # exercised, but the model bypassed our round-trip
            # instruction. That likely means a broken instruction
            # template, not a broken provider — fail loud anyway.
            ok = False
            error_type = "MissingCanonicalDownload"
            error_message = (
                "Author file-path round-trip: container_id and "
                "containers.files.list succeeded, but the model did not "
                "write /mnt/data/notes.tex (so the per-file download has "
                "nothing to compare against the upload). Got files: "
                f"{downloaded_names}."
            )
        elif probe_token not in notes_body:
            ok = False
            error_type = "ProbeTokenMismatch"
            error_message = (
                "Author file-path round-trip: notes.tex was downloaded "
                "from the container but its contents did not include the "
                f"probe token {probe_token!r}. The model may have "
                "overwritten the file with unrelated content, or the "
                "container is returning stale data."
            )
        else:
            ok = True
    except asyncio.TimeoutError:
        ok = False
        error_type = "Timeout"
        error_message = (
            f"Author file-path probe exceeded {API_PROBE_TIMEOUT_S}s."
        )
    except Exception as exc:  # noqa: BLE001
        ok = False
        error_type = type(exc).__name__
        notes = " ".join(getattr(exc, "__notes__", []) or [])
        suffix = f" ({notes})" if notes else ""
        error_message = (str(exc) or repr(exc)) + suffix
    finally:
        if sink_id is not None and _logger is not None:
            _logger.remove(sink_id)
        if workspace_dir is not None:
            shutil.rmtree(workspace_dir, ignore_errors=True)

    if not ok and captured_logs:
        detail["upstream_errors"] = captured_logs
        signal = _pick_signal_log(captured_logs)
        if signal and error_type in ("EmptyResponse", "MissingSentinel"):
            error_message = (
                f"{error_message} Upstream provider error: {signal[:400]}"
            )

    duration = round(time.monotonic() - started, 3)
    return ProbeResult(
        role=probe.role,
        model_ref=probe.model_ref,
        ok=ok,
        duration_s=duration,
        started_at=started_at,
        finished_at=_utc_now(),
        response_excerpt=_excerpt(response_text),
        error_type=error_type,
        error_message=error_message,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cached_input_tokens=int(usage.get("cached_input_tokens") or 0),
        reasoning_tokens=int(usage.get("reasoning_tokens") or 0),
        cost_usd=float(usage.get("cost") or 0.0),
        detail=detail,
    )


async def run_all_probes(probes: list[Probe]) -> list[ProbeResult]:
    """Run all probes serially.

    Serial (not parallel) because we attach a loguru sink during each
    probe to capture the upstream provider exception, and overlapping
    sinks would attribute one probe's 401 to a different probe's
    ``upstream_errors`` field. Each API probe completes in ~1–3s with
    ``max_retries=1`` + ``max_retries_inner=1``, so total wallclock is
    typically under 20s for the five-role default. The compute probe
    is naturally last because of where it sits in the discovery order.
    """
    return [await run_probe(p) for p in probes]


def build_report(
    settings: Settings,
    results: list[ProbeResult],
    *,
    attempt: int,
    started_at: str,
) -> dict[str, Any]:
    failed = [r for r in results if not r.ok]
    return {
        "started_at": started_at,
        "updated_at": _utc_now(),
        "workflow": settings.workflow,
        "mode": settings.mode,
        "attempt": attempt,
        "all_ok": len(failed) == 0,
        "n_failures": len(failed),
        "failed_roles": [r.role for r in failed],
        "probes": [asdict(r) for r in results],
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_failure_summary(results: list[ProbeResult]) -> None:
    print("\nFirstProof healthcheck — FAILURES", file=sys.stderr)
    print("-" * 72, file=sys.stderr)
    for r in results:
        if r.ok:
            continue
        print(f"  role={r.role}  model_ref={r.model_ref}", file=sys.stderr)
        print(f"    {r.error_type}: {r.error_message}", file=sys.stderr)
    print("-" * 72, file=sys.stderr)


async def wait_for_recovery_or_signal(
    settings: Settings,
    initial_results: list[ProbeResult],
    started_at: str,
) -> list[ProbeResult]:
    """Halt-and-wait loop. Returns the last result snapshot when proceed-conditions are met."""
    proceed_path = settings.output_dir / PROCEED_FILE
    report_path = settings.output_dir / REPORT_FILE
    attempt = 1
    last_heartbeat = 0.0
    results = list(initial_results)

    while True:
        if proceed_path.exists():
            print(
                f"FirstProof healthcheck: operator signal detected at {proceed_path};"
                " proceeding despite failures.",
                file=sys.stderr,
            )
            return results

        now = time.monotonic()
        if now - last_heartbeat >= settings.heartbeat_interval_s:
            failures = [r.role for r in results if not r.ok]
            print(
                f"[{_utc_now()}] FirstProof healthcheck still waiting; "
                f"failing roles: {failures or '(none — last re-probe was clean?)'}",
                file=sys.stderr,
            )
            last_heartbeat = now

        await asyncio.sleep(settings.wait_interval_s)

        # Re-probe the still-failing roles.
        from proofstack.registry import load_preset

        try:
            preset = load_preset(settings.workflow)
            all_probes = discover_probes(preset, skip_compute=settings.skip_compute)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{_utc_now()}] FirstProof healthcheck: re-probe could not load preset: {exc}",
                file=sys.stderr,
            )
            continue

        attempt += 1
        failing_roles = {r.role for r in results if not r.ok}
        retry_probes = [p for p in all_probes if p.role in failing_roles]
        retry_results = await run_all_probes(retry_probes)
        retry_by_role = {r.role: r for r in retry_results}
        results = [retry_by_role.get(r.role, r) if not r.ok else r for r in results]

        report = build_report(settings, results, attempt=attempt, started_at=started_at)
        write_report(report_path, report)

        if all(r.ok for r in results):
            print(
                f"[{_utc_now()}] FirstProof healthcheck: all probes recovered on attempt"
                f" {attempt}; proceeding.",
                file=sys.stderr,
            )
            return results


async def main_async() -> int:
    settings = _settings_from_env()
    if settings.mode == "off":
        print("FirstProof healthcheck: mode=off, skipping")
        return 0

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()

    from proofstack.registry import load_preset

    try:
        preset = load_preset(settings.workflow)
    except Exception as exc:  # noqa: BLE001
        print(
            f"FirstProof healthcheck: could not load preset {settings.workflow!r}: {exc}",
            file=sys.stderr,
        )
        # Cannot probe without a preset; write a degenerate report and
        # respect mode. In strict mode this is a hard failure, but we
        # still cannot "wait and retry" usefully — fall through to the
        # standard wait loop so the operator notices.
        results: list[ProbeResult] = []
        report = build_report(settings, results, attempt=1, started_at=started_at)
        report["preset_load_error"] = str(exc)
        report["all_ok"] = False
        report["n_failures"] = 1
        report["failed_roles"] = ["preset"]
        write_report(settings.output_dir / REPORT_FILE, report)
        if settings.mode == "warn":
            return 0
        print(
            "FirstProof healthcheck (strict): preset load failed. "
            "Halting; touch /data/output/healthcheck.proceed to continue.",
            file=sys.stderr,
        )
        await _wait_for_proceed_only(settings)
        return 0

    probes = discover_probes(preset, skip_compute=settings.skip_compute)
    print(
        f"FirstProof healthcheck: probing {len(probes)} role(s) for workflow "
        f"{settings.workflow!r} (mode={settings.mode})"
    )
    for p in probes:
        print(f"  - {p.role}  ->  {p.model_ref}")

    results = await run_all_probes(probes)
    report = build_report(settings, results, attempt=1, started_at=started_at)
    write_report(settings.output_dir / REPORT_FILE, report)

    if all(r.ok for r in results):
        print("FirstProof healthcheck: all probes ok.")
        return 0

    _print_failure_summary(results)

    if settings.mode == "warn":
        print("FirstProof healthcheck: mode=warn — continuing despite failures.")
        return 0

    print(
        "\nFirstProof healthcheck (strict): HALTING.\n"
        "  The container will not exit — your AWS instance stays up.\n"
        f"  Report: {settings.output_dir / REPORT_FILE}\n"
        f"  To proceed-anyway: touch {settings.output_dir / PROCEED_FILE}\n"
        f"  Otherwise: failures will be re-probed every {settings.wait_interval_s}s.\n",
        file=sys.stderr,
    )
    await wait_for_recovery_or_signal(settings, results, started_at)
    return 0


async def _wait_for_proceed_only(settings: Settings) -> None:
    proceed_path = settings.output_dir / PROCEED_FILE
    last_heartbeat = 0.0
    while not proceed_path.exists():
        now = time.monotonic()
        if now - last_heartbeat >= settings.heartbeat_interval_s:
            print(
                f"[{_utc_now()}] FirstProof healthcheck still waiting on operator signal "
                f"({proceed_path}).",
                file=sys.stderr,
            )
            last_heartbeat = now
        await asyncio.sleep(settings.wait_interval_s)
    print(
        f"FirstProof healthcheck: operator signal detected at {proceed_path}; proceeding.",
        file=sys.stderr,
    )


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
