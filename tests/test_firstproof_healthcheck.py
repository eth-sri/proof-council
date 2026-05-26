"""Unit tests for ``scripts/firstproof_healthcheck.py``.

Cover the parts that don't require an actual API client or codex CLI:

  - probe discovery picks up Author/Critic/Council/Compute roles
  - the proceed-sentinel short-circuits the wait loop
  - re-probe replaces only the failing roles
  - report shape (paths, fields)
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "firstproof_healthcheck.py"


def _load_module():
    """Load ``scripts/firstproof_healthcheck.py`` as a module under a safe name."""
    spec = importlib.util.spec_from_file_location("_firstproof_healthcheck_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hc = _load_module()


@dataclass
class _StubPreset:
    component_configs: dict[str, dict[str, Any]]
    inputs: dict[str, Any]
    model_overrides: dict[str, str] = field(default_factory=dict)


def test_discover_probes_default_long_preset():
    preset = _StubPreset(
        component_configs={
            "Author": {"model": "models/openai/gpt-55-pro"},
            "ACCritic": {"model": "models/openai/gpt-55-pro"},
        },
        inputs={
            "enable_council": True,
            "council_models": [
                "models/openai/gpt-55-pro",
                "models/anthropic/opus_47_max",
                "models/gemini/gemini-31-pro",
            ],
            "enable_compute": True,
            "compute_model": "gpt-5.5",
            "compute_cost_config": "models/openai/gpt-55-high",
            "compute_reasoning_effort": "xhigh",
            "compute_sandbox_backend": "subprocess",
            "compute_codex_sandbox": "docker-bypass",
        },
    )
    probes = hc.discover_probes(preset)
    roles = [p.role for p in probes]
    assert roles == [
        hc.ROLE_AUTHOR,
        hc.ROLE_CRITIC,
        f"{hc.ROLE_COUNCIL_PREFIX}gpt-55-pro",
        f"{hc.ROLE_COUNCIL_PREFIX}opus_47_max",
        f"{hc.ROLE_COUNCIL_PREFIX}gemini-31-pro",
        hc.ROLE_COMPUTE,
    ]
    author = probes[0]
    assert author.model_ref == "models/openai/gpt-55-pro"
    assert author.tools, "Author probe must include the runtime tools list"
    council = probes[2]
    assert council.tools == [], "Council probes are no-tools at runtime"
    compute = probes[-1]
    assert compute.compute_cost_config == "models/openai/gpt-55-high"
    assert compute.compute_sandbox_backend == "subprocess"


def test_discover_probes_skips_disabled_pieces():
    preset = _StubPreset(
        component_configs={},
        inputs={
            "enable_council": False,
            "enable_compute": False,
        },
    )
    probes = hc.discover_probes(preset)
    roles = [p.role for p in probes]
    assert roles == [hc.ROLE_AUTHOR, hc.ROLE_CRITIC]


def test_discover_probes_honors_model_overrides():
    """Per-agent override beats components beats default. A bare ``"*"``
    wildcard also propagates to Author and Critic but not to Compute
    (Compute receives its model as a workflow Input, not via
    ``RunContext.model_for``)."""
    preset = _StubPreset(
        component_configs={
            "Author": {"model": "models/openai/gpt-55-pro"},
            "ACCritic": {"model": "models/openai/gpt-55-pro"},
        },
        inputs={
            "enable_council": True,
            "council_models": ["models/anthropic/opus_47_max"],
            "enable_compute": True,
            "compute_model": "gpt-5.5",
        },
    )
    preset.model_overrides = {  # type: ignore[attr-defined]
        "Author": "models/anthropic/opus_47_max",
        "*": "models/openai/gpt-55-medium",
    }
    probes = hc.discover_probes(preset)
    by_role = {p.role: p.model_ref for p in probes}
    # Per-agent override beats wildcard for Author.
    assert by_role[hc.ROLE_AUTHOR] == "models/anthropic/opus_47_max"
    # Critic has no per-agent override -> wildcard applies.
    assert by_role[hc.ROLE_CRITIC] == "models/openai/gpt-55-medium"
    # Council member: wildcard applies (the seat itself has no override).
    council_ref = next(v for k, v in by_role.items() if k.startswith(hc.ROLE_COUNCIL_PREFIX))
    assert council_ref == "models/openai/gpt-55-medium"
    # Compute is not routed through model_for -> uses input compute_model literally.
    assert by_role[hc.ROLE_COMPUTE] == "gpt-5.5"


def test_discover_probes_compute_docker_image_passthrough():
    preset = _StubPreset(
        component_configs={},
        inputs={
            "enable_council": False,
            "enable_compute": True,
            "compute_model": "gpt-5.5",
            "compute_sandbox_backend": "docker",
            "compute_docker_image": "custom/cas-sandbox:v3",
        },
    )
    probes = hc.discover_probes(preset)
    compute = next(p for p in probes if p.role == hc.ROLE_COMPUTE)
    assert compute.compute_sandbox_backend == "docker"
    assert compute.compute_docker_image == "custom/cas-sandbox:v3"


def test_discover_probes_compute_docker_image_absent_when_not_set():
    preset = _StubPreset(
        component_configs={},
        inputs={"enable_council": False, "enable_compute": True, "compute_model": "gpt-5.5"},
    )
    probes = hc.discover_probes(preset)
    compute = next(p for p in probes if p.role == hc.ROLE_COMPUTE)
    assert compute.compute_docker_image is None


def test_compute_probe_instructions_contain_no_inline_comments():
    """The example JSON inside the prompt must be valid JSON literal —
    if a worker copies it verbatim, ``json.loads`` must succeed.
    Reviewer's catch: a previous version embedded ``//`` comments in
    the example so a literal copy would fail strict validation."""
    text = hc.COMPUTE_PROBE_INSTRUCTIONS
    # Pull the example block: collect contiguous indented lines (7+
    # leading spaces) starting at the first such line containing ``"sentinel"``.
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if '"sentinel"' in line and line.startswith(" " * 6)),
        None,
    )
    assert start is not None, "could not locate JSON example in prompt"
    # Walk forward until the first non-indented non-blank line.
    end = start
    while end < len(lines) and (lines[end].startswith(" " * 6) or lines[end].strip() == ""):
        end += 1
    # Step back to include the opening "{" line.
    open_idx = start - 1
    while open_idx >= 0 and "{" not in lines[open_idx]:
        open_idx -= 1
    assert open_idx >= 0, "could not locate opening brace of JSON example"
    raw_block = "\n".join(lines[open_idx:end])
    # Strip the 7-space indent the docstring adds.
    dedented = "\n".join(
        line[7:] if line.startswith(" " * 7) else line.lstrip() for line in raw_block.splitlines()
    )
    # No JSON-killer markers.
    assert "//" not in dedented, f"`//` comments leak into JSON example:\n{dedented}"
    # And the example must actually parse.
    json.loads(dedented)


def test_prepare_output_dir_removes_stale_healthcheck(tmp_path):
    """Re-running into a re-used output dir must not let a previous
    run's healthcheck.json or healthcheck.proceed leak into the new
    one (reviewer R4)."""
    import importlib.util
    import sys
    ep_path = REPO_ROOT / "scripts" / "firstproof_entrypoint.py"
    spec = importlib.util.spec_from_file_location("_firstproof_ep_test", ep_path)
    assert spec is not None and spec.loader is not None
    ep = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ep
    spec.loader.exec_module(ep)

    (tmp_path / "healthcheck.json").write_text("{\"stale\": true}", encoding="utf-8")
    (tmp_path / "healthcheck.proceed").write_text("", encoding="utf-8")
    ep._prepare_output_dir(tmp_path)
    assert not (tmp_path / "healthcheck.json").exists()
    assert not (tmp_path / "healthcheck.proceed").exists()
    # Subdirs are created lazily by ``_run_problem``/the workflow
    # subprocess so that a crashed adapter cannot pad ``OUTPUT_COUNT > 0``
    # (the bundled run.sh's success heuristic) before any solution exists.
    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "workflow_runs").exists()


def test_discover_probes_skip_compute_flag():
    preset = _StubPreset(
        component_configs={},
        inputs={
            "enable_council": False,
            "enable_compute": True,
            "compute_model": "gpt-5.5",
        },
    )
    probes = hc.discover_probes(preset, skip_compute=True)
    assert all(p.role != hc.ROLE_COMPUTE for p in probes)


def test_build_report_shape():
    settings = hc.Settings(
        output_dir=Path("/tmp/x"),
        workflow="author_critic_long",
        mode="strict",
        wait_interval_s=1,
        heartbeat_interval_s=1,
        skip_compute=False,
    )
    results = [
        hc.ProbeResult(
            role=hc.ROLE_AUTHOR,
            model_ref="models/openai/gpt-55-pro",
            ok=True,
            duration_s=0.1,
            started_at="2026-05-20T16:00:00Z",
            finished_at="2026-05-20T16:00:00Z",
            response_excerpt=hc.HEALTHCHECK_SENTINEL,
        ),
        hc.ProbeResult(
            role=f"{hc.ROLE_COUNCIL_PREFIX}opus_47_max",
            model_ref="models/anthropic/opus_47_max",
            ok=False,
            duration_s=0.2,
            started_at="2026-05-20T16:00:00Z",
            finished_at="2026-05-20T16:00:00Z",
            error_type="AuthenticationError",
            error_message="invalid x-api-key",
        ),
    ]
    report = hc.build_report(settings, results, attempt=1, started_at="2026-05-20T16:00:00Z")
    assert report["all_ok"] is False
    assert report["n_failures"] == 1
    assert report["failed_roles"] == [f"{hc.ROLE_COUNCIL_PREFIX}opus_47_max"]
    assert report["workflow"] == "author_critic_long"
    assert report["mode"] == "strict"
    assert report["attempt"] == 1
    assert len(report["probes"]) == 2


def test_write_report_creates_file(tmp_path: Path):
    path = tmp_path / "out" / "healthcheck.json"
    hc.write_report(path, {"all_ok": True, "probes": []})
    assert path.exists()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["all_ok"] is True


def test_excerpt_truncates_long_text():
    short = hc._excerpt("abc")
    assert short == "abc"
    long = hc._excerpt("x" * 2000, n=100)
    assert long.startswith("x" * 100)
    assert long.endswith("chars]")


def test_wait_for_proceed_only_returns_on_sentinel(tmp_path: Path):
    settings = hc.Settings(
        output_dir=tmp_path,
        workflow="author_critic_long",
        mode="strict",
        wait_interval_s=0,  # tight loop
        heartbeat_interval_s=999,
        skip_compute=False,
    )

    async def driver():
        # Touch the sentinel slightly after we start waiting.
        async def touch_later():
            await asyncio.sleep(0.05)
            (tmp_path / hc.PROCEED_FILE).touch()

        await asyncio.gather(hc._wait_for_proceed_only(settings), touch_later())

    asyncio.run(driver())


def test_wait_loop_proceeds_when_reprobe_recovers(tmp_path: Path, monkeypatch):
    settings = hc.Settings(
        output_dir=tmp_path,
        workflow="author_critic_long",
        mode="strict",
        wait_interval_s=0,
        heartbeat_interval_s=999,
        skip_compute=True,
    )

    initial = [
        hc.ProbeResult(
            role=hc.ROLE_AUTHOR,
            model_ref="models/openai/gpt-55-pro",
            ok=True,
            duration_s=0.0,
            started_at="t0",
            finished_at="t0",
        ),
        hc.ProbeResult(
            role=f"{hc.ROLE_COUNCIL_PREFIX}opus_47_max",
            model_ref="models/anthropic/opus_47_max",
            ok=False,
            duration_s=0.0,
            started_at="t0",
            finished_at="t0",
            error_type="AuthenticationError",
            error_message="invalid x-api-key",
        ),
    ]

    fake_preset = _StubPreset(
        component_configs={
            "Author": {"model": "models/openai/gpt-55-pro"},
            "ACCritic": {"model": "models/openai/gpt-55-pro"},
        },
        inputs={
            "enable_council": True,
            "council_models": ["models/anthropic/opus_47_max"],
            "enable_compute": False,
        },
    )

    # Patch load_preset that wait_for_recovery_or_signal imports lazily.
    import proofstack.registry as registry_mod

    monkeypatch.setattr(registry_mod, "load_preset", lambda *_a, **_k: fake_preset)

    # Patch the probe runner so the re-probe "recovers".
    async def fake_run_all_probes(probes):
        return [
            hc.ProbeResult(
                role=p.role,
                model_ref=p.model_ref,
                ok=True,
                duration_s=0.0,
                started_at="t1",
                finished_at="t1",
            )
            for p in probes
        ]

    monkeypatch.setattr(hc, "run_all_probes", fake_run_all_probes)

    final = asyncio.run(
        hc.wait_for_recovery_or_signal(settings, initial, started_at="t0")
    )
    assert all(r.ok for r in final)
    # Report was written.
    report = json.loads((tmp_path / hc.REPORT_FILE).read_text(encoding="utf-8"))
    assert report["all_ok"] is True
    assert report["attempt"] >= 2


def test_settings_from_env_off_mode(monkeypatch):
    monkeypatch.setenv("FIRSTPROOF_HEALTHCHECK", "off")
    settings = hc._settings_from_env([])
    assert settings.mode == "off"


def test_settings_from_env_default_off(monkeypatch):
    """Default mode is ``off`` so the official First Proof harness does
    not accidentally halt on a probe failure (the halted container's
    output dir would be misinterpreted as a successful submission)."""
    monkeypatch.delenv("FIRSTPROOF_HEALTHCHECK", raising=False)
    settings = hc._settings_from_env([])
    assert settings.mode == "off"


def test_settings_from_env_unknown_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("FIRSTPROOF_HEALTHCHECK", "bogus")
    settings = hc._settings_from_env([])
    assert settings.mode == "off"


def test_settings_from_env_strict_opt_in(monkeypatch):
    monkeypatch.setenv("FIRSTPROOF_HEALTHCHECK", "strict")
    settings = hc._settings_from_env([])
    assert settings.mode == "strict"


def test_is_openai_model_ref():
    assert hc._is_openai_model_ref("models/openai/gpt-55-pro") is True
    assert hc._is_openai_model_ref("models/anthropic/opus_47_max") is False
    assert hc._is_openai_model_ref("models/gemini/gemini-31-pro") is False
    assert hc._is_openai_model_ref("") is False


def _good_compute_report(**overrides):
    base = {
        "sentinel": hc.COMPUTE_HEALTHCHECK_SENTINEL,
        "binaries": {"sage": "/usr/bin/sage", "gap": "", "singular": "", "gp": ""},
        "computation": {"engine": "sage", "expr": "1+1", "result": "2"},
    }
    for k, v in overrides.items():
        if isinstance(v, dict):
            base[k] = {**base.get(k, {}), **v}
        else:
            base[k] = v
    return json.dumps(base)


def test_validate_compute_report_json_happy_path():
    detail: dict = {}
    report = _good_compute_report()
    assert hc._validate_compute_report(report, detail) is None
    assert detail["report_parsed"]["binaries"]["sage"] == "/usr/bin/sage"


def test_validate_compute_report_missing_sentinel():
    detail: dict = {}
    report = json.dumps({
        "binaries": {"sage": "/usr/bin/sage"},
        "computation": {"engine": "sage", "expr": "1+1", "result": "2"},
    })
    err = hc._validate_compute_report(report, detail)
    assert err is not None and "sentinel" in err


def test_validate_compute_report_empty():
    assert hc._validate_compute_report("", {}) == "response file is empty"


def test_validate_compute_report_fenced_json():
    detail: dict = {}
    report = "```json\n" + _good_compute_report() + "\n```"
    assert hc._validate_compute_report(report, detail) is None


def test_validate_compute_report_rejects_plain_text():
    """The prompt mandates a JSON object. Plain text that happens to
    contain the sentinel + a CAS name is NOT enough — there's no proof
    the worker actually probed binaries or ran a computation."""
    err = hc._validate_compute_report(
        f"sage CAS_HEALTHCHECK_OK", {}
    )
    assert err is not None and "JSON" in err
    err = hc._validate_compute_report(
        f"sage at /usr/bin/sage. {hc.COMPUTE_HEALTHCHECK_SENTINEL}", {}
    )
    assert err is not None and "JSON" in err


def test_validate_compute_report_rejects_non_object_json():
    """A JSON array or scalar must not pass."""
    detail: dict = {}
    err = hc._validate_compute_report(json.dumps([1, 2, 3]), detail)
    assert err is not None and "JSON object" in err
    assert "report_raw_excerpt" in detail
    err = hc._validate_compute_report(json.dumps("just a string"), {})
    assert err is not None


def test_validate_compute_report_rejects_all_empty_binaries():
    """The whole point of the preflight is to catch a CAS stack that
    isn't actually installed. A report claiming every CAS path is empty
    must fail, not pass on structural validity."""
    report = _good_compute_report(
        binaries={"sage": "", "gap": "", "singular": "", "gp": ""}
    )
    err = hc._validate_compute_report(report, {})
    assert err is not None and "every entry in 'binaries' is" in err


def test_validate_compute_report_engine_must_be_non_empty_binary():
    """``computation.engine`` must name a binary whose path is non-empty,
    so the worker can't claim to have run sage while reporting sage=""."""
    report = _good_compute_report(
        binaries={"sage": "", "gap": "/usr/bin/gap", "singular": "", "gp": ""},
        computation={"engine": "sage", "expr": "1+1", "result": "2"},
    )
    err = hc._validate_compute_report(report, {})
    assert err is not None and "engine" in err


def test_validate_compute_report_rejects_unknown_expression():
    report = _good_compute_report(
        computation={"engine": "sage", "expr": "factor(42)", "result": "2 * 3 * 7"}
    )
    err = hc._validate_compute_report(report, {})
    assert err is not None and "allowed expressions" in err


def test_validate_compute_report_rejects_wrong_result():
    report = _good_compute_report(
        computation={"engine": "sage", "expr": "1+1", "result": "3"}
    )
    err = hc._validate_compute_report(report, {})
    assert err is not None and "does not match" in err


def test_validate_compute_report_accepts_each_allowed_expression():
    for expr, result in hc._COMPUTE_PROBE_ALLOWED_EXPRS.items():
        report = _good_compute_report(
            computation={"engine": "sage", "expr": expr, "result": result}
        )
        assert hc._validate_compute_report(report, {}) is None, expr


def test_run_probe_dispatch_to_author_path(monkeypatch):
    """The author probe must dispatch to ``probe_author_files_path`` when
    the configured model is OpenAI (so we exercise files.create + container
    download), and fall back to the sentinel probe otherwise."""
    called: dict[str, str] = {}

    async def fake_author(p):
        called["which"] = "author"
        return hc.ProbeResult(
            role=p.role, model_ref=p.model_ref, ok=True, duration_s=0,
            started_at="t", finished_at="t",
        )

    async def fake_sentinel(p):
        called["which"] = "sentinel"
        return hc.ProbeResult(
            role=p.role, model_ref=p.model_ref, ok=True, duration_s=0,
            started_at="t", finished_at="t",
        )

    monkeypatch.setattr(hc, "probe_author_files_path", fake_author)
    monkeypatch.setattr(hc, "probe_api_model", fake_sentinel)

    # OpenAI Author → files-path probe
    p_openai = hc.Probe(role=hc.ROLE_AUTHOR, model_ref="models/openai/gpt-55-pro", tools=[])
    asyncio.run(hc.run_probe(p_openai))
    assert called["which"] == "author"

    # Non-OpenAI Author (hypothetical preset edit) → sentinel fallback
    called.clear()
    p_anthropic = hc.Probe(role=hc.ROLE_AUTHOR, model_ref="models/anthropic/opus_47_max", tools=[])
    asyncio.run(hc.run_probe(p_anthropic))
    assert called["which"] == "sentinel"
