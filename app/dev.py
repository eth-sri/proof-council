"""Local developer dashboard for ProofCouncil.

Localhost-only Flask app for editing workflow-backed agents and browsing
run outputs.

Run::

    uv run python app/dev.py --port 5002
    uv run python app/dev.py --runs-root outputs --runs-root smoke-output
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
# Launching as a plain script (`uv run python app/dev.py …`) puts
# `app/` first on sys.path. Prepend the repo root so
# `from app.dev_data …` resolves the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_file, url_for
from mathagents.config_loader import load_solver_config
from proofstack.monitor import DEFAULT_MONITOR_MODEL, normalize_monitor_model_spec

from app.dev_data import (
    clear_stopped_marker,
    create_tool_definition,
    delete_preset,
    discover_agent_palette_items,
    discover_agents,
    discover_exported_presets,
    discover_model_options,
    discover_presets,
    discover_runs,
    discover_tool_definitions,
    find_agent,
    estimate_prunable_bytes,
    find_preset,
    find_run,
    find_runs_awaiting_human,
    run_process_alive,
    stop_run_process,
    write_stopped_marker,
    load_call_detail,
    load_event_tree,
    load_monitor_summaries,
    load_pending_human_tasks,
    load_execution_graph,
    mutate_preset_yaml,
    presets_registry_version,
    preset_file_version,
    preset_dag_report,
    prune_run_artifacts,
    render_recorded_messages,
    resolve_output_refs,
    safe_blob_path,
    save_preset_yaml,
    save_tool_definition,
    tool_definition_to_dict,
    validate_preset_yaml,
    workflow_input_from_tree,
    workflow_output_field_text,
    workflow_output_from_tree,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOTS = (REPO_ROOT / "outputs",)
# Normalized run statuses (see dev_data._normalize_run_status) that mean the run
# is over, so the human-task poller can stop.
_TERMINAL_RUN_STATUSES = {"finished", "error"}
PROBLEMS_ROOT = REPO_ROOT / "problems"
LOCAL_TIMEZONE = ZoneInfo(os.environ.get("PROOFSTACK_TIMEZONE") or "Europe/Zurich")
PROVIDER_API_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "deepseek_special": "DEEPSEEK_API_KEY",
    "glm": "GLM_API_KEY",
    "google": "GOOGLE_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "sri": "SRI_API_KEY",
    "stepfun": "STEPFUN_API_KEY",
    "tiiuae": "TIIUAE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "xai": "XAI_API_KEY",
}


def _relative_model_ref(model: Any) -> str:
    raw = str(model or "").strip()
    if raw.startswith("models/"):
        return raw.removeprefix("models/")
    return raw


def _monitor_model_options() -> list[str]:
    seen: set[str] = set()
    options: list[str] = []
    for ref in discover_model_options():
        rel = _relative_model_ref(ref)
        if rel and rel not in seen:
            seen.add(rel)
            options.append(rel)
    default = _relative_model_ref(DEFAULT_MONITOR_MODEL)
    if default and default not in seen:
        options.append(default)
    return sorted(options)


def _monitor_key_requirements() -> dict[str, list[dict[str, Any]]]:
    env = _dashboard_subprocess_env()
    return {
        model: _api_key_requirements_for_model(normalize_monitor_model_spec(model), env)
        for model in _monitor_model_options()
    }


def create_app(runs_roots: tuple[Path, ...] = DEFAULT_RUNS_ROOTS) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["RUNS_ROOTS"] = list(runs_roots)

    @app.context_processor
    def inject_human_waiting_runs():
        # Powers the global "waiting on you" nav indicator on every page.
        # Only runs blocked on human input are returned (cheap: non-terminal
        # runs only). Failures here must never break page rendering.
        try:
            waiting = find_runs_awaiting_human(app.config["RUNS_ROOTS"])
        except Exception:
            waiting = []
        return {"human_waiting_runs": waiting}

    @app.template_filter("display_scalar")
    def display_scalar(value):
        if value is True:
            return "true"
        if value is False:
            return "false"
        if value is None:
            return "none"
        return str(value)

    @app.template_filter("display_time")
    def display_time(value):
        if not value:
            return "—"
        text = str(value)
        try:
            is_utc = text.endswith("Z")
            parsed = datetime.fromisoformat(text[:-1] + "+00:00" if is_utc else text)
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(LOCAL_TIMEZONE)
            return parsed.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            cleaned = text.replace("T", " ").replace("Z", "").split(".", 1)[0]
            return re.sub(r"(\d{1,2}:\d{2}):\d{2}\b", r"\1", cleaned)

    @app.template_filter("display_duration")
    def display_duration(value):
        if value is None:
            return "—"
        try:
            total_seconds = max(0.0, float(value))
        except (TypeError, ValueError):
            return "—"
        total_minutes = int(round(total_seconds / 60.0))
        if total_seconds > 0 and total_minutes == 0:
            return "<1 min"
        hours, minutes = divmod(total_minutes, 60)
        if hours and minutes:
            return f"{hours} h {minutes} min"
        if hours:
            return f"{hours} h"
        return f"{total_minutes} min"

    @app.template_filter("display_tokens")
    def display_tokens(value):
        if value is None:
            return "—"
        try:
            n = int(value)
        except (TypeError, ValueError):
            return "—"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.0f}k"
        return str(n)

    @app.route("/")
    def index():
        return render_template("dev_home.html")

    @app.route("/catalog")
    def catalog():
        agents = discover_agents()
        return render_template("dev_agents.html", agents=agents)

    @app.route("/agent/<qualname>")
    def agent_detail(qualname: str):
        match = find_agent(qualname)
        if match is None:
            abort(404)
        return render_template("dev_agent_detail.html", agent=match)

    # --- UI-2: workflow presets ----------------------------------------------

    @app.route("/presets")
    def presets_index():
        presets = discover_presets()
        return render_template(
            "dev_presets.html",
            presets=presets,
            preset_signature=presets_registry_version(),
        )

    @app.route("/presets/data")
    def presets_data():
        signature = presets_registry_version()
        if request.args.get("signature") == signature:
            return jsonify({"ok": True, "changed": False, "signature": signature})
        presets = discover_presets()
        payload = {
            "ok": True,
            "changed": True,
            "signature": signature,
            "presets": [_preset_payload(p) for p in presets],
            "exported_presets": discover_exported_presets(),
        }
        if request.args.get("include_keys") == "1":
            runnable = [p for p in presets if not p.error]
            payload["key_requirements"] = {
                p.name: _api_key_requirements_for_preset(p.name)
                for p in runnable
            }
        return jsonify(payload)

    @app.route("/run-agent")
    def run_agent():
        presets = [p for p in discover_presets() if not p.error]
        return render_template(
            "dev_run_agent.html",
            presets=presets,
            preset_inputs={p.name: p.inputs for p in presets},
            preset_signature=presets_registry_version(),
            problems=_discover_problem_files(),
            monitor_model_options=_monitor_model_options(),
            monitor_key_requirements=_monitor_key_requirements(),
            key_requirements={
                p.name: _api_key_requirements_for_preset(p.name)
                for p in presets
            },
        )

    @app.route("/run-agent/start", methods=["POST"])
    def run_agent_start():
        payload = request.get_json(silent=True) or {}
        preset_name = str(payload.get("preset") or "").strip()
        preset = find_preset(preset_name)
        if preset is None:
            return jsonify({"ok": False, "errors": ["Choose an agent to run."]}), 400

        try:
            max_parallel = max(1, int(payload.get("max_parallel") or 1))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "errors": ["Max parallel runs must be a number."]}), 400

        monitor_enabled = bool(payload.get("monitor"))
        monitor_model = str(normalize_monitor_model_spec(payload.get("monitor_model") or DEFAULT_MONITOR_MODEL))

        try:
            problems = _selected_run_problems(payload)
        except ValueError as e:
            return jsonify({"ok": False, "errors": [str(e)]}), 400
        if not problems:
            return jsonify({"ok": False, "errors": ["Select or create at least one problem."]}), 400

        env = _dashboard_subprocess_env()
        for key, value in (payload.get("api_keys") or {}).items():
            clean_key = str(key or "").strip()
            clean_value = str(value or "").strip()
            if clean_key and clean_value:
                env[clean_key] = clean_value

        missing = [
            req["env"]
            for req in _api_key_requirements_for_preset(preset_name, env=env)
            if not req["present"]
        ]
        if monitor_enabled:
            missing.extend(
                req["env"]
                for req in _api_key_requirements_for_model(monitor_model, env)
                if not req["present"]
            )
            missing = sorted(set(missing))
        if missing:
            return jsonify({"ok": False, "errors": [f"Missing API keys: {', '.join(missing)}"]}), 400

        display_name = _run_display_name(preset.label, problems)
        outputs_root = REPO_ROOT / "outputs"
        run_id = _next_run_id(_slug(display_name).lower(), outputs_root)
        batch_dir = outputs_root / run_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        problems_file = batch_dir / "problems.json"
        problems_file.write_text(
            json.dumps({"problems": problems}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (batch_dir / "run-metadata.json").write_text(
            json.dumps(
                {
                    "status": "starting",
                    "display_name": display_name,
                    "started_by": "dashboard",
                    "preset": preset_name,
                    "monitor": {"enabled": monitor_enabled, "model": monitor_model if monitor_enabled else None},
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "manifest": {
                        "started_at": datetime.now().isoformat(timespec="seconds"),
                        "preset": preset_name,
                        "max_parallel": max_parallel,
                        "problems": {
                            problem["id"]: {
                                "status": "queued",
                                "problem_id": problem["id"],
                                "display_name": problem.get("display_name") or _human_label(problem["id"]),
                                "run_id": f"{run_id}-{problem['id']}",
                            }
                            for problem in problems
                        },
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log_path = batch_dir / "dashboard-subprocess.log"
        cmd = [
            sys.executable,
            "scripts/run_workflow_batch.py",
            "--workflow",
            preset_name,
            "--problems-file",
            str(problems_file),
            "--output",
            str(outputs_root),
            "--run-id",
            run_id,
            "--run-name",
            display_name,
            "--max-parallel",
            str(max_parallel),
        ]
        if monitor_enabled:
            cmd.extend(["--monitor", "--monitor-model", monitor_model])
        # Per-run workflow input overrides (e.g. claude_model=haiku). Only
        # non-empty values are forwarded; blanks fall back to the preset default.
        for key, value in (payload.get("inputs") or {}).items():
            clean_key = str(key or "").strip()
            clean_value = str("" if value is None else value).strip()
            if clean_key and clean_value:
                cmd.extend(["--input", f"{clean_key}={clean_value}"])
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return jsonify(
            {
                "ok": True,
                "run_id": run_id,
                "display_name": display_name,
                "url": url_for("run_detail", run_id=run_id),
            }
        )

    @app.route("/run-agent/problem", methods=["POST"])
    def run_agent_problem():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text") or "").strip()
        if not text:
            return jsonify({"ok": False, "errors": ["Problem statement is required."]}), 400
        problem = _save_problem_text(str(payload.get("id") or ""), text)
        return jsonify({"ok": True, "problem": problem})

    @app.route("/agents/new")
    def new_agent():
        label = _random_agent_label()
        name = _next_preset_name(_slug(label).lower())
        save_preset_yaml(name, _starter_agent_yaml(name, label))
        return redirect(url_for("preset_editor", name=name))

    @app.route("/preset/<name>/editor")
    def preset_editor(name: str):
        preset = find_preset(name)
        if preset is None:
            abort(404)
        report = preset_dag_report(name)
        return render_template(
            "dev_preset_editor.html",
            preset=preset,
            report=report.to_dict(),
            model_options=discover_model_options(),
            monitor_model_options=_monitor_model_options(),
            preset_registry_version=presets_registry_version(),
            agent_palette_items=discover_agent_palette_items(),
            exported_presets=discover_exported_presets(),
            tool_definitions=[
                tool_definition_to_dict(tool) for tool in discover_tool_definitions()
            ],
        )

    @app.route("/preset/<name>/editor/data")
    def preset_editor_data(name: str):
        try:
            file_version = preset_file_version(name)
        except FileNotFoundError:
            abort(404)
        registry_version = presets_registry_version()
        registry_changed = request.args.get("registry_version") != registry_version
        if request.args.get("file_version") == file_version and not registry_changed:
            return jsonify(
                {
                    "ok": True,
                    "changed": False,
                    "file_version": file_version,
                    "registry_version": registry_version,
                }
            )
        if request.args.get("file_version") == file_version and registry_changed:
            return jsonify(
                {
                    "ok": True,
                    "changed": False,
                    "registry_changed": True,
                    "file_version": file_version,
                    "registry_version": registry_version,
                    "exported_presets": discover_exported_presets(),
                }
            )
        preset = find_preset(name)
        if preset is None:
            abort(404)
        return jsonify(
            {
                "ok": True,
                "changed": True,
                "preset": {
                    "name": preset.name,
                    "label": preset.label,
                    "raw_yaml": preset.raw_yaml,
                    "file_version": file_version,
                    "component_configs": preset.component_configs,
                },
                "report": preset_dag_report(name).to_dict(),
                "model_options": discover_model_options(),
                "registry_version": registry_version,
                "exported_presets": discover_exported_presets(),
                "tool_definitions": [
                    tool_definition_to_dict(tool)
                    for tool in discover_tool_definitions()
                ],
            }
        )

    @app.route("/tools/new", methods=["POST"])
    def tool_create():
        try:
            tool = create_tool_definition()
        except Exception as e:
            return jsonify({"ok": False, "errors": [str(e)]}), 400
        return jsonify({"ok": True, "tool": tool_definition_to_dict(tool)})

    @app.route("/tools/<name>/save", methods=["POST"])
    def tool_save(name: str):
        payload = request.get_json(silent=True) or {}
        try:
            tool = save_tool_definition(
                name,
                str(payload.get("name") or name),
                str(payload.get("yaml") or ""),
                str(payload.get("python") or ""),
            )
        except Exception as e:
            return jsonify({"ok": False, "errors": [str(e)]}), 400
        return jsonify({"ok": True, "tool": tool_definition_to_dict(tool)})

    @app.route("/preset/<name>/editor/validate", methods=["POST"])
    def preset_editor_validate(name: str):
        payload = request.get_json(silent=True) or {}
        raw_yaml = str(payload.get("raw_yaml", ""))
        return jsonify(validate_preset_yaml(raw_yaml))

    @app.route("/preset/<name>/editor/mutate", methods=["POST"])
    def preset_editor_mutate(name: str):
        if find_preset(name) is None:
            abort(404)
        payload = request.get_json(silent=True) or {}
        raw_yaml = str(payload.get("raw_yaml", ""))
        operation = payload.get("operation") or {}
        return jsonify(mutate_preset_yaml(raw_yaml, operation))

    @app.route("/preset/<name>/editor/save", methods=["POST"])
    def preset_editor_save(name: str):
        payload = request.get_json(silent=True) or {}
        raw_yaml = str(payload.get("raw_yaml", ""))
        base_file_version = str(payload.get("base_file_version") or "")
        preset = find_preset(name)
        if preset is None:
            abort(404)
        if (
            base_file_version
            and preset.file_version != base_file_version
            and preset.raw_yaml != raw_yaml
        ):
            return jsonify(
                {
                    "ok": False,
                    "conflict": True,
                    "errors": ["YAML changed on disk; reloaded the latest version."],
                    "preset": {
                        "name": preset.name,
                        "label": preset.label,
                        "raw_yaml": preset.raw_yaml,
                        "file_version": preset.file_version,
                        "component_configs": preset.component_configs,
                    },
                    "report": preset_dag_report(name).to_dict(),
                    "model_options": discover_model_options(),
                    "exported_presets": discover_exported_presets(),
                    "tool_definitions": [
                        tool_definition_to_dict(tool)
                        for tool in discover_tool_definitions()
                    ],
                }
            ), 409
        try:
            save_preset_yaml(name, raw_yaml)
        except Exception as e:
            return jsonify({"ok": False, "errors": [str(e)]}), 400
        preset = find_preset(name)
        file_version = preset.file_version if preset else preset_file_version(name)
        return jsonify(
            {
                "ok": True,
                "file_version": file_version,
                "preset": {
                    "name": name,
                    "label": preset.label if preset else name.replace("_", " ").title(),
                    "raw_yaml": raw_yaml,
                    "file_version": file_version,
                    "component_configs": preset.component_configs if preset else {},
                },
                "report": validate_preset_yaml(raw_yaml),
            }
        )

    @app.route("/preset/<name>/delete", methods=["POST"])
    def preset_delete(name: str):
        try:
            delete_preset(name)
        except Exception as e:
            return jsonify({"ok": False, "errors": [str(e)]}), 400
        return jsonify({"ok": True, "url": url_for("presets_index")})

    @app.route("/preset/<name>/editor/run-sample", methods=["POST"])
    def preset_editor_run_sample(name: str):
        preset = find_preset(name)
        if preset is None:
            abort(404)
        payload = request.get_json(silent=True) or {}
        problem = str(
            payload.get("problem")
            or "Prove that the square root of 2 is irrational."
        ).strip()
        problem_id = _slug(str(payload.get("problem_id") or "editor_sample"))
        monitor_enabled = bool(payload.get("monitor"))
        monitor_model = str(normalize_monitor_model_spec(payload.get("monitor_model") or DEFAULT_MONITOR_MODEL))
        display_name = _run_display_name(preset.label, [{"id": problem_id}])
        outputs_root = REPO_ROOT / "outputs"
        run_id = _next_run_id(_slug(display_name).lower(), outputs_root)
        outputs_root.mkdir(parents=True, exist_ok=True)
        run_dir = outputs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run-metadata.json").write_text(
            json.dumps(
                {
                    "status": "starting",
                    "display_name": display_name,
                    "started_by": "dashboard",
                    "preset": name,
                    "monitor": {"enabled": monitor_enabled, "model": monitor_model if monitor_enabled else None},
                    "problem_id": problem_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log_path = run_dir / "dashboard-subprocess.log"
        env = _dashboard_subprocess_env()
        cmd = [
            sys.executable,
            "scripts/run_workflow.py",
            "--workflow",
            name,
            "--problem-text",
            problem,
            "--problem-id",
            problem_id,
            "--run-id",
            run_id,
            "--run-name",
            display_name,
            "--output",
            str(outputs_root),
        ]
        if monitor_enabled:
            cmd.extend(["--monitor", "--monitor-model", monitor_model])
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return jsonify(
            {
                "ok": True,
                "run_id": run_id,
                "display_name": display_name,
                "url": url_for("run_detail", run_id=run_id),
                "log": str(log_path),
            }
        )

    # --- UI-0: run viewer -----------------------------------------------------

    @app.route("/runs")
    def runs_index():
        runs = discover_runs(app.config["RUNS_ROOTS"])
        return render_template("dev_runs.html", runs=runs)

    @app.route("/run/<run_id>")
    def run_detail(run_id: str):
        run = find_run(app.config["RUNS_ROOTS"], run_id)
        if run is None:
            abort(404)
        tree = load_event_tree(run.path)
        exec_graph = load_execution_graph(run.path, tree=tree)
        workflow_input = resolve_output_refs(run.path, workflow_input_from_tree(tree))
        workflow_output = resolve_output_refs(run.path, workflow_output_from_tree(tree))
        monitor_summaries = load_monitor_summaries(run.path, graph=exec_graph)
        show_monitor = bool(run.monitor_enabled or monitor_summaries)
        show_execution_graph = bool(run.has_events or (run.status == "running" and not run.problems))
        return render_template(
            "dev_run_detail.html",
            run=run,
            tree=tree,
            exec_graph=exec_graph,
            workflow_input=workflow_input,
            workflow_output=workflow_output,
            monitor_summaries=monitor_summaries,
            show_monitor=show_monitor,
            show_execution_graph=show_execution_graph,
            human_tasks=load_pending_human_tasks(run.path),
            run_finished=(run.status or "running") in _TERMINAL_RUN_STATUSES,
            run_running=(run.status or "running") == "running",
            run_alive=run_process_alive(run.path),
            can_resume=(run.path / "resume.json").exists(),
            prunable_bytes=estimate_prunable_bytes(run.path),
            pruned_kb=request.args.get("pruned_kb", type=int),
        )

    @app.route("/run/<run_id>/human", methods=["POST"])
    def run_human_submit(run_id: str):
        run = find_run(app.config["RUNS_ROOTS"], run_id)
        if run is None:
            abort(404)
        filename = str(request.form.get("response_filename") or "")
        # Only allow writing a *.response.json directly inside this run's inbox.
        if "/" in filename or "\\" in filename or not filename.endswith(".response.json"):
            abort(400, description="invalid response filename")
        inbox = (run.path / "human_inbox").resolve()
        target = (inbox / filename).resolve()
        if target.parent != inbox:
            abort(400, description="response path escapes inbox")
        values: dict[str, Any] = {}
        for key, value in request.form.items():
            if key.startswith("f_"):
                values[key[2:]] = value
        values.setdefault("status", "done")
        inbox.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.route("/run/<run_id>/resume", methods=["POST"])
    def run_resume(run_id: str):
        # Relaunch a stopped/crashed run in place: same run_id, --resume-from
        # itself, so completed nodes (incl. answered human prompts) replay from
        # the resume_cache and the run continues from the first unfinished node.
        run = find_run(app.config["RUNS_ROOTS"], run_id)
        if run is None:
            abort(404)
        spec_path = run.path / "resume.json"
        if not spec_path.exists():
            abort(400, description="no resume spec recorded for this run")
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            argv = spec.get("argv") or []
        except (json.JSONDecodeError, OSError):
            abort(400, description="invalid resume spec")
        if not argv:
            abort(400, description="empty resume spec")
        cmd = [sys.executable, *[str(a) for a in argv], "--resume-from", run_id]
        # Drop any "stopped" marker so the relaunched run reads as running again,
        # not as the stopped run it was a moment ago.
        clear_stopped_marker(run.path)
        env = _dashboard_subprocess_env()
        log_path = run.path / "dashboard-resume.log"
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return redirect(url_for("run_detail", run_id=run_id))

    @app.route("/run/<run_id>/stop", methods=["POST"])
    def run_stop(run_id: str):
        # Stop a live run: terminate its process group (worker + sandbox CLIs)
        # and mark it "stopped" (non-terminal) so the Resume button shows. We do
        # NOT freeze it (SIGSTOP) — a clean kill + durable resume is more robust
        # than a frozen process the OS might reap. Resume replays finished nodes
        # from the cache and continues from where it stopped.
        run = find_run(app.config["RUNS_ROOTS"], run_id)
        if run is None:
            abort(404)
        result = stop_run_process(run.path)
        write_stopped_marker(run.path, signalled=result["signalled"])
        return redirect(url_for("run_detail", run_id=run_id))

    @app.route("/run/<run_id>/prune", methods=["POST"])
    def run_prune(run_id: str):
        # Reclaim disk: delete finished nodes' throwaway sandboxes + CLI logs.
        # Safe — the dashboard reads input/output/messages.json (kept) and
        # resume replays from resume_cache/ (kept). Proof text lives in
        # output.json, so nothing visible or replayable is lost.
        run = find_run(app.config["RUNS_ROOTS"], run_id)
        if run is None:
            abort(404)
        result = prune_run_artifacts(run.path)
        pruned_kb = result["bytes_freed"] // 1024
        return redirect(url_for("run_detail", run_id=run_id, pruned_kb=pruned_kb))

    @app.route("/run/<run_id>/human-pending")
    def run_human_pending(run_id: str):
        # Cheap poll target so the run page can notice a NEW human ask appear
        # (e.g. the second time a loop reaches a human node) and reload itself.
        run = find_run(app.config["RUNS_ROOTS"], run_id)
        if run is None:
            abort(404)
        tasks = load_pending_human_tasks(run.path)
        finished = (run.status or "running") in _TERMINAL_RUN_STATUSES
        return jsonify(
            {
                "pending": sorted(str(t["response_filename"]) for t in tasks),
                "finished": finished,
                # The worker's live state, so the page can swap the Stop/Resume
                # button when a resume's process comes up (or a stop's goes down).
                "alive": run_process_alive(run.path),
            }
        )

    @app.route("/run/<run_id>/graph-fragment")
    def run_graph_fragment(run_id: str):
        run = find_run(app.config["RUNS_ROOTS"], run_id)
        if run is None:
            abort(404)
        tree = load_event_tree(run.path)
        exec_graph = load_execution_graph(run.path, tree=tree)
        return render_template("dev_run_graph.html", run=run, tree=tree, exec_graph=exec_graph)

    @app.route("/run/<run_id>/monitor-fragment")
    def run_monitor_fragment(run_id: str):
        run = find_run(app.config["RUNS_ROOTS"], run_id)
        if run is None:
            abort(404)
        tree = load_event_tree(run.path)
        exec_graph = load_execution_graph(run.path, tree=tree)
        return render_template(
            "dev_run_monitor.html",
            run=run,
            monitor_summaries=load_monitor_summaries(run.path, graph=exec_graph),
        )

    @app.route("/run/<run_id>/call/<call_ref>")
    def call_detail(run_id: str, call_ref: str):
        run = find_run(app.config["RUNS_ROOTS"], run_id)
        if run is None:
            abort(404)
        tree = load_event_tree(run.path)
        load_execution_graph(run.path, tree=tree)
        node = tree.by_ref.get(call_ref)
        if node is None:
            abort(404)
        detail = load_call_detail(run.path, node)
        rendered = render_recorded_messages(detail.messages_json)
        input_payload = detail.input_json if detail.input_json is not None else node.input
        input_problem = _extract_problem_text(input_payload)
        parent_ref = ""
        if node.parent_call_id and node.parent_call_id in tree.by_id:
            parent_ref = tree.by_id[node.parent_call_id].display_ref
        return render_template(
            "dev_call_detail.html",
            run=run,
            tree=tree,
            node=node,
            detail=detail,
            rendered=rendered,
            input_problem=input_problem,
            input_fields=_input_without_rendered_problem(input_payload, input_problem),
            parent_ref=parent_ref,
        )

    @app.route("/run/<run_id>/blob")
    def run_blob(run_id: str):
        run = find_run(app.config["RUNS_ROOTS"], run_id)
        if run is None:
            abort(404)
        ref = request.args.get("ref", "")
        try:
            path = safe_blob_path(run.path, ref)
        except ValueError as e:
            abort(400, description=str(e))
        return send_file(path, mimetype="text/plain")

    @app.route("/run/<run_id>/output/<field>/download")
    def run_output_download(run_id: str, field: str):
        run = find_run(app.config["RUNS_ROOTS"], run_id)
        if run is None:
            abort(404)
        if not re.fullmatch(r"[A-Za-z0-9_]+", field):
            abort(400, description="invalid field")
        text = workflow_output_field_text(run.path, load_event_tree(run.path), field)
        if text is None:
            abort(404)
        return Response(
            _wrap_latex_body(text),
            mimetype="text/x-tex",
            headers={"Content-Disposition": f'attachment; filename="{field}.tex"'},
        )

    @app.route("/run/<run_id>/output/<field>/pdf")
    def run_output_pdf(run_id: str, field: str):
        run = find_run(app.config["RUNS_ROOTS"], run_id)
        if run is None:
            abort(404)
        if not re.fullmatch(r"[A-Za-z0-9_]+", field):
            abort(400, description="invalid field")
        text = workflow_output_field_text(run.path, load_event_tree(run.path), field)
        if text is None:
            abort(404)
        pdf, log = _compile_latex_pdf(_wrap_latex_body(text), field)
        if pdf is None:
            return Response("LaTeX compile failed:\n\n" + log, status=422, mimetype="text/plain")
        return Response(
            pdf,
            mimetype="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{field}.pdf"'},
        )

    def _human_task_proof(run_id: str) -> tuple[str, str]:
        # The current proof draft shown to the human, plus a filename stem.
        run = find_run(app.config["RUNS_ROOTS"], run_id)
        if run is None:
            abort(404)
        wanted = request.args.get("task", "")
        tasks = load_pending_human_tasks(run.path)
        if wanted:
            # A stale/invalid task id must 404, not silently serve another
            # pending task's proof under the requested filename.
            task = next((t for t in tasks if t.get("response_filename") == wanted), None)
            if task is None:
                abort(404, description="unknown or already-answered human task")
        else:
            task = tasks[0] if tasks else None
        proof = str(((task or {}).get("inputs") or {}).get("proof") or "")
        if not proof.strip():
            abort(404, description="no proof draft on this task yet")
        stem = re.sub(r"[^A-Za-z0-9_-]", "", Path(wanted).name.replace(".response.json", "")) or "proof"
        return proof, stem

    @app.route("/run/<run_id>/human-proof.tex")
    def run_human_proof_tex(run_id: str):
        # Download the current proof draft as a compilable .tex (body-only drafts
        # get the standard preamble; full documents pass through).
        proof, stem = _human_task_proof(run_id)
        return Response(
            _wrap_latex_body(proof),
            mimetype="text/x-tex",
            headers={"Content-Disposition": f'attachment; filename="{stem}.tex"'},
        )

    @app.route("/run/<run_id>/human-proof.pdf")
    def run_human_proof_pdf(run_id: str):
        # Compile the current proof draft and show the PDF inline.
        proof, stem = _human_task_proof(run_id)
        pdf, log = _compile_latex_pdf(_wrap_latex_body(proof), stem)
        if pdf is None:
            return Response("LaTeX compile failed:\n\n" + log, status=422, mimetype="text/plain")
        return Response(
            pdf,
            mimetype="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{stem}.pdf"'},
        )

    return app


# pdflatex/latexmk are installed under /Library/TeX/texbin on macOS, which a GUI-
# launched dashboard may not have on PATH; add it so the compiler is found.
_TEX_BIN = "/Library/TeX/texbin"

# Standard preamble wrapped around a proof BODY on download so it compiles, and
# so working documents can be kept body-only (no per-round preamble noise).
_STANDARD_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm,amsfonts,mathtools}
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{example}[theorem]{Example}"""

# Visible flag for an unresolved point in a reconciled proof. Core LaTeX only
# (fbox/parbox), so it works under any preamble; \providecommand so a document
# that defines its own \dispute wins.
_DISPUTE_MACRO = (
    r"\providecommand{\dispute}[1]{\par\medskip\noindent"
    r"\fbox{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}"
    r"{\textbf{>>> DISPUTE:} #1}}\par\medskip}"
)

_DISPUTE_COMMENT_RE = re.compile(r"^\s*%+\s*>{2,3}\s*DISPUTE:\s*(.*)$", re.MULTILINE)


def _surface_dispute_markers(text: str) -> str:
    """Turn comment-style dispute markers into visible \\dispute boxes.

    The reconciler is prompted to emit \\dispute{...} directly, but older docs
    (and model slip-ups) carry '% >>> DISPUTE: ...' comment lines, which vanish
    in a compiled PDF — the opposite of their purpose. Rewrite them and make
    sure the macro exists.
    """
    replaced = _DISPUTE_COMMENT_RE.sub(lambda m: r"\dispute{" + m.group(1).strip() + "}", text)
    already_defined = re.search(
        r"\\(?:providecommand|newcommand|renewcommand|def)\s*\{?\\dispute", replaced
    )
    if "\\dispute" in replaced and not already_defined:
        begin = replaced.find("\\begin{document}")
        if begin >= 0:
            replaced = replaced[:begin] + _DISPUTE_MACRO + "\n" + replaced[begin:]
    return replaced


def _wrap_latex_body(text: str) -> str:
    """Return a compilable LaTeX document. If the text is already a full document
    (has \\documentclass) it is returned unchanged apart from dispute surfacing;
    otherwise it is treated as a body and wrapped in the standard preamble."""
    if "\\documentclass" in text:
        return _surface_dispute_markers(text)
    body = _surface_dispute_markers(text.strip())
    return (
        _STANDARD_PREAMBLE
        + "\n"
        + _DISPUTE_MACRO
        + "\n\\begin{document}\n\n"
        + body
        + "\n\n\\end{document}\n"
    )


def _compile_latex_pdf(
    tex_source: str, name: str, *, extra_files: dict[str, str] | None = None
) -> tuple[bytes | None, str]:
    """Compile a LaTeX document to PDF in a throwaway dir. Returns (pdf_bytes,
    log); pdf_bytes is None on failure with the tail of the compiler log.
    extra_files are written alongside name.tex (e.g. a sidecar text file)."""
    search_path = os.environ.get("PATH", "") + os.pathsep + _TEX_BIN
    latexmk = shutil.which("latexmk", path=search_path)
    pdflatex = shutil.which("pdflatex", path=search_path)
    if latexmk:
        cmd = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"]
    elif pdflatex:
        cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"]
    else:
        return None, "No LaTeX compiler (latexmk or pdflatex) found."
    env = dict(os.environ)
    env["PATH"] = search_path
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / f"{name}.tex").write_text(tex_source, encoding="utf-8")
        for fname, content in (extra_files or {}).items():
            (tmp_path / fname).write_text(content, encoding="utf-8")
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp,
                capture_output=True,
                # Decode the compiler log tolerantly: pdflatex wraps lines at ~79
                # chars and can split a multi-byte UTF-8 char (e.g. an em dash),
                # which strict decoding would crash on.
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                env=env,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            return None, f"compile error: {type(e).__name__}: {e}"
        pdf_path = tmp_path / f"{name}.pdf"
        if pdf_path.exists():
            return pdf_path.read_bytes(), proc.stdout[-2000:]
        return None, (proc.stdout + "\n" + proc.stderr)[-4000:]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ProofCouncil local dashboard",
    )
    p.add_argument("--port", type=int, default=5002)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--runs-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "Directory holding run dirs (or a single run dir itself). "
            "Repeatable. Defaults to ./outputs/."
        ),
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _dashboard_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(_read_dotenv(REPO_ROOT / ".env"))
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    return env


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def _discover_problem_files() -> list[dict[str, str]]:
    if not PROBLEMS_ROOT.exists():
        return []
    problems: list[dict[str, str]] = []
    for path in sorted(PROBLEMS_ROOT.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        preview = " ".join(text.split())
        problem_id = path.stem
        problems.append(
            {
                "id": path.stem if path.suffix == ".txt" else path.name,
                "title": problem_id.replace("_", " "),
                "preview": preview[:180],
            }
        )
    return problems


def _problem_file_for_id(file_id: str) -> Path | None:
    candidates = [PROBLEMS_ROOT / file_id]
    if Path(file_id).suffix == "":
        candidates.append(PROBLEMS_ROOT / f"{file_id}.txt")
    try:
        root = PROBLEMS_ROOT.resolve()
    except OSError:
        return None
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved.parent == root and resolved.is_file():
            return resolved
    return None


def _selected_run_problems(payload: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_id in payload.get("problems") or []:
        file_id = str(raw_id or "").strip()
        if not file_id or Path(file_id).name != file_id or "/" in file_id or "\\" in file_id:
            continue
        path = _problem_file_for_id(file_id)
        if path is None:
            raise ValueError(f"Problem not found: {file_id}")
        problem_id = _slug(path.stem)
        if problem_id in seen:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            out.append({"id": problem_id, "text": text, "latex": text, "display_name": _human_label(problem_id)})
            seen.add(problem_id)
    return out


def _extract_problem_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("problem", "problem_statement", "problem_text", "statement"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
        for key in ("input", "inputs", "payload"):
            nested = _extract_problem_text(value.get(key))
            if nested:
                return nested
    if isinstance(value, list):
        for item in value:
            nested = _extract_problem_text(item)
            if nested:
                return nested
    return ""


def _input_without_rendered_problem(value: Any, rendered_problem: str) -> Any:
    if value is None:
        return None
    problem_keys = {"problem", "problem_statement", "problem_text", "statement"}

    def clean(item: Any) -> Any:
        if item is None:
            return None
        if isinstance(item, dict):
            out: dict[str, Any] = {}
            for key, nested in item.items():
                if (
                    rendered_problem
                    and str(key) in problem_keys
                    and isinstance(nested, str)
                    and nested.strip() == rendered_problem
                ):
                    continue
                cleaned = clean(nested)
                if cleaned is not None:
                    out[key] = cleaned
            return out or None
        if isinstance(item, list):
            cleaned_items = [clean(nested) for nested in item]
            return [nested for nested in cleaned_items if nested is not None]
        return item

    return clean(value)


def _save_problem_text(requested_id: str, text: str) -> dict[str, str]:
    problem_id = _next_problem_id(_slug(requested_id or text[:40] or "problem"))
    PROBLEMS_ROOT.mkdir(parents=True, exist_ok=True)
    (PROBLEMS_ROOT / f"{problem_id}.txt").write_text(text.strip() + "\n", encoding="utf-8")
    preview = " ".join(text.split())
    return {
        "id": problem_id,
        "title": problem_id.replace("_", " "),
        "preview": preview[:180],
    }


def _next_problem_id(base: str) -> str:
    base = _slug(base)
    if not (PROBLEMS_ROOT / f"{base}.txt").exists():
        return base
    idx = 2
    while (PROBLEMS_ROOT / f"{base}_{idx}.txt").exists():
        idx += 1
    return f"{base}_{idx}"


def _preset_payload(preset) -> dict[str, Any]:
    return {
        "name": preset.name,
        "label": preset.label,
        "description": preset.description,
        "inputs": preset.inputs,
        "budget": preset.budget,
        "model_overrides": preset.model_overrides,
        "error": preset.error,
        "edit_url": url_for("preset_editor", name=preset.name),
        "delete_url": url_for("preset_delete", name=preset.name),
    }


def _api_key_requirements_for_preset(name: str, env: dict[str, str] | None = None) -> list[dict[str, Any]]:
    preset = find_preset(name)
    if preset is None:
        return []
    env = env or _dashboard_subprocess_env()
    try:
        raw = yaml.safe_load(preset.raw_yaml) or {}
    except yaml.YAMLError:
        return []
    specs = _model_specs_from_value(raw)
    by_env: dict[str, dict[str, Any]] = {}
    for spec in specs:
        for requirement in _api_key_requirements_for_model(spec, env):
            item = by_env.setdefault(
                requirement["env"],
                {
                    "env": requirement["env"],
                    "provider": requirement["provider"],
                    "label": requirement["label"],
                    "models": [],
                    "present": bool(env.get(requirement["env"])),
                },
            )
            for model in requirement["models"]:
                if model not in item["models"]:
                    item["models"].append(model)
    return sorted(by_env.values(), key=lambda item: item["env"])


def _model_specs_from_value(value: Any) -> list[Any]:
    specs: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_model_spec_key(key):
                if isinstance(item, list):
                    specs.extend(item)
                else:
                    specs.append(item)
            elif key == "model_overrides" and isinstance(item, dict):
                specs.extend(item.values())
            else:
                specs.extend(_model_specs_from_value(item))
    elif isinstance(value, list):
        for item in value:
            specs.extend(_model_specs_from_value(item))
    return specs


def _is_model_spec_key(key: Any) -> bool:
    normalized = str(key)
    return normalized in {"model", "model_config"} or normalized.endswith(
        ("_model", "_models")
    )


def _api_key_requirements_for_model(spec: Any, env: dict[str, str]) -> list[dict[str, Any]]:
    try:
        cfg = load_solver_config(spec)
    except Exception:
        return []
    if cfg.get("type") == "agent" and isinstance(cfg.get("model_config"), dict):
        return _api_key_requirements_for_model(cfg["model_config"], env)
    api = str(cfg.get("api") or "openai")
    key_env = str(cfg.get("api_key_env") or PROVIDER_API_KEYS.get(api) or "")
    if not key_env or api in {"custom", "vllm"}:
        return []
    model = str(cfg.get("model") or spec)
    return [
        {
            "env": key_env,
            "provider": api,
            "label": api.replace("_", " ").title(),
            "models": [model],
            "present": bool(env.get(key_env)),
        }
    ]


def _slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    safe = safe.strip("_")
    return safe or "editor_sample"


def _human_label(value: str) -> str:
    text = re.sub(r"[_-]+", " ", str(value or "").strip()).strip()
    return text.title() if text else "Problem"


def _run_display_name(agent_label: str, problems: list[dict[str, Any]]) -> str:
    label = str(agent_label or "Agent").strip() or "Agent"
    if len(problems) == 1:
        problem = problems[0]
        problem_name = str(problem.get("display_name") or _human_label(str(problem.get("id") or "problem")))
        return f"{label} · {problem_name}"
    return f"{label} · {len(problems)} problems"


def _next_preset_name(base: str) -> str:
    existing = {p.name for p in discover_presets()}
    if base not in existing:
        return base
    idx = 2
    while f"{base}_{idx}" in existing:
        idx += 1
    return f"{base}_{idx}"


def _next_run_id(base: str, outputs_root: Path) -> str:
    safe = _slug(base).lower()
    if not (outputs_root / safe).exists():
        return safe
    idx = 2
    while (outputs_root / f"{safe}-{idx}").exists():
        idx += 1
    return f"{safe}-{idx}"


def _random_agent_label() -> str:
    adjectives = [
        "Brisk",
        "Cheeky",
        "Clever",
        "Cosmic",
        "Dapper",
        "Dizzy",
        "Jaunty",
        "Nimble",
        "Plucky",
        "Zesty",
    ]
    nouns = [
        "Axiom",
        "Lemma",
        "Proof",
        "Quibble",
        "Riddle",
        "Scheme",
        "Spark",
        "Theorem",
        "Twist",
        "Zigzag",
    ]
    return f"{random.choice(adjectives)} {random.choice(nouns)}"


def _starter_agent_yaml(name: str, label: str | None = None) -> str:
    label = label or name.replace("_", " ").title()
    return f"""workflow: proofstack.agents.dag_workflow.DAGWorkflow
description: >
  Draft proof agent. Edit the prompt, graph, and settings in the visual editor.

export:
  visible_as_node: true
  label: {label}
  description: Draft proof agent.

inputs:
  problem: ""

budget:
  max_usd: 2.0
  max_wallclock_s: 600

components:
  cfg_solver:
    model: models/openai/gpt-54-mini
    system_prompt: |
      You are an expert research mathematician. Produce a clear, rigorous proof attempt.
      Return only the proof inside <solution>...</solution>.
    user_prompt: |
      Problem:
      {{problem}}

      Write a complete proof inside <solution>...</solution>.
    input_schema:
      problem: string
    output:
      xml_tags: [solution]
      default_field: solution

dag:
  ui:
    workflow_output:
      x: 520
      y: 90
  nodes:
    - id: solver
      kind: agent
      agent: proofstack.agents.configurable_prompt.ConfigurablePromptAgent
      name: cfg_solver
      inputs:
        problem: $input.problem
      best_tex: $output.solution
      ui:
        x: 80
        y: 90
        label: Draft solver

  outputs:
    solution: $node.solver.solution
"""


if __name__ == "__main__":
    args = _parse_args()
    roots = tuple(args.runs_root) if args.runs_root else DEFAULT_RUNS_ROOTS
    app = create_app(runs_roots=roots)
    app.run(host=args.host, port=args.port, debug=args.debug)
