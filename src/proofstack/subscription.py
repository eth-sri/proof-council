"""Subscription-only routing; never constructs a provider API client."""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import replace
from pathlib import Path

from proofstack.atomic import write_text_atomic
from proofstack.budget import BudgetSpec
from proofstack.codex_auth import classify_codex_auth


CODEX_MODEL = "gpt-6-astra"
CLAUDE_MODEL = "claude-fable-5-1"
CODEX_COST = "models/openai/gpt-6-astra"
CLAUDE_COST = "models/anthropic/fable_51"
COUNCIL_MODELS = [CODEX_COST, CLAUDE_COST]


class SubscriptionBlocked(RuntimeError):
    requires_reconciliation = True


class SubscriptionCallFailed(RuntimeError):
    """The worker is stopped and accounted, but did not deliver a usable report."""


_active_receipts: set[Path] = set()


def _read_receipt(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("not an object")
        return value
    except (OSError, ValueError) as exc:
        raise SubscriptionBlocked(f"Invalid subscription receipt: {path.name}") from exc


def _completed(saved):
    output = saved.get("output")
    return (saved.get("status") == "completed" and isinstance(output, dict)
            and output.get("status") == "done" and isinstance(output.get("text"), str)
            and bool(output["text"].strip()))


def _settlement_path(receipt):
    return receipt.parent / "settlements" / receipt.name


def _settled(receipt):
    path = _settlement_path(receipt)
    return path.exists() and _read_receipt(path).get("settled") is True


def recover_packet(root, receipt_id):
    """Return settlement and any completed raw output without starting a worker."""
    if not isinstance(receipt_id, str) or not re.fullmatch(r"[0-9a-f]{64}", receipt_id):
        raise SubscriptionBlocked("Invalid subscription receipt identifier")
    receipt = root / "subscription_calls" / f"{receipt_id}.json"
    saved = reconcile_receipt(receipt)
    if _completed(saved):
        return True, saved["output"]
    return saved.get("status") == "failed" and _settled(receipt), None


def reconcile_receipt(receipt):
    saved = _read_receipt(receipt)
    if _completed(saved):
        return saved
    if saved.get("status") in {"started", "failed"} and _settled(receipt):
        settlement = _read_receipt(_settlement_path(receipt))
        folder = (receipt.parent.parent / settlement["workdir"]).resolve()
        if not folder.is_relative_to((receipt.parent.parent / "agents").resolve()):
            raise SubscriptionBlocked("Invalid subscription worker artifact path")
        output_path = folder / "output.json"
        recovered = {**saved, "status": "completed", "output": _read_receipt(output_path)} if output_path.exists() else {}
        saved = recovered if _completed(recovered) else {**saved, "status": "failed"}
        write_text_atomic(receipt, json.dumps(saved))
    return saved


def reconcile_stopped_call(root, receipt_id, *, unreported_allowance_usd, workers_stopped=False):
    """Explicit operator recovery; never starts workers or changes a run limit."""
    if not workers_stopped:
        raise SubscriptionBlocked("Confirm all workers and descendants are stopped before reconciliation")
    if not re.fullmatch(r"[0-9a-f]{64}", receipt_id):
        raise ValueError("Expected the 64-character subscription receipt identifier")
    if (type(unreported_allowance_usd) not in (int, float)
            or not math.isfinite(unreported_allowance_usd) or unreported_allowance_usd < 0):
        raise ValueError("Unreported allowance must be a finite nonnegative planning charge")
    root = Path(root).resolve()
    receipt = root / "subscription_calls" / f"{receipt_id}.json"
    with receipt.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SubscriptionBlocked("Subscription controller is still running") from exc
        if receipt.resolve() in _active_receipts:
            raise SubscriptionBlocked("Subscription controller is still running")
        saved = _read_receipt(receipt)
        if _completed(saved) or (saved.get("status") == "failed" and _settled(receipt)):
            return saved
        if saved.get("status") != "started":
            raise SubscriptionBlocked("Only an interrupted invocation can be reconciled manually")
        attempt = saved.get("attempts", 1)
        if type(attempt) is not int or attempt < 1:
            raise SubscriptionBlocked("Invalid subscription attempt number")
        # Debit each attempt once, even across a crash before guard release.
        # Keep the first attempt's legacy id to avoid charging it again.
        usage_id = "reconciliation-" + receipt_id
        if attempt > 1:
            usage_id += f"-attempt-{attempt}"
        usage = root / "subscription_usage" / f"{usage_id}.json"
        usage.parent.mkdir(exist_ok=True)
        if usage.exists():
            if _read_receipt(usage)["payload"]["api_equivalent_usd"] != unreported_allowance_usd:
                raise SubscriptionBlocked("Saved reconciliation allowance cannot change")
        else:
            write_text_atomic(usage, json.dumps({
                "kind": "model.call", "subscription_usage_id": usage_id,
                "payload": {"subscription": True, "cost_usd": 0, "metered_tokens": 0,
                            "api_equivalent_usd": unreported_allowance_usd, "cost_estimated": True,
                            "via": "operator_reconciliation", "receipt_id": receipt_id, "attempt": attempt},
            }))
        settlement = _settlement_path(receipt)
        settlement.parent.mkdir(exist_ok=True)
        # No fabricated output is recovered from this synthetic directory.
        write_text_atomic(settlement, json.dumps({
            "settled": True, "process_stopped": True, "operator_confirmed": True,
            "workdir": f"agents/reconciled-{receipt_id}",
        }))
        saved = {**saved, "status": "failed", "operator_reconciled": True}
        write_text_atomic(receipt, json.dumps(saved))
        return saved


def deny_api(*args, **kwargs):
    raise SubscriptionBlocked("Paid API calls are forbidden in subscription mode")


def check_cli_environment(env):
    forbidden = {"ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                 "CLAUDE_CODE_USE_FOUNDRY", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL",
                 "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "GOOGLE_APPLICATION_CREDENTIALS"}
    if any(value and (key.endswith("API_KEY") or key in forbidden) for key, value in env.items()):
        raise SubscriptionBlocked("Paid provider credentials or gateways are forbidden in subscription workers")


def codex_auth_path():
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser() / "auth.json"


def claude_auth_path():
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude").expanduser() / ".credentials.json"


def read_claude_auth():
    try:
        text = claude_auth_path().read_text(encoding="utf-8")
        oauth = json.loads(text)["claudeAiOauth"]
        if (not isinstance(oauth, dict) or not isinstance(oauth.get("accessToken"), str)
                or not oauth["accessToken"].strip() or oauth.get("subscriptionType") not in {"pro", "max", "team", "enterprise"}):
            raise ValueError("not a subscription login")
        # Copy only OAuth, never API key helpers, gateways, settings, or hooks.
        return json.dumps({"claudeAiOauth": oauth})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SubscriptionBlocked(
            "Claude subscription credentials unavailable; log in with Claude Code on this host "
            "(file-based CLAUDE_CONFIG_DIR credentials are required)"
        ) from exc


def preflight():
    for binary in ("codex", "claude"):
        if shutil.which(binary) is None:
            raise SubscriptionBlocked(f"Subscription mode requires the {binary} CLI")
    try:
        auth = codex_auth_path().read_text(encoding="utf-8")
    except OSError as exc:
        raise SubscriptionBlocked("Codex subscription login unavailable; run codex login first") from exc
    if classify_codex_auth(auth) != "subscription":
        raise SubscriptionBlocked("Codex must use a ChatGPT subscription login, not an API key")
    read_claude_auth()


def configure_workflow(agent, inp):
    requested = inp.execution_backend
    current = agent.ctx.execution_backend
    if current == "subscription" and requested != current:
        raise SubscriptionBlocked("Nested workflows cannot leave subscription mode")
    path = agent.ctx.root_workdir / "execution-backend.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["backend"] != requested:
            raise SubscriptionBlocked("Cannot change execution backend in an existing run; use a new run directory")
    elif requested == "subscription" and (getattr(inp, "resume_run", False)
            or (agent.ctx.root_workdir / "ac_workspaces").exists()):
        raise SubscriptionBlocked("Cannot resume a legacy API run as subscription work; use a new run directory")
    if requested == "api":
        return inp
    if agent.ctx.monitor is not None:
        raise SubscriptionBlocked("The API-backed LLM monitor is unavailable in subscription mode; omit --monitor")
    if agent.ctx.resume_from and agent.ctx.resume_from != agent.ctx.root_workdir.name:
        raise SubscriptionBlocked("Subscription resume requires the complete run directory and ledger, not an external cache")
    if inp.author_parallelism != 1:
        raise ValueError("Subscription mode currently requires author_parallelism=1; native API delegation is disabled")
    config = {"codex_model": CODEX_MODEL, "codex_effort": "xhigh", "claude_model": CLAUDE_MODEL,
              "claude_effort": "xhigh", "max_parallel": inp.subscription_max_parallel,
              "call_seconds": inp.subscription_call_seconds, "planning_usd": inp.subscription_planning_usd}
    if path.exists() and saved.get("settings") != config:
        raise SubscriptionBlocked("Subscription settings/allowance cannot change on resume")
    check_pending_calls(agent.ctx.root_workdir)
    if current != "subscription":
        preflight()
        root = agent.tracker
        while root.parent is not None:
            root = root.parent
        if root.counters.usd:
            raise SubscriptionBlocked("Cannot change billing mode after paid usage")
        root.subscription_mode = True
        root.spec = (root.spec or BudgetSpec()).model_copy(update={"max_usd": inp.subscription_planning_usd})
        restore_usage(agent.ctx, root)
        agent.tracker.counters.api_equivalent_usd = root.counters.api_equivalent_usd
        agent.ctx = replace(agent.ctx, execution_backend="subscription", subscription_config=config,
                            subscription_limiter=asyncio.Semaphore(inp.subscription_max_parallel),
                            api_client_factory=deny_api)
        # ACWorkflow constructs its seats in __init__, before input selection.
        for name in ("author", "critic", "council", "compute"):
            child = getattr(agent, name, None)
            if child is not None:
                child.ctx = agent.ctx
        write_text_atomic(path, json.dumps({"backend": requested, "settings": config,
                                           "usage_receipts": 1 if not path.exists() else saved.get("usage_receipts", 0)}))
    updates = {"council_models": COUNCIL_MODELS, "compute_model": CODEX_MODEL,
               "compute_reasoning_effort": "xhigh", "compute_cost_config": CODEX_COST,
               "compute_codex_sandbox": "auto"}
    if hasattr(inp, "cleanup_backend"):
        updates["cleanup_backend"] = "claude_code"
    return inp.model_copy(update=updates)


def check_pending_calls(root):
    # Check the whole run, not just the next packet's hash: changed inputs must
    # not bypass unresolved work left behind by a crashed process.
    for receipt in (root / "subscription_calls").glob("*.json"):
        if receipt.resolve() in _active_receipts:
            continue
        try:
            saved = reconcile_receipt(receipt)
        except SubscriptionBlocked as exc:
            raise SubscriptionBlocked(f"Unresolved subscription invocation ({receipt.name}): {exc}") from exc
        if _completed(saved) or (saved.get("status") == "failed" and _settled(receipt)):
            continue
        raise SubscriptionBlocked(
            f"Unresolved subscription invocation ({receipt.name}); reconcile before resuming this run"
        )


def restore_usage(ctx, root):
    path = ctx.root_workdir / "events.jsonl"
    nominal = tokens = 0
    events, damaged = [], False
    for line in path.read_bytes().splitlines() if path.exists() else []:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            damaged = True
            continue
        if not isinstance(event, dict):
            damaged = True
            continue
        if event.get("subscription_usage_id"):
            receipt = ctx.root_workdir / "subscription_usage" / f"{event['subscription_usage_id']}.json"
            if not receipt.exists():
                raise SubscriptionBlocked("Missing subscription usage receipt")
            continue
        events.append(event)
    backend_path = ctx.root_workdir / "execution-backend.json"
    if damaged and (not backend_path.exists() or _read_receipt(backend_path).get("usage_receipts") != 1):
        raise SubscriptionBlocked("Incomplete legacy subscription usage log; reconcile before resume")
    events.extend(_read_receipt(p) for p in (ctx.root_workdir / "subscription_usage").glob("*.json"))
    for event in events:
        p = event.get("payload", {})
        if event.get("kind") != "model.call" or not p.get("subscription"):
            continue
        value = p.get("api_equivalent_usd")
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                or p.get("cost_usd", 0) != 0):
            raise SubscriptionBlocked("Invalid subscription usage ledger")
        nominal += value
        if not math.isfinite(nominal):
            raise SubscriptionBlocked("Invalid cumulative subscription usage ledger")
        count = p.get("metered_tokens", (p.get("in_tokens", 0) or 0) + (p.get("out_tokens", 0) or 0))
        if type(count) is not int or count < 0:
            raise SubscriptionBlocked("Invalid subscription token ledger")
        tokens += count
    root.counters.api_equivalent_usd = max(root.counters.api_equivalent_usd, nominal)
    root.counters.tokens = max(root.counters.tokens, tokens)


def cli_config(provider, seconds):
    common = {"prompt": "{prompt}", "completion_signal": "exit", "cache_enabled": False,
              "sandbox": {"backend": "subprocess", "timeout_s": max(1, int(seconds)),
                          "limit_address_space": False, "provider_keys": [],
                          "env_allowlist": ["PATH", "LANG", "LC_ALL"]},
              "input_files": {"request.md": {"from_input": "packet"}},
              "output_files": {"text": "report.md"}, "done_outputs": {"status": "status"}}
    if provider == "codex":
        return {**common, "cmd": ["codex", "exec", "--ignore-user-config", "--ephemeral",
                                  "--skip-git-repo-check", "--json", "--output-last-message", "last-message.txt",
                                  "-c", 'forced_login_method="chatgpt"', "-c", 'model_provider="openai"',
                                  "-c", "tools.web_search=true"],
                "model": CODEX_MODEL, "model_reasoning_effort": "xhigh", "codex_sandbox": "auto",
                "copy_codex_auth": True, "usage": {"type": "codex_jsonl", "cost_config": CODEX_COST}}
    return {**common, "cmd": ["claude", "-p", "--setting-sources", "", "--disable-slash-commands",
                              "--settings", '{"disableAllHooks":true}', "--strict-mcp-config",
                              "--mcp-config", '{"mcpServers":{}}', "--no-session-persistence",
                              "--output-format", "stream-json", "--verbose", "--effort", "xhigh",
                              "--tools", "Read,Write,WebSearch,WebFetch", "--allowedTools", "Read,Write,WebSearch,WebFetch"],
            "model": CLAUDE_MODEL, "copy_claude_auth": True,
            "usage": {"type": "claude_json", "auth_mode": "subscription", "cost_config": CLAUDE_COST}}


async def run_packet(agent, packet, *, provider="codex", files=None, outputs=None, extra_prompt="", key=None, attachment=None, tools=True):
    from proofstack.agents.configurable_cli import ConfigurableCLIAgent
    from proofstack.budget import BudgetExhausted

    seconds = agent.ctx.subscription_config["call_seconds"]
    remaining = agent.tracker.remaining_wallclock_s()
    if remaining is not None:
        seconds = min(seconds, remaining)
    if seconds <= 0:
        raise TimeoutError("Subscription phase deadline reached")
    config = cli_config(provider, seconds)
    if not tools:
        if provider != "claude" or files or outputs or attachment:
            raise ValueError("Tool-free packets require Claude and no file inputs/outputs")
        for flag in ("--tools", "--allowedTools"):
            config["cmd"][config["cmd"].index(flag) + 1] = ""
        config.update(output_files={}, claude_result_output="text")
    config["input_files"].update({name: {"from_input": f"file_{i}"} for i, name in enumerate(files or {})})
    config["output_files"].update(outputs or {})
    attachment_hash = None
    if attachment is not None:
        attachment = Path(attachment)
        if not attachment.resolve().is_relative_to(agent.ctx.root_workdir.resolve()) or attachment.stat().st_size > 512 * 1024**2:
            raise SubscriptionBlocked("Invalid or oversized Compute handoff")
        with attachment.open("rb") as stream:
            attachment_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        config["input_files"]["compute-workspace.zip"] = {"from_binary_path_input": "attachment"}
    # Older Codex receipts may contain a final chat summary that overwrote report.md.
    identity = json.dumps((["separate-report-v2"] if provider == "codex" else []) + [key or agent.name, packet, files or {}, outputs, provider,
                           attachment_hash, tools, agent.ctx.subscription_config], sort_keys=True)
    digest = hashlib.sha256(identity.encode()).hexdigest()
    receipt = agent.ctx.root_workdir / "subscription_calls" / f"{digest}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    check_pending_calls(agent.ctx.root_workdir)
    previous = None
    if receipt.exists():
        previous = reconcile_receipt(receipt)
        if _completed(previous):
            write_text_atomic(agent.workdir / "subscription-settlement.json",
                              json.dumps({"settled": True, "receipt_id": digest}))
            return previous["output"]
        if previous.get("status") != "failed" or not _settled(receipt):
            raise SubscriptionBlocked("Interrupted subscription invocation; reconcile saved CLI artifacts before retrying")
        if previous.get("attempts", 1) >= 2:
            raise SubscriptionCallFailed("Subscription invocation failed twice; bounded retry exhausted; no API fallback")
    name = f"subscription-{provider}-{digest[:16]}"
    settlement = _settlement_path(receipt)
    settlement.parent.mkdir(exist_ok=True)
    config["subscription_settlement_path"] = str(settlement)
    ctx = replace(agent.ctx, component_configs={name: config}, model_overrides={})
    worker = ConfigurableCLIAgent(ctx, name=name, parent_budget_scope=agent.tracker.scope)
    worker.tracker.parent = agent.tracker
    prompt = ("Read request.md completely and carry out its role and instructions. The labelled messages "
              "are the exact supplied conversation, not another session. Do not read any other agent's files. "
              "Provider-hosted code tools mean local tools here; there are no API credentials. "
              "Write the complete final response, including required verdict tags, to report.md. "
              "Do not truncate the manuscript or review. " + extra_prompt)
    if not tools:
        prompt = packet + "\nReturn the requested response directly. No tools are available."
    # Deadline includes time waiting for the shared subscription concurrency slot.
    async with asyncio.timeout(seconds):
        async with agent.ctx.subscription_limiter:
            agent.tracker.check()
            lock = receipt.with_suffix(".lock").open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                lock.close()
                raise SubscriptionBlocked("Subscription invocation is already running") from exc
            try:
                if receipt.exists():
                    saved = reconcile_receipt(receipt)
                    if _completed(saved):
                        write_text_atomic(agent.workdir / "subscription-settlement.json",
                                          json.dumps({"settled": True, "receipt_id": digest}))
                        return saved["output"]
                    if saved != previous:
                        raise SubscriptionBlocked("Subscription invocation changed while queued")
                settlement.unlink(missing_ok=True)
                started = {"status": "started", "worker": name,
                           "attempts": (previous or {}).get("attempts", 0) + 1}
                write_text_atomic(receipt, json.dumps(started))
                write_text_atomic(agent.workdir / "subscription-settlement.json",
                                  json.dumps({"settled": False, "receipt_id": digest}))
                _active_receipts.add(receipt.resolve())
                out = await worker(prompt=prompt, packet=packet, attachment=str(attachment) if attachment else "",
                                   **{f"file_{i}": value for i, value in enumerate((files or {}).values())})
            except BudgetExhausted as exc:
                out = getattr(exc, "completed_output", None)
                if out is not None and getattr(out, "text", "").strip() and getattr(out, "status", None) == "done":
                    write_text_atomic(receipt, json.dumps({**started, "status": "completed", "output": out.model_dump(mode="json")}))
                raise
            except Exception as exc:
                if receipt.exists() and _settled(receipt):
                    reconcile_receipt(receipt)
                    raise SubscriptionCallFailed(
                        f"Subscription worker failed ({type(exc).__name__}), stopped and accounted; no API fallback"
                    ) from exc
                raise SubscriptionBlocked(
                    f"Subscription worker failed ({type(exc).__name__}); inspect {name} and reconcile before retrying; no API fallback"
                ) from exc
            finally:
                _active_receipts.discard(receipt.resolve())
                lock.close()
                if receipt.exists():
                    saved = reconcile_receipt(receipt)
                    write_text_atomic(agent.workdir / "subscription-settlement.json",
                                      json.dumps({"settled": _completed(saved) or _settled(receipt), "receipt_id": digest}))
            if out.status != "done" or not (out.text or "").strip():
                error = SubscriptionCallFailed if _settled(receipt) else SubscriptionBlocked
                raise error("Subscription CLI did not deliver a completed report; no API fallback")
            result = out.model_dump(mode="json")
            write_text_atomic(receipt, json.dumps({**started, "status": "completed", "output": result}))
            write_text_atomic(agent.workdir / "subscription-settlement.json",
                              json.dumps({"settled": True, "receipt_id": digest}))
            try:
                agent.tracker.check()
            except BudgetExhausted as exc:
                exc.completed_output = out
                raise
            return result


async def run_model_role(agent, inp):
    from proofstack.agents.ac.author import Author
    from proofstack.agents.ac.council import CouncilMember
    from proofstack.budget import BudgetExhausted

    is_author = isinstance(agent, Author)
    provider = "claude" if isinstance(agent, CouncilMember) and "anthropic/" in str(agent.MODEL) else "codex"
    messages = agent.render_messages(inp)
    packet = "\n\n".join(f"# {m['role']}\n{m['content']}" for m in messages)
    files, outputs, instruction = {}, {}, ""
    if is_author:
        for field, name in (("answer_tex", "answer.tex"), ("research_notes_tex", "research_notes.tex"),
                            ("references_bib", "references.bib")):
            files[name] = getattr(inp, field)
            outputs[field] = name
        instruction = ("TRANSPORT OVERRIDE: ignore request.md's fenced file-output convention; its mathematical "
                       "instructions and readiness rules still apply. "
                       "Edit answer.tex, research_notes.tex and references.bib directly. Do not repeat their contents "
                       "in report.md; retain the thinking summary, ready tag, council and compute requests there. "
                       "The council has only Astra via Codex and Fable via Claude Code, not Pro or Gemini. "
                       "Any budget figures are API-equivalent planning allowances, not subscription charges.")
        if inp.compute_zip_path:
            instruction += " The previous Compute workspace is provided as compute-workspace.zip; inspect it as needed."
    exhausted = None
    try:
        raw = await run_packet(agent, packet, provider=provider, files=files, outputs=outputs, extra_prompt=instruction,
                               key=agent._cache_key(inp), attachment=inp.compute_zip_path if is_author else None)
    except BudgetExhausted as exc:
        completed = getattr(exc, "completed_output", None)
        if completed is None or getattr(completed, "status", None) != "done" or not getattr(completed, "text", "").strip():
            raise
        raw, exhausted = completed.model_dump(mode="json"), exc
    agent._on_response(raw["text"], inp)
    result = agent.parse_output(raw["text"], inp)
    if is_author:
        if any(not isinstance(raw.get(field), str) for field in outputs):
            raise SubscriptionBlocked("Author did not preserve all canonical manuscript files")
        result = result.model_copy(update={**{field: raw[field] for field in outputs},
                                           "files_changed": list(files), "via": "subscription_cli", "artifact_status": "complete"})
    write_text_atomic(agent.workdir / "raw_response.txt", raw["text"])
    if exhausted is not None:
        exhausted.completed_output = result
        raise exhausted
    return result


async def linear_read_subscription(agent, tracker, workspace, deadline):
    import time
    from types import SimpleNamespace
    from proofstack.agents.linear_read import split_document, build_query, parse_reply, render_report
    # Each fresh worker sees only its prefix, never the later manuscript.
    preamble, blocks = split_document((workspace / "answer.tex").read_text())
    results = {}
    for i, block in enumerate(blocks):
        tracker.check()
        if time.monotonic() >= deadline:
            raise TimeoutError("Linear-reader deadline reached")
        messages = build_query(preamble, blocks[:i], block, i)
        packet = "\n\n".join(f"# {m['role']}\n{m['content']}" for m in messages)
        folder = agent.workdir / "linear_read" / hashlib.sha256(packet.encode()).hexdigest()
        proxy = SimpleNamespace(ctx=agent.ctx, tracker=tracker, name="cleanup-linear-read", workdir=folder)
        raw = await run_packet(proxy, packet, provider="claude", key=f"linear-{i}", tools=False)
        results[i] = parse_reply(raw["text"])
        if "parse_error" in results[i]:
            raise SubscriptionBlocked("Linear reader returned an invalid report")
    report, findings = render_report("answer.tex", CLAUDE_MODEL, blocks, results, 0)
    digest = hashlib.sha256(report.encode()).hexdigest()
    path = f"reviews/linear-{digest}.md"
    (workspace / "reviews").mkdir(exist_ok=True)
    write_text_atomic(workspace / path, report)
    await agent.events.emit("cleanup.linear_read", {"model": CLAUDE_MODEL, "passages": len(blocks),
                             "findings": len(findings), "cost_usd": 0, "subscription": True})
    return {"report": report, "path": path, "findings": len(findings), "cost_usd": 0}
