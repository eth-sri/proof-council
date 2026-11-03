"""Serialized accounting for ordinary API calls and provider receipt recovery."""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from mathagents.provider_trace import COUNTS, usage_adjustments


USAGE_FIELDS = ("cost_usd", "in_tokens", "out_tokens", "reasoning_tokens")
logger = logging.getLogger(__name__)


def provider_attempt_terminal(row):
    if row.get("chat_completion_terminal") is True:
        return True
    provider = row.get("provider", "openai")
    if provider == "openai":
        return row.get("status") in {"failed", "cancelled", "incomplete", "completed"}
    if provider == "anthropic":
        return row.get("stop_reason") in {
            "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal",
            "model_context_window_exceeded",
        }
    if provider == "google":
        return row.get("finish_reason") in {
            "STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER", "BLOCKLIST",
            "PROHIBITED_CONTENT", "SPII", "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS",
            "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_OTHER", "NO_IMAGE", "IMAGE_RECITATION",
        }
    return False


def transport_uncertain(row):
    # Older receipts used the remote-reconciliation flag for no-ID outages.
    return bool((row.get("transport_uncertain") and not row.get("response_id")
                 and not provider_attempt_terminal(row)) or (
        row.get("reconciliation_pending") and not row.get("response_id")
        and row.get("error_type") and row.get("retry_decision") in {"retry", "stop"}))


def provider_attempts_settled(rows):
    def settled(row):
        if row.get("reconciliation_pending") or transport_uncertain(row):
            return False
        if provider_attempt_terminal(row):
            return not row.get("usage_unavailable", True) or row.get("cost_estimated", False)
        # An explicit failed create/retry is not an open background response.
        # Never discard a response ID, partial usage, or a merely started call.
        return (not row.get("response_id") and row.get("error_type")
                and row.get("retry_decision") in {"retry", "stop"}
                and not any(row.get(key) for key in ("cost", *COUNTS)))
    return all(settled(row) for row in rows)


async def reconcile_provider_call(ctx, tracker, folder):
    """Re-read and charge durable evidence, without making a provider request."""
    from mathagents.provider_trace import latest_attempts
    path = folder / "provider-attempts.jsonl"
    marker = folder / "provider-settlement.json"
    if _operator_reconciled(ctx.root_workdir, folder):
        await settle_provider_usage(ctx, tracker)
        return True
    if not path.exists():
        return not marker.exists() or json.loads(marker.read_text()).get("settled") is True
    rows = latest_attempts(path)
    from mathagents.provider_trace import ProviderTrace
    for invocation in dict.fromkeys(r["invocation_id"] for r in rows):
        ProviderTrace.restore(path, invocation).estimate_terminal_usage()
    rows = latest_attempts(path)
    await settle_provider_usage(ctx, tracker, call_ids={r.get("call_id") or r["invocation_id"] for r in rows})
    if rows and provider_attempts_settled(rows):
        return True
    return bool(not rows and marker.exists() and json.loads(marker.read_text()).get("settled") is True)


def _reconciliation_identity(root, folder):
    root, folder = Path(root).resolve(), Path(folder).resolve()
    if folder.parent != root / "agents" or not folder.is_dir():
        raise ValueError("Select an existing call directory directly under RUN_DIR/agents")
    evidence = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                if (folder / name).exists() else None
                for name in ("input.json", "provider-attempts.jsonl")}
    identity = {"workdir": str(folder.relative_to(root)), "evidence": evidence}
    key = "api-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity, key


def _operator_reconciled(root, folder):
    receipt = folder / "provider-reconciliation.json"
    if not receipt.exists():
        return False
    identity, key = _reconciliation_identity(root, folder)
    saved = json.loads(receipt.read_text())
    ledger = Path(root) / "recovery-debits.json"
    debits = json.loads(ledger.read_text()) if ledger.exists() else {}
    amount = saved.get("unreported_allowance_usd")
    return (saved.get("identity") == identity and saved.get("debit_key") == key
            and saved.get("workers_stopped") is True and saved.get("provider_work_terminal") is True
            and type(amount) in (int, float) and math.isfinite(amount) and amount >= 0
            and debits.get(key, {}).get("cost_usd") == amount)


def reconcile_stopped_provider_call(root, folder, *, workers_stopped=False,
                                    provider_work_terminal=False, unreported_allowance_usd):
    """Operator-only recovery after independently settling remote provider work."""
    from proofstack.atomic import write_text_atomic
    if not workers_stopped or not provider_work_terminal:
        raise ValueError("Confirm both stopped local workers and terminal remote provider work")
    amount = unreported_allowance_usd
    if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
        raise ValueError("Unreported allowance must be finite and nonnegative")
    root, folder = Path(root).resolve(), Path(folder).resolve()
    identity, key = _reconciliation_identity(root, folder)
    backend = root / "execution-backend.json"
    if backend.exists() and json.loads(backend.read_text()).get("backend") == "subscription":
        raise ValueError("Use reconcile_subscription.py for subscription runs")
    with (root / ".api-reconciliation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = root / "recovery-debits.json"
        debits = json.loads(ledger.read_text()) if ledger.exists() else {}
        if key in debits and debits[key].get("cost_usd") != amount:
            raise ValueError("This evidence already has a different recovery debit")
        if key not in debits:
            debits[key] = {"cost_usd": amount, "reason": "Operator reconciliation: " + identity["workdir"]}
            write_text_atomic(ledger, json.dumps(debits, sort_keys=True))
        saved = {"identity": identity, "debit_key": key, "workers_stopped": True,
                 "provider_work_terminal": True, "unreported_allowance_usd": amount}
        write_text_atomic(folder / "provider-reconciliation.json", json.dumps(saved, sort_keys=True))
        return saved


def _normalized_usage(payload):
    normalized = {}
    for key in (*USAGE_FIELDS, "metered_tokens"):
        if key == "metered_tokens" and key not in payload:
            continue
        value = payload.get(key)
        number = float(value if value is not None else 0)
        if isinstance(value, bool) or not math.isfinite(number) or number < 0:
            raise ValueError(f"Invalid usage field: {key}")
        if key != "cost_usd" and not number.is_integer():
            raise ValueError(f"Nonintegral token count: {key}")
        normalized[key] = number if key == "cost_usd" else int(number)
    return normalized


def _usage_events(run_dir: Path):
    path = run_dir / "events.jsonl"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []
    events, damaged = [], []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("event is not an object")
            if event.get("kind") == "model.call":
                payload = event.get("payload") or {}
                if not isinstance(payload, dict):
                    raise ValueError("model usage is not an object")
                if event.get("call_id") is not None and not isinstance(event["call_id"], str):
                    raise ValueError("call_id is not a string")
                event["payload"] = {**payload, **_normalized_usage(payload)}
        except (ValueError, TypeError, OverflowError, RecursionError):
            damaged.append(number)
            continue
        events.append(event)
    return events, damaged


def _tokens(payload):
    # Native CLI accounting may use a distinct metered total; reasoning tokens
    # are otherwise already included in output tokens.
    ordinary = (payload.get("in_tokens", 0) or 0) + (payload.get("out_tokens", 0) or 0)
    return payload.get("metered_tokens", ordinary) or 0


def _billed_usage(events):
    billed = {}
    for event in events:
        if event.get("kind") != "model.call" or not event.get("call_id"):
            continue
        previous = billed.setdefault(event["call_id"], dict.fromkeys(USAGE_FIELDS, 0))
        payload = event.get("payload") or {}
        for key in USAGE_FIELDS:
            previous[key] += payload.get(key, 0) or 0
    return billed


def _charge_usage(ctx, tracker, call_id, current, billed):
    charged = ctx._provider_usage_charged.get(call_id, {})
    accounted = {key: max(billed.get(key, 0), charged.get(key, 0)) for key in USAGE_FIELDS}
    delta = {key: max(0, current.get(key, 0) - accounted[key]) for key in USAGE_FIELDS}
    tracker.add_usd(delta["cost_usd"])
    tracker.add_tokens(delta["in_tokens"] + delta["out_tokens"])
    ctx._provider_usage_charged[call_id] = {
        key: accounted[key] + delta[key] for key in USAGE_FIELDS}


@dataclass(frozen=True)
class UsageSnapshot:
    cost_usd: float
    tokens: int
    adjustments: list[dict]
    damaged: list[int]
    billed_usage: dict[str, dict[str, float | int]]


def provider_usage_snapshot(run_dir: Path) -> UsageSnapshot:
    events, damaged = _usage_events(run_dir)
    # Provider receipts remain strict: only the supplementary event log is
    # tolerant of crash fragments, including fragments followed by new events.
    adjustments = usage_adjustments(run_dir, events)
    payloads = [event.get("payload") or {} for event in events if event.get("kind") == "model.call"]
    payloads.extend(item["payload"] for item in adjustments)
    return UsageSnapshot(
        cost_usd=sum(float(p.get("cost_usd", 0.0) or 0.0) for p in payloads),
        tokens=sum(_tokens(p) for p in payloads),
        adjustments=adjustments, damaged=damaged, billed_usage=_billed_usage(events),
    )


async def _finish_settlement(operation):
    # Join the entire append/counter transaction, including failed writes.
    task = asyncio.create_task(operation)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break  # Inspect the failure below without replacing cancellation.
    try:
        result = task.result()
    except BaseException as exc:
        if cancelled:
            raise asyncio.CancelledError from exc
        raise
    if cancelled:
        raise asyncio.CancelledError
    return result


async def record_provider_usage(ctx, tracker, payload, *, call_id, emitter):
    """Record cumulative call usage, subtracting any already-settled amounts."""
    async def record():
        async with ctx._usage_settlement_lock:
            current = _normalized_usage(payload)
            try:
                events, _ = _usage_events(ctx.root_workdir)
            except OSError as exc:
                # Without history we cannot safely append a billing delta.
                # Keep the paid reply and its known charge; receipts allow a
                # later settlement once the supplementary log is readable.
                _charge_usage(ctx, tracker, call_id, current, {})
                logger.warning("Cannot read events.jsonl (%s); usage settlement deferred for %s",
                               type(exc).__name__, call_id)
                try:
                    await emitter.emit("accounting.events_unreadable", {
                        "path": "events.jsonl", "error_type": type(exc).__name__,
                        "reported_usage": current,
                        "warning": "Known usage retained; billing event deferred until history is readable",
                    }, call_id=call_id)
                except Exception:
                    logger.warning("Could not log deferred usage settlement for %s", call_id)
                return
            previous = _billed_usage(events).get(call_id, {})
            delta = {key: max(0, current[key] - previous.get(key, 0)) for key in USAGE_FIELDS}
            details = {**payload, **delta}
            if any(previous.values()):
                details["reported_usage"] = current
            # A paid reply still counts if the supplementary event writer fails.
            # The in-process floor prevents either retry or recovery charging it
            # again; provider receipts remain the durable source after a restart.
            _charge_usage(ctx, tracker, call_id, current, previous)
            await emitter.emit("model.call", details, call_id=call_id)

    await _finish_settlement(record())


async def settle_provider_usage(ctx, tracker, *, call_ids=None, emitter=None):
    """Restore a run snapshot, or settle only the specified critic calls."""
    emitter = emitter or ctx.events

    async def settle():
        async with ctx._usage_settlement_lock:
            snapshot = provider_usage_snapshot(ctx.root_workdir)
            # Native CLI charges can still advance counters while we await a
            # write; compare the snapshot with its own pre-await baseline.
            accounted = max(node.counters.usd for node in tracker.chain())
            accounted_tokens = max(node.counters.tokens for node in tracker.chain())
            if snapshot.damaged:
                await emitter.emit("accounting.events_corrupt", {
                    "path": "events.jsonl", "skipped_lines": snapshot.damaged,
                    "warning": "Damaged event rows skipped; provider receipts are still reconciled",
                })
            for adjustment in snapshot.adjustments:
                call_id, payload = adjustment["call_id"], adjustment["payload"]
                if call_id.startswith("recovery:"):
                    continue  # Legacy debits stay in their own durable ledger.
                if call_ids is not None and call_id not in call_ids:
                    continue
                # Charge known usage even if another attempt has unknown usage.
                # Settlement eligibility is a separate check; this is a floor,
                # never a claim that the unresolved work was free.
                if not any(payload.get(k, 0) for k in ("cost_usd", "in_tokens", "out_tokens", "reasoning_tokens")):
                    continue
                await emitter.emit("model.call", {
                    **payload, "via": "resume_reconciliation" if call_ids is None else "critic_reconciliation",
                }, call_id=call_id)
                if call_ids is not None:
                    previous = snapshot.billed_usage.get(call_id, {})
                    current = {key: previous.get(key, 0) + payload.get(key, 0) for key in USAGE_FIELDS}
                    _charge_usage(ctx, tracker, call_id, current, previous)
            if call_ids is None:
                tracker.add_usd(max(0.0, snapshot.cost_usd - accounted))
                tracker.add_tokens(max(0, snapshot.tokens - accounted_tokens))
                # Remember restored call totals too: a later ordinary call can
                # finish while the event history is temporarily unreadable.
                totals = {key: dict(value) for key, value in snapshot.billed_usage.items()}
                for adjustment in snapshot.adjustments:
                    total = totals.setdefault(adjustment["call_id"], dict.fromkeys(USAGE_FIELDS, 0))
                    for key in USAGE_FIELDS:
                        total[key] += adjustment["payload"].get(key, 0)
                for call_id, total in totals.items():
                    charged = ctx._provider_usage_charged.get(call_id, {})
                    ctx._provider_usage_charged[call_id] = {
                        key: max(total[key], charged.get(key, 0)) for key in USAGE_FIELDS}
            return snapshot.cost_usd

    # The append and corresponding counter change must finish together, even
    # if cancellation lands while waiting for the sink lock or its writer thread.
    return await _finish_settlement(settle())
