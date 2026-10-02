"""Serialized accounting for ordinary API calls and provider receipt recovery."""
from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from mathagents.provider_trace import usage_adjustments


USAGE_FIELDS = ("cost_usd", "in_tokens", "out_tokens", "reasoning_tokens")
logger = logging.getLogger(__name__)


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
                if call_ids is not None and (call_id not in call_ids or (
                        payload["usage_unavailable"] and not payload.get("cost_estimated"))):
                    continue
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
