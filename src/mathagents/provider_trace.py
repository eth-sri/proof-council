"""Per-invocation, append-only provider accounting independent of debug logs."""
from contextvars import ContextVar
import json
import hashlib
import math
import os
from pathlib import Path
import threading
import time
import uuid
import tempfile

from loguru import logger


active_trace: ContextVar["ProviderTrace | None"] = ContextVar("provider_trace", default=None)
active_attempt: ContextVar[tuple | None] = ContextVar("provider_attempt", default=None)
COUNTS = ("input_tokens", "output_tokens", "cached_input_tokens", "cached_write_tokens", "reasoning_tokens")
# A planning heuristic, not measured throughput or a guaranteed billing bound.
ESTIMATED_OUTPUT_TOKENS_PER_SECOND = 50
PRE_INFERENCE_REJECTIONS = {"context_length_exceeded", "invalid_file", "file_not_found", "array_above_max_length"}


def _terminal_estimate(row):
    start, end = row.get("request_started_at"), row.get("terminal_observed_at")
    input_cost, output_rate = row.get("estimate_input_usd"), row.get("estimate_output_usd_per_token")
    if not all(type(x) in (int, float) and math.isfinite(x) and x >= 0
               for x in (start, end, input_cost, output_rate)):
        return {}
    elapsed = max(0.0, end - start)
    visible = row.get("estimate_visible_output_tokens", 0)
    allowance = row.get("estimate_output_allowance_tokens", 0)
    if any(type(x) is not int or x < 0 for x in (visible, allowance)):
        return {}
    unseen = 0 if row.get("pre_inference_rejection") else min(
        allowance, math.ceil(min(elapsed, allowance / ESTIMATED_OUTPUT_TOKENS_PER_SECOND)
                             * ESTIMATED_OUTPUT_TOKENS_PER_SECOND))
    # Visible reasoning is already output: do not add it a second time.
    outputs = max(visible, unseen)
    cost = (0 if row.get("pre_inference_rejection") else input_cost) + outputs * output_rate
    if not math.isfinite(cost):
        return {}
    return {"terminal_usage_estimate_usd": cost,
            "estimate_output_tokens": outputs, "estimate_elapsed_s": elapsed,
            "estimate_output_tokens_per_second": ESTIMATED_OUTPUT_TOKENS_PER_SECOND,
            "terminal_estimate_source": "request_elapsed_v1"}


def _bounded_tool_evidence(item):
    if len(json.dumps(item).encode()) <= 32_000:
        return dict(item)
    # Preserve checker identity even when unrelated diagnostics are enormous.
    kept = {key: item.get(key) for key in ("type", "id", "status", "container_id", "code")}
    code = kept["code"]
    if isinstance(code, str) and len(code.encode()) > 16_000:
        kept["code"] = None
    kept["outputs"] = []
    remaining = 8_000
    for output in item.get("outputs") or []:
        logs = output.get("logs") if output.get("type") == "logs" else None
        if not isinstance(logs, str) or remaining <= 0:
            continue
        data = logs.encode()
        if len(data) > remaining:
            half = remaining // 2
            logs = data[:half].decode(errors="ignore") + "\n[truncated]\n" + data[-half:].decode(errors="ignore")
        remaining -= len(logs.encode())
        kept["outputs"].append({"type": "logs", "logs": logs})
    kept["evidence_truncated"] = True
    if len(json.dumps(kept).encode()) > 32_000:
        kept["outputs"] = []
    if len(json.dumps(kept).encode()) > 32_000:
        kept["code"] = None
    return kept


class ProviderAccountingError(RuntimeError):
    """Durable accounting failed; retrying the model would hide spending."""


class ProviderTrace:
    def __init__(self, path: Path | None = None, *, on_failure=None, on_response=None, **metadata):
        self.id = uuid.uuid4().hex
        self.path = path
        self.metadata = metadata
        self.attempts = {}
        self.lock = threading.RLock()
        self.completed_report = None
        self._latest_request = None
        self.on_failure = on_failure
        self.on_response = on_response

    @classmethod
    def restore(cls, path, invocation_id):
        rows = [r for r in latest_attempts(path) if r["invocation_id"] == invocation_id]
        trace = cls(path, **{k: rows[0][k] for k in ("run_id", "agent", "call_id") if k in rows[0]})
        trace.id = invocation_id
        for i, row in enumerate(rows):
            key = (str(i), 0)
            trace.attempts[key] = dict(row)
            trace._latest_request = key
        return trace

    def tool_evidence(self):
        with self.lock:
            return [item for row in self.attempts.values() for item in row.get("code_interpreter_calls", [])]

    def retain_tool_receipt(self, item, receipt):
        with self.lock:
            for key, row in list(self.attempts.items()):
                calls = row.get("code_interpreter_calls", [])
                updated = [dict(call, notes_verification_receipt=receipt) if all(
                    call.get(k) == item.get(k) for k in ("id", "code", "container_id", "status"))
                    else call for call in calls]
                if updated != calls:
                    self.update(key, code_interpreter_calls=updated)

    def failure(self, key, **details):
        self.update(key, **details)
        if self.on_failure is not None:
            # Accounting above is durable; supplementary live monitoring must
            # never turn a provider failure into a new retryable exception.
            try:
                self.on_failure({"invocation_id": self.id,
                                 "attempt_id": f"{self.id}:{key[0]}:{key[1]}", **details})
            except Exception as exc:
                logger.warning("Could not emit provider attempt failure: {}", type(exc).__name__)

    def update(self, key, **changes):
        with self.lock:
            record = self.attempts.setdefault(key, {"attempt_id": f"{self.id}:{key[0]}:{key[1]}", "usage_unavailable": True})
            record.update(changes)
            row = {"invocation_id": self.id, **self.metadata, **record, "timestamp": time.time()}
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    data = (json.dumps(row, default=str, ensure_ascii=True) + "\n").encode()
                    fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                    try:
                        view = memoryview(data)
                        while view:
                            written = os.write(fd, view)
                            if not written:
                                raise OSError("short provider receipt write")
                            view = view[written:]
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                except OSError as exc:
                    raise ProviderAccountingError("Could not persist provider accounting; no further model retry is safe") from exc

    def totals(self, idx=None):
        with self.lock:
            rows = [row for key, row in self.attempts.items() if idx is None or key[1] == idx]
            return {
                **{name: sum(row.get(name, 0) for row in rows) for name in COUNTS},
                "cost": sum(row.get("cost", 0.0) for row in rows),
                "usage_unavailable": any(row.get("usage_unavailable", True) for row in rows),
                "provider_attempts": len(rows),
                "invocation_id": self.id,
                "provider_outcomes": [
                    {name: row[name] for name in ("outcome", "status", "stop_reason", "finish_reason", "response_id", "cancellation_acknowledged") if name in row}
                    for row in rows
                ],
            }

    def request(self, client, ts, idx, payload):
        key = (ts, idx)
        active_attempt.set(key)
        with self.lock:
            self._latest_request = key
            self.completed_report = None
            self.update(key, provider=client.api, model=payload.get("model", client.model),
                        reasoning=payload.get("reasoning"), outcome="started", ever_in_progress=False,
                        request_started_at=time.time())
            # Save pricing while the exact client/model is available. Resume can
            # settle a terminal response without credentials or another request.
            if callable(getattr(client, "_get_cost", None)):
                from .api_client import _estimate_salvaged_usage
                inputs, _ = _estimate_salvaged_usage(payload, [])
                output_allowance = payload.get("max_output_tokens", 0)
                if type(output_allowance) is not int or output_allowance < 0:
                    output_allowance = 0
                input_cost = float(client._get_cost(inputs, 0, 0, 0))
                output_rate = float(client._get_cost(inputs, 1, 0, 0)) - input_cost
                if all(math.isfinite(x) and x >= 0 for x in (input_cost, output_rate)):
                    self.update(key, estimate_input_usd=input_cost, estimate_output_usd_per_token=output_rate,
                                estimate_input_tokens=inputs,
                                estimate_output_allowance_tokens=output_allowance,
                                terminal_estimate_source="request_elapsed_v1")

    def estimate_terminal_usage(self):
        with self.lock:
            for key, row in list(self.attempts.items()):
                if (row.get("provider") == "openai" and row.get("response_id")
                        and row.get("status") in {"failed", "cancelled", "incomplete", "completed"}
                        and row.get("usage_unavailable", True) and not row.get("cost_estimated")):
                    fields = _terminal_estimate(row)
                    estimate = fields.get("terminal_usage_estimate_usd")
                    if estimate is None or not math.isfinite(estimate):
                        continue
                    self.update(key, **fields, cost=max(estimate, row.get("cost", 0)), cost_estimated=True,
                                estimated_cost_usd=estimate, reconciliation_pending=False,
                                estimate_reason="terminal_response_without_usage")

    def response(self, client, ts, idx, response):
        if not isinstance(response, dict):
            return
        key = (ts, idx)
        self._record_usage(client, key, response)
        # Persist a final report before any bounded artifact I/O can be interrupted.
        with self.lock:
            self._retain_completed_report(response, key)
        if self.on_response is not None:
            try:
                self.on_response(client, self)
            except ProviderAccountingError:
                raise
            except Exception as exc:
                logger.warning("Could not capture provider tool evidence: {}", type(exc).__name__)
            with self.lock:
                self._retain_completed_report(response, key)

    def _retain_completed_report(self, response, key):
        output = response.get("output") or []
        if response.get("status") == "completed" and not any(item.get("type") == "function_call" for item in output):
            text = "\n\n".join(
                part.get("text", "") for item in output if item.get("type") == "message"
                for part in item.get("content", []) if part.get("type") == "output_text"
            )
            if text.strip() and len(text.encode("utf-8")) <= 2_000_000:
                latest = self._latest_request in (None, key)
                if latest:
                    self.completed_report = text
                if self.path is not None:
                    tmp = None
                    try:
                        self.path.parent.mkdir(parents=True, exist_ok=True)
                        attempt = self.attempts[key]["attempt_id"]
                        report = {"invocation_id": self.id, "attempt_id": attempt, "response_id": response.get("id"),
                                  "report": text, "tool_evidence": self.tool_evidence()}
                        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as f:
                            tmp = f.name
                            json.dump(report, f)
                            f.flush()
                            os.fsync(f.fileno())
                        suffix = hashlib.sha256(attempt.encode()).hexdigest()
                        os.replace(tmp, self.path.parent / f"provider-completed-{self.id}-{suffix}.json")
                        if latest:
                            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as f:
                                tmp = f.name
                                json.dump(report, f)
                                f.flush()
                                os.fsync(f.fileno())
                            os.replace(tmp, self.path.parent / f"provider-completed-{self.id}.json")
                    except OSError as exc:
                        logger.warning("Could not persist supplementary completed provider report: {}", type(exc).__name__)
                    finally:
                        if tmp is not None:
                            try:
                                os.unlink(tmp)
                            except FileNotFoundError:
                                pass
                            except OSError as exc:
                                logger.warning("Could not remove temporary provider report: {}", type(exc).__name__)

    def _record_usage(self, client, key, response):
        fields = {}
        evidence = [_bounded_tool_evidence(item) for item in response.get("output") or []
                    if item.get("type") == "code_interpreter_call"]
        if evidence:
            with self.lock:
                previous = self.attempts.get(key, {}).get("code_interpreter_calls", [])
                by_id = {item["id"]: item for item in previous if item.get("id")}
                for item in evidence:
                    old = by_id.get(item.get("id"), {})
                    if old.get("container_id") == item.get("container_id"):
                        if item.get("code") is None and not item.get("evidence_truncated"):
                            item["code"] = old.get("code")
                        if old.get("code") == item.get("code") and not item.get("outputs"):
                            item["outputs"] = old.get("outputs", item.get("outputs"))
                    if old.get("notes_verification_receipt") and all(
                            old.get(k) == item.get(k) for k in ("code", "container_id", "status")):
                        item["notes_verification_receipt"] = old["notes_verification_receipt"]
                incoming = {item.get("id") for item in evidence}
                fields["code_interpreter_calls"] = [item for item in previous if item.get("id") not in incoming] + evidence
        usage = response.get("usage")
        if (response.get("status") in {"failed", "cancelled", "incomplete", "completed"}
                and (usage is None or response.get("usage_estimated"))):
            from .api_client import _estimate_salvaged_usage
            with self.lock:
                row = self.attempts.get(key, {})
                _, outputs = _estimate_salvaged_usage({}, response.get("output") or [])
                error = response.get("error") or {}
                fields.update(terminal_observed_at=row.get("terminal_observed_at", time.time()),
                              estimate_visible_output_tokens=max(outputs, row.get("estimate_visible_output_tokens", 0)),
                              pre_inference_rejection=(error.get("code") in PRE_INFERENCE_REJECTIONS
                                  and not response.get("output") and not row.get("ever_in_progress")))
                fields.update(_terminal_estimate({**row, **fields}))
        google_usage = response.get("usageMetadata")
        if google_usage is not None:
            usage = {
                "input_tokens": (google_usage.get("promptTokenCount", 0) or 0) + (google_usage.get("toolUsePromptTokenCount", 0) or 0),
                "output_tokens": (google_usage.get("candidatesTokenCount", 0) or 0) + (google_usage.get("thoughtsTokenCount", 0) or 0),
                "cached_input_tokens": google_usage.get("cachedContentTokenCount", 0),
                "reasoning_tokens": google_usage.get("thoughtsTokenCount", 0),
            }
        if usage is not None:
            inputs, outputs, cached, written = client._extract_usage_tokens(usage)
            fields.update(input_tokens=inputs, output_tokens=outputs, cached_input_tokens=cached,
                          cached_write_tokens=written, reasoning_tokens=client._extract_reasoning_tokens(usage),
                          cost=client._get_cost(inputs, outputs, cached, written),
                          usage_unavailable=bool(response.get("usage_partial", False) or response.get("usage_estimated", False)))
            if response.get("usage_estimated"):
                with self.lock:
                    allowance = fields.get("terminal_usage_estimate_usd", 0)
                fields["cost"] = max(fields["cost"], allowance)
                fields.update(cost_estimated=True, estimated_cost_usd=fields["cost"],
                              estimate_reason="salvaged_response_text", reconciliation_pending=False)
            with self.lock:
                previous = self.attempts.get(key, {})
                for name in (*COUNTS, "cost"):
                    fields[name] = max(fields.get(name, 0), previous.get(name, 0))
        for name in ("id", "status", "stop_reason"):
            if response.get(name) is not None:
                fields["response_id" if name == "id" else name] = response[name]
        if response.get("status") == "in_progress":
            fields["ever_in_progress"] = True
        candidates = response.get("candidates") or []
        if candidates:
            fields["finish_reason"] = candidates[0].get("finishReason")
        choices = response.get("choices") or []
        if choices:
            fields["chat_completion_terminal"] = all(choice.get("finish_reason") is not None for choice in choices)
        fields["outcome"] = "error" if response.get("exception") or response.get("error") else "response"
        self.update(key, **fields)


def latest_attempts(path: Path):
    """Last durable revision per attempt; ignore only a crash-truncated last line."""
    attempts = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        attempts[row["attempt_id"]] = row
    return list(attempts.values())


def usage_adjustments(run_dir: Path, events: list[dict]) -> list[dict]:
    """Reconcile receipts against already-charged calls without double billing."""
    billed = {}
    for event in events:
        if event.get("kind") != "model.call":
            continue
        key = event.get("call_id")
        payload = event.get("payload") or {}
        totals = billed.setdefault(key, {})
        for receipt, field in (("cost", "cost_usd"), ("input_tokens", "in_tokens"),
                               ("output_tokens", "out_tokens"), ("reasoning_tokens", "reasoning_tokens")):
            totals[receipt] = totals.get(receipt, 0) + (payload.get(field, 0) or 0)
    calls = {}
    for path in sorted((run_dir / "agents").glob("*/provider-attempts.jsonl")):
        for row in latest_attempts(path):
            key = row.get("call_id") or row["invocation_id"]
            call = calls.setdefault(key, {"cost": 0.0, "input_tokens": 0, "output_tokens": 0,
                                          "reasoning_tokens": 0, "usage_unavailable": False})
            for field in ("cost", "input_tokens", "output_tokens", "reasoning_tokens"):
                call[field] += row.get(field, 0) or 0
            call["usage_unavailable"] |= row.get("usage_unavailable", True)
            call["cost_estimated"] = call.get("cost_estimated", False) or row.get("cost_estimated", False)
            call["model"] = row.get("model")
    adjustments = []
    for key, total in calls.items():
        charged = billed.get(key, {})
        payload = {
            field: max(0, total[source] - charged.get(source, 0))
            for source, field in (("cost", "cost_usd"), ("input_tokens", "in_tokens"),
                                  ("output_tokens", "out_tokens"), ("reasoning_tokens", "reasoning_tokens"))
        }
        if any(payload.values()) or total["usage_unavailable"]:
            adjustments.append({"kind": "provider.usage_adjustment", "call_id": key,
                                "payload": {**payload, "usage_unavailable": total["usage_unavailable"],
                                            "cost_estimated": total.get("cost_estimated", False), "model": total["model"]}})
    # Operator-reconciled legacy charges have stable IDs and are never appended
    # to model.call, so repeated restarts and exports count each debit once.
    ledger = run_dir / "recovery-debits.json"
    if ledger.exists():
        for key, debit in json.loads(ledger.read_text()).items():
            cost = float(debit["cost_usd"])
            if not math.isfinite(cost) or cost < 0 or not debit.get("reason"):
                raise ValueError("Invalid recovery debit")
            adjustments.append({"kind": "provider.usage_adjustment", "call_id": f"recovery:{key}",
                                "payload": {"cost_usd": cost, "in_tokens": 0, "out_tokens": 0,
                                            "reasoning_tokens": 0, "usage_unavailable": False,
                                            "model": "legacy-reconciliation", "reason": debit["reason"]}})
    return adjustments
