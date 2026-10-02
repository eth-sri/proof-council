"""This module provides a unified API for querying various large language models."""

import inspect
import json
import os
import re
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar, copy_context
import threading
from queue import Queue, Empty
import math
from dataclasses import dataclass
from datetime import datetime

import anthropic
import requests
from anthropic.types import TextBlock, ThinkingBlock
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from loguru import logger
from openai import OpenAI, RateLimitError
from together import Together
from tqdm import tqdm
from transformers import AutoTokenizer

from mathagents.request_logger import request_logger
from mathagents.provider_trace import ProviderTrace, ProviderAccountingError, active_trace, active_attempt, COUNTS
from mathagents.utils import check_for_extra_keys

# Ask for the visible reply, not extraction of private reasoning. The latter
# can trigger stop_details.category=reasoning_extraction even when replaying
# a valid thinking block.
_ANTHROPIC_FINAL_ANSWER_SALVAGE_PROMPT = (
    "Your previous turn ended without any visible reply. Please now write your reply to the "
    "request above, in the format the task requires. Target at most 1000 words unless the task "
    "genuinely needs more (for example complete file contents). If you are unsure about a point, say so."
)

_OPENAI_MAX_OUTPUT_TOKENS_WRAPUP_PROMPT = (
    "Your previous turn stopped because it reached the maximum output-token limit "
    "(incomplete_details.reason=max_output_tokens), so its visible output is missing or truncated "
    "and will be discarded; only this reply is read. Do not redo research or tool work that is "
    "already done above - build on it. Wrap up now: finish any pending file edits (if your work is "
    "transmitted via files, write the complete final versions), then write your complete, "
    "self-contained final response in the exact output format the task requires. Everything the "
    "format needs, including any full file contents, must appear in this single reply; do not write "
    "a continuation fragment. Keep any further reasoning brief."
)

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None


# Pattern matches "try again in 478ms" or "try again in 5.2s" — the
# OpenAI rate-limit error message format. Returned in seconds.
_RETRY_AFTER_RE = re.compile(
    r"try again in\s+([\d.]+)\s*(ms|s)\b",
    re.IGNORECASE,
)


def _parse_retry_after_seconds(msg: str | None) -> float:
    """Extract retry-after in seconds from a rate-limit error message.
    Falls back to a 30 s default when the format is unrecognised."""
    if not msg:
        return 30.0
    m = _RETRY_AFTER_RE.search(msg)
    if not m:
        return 30.0
    val = float(m.group(1))
    if m.group(2).lower() == "ms":
        val /= 1000.0
    return val


class _SalvageUsage:
    """Stand-in for ``response.usage`` when it is ``None`` due to
    ``status="failed"``. Holds either zero (no estimate available) or
    a conservative char-count estimate of input + output tokens, so
    the salvaged turn does NOT bill as $0 against the BudgetTracker.

    OpenAI charged us server-side for the work the model actually
    did; the original code recorded $0 for it, which let the
    workflow happily march into another expensive round past a hard
    run budget. A rough estimate is far better than zero — at Pro
    rates one salvaged call can be $5-15.
    """

    def __init__(self, input_tokens: int = 0, output_tokens: int = 0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        self.input_tokens_details = None
        self.output_tokens_details = None

    def model_dump(self) -> dict:
        """Match the Pydantic-usage shape ``_usage_to_dict`` looks for.
        Without this, ``_usage_to_dict`` falls through to ``return {}``
        and ``_extract_usage_tokens`` reads zero — silently zero-billing
        the salvaged turn against the BudgetTracker (which defeats the
        whole point of estimating salvaged usage in the first place).
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


class ProviderAttachmentRejectedError(ValueError):
    """Terminal provider rejection carrying usage from the failed call.

    Attachment failures are retried by the Author without the optional
    artifact. The first attempt may nevertheless have completed substantial
    reasoning or tool work, so callers must charge ``cost`` before retrying.
    """

    def __init__(self, message: str, *, cost: dict):
        super().__init__(message)
        self.cost = dict(cost)


class _BackgroundResponseTimeout(TimeoutError):
    pass


class _ClientTerminated(Exception):
    """Raised inside workers when ``terminate()`` was called; must escape
    the retry loops instead of being retried."""
    pass


class _ProviderWallclockTimeout(TimeoutError):
    pass


class RequiredHostedToolUnavailable(ValueError):
    """A retry cannot perform mandatory tool work within its remaining allowance."""


@dataclass
class _ResponsesRetryState:
    owner: object
    started_at: float
    pro_failures: int = 0
    background_timeouts: int = 0
    hosted_tool_calls_used: int = 0


# One state per logical query, shared by its inner/outer retries but not by
# concurrent queries using the same APIClient (or by nested tool calls).
_responses_retry_state: ContextVar[_ResponsesRetryState | None] = ContextVar(
    "responses_retry_state", default=None,
)


def _estimate_salvaged_usage(
    payload: dict, output_items: list
) -> tuple[int, int]:
    """Conservative (in_tokens, out_tokens) estimate from a failed
    Responses-API response, using char-count / 3 as the rule. 3
    chars/token is intentionally on the over-counting side compared
    to the 4 chars/token rule of thumb; we want to err toward
    *over*-billing the budget tracker for a salvaged turn.

    Walks ``payload["input"]`` for message content and ``output_items``
    for reasoning / code_interpreter_call / message blocks. Tool-call
    metadata (web_search_call action fields etc.) is not counted.
    """

    def _add_text(acc: list[int], text) -> None:
        if isinstance(text, str):
            acc[0] += len(text)

    def _walk_content(content, acc: list[int]) -> None:
        if isinstance(content, str):
            _add_text(acc, content)
            return
        if not isinstance(content, list):
            return
        for c in content:
            if isinstance(c, dict):
                t = c.get("text", "")
                _add_text(acc, t)
            else:
                t = getattr(c, "text", None)
                if t is not None:
                    _add_text(acc, t)

    in_chars = [0]
    inp = payload.get("input", []) if isinstance(payload, dict) else []
    if isinstance(inp, list):
        for msg in inp:
            if not isinstance(msg, dict):
                continue
            _walk_content(msg.get("content", ""), in_chars)

    out_chars = [0]
    for item in output_items or []:
        t = getattr(item, "type", None)
        if t is None and isinstance(item, dict):
            t = item.get("type")
        if t == "message":
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            _walk_content(content, out_chars)
        elif t == "code_interpreter_call":
            code = getattr(item, "code", None)
            if code is None and isinstance(item, dict):
                code = item.get("code", "")
            _add_text(out_chars, code or "")
        elif t == "reasoning":
            for attr in ("summary", "content"):
                val = getattr(item, attr, None)
                if val is None and isinstance(item, dict):
                    val = item.get(attr)
                _walk_content(val, out_chars)

    return max(in_chars[0] // 3, 0), max(out_chars[0] // 3, 0)


def _openai_responses_tool_descriptor(tool_desc):
    desc = dict(tool_desc)
    if desc.get("type") == "code_interpreter" and "container" not in desc:
        desc["container"] = {"type": "auto"}
    return desc


def _openai_response_hit_max_output_tokens(response) -> bool:
    if getattr(response, "status", None) != "incomplete":
        return False
    details = getattr(response, "incomplete_details", None)
    if isinstance(details, dict):
        reason = details.get("reason")
    else:
        reason = getattr(details, "reason", None)
    return reason == "max_output_tokens"


def _drop_trailing_reasoning_items(conversation: list) -> int:
    """Remove reasoning items at the tail of a Responses conversation.

    A truncated (``max_output_tokens``) response typically ends in
    reasoning items with no following message or tool call; the API
    rejects such a tail when it is replayed as input.
    """
    dropped = 0
    while conversation and conversation[-1].get("type") == "reasoning":
        conversation.pop()
        dropped += 1
    return dropped


_TERMINAL_API_ERROR_HINTS: tuple[str, ...] = (
    "401",
    "invalid_api_key",
    "invalid api key",
    "no api key",
    "model_not_found",
    "model not found",
    "does not exist",
    "permission_denied",
    "permission denied",
    "unauthorized",
    "insufficient_quota",
    "billing_hard_limit_reached",
    "array_above_max_length",
    "container has too many files",
    "maximum of 1000 files",
)


class _ToolRecoveryFailed(RuntimeError):
    """Do not restart an exhausted hosted-tool loop after its one wrap-up."""


def _is_context_length_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(hint in text for hint in (
        "maximum context length", "input token count", "context_length_exceeded",
    ))


def _is_terminal_api_error(
    exc: BaseException, *, response_poll: bool = False, context_length_recovery: bool = False,
) -> bool:
    """True when an SDK error is bad-key / missing-model / forbidden.

    Provider SDKs surface these inconsistently — sometimes as raw HTTP
    status code, sometimes as nested error.code strings — so we match
    on the error text rather than the exception class. Used in both
    the outer retry supervisor (``_run_query_with_retry``) and inside
    each provider's inner loop so a 401 on the first inner attempt
    raises out of the loop immediately instead of sleeping 60s per
    inner retry before the outer supervisor sees it.
    """
    if isinstance(exc, (_ClientTerminated, _ProviderWallclockTimeout, ProviderAccountingError,
                        _ToolRecoveryFailed, RequiredHostedToolUnavailable)):
        return True
    text = str(exc).lower()
    # A just-created background response can briefly be absent from the GET
    # replica. This exception applies only to polling, never model creation.
    if response_poll and (getattr(exc, "status_code", None) == 404 or "error code: 404" in text):
        return False
    if context_length_recovery and _is_context_length_error(exc) and getattr(exc, "status_code", None) in (None, 400):
        return False
    # Failed background Responses have no HTTP status on their ValueError.
    # Neither retries nor Pro -> standard fallback can fix the same input.
    if _is_context_length_error(exc):
        return True
    if getattr(exc, "status_code", None) in (400, 401, 403, 404, 422):
        return True
    if any(hint in text for hint in ("error code: 400", "error: 400", "invalid_request_error")):
        return True
    return any(hint in text for hint in _TERMINAL_API_ERROR_HINTS)


class APIClient:
    def _report_attempt_failure(self, ts, idx, exc, *, retry_decision):
        trace = active_trace.get()
        if trace is not None:
            trace.failure(
                (ts, idx), error_type=type(exc).__name__,
                error_code="context_length_exceeded" if _is_context_length_error(exc) else getattr(exc, "code", None),
                error_message=str(exc)[:500],
                retry_decision=retry_decision,
            )

    def _provider_response_dict(self, response):
        if hasattr(response, "model_dump"):
            return response.model_dump()
        return {"id": getattr(response, "id", None), "status": getattr(response, "status", None),
                "usage": getattr(response, "usage", None)}

    def _cancel_background_response(self, client, response_id, ts, idx):
        trace = active_trace.get()
        if trace is not None:
            trace.update((ts, idx), response_id=response_id, cancellation_requested=True)
        # Independent of self.terminated: cleanup must still GET the same ID.
        # A daemon bounds SDK retries/transport hangs as well as socket timeouts.
        deadline = time.monotonic() + 2.5

        def bounded(operation):
            results = Queue(maxsize=1)
            def run():
                try:
                    results.put(operation())
                except Exception:
                    results.put(None)
            threading.Thread(target=run, daemon=True).start()
            try:
                return results.get(timeout=max(0.001, min(1.2, deadline - time.monotonic())))
            except Empty:
                return None

        receipt = bounded(lambda: client.responses.cancel(response_id, timeout=1.0))
        if trace is not None:
            trace.update((ts, idx), cancellation_acknowledged=receipt is not None)
        if receipt is not None:
            self._log_provider_response(ts=ts, batch_idx=idx, response=self._provider_response_dict(receipt))
        data = self._provider_response_dict(receipt) if receipt is not None else {}
        if (data.get("status") not in {"cancelled", "failed", "incomplete"}
                or data.get("usage") is None):
            fetched = bounded(lambda: client.responses.retrieve(
                response_id, timeout=1.0, include=["code_interpreter_call.outputs"]))
            if fetched is not None:
                receipt = fetched
                self._log_provider_response(ts=ts, batch_idx=idx, response=self._provider_response_dict(receipt))
        data = self._provider_response_dict(receipt) if receipt is not None else {}
        terminal = data.get("status") in {"completed", "cancelled", "failed", "incomplete"}
        if trace is not None:
            trace.update((ts, idx), reconciliation_pending=not terminal)
            if not terminal or data.get("usage") is None:
                trace.update((ts, idx), usage_unavailable=True)
        if not terminal:
            logger.warning(f"Background response {response_id} remains unresolved after bounded cancellation reconciliation")
        return receipt

    def _sleep(self, seconds):
        remaining = max(0.0, float(seconds))
        while remaining:
            if self.terminated:
                raise _ClientTerminated("Cancelled during provider retry backoff")
            step = min(1.0, remaining)
            time.sleep(step)
            remaining -= step

    def _log_provider_request(self, *, ts, batch_idx, request, **info):
        trace = active_trace.get()
        if trace is not None:
            trace.request(self, ts, batch_idx, request)
            info.update(trace.metadata, invocation_id=trace.id)
        try:
            request_logger.log_request(ts=ts, batch_idx=batch_idx, request=request, **info)
        except OSError as exc:
            logger.warning(f"Provider debug request log failed: {type(exc).__name__}")

    def _log_provider_response(self, *, ts, batch_idx, response=None, **info):
        trace = active_trace.get()
        if trace is not None:
            trace.response(self, ts, batch_idx, response or info)
            if "cancellation_acknowledged" in info:
                trace.update((ts, batch_idx), cancellation_acknowledged=info["cancellation_acknowledged"])
        try:
            request_logger.log_response(ts=ts, batch_idx=batch_idx, response=response, **info)
        except OSError as exc:
            logger.warning(f"Provider debug response log failed: {type(exc).__name__}")

    def _bounded_provider_operation(self, operation, *, timeout, cancel=None, on_abandoned_result=None):
        """Bound wall time even when an SDK blocks between stream events.

        SDK I/O runs in a daemon, never the asyncio executor's shutdown path.
        Closing a broken transport is best-effort and must not block the caller.
        """
        results = Queue(maxsize=1)
        context = copy_context()
        ownership = threading.Lock()
        abandoned = False

        def abandon_result(value):
            if on_abandoned_result is not None:
                try:
                    on_abandoned_result(value)
                except Exception as exc:
                    logger.warning(f"Late provider response cleanup failed: {type(exc).__name__}")

        def run():
            try:
                result = (True, operation())
            except BaseException as exc:
                result = (False, exc)
            with ownership:
                if not abandoned:
                    results.put(result)
                    return
            if result[0]:
                abandon_result(result[1])

        threading.Thread(target=lambda: context.run(run), daemon=True).start()
        deadline = time.monotonic() + float(timeout) if timeout is not None else float("inf")
        while True:
            remaining = deadline - time.monotonic()
            if self.terminated or remaining <= 0:
                with ownership:
                    abandoned = True
                    try:
                        completed = results.get_nowait()
                    except Empty:
                        completed = None
                if completed is not None and completed[0]:
                    cleanup_context = copy_context()
                    threading.Thread(target=lambda: cleanup_context.run(abandon_result, completed[1]), daemon=True).start()
                if cancel is not None:
                    def close():
                        try:
                            cancel()
                        except Exception:
                            pass
                    threading.Thread(target=close, daemon=True).start()
                if self.terminated:
                    raise _ClientTerminated("Provider operation cancelled")
                raise _ProviderWallclockTimeout("Provider wall-clock deadline exceeded")
            try:
                ok, value = results.get(timeout=min(0.05, remaining))
            except Empty:
                continue
            if ok:
                return value
            raise value

    def reconcile_background_response(self, response_id, *, timeout=15.0, on_response=None):
        """Retrieve, then boundedly cancel an existing response; never create one."""
        if self.api != "openai" or not self.use_openai_responses_api:
            raise ValueError("Background reconciliation requires OpenAI Responses")

        def reconcile():
            deadline = time.monotonic() + timeout
            terminal = {"completed", "failed", "cancelled", "incomplete"}
            with OpenAI(api_key=self.api_key, base_url=self.base_url,
                        timeout=min(3.0, timeout), max_retries=0) as client:
                data = {}
                for operation in ("retrieve", "retrieve", "cancel", "retrieve"):
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    if operation == "cancel" and data.get("status") in terminal:
                        continue
                    kwargs = {"timeout": min(3.0, left)}
                    if operation == "retrieve":
                        kwargs["include"] = ["code_interpreter_call.outputs"]
                    try:
                        response = getattr(client.responses, operation)(response_id, **kwargs)
                    except Exception:
                        continue
                    data = self._provider_response_dict(response)
                    if data.get("id") != response_id:
                        raise ValueError("Provider returned a different response during reconciliation")
                    if on_response is not None:
                        on_response(data)
                    if data.get("status") in terminal and data.get("usage") is not None:
                        return data
                return data

        return self._bounded_provider_operation(reconcile, timeout=timeout)

    def attach_code_interpreter_file(self, path, *, timeout=120.0):
        """Upload an expiring snapshot and attach it to this client's Python tool.

        Uses the same credentials/base URL as model requests. This is storage
        I/O only; model usage remains accounted by run_queries/ProviderTrace.
        """
        if self.api != "openai" or not self.use_openai_responses_api:
            raise ValueError("File-backed critic notes require OpenAI Responses with Code Interpreter")
        python_tools = [desc for desc in self.tool_descriptions if desc.get("type") == "code_interpreter"]
        container = python_tools[0].get("container") if len(python_tools) == 1 else None
        if not isinstance(container, dict) or container.get("type") != "auto":
            raise ValueError("File-backed notes require exactly one automatic Code Interpreter container")

        def upload():
            with OpenAI(api_key=self.api_key, base_url=self.base_url,
                        timeout=timeout, max_retries=0) as client:
                with open(path, "rb") as source:
                    return client.files.create(
                        file=source, purpose="user_data",
                        # extra_body also supports SDKs predating the typed option.
                        extra_body={"expires_after": {"anchor": "created_at", "seconds": 172800}},
                    )

        uploaded = self._bounded_provider_operation(upload, timeout=timeout)
        if not uploaded.id:
            raise ValueError("Notes upload returned no file ID")
        python_tools[0]["container"] = {"type": "auto", "file_ids": [uploaded.id]}
        return uploaded.id

    def discard_code_interpreter_container(self, container_id):
        """Best-effort cleanup of an owned container that never reached a model."""
        def discard():
            try:
                with OpenAI(api_key=self.api_key, base_url=self.base_url,
                            timeout=5.0, max_retries=0) as client:
                    client.containers.delete(container_id)
            except Exception as exc:
                logger.warning("Unused notes container cleanup failed: {}", type(exc).__name__)

        # Cleanup must also work after terminate(), without extending shutdown.
        threading.Thread(target=discard, daemon=True).start()

    def create_code_interpreter_container_with_file(self, path, *, timeout=120.0):
        """Bind notes to an explicit container whose ID is known before reasoning."""
        if self.api != "openai" or not self.use_openai_responses_api:
            raise ValueError("File-backed critic notes require OpenAI Responses with Code Interpreter")
        python_tools = [desc for desc in self.tool_descriptions if desc.get("type") == "code_interpreter"]
        container = python_tools[0].get("container") if len(python_tools) == 1 else None
        if not isinstance(container, dict) or container.get("type") != "auto":
            raise ValueError("File-backed notes require exactly one unbound Code Interpreter container")
        deadline = time.monotonic() + timeout

        def remaining():
            left = deadline - time.monotonic()
            if left <= 0 or self.terminated:
                raise TimeoutError("Notes container setup deadline exceeded")
            return left

        def create():
            container_id = None
            try:
                with OpenAI(api_key=self.api_key, base_url=self.base_url,
                            timeout=remaining(), max_retries=0) as client:
                    created = client.containers.create(
                        name="critic-notes", expires_after={"anchor": "last_active_at", "minutes": 20},
                        timeout=remaining())
                    container_id = created.id
                    if not container_id:
                        raise ValueError("Notes container creation returned no ID")
                    with open(path, "rb") as source:
                        uploaded = client.containers.files.create(
                            container_id, file=(os.path.basename(path), source), timeout=remaining())
                    if not uploaded.id:
                        raise ValueError("Notes upload returned no file ID")
                    return container_id, uploaded.id
            except BaseException:
                if container_id:
                    self.discard_code_interpreter_container(container_id)
                raise

        container_id, file_id = self._bounded_provider_operation(
            create, timeout=timeout,
            on_abandoned_result=lambda result: self.discard_code_interpreter_container(result[0]))
        python_tools[0]["container"] = container_id
        return container_id, file_id

    def touch_code_interpreter_container(self, container_id, *, timeout=15.0):
        """Refresh an existing container's idle timer without a model request."""
        if self.api != "openai" or not self.use_openai_responses_api:
            raise ValueError("Container keepalive requires OpenAI Responses")

        def touch():
            with OpenAI(api_key=self.api_key, base_url=self.base_url,
                        timeout=timeout, max_retries=0) as client:
                return client.containers.retrieve(container_id).status

        return self._bounded_provider_operation(touch, timeout=timeout)

    def read_code_interpreter_file(self, container_id, path, *, timeout=15.0, max_bytes=4096):
        """Read one bounded artifact from an existing container; never run a model."""
        if self.api != "openai" or not self.use_openai_responses_api:
            raise ValueError("Container file retrieval requires OpenAI Responses")
        if not container_id or not path.startswith("/mnt/data/") or max_bytes <= 0:
            raise ValueError("Invalid container artifact request")

        def download():
            deadline = time.monotonic() + timeout

            def remaining():
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("Container artifact retrieval deadline exceeded")
                return min(5.0, left)

            with OpenAI(api_key=self.api_key, base_url=self.base_url,
                        timeout=remaining(), max_retries=0) as client:
                matches, after = [], None
                for _ in range(10):
                    kwargs = {"limit": 100, "timeout": remaining()}
                    if after is not None:
                        kwargs["after"] = after
                    page = client.containers.files.list(container_id, **kwargs)
                    matches.extend(item for item in page.data if item.path == path)
                    if page.has_more is False:
                        break
                    if page.has_more is not True:
                        raise ValueError("Incomplete container artifact listing")
                    if not page.data or page.data[-1].id == after:
                        raise ValueError("Incomplete container artifact listing")
                    after = page.data[-1].id
                else:
                    raise ValueError("Container artifact listing exceeded its limit")
                if len(matches) != 1 or not matches[0].id:
                    raise ValueError("Container artifact missing or ambiguous")
                if matches[0].container_id != container_id:
                    raise ValueError("Container artifact belongs to a different container")
                # Provider-created files can omit their size. The stream limit
                # below remains authoritative even when metadata is absent.
                declared_bytes = getattr(matches[0], "bytes", None)
                if declared_bytes is not None and declared_bytes > max_bytes:
                    raise ValueError("Container artifact exceeded its size limit")
                file_id = matches[0].id
                data = bytearray()
                with client.containers.files.content.with_streaming_response.retrieve(
                        file_id, container_id=container_id, timeout=remaining()) as response:
                    for chunk in response.iter_bytes(chunk_size=max_bytes + 1):
                        remaining()
                        if len(data) + len(chunk) > max_bytes:
                            raise ValueError("Container artifact exceeded its size limit")
                        data.extend(chunk)
                return {"container_id": container_id, "file_id": file_id,
                        "path": path, "content": data.decode("utf-8")}

        return self._bounded_provider_operation(download, timeout=timeout)

    """A client that queries various LLM APIs."""

    def __init__(
        self,
        model,
        timeout=30000,
        max_tokens=None,
        api="openai",
        api_key_env=None,
        base_url=None,
        max_retries=3,
        max_retries_inner=25,
        max_wallclock_per_call_s=600.0,
        concurrent_requests=30,
        no_system_messages=False,
        context_limit=None,
        background=False,
        read_cost=1,
        cache_read_cost=None,
        write_cost=1,
        sleep_on_error=60,
        sleep_on_error_max=600,
        sleep_after_request=0.1,
        include_max_tool_calls=True,
        throw_error_on_failure=False,
        max_tokens_param="max_tokens",
        reasoning_effort=None,
        batch_processing=False,
        use_openai_responses_api=False,
        use_gdm_tools=False,
        stream_openai_chat_completions=False,
        stream_anthropic_messages=False,
        anthropic_betas=None,
        anthropic_salvage_empty_max_tokens=False,
        anthropic_continue_on_max_tokens=False,
        anthropic_max_token_continuations=3,
        anthropic_max_server_tool_continuations=5,
        max_tool_calls=float("inf"),
        max_hosted_tool_calls=None,
        required_hosted_tool_types=(),
        required_hosted_tool_validator=None,
        tool_wrapup_reserve_s=0,
        cache_write_cost=0,
        cache_write_tokens_in_input=False,
        long_context_threshold_tokens=None,
        long_context_input_multiplier=1,
        long_context_output_multiplier=1,
        background_timeout_downgrade_after=0,
        background_timeout_reasoning_efforts=None,
        background_server_kill_after_s=None,
        openai_pro_fallback_after_failures=0,
        openai_continue_on_max_output_tokens=False,
        openai_max_output_token_continuations=1,
        openai_max_output_token_continuation_effort=None,
        tools=None,
        **kwargs,
    ):
        """Initializes the APIClient object and params. All prompts are set at run_queries invocation.

        Args:
            model (str): The name of the model to use.
            timeout (int, optional): The timeout for API requests in seconds. Defaults to 9000.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to None.
            api (str, optional): The API to use. Defaults to 'openai'.
            max_retries (int, optional): The maximum number of retries for a failed query. Defaults to 50.
            concurrent_requests (int, optional): The number of concurrent requests to make. Defaults to 30.
            no_system_messages (bool, optional): Whether to disable system messages. Defaults to False.
            read_cost (int, optional): The cost of reading a token. Defaults to 1.
            cache_read_cost (int, optional): The cost of reading a cached input token. Defaults to read_cost.
            write_cost (int, optional): The cost of writing a token. Defaults to 1.
            sleep_on_error (int, optional): The number of seconds to sleep on an error. Defaults to 60.
            sleep_after_request (float, optional): The number of seconds to sleep after a request. Defaults to 0.1.
            throw_error_on_failure (bool, optional): Whether to throw an error on failure. Defaults to False.
            max_tokens_param (str, optional): The name of the max_tokens parameter for the API. Defaults to "max_tokens".
            reasoning_effort (str, optional): The reasoning effort to use. Defaults to None.
            batch_processing (bool, optional): Whether to use batch processing. Defaults to False.
            use_openai_responses_api (bool, optional): Whether to use OpenAI responses. Defaults to False.
            stream_openai_chat_completions (bool, optional): Whether to stream OpenAI chat completions internally.
            stream_anthropic_messages (bool, optional): Whether to use Anthropic's streaming Messages API.
            anthropic_betas (list[str], optional): Anthropic beta headers to pass through
                ``client.beta.messages``. Required for Files API ``container_upload``.
            anthropic_salvage_empty_max_tokens (bool, optional): Whether to follow up an
                Anthropic max-token response with no visible text by asking for only the final answer.
            anthropic_continue_on_max_tokens (bool, optional): Whether to continue a visible
                Anthropic response that stopped at max_tokens and stitch the visible chunks.
            anthropic_max_token_continuations (int, optional): Maximum number of visible
                max_tokens continuation turns.
            anthropic_max_server_tool_continuations (int, optional): Maximum number of
                Anthropic pause_turn continuations for server tools.
            max_tool_calls (int|float|dict, optional): The maximum number of tool calls to make.
                Defaults to unlimited.
                Could also be a dict that specifies max calls per tool name.
            max_hosted_tool_calls (int, optional): Total hosted-tool allowance across
                an OpenAI Responses query and its continuations/retries. Zero disables
                hosted tools; None leaves them uncapped. Local functions are separate.
            required_hosted_tool_types (iterable[str], optional): Fail rather than
                submit a tool-less retry if mandatory hosted work has not yet
                produced a completed tool item. Does not certify that item's result.
            required_hosted_tool_validator (callable, optional): A stricter local
                evidence gate for satisfying mandatory hosted work. No API calls.
            cache_write_tokens_in_input (bool, optional): Whether reported cache-write
                tokens are already included in the input-token total.
            long_context_threshold_tokens (int, optional): Input-token threshold above
                which the configured long-context multipliers apply to the full request.
            long_context_input_multiplier (float, optional): Long-context multiplier for
                all input categories, including cache reads and writes.
            long_context_output_multiplier (float, optional): Long-context output multiplier.
            background_timeout_downgrade_after (int, optional): Number of OpenAI Responses
                background poll timeouts before using background_timeout_reasoning_efforts.
            background_timeout_reasoning_efforts (list[str], optional): Reasoning efforts
                to use on subsequent OpenAI Responses background retries after timeout.
            background_server_kill_after_s (float, optional): Treat an OpenAI Responses
                background response that ends ``status=failed`` with no output after at
                least this many seconds of polling as a server-side duration kill, i.e.
                like a background timeout (so the effort downgrade applies to the retry).
            openai_pro_fallback_after_failures (int, optional): Retryable failed Pro
                requests before switching this query to reasoning.mode=standard.
                Zero disables fallback. New independent queries start in Pro again.
            openai_continue_on_max_output_tokens (bool, optional): When an OpenAI
                Responses call ends ``status=incomplete`` with reason ``max_output_tokens``,
                send one follow-up turn asking the model to wrap up instead of accepting
                the truncated output.
            openai_max_output_token_continuations (int, optional): Maximum number of such
                wrap-up turns per call.
            openai_max_output_token_continuation_effort (str, optional): Reasoning effort
                for the wrap-up turn(s); None keeps the call's configured effort.
            tools (list, optional): A list of tools to use. Defaults to None.
            **kwargs: Additional keyword arguments for the API.
        """
        if not isinstance(cache_write_tokens_in_input, bool):
            raise ValueError("cache_write_tokens_in_input must be a boolean")
        if cache_write_tokens_in_input:
            try:
                cache_write_cost = float(cache_write_cost)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "cache_write_cost must be positive when cache_write_tokens_in_input is true"
                ) from e
            if cache_write_cost <= 0:
                raise ValueError(
                    "cache_write_cost must be positive when cache_write_tokens_in_input is true"
                )
        if long_context_threshold_tokens is not None:
            long_context_threshold_tokens = int(long_context_threshold_tokens)
            if long_context_threshold_tokens <= 0:
                raise ValueError("long_context_threshold_tokens must be positive")
        long_context_input_multiplier = float(long_context_input_multiplier)
        long_context_output_multiplier = float(long_context_output_multiplier)
        if long_context_input_multiplier <= 0 or long_context_output_multiplier <= 0:
            raise ValueError("long-context cost multipliers must be positive")

        # Max tool calls
        has_local_tools = any(func is not None for func, _ in (tools or []))
        max_tool_calls = self._normalize_max_tool_calls(max_tool_calls)
        self.tool_calls_allowed = False
        if isinstance(max_tool_calls, (int, float)):
            self.max_tool_calls = {"any": max_tool_calls}
            self.max_tool_calls_mode = "total"
            if has_local_tools and max_tool_calls > 0:
                self.tool_calls_allowed = True
        elif isinstance(max_tool_calls, dict):
            self.max_tool_calls = {str(k): self._normalize_max_tool_calls(v) for k, v in max_tool_calls.items()}
            self.max_tool_calls_mode = "per_tool"
            if has_local_tools and sum(self.max_tool_calls.values()) > 0:
                self.tool_calls_allowed = True
        if max_hosted_tool_calls is not None:
            if isinstance(max_hosted_tool_calls, bool) or not isinstance(max_hosted_tool_calls, int) or max_hosted_tool_calls < 0:
                raise ValueError("max_hosted_tool_calls must be a nonnegative integer or None")
            if api not in {"openai", "openrouter", "together", "custom"} or not use_openai_responses_api or batch_processing:
                raise ValueError("max_hosted_tool_calls requires the OpenAI Responses API without batch processing")
        self.max_hosted_tool_calls = max_hosted_tool_calls
        if isinstance(required_hosted_tool_types, str) or not isinstance(required_hosted_tool_types, (list, tuple, set, frozenset)):
            raise ValueError("required_hosted_tool_types must be a collection of tool types")
        if any(kind not in {"code_interpreter", "web_search", "web_search_preview"} for kind in required_hosted_tool_types):
            raise ValueError("Unsupported required hosted tool type")
        self.required_hosted_tool_types = frozenset(required_hosted_tool_types)
        if required_hosted_tool_validator is not None and not callable(required_hosted_tool_validator):
            raise ValueError("required_hosted_tool_validator must be callable")
        self.required_hosted_tool_validator = required_hosted_tool_validator

        # Adapt model name and other args to the model
        if "--" in model:
            model, reasoning_effort = model.split("--")
            logger.info(f"Model: {model}, Reasoning effort: {reasoning_effort}")
        if (api not in ["anthropic", "openai"] or self.tool_calls_allowed) and batch_processing:
            logger.warning("Batch processing is only supported for the Anthropic API and OpenAI API without tool calling.")
            batch_processing = False
        is_astra = model == "gpt-6-astra" or model.startswith("gpt-6-astra-")
        if api == "openai" and is_astra and (not use_openai_responses_api or batch_processing):
            if tools or (kwargs.get("reasoning") or {}).get("mode") == "pro":
                raise ValueError("GPT-6 Astra tools and Pro mode require the Responses API with batch_processing=False.")
        if ("o1" in model or "o3" in model or "o4" in model or "gpt-5" in model or is_astra) and api == "openai":
            logger.info(f"Preserving developer messages for {model}.")
            no_system_messages = True
            if not use_openai_responses_api:
                max_tokens_param = "max_completion_tokens"
        if use_openai_responses_api and not batch_processing:
            max_tokens_param = "max_output_tokens"
        if self.tool_calls_allowed and not use_openai_responses_api and not api == "anthropic":
            max_tokens_param = "max_completion_tokens"
        self._kwarg_remover(api, model, kwargs)

        self.model = model
        self.kwargs = kwargs
        self.max_tokens_param = max_tokens_param
        self.context_limit = context_limit
        self.max_tokens = max_tokens
        if max_tokens is not None:
            self.kwargs[max_tokens_param] = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retries_inner = max_retries_inner
        self.max_wallclock_per_call_s = max_wallclock_per_call_s
        self.tool_wrapup_reserve_s = float(tool_wrapup_reserve_s)
        if not math.isfinite(self.tool_wrapup_reserve_s) or self.tool_wrapup_reserve_s < 0:
            raise ValueError("tool_wrapup_reserve_s must be finite and nonnegative")
        self.throw_error_on_failure = throw_error_on_failure
        self.concurrent_requests = concurrent_requests
        self.no_system_messages = no_system_messages
        self.sleep_on_error = sleep_on_error
        self.sleep_on_error_max = sleep_on_error_max
        self.sleep_after_request = sleep_after_request
        self.read_cost = read_cost
        self.cache_read_cost = read_cost if cache_read_cost is None else cache_read_cost
        self.write_cost = write_cost
        self.background = background
        self.batch_processing = batch_processing
        self.use_openai_responses_api = use_openai_responses_api
        self.use_gdm_tools = use_gdm_tools
        self.use_google_internal_tools = False
        self.stream_openai_chat_completions = stream_openai_chat_completions
        self.stream_anthropic_messages = stream_anthropic_messages
        self.anthropic_betas = list(anthropic_betas or [])
        self.anthropic_salvage_empty_max_tokens = anthropic_salvage_empty_max_tokens
        self.anthropic_continue_on_max_tokens = anthropic_continue_on_max_tokens
        self.anthropic_max_token_continuations = max(0, int(anthropic_max_token_continuations or 0))
        self.anthropic_max_server_tool_continuations = max(0, int(anthropic_max_server_tool_continuations or 0))
        self.include_max_tool_calls = include_max_tool_calls
        self.cache_write_cost = cache_write_cost
        self.cache_write_tokens_in_input = cache_write_tokens_in_input
        self.long_context_threshold_tokens = long_context_threshold_tokens
        self.long_context_input_multiplier = long_context_input_multiplier
        self.long_context_output_multiplier = long_context_output_multiplier
        self.background_timeout_downgrade_after = max(0, int(background_timeout_downgrade_after or 0))
        self.background_timeout_reasoning_efforts = list(background_timeout_reasoning_efforts or [])
        self.background_server_kill_after_s = (
            None if background_server_kill_after_s is None else float(background_server_kill_after_s)
        )
        self.openai_pro_fallback_after_failures = max(0, int(openai_pro_fallback_after_failures or 0))
        self.openai_continue_on_max_output_tokens = bool(openai_continue_on_max_output_tokens)
        self.openai_max_output_token_continuations = max(0, int(openai_max_output_token_continuations or 0))
        self.openai_max_output_token_continuation_effort = openai_max_output_token_continuation_effort
        self.background = background
        if max_tokens is not None:
            self.max_tokens_param = max_tokens_param
        if reasoning_effort is not None:
            if not self.use_openai_responses_api or self.batch_processing:
                self.kwargs["reasoning_effort"] = reasoning_effort
            elif "reasoning" in self.kwargs:
                self.kwargs["reasoning"]["effort"] = reasoning_effort
            else:
                self.kwargs["reasoning"] = {"effort": reasoning_effort}

        # Save tools: user should forward all (even if mix of competition-given and scaffold-given)
        self.tools = tools if tools is not None else []
        self.tool_functions = {
            tool_desc["function"]["name"]: func for func, tool_desc in self.tools if "function" in tool_desc
        }
        self.tool_descriptions = [tool_desc for _, tool_desc in self.tools]
        if (not self.tool_calls_allowed or len(self.tool_descriptions) == 0) and "tool_choice" in self.kwargs:
            del self.kwargs["tool_choice"]

        # Prep api
        self.api = api
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.api_key = None
        self.terminated = False
        self._initialize_api_keys()

        
        # VLLM-specific initialization
        if self.api == "vllm":
            self.vllm_model = None

    def _background_timeout_retry_reasoning_effort(self, timeout_count):
        if timeout_count < self.background_timeout_downgrade_after:
            return None
        if not self.background_timeout_reasoning_efforts:
            return None
        idx = timeout_count - self.background_timeout_downgrade_after
        idx = min(idx, len(self.background_timeout_reasoning_efforts) - 1)
        return self.background_timeout_reasoning_efforts[idx]

    def _kwargs_for_background_timeout_retry(self, timeout_count):
        return self._kwargs_with_reasoning_effort(
            self._background_timeout_retry_reasoning_effort(timeout_count)
        )

    def _kwargs_with_reasoning_effort(self, effort):
        if effort is None:
            return self.kwargs
        kwargs = self.kwargs.copy()
        reasoning = dict(kwargs.get("reasoning") or {})
        reasoning["effort"] = effort
        kwargs["reasoning"] = reasoning
        return kwargs

    def _responses_request_kwargs(self, state, *, wrapup=False):
        if wrapup and self.openai_max_output_token_continuation_effort is not None:
            kwargs = self._kwargs_with_reasoning_effort(self.openai_max_output_token_continuation_effort)
        else:
            kwargs = self._kwargs_for_background_timeout_retry(state.background_timeouts)
        if (
            self.openai_pro_fallback_after_failures
            and state.pro_failures >= self.openai_pro_fallback_after_failures
            and (kwargs.get("reasoning") or {}).get("mode") == "pro"
        ):
            kwargs = {**kwargs, "reasoning": {**kwargs["reasoning"], "mode": "standard"}}
        return kwargs

    def _record_pro_failure(self, state, *, pro_attempt, idx):
        if not pro_attempt or not self.openai_pro_fallback_after_failures:
            return
        state.pro_failures += 1
        if state.pro_failures == self.openai_pro_fallback_after_failures:
            logger.warning(
                f"OpenAI Pro fallback on call idx={idx}: {state.pro_failures} failed Pro attempts; "
                f"subsequent requests use {self.model} reasoning.mode=standard within the existing deadline."
            )

    def _is_background_server_kill(self, err_code, elapsed_s):
        """OpenAI kills long sol/astra background calls at ~60 min with
        ``status=failed``, ``error.code=server_error``, empty output and
        ``usage=None``. Classify that like a background timeout so the
        retry runs at the downgraded effort instead of dying again."""
        if self.background_server_kill_after_s is None:
            return False
        if str(err_code or "server_error").lower() != "server_error":
            return False
        return elapsed_s >= self.background_server_kill_after_s

    def _initialize_vllm(self):
            if LLM is None:
                raise ImportError("vllm is not installed. pip install vllm")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            vllm_args = {}

            for p in ("temperature", "top_p", "max_tokens", "top_k", "repetition_penalty", "presence_penalty"):
                if p in self.kwargs:
                    vllm_args[p] = self.kwargs.pop(p)
            self.sampling_params = SamplingParams(**vllm_args)
            self.vllm_model = LLM(
                model=self.model, tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
            )
            logger.info(f"Loaded local vllm model `{self.model}` with sampling {self.kwargs}")

    def terminate(self):
        """Terminates the APIClient."""
        self.terminated = True

    @staticmethod
    def _normalize_max_tool_calls(value):
        if value is None:
            return float("inf")
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in {"inf", "+inf", "infinity", "+infinity", "unlimited"}:
                return float("inf")
            if cleaned == "":
                return float("inf")
            return int(cleaned)
        return value

    def _kwarg_remover(self, api, model, kwargs):
        """Removes kwargs that are not supported by the API or model.

        Args:
            api (str): The API to use.
            model (str): The model to use.
            kwargs (dict): The kwargs to clean.
        """
        if "use_openai_responses_api_tools" in kwargs:
            del kwargs["use_openai_responses_api_tools"]
        if any([kw in model for kw in ["o1", "o3", "o4"]]) and "temperature" in kwargs:
            del kwargs["temperature"]
        if api == "openai" and (model == "gpt-6-astra" or model.startswith("gpt-6-astra-")):
            for kwarg in ("temperature", "top_p", "top_logprobs", "logprobs"):
                kwargs.pop(kwarg, None)
            if kwargs.get("include") is not None:
                kwargs["include"] = [
                    item for item in kwargs["include"] if item != "message.output_text.logprobs"
                ]
        for kwarg in ["top_p", "top_k", "temperature"]:
            if kwarg in kwargs and kwargs[kwarg] is None:
                del kwargs[kwarg]
        if (api == "anthropic" and "claude-3-7" in model) or (("o1" in model or "o3" in model) and api == "openai"):
            for kwarg_to_remove in ["top_p", "top_k", "temperature"]:
                if kwarg_to_remove in kwargs:
                    logger.info(f"Removing {kwarg_to_remove} parameter for {model} model.")
                    del kwargs[kwarg_to_remove]

    def _initialize_api_keys(self):
        """Initializes the API keys and base URLs for the selected API."""
        if self.api == "sri":
            self.api_key = os.getenv("SRI_API_KEY")
            self.base_url = "https://srlx.inf.ethz.ch/openai"
            self.api = "openai"
        elif self.api == "xai":
            self.api_key = os.getenv("XAI_API_KEY")
            self.base_url = "https://api.x.ai/v1"
            self.api = "openai"
        elif self.api == "stepfun":
            self.api_key = os.getenv("STEPFUN_API_KEY")
            self.base_url = "https://api.stepfun.ai/v1"
            self.api = "openai"
        elif self.api == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif self.api == "together":
            self.api_key = os.getenv("TOGETHER_API_KEY")
            self.base_url = "https://api.together.xyz/v1"
        elif self.api == "google":
            self.api_key = os.getenv("GOOGLE_API_KEY")
            # The native generateContent path records but does not execute
            # local function tools, so ``use_gdm_tools`` only routes calls
            # whose tools are all provider-managed (codeExecution /
            # googleSearch); local function tools keep the Chat Completions
            # tool loop. The legacy BCN model keeps its original routing.
            provider_tools_only = bool(self.tool_descriptions) and not self.tool_calls_allowed
            has_provider_managed_tools = any(
                desc.get("type") in ("code_interpreter", "web_search_preview", "web_search")
                for desc in self.tool_descriptions
            )
            if self.use_gdm_tools and has_provider_managed_tools and self.tool_calls_allowed:
                raise ValueError(
                    "Gemini provider-managed tools (code_interpreter / web_search) cannot be "
                    "combined with local function tools: the native generateContent path does "
                    "not execute local functions and the OpenAI-compatible endpoint rejects the "
                    "provider tools."
                )
            if (self.tool_calls_allowed and "gdm-eval-model-bcn" in self.model) or (
                self.use_gdm_tools and provider_tools_only
            ):
                self.use_google_internal_tools = True
                self.api = "google"
                self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            else:
                self.api = "openai"  # !
                self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif self.api == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif self.api == "glm":
            self.api_key = os.getenv("GLM_API_KEY")
            self.base_url = "https://api.z.ai/api/paas/v4/"
            self.api = "openai"
        elif self.api == "tiiuae":
            self.api_key = os.getenv("TIIUAE_API_KEY")
            self.base_url = "https://falcon-stage.blueoc.tech/v1"
            self.api = "openai"
        elif self.api == "moonshot":
            self.api_key = os.getenv("MOONSHOT_API_KEY")
            self.base_url = "https://api.moonshot.ai/v1"
            self.api = "openai"
        elif self.api == "deepseek":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.base_url = "https://api.deepseek.com"
            self.api = "openai"
        elif self.api == "deepseek_special":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.base_url = "https://api.deepseek.com/v3.2_speciale_expires_on_20251215"
            self.api = "openai"
        elif self.api == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
            if "via_openai" in self.kwargs:
                del self.kwargs["via_openai"]
                self.api = "openai"
        elif self.api == "custom":
            self.api = "openai"
            self.base_url = self.base_url
            self.api_key = os.getenv(self.api_key_env) if self.api_key_env is not None else "EMPTY"
        elif self.api == "vllm":
            return
        else:
            raise ValueError(f"API {self.api} not supported.")


        assert self.api_key is not None, "API key not found."

    class InternalRequestResult:
        """A class to hold the result of a request internally (below run_queries)."""

        def __init__(self, conversation, input_tokens, output_tokens, cached_input_tokens=0, cached_write_tokens=0, reasoning_tokens=0, n_retries=0, time=0, cost_usd=None):
            self.conversation = conversation
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.cached_input_tokens = cached_input_tokens
            self.cached_write_tokens = cached_write_tokens
            # First Proof spec requires per-call reasoning token counts.
            # Provider-specific extraction lives in ``_extract_reasoning_tokens``.
            # Defaults to 0 for paths where no usage was reported (e.g.
            # request timeouts) or providers that don't expose it.
            self.reasoning_tokens = reasoning_tokens
            self.n_retries = n_retries
            self.time = time
            self.cost_usd = cost_usd

    def run_queries(self, queries, no_tqdm=False, ignore_tool_calls=False, custom_indices=None):
        """Only entry point: runs a given list of queries through the API.

        Args:
            queries (list[MessageList]): A list of queries to run. Each query is a MessageList that we will mangle into
            the right format for this API.
            no_tqdm (bool, optional): Whether to disable the tqdm progress bar. Defaults to False.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction. Defaults to False.

        Yields:
            tuple: An (idx, conversation, detailed_cost) tuple.
                idx: Query index, or its supplied custom_indices value.
                conversation: Full list of messages (including those from the query) in the API format (incl. CoT).
                detailed_cost: A dict with total "cost" ($), "input_tokens", "output_tokens", and "time" (seconds).
        """
        if not no_tqdm:
            logger.info(f"Running {len(queries)} queries.")

        # For now only switches between system/developer, keeps rest intact
        queries = [self._validate_and_prepare_query(query) for query in queries]

        # Prepare batch indices, agents use custom ones for request_logger
        if custom_indices is not None:
            indices = custom_indices
        else:
            indices = list(range(len(queries)))

        # Case 1: VLLM
        if self.api == "vllm":
            # Bypass threading and batch everything into one local generate
            # TODO request logger
            for idx, conversation, detailed in self._run_vllm_queries(queries):
                yield indices[idx], conversation, detailed
            return

        # Case 2: Batch API
        if self.batch_processing:
            start_time = time.time()
            if self.api == "openai":
                results_batch = self._openai_batch_processing(queries, indices)
            else:
                results_batch = self._anthropic_batch_processing(queries, indices)
            end_time = time.time()  # pack all time into first index (since batch)
            for idx, result in enumerate(results_batch):
                if result is None:
                    conversation = [m.copy() for m in queries[idx]] + [{"role": "assistant", "content": ""}]
                    result = self.InternalRequestResult(conversation, input_tokens=0, output_tokens=0)
                detailed_cost = {
                    "cost": result.cost_usd if result.cost_usd is not None else self._get_cost(
                        result.input_tokens, result.output_tokens, result.cached_input_tokens, result.cached_write_tokens
                    ),
                    "input_tokens": result.input_tokens,
                    "cached_input_tokens": result.cached_input_tokens,
                    "cached_write_tokens": result.cached_write_tokens,
                    "output_tokens": result.output_tokens,
                    "reasoning_tokens": result.reasoning_tokens,
                    "time": end_time - start_time,
                    "n_retries": result.n_retries,
                }
                yield indices[idx], result.conversation, detailed_cost
            return

        # Case 3: Standard API; parallelize manually
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
            future_to_query = {
                executor.submit(copy_context().run, self._run_query_with_retry, idx, query, ignore_tool_calls): (idx, query)
                for idx, query in zip(indices, queries)
            }

            iterator = as_completed(future_to_query)
            if not no_tqdm:
                iterator = tqdm(iterator, total=len(future_to_query))

            for future in iterator:
                idx, query = future_to_query[future]
                result = future.result()
                if result is None:
                    conversation = [m.copy() for m in query] + [{"role": "assistant", "content": ""}]
                    result = self.InternalRequestResult(conversation, input_tokens=0, output_tokens=0)
                detailed_cost = {
                    "cost": result.cost_usd if result.cost_usd is not None else self._get_cost(
                        result.input_tokens,
                        result.output_tokens,
                        result.cached_input_tokens,
                        result.cached_write_tokens,
                    ),
                    "input_tokens": result.input_tokens,
                    "cached_input_tokens": result.cached_input_tokens,
                    "cached_write_tokens": result.cached_write_tokens,
                    "output_tokens": result.output_tokens,
                    "reasoning_tokens": result.reasoning_tokens,
                    "n_retries": result.n_retries,
                    "time": time.time() - start_time,
                    "request_time": result.time,
                    "provider_outcomes": getattr(result, "provider_outcomes", []),
                    "usage_unavailable": getattr(result, "usage_unavailable", False),
                }
                yield idx, result.conversation, detailed_cost

    def _validate_and_prepare_query(self, query):
        """Prepares a query for the API.
            All "tool_response" and "assistant" blocks must have come straight from this APIClient
                => We only need to normalize the developer and user messages
            We will assume they arrive in normalized format (only role: "user"/"developer" and content:str fields)

        Args:
            query (MessageList): List of messages to prepare.

        Returns:
            query_prepared (MessageList): The prepared conversation in the format for this API.
        """
        query_prepared = []
        for m in query:
            query_prepared.append(m.copy())
            if m.get("role", "") == "developer" and not self.no_system_messages:
                query_prepared[-1]["role"] = "system"  # use system if expected by API
            if m.get("role", "") == "user":
                check_for_extra_keys(m, ["role", "content", "tool_context"])

            # Fix images into another format for gemini and grok and qwen and glm
            if (
                m.get("role", "") == "user"
                and isinstance(m.get("content", ""), list)
                and (
                    "gemini-" in self.model
                    or "riftrunner" in self.model
                    or "gdm-eval" in self.model
                    or "grok-4" in self.model
                    or "qwen" in self.model
                    or "glm" in self.model
                    or "moonshot" in self.model
                )
            ):
                new_content = []
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type", "") == "input_image" and "image_url" in block:
                        inner = {"url": block["image_url"]}
                        if "grok-4" in self.model:
                            inner["detail"] = "high"
                        new_content.append({"type": "image_url", "image_url": inner})
                    elif isinstance(block, dict) and block.get("type", "") == "input_text" and "text" in block:
                        new_content.append({"type": "text", "text": block["text"]})
                    elif isinstance(block, dict) and block.get("type", "") in ["text", "image_url"]:
                        # Already transformed (e.g., during last_chance), keep as-is
                        new_content.append(block)
                    else:
                        # Unknown block type, log warning but keep the block to avoid data loss
                        logger.warning(f"Unknown content block type in user message: {block}")
                        new_content.append(block)
                query_prepared[-1]["content"] = new_content

            # Fix images for anthropic
            if m.get("role", "") == "user" and isinstance(m.get("content", ""), list) and self.api == "anthropic":
                new_content = []
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type", "") == "input_image" and "image_url" in block:
                        b64_full = block["image_url"]
                        tag = "data:image/png;base64,"
                        assert b64_full.startswith(tag)
                        source = {"type": "base64", "media_type": "image/png", "data": b64_full[len(tag) :]}
                        new_content.append({"type": "image", "source": source})
                    elif isinstance(block, dict) and block.get("type", "") == "input_text" and "text" in block:
                        new_content.append({"type": "text", "text": block["text"]})
                    elif isinstance(block, dict) and block.get("type", "") in ["text", "image", "container_upload"]:
                        # Already transformed (e.g., during last_chance), keep as-is
                        new_content.append(block)
                    else:
                        # Unknown block type, log warning but keep the block to avoid data loss
                        logger.warning(f"Unknown content block type in user message: {block}")
                        new_content.append(block)
                query_prepared[-1]["content"] = new_content
        return query_prepared

    def _usage_to_dict(self, usage):
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        # Fallback: any plain object that exposes token attributes
        # (covers ad-hoc usage stand-ins like ``_SalvageUsage`` and any
        # SDK / wrapper that returns a duck-typed object without
        # ``model_dump``). Without this fallback an object that lacks
        # ``model_dump`` silently bills as zero, masking budget
        # exhaustion on salvaged turns AND wiping reasoning-token
        # counts for any nested details object that does not get
        # surfaced.
        out: dict = {}
        for key in (
            # Standard token counts.
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "cached_input_tokens",
            "cached_tokens",
            "cache_write_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            # Reasoning / thinking flat fields. ``_extract_reasoning_tokens``
            # checks both snake_case and camelCase, so we preserve any
            # variant the wrapper happens to expose.
            "reasoning_tokens",
            "reasoning_out_tokens",
            "thinking_tokens",
            "thoughts_token_count",
            "thoughtsTokenCount",
        ):
            val = getattr(usage, key, None)
            if val is not None:
                out[key] = val
        # Keep nested details intact because both token extractors accept either
        # dictionaries or attribute-bearing SDK objects.
        for nested in (
            "input_tokens_details",
            "prompt_tokens_details",
            "output_tokens_details",
            "completion_tokens_details",
        ):
            sub = getattr(usage, nested, None)
            if sub is not None:
                out[nested] = sub
        return out

    def _extract_usage_tokens(self, usage):
        usage_dict = self._usage_to_dict(usage)
        input_tokens = usage_dict.get("input_tokens", usage_dict.get("prompt_tokens", 0)) or 0
        output_tokens = usage_dict.get("output_tokens")
        if output_tokens is None:
            completion_tokens = usage_dict.get("completion_tokens")
            if completion_tokens is not None and "total_tokens" not in usage_dict:
                output_tokens = completion_tokens
            else:
                total_tokens = usage_dict.get("total_tokens")
                output_tokens = (total_tokens - input_tokens) if total_tokens is not None else 0

        input_details = usage_dict.get("input_tokens_details", {})
        if not isinstance(input_details, dict):
            input_details = self._usage_to_dict(input_details)
        prompt_details = usage_dict.get("prompt_tokens_details", {})
        if not isinstance(prompt_details, dict):
            prompt_details = self._usage_to_dict(prompt_details)

        cached_input_tokens = (
            usage_dict.get("cached_input_tokens")
            or usage_dict.get("cache_read_input_tokens")
            or usage_dict.get("cached_tokens")
            or input_details.get("cached_tokens")
            or prompt_details.get("cached_tokens")
            or 0
        )
        if usage_dict.get("cache_read_input_tokens"):
            input_tokens += cached_input_tokens  # if cache_read_input_tokens is provided, Anthropic doesn't count those in input_tokens, so we add them back for consistency with other APIs.

        cached_input_tokens = min(max(cached_input_tokens, 0), max(input_tokens, 0))
        cached_creation_input_tokens = (
            usage_dict.get("cache_creation_input_tokens")
            or usage_dict.get("cache_write_tokens")
            or input_details.get("cache_write_tokens")
            or prompt_details.get("cache_write_tokens")
            or 0
        )
        return input_tokens, output_tokens, cached_input_tokens, cached_creation_input_tokens

    def _extract_reasoning_tokens(self, usage) -> int:
        """Best-effort extraction of provider-reported reasoning tokens.

        First Proof's spec requires logging input + output + reasoning
        tokens per API call. The standard token counts (input/output)
        are returned by ``_extract_usage_tokens``; the reasoning count
        lives under provider-specific fields:

          * OpenAI Responses API → ``output_tokens_details.reasoning_tokens``
          * OpenAI Chat Completions (o1/o3) → ``completion_tokens_details.reasoning_tokens``
          * Gemini → ``thinking_tokens`` or ``thoughts_token_count`` (if exposed)
          * Anthropic — thinking tokens are bundled into ``output_tokens``
            and not separately reported, so this returns 0 there.

        Returns 0 when no recognised field is present (so the caller
        can still emit a stable ``reasoning_tokens`` field).
        """
        usage_dict = self._usage_to_dict(usage)
        if not usage_dict:
            return 0

        def _coerce(value) -> int | None:
            if value is None:
                return None
            try:
                n = int(value)
            except (TypeError, ValueError):
                return None
            return max(n, 0)

        def _from_details(value) -> int | None:
            if value is None:
                return None
            if isinstance(value, dict):
                return _coerce(value.get("reasoning_tokens", value.get("thinking_tokens")))
            return _coerce(getattr(value, "reasoning_tokens", getattr(value, "thinking_tokens", None)))

        # OpenAI Responses + Chat-Completions reasoning fields live under
        # different nested objects; check both.
        for nested_key in ("output_tokens_details", "completion_tokens_details"):
            n = _from_details(usage_dict.get(nested_key))
            if n is not None:
                return n

        # Gemini / top-level fallbacks. ``reasoning_out_tokens`` is the
        # name our own Compute Worker code uses when it parses codex
        # JSONL output. Google's native API returns camelCase
        # ``thoughtsTokenCount``; the snake_case alias is for
        # third-party Gemini wrappers that re-key responses.
        for top_key in (
            "thinking_tokens",
            "thoughts_token_count",
            "thoughtsTokenCount",
            "reasoning_tokens",
            "reasoning_out_tokens",
        ):
            n = _coerce(usage_dict.get(top_key))
            if n is not None:
                return n

        return 0

    def _get_cost(self, input_tokens, output_tokens, cached_input_tokens=0, cached_creation_input_tokens=0):
        input_tokens = max(input_tokens, 0)
        output_tokens = max(output_tokens, 0)
        cached_input_tokens = min(max(cached_input_tokens, 0), input_tokens)
        uncached_input_tokens = input_tokens - cached_input_tokens
        if self.cache_write_tokens_in_input:
            cached_creation_input_tokens = min(
                max(cached_creation_input_tokens, 0),
                uncached_input_tokens,
            )
            uncached_input_tokens -= cached_creation_input_tokens
        long_context = (
            self.long_context_threshold_tokens is not None
            and input_tokens > self.long_context_threshold_tokens
        )
        input_multiplier = self.long_context_input_multiplier if long_context else 1
        output_multiplier = self.long_context_output_multiplier if long_context else 1
        return (
            (
                uncached_input_tokens * self.read_cost
                + cached_input_tokens * self.cache_read_cost
                + cached_creation_input_tokens * self.cache_write_cost
            )
            * input_multiplier
            + output_tokens * self.write_cost * output_multiplier
        ) / 1e6

    def _get_messages_from_anthropic_content(self, content):
        """Postprocesses the content from an Anthropic API query.

        Args:
            content: The content from the Anthropic API.

        Returns:
            str: The textual representation.
        """
        messages = []
        for content_block in content:
            if isinstance(content_block, ThinkingBlock):
                messages.append({"role": "assistant", "type": "reasoning", "content": content_block.thinking})
            elif isinstance(content_block, TextBlock):
                messages.append({"role": "assistant", "type": "response", "content": content_block.text})
                break
        return messages

    def _drop_cot(self, messages):
        """Drops all CoT/thinking/reasoning messages from a conversation.
        This is a cost saving measure at API call site; conversations that are maintained will have this.

        Args:
            messages (MessageList): The conversation to drop CoT from.
        Returns:
            MessageList: The conversation without CoT messages.
        """
        new_messages = []
        for m in messages:
            if m.get("role", "") == "assistant" and m.get("type", "response") == "cot":
                continue
            new_messages.append(m.copy())
            new_messages[-1].pop("tool_context", None)

            # Remove thought signatures for riftrunner and bcn
            if m.get("thought_signature", None) is not None:
                new_messages[-1].pop("thought_signature")

            # Remove thought signatures for riftrunner and bcn
            if m.get("extra_content", None) is not None:
                if m["extra_content"].get("google", None) is not None:
                    if m["extra_content"]["google"].get("thought_signature", None) is not None:
                        new_messages[-1]["extra_content"]["google"].pop("thought_signature")

        return new_messages

    def _get_tool_context(self, messages):
        tool_context = {}
        for m in messages:
            if m.get("role", "") == "user" and isinstance(m.get("tool_context"), dict):
                tool_context.update(m["tool_context"])
        return tool_context

    def _tool_call_deadline(self, inner_start):
        if self.max_wallclock_per_call_s is None:
            return None
        state = _responses_retry_state.get()
        if state is not None and state.owner is self:
            inner_start = min(inner_start, state.started_at)
        remaining = self.max_wallclock_per_call_s - (time.time() - inner_start)
        return time.monotonic() + max(0.0, remaining)

    def _execute_tool_function(self, tool_name, arguments, messages, *, call_deadline_monotonic_s=None):
        tool_func = self.tool_functions[tool_name]
        try:
            signature = inspect.signature(tool_func)
            arguments = arguments.copy()
            for param_name, param in signature.parameters.items():
                if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    continue
                if param_name == "call_deadline_monotonic_s":
                    arguments[param_name] = call_deadline_monotonic_s
                    continue
                if param_name == "messages":
                    # Execution provenance must come from the harness, not model arguments.
                    arguments["messages"] = messages
                    continue
                if param_name in arguments:
                    continue
                tool_context = self._get_tool_context(messages)
                if param_name in tool_context:
                    arguments[param_name] = tool_context[param_name]
        except (TypeError, ValueError):
            pass
        return tool_func(**arguments)

    """
        Case 1: VLLM
    """

    def _run_vllm_queries(self, queries):
        """
        Batch all queries into one vllm.generate(...) call, collect final states, yield in order.

        Args:
            queries (List[str]): A list of queries (chat formatted convos) to run.

        Yields:
            tuple: An (idx, conversation, detailed_cost) tuple.
        """
        if self.vllm_model is None:
            self._initialize_vllm()
        tasks = []
        for idx, query in enumerate(queries):
            query_in_template = self.tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True)
            tasks.append({"id": str(idx), "prompt": query_in_template})

        logger.info(f"Running {len(tasks)} queries on local vllm…")
        last_outputs = []
        time_start = time.time()
        for batch in self.vllm_model.generate(tasks, sampling_params=self.sampling_params):
            for out in batch.outputs:
                last_outputs.append(out)
            if self.terminated:
                last_outputs += [None for _ in range(len(tasks) - len(last_outputs))]
        time_end = time.time()

        for idx, out in enumerate(last_outputs):
            if out is None:
                text = ""
            else:
                text = out.text
            conversation = [m.copy() for m in queries[idx]] + [{"role": "assistant", "content": text}]
            inp = len(self.tokenizer.apply_chat_template(queries[idx], tokenize=True, add_generation_prompt=True,enable_thinking=True)['input_ids'])
            outp = len(out.token_ids)
            detailed_cost = {
                "cost": self._get_cost(inp, outp, cached_input_tokens=0),
                "input_tokens": inp,
                "cached_input_tokens": 0,
                "output_tokens": outp,
                "n_retries": 0,
                "time": time_end - time_start if idx == 0 else 0,  # put all in first since batch
            }
            yield idx, conversation, detailed_cost

    """
        Case 2: Batch API
    """

    def _openai_batch_processing(self, queries, indices, retry_idx=0):
        """Processes a batch of queries  using the OpenAI API.

        Args:
            queries (list[MessageList]): A list of queries to run, each is a MessageList in API format.
            retry_idx (int, optional): Current retry index starting from 0.

        Returns:
            list: A list of InternalRequestResult or None.
        """
        if retry_idx >= self.max_retries:
            return [None for _ in range(len(queries))]

        # custom_id encodes the *original* idx (so callers that recurse with
        # only a subset of the queries can still identify which original
        # request a row in the batch output came from). Inside this function
        # we still need to index ``results`` (which is the same length as
        # ``queries``) *positionally*, so keep a reverse map.
        idx_to_pos = {idx: pos for pos, idx in enumerate(indices)}

        jsonl_queries = []
        for idx, query in zip(indices, queries):
            request = {
                "custom_id": f"apiquery-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": self.model, "messages": self._drop_cot(query), **self.kwargs},
            }
            jsonl_queries.append(request)

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)

        # create temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        with open(tmp.name, "wb") as f:
            for i, query in enumerate(jsonl_queries):
                f.write(json.dumps(query).encode("utf-8"))
                f.write(b"\n")

        batch_input_file = client.files.create(file=open(tmp.name, "rb"), purpose="batch")

        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        # close tmp file
        tmp.close()

        logger.info(
            f"Running {len(queries)} queries with batch ID {batch.id} using file with File ID {batch_input_file.id}."
        )

        request_counts = dict(batch.request_counts)

        while True:
            try:
                batch = client.batches.retrieve(batch.id)
            except Exception as e:  # noqa: E722 F841
                logger.warning(f"Error connecting to batch OpenAI. Retrying in 10s. Exception: {e}")
                pass
            if any([request_counts[key] != dict(batch.request_counts)[key] for key in request_counts]):
                request_counts = dict(batch.request_counts)
                logger.info(
                    f"Completed Requests Progress: {request_counts['completed']}/{len(queries)}. Errors: {request_counts['failed']}/{len(queries)}"
                )
            if batch.status == "completed":
                break
            self._sleep(10)

        results = [None for _ in range(len(queries))]
        repeat_indices = []

        if batch.output_file_id is None:
            return results
        while True:
            try:
                file_response = client.files.content(file_id=batch.output_file_id)
                break
            except Exception as e:
                logger.error(f"Error connecting to batch OpenAI. Retrying in 10s. Exception: {e}")
                self._sleep(10)
                continue

        json_response = []
        for line in file_response.iter_lines():
            json_response.append(json.loads(line))

        for result in json_response:
            original_idx = int(result["custom_id"].split("-")[-1])
            pos = idx_to_pos.get(original_idx)
            if pos is None:
                logger.error(f"Batch result has unknown custom_id idx={original_idx}; skipping.")
                continue
            if result["response"]["status_code"] != 200:
                repeat_indices.append(original_idx)
                logger.error(f"Error: {result['response']['status_code']}")
            else:
                try:
                    content = result["response"]["body"]["choices"][0]["message"]["content"]
                    conversation = [m.copy() for m in queries[pos]] + [{"role": "assistant", "content": content}]
                    usage = result["response"]["body"]["usage"]
                    (
                        input_tokens,
                        output_tokens,
                        cached_input_tokens,
                        cached_write_tokens,
                    ) = self._extract_usage_tokens(usage)
                    reasoning_tokens = self._extract_reasoning_tokens(usage)
                    results[pos] = self.InternalRequestResult(
                        conversation,
                        input_tokens,
                        output_tokens,
                        cached_input_tokens=cached_input_tokens,
                        cached_write_tokens=cached_write_tokens,
                        reasoning_tokens=reasoning_tokens,
                        n_retries=retry_idx,
                    )
                except Exception as e:
                    logger.error(f"Error when unpacking batch OpenAI response, will repeat. Exception: {e}")
                    repeat_indices.append(original_idx)

        for pos in range(len(results)):
            if results[pos] is None:
                repeat_indices.append(indices[pos])
        # ``repeat_indices`` may contain duplicates if a row both 200'd and
        # then failed to unpack; dedupe while preserving order.
        seen: set[int] = set()
        repeat_indices = [i for i in repeat_indices if not (i in seen or seen.add(i))]
        if len(repeat_indices) > 0:
            logger.info(f"Repeating {len(repeat_indices)} queries.")
            repeat_queries = [queries[idx_to_pos[i]] for i in repeat_indices]
            repeat_results = self._openai_batch_processing(repeat_queries, repeat_indices, retry_idx + 1)
            for original_idx, result in zip(repeat_indices, repeat_results):
                results[idx_to_pos[original_idx]] = result

        return results

    def _anthropic_batch_processing(self, queries, indices, retry_idx=0):
        """Processes a batch of queries using the Anthropic API.

        Args:
            queries (list[MessageList]): A list of queries to run, each is a MessageList in API format.
            retry_idx (int, optional): Current retry index starting from 0.

        Returns:
            list: A list of InternalRequestResult or None.
        """
        if retry_idx >= self.max_retries:
            return [None for _ in range(len(queries))]

        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
        )

        requests = []
        ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
        ts += f".{datetime.now().microsecond:06d}"
        for idx, query in zip(indices, queries):
            kwargs_here = self.kwargs.copy()
            if query[0]["role"] == "system":
                kwargs_here["system"] = query[0]["content"]
                query = query[1:]

            payload = {
                "custom_id": f"apiquery-{idx}",
                "params": {"model": self.model, "messages": self._drop_cot(query), **kwargs_here},
            }
            self._log_provider_request(ts=ts, batch_idx=idx, request=payload)
            request = Request(
                custom_id=f"apiquery-{idx}",
                params=MessageCreateParamsNonStreaming(model=self.model, messages=self._drop_cot(query), **kwargs_here),
            )
            requests.append(request)

        message_batch = client.messages.batches.create(requests=requests)

        logger.info(f"Running {len(queries)} queries with batch ID {message_batch.id}")

        current_request_counts = dict(message_batch.request_counts)

        while True:
            try:
                message_batch = client.messages.batches.retrieve(
                    message_batch_id=message_batch.id,
                )
            except Exception as e:  # noqa: E722 E841
                logger.warning(f"Error connecting to Anthropic. Retrying in 10s. Exception: {e}")
                pass
            if any(
                [
                    current_request_counts[key] != dict(message_batch.request_counts)[key]
                    for key in current_request_counts
                ]
            ):
                current_request_counts = dict(message_batch.request_counts)
                error_sum = sum([current_request_counts[key] for key in current_request_counts if "succeeded" != key])
                logger.info(
                    f"Succeeded Requests Progress: {current_request_counts['succeeded']}/{len(queries)}. Errors: {error_sum}"
                )
            if message_batch.processing_status == "ended":
                break
            self._sleep(10)

        results = []
        repeat_indices = []

        while True:
            try:
                raw_results = client.messages.batches.results(
                    message_batch_id=message_batch.id,
                )
                break
            except Exception as e:
                logger.error(f"Error connecting to batch Anthropic. Retrying in 10 seconds. Exception: {e}")
                self._sleep(10)

        for i, raw_result in enumerate(raw_results):
            self._log_provider_response(ts=ts, batch_idx=i, response=raw_result.model_dump())
            if raw_result.result.type == "succeeded":
                new_messages = self._get_messages_from_anthropic_content(raw_result.result.message.content)
                conversation = [m.copy() for m in queries[i]] + new_messages
                (
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                    cached_write_tokens,
                ) = self._extract_usage_tokens(raw_result.result.message.usage)
                reasoning_tokens = self._extract_reasoning_tokens(raw_result.result.message.usage)
                results.append(
                    self.InternalRequestResult(
                        conversation,
                        input_tokens,
                        output_tokens,
                        cached_input_tokens=cached_input_tokens,
                        cached_write_tokens=cached_write_tokens,
                        reasoning_tokens=reasoning_tokens,
                        n_retries=retry_idx,
                    )
                )
            else:
                results.append(None)
                repeat_indices.append(i)
                if raw_result.result.type == "errored":
                    logger.error(raw_result.result.error)

        if len(repeat_indices) > 0:
            logger.info(f"Repeating {len(repeat_indices)} queries.")
            repeat_queries = [queries[i] for i in repeat_indices]
            # The recursive call needs the *original* idx values so that
            # custom_id round-trips correctly, while the local ``results``
            # list (positional) is the one we assign into here.
            repeat_original_indices = [indices[i] for i in repeat_indices]
            repeat_results = self._anthropic_batch_processing(
                repeat_queries, repeat_original_indices, retry_idx + 1
            )
            for i, result in zip(repeat_indices, repeat_results):
                results[i] = result

        return results

    """
        Case 3: Standard API
    """

    def _run_query_with_retry(self, idx, query, ignore_tool_calls=False):
        """Runs a query on standard API with retries on failure.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """
        state = _ResponsesRetryState(owner=self, started_at=time.time())
        token = _responses_retry_state.set(state)
        trace = active_trace.get() or ProviderTrace()
        trace_token = active_trace.set(trace)
        try:
            result = self._run_query_retry_loop(idx, query, ignore_tool_calls=ignore_tool_calls)
            totals = trace.totals(idx)
            if result is None and totals["provider_attempts"]:
                result = self.InternalRequestResult(query + [{"role": "assistant", "content": ""}], 0, 0)
            if result is not None and totals["provider_attempts"]:
                result.provider_outcomes = totals["provider_outcomes"]
                result.usage_unavailable = totals["usage_unavailable"]
                result_cost = result.cost_usd if result.cost_usd is not None else self._get_cost(
                    result.input_tokens, result.output_tokens, result.cached_input_tokens, result.cached_write_tokens,
                )
                for name in COUNTS:
                    setattr(result, name, max(getattr(result, name), totals[name]))
                result.cost_usd = max(result_cost, totals["cost"])
            return result
        except Exception as exc:
            exc.partial_usage = trace.totals(idx)
            raise
        finally:
            active_trace.reset(trace_token)
            _responses_retry_state.reset(token)

    def _run_query_retry_loop(self, idx, query, ignore_tool_calls=False):
        retry_idx = 0
        total_retries = 0
        start_time = time.time()
        while retry_idx < self.max_retries:
            if self.terminated:
                return None
            # Wallclock budget check: the original retry tree
            # (outer=3 x inner=25 x sleep=60s) could spend ~75 min on a
            # single failing call. That eats the 24h benchmark wallclock
            # on a provider hiccup and blocks any tandem that relies on
            # this call completing. Cap total wallclock per call here so
            # rate-limit / overload storms surface as a single failure
            # in minutes, not hours.
            if self.max_wallclock_per_call_s is not None:
                elapsed = time.time() - start_time
                if elapsed >= self.max_wallclock_per_call_s:
                    logger.error(
                        f"Wallclock budget ({self.max_wallclock_per_call_s}s) "
                        f"exhausted after {elapsed:.0f}s on call idx={idx} "
                        f"(retry_idx={retry_idx}, total_retries={total_retries}). Giving up."
                    )
                    break
            try:
                result = self._run_query(idx, query, ignore_tool_calls=ignore_tool_calls)
                result.n_retries += total_retries
                result.time = time.time() - start_time
                self._sleep(self.sleep_after_request)
                return result
            except Exception as e:
                # These are control-flow outcomes, not retryable provider
                # failures.  Retrying a terminated client delays shutdown, and
                # wrapping an attachment rejection loses the partial-call cost
                # carried by ProviderAttachmentRejectedError.
                if isinstance(e, _ClientTerminated):
                    return None
                if isinstance(e, ProviderAttachmentRejectedError):
                    raise
                if "Max inner retries reached." in str(e):
                    total_retries += self.max_retries_inner
                elif "rate limit" not in str(e).lower() and "429" not in str(e):
                    total_retries += 1
                logger.error(f"Error in outer retries. Exception: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Terminal errors: bad key, missing model, malformed payload.
                # Retrying these just wastes 60+120+240s of wallclock before
                # giving up. *Raise* the original exception rather than
                # ``break``-ing into the throw_error_on_failure fall-through:
                # otherwise the failure surfaces only as ``None`` →
                # blank-text assistant in ``run_queries``, and Council seats
                # render as "empty response from provider" instead of the
                # real 401 / model_not_found. Re-raising lets ``Council._one``
                # catch the actual error and report it as
                # ``CouncilReply(error=...)``.
                if _is_terminal_api_error(e):
                    logger.error(
                        f"Terminal API error on call idx={idx}; not retrying. Hint: {e}"
                    )
                    raise
                # Exponential backoff between outer retries. ``retry_idx``
                # at this point is the count of *non-rate-limit* errors
                # observed so far in this call (0 on first failure,
                # 1 on second, etc.). Rate-limit failures don't grow
                # the backoff — those already wait the server-suggested
                # retry-after at the inner-loop level.
                #
                # IMPORTANT: salvageable responses (Responses API
                # ``status=failed`` with non-empty ``output``) are
                # handled inside ``_openai_query_responses_api`` and
                # returned successfully — they never reach this except
                # block, so they don't trigger a retry. See
                # ``_estimate_salvaged_usage`` / ``_SalvageUsage``.
                backoff_step = retry_idx
                backoff_sleep_s = min(
                    self.sleep_on_error * (2 ** backoff_step),
                    self.sleep_on_error_max,
                )
                # Skip the sleep entirely if no further retry will run.
                # Rate-limit errors don't increment ``retry_idx`` (they
                # always retry), but a non-rate-limit error on the last
                # allowed attempt would otherwise burn up to
                # ``sleep_on_error_max`` seconds of wallclock immediately
                # before exiting the loop. Codex review P2.
                is_rate_limit_err = (
                    "rate limit" in str(e).lower() or "429" in str(e)
                )
                if not is_rate_limit_err and retry_idx + 1 >= self.max_retries:
                    logger.info(
                        f"Retries exhausted on call idx={idx} "
                        f"(retry_idx={retry_idx}, max_retries={self.max_retries}); "
                        f"skipping final {backoff_sleep_s}s backoff and giving up."
                    )
                    break
                # Also short-circuit the sleep if it would push us past
                # the wallclock budget — otherwise we'd sleep N seconds
                # only to give up immediately on the next iteration.
                if self.max_wallclock_per_call_s is not None:
                    elapsed = time.time() - start_time
                    if elapsed + backoff_sleep_s >= self.max_wallclock_per_call_s:
                        logger.error(
                            f"Wallclock budget would be exhausted mid-sleep "
                            f"({elapsed:.0f}s + {backoff_sleep_s}s sleep > "
                            f"{self.max_wallclock_per_call_s}s budget). Giving up."
                        )
                        break
                logger.info(
                    f"Backing off {backoff_sleep_s}s before outer-retry "
                    f"{retry_idx + 1} on call idx={idx}."
                )
                self._sleep(backoff_sleep_s)
                # if api error is not due to rate limit, try again
                if "rate limit" not in str(e).lower() and "429" not in str(e):
                    retry_idx += 1
                if "violating our usage policy" in str(e).lower():
                    print("Stopping - prompt repeatedly violated usage policy -- ", query)
                    if retry_idx > 3:
                        break
                continue
        if self.throw_error_on_failure:
            raise ValueError("Max outer retries reached.")
        else:
            return None

    def _run_query(self, idx, query, ignore_tool_calls=False):
        """Runs a query on standard API.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """
        if self.api == "google":
            return self._google_query_with_internal_tools(idx, query)
        if self.api == "openai":
            return self._openai_query_with_tools(idx, query, ignore_tool_calls=ignore_tool_calls)
        elif self.api == "together":
            return self._openai_query_with_tools(idx, query, is_together=True, ignore_tool_calls=ignore_tool_calls)
        elif self.api == "anthropic":
            return self._anthropic_query_with_tools(idx, query, ignore_tool_calls=ignore_tool_calls)
        elif self.api == "openrouter":
            return self._openai_query_with_tools(idx, query)

    def _anthropic_query(self, idx, query):
        """Backward-compatible wrapper; Anthropic now uses one unified Messages API path."""
        return self._anthropic_query_with_tools(idx, query, ignore_tool_calls=True)

    def _anthropic_request_timeout(self, inner_start: float):
        state = _responses_retry_state.get()
        if state is not None and state.owner is self:
            inner_start = state.started_at
        if self.max_wallclock_per_call_s is None:
            return self.timeout

        elapsed = time.time() - inner_start
        remaining = self.max_wallclock_per_call_s - elapsed
        if remaining <= 0:
            raise _ProviderWallclockTimeout(
                f"Wallclock budget ({self.max_wallclock_per_call_s}s) "
                f"exhausted before Anthropic request attempt."
            )
        try:
            return min(float(self.timeout), remaining)
        except (TypeError, ValueError):
            return remaining

    def _anthropic_payload_with_betas(self, payload: dict) -> dict:
        anthropic_betas = list(getattr(self, "anthropic_betas", []) or [])
        if not anthropic_betas:
            return payload
        out = dict(payload)
        configured = list(out.get("betas") or [])
        for beta in anthropic_betas:
            if beta not in configured:
                configured.append(beta)
        out["betas"] = configured
        return out

    def _anthropic_messages_api(self, client):
        if getattr(self, "anthropic_betas", None):
            return client.beta.messages
        return client.messages

    def _create_anthropic_message(self, client, payload, *, inner_start: float):
        messages_api = self._anthropic_messages_api(client)
        if not self.stream_anthropic_messages:
            return messages_api.create(**payload)

        opened = []
        trace, attempt = active_trace.get(), active_attempt.get()
        stopped = threading.Event()
        partial_text = []
        partial_chars = 0
        partial_lock = threading.Lock()

        def consume():
            nonlocal partial_chars
            start_usage = None
            usage = {}
            response_id = None
            with messages_api.stream(**payload) as stream:
                opened.append(stream)
                if stopped.is_set():
                    raise _ClientTerminated("Stream opened after cancellation")
                for event in stream:
                    if stopped.is_set():
                        raise _ClientTerminated("Stream cancelled")
                    if getattr(event, "type", None) == "message_start":
                        message = getattr(event, "message", None)
                        start_usage = getattr(message, "usage", None)
                        response_id = getattr(message, "id", None)
                        usage.update(self._usage_to_dict(start_usage))
                    elif getattr(event, "type", None) == "message_delta":
                        usage.update(self._usage_to_dict(getattr(event, "usage", None)))
                    elif getattr(event, "type", None) == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if getattr(delta, "type", None) == "text_delta":
                            text = getattr(delta, "text", "")
                            with partial_lock:
                                if partial_chars < 65536:
                                    partial_text.append(text[:65536 - partial_chars])
                                partial_chars += len(text)
                    if trace is not None and attempt is not None and getattr(event, "type", None) in ("message_start", "message_delta"):
                        trace.response(self, *attempt, {"id": response_id, "usage": usage, "usage_partial": True})
                response = stream.get_final_message()
            self._merge_anthropic_stream_start_usage(response, start_usage)
            if trace is not None and attempt is not None:
                trace.response(self, *attempt, response.model_dump())
            return response

        def cancel():
            stopped.set()
            for stream in opened:
                stream.close()

        try:
            return self._bounded_provider_operation(
                consume, timeout=self._anthropic_request_timeout(inner_start), cancel=cancel,
            )
        except Exception as exc:
            if trace is not None and attempt is not None:
                trace.update(attempt, outcome=type(exc).__name__)
                with partial_lock:
                    text = "".join(partial_text)
                    truncated = partial_chars > 65536
                # An audit artifact only: never promote an interrupted fragment
                # into a completed council reply or manuscript.
                self._log_provider_response(ts=attempt[0], batch_idx=attempt[1],
                                            partial_text=text, partial_text_truncated=truncated,
                                            exception={"type": type(exc).__name__, "msg": str(exc)})
            raise
        finally:
            stopped.set()

    def _merge_anthropic_stream_start_usage(self, response, start_usage) -> None:
        if start_usage is None:
            return
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        for attr in (
            "input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            start_value = getattr(start_usage, attr, None)
            if start_value is None:
                continue
            current_value = getattr(usage, attr, None)
            try:
                should_update = current_value is None or int(start_value) > int(current_value)
            except (TypeError, ValueError):
                should_update = True
            if should_update:
                try:
                    setattr(usage, attr, start_value)
                except Exception:
                    object.__setattr__(usage, attr, start_value)

    def _anthropic_response_needs_final_answer_salvage(self, response, text_blocks) -> bool:
        if not self.anthropic_salvage_empty_max_tokens:
            return False
        if any(str(text).strip() for text in text_blocks):
            return False
        if getattr(response, "stop_reason", None) == "max_tokens":
            return True
        usage = getattr(response, "usage", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if self.max_tokens is None or output_tokens is None:
            return False
        try:
            return int(output_tokens) >= int(self.max_tokens)
        except (TypeError, ValueError):
            return False

    def _anthropic_final_answer_salvage_kwargs(self):
        kwargs = self.kwargs.copy()
        kwargs.pop("thinking", None)
        kwargs.pop("output_config", None)
        if self.max_tokens is not None:
            kwargs[self.max_tokens_param] = self.max_tokens
        return kwargs

    def _anthropic_content_summary(self, response):
        assistant_blocks = []
        text_blocks = []
        tool_uses = []
        for block in response.content:
            block_dict = block.model_dump() if hasattr(block, "model_dump") else block
            if isinstance(block_dict, dict):
                assistant_blocks.append(block_dict)
            block_type = getattr(block, "type", block_dict.get("type", "") if isinstance(block_dict, dict) else "")
            if block_type == "text":
                text_blocks.append(getattr(block, "text", block_dict.get("text", "")))
            elif block_type == "tool_use":
                tool_uses.append(block)
        return assistant_blocks, text_blocks, tool_uses

    def _anthropic_fetch_final_answer_salvage(
        self,
        client,
        idx,
        system_message,
        anthropic_messages,
        inner_start,
        nb_executed_tool_calls,
    ):
        self._append_anthropic_message(anthropic_messages, "user", _ANTHROPIC_FINAL_ANSWER_SALVAGE_PROMPT)
        response = None
        n_retries = -1
        total_retries = 0
        while response is None and n_retries < self.max_retries_inner:
            n_retries += 1
            ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
            ts += f".{datetime.now().microsecond:06d}"
            request_logged = False
            try:
                payload = {
                    "model": self.model,
                    "messages": anthropic_messages,
                    **self._anthropic_final_answer_salvage_kwargs(),
                }
                payload["timeout"] = self._anthropic_request_timeout(inner_start)
                if system_message is not anthropic.NOT_GIVEN:
                    payload["system"] = system_message
                payload = self._anthropic_payload_with_betas(payload)
                self._log_provider_request(
                    ts=ts,
                    batch_idx=idx,
                    request=payload,
                    nb_executed_tool_calls=nb_executed_tool_calls,
                    n_retries=n_retries,
                    stream_anthropic_messages=self.stream_anthropic_messages,
                    anthropic_final_answer_salvage=True,
                )
                request_logged = True
                response = self._create_anthropic_message(client, payload, inner_start=inner_start)
                self._log_provider_response(ts=ts, batch_idx=idx, response=self._provider_response_dict(response))
            except Exception as e:
                if "rate limit" not in str(e).lower() and "429" not in str(e):
                    total_retries += 1
                if request_logged:
                    self._log_provider_response(ts=ts, batch_idx=idx, response={"exception": str(e)})
                logger.error(f"Got Anthropic error in final-answer salvage. Exception: {e}")
                if (
                    isinstance(e, _ProviderWallclockTimeout)
                    or (
                        self.max_wallclock_per_call_s is not None
                        and time.time() - inner_start >= self.max_wallclock_per_call_s
                    )
                    or _is_terminal_api_error(e)
                ):
                    raise
                self._sleep(60)
        if response is None:
            raise ValueError("Max inner retries reached during Anthropic final-answer salvage.")
        return response, total_retries

    def _anthropic_tool_descriptions(self):
        anthropic_tools = []
        for tool_desc in self.tool_descriptions:
            if tool_desc.get("type") == "function" and "function" in tool_desc:
                fn = tool_desc["function"]
                anthropic_tools.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
            elif tool_desc.get("type") == "web_search_preview":
                anthropic_tools.append(
                    {
                        "type": "web_search_20260209",
                        "name": "web_search",
                        "max_uses": int(tool_desc.get("max_uses", 20)),
                    }
                )
            elif tool_desc.get("type") == "code_interpreter":
                anthropic_tools.append(
                    {"type": "code_execution_20260521", "name": "code_execution"}
                )
            elif str(tool_desc.get("type", "")).startswith(
                ("web_search_", "web_fetch_", "code_execution_")
            ) and "name" in tool_desc:
                anthropic_tools.append(tool_desc)
            elif "name" in tool_desc and "input_schema" in tool_desc:
                anthropic_tools.append(tool_desc)
            else:
                logger.warning(f"Skipping unsupported Anthropic tool descriptor: {tool_desc}")
        return anthropic_tools

    def _anthropic_response_needs_continuation(self, response, text_blocks) -> bool:
        if not self.anthropic_continue_on_max_tokens:
            return False
        if getattr(response, "stop_reason", None) != "max_tokens":
            return False
        return any(str(text).strip() for text in text_blocks)

    def _to_anthropic_content_blocks(self, content):
        def _safe_text_block(text, cache_control=None):
            if text is None:
                return None
            text_str = str(text)
            if text_str.strip() == "":
                return None
            block = {"type": "text", "text": text_str}
            if cache_control is not None:
                block["cache_control"] = cache_control
            return block

        if isinstance(content, str):
            block = _safe_text_block(content)
            return [] if block is None else [block]
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            block = _safe_text_block(content)
            return [] if block is None else [block]

        out = []
        for block in content:
            if not isinstance(block, dict):
                text_block = _safe_text_block(block)
                if text_block is not None:
                    out.append(text_block)
                continue
            block_type = block.get("type", "")
            if block_type in ["text", "input_text", "output_text"] and "text" in block:
                mapped = _safe_text_block(block.get("text", ""), cache_control=block.get("cache_control", None))
                if mapped is not None:
                    out.append(mapped)
            elif block_type in ["image", "tool_use", "tool_result"]:
                out.append(block)
            else:
                # Preserve known structured blocks; otherwise avoid introducing invalid empty text blocks.
                if "text" in block:
                    mapped = _safe_text_block(block.get("text", ""), cache_control=block.get("cache_control", None))
                    if mapped is not None:
                        out.append(mapped)
                else:
                    out.append(block)
        return out

    def _append_anthropic_message(self, anthropic_messages, role, content):
        content_blocks = self._to_anthropic_content_blocks(content)
        if len(content_blocks) == 0:
            return
        if len(anthropic_messages) > 0 and anthropic_messages[-1]["role"] == role:
            anthropic_messages[-1]["content"].extend(content_blocks)
        else:
            anthropic_messages.append({"role": role, "content": content_blocks})

    def _convert_to_anthropic_messages(self, messages):
        system_message = anthropic.NOT_GIVEN
        anthropic_messages = []

        for m in self._drop_cot(messages):
            role = m.get("role", "")
            if role in ["system", "developer"]:
                if system_message is anthropic.NOT_GIVEN and len(anthropic_messages) == 0:
                    system_blocks = self._to_anthropic_content_blocks(m.get("content", ""))
                    if len(system_blocks) > 0:
                        system_message = system_blocks
                else:
                    self._append_anthropic_message(anthropic_messages, "user", m.get("content", ""))
                continue

            if role == "user":
                self._append_anthropic_message(anthropic_messages, "user", m.get("content", ""))
                continue

            if role == "assistant" and m.get("type", "") in ["function_call", "tool_call"]:
                tool_use_id = m.get("tool_call_id", m.get("id", m.get("call_id", None)))
                if tool_use_id is None:
                    logger.warning(f"Anthropic tool call missing id; skipping block: {m}")
                    continue
                arguments = m.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                self._append_anthropic_message(
                    anthropic_messages,
                    "assistant",
                    [
                        {
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": m.get("tool_name", m.get("name", "")),
                            "input": arguments,
                        }
                    ],
                )
                continue

            if role == "assistant" and m.get("tool_calls", None):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    arguments = fn.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {"raw": arguments}
                    self._append_anthropic_message(
                        anthropic_messages,
                        "assistant",
                        [
                            {
                                "type": "tool_use",
                                "id": tc.get("id"),
                                "name": fn.get("name", ""),
                                "input": arguments,
                            }
                        ],
                    )
                continue

            if role in ["tool", "tool_response", "function_call_output"] or m.get("type", "") == "function_call_output":
                tool_use_id = m.get("tool_call_id", m.get("id", m.get("call_id", None)))
                if tool_use_id is None:
                    logger.warning(f"Anthropic tool response missing tool_call_id; converting to user text: {m}")
                    self._append_anthropic_message(anthropic_messages, "user", m.get("content", m.get("output", "")))
                    continue
                self._append_anthropic_message(
                    anthropic_messages,
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": m.get("content", m.get("output", "")),
                        }
                    ],
                )
                continue

            if role == "assistant":
                if m.get("content", None) is None:
                    continue
                self._append_anthropic_message(anthropic_messages, "assistant", m.get("content", ""))
            elif role != "":
                self._append_anthropic_message(anthropic_messages, "user", m.get("content", ""))

        return system_message, anthropic_messages

    def _anthropic_query_with_tools(self, idx, messages, ignore_tool_calls=False):
        client = anthropic.Anthropic(
            api_key=self.api_key,
            max_retries=0,
            timeout=self.timeout,
        )

        if ignore_tool_calls:
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        else:
            max_tool_calls_mode, max_tool_calls = self.max_tool_calls_mode, self.max_tool_calls
        total_max_tool_calls = sum(max_tool_calls.values())
        if max_tool_calls_mode == "total":
            nb_executed_tool_calls = {"any": 0}
        else:
            nb_executed_tool_calls = {t: 0 for t in self.tool_functions.keys()}

        conversation = [m.copy() for m in messages]
        system_message, anthropic_messages = self._convert_to_anthropic_messages(conversation)
        anthropic_tools = self._anthropic_tool_descriptions()

        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        cached_write_tokens = 0
        reasoning_tokens = 0
        cost_usd = 0.0
        total_retries = 0
        inner_start = time.time()
        server_tool_continuations = 0
        max_token_continuations = 0
        continued_text_chunks = []

        while not self.terminated:
            response = None
            n_retries = -1
            while response is None and n_retries < self.max_retries_inner:
                n_retries += 1
                ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
                ts += f".{datetime.now().microsecond:06d}"
                request_logged = False
                # Wallclock cap: bail out of the inner loop so the outer
                # retry supervisor can record the failure and stop
                # instead of blocking for up to 25*60s on a single call.
                if self.max_wallclock_per_call_s is not None:
                    if time.time() - inner_start >= self.max_wallclock_per_call_s:
                        raise ValueError(
                            f"Wallclock budget ({self.max_wallclock_per_call_s}s) "
                            f"exhausted in inner anthropic retry loop after "
                            f"{n_retries} attempts."
                        )
                try:
                    payload = {"model": self.model, "messages": anthropic_messages, **self.kwargs}
                    payload["timeout"] = self._anthropic_request_timeout(inner_start)
                    if system_message is not anthropic.NOT_GIVEN:
                        payload["system"] = system_message
                    if len(anthropic_tools) > 0:
                        payload["tools"] = anthropic_tools
                    if ignore_tool_calls:
                        payload["tool_choice"] = {"type": "none"}
                    payload = self._anthropic_payload_with_betas(payload)
                    info = {
                        "nb_executed_tool_calls": nb_executed_tool_calls,
                        "n_retries": n_retries,
                        "stream_anthropic_messages": self.stream_anthropic_messages,
                    }
                    self._log_provider_request(ts=ts, batch_idx=idx, request=payload, **info)
                    request_logged = True
                    response = self._create_anthropic_message(client, payload, inner_start=inner_start)
                    self._log_provider_response(ts=ts, batch_idx=idx, response=self._provider_response_dict(response))
                except Exception as e:
                    if "rate limit" not in str(e).lower() and "429" not in str(e):
                        total_retries += 1
                    if request_logged:
                        self._log_provider_response(ts=ts, batch_idx=idx, response={"exception": str(e)})
                    logger.error(f"Got Anthropic error in tools inner loop. Exception: {e}")
                    if (
                        isinstance(e, _ProviderWallclockTimeout)
                        or (
                            self.max_wallclock_per_call_s is not None
                            and time.time() - inner_start >= self.max_wallclock_per_call_s
                        )
                    ):
                        raise
                    # Terminal errors raise out of the inner loop so the
                    # outer supervisor sees them immediately (and so a
                    # Council seat with a bad key resolves to an explicit
                    # CouncilReply(error=...) rather than blank text).
                    if _is_terminal_api_error(e):
                        raise
                    self._sleep(60)
                    continue
            if response is None:
                raise ValueError("Max inner retries reached.")

            step_input, step_output, step_cached_input, step_write_cache = self._extract_usage_tokens(response.usage)
            step_reasoning = self._extract_reasoning_tokens(response.usage)
            input_tokens += step_input
            output_tokens += step_output
            cached_input_tokens += step_cached_input
            cached_write_tokens += step_write_cache
            reasoning_tokens += step_reasoning
            cost_usd += self._get_cost(
                step_input,
                step_output,
                step_cached_input,
                step_write_cache,
            )

            assistant_blocks = []
            text_blocks = []
            tool_uses = []
            for block in response.content:
                block_dict = block.model_dump() if hasattr(block, "model_dump") else block
                if isinstance(block_dict, dict):
                    assistant_blocks.append(block_dict)
                block_type = getattr(block, "type", block_dict.get("type", "") if isinstance(block_dict, dict) else "")
                if block_type == "text":
                    text_blocks.append(getattr(block, "text", block_dict.get("text", "")))
                elif block_type == "thinking":
                    conversation.append({"role": "assistant", "type": "cot", "content": getattr(block, "thinking", "")})
                elif block_type == "tool_use":
                    tool_uses.append(block)
                elif isinstance(block_dict, dict):
                    conversation.append({"role": "assistant", **block_dict})

            if len(text_blocks) > 0:
                text_joined = "\n\n".join(text_blocks)
                conversation.append({"role": "assistant", "content": text_joined})
                continued_text_chunks.append(text_joined)
            # A truncated server-tool call is not a resumable tool result.
            # Remove the incomplete tail, never fabricate an execution result.
            unresolved_server_tool = False
            if getattr(response, "stop_reason", None) == "max_tokens":
                results = {block.get("tool_use_id") for block in assistant_blocks if block.get("tool_use_id")}
                for index, block in enumerate(assistant_blocks):
                    if block.get("type") == "server_tool_use" and block.get("id") not in results:
                        assistant_blocks = assistant_blocks[:index]
                        unresolved_server_tool = True
                        break
            if len(assistant_blocks) > 0:
                self._append_anthropic_message(anthropic_messages, "assistant", assistant_blocks)

            if getattr(response, "stop_reason", None) == "pause_turn":
                if server_tool_continuations >= self.anthropic_max_server_tool_continuations:
                    logger.warning(
                        "Anthropic server-tool response returned pause_turn after "
                        f"{server_tool_continuations} continuation(s); stopping."
                    )
                    break
                server_tool_continuations += 1
                continue

            if len(tool_uses) == 0:
                if not unresolved_server_tool and self._anthropic_response_needs_continuation(response, text_blocks):
                    if max_token_continuations >= self.anthropic_max_token_continuations:
                        logger.warning(
                            "Anthropic response stopped at max_tokens after "
                            f"{max_token_continuations} continuation(s); accepting partial text."
                        )
                        break
                    max_token_continuations += 1
                    self._append_anthropic_message(
                        anthropic_messages,
                        "user",
                        "Please continue exactly from where you left off. Do not repeat earlier text.",
                    )
                    continue
                if unresolved_server_tool or self._anthropic_response_needs_final_answer_salvage(response, text_blocks):
                    salvage_response, salvage_retries = self._anthropic_fetch_final_answer_salvage(
                        client,
                        idx,
                        system_message,
                        anthropic_messages,
                        inner_start,
                        nb_executed_tool_calls,
                    )
                    total_retries += salvage_retries
                    step_input, step_output, step_cached_input, step_write_cache = self._extract_usage_tokens(
                        salvage_response.usage
                    )
                    step_reasoning = self._extract_reasoning_tokens(salvage_response.usage)
                    input_tokens += step_input
                    output_tokens += step_output
                    cached_input_tokens += step_cached_input
                    cached_write_tokens += step_write_cache
                    reasoning_tokens += step_reasoning
                    cost_usd += self._get_cost(
                        step_input,
                        step_output,
                        step_cached_input,
                        step_write_cache,
                    )
                    salvage_assistant_blocks, salvage_text_blocks, _ = self._anthropic_content_summary(
                        salvage_response
                    )
                    if len(salvage_text_blocks) > 0:
                        conversation.append({"role": "assistant", "content": "\n\n".join(salvage_text_blocks)})
                    if len(salvage_assistant_blocks) > 0:
                        self._append_anthropic_message(anthropic_messages, "assistant", salvage_assistant_blocks)
                break
            if total_max_tool_calls <= sum(nb_executed_tool_calls.values()):
                break

            tool_result_blocks = []
            for tool_use in tool_uses:
                tool_name = getattr(tool_use, "name", "")
                tool_id = getattr(tool_use, "id", None)
                arguments = getattr(tool_use, "input", {})
                output = ""

                if tool_name not in self.tool_functions:
                    output = f"Error: Tool {tool_name} not found."
                    logger.warning(output)
                else:
                    tool_key = "any" if max_tool_calls_mode == "total" else tool_name
                    if nb_executed_tool_calls[tool_key] >= max_tool_calls[tool_key]:
                        output = f"Error: Tool call after exceeding max # of tool calls ({max_tool_calls[tool_key]})."
                    else:
                        try:
                            output = self._execute_tool_function(
                                tool_name, arguments, conversation,
                                call_deadline_monotonic_s=self._tool_call_deadline(inner_start),
                            )
                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name}. Exception: {e}")
                            output = f"Error executing tool {tool_name}. Exception: {e}"
                        if isinstance(output, tuple):
                            output, additional_cost = output
                            input_tokens += additional_cost["input_tokens"]
                            output_tokens += additional_cost["output_tokens"]
                            cost_usd += float(
                                additional_cost.get(
                                    "cost",
                                    self._get_cost(
                                        additional_cost["input_tokens"],
                                        additional_cost["output_tokens"],
                                    ),
                                )
                            )
                        nb_executed_tool_calls[tool_key] += 1

                nb_tool_calls_left = max_tool_calls.get("any", 0) - nb_executed_tool_calls.get("any", 0)
                if max_tool_calls_mode != "total":
                    nb_tool_calls_left = max_tool_calls.get(tool_name, 0) - nb_executed_tool_calls.get(tool_name, 0)
                detail = "for this tool" if max_tool_calls_mode == "per_tool" else "(across all tools)"
                info = f"\n\n### INFO ###\nYou have {nb_tool_calls_left} tool executions left {detail}."
                if not self.include_max_tool_calls:
                    info = ""

                output_str = output if isinstance(output, str) else str(output)
                conversation.append(
                    {
                        "role": "assistant",
                        "type": "function_call",
                        "id": tool_id,
                        "name": tool_name,
                        "arguments": arguments,
                    }
                )
                conversation.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "tool_call_id": tool_id,
                        "content": output_str + info,
                    }
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": output_str + info,
                    }
                )

            if len(tool_result_blocks) == 0:
                break
            self._append_anthropic_message(anthropic_messages, "user", tool_result_blocks)

            if total_max_tool_calls <= sum(nb_executed_tool_calls.values()):
                break
        
        if output_tokens == self.max_tokens and (conversation[-1].get("content", "") == "\n\n" or conversation[-1].get("content", "") == "\n"):
            conversation[-1]["content"] = "Model reached its max allowed token limit. \\boxed{None}"
        elif len(conversation) == len(messages) or conversation[-1].get("type") == "cot":
            conversation.append({"role": "assistant", "content": ""})
        elif max_token_continuations > 0 and continued_text_chunks:
            stitched = "\n\n".join(continued_text_chunks)
            if conversation[-1].get("content") != stitched:
                conversation.append({"role": "assistant", "content": stitched})
        return self.InternalRequestResult(
            conversation,
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
            cached_write_tokens=cached_write_tokens,
            reasoning_tokens=reasoning_tokens,
            n_retries=total_retries,
            cost_usd=cost_usd,
        )

    def _openai_query_with_tools(self, idx, query, is_together=False, ignore_tool_calls=False):
        """Queries the OpenAI API with tools.

        Args:
            idx (int): The index of the query in the batch of queries given to run_queries.
            query (MessageList): The query to run.
            is_together (bool, optional): Whether to use the Together API. Defaults to False.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """
        if is_together:
            client = Together(timeout=self.timeout, max_retries=0)
        else:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0)

        if self.use_openai_responses_api:
            return self._openai_query_responses_api(client, idx, query, ignore_tool_calls=ignore_tool_calls)
        else:
            return self._openai_query_chat_completions_api(client, idx, query, ignore_tool_calls=ignore_tool_calls)

    def _openai_query_responses_api(self, client, idx, messages, ignore_tool_calls=False):
        """Queries the OpenAI API with the responses API.

        Args:
            client: The OpenAI client.
            idx (int): The index of the query in the batch of queries given to run_queries.
            messages (list): The messages to send.
            ignore_tool_calls (bool, optional): Whether to ignore tool calls in this interaction.

        Returns:
            InternalRequestResult or None
        """

        # Set up tools
        response_tools = []
        for tool_desc in self.tool_descriptions:
            if tool_desc["type"] != "function":
                response_tools.append(_openai_responses_tool_descriptor(tool_desc))
            else:
                response_tools.append({"type": "function", **tool_desc["function"]})
        if ignore_tool_calls:
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        elif len(response_tools) == 1 and response_tools[0]["type"] == "code_interpreter":
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        else:
            max_tool_calls_mode, max_tool_calls = self.max_tool_calls_mode, self.max_tool_calls

        # State
        total_max_tool_calls = sum(max_tool_calls.values())
        if max_tool_calls_mode == "total":
            nb_executed_tool_calls = {"any": 0}
        else:
            nb_executed_tool_calls = {t: 0 for t in self.tool_functions.keys()}
        conversation = [m.copy() for m in messages]
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        cached_write_tokens = 0
        reasoning_tokens = 0
        cost_usd = 0.0
        total_retries = 0
        retry_state = _responses_retry_state.get()
        if retry_state is None or retry_state.owner is not self:
            retry_state = _ResponsesRetryState(owner=self, started_at=time.time())
        max_output_token_continuations = 0
        in_wrapup = False
        inner_start = retry_state.started_at
        delivery_notice_sent = False

        def delivery_only():
            return (self.tool_wrapup_reserve_s > 0 and self.max_wallclock_per_call_s is not None
                    and time.time() >= inner_start + max(0, self.max_wallclock_per_call_s - self.tool_wrapup_reserve_s))

        def _tool_is_available_for_request(tool):
            if delivery_only() and tool.get("name") not in {"read_context", "publish_artifact", "publish_sandbox_artifact"}:
                return False
            if tool.get("type") != "function":
                return self.max_hosted_tool_calls is None or retry_state.hosted_tool_calls_used < self.max_hosted_tool_calls
            if max_tool_calls_mode == "total":
                return nb_executed_tool_calls.get("any", 0) < max_tool_calls.get("any", 0)
            name = str(tool.get("name", ""))
            return nb_executed_tool_calls.get(name, 0) < max_tool_calls.get(name, 0)

        while not self.terminated:
            # Inner retry to get a response
            deadline_recovered = False
            response = None
            n_retries = -1
            while response is None and n_retries < self.max_retries_inner:
                n_retries += 1
                if self.terminated:
                    raise _ClientTerminated("APIClient terminated during OpenAI responses retry loop.")
                # Wallclock cap (shared across all inner attempts for this
                # call): stop ~retries x 60s + provider-overload storms
                # from running longer than the per-call budget set on
                # APIClient. Observed 8+ min stuck on
                # "No usage info in response" before this cap.
                if self.max_wallclock_per_call_s is not None:
                    if time.time() - inner_start >= self.max_wallclock_per_call_s:
                        if in_wrapup:
                            break
                        raise ValueError(
                            f"Wallclock budget ({self.max_wallclock_per_call_s}s) "
                            f"exhausted in inner openai responses retry loop after "
                            f"{n_retries} attempts."
                        )
                pro_attempt = False
                request_tools = [tool for tool in response_tools if _tool_is_available_for_request(tool)]
                available = {tool.get("type") for tool in request_tools}
                completed = {item.get("type", "").removesuffix("_call") for item in conversation
                             if item.get("status") == "completed"}
                if "web_search" in completed:
                    completed.add("web_search_preview")
                if self.required_hosted_tool_validator is not None:
                    completed = self.required_hosted_tool_types if self.required_hosted_tool_validator() else set()
                if self.required_hosted_tool_types - available - completed:
                    raise RequiredHostedToolUnavailable(
                        "Mandatory hosted tools are unavailable; reconcile prior work before retrying")
                try:
                    if delivery_only() and not delivery_notice_sent:
                        conversation.append({"role": "user", "content": (
                            "The remaining time is reserved for delivery. Do not start new research. "
                            "Publish existing findings/artifacts now, state outstanding gaps, and return your final report. "
                            "The original hard deadline still applies."
                        )})
                        delivery_notice_sent = True
                    request_kwargs = self._responses_request_kwargs(retry_state, wrapup=in_wrapup)
                    payload = {
                        "model": self.model,
                        "tools": request_tools,
                        "input": self._drop_cot(conversation),  # Drop CoT here to save cost (stays in convo)
                        "timeout": self.timeout,
                        **request_kwargs,
                    }
                    hosted_allowance = 0
                    if self.max_hosted_tool_calls is not None and any(tool.get("type") != "function" for tool in request_tools):
                        hosted_allowance = max(0, self.max_hosted_tool_calls - retry_state.hosted_tool_calls_used)
                        payload["max_tool_calls"] = hosted_allowance
                    if self.background:
                        payload["background"] = self.background
                    if any(tool.get("type") == "code_interpreter" for tool in request_tools):
                        payload["include"] = list(dict.fromkeys([*payload.get("include", []), "code_interpreter_call.outputs"]))
                    ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
                    ts += f".{datetime.now().microsecond:06d}"
                    info = {"nb_executed_tool_calls": nb_executed_tool_calls, "n_retries": n_retries}
                    self._log_provider_request(ts=ts, batch_idx=idx, request=payload, **info)
                    pro_attempt = (request_kwargs.get("reasoning") or {}).get("mode") == "pro"
                    # An interrupted request may already have executed tools remotely.
                    # Return unused allowance only after receiving its final output.
                    retry_state.hosted_tool_calls_used += hosted_allowance
                    def create_response(payload=payload, ts=ts):
                        created = client.responses.create(**payload)
                        self._log_provider_response(ts=ts, batch_idx=idx, response=self._provider_response_dict(created))
                        return created

                    def cancel_late_response(created, ts=ts):
                        if self.background and getattr(created, "id", None):
                            self._cancel_background_response(client, created.id, ts, idx)

                    create_timeout = self._anthropic_request_timeout(inner_start)
                    if self.background:
                        payload["timeout"] = min(60.0, create_timeout)
                    response = self._bounded_provider_operation(
                        create_response, timeout=create_timeout, on_abandoned_result=cancel_late_response,
                    )
                    if self.background:
                        time_start = time.time()
                        # Absolute per-attempt deadline: min of the
                        # per-attempt ``self.timeout`` and the inner-loop
                        # wallclock cap. Computed ONCE before polling.
                        # A previous version recomputed the wallclock
                        # remainder on every retrieve and compared a
                        # shrinking duration against the per-attempt
                        # elapsed; that incorrectly truncated attempt 1
                        # to roughly half the wallclock budget and
                        # produced a cascade of short retries before
                        # the start-of-attempt guard fired.
                        attempt_deadline_at = time_start + self.timeout
                        if self.max_wallclock_per_call_s is not None:
                            attempt_deadline_at = min(
                                attempt_deadline_at,
                                inner_start + self.max_wallclock_per_call_s,
                            )
                        while response.status in {"queued", "in_progress"}:
                            # Sleep in short slices so terminate() can
                            # interrupt the up-to-hours-long poll promptly.
                            for _ in range(60):
                                if self.terminated or time.time() >= attempt_deadline_at:
                                    break
                                time.sleep(1)
                            if self.terminated:
                                response_id = getattr(response, "id", None)
                                if response_id is not None:
                                    recovered = self._cancel_background_response(client, response_id, ts, idx)
                                    if getattr(recovered, "status", None) == "completed":
                                        response, deadline_recovered = recovered, True
                                        break
                                raise _ClientTerminated("APIClient terminated while polling background response.")
                            if time.time() >= attempt_deadline_at:
                                recovered = self._cancel_background_response(client, response.id, ts, idx)
                                if getattr(recovered, "status", None) == "completed":
                                    response, deadline_recovered = recovered, True
                                    break
                                raise _BackgroundResponseTimeout("Timeout waiting for background response.")
                            try:
                                retrieve_kwargs = {"include": payload["include"]} if "include" in payload else {}
                                remaining = max(0.01, attempt_deadline_at - time.time())
                                response_id = response.id
                                retrieve_kwargs["timeout"] = min(30.0, remaining)

                                def retrieve_response(response_id=response_id, kwargs=retrieve_kwargs, ts=ts):
                                    retrieved = client.responses.retrieve(response_id, **kwargs)
                                    self._log_provider_response(ts=ts, batch_idx=idx, response=self._provider_response_dict(retrieved))
                                    return retrieved

                                response = self._bounded_provider_operation(
                                    retrieve_response, timeout=remaining,
                                )
                            except (_ClientTerminated, _ProviderWallclockTimeout) as interrupted:
                                recovered = self._cancel_background_response(client, response_id, ts, idx)
                                if getattr(recovered, "status", None) == "completed":
                                    response, deadline_recovered = recovered, True
                                    break
                                if isinstance(interrupted, _ClientTerminated):
                                    raise
                                raise _BackgroundResponseTimeout("Timeout waiting for background response.") from interrupted
                            except Exception as poll_exc:
                                if _is_terminal_api_error(poll_exc, response_poll=True):
                                    raise
                                # A failed GET does not mean generation failed.
                                # Keep polling this ID, bounded by the same deadline.
                                logger.warning(
                                    f"OpenAI background poll failed for {response.id}; "
                                    f"retaining the response for retry: {poll_exc}"
                                )
                                self._log_provider_response(
                                    ts=ts, batch_idx=idx,
                                    exception={"poll_exception": str(poll_exc), "response_id": response.id},
                                )
                            if time.time() > attempt_deadline_at and response.status in {"queued", "in_progress"}:
                                response_id = getattr(response, "id", None)
                                if response_id is not None:
                                    recovered = self._cancel_background_response(client, response_id, ts, idx)
                                    if getattr(recovered, "status", None) == "completed":
                                        response, deadline_recovered = recovered, True
                                        break
                                raise _BackgroundResponseTimeout("Timeout waiting for background response.")
                        deadline_recovered |= self.terminated or time.time() >= attempt_deadline_at
                        self._log_provider_response(ts=ts, batch_idx=idx, response=self._provider_response_dict(response))
                        # Resilience: handle status="failed" without losing
                        # the partial output. The Responses API marks the
                        # WHOLE response as failed when the LAST tool call
                        # blows a rate-limit (TPM), even though dozens of
                        # earlier output items (reasoning, code_interpreter,
                        # the final assistant message) are present. We want
                        # those — and *prefer* using them to retrying. A
                        # retry on a rate-limit-failed long-running Pro
                        # call risks (a) blowing the wallclock budget on
                        # the next attempt, (b) hitting the same TPM
                        # ceiling because peak reasoning tokens are still
                        # in the rolling window, and (c) discarding a
                        # fully-formed model reply that's already on the
                        # wire. Only retry if the output is actually empty.
                        if response.status == "failed":
                            err = getattr(response, "error", None)
                            err_code = err.get("code") if isinstance(err, dict) else getattr(err, "code", None)
                            err_msg = (err.get("message", "") if isinstance(err, dict) else getattr(err, "message", "")) or ""
                            failed_error = ValueError(
                                f"OpenAI response.status=failed. Error: {err_code} {err_msg[:200]}"
                            )
                            if _is_context_length_error(failed_error):
                                # Do not replay partial tool output into another
                                # oversized request. Usage/output are retained
                                # in the provider trace and raw response log.
                                raise failed_error
                            output_items = getattr(response, "output", None) or []
                            file_limit_error = (
                                str(err_code or "").lower() == "array_above_max_length"
                                or "container has too many files" in err_msg.lower()
                                or "maximum of 1000 files" in err_msg.lower()
                            )
                            if file_limit_error:
                                failed_usage = response.usage
                                if failed_usage is None:
                                    est_in, est_out = _estimate_salvaged_usage(
                                        payload,
                                        output_items,
                                    )
                                    failed_usage = _SalvageUsage(est_in, est_out)
                                (
                                    step_input,
                                    step_output,
                                    step_cached_input,
                                    step_cached_write,
                                ) = self._extract_usage_tokens(failed_usage)
                                step_reasoning = self._extract_reasoning_tokens(
                                    failed_usage
                                )
                                failed_cost = self._get_cost(
                                    step_input,
                                    step_output,
                                    step_cached_input,
                                    step_cached_write,
                                )
                                # Partial tool output cannot make an overfull
                                # container usable. Carry all usage accumulated
                                # by this APIClient call so the Author can bill
                                # the failed attempt before retrying without the
                                # optional attachment.
                                raise ProviderAttachmentRejectedError(
                                    "OpenAI container file limit rejected the request. "
                                    f"Error: {err_code} {err_msg[:200]}",
                                    cost={
                                        "cost": cost_usd + failed_cost,
                                        "input_tokens": input_tokens + step_input,
                                        "cached_input_tokens": (
                                            cached_input_tokens + step_cached_input
                                        ),
                                        "cached_write_tokens": (
                                            cached_write_tokens + step_cached_write
                                        ),
                                        "output_tokens": output_tokens + step_output,
                                        "reasoning_tokens": (
                                            reasoning_tokens + step_reasoning
                                        ),
                                        "time": time.time() - inner_start,
                                        "n_retries": total_retries,
                                    },
                                )
                            if output_items:
                                # The downstream code reads response.usage
                                # to bill the call. The failed response
                                # has usage=None — substitute a char-count
                                # estimate (NOT zero) so the BudgetTracker
                                # actually decrements for the work that
                                # was done. Recording $0 would let the
                                # run march straight into another
                                # expensive round past the budget.
                                est_in, est_out = _estimate_salvaged_usage(payload, output_items)
                                logger.warning(
                                    f"OpenAI response.status=failed but response.output has "
                                    f"{len(output_items)} items; salvaging instead of retrying "
                                    f"(estimated usage: {est_in} in / {est_out} out tokens). "
                                    f"Error: {err_code} {err_msg[:200]}"
                                )
                                if response.usage is None:
                                    response.usage = _SalvageUsage(est_in, est_out)
                            elif self._is_background_server_kill(err_code, time.time() - time_start):
                                raise _BackgroundResponseTimeout(
                                    f"OpenAI background response {getattr(response, 'id', '?')} failed "
                                    f"server-side with no output after {time.time() - time_start:.0f}s "
                                    f"(error={err_code}); treating as a duration kill."
                                )
                            elif err_code == "rate_limit_exceeded" and n_retries < self.max_retries_inner:
                                sleep_s = max(_parse_retry_after_seconds(err_msg) + 1.0, 5.0)
                                logger.warning(
                                    f"OpenAI rate_limit_exceeded after background poll with no "
                                    f"output items; sleeping {sleep_s:.1f}s before retry. "
                                    f"Message: {err_msg[:200]}"
                                )
                                self._record_pro_failure(retry_state, pro_attempt=pro_attempt, idx=idx)
                                self._report_attempt_failure(ts, idx, ValueError(err_code), retry_decision="retry")
                                self._sleep(sleep_s)
                                response = None
                                continue
                            else:
                                raise ValueError(
                                    f"OpenAI response.status=failed with no output items. "
                                    f"Error: {err_code} {err_msg[:200]}"
                                )
                        else:
                            try:
                                response.usage.input_tokens
                            except:
                                raise ValueError("No usage info in response -> if in background, this mean exception occured.")
                    else:
                        self._log_provider_response(ts=ts, batch_idx=idx, response=self._provider_response_dict(response))
                    if hosted_allowance:
                        hosted_used = sum(
                            getattr(item, "type", "") in {"code_interpreter_call", "web_search_call"}
                            for item in response.output
                        )
                        retry_state.hosted_tool_calls_used += hosted_used - hosted_allowance
                except Exception as e:
                    if isinstance(
                        e,
                        (_ClientTerminated, ProviderAttachmentRejectedError),
                    ):
                        raise
                    if isinstance(e, _BackgroundResponseTimeout):
                        retry_state.background_timeouts += 1
                        next_effort = self._background_timeout_retry_reasoning_effort(retry_state.background_timeouts)
                        if next_effort is not None:
                            logger.warning(
                                f"OpenAI background response timed out {retry_state.background_timeouts} time(s); "
                                f"retrying with reasoning.effort={next_effort}."
                            )
                    if "rate limit" not in str(e).lower() and "429" not in str(e):
                        total_retries += 1
                    self._log_provider_response(ts=ts, batch_idx=idx, exception={"exception": str(e)})
                    logger.error(f"Got OpenAI error in responses api inner. Exception: {e}")
                    terminal = _is_terminal_api_error(e)
                    self._report_attempt_failure(ts, idx, e, retry_decision="stop" if terminal else "retry")
                    if terminal:
                        if in_wrapup:
                            response = None
                            break
                        raise
                    self._record_pro_failure(retry_state, pro_attempt=pro_attempt, idx=idx)
                    self._sleep(60)
                    response = None
                    continue
            if response is None:
                if in_wrapup:
                    # Keep the usage and partial output already accrued
                    # instead of raising into an outer retry that would
                    # redo the whole (expensive) call from scratch.
                    logger.warning(
                        "OpenAI max_output_tokens wrap-up turn failed; accepting the truncated output."
                    )
                    if conversation and conversation[-1].get("role") == "user":
                        conversation.pop()
                    break
                raise ValueError("Max inner retries reached.")

            # Update state: token counts and conversation (potentially execute tool calls)
            step_input, step_output, step_cached_input, step_cached_write = self._extract_usage_tokens(response.usage)
            step_reasoning = self._extract_reasoning_tokens(response.usage)
            input_tokens += step_input
            output_tokens += step_output
            cached_input_tokens += step_cached_input
            cached_write_tokens += step_cached_write
            reasoning_tokens += step_reasoning
            cost_usd += self._get_cost(
                step_input,
                step_output,
                step_cached_input,
                step_cached_write,
            )

            was_tool_call_executed = False
            for out in response.output:
                if out.type == "message":
                    all_messages = ""  # need all together + ID because API crashes otherwise
                    for c in out.content:
                        if c.type == "output_text":
                            all_messages += f"{c.text}\n\n"
                    conversation.append({"role": "assistant", "content": all_messages, "id": out.id})
                elif out.type == "code_interpreter_call":
                    status = getattr(out, "status", None) or "completed"
                    if status not in {"in_progress", "interpreting", "completed", "incomplete", "failed"}:
                        status = "failed"
                    outputs = getattr(out, "outputs", None)
                    if outputs is not None:
                        outputs = [item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in outputs]
                        outputs = [{"type": item["type"], field: item[field]}
                                   for item in outputs for field in ("logs", "url")
                                   if item.get("type") == {"logs": "logs", "url": "image"}[field]
                                   and isinstance(item.get(field), str)]
                    conversation.append(
                        {
                            "type": "code_interpreter_call",
                            "id": out.id,
                            "code": out.code,
                            "container_id": out.container_id,
                            "status": status,
                            "outputs": outputs,
                        }
                    )
                elif out.type == "web_search_call":
                    status = getattr(out, "status", None) or "completed"
                    if status not in {"in_progress", "searching", "completed", "failed"}:
                        status = "completed"
                    conversation.append({"type": "web_search_call", "id": out.id, "status": status})
                elif out.type == "function_call":
                    if deadline_recovered or self.terminated:
                        continue  # Retain usage/output, but never execute post-deadline work.
                    function_name = out.name
                    arguments = json.loads(out.arguments)
                    tool_func = self.tool_functions[function_name]
                    tool_key = "any" if max_tool_calls_mode == "total" else function_name
                    if delivery_only() and function_name not in {"read_context", "publish_artifact", "publish_sandbox_artifact"}:
                        output = "Delivery interval: new research is disabled. Publish existing work and finish."
                    elif nb_executed_tool_calls[tool_key] >= max_tool_calls[tool_key]:
                        output = f"Error: Tool call after exceeding max # of tool calls ({max_tool_calls[tool_key]})."
                    else:
                        try:
                            output = self._execute_tool_function(
                                function_name, arguments, conversation,
                                call_deadline_monotonic_s=self._tool_call_deadline(inner_start),
                            )
                        except Exception as e:
                            logger.error(f"Error executing tool {function_name}. Exception: {e}")
                            output = f"Error executing tool {function_name}. Exception: {e}"
                    if isinstance(output, tuple):
                        additional_cost = output[1]
                        input_tokens += additional_cost["input_tokens"]
                        output_tokens += additional_cost["output_tokens"]
                        cost_usd += float(
                            additional_cost.get(
                                "cost",
                                self._get_cost(
                                    additional_cost["input_tokens"],
                                    additional_cost["output_tokens"],
                                ),
                            )
                        )
                        output = output[0]
                    conversation.append(
                        {
                            "type": "function_call",
                            "id": out.id,
                            "call_id": out.call_id,
                            "arguments": out.arguments,
                            "name": out.name,
                        }
                    )
                    was_tool_call_executed = True
                    nb_executed_tool_calls[tool_key] += 1
                    nb_tool_calls_left = max_tool_calls[tool_key] - nb_executed_tool_calls[tool_key]
                    detail = "for this tool" if max_tool_calls_mode == "per_tool" else "(across all tools)"
                    if self.include_max_tool_calls:
                        info = f"\n\n### INFO ###\nYou have {nb_tool_calls_left} tool executions left {detail}."
                    else:
                        info = ""
                    conversation.append(
                        {"type": "function_call_output", "call_id": out.call_id, "output": str(output) + info}
                    )
                elif out.type == "reasoning":
                    """
                    API change - each code_interpreter_call now comes with this block beforehand and if you don't
                    keep it it crashes. So we keep it intact (so it does not get stripped in _drop_cot
                    and clean_conversation later will do what the code below used to do).
                    Still need to convert to dict.
                    """
                    reasoning_block = {
                        "id": out.id,
                        "summary": [{"text": b.text, "type": b.type} for b in out.summary],
                        "type": out.type,
                        # "status": out.status,
                    }
                    encrypted_content = getattr(out, "encrypted_content", None)
                    if encrypted_content is not None:
                        reasoning_block["encrypted_content"] = encrypted_content
                    # NOTE: you get status from OpenAI but if you send it back to them they crash out
                    conversation.append(reasoning_block)
                    # summary = ""
                    # for thought in out.summary:
                    #    if thought.text is not None:
                    #        summary += "<thought>" + "\n" + thought.text + "\n" + "</thought>\n"
                    # conversation.append({"role": "assistant", "type": "cot", "content": summary, "id": out.id})
                else:
                    raise ValueError(f"Unknown output type {out.type}")

            if deadline_recovered:
                _drop_trailing_reasoning_items(conversation)
                break
            if _openai_response_hit_max_output_tokens(response):
                dropped = _drop_trailing_reasoning_items(conversation)
                if (
                    self.openai_continue_on_max_output_tokens
                    and max_output_token_continuations < self.openai_max_output_token_continuations
                    and not self.terminated
                ):
                    max_output_token_continuations += 1
                    in_wrapup = True
                    logger.warning(
                        f"OpenAI response hit max_output_tokens; dropped {dropped} trailing reasoning "
                        f"item(s), requesting wrap-up turn {max_output_token_continuations}/"
                        f"{self.openai_max_output_token_continuations}."
                    )
                    conversation.append({"role": "user", "content": _OPENAI_MAX_OUTPUT_TOKENS_WRAPUP_PROMPT})
                    continue
                logger.warning(
                    f"OpenAI response hit max_output_tokens after {max_output_token_continuations} "
                    f"wrap-up turn(s); accepting the truncated output."
                )
                break

            # If nothing was run this was the last iteration, stop
            if not was_tool_call_executed or self.terminated:
                break

        if conversation[-1].get("type", "") == "reasoning":
            raise ValueError("Conversation ended with reasoning block.")
        # Recovery prompts and tool items do not constitute a new answer.
        # Prevent consumers from falling back to an earlier invocation's reply.
        if not any(m.get("role") == "assistant" for m in conversation[len(messages):]):
            conversation.append({"role": "assistant", "content": ""})
        return self.InternalRequestResult(
            conversation,
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
            cached_write_tokens=cached_write_tokens,
            reasoning_tokens=reasoning_tokens,
            n_retries=total_retries,
            cost_usd=cost_usd,
        )

    def _openai_query_chat_completions_api(self, client, idx, messages, ignore_tool_calls=False):
        """Queries the OpenAI API using chat completions API.

        Args:
            client: The OpenAI client.
            idx (int): The index of the query in the batch of queries given to run_queries.
            messages (list): The messages to send.
            ignore_tool_calls (bool): Whether to ignore tool calls.

        Returns:
            InternalRequestResult or None
        """
        if self.stream_openai_chat_completions:
            return self._openai_query_chat_completions_streaming(client, idx, messages)

        # Set up tools
        if ignore_tool_calls:
            max_tool_calls_mode, max_tool_calls = "total", {"any": 0}
        else:
            max_tool_calls_mode, max_tool_calls = self.max_tool_calls_mode, self.max_tool_calls

        # State
        total_max_tool_calls = sum(max_tool_calls.values())
        if max_tool_calls_mode == "total":
            nb_executed_tool_calls = {"any": 0}
        else:
            nb_executed_tool_calls = {t: 0 for t in self.tool_functions.keys()}
        conversation = [m.copy() for m in messages]
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        cached_write_tokens = 0
        reasoning_tokens = 0
        cost_usd = 0.0
        total_retries = 0
        max_output_tokens = self.kwargs.get(self.max_tokens_param, None)
        inner_start = time.time()

        # As long as we just had a tool response do another request
        was_tool_call_executed = True
        while was_tool_call_executed and not self.terminated:
            was_tool_call_executed = False
            # Inner retry to get a response
            response = None
            n_retries = -1
            while response is None and n_retries < self.max_retries_inner:
                n_retries += 1
                if self.max_wallclock_per_call_s is not None:
                    if time.time() - inner_start >= self.max_wallclock_per_call_s:
                        raise ValueError(
                            f"Wallclock budget ({self.max_wallclock_per_call_s}s) "
                            f"exhausted in inner openai chat-completions retry loop "
                            f"after {n_retries} attempts."
                        )
                try:
                    kwargs = self.kwargs.copy()
                    kwargs[self.max_tokens_param] = max_output_tokens
                    payload = {
                        "model": self.model,
                        "messages": self._drop_cot(conversation),  # Drop CoT here to save cost (stays in convo)
                        "tools": self.tool_descriptions if len(self.tool_descriptions) > 0 else None,
                        "timeout": self.timeout,
                        **kwargs,
                    }
                    ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
                    ts += f".{datetime.now().microsecond:06d}"
                    info = {"nb_executed_tool_calls": nb_executed_tool_calls, "n_retries": n_retries}
                    self._log_provider_request(ts=ts, batch_idx=idx, request=payload, **info)
                    response = client.chat.completions.create(**payload)
                    self._log_provider_response(ts=ts, batch_idx=idx, response=self._provider_response_dict(response))
                except Exception as e:
                    if "rate limit" not in str(e).lower() and "429" not in str(e):
                        total_retries += 1
                    self._log_provider_response(ts=ts, batch_idx=idx, response={"exception": str(e)})
                    if isinstance(e, RateLimitError):
                        logger.info(f"Got OpenAI CC rate limit error. Sleeping for 60 seconds. Exception: {e}")
                        self._sleep(60)
                        continue
                    if _is_terminal_api_error(e, context_length_recovery=True):
                        raise
                    if _is_context_length_error(e):
                        if max_output_tokens is None or max_output_tokens <= 1:
                            raise
                        max_output_tokens = max_output_tokens // 2
                        logger.info(
                            f"Got OpenAI CC max context length error. Reducing max output tokens to {max_output_tokens} and retrying. Exception: {e}"
                        )
                    logger.info(f"Got OpenAI CC non ratelimit error. Sleeping for 20 seconds: {e}")
                    self._sleep(60)
                    continue
            if response is None:
                raise ValueError("Max inner retries reached.")

            # Update state: token counts and conversation (potentially execute tool calls)
            step_input, step_output, step_cached_input, step_cached_write = (
                self._extract_usage_tokens(response.usage)
            )
            step_reasoning = self._extract_reasoning_tokens(response.usage)
            input_tokens += step_input
            output_tokens += step_output
            cached_input_tokens += step_cached_input
            cached_write_tokens += step_cached_write
            reasoning_tokens += step_reasoning
            cost_usd += self._get_cost(
                step_input,
                step_output,
                step_cached_input,
                step_cached_write,
            )
            message = response.choices[0].message
            if self.context_limit is not None:
                max_output_tokens = self.context_limit
                if max_output_tokens is not None:
                    max_output_tokens -= response.usage.total_tokens
                    max_output_tokens = min(max_output_tokens, 
                                            self.kwargs.get(self.max_tokens_param, float("inf")))
            else:
                max_output_tokens = self.kwargs.get(self.max_tokens_param, None)
            # Add CoT and rest of message separately
            # TODO: if we notice new ways to return CoT they should be mangled here. Likely missing something.
            if hasattr(message, "reasoning_details") and message.reasoning_details:
                for detail in message.reasoning_details:
                    if isinstance(detail, dict):
                        type_ = detail.get("type", "")
                        text_ = detail.get("text", "")
                        sumary_ = detail.get("sumary", "")
                    else:
                        type_ = detail.type
                        text_ = detail.text
                        sumary_ = detail.sumary
                    if type_ == "reasoning.text":
                        conversation.append({"role": "assistant", "type": "cot", "content": text_})
                    elif type_ == "reasoning.summary":
                        conversation.append({"role": "assistant", "type": "cot", "content": sumary_})
                    else:
                        pass  # encrypted
            else:
                # Reasoning details trumps others as there is repetition
                if hasattr(message, "reasoning") and message.reasoning:
                    conversation.append({"role": "assistant", "type": "cot", "content": message.reasoning})
                if hasattr(message, "reasoning_content") and message.reasoning_content:
                    conversation.append({"role": "assistant", "type": "cot", "content": message.reasoning_content})

            message_dict = message.model_dump()
            message_dict = {k: v for k, v in message_dict.items() if v is not None}  # Drop nulls
            conversation.append(message_dict)  # Should have tool calls inside too; and reasoning_details

            # Try to execute all tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if isinstance(tool_call, dict):
                        function_name = tool_call.get("function", dict()).get("name", "")
                        tool_call_id = tool_call.get("id")
                    else:
                        function_name = tool_call.function.name
                        tool_call_id = tool_call.id
                    if function_name not in self.tool_functions:
                        logger.warning(f"Tool {function_name} not found.")
                        conversation.append(
                            {"role": "tool", "tool_name": function_name, "tool_call_id": tool_call_id, "content": "Error: Tool not found."}
                        )
                        continue

                    # If no budget return error
                    # NOTE: just erroring out here might stop the request loop but the model will be given last chance.
                    tool_key = "any" if max_tool_calls_mode == "total" else function_name
                    if nb_executed_tool_calls[tool_key] >= max_tool_calls[tool_key]:
                        error = f"Error: Exceeded maximum number of tool calls ({max_tool_calls[tool_key]})."
                        conversation.append({"role": "tool", "tool_call_id": tool_call.id, "content": error})
                    else:
                        # Execute tool
                        arguments = json.loads(tool_call.function.arguments)
                        try:
                            output = self._execute_tool_function(
                                function_name, arguments, conversation,
                                call_deadline_monotonic_s=self._tool_call_deadline(inner_start),
                            )
                        except Exception as e:
                            logger.error(f"Error executing tool {function_name}. Exception: {e}")
                            output = f"Error executing tool {function_name}. Exception: {e}"
                        # Tools can return additional cost
                        if isinstance(output, tuple):
                            output, extra_cost = output
                            input_tokens += extra_cost["input_tokens"]
                            output_tokens += extra_cost["output_tokens"]
                            cost_usd += float(
                                extra_cost.get(
                                    "cost",
                                    self._get_cost(
                                        extra_cost["input_tokens"],
                                        extra_cost["output_tokens"],
                                    ),
                                )
                            )

                        nb_executed_tool_calls[tool_key] += 1

                        nb_tool_calls_left = max_tool_calls[tool_key] - nb_executed_tool_calls[tool_key]
                        detail = "for this tool" if max_tool_calls_mode == "per_tool" else "(across all tools)"
                        info = f"\n\n### INFO ###\nYou have {nb_tool_calls_left} tool executions left {detail}."
                        conversation.append(
                            {
                                "role": "tool",
                                "tool_name": function_name,
                                "tool_call_id": tool_call.id,
                                "content": str(output) + info,
                            }
                        )

                    was_tool_call_executed = True

                if total_max_tool_calls <= sum(nb_executed_tool_calls.values()):
                    if conversation[-1]["role"] != "assistant":
                        conversation.append({"role": "assistant", "content": "Assistant executed more tools than allowed. \\boxed{None}"})
                    break  # No more tool calls allowed, stop here

        if total_max_tool_calls > 0:
            logger.info(f"Finished on a loop without tool calls, after executing {nb_executed_tool_calls} calls total.")

        return self.InternalRequestResult(
            self.clean_cot_from_conversation(conversation),
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
            cached_write_tokens=cached_write_tokens,
            reasoning_tokens=reasoning_tokens,
            n_retries=total_retries,
            cost_usd=cost_usd,
        )

    def _openai_query_chat_completions_streaming(self, client, idx, messages):
        """Queries the OpenAI API using chat completions API with streaming.

        Tool calls are ignored in streaming mode; we only aggregate the final text.
        """
        conversation = [m.copy() for m in messages]
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        cached_write_tokens = 0
        reasoning_tokens = 0
        total_retries = 0
        max_output_tokens = self.kwargs.get(self.max_tokens_param, None)
        inner_start = time.time()

        response = None
        n_retries = -1
        while response is None and n_retries < self.max_retries_inner:
            n_retries += 1
            if self.max_wallclock_per_call_s is not None:
                if time.time() - inner_start >= self.max_wallclock_per_call_s:
                    raise ValueError(
                        f"Wallclock budget ({self.max_wallclock_per_call_s}s) "
                        f"exhausted in inner openai streaming retry loop after "
                        f"{n_retries} attempts."
                    )
            try:
                kwargs = self.kwargs.copy()
                kwargs[self.max_tokens_param] = max_output_tokens
                payload = {
                    "model": self.model,
                    "messages": self._drop_cot(conversation),
                    "timeout": self.timeout,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    **kwargs,
                }
                ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
                ts += f".{datetime.now().microsecond:06d}"
                self._log_provider_request(ts=ts, batch_idx=idx, request=payload, n_retries=n_retries)
                response = client.chat.completions.create(**payload)
            except Exception as e:
                if "rate limit" not in str(e).lower() and "429" not in str(e):
                    total_retries += 1
                self._log_provider_response(ts=ts, batch_idx=idx, response={"exception": str(e)})
                if isinstance(e, RateLimitError):
                    logger.info(f"Got OpenAI CC rate limit error. Sleeping for 60 seconds. Exception: {e}")
                    self._sleep(60)
                    continue
                if _is_terminal_api_error(e, context_length_recovery=True):
                    raise
                if _is_context_length_error(e):
                    if max_output_tokens is None or max_output_tokens <= 1:
                        raise
                    max_output_tokens = max_output_tokens // 2
                    logger.info(
                        f"Got OpenAI CC max context length error. Reducing max output tokens to {max_output_tokens} and retrying. Exception: {e}"
                    )
                logger.info(f"Got OpenAI CC non ratelimit error. Sleeping for 20 seconds: {e}")
                self._sleep(60)
                continue
        if response is None:
            raise ValueError("Max inner retries reached.")

        text_parts = []
        reasoning_parts = []
        final_usage = None
        for event in response:
            if getattr(event, "usage", None) is not None:
                final_usage = event.usage
            if not getattr(event, "choices", None):
                continue
            delta = event.choices[0].delta
            if getattr(delta, "content", None):
                text_parts.append(delta.content)
            if getattr(delta, "reasoning", None):
                reasoning_parts.append(delta.reasoning)

        reasoning = "".join(reasoning_parts)
        if reasoning:
            conversation.append({"role": "assistant", "type": "cot", "content": reasoning})
        content = "".join(text_parts)   
        conversation.append({"role": "assistant", "content": content})

        if final_usage is not None:
            input_tokens, output_tokens, cached_input_tokens, cached_write_tokens = (
                self._extract_usage_tokens(final_usage)
            )
            reasoning_tokens = self._extract_reasoning_tokens(final_usage)

        self._log_provider_response(
            ts=ts,
            batch_idx=idx,
            response={
                "streamed": True,
                "content": content,
                "reasoning": reasoning,
                "usage": final_usage.model_dump() if final_usage is not None else None,
            },
        )

        return self.InternalRequestResult(
            conversation,
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
            cached_write_tokens=cached_write_tokens,
            reasoning_tokens=reasoning_tokens,
            n_retries=total_retries,
        )

    def clean_cot_from_conversation(self, conversation):
        """Cleans CoT from conversation for cost saving purposes.

        Args:
            conversation (list): The conversation to clean.
        Returns:
            list: The cleaned conversation.
        """
        cleaned_conversation = []
        for message in conversation:
            new_message = message.copy()
            if "reasoning_details" in new_message:
                del new_message["reasoning_details"]
            if "reasoning" in new_message:
                del new_message["reasoning"]
            if "reasoning_content" in new_message:
                del new_message["reasoning_content"]
            cleaned_conversation.append(new_message)
        return cleaned_conversation

    def _google_generation_config(self):
        """``generationConfig`` for the native generateContent path.

        Model configs carry the thinking settings in the OpenAI-compatible
        ``extra_body.extra_body.google.thinking_config`` (snake_case) form;
        the native API wants them camelCased under ``thinkingConfig``.
        """
        config = {}
        if self.max_tokens is not None:
            config["maxOutputTokens"] = int(self.max_tokens)
        extra_body = self.kwargs.get("extra_body") or {}
        google_extra = (extra_body.get("extra_body") or {}).get("google") or {}
        thinking = google_extra.get("thinking_config") or google_extra.get("thinkingConfig")
        if isinstance(thinking, dict) and thinking:
            config["thinkingConfig"] = {
                re.sub(r"_([a-z])", lambda m: m.group(1).upper(), str(k)): v
                for k, v in thinking.items()
            }
        return config

    def _google_query_with_internal_tools(self, idx, messages, *, _final_answer_only=False):
        """Query native generateContent, including prior conversation turns."""
        system_messages = [m for m in messages if m.get("role") in ("developer", "system")]
        contents = []
        for msg in messages:
            role = msg.get("role")
            if role in ("developer", "system"):
                continue
            if "google_parts" in msg:
                role, parts = "model", msg["google_parts"]
            elif role in ("user", "assistant"):
                content = msg.get("content", "")
                if not isinstance(content, str):
                    raise ValueError("Native Gemini history requires text message content.")
                role = "model" if role == "assistant" else "user"
                parts = [{"text": content}]
            else:
                raise ValueError(f"Unsupported native Gemini history role: {role!r}")
            if not parts:
                continue
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": list(parts)})
        if not contents or contents[-1]["role"] != "user":
            raise ValueError("Native Gemini history must end with a user request.")

        conversation = [m.copy() for m in messages]

        headers = {
            "x-goog-api-key": f"{self.api_key}",
            "Content-Type": "application/json",
        }

        function_declarations = []
        google_tools = []
        has_web_search = False
        for tool_desc in self.tool_descriptions:
            if tool_desc.get("type") == "function" and "function" in tool_desc:
                fn = tool_desc["function"]
                function_declarations.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
            elif tool_desc.get("type") in ("web_search", "web_search_preview"):
                google_tools.append({"googleSearch": {}})
                has_web_search = True
            elif tool_desc.get("type") == "code_interpreter":
                google_tools.append({"codeExecution": {}})
            elif "google_search" in tool_desc:
                google_tools.append({"googleSearch": tool_desc["google_search"]})
                has_web_search = True
            else:
                google_tools.append(tool_desc)
                if "googleSearch" in tool_desc:
                    has_web_search = True
        if function_declarations:
            google_tools.append({"functionDeclarations": function_declarations})

        payload = {"contents": contents}
        if system_messages:
            payload["systemInstruction"] = {
                "parts": [{"text": m["content"]} for m in system_messages]
            }
        if google_tools:
            payload["tools"] = google_tools
        if _final_answer_only:
            payload.pop("tools", None)
        generation_config = self._google_generation_config()
        if generation_config:
            payload["generationConfig"] = generation_config

        ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
        ts += f".{datetime.now().microsecond:06d}"
        self._log_provider_request(ts=ts, batch_idx=idx, request=payload)

        timeout = self._anthropic_request_timeout(time.time())
        def generate():
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=timeout)
            self._log_provider_response(ts=ts, batch_idx=idx, response=response.json())
            return response

        response = self._bounded_provider_operation(generate, timeout=timeout)

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        json_response = response.json()

        if "candidates" not in json_response:
            raise Exception(f"Error: {json_response}")

        candidate = json_response["candidates"][0]
        message = candidate.get("content") or {}
        parts = message.get("parts") or []
        role = message.get("role", "model")
        assert role == "model"
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        if _final_answer_only and not any(part.get("text", "").strip() for part in parts if not part.get("thought")):
            raise _ToolRecoveryFailed(f"Gemini wrap-up returned no final text ({finish_reason})")

        if finish_reason == "TOO_MANY_TOOL_CALLS" and not _final_answer_only:
            # One bounded wrap-up request, with hosted tools physically absent.
            # Reuse actual findings, not a claim that the tool loop succeeded.
            findings = []
            size = 0
            for part in reversed(parts):
                encoded_size = len(json.dumps(part))
                if size + encoded_size > 120_000:
                    continue
                findings.insert(0, part)
                size += encoded_size
            recovery_messages = [*messages]
            if findings:
                recovery_messages.append({"role": "assistant", "google_parts": findings})
            recovery_messages.append({"role": "user", "content": (
                "The hosted-tool limit ended the previous attempt. Tools are disabled. "
                "Using only results actually obtained, write the requested final council reply now. "
                "State any remaining gaps; do not claim unexecuted computations succeeded."
            )})
            try:
                recovered = self._google_query_with_internal_tools(idx, recovery_messages, _final_answer_only=True)
            except Exception as exc:
                raise _ToolRecoveryFailed(f"Gemini tools-disabled recovery failed: {exc}") from exc
            usage = json_response.get("usageMetadata") or {}
            inputs = int(usage.get("promptTokenCount", 0) or 0) + int(usage.get("toolUsePromptTokenCount", 0) or 0)
            reasoning = int(usage.get("thoughtsTokenCount", 0) or 0)
            outputs = int(usage.get("candidatesTokenCount", 0) or 0) + reasoning
            cached = int(usage.get("cachedContentTokenCount", 0) or 0)
            recovered.cost_usd = (recovered.cost_usd if recovered.cost_usd is not None else self._get_cost(
                recovered.input_tokens, recovered.output_tokens, recovered.cached_input_tokens,
            )) + self._get_cost(inputs, outputs, cached)
            recovered.input_tokens += inputs
            recovered.output_tokens += outputs
            recovered.cached_input_tokens += cached
            recovered.reasoning_tokens += reasoning
            return recovered

        usage_metadata = json_response.get("usageMetadata") or {}
        input_tokens = int(usage_metadata.get("promptTokenCount", 0) or 0)
        # Built-in tool rounds (codeExecution / googleSearch) re-prompt the
        # model; Gemini bills those tokens as input but reports them apart
        # from promptTokenCount (a council seat showed 727k such tokens next
        # to a 56k prompt).
        input_tokens += int(usage_metadata.get("toolUsePromptTokenCount", 0) or 0)
        output_tokens = int(usage_metadata.get("candidatesTokenCount", 0) or 0)
        # Gemini bills thinking tokens at the output rate but reports them
        # separately from candidatesTokenCount; fold them into the billable
        # output total (reasoning_tokens below keeps the separate metric).
        output_tokens += int(usage_metadata.get("thoughtsTokenCount", 0) or 0)
        # cachedContentTokenCount is the cache-hit subset of promptTokenCount.
        cached_input_tokens = int(usage_metadata.get("cachedContentTokenCount", 0) or 0)
        # Google's native API reports thinking budget as a separate
        # ``thoughtsTokenCount`` field (NOT included in
        # candidatesTokenCount, unlike OpenAI's output_tokens). Surface
        # it so the First-Proof token report can record reasoning per
        # call for Gemini just like the other providers.
        reasoning_tokens = int(usage_metadata.get("thoughtsTokenCount", 0) or 0)

        def _clear_buffer(conversation, buffer, mode):
            if buffer is None or len(buffer) == 0:
                return
            if mode == "cot":
                conversation.append({"role": "assistant", "type": "cot", "content": buffer})
            elif mode == "response":
                conversation.append({"role": "assistant", "type": "response", "content": buffer})
            else:
                return

        mode = None
        buffer = ""
        if has_web_search and candidate.get("groundingMetadata"):
            conversation.append({"role": "assistant", "type": "web_search_call", "query": messages[0]["content"]})
        for part in parts:
            if "thought" in part and part["thought"]:
                if mode != "cot":
                    _clear_buffer(conversation, buffer, mode)
                    buffer = ""
                    mode = "cot"
                buffer += f"{part['text']}"
            elif "functionCall" in part:
                _clear_buffer(conversation, buffer, mode)
                buffer = ""
                mode = None
                conversation.append(
                    {
                        "type": "tool_call",
                        "role": "assistant",
                        "call_id": "1",
                        "arguments": part["functionCall"]["args"],
                        "name": part["functionCall"]["name"],
                    }
                )
            elif "executableCode" in part:
                _clear_buffer(conversation, buffer, mode)
                buffer = ""
                mode = None
                conversation.append(
                    {
                        "type": "code_interpreter_call",
                        "role": "assistant",
                        "language": part["executableCode"].get("language", ""),
                        "code": part["executableCode"].get("code", ""),
                    }
                )
            elif "codeExecutionResult" in part:
                _clear_buffer(conversation, buffer, mode)
                buffer = ""
                mode = None
                conversation.append(
                    {
                        "type": "code_interpreter_result",
                        "role": "tool",
                        "outcome": part["codeExecutionResult"].get("outcome", ""),
                        "content": part["codeExecutionResult"].get("output", ""),
                    }
                )
            elif "functionResponse" in part:
                _clear_buffer(conversation, buffer, mode)
                buffer = ""
                mode = None
                name = part["functionResponse"]["name"]
                resp = part["functionResponse"]["response"]
                if "result" in resp:
                    result = resp["result"]
                elif "content" in resp:
                    result = resp["content"]
                else:
                    result = ""
                conversation.append({"role": "tool", "tool_name": f"{name}", "tool_call_id": "1", "content": result})
            elif "text" in part:
                if mode != "response":
                    _clear_buffer(conversation, buffer, mode)
                    buffer = ""
                    mode = "response"
                buffer += f"{part['text']}"
        _clear_buffer(conversation, buffer, mode)

        if len(conversation) == len(messages) or conversation[-1].get("type") != "response":
            conversation.append({"role": "assistant", "type": "response", "content": ""})
        # Display traces are not separate native turns. Replay the original
        # parts once, unchanged (including tool parts and thought signatures).
        for msg in conversation[len(messages):]:
            msg["google_parts"] = []
        conversation[-1]["google_parts"] = parts

        return self.InternalRequestResult(
            conversation=conversation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
        )
