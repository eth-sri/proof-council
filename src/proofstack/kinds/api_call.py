"""APICallAgent — single mathagents.APIClient call wrapped as an Agent.

Subclasses set ``SYSTEM_PROMPT`` (optional), ``USER_PROMPT`` (template
formatted with ``Inputs.model_dump(mode='json')``), and ``MODEL`` (a
config reference understood by ``mathagents.config_loader``).

Defaults cover the trivial case (one input field substituted into a
template, one output field holding the assistant's text). Override
``render_messages``, ``parse_output``, or ``extra_client_kwargs`` for
richer behavior.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import re
import time
from typing import Any, ClassVar

from pydantic import BaseModel

from proofstack.agent import Agent
from proofstack.budget import BudgetExhausted, budget_overrun_allowed
from proofstack.context import ModelSpec
from proofstack.events import new_call_id
from proofstack.provider_accounting import record_provider_usage

Message = dict[str, Any]


class APICallAgent(Agent):
    """One-shot API call against a single model.

    Class-level config (override in subclass):
      - ``SYSTEM_PROMPT``: optional system / developer message
      - ``USER_PROMPT``: template, formatted with ``Inputs.model_dump(mode='json')``
        via ``str.format(**fields)``
      - ``MODEL``: config ref understood by ``mathagents.load_solver_config``
    """

    SYSTEM_PROMPT: ClassVar[str | None] = None
    USER_PROMPT: ClassVar[str] = "{problem}"
    MODEL: ClassVar[ModelSpec] = "models/openai/gpt-54"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client: Any | None = None

    # --- subclass hooks --------------------------------------------------------

    def render_messages(self, inp: BaseModel) -> list[Message]:
        fields = inp.model_dump(mode="json")
        user_text = self.USER_PROMPT.format(**fields)
        msgs: list[Message] = []
        if self.SYSTEM_PROMPT:
            msgs.append({"role": "developer", "content": self.SYSTEM_PROMPT})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def parse_output(self, raw_text: str, inp: BaseModel) -> BaseModel:
        # Fallback: drop raw text into the single non-``reasoning`` field.
        # If that field is named ``solution``, still sanitize LaTeX cruft
        # so a downstream wrapper can't end up with nested documents or
        # undefined ``proof`` environments.
        target = _single_string_field(self.Outputs)
        body = raw_text
        if target == "solution":
            body = _sanitize_solution_body(body)
        return self.Outputs.model_validate({target: body})

    def extra_client_kwargs(self) -> dict[str, Any]:
        return {}

    def _on_response(self, raw_text: str, inp: BaseModel) -> None:
        """Observe a completed reply synchronously, before logging awaits.

        Overrides must not raise or perform I/O. This is for callers that
        must retain a reply even when later bookkeeping is interrupted.
        """

    def _on_provider_response(self, client, trace) -> None:
        """Optional bounded, synchronous artifact capture during provider polling."""

    # --- framework-managed ----------------------------------------------------

    async def run(self, inp: BaseModel) -> BaseModel:
        warnings = self.tracker.check()
        for scope, kind, used, limit in warnings:
            await self.events.emit(
                "budget.warn",
                {"scope": scope, "kind": kind, "used": used, "limit": limit},
            )

        messages = self.render_messages(inp)
        try:
            (self.workdir / "messages.json").write_text(
                json.dumps(messages, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError:
            pass
        client_messages = self._messages_with_tool_context(messages)
        client = await self._get_client()

        call_id = new_call_id()
        await self.events.emit(
            "model.call.start",
            {"model": getattr(client, "model", str(self.MODEL))},
            call_id=call_id,
        )

        start = time.monotonic()
        result = await self._query(client, client_messages, _one_shot_query, call_id=call_id)
        elapsed = time.monotonic() - start

        # result: (idx, conversation, detailed_cost)
        _idx, conversation, cost = result
        usd = float(cost.get("cost", 0.0))
        in_tok = int(cost.get("input_tokens", 0) or 0)
        out_tok = int(cost.get("output_tokens", 0) or 0)
        # First Proof spec requires per-call reasoning tokens. APIClient
        # surfaces them on ``detailed_cost`` when the provider reports
        # them (OpenAI Responses/Chat-Completions reasoning models,
        # Gemini thinking). 0 when not reported.
        reasoning_tok = int(cost.get("reasoning_tokens", 0) or 0)
        raw_text = _assistant_text(conversation)
        self._on_response(raw_text, inp)
        await self._record_model_usage(
            {
                "model": getattr(client, "model", str(self.MODEL)),
                "in_tokens": in_tok,
                "out_tokens": out_tok,
                "reasoning_tokens": reasoning_tok,
                "cost_usd": usd,
                "duration_s": elapsed,
                "status": "completed" if raw_text.strip() else "empty",
                "provider_outcomes": cost.get("provider_outcomes", []),
                "usage_unavailable": cost.get("usage_unavailable", False),
            },
            call_id=call_id,
        )

        # Best-effort post-call check (raises if we just blew a limit).
        try:
            post_warnings = self.tracker.check()
        except BudgetExhausted as e:
            # The call has already completed and been charged. A caller that
            # branches on the reply (WriteupLoop's UNABLE: catastrophe signal)
            # must not lose it just because the charge crossed a limit, so
            # attach it. The raise itself is unchanged.
            with contextlib.suppress(Exception):
                e.completed_output = self.parse_output(
                    _assistant_text(conversation), inp)
            raise
        for scope, kind, used, limit in post_warnings:
            await self.events.emit(
                "budget.warn",
                {"scope": scope, "kind": kind, "used": used, "limit": limit},
            )

        if not raw_text.strip():
            await self.events.emit(
                "model.empty_response",
                {
                    "type": "EmptyResponse",
                    "msg": f"model {getattr(client, 'model', '?')} returned an empty response",
                },
                call_id=call_id,
            )
        return self.parse_output(raw_text, inp)

    async def _query(self, client, messages, query, *, call_id=None):
        from mathagents.provider_trace import ProviderTrace, active_trace

        call_id = call_id or new_call_id()
        # Authors may reuse a client across rounds; construction-time caps alone
        # would give a late call the first round's much larger timeout.
        remaining = self.tracker.remaining_wallclock_s()
        if remaining is not None and not budget_overrun_allowed():
            self.tracker.check()
            for key in ("timeout", "max_wallclock_per_call_s"):
                if hasattr(client, key):
                    configured = getattr(client, key)
                    setattr(client, key, min(float(configured), remaining) if configured is not None else remaining)
        loop = asyncio.get_running_loop()

        def attempt_failed(payload):
            # APIClient runs in a worker thread. Flush the small diagnostic
            # before its retry sleep so monitors need not wait for query exit.
            if loop.is_closed():
                return
            emission = self.events.emit("model.attempt.failed", payload, call_id=call_id)
            try:
                pending = asyncio.run_coroutine_threadsafe(emission, loop)
            except RuntimeError:
                emission.close()
                raise
            try:
                pending.result(timeout=5)
            except Exception:
                pending.cancel()
                raise

        trace = ProviderTrace(self.workdir / "provider-attempts.jsonl",
                              run_id=self.ctx.root_workdir.name, agent=self.workdir.name, call_id=call_id,
                              on_failure=attempt_failed, on_response=self._on_provider_response)
        token = active_trace.set(trace)
        task = asyncio.create_task(asyncio.to_thread(query, client, messages))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            # Cancelling to_thread does not stop its provider polling loop.
            with contextlib.suppress(Exception):
                client.terminate()
            # Give background cancellation a bounded chance to persist its
            # acknowledgement and final usage; never wait for a stuck transport.
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
            if trace.completed_report:
                exc.completed_report = trace.completed_report
            try:
                await self._charge_failed_provider_trace(trace, client, call_id, "cancelled")
            except Exception as accounting_error:
                # Cancellation was already delivered before settlement began.
                # A failed append must not turn this stop into a retryable error.
                raise exc from accounting_error
            with contextlib.suppress(Exception):
                await self.events.emit("model.call.cancelled", {
                    "model": getattr(client, "model", str(self.MODEL)),
                    "usage_unavailable": not trace.totals()["provider_attempts"] or trace.totals()["usage_unavailable"],
                    "note": "Cancellation requested; provider charges may still apply.",
                }, call_id=call_id)
            raise
        except Exception as exc:
            if isinstance(getattr(exc, "cost", None), dict):
                # Author owns attachment-rejection charging; enrich its receipt
                # rather than charge the same response twice.
                totals = trace.totals()
                for name, value in totals.items():
                    if name in exc.cost and isinstance(value, (int, float)):
                        exc.cost[name] = max(exc.cost[name] or 0, value)
            else:
                await self._charge_failed_provider_trace(trace, client, call_id, "failed")
            raise
        finally:
            self._provider_tool_evidence = trace.tool_evidence()
            active_trace.reset(token)
            # Retrieve a late exception without making shutdown await that task.
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)

    async def _charge_failed_provider_trace(self, trace, client, call_id, status):
        totals = trace.totals()
        if not totals["provider_attempts"]:
            return
        await self._record_model_usage({
            "model": getattr(client, "model", str(self.MODEL)),
            "cost_usd": totals["cost"], "in_tokens": totals["input_tokens"],
            "out_tokens": totals["output_tokens"], "reasoning_tokens": totals["reasoning_tokens"],
            "status": status, "usage_unavailable": totals["usage_unavailable"],
            "invocation_id": totals["invocation_id"],
            "provider_outcomes": totals["provider_outcomes"],
        }, call_id=call_id)

    async def _record_model_usage(self, payload, *, call_id):
        await record_provider_usage(self.ctx, self.tracker, payload, call_id=call_id, emitter=self.events)

    def _limit_client_deadline(self, cfg):
        remaining = self.tracker.remaining_wallclock_s()
        if remaining is not None and not budget_overrun_allowed():
            for key in ("timeout", "max_wallclock_per_call_s"):
                # Leave absent defaults to APIClient; _query caps the actual
                # client immediately before use without widening those defaults.
                if key in cfg:
                    configured = cfg[key]
                    cfg[key] = min(float(configured), remaining) if configured is not None else remaining
        return cfg

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        spec = self.ctx.model_for(self, self.MODEL)
        # APIClient construction touches network credentials and may sleep
        # on cold starts; offload to a thread.
        self._client = await asyncio.to_thread(self._build_client, spec)
        return self._client

    def _build_client(self, spec: ModelSpec) -> Any:
        """Build the APIClient with ``extra_client_kwargs`` merged in.

        We always go through a config dict so subclass overrides like
        ``tools=[(None, {"type": "web_search_preview"})]`` and
        ``max_tool_calls=N`` actually reach ``APIClient.__init__`` —
        post-hoc ``setattr`` would not reconfigure the tool loop.
        """
        from mathagents import APIClient, load_solver_config

        cfg = load_solver_config(spec)
        cfg = {k: v for k, v in cfg.items() if not k.startswith("__")}
        extra = dict(self.extra_client_kwargs())
        defaults = inspect.signature(APIClient).parameters
        for key in ("timeout", "max_wallclock_per_call_s"):
            if key in extra and extra[key] is not None:
                configured = cfg.get(key, defaults[key].default)
                if configured is not None:
                    extra[key] = min(float(configured), float(extra[key]))
        cfg.update(extra)
        return self.ctx.api_client_factory(self._limit_client_deadline(cfg))

    def _messages_with_tool_context(self, messages: list[Message]) -> list[Message]:
        copied = [msg.copy() for msg in messages]
        context = {
            "persisted_file_root": str(self.ctx.root_workdir / "persisted_files"),
        }
        for msg in copied:
            if msg.get("role") == "user":
                existing = msg.get("tool_context") if isinstance(msg.get("tool_context"), dict) else {}
                msg["tool_context"] = {**existing, **context}
                break
        return copied


# --- helpers -----------------------------------------------------------------


def _one_shot_query(client: Any, messages: list[Message]) -> tuple[int, list[Message], dict]:
    """Drain a single APIClient.run_queries iteration."""
    iterator = client.run_queries([messages], no_tqdm=True)
    return next(iter(iterator))


def _assistant_text(conversation: list[Message]) -> str:
    """Pick the last assistant turn's text content from a conversation."""
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


def _single_string_field(model_cls: type[BaseModel]) -> str:
    """Pick the single non-reasoning string field of an Outputs model."""
    for name, info in model_cls.model_fields.items():
        if name == "reasoning":
            continue
        return name
    raise TypeError(f"{model_cls.__name__} has no fields to receive raw text")


_TAG_CACHE: dict[tuple[str, ...], list[re.Pattern[str]]] = {}

# Models routinely confuse the XML closing tag with a LaTeX
# ``\end{solution}`` environment marker, drop the opening ``<solution>``
# entirely and emit ``\begin{proof}…\end{proof}`` instead, or wrap their
# answer in a full ``\documentclass…\begin{document}…\end{document}``.
# These fallbacks try the strict tag first, then progressively looser
# patterns, and finally sanitize the body so downstream LaTeX wrapping
# does not get nested-environment errors.

_FALLBACK_PATTERNS_FOR_TAG: dict[str, list[re.Pattern[str]]] = {
    "solution": [
        # Opening tag present, closing tag confused with \end{solution}.
        re.compile(r"<solution>(?P<body>.*?)\\end\{solution\}", re.DOTALL),
        # Opening missing/typo'd, closing mangled.
        re.compile(r"\\begin\{solution\}(?P<body>.*?)\\end\{solution\}", re.DOTALL),
        # No XML/env markers at all — pull the proof body.
        re.compile(r"\\begin\{proof\}(?P<body>.*?)\\end\{proof\}", re.DOTALL),
    ],
}

# Strippable LaTeX cruft that sometimes leaks into the extracted body
# even after a successful match (e.g. a stray ``\end{solution}`` after
# the model's actual closing tag, or a ``\documentclass`` echoed inside
# the answer). Order matters: strip preamble before environment markers.
_LATEX_CRUFT_STRIPPERS: list[re.Pattern[str]] = [
    re.compile(r"\\documentclass\b\s*(?:\[[^\]]*\])?\s*(?:\{[^}]*\})?\s*"),
    re.compile(r"\\usepackage\b\s*(?:\[[^\]]*\])?\s*\{[^}]*\}\s*"),
    re.compile(r"\\title\s*\{[^}]*\}\s*"),
    re.compile(r"\\author\s*\{[^}]*\}\s*"),
    re.compile(r"\\date\s*\{[^}]*\}\s*"),
    re.compile(r"\\maketitle\b\s*"),
    re.compile(r"\\(?:begin|end)\{document\}\s*"),
    re.compile(r"\\(?:begin|end)\{solution\}\s*"),
    re.compile(r"\\(?:begin|end)\{proof\}\s*"),
    re.compile(r"</?solution>\s*"),
]


def _sanitize_solution_body(body: str) -> str:
    """Strip preamble / document / proof-env cruft from an extracted body.

    Defensive: compile nodes usually wrap proof bodies in an ``article``
    preamble. If the model echoed any wrappers in its answer, the wrapped
    result would have nested ``\\begin{document}`` or undefined ``proof``
    environments.
    """
    cleaned = body
    for pat in _LATEX_CRUFT_STRIPPERS:
        cleaned = pat.sub("", cleaned)
    return cleaned.strip()


def _extract_xml_tags(text: str, tags: tuple[str, ...]) -> dict[str, str]:
    if tags not in _TAG_CACHE:
        _TAG_CACHE[tags] = [
            re.compile(rf"<{t}>(?P<body>.*?)</{t}>", re.DOTALL) for t in tags
        ]
    out: dict[str, str] = {}
    for tag, pat in zip(tags, _TAG_CACHE[tags]):
        m = pat.search(text)
        if m is None:
            for fb in _FALLBACK_PATTERNS_FOR_TAG.get(tag, ()):
                m = fb.search(text)
                if m is not None:
                    break
        if m is not None:
            body = m.group("body")
            if tag == "solution":
                body = _sanitize_solution_body(body)
            else:
                body = body.strip()
            out[tag] = body
    return out


__all__ = ["APICallAgent"]
