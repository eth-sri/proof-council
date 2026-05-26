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
import json
import re
import time
from typing import Any, ClassVar

from pydantic import BaseModel

from proofstack.agent import Agent
from proofstack.context import ModelSpec
from proofstack.events import new_call_id

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
        result = await asyncio.to_thread(_one_shot_query, client, client_messages)
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
        self.tracker.add_usd(usd)
        self.tracker.add_tokens(in_tok + out_tok)
        await self.events.emit(
            "model.call",
            {
                "model": getattr(client, "model", str(self.MODEL)),
                "in_tokens": in_tok,
                "out_tokens": out_tok,
                "reasoning_tokens": reasoning_tok,
                "cost_usd": usd,
                "duration_s": elapsed,
            },
            call_id=call_id,
        )

        # Best-effort post-call check (raises if we just blew a limit).
        post_warnings = self.tracker.check()
        for scope, kind, used, limit in post_warnings:
            await self.events.emit(
                "budget.warn",
                {"scope": scope, "kind": kind, "used": used, "limit": limit},
            )

        raw_text = _assistant_text(conversation)
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
        from mathagents import load_solver_config

        cfg = load_solver_config(spec)
        cfg = {k: v for k, v in cfg.items() if not k.startswith("__")}
        cfg.update(self.extra_client_kwargs())
        return self.ctx.api_client_factory(cfg)

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
