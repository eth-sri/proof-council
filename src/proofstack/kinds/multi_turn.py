"""MultiTurnAgent — model-directed tool loop with sub-agents-as-tools.

Subclasses declare a ``tools`` list of either ``Agent`` instances or
plain callables. Each ``Agent`` is exposed to the model as a tool whose
JSON schema is auto-derived from its ``Inputs.model_json_schema()`` and
whose description is the agent's ``description``.

Implementation: we lean on ``mathagents.APIClient``'s built-in tool
loop (``tools=...`` + ``max_tool_calls > 0``). Async sub-agents are
bridged into the synchronous tool callbacks via
``asyncio.run_coroutine_threadsafe``, so context vars (parent call id,
agent path) are preserved end-to-end.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import Any, Callable, ClassVar

from pydantic import BaseModel

from proofstack.agent import Agent
from proofstack.context import ModelSpec
from proofstack.events import new_call_id
from proofstack.kinds.api_call import _assistant_text


ToolEntry = Agent | Callable


class MultiTurnAgent(Agent):
    """A loop of model call → tool calls → … terminating on final answer.

    Subclass overrides:
      - ``SYSTEM_PROMPT`` (optional)
      - ``USER_PROMPT``  — formatted with ``Inputs.model_dump(mode='json')``
      - ``MODEL``        — config ref understood by ``mathagents.load_solver_config``
      - ``MAX_STEPS``    — upper bound on tool calls per run
      - ``tools``        — property returning a list of Agent or callable

    Subclasses may override ``parse_final(messages, inp)`` for richer
    Outputs parsing; the default puts the final assistant text into the
    single non-``reasoning`` Outputs field.
    """

    SYSTEM_PROMPT: ClassVar[str | None] = None
    USER_PROMPT: ClassVar[str] = "{problem}"
    MODEL: ClassVar[ModelSpec] = "models/openai/gpt-54"
    MAX_STEPS: ClassVar[int] = 30

    @property
    def tools(self) -> list[ToolEntry]:
        return []

    # --- subclass hooks --------------------------------------------------------

    def render_initial_messages(self, inp: BaseModel) -> list[dict[str, Any]]:
        fields = inp.model_dump(mode="json")
        user_text = self.USER_PROMPT.format(**fields)
        msgs: list[dict[str, Any]] = []
        if self.SYSTEM_PROMPT:
            msgs.append({"role": "developer", "content": self.SYSTEM_PROMPT})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def parse_final(self, messages: list[dict[str, Any]], inp: BaseModel) -> BaseModel:
        text = _assistant_text(messages)
        for name, _info in self.Outputs.model_fields.items():
            if name == "reasoning":
                continue
            return self.Outputs.model_validate({name: text})
        raise TypeError(f"{type(self).__name__}.Outputs has no fields to receive final text")

    def extra_client_kwargs(self) -> dict[str, Any]:
        return {}

    # --- framework-managed -----------------------------------------------------

    async def run(self, inp: BaseModel) -> BaseModel:  # type: ignore[override]
        loop = asyncio.get_running_loop()
        spec = self.ctx.model_for(self, self.MODEL)
        tool_pairs = self._build_tool_pairs(loop)

        client = await asyncio.to_thread(
            self._build_client, spec, tool_pairs
        )

        messages = self._messages_with_tool_context(self.render_initial_messages(inp))

        await self.events.emit(
            "multiturn.start",
            {"model": getattr(client, "model", str(spec)), "n_tools": len(tool_pairs), "max_steps": self.MAX_STEPS},
        )
        start = time.monotonic()
        # APIClient drives the entire tool loop internally.
        result = await asyncio.to_thread(_one_query, client, messages)
        elapsed = time.monotonic() - start

        _idx, conversation, cost = result
        usd = float(cost.get("cost", 0.0))
        in_tok = int(cost.get("input_tokens", 0) or 0)
        out_tok = int(cost.get("output_tokens", 0) or 0)
        self.tracker.add_usd(usd)
        self.tracker.add_tokens(in_tok + out_tok)
        await self.events.emit(
            "multiturn.end",
            {
                "model": getattr(client, "model", str(spec)),
                "in_tokens": in_tok,
                "out_tokens": out_tok,
                "cost_usd": usd,
                "duration_s": elapsed,
                "n_messages": len(conversation),
            },
        )

        # Best-effort budget check after the multi-turn block.
        warnings = self.tracker.check()
        for scope, kind, used, limit in warnings:
            await self.events.emit(
                "budget.warn",
                {"scope": scope, "kind": kind, "used": used, "limit": limit},
            )

        try:
            (self.workdir / "conversation.json").write_text(
                json.dumps(conversation, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError:
            pass

        return self.parse_final(conversation, inp)

    # --- helpers --------------------------------------------------------------

    def _build_client(self, spec: ModelSpec, tool_pairs: list[tuple]) -> Any:
        if isinstance(spec, dict):
            cfg = dict(spec)
        else:
            from mathagents import load_solver_config

            cfg = load_solver_config(spec)
        cfg = {k: v for k, v in cfg.items() if not k.startswith("__")}
        cfg["tools"] = tool_pairs
        cfg["max_tool_calls"] = self.MAX_STEPS
        cfg.update(self.extra_client_kwargs())
        from mathagents import APIClient

        return APIClient(**cfg)

    def _build_tool_pairs(self, loop: asyncio.AbstractEventLoop) -> list[tuple]:
        pairs = []
        for tool in self.tools:
            if isinstance(tool, Agent):
                pairs.append(_agent_to_tool_pair(tool, loop))
            elif callable(tool):
                pairs.append(_callable_to_tool_pair(tool, loop))
            else:
                raise TypeError(f"unsupported tool entry: {tool!r}")
        return pairs

    def _messages_with_tool_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _one_query(client, messages):
    """Drain the single-result iterator returned by APIClient.run_queries."""
    for tup in client.run_queries([messages], no_tqdm=True):
        return tup
    raise RuntimeError("APIClient.run_queries yielded no result")


def _agent_to_tool_pair(agent: Agent, loop: asyncio.AbstractEventLoop):
    """Wrap a sub-agent as a (sync_callable, openai_tool_desc) pair.

    The sync callable bridges into the parent event loop via
    ``run_coroutine_threadsafe``. This preserves context vars and
    ensures the sub-agent's events thread under the parent agent's
    call_id.
    """

    schema = agent.Inputs.model_json_schema()
    schema.setdefault("type", "object")

    def sync_tool(**kwargs):
        fut = asyncio.run_coroutine_threadsafe(agent(**kwargs), loop)
        out = fut.result()
        if isinstance(out, BaseModel):
            return out.model_dump_json()
        return json.dumps(out, default=str, ensure_ascii=False)

    desc = {
        "type": "function",
        "function": {
            "name": agent.name,
            "description": agent.description or f"Sub-agent {type(agent).__name__}",
            "parameters": schema,
        },
    }
    return sync_tool, desc


def _callable_to_tool_pair(fn: Callable, loop: asyncio.AbstractEventLoop):
    """Wrap a plain callable as a (sync_callable, openai_tool_desc) pair.

    Coroutine functions are bridged via ``run_coroutine_threadsafe``;
    sync functions are passed through. The JSON schema is derived from
    the function's signature (best effort — annotated typed params are
    reflected; ``Any`` and missing annotations become free-form strings).
    """

    name = getattr(fn, "__name__", "tool")
    description = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else f"Tool {name}"
    sig = inspect.signature(fn)
    schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        annotation = param.annotation if param.annotation is not inspect._empty else str
        schema["properties"][pname] = _json_schema_for(annotation)
        if param.default is inspect._empty:
            schema["required"].append(pname)

    if asyncio.iscoroutinefunction(fn):
        def sync_tool(**kwargs):
            fut = asyncio.run_coroutine_threadsafe(fn(**kwargs), loop)
            res = fut.result()
            if isinstance(res, BaseModel):
                return res.model_dump_json()
            return json.dumps(res, default=str, ensure_ascii=False)
    else:
        def sync_tool(**kwargs):
            res = fn(**kwargs)
            if isinstance(res, BaseModel):
                return res.model_dump_json()
            if isinstance(res, str):
                return res
            return json.dumps(res, default=str, ensure_ascii=False)

    return sync_tool, {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }


_PRIMITIVE_TO_JSON: dict[Any, dict[str, str]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}


def _json_schema_for(annotation: Any) -> dict[str, Any]:
    if annotation in _PRIMITIVE_TO_JSON:
        return dict(_PRIMITIVE_TO_JSON[annotation])
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.model_json_schema()
    return {"type": "string"}


__all__ = ["MultiTurnAgent"]
