"""Agent base class.

Every agent is a typed async function ``Inputs -> Outputs``. The
``__call__`` wrapper handles event bracketing, deterministic resume
caching, per-call workdir allocation, and budget setup; subclasses
only override ``run``.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from proofstack.budget import BudgetExhausted, BudgetSpec, BudgetTracker
from proofstack.context import RunContext
from proofstack.events import (
    EventEmitter,
    _AGENT_PATH,
    _CURRENT_WORKDIR,
    _PARENT_CALL_ID,
    new_call_id,
)


ExecutionMode = Literal["workflow", "agent", "deterministic_tool", "human_assisted"]


_CONFIG_ATTR_ALIASES: dict[str, str] = {
    "system_prompt": "SYSTEM_PROMPT",
    "user_prompt": "USER_PROMPT",
    "model": "MODEL",
    "sandbox": "SANDBOX",
}

_CACHE_CONFIG_ATTRS: tuple[str, ...] = (
    "SYSTEM_PROMPT",
    "USER_PROMPT",
    "MODEL",
    "SANDBOX",
)


class Agent(ABC):
    """Async function from Inputs to Outputs.

    Subclasses define:
      - ``Inputs``:  Pydantic model of declared input channels
      - ``Outputs``: Pydantic model of declared output channels
      - ``run()``:   the actual work

    Set ``description`` (used as a tool description when this agent is
    exposed to a parent MultiTurnAgent) and ``execution_mode`` (used
    for cost attribution and UI grouping).
    """

    Inputs: ClassVar[type[BaseModel]]
    Outputs: ClassVar[type[BaseModel]]

    description: ClassVar[str] = ""
    execution_mode: ClassVar[ExecutionMode] = "agent"
    default_budget: ClassVar[BudgetSpec | None] = None
    # Whether ``__call__`` should consult and write the resume_cache. Subclasses
    # whose effects are not captured by their JSON Outputs (e.g. CLIAgent
    # variants that mutate a persistent on-disk workspace) must set this to
    # False to avoid silently replaying stale outputs against an empty
    # workspace on rerun.
    cache_enabled: ClassVar[bool] = True

    def __init__(
        self,
        ctx: RunContext,
        *,
        name: str | None = None,
        budget: BudgetSpec | None = None,
        parent_budget_scope: str = "run",
    ):
        self.ctx = ctx
        self.name = name or type(self).__name__
        self.component_config: dict[str, Any] = ctx.component_config_for(self)
        config_budget = _budget_from_config(self.component_config)
        self.budget: BudgetSpec | None = budget or config_budget or self.default_budget
        self._apply_component_config(self.component_config)
        self.events: EventEmitter = ctx.events.scoped(agent=self.name)
        parent = ctx.budgets.get(parent_budget_scope) or ctx.budgets.root("run")
        self.tracker: BudgetTracker = ctx.budgets.child(
            scope=f"agent:{self.name}", parent=parent, spec=self.budget
        )
        # ``self.workdir`` reads from the per-call contextvar set in
        # ``__call__``. The fallback below is only allocated on demand
        # (lazy) for the rare case the property is read outside of any
        # ``__call__``, so we don't burn an empty directory per Agent.
        self._fallback_workdir: Path | None = None

    @property
    def workdir(self) -> Path:
        wd = _CURRENT_WORKDIR.get()
        if isinstance(wd, Path):
            return wd
        if self._fallback_workdir is None:
            self._fallback_workdir = self.ctx.workdir_for(self)
        return self._fallback_workdir

    @abstractmethod
    async def run(self, inp: BaseModel) -> BaseModel: ...

    async def __call__(self, **kwargs: Any) -> BaseModel:
        inp = self.Inputs(**kwargs)
        call_id = new_call_id()
        cache_key = self._cache_key(inp)
        # Capture the caller's call_id BEFORE we shadow it; events
        # emitted at the boundary (start, end, cache_hit, error) must
        # carry the caller as parent_call_id, not ourselves.
        parent_call_id = _PARENT_CALL_ID.get()

        if type(self).cache_enabled:
            cached = self.ctx.resume_cache.get(cache_key)
            if cached is not None:
                await self.events.emit(
                    "agent.cache_hit",
                    {"key": cache_key},
                    call_id=call_id,
                    execution_mode=type(self).execution_mode,
                    parent_call_id=parent_call_id,
                )
                return self._coerce_output(cached)

        # Allocate a fresh workdir per invocation. This is the fix for
        # parallel re-use of the same agent instance overwriting its
        # input/output checkpoints.
        workdir = self.ctx.workdir_for(self, call_id=call_id)
        inp_json = inp.model_dump(mode="json")
        self._persist_input(inp_json, workdir)

        await self.events.emit(
            "agent.start",
            {"input": inp_json},
            call_id=call_id,
            execution_mode=type(self).execution_mode,
            parent_call_id=parent_call_id,
        )

        token_workdir = _CURRENT_WORKDIR.set(workdir)
        token_parent = _PARENT_CALL_ID.set(call_id)
        token_path = _AGENT_PATH.set(_AGENT_PATH.get() + (self.name,))
        try:
            try:
                out = await self.run(inp)
            except BudgetExhausted as e:
                # Reset contextvars first so error event records the
                # caller as parent_call_id.
                _PARENT_CALL_ID.reset(token_parent)
                _AGENT_PATH.reset(token_path)
                _CURRENT_WORKDIR.reset(token_workdir)
                token_parent = token_path = token_workdir = None  # type: ignore[assignment]
                await self.events.emit(
                    "agent.error",
                    {"type": "BudgetExhausted", "msg": str(e), "scope": e.scope},
                    call_id=call_id,
                    parent_call_id=parent_call_id,
                )
                raise
            except Exception as e:
                _PARENT_CALL_ID.reset(token_parent)
                _AGENT_PATH.reset(token_path)
                _CURRENT_WORKDIR.reset(token_workdir)
                token_parent = token_path = token_workdir = None  # type: ignore[assignment]
                await self.events.emit(
                    "agent.error",
                    {"type": type(e).__name__, "msg": str(e)},
                    call_id=call_id,
                    parent_call_id=parent_call_id,
                )
                raise
        finally:
            for tok, var in (
                (token_path, _AGENT_PATH),
                (token_parent, _PARENT_CALL_ID),
                (token_workdir, _CURRENT_WORKDIR),
            ):
                if tok is not None:
                    try:
                        var.reset(tok)
                    except (LookupError, ValueError):
                        pass

        out_json = self._dump_output(out)
        if type(self).cache_enabled:
            self.ctx.resume_cache.put(cache_key, out_json)
        await self._persist_output(out, workdir)
        await self.events.emit(
            "agent.end",
            {"output": out_json},
            call_id=call_id,
            execution_mode=type(self).execution_mode,
            parent_call_id=parent_call_id,
        )
        return out

    # --- helpers --------------------------------------------------------------

    def _cache_key(self, inp: BaseModel) -> str:
        payload = {
            "agent_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "agent_name": self.name,
            "agent_config": self._cache_config_snapshot(),
            "inputs": inp.model_dump(mode="json"),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        return digest

    def _coerce_output(self, raw: Any) -> BaseModel:
        return self.Outputs.model_validate(raw)

    def _dump_output(self, out: Any) -> Any:
        if isinstance(out, BaseModel):
            return out.model_dump(mode="json")
        return out

    def _apply_component_config(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            if key == "budget" or key.startswith("__"):
                continue
            self._apply_component_config_value(str(key), value)

    def _apply_component_config_value(self, key: str, value: Any) -> None:
        attr = _CONFIG_ATTR_ALIASES.get(key, key)
        if attr == "SANDBOX" and isinstance(value, dict):
            from proofstack.sandbox.base import SandboxSpec

            value = SandboxSpec(**value)
        if attr.isupper() or hasattr(type(self), attr):
            setattr(self, attr, value)

    def _cache_config_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        if self.component_config:
            snapshot["component_config"] = self.component_config
        for attr in _CACHE_CONFIG_ATTRS:
            if hasattr(self, attr):
                try:
                    snapshot[attr] = getattr(self, attr)
                except Exception:
                    continue
        return snapshot

    def _persist_input(self, inp_json: Any, workdir: Path) -> None:
        try:
            (workdir / "input.json").write_text(
                json.dumps(inp_json, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError:
            pass

    async def _persist_output(self, out: Any, workdir: Path) -> None:
        try:
            (workdir / "output.json").write_text(
                json.dumps(self._dump_output(out), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError:
            pass


__all__ = ["Agent", "ExecutionMode"]


def _budget_from_config(config: dict[str, Any]) -> BudgetSpec | None:
    raw = config.get("budget")
    if raw is None:
        return None
    if isinstance(raw, BudgetSpec):
        return raw
    if isinstance(raw, dict):
        return BudgetSpec(**raw)
    raise TypeError("component config 'budget' must be a mapping")
