"""Optional LLM run monitor for live dashboard summaries."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from proofstack.context import ModelSpec, RunContext
from proofstack.events import new_call_id


DEFAULT_MONITOR_MODEL: ModelSpec = "models/openai/gpt-54-mini"


@dataclass
class RunMonitor:
    ctx: RunContext
    model: ModelSpec = DEFAULT_MONITOR_MODEL
    problem: str = ""
    problem_id: str = ""
    workflow_structure: Any = None
    max_output_chars: int = 5000
    summaries: list[dict[str, Any]] = field(default_factory=list)
    _client: Any | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.model = normalize_monitor_model_spec(self.model)

    def schedule_agent_end(
        self,
        *,
        call_id: str,
        agent: str,
        agent_path: str,
        execution_mode: str | None,
        input_json: Any,
        output_json: Any,
        status: str = "ok",
        error: dict[str, Any] | None = None,
    ) -> None:
        task = asyncio.create_task(
            self.record_agent_end(
                call_id=call_id,
                agent=agent,
                agent_path=agent_path,
                execution_mode=execution_mode,
                input_json=input_json,
                output_json=output_json,
                status=status,
                error=error,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(_consume_task_exception)

    async def drain(self) -> None:
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def record_agent_end(
        self,
        *,
        call_id: str,
        agent: str,
        agent_path: str,
        execution_mode: str | None,
        input_json: Any,
        output_json: Any,
        status: str = "ok",
        error: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            await self._record(
                call_id=call_id,
                agent=agent,
                agent_path=agent_path,
                execution_mode=execution_mode,
                input_json=input_json,
                output_json=output_json,
                status=status,
                error=error,
            )

    async def _record(
        self,
        *,
        call_id: str,
        agent: str,
        agent_path: str,
        execution_mode: str | None,
        input_json: Any,
        output_json: Any,
        status: str,
        error: dict[str, Any] | None,
    ) -> None:
        monitor_call_id = new_call_id()
        display_agent = self._display_agent(agent=agent, agent_path=agent_path)
        prompt = self._prompt(
            agent=agent,
            agent_path=agent_path,
            display_agent=display_agent,
            execution_mode=execution_mode,
            input_json=input_json,
            output_json=output_json,
            status=status,
            error=error,
        )
        messages = [
            {
                "role": "developer",
                "content": (
                    "You summarize an agentic math workflow for a human watching it live. "
                    "Write exactly 3-4 concise sentences. Do not invent results; say what changed, "
                    "what the finished node appears to have done, and what seems important next. "
                    "Use user-facing node labels only; never mention internal identifiers, component names, "
                    "file paths, or strings containing 'DAGWorkflow'."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            client = await self._get_client()
            await self.ctx.events.emit(
                "model.call.start",
                {"model": getattr(client, "model", str(self.model)), "via": "run_monitor"},
                call_id=monitor_call_id,
                parent_call_id=call_id,
            )
            start = time.monotonic()
            _idx, conversation, cost = await asyncio.to_thread(_one_shot_query, client, messages)
            elapsed = time.monotonic() - start
            summary = _assistant_text(conversation).strip()
            usd = float(cost.get("cost", 0.0) or 0.0)
            in_tokens = int(cost.get("input_tokens", 0) or 0)
            out_tokens = int(cost.get("output_tokens", 0) or 0)
            self.ctx.budgets.root("run").add_usd(usd)
            self.ctx.budgets.root("run").add_tokens(in_tokens + out_tokens)
            await self.ctx.events.emit(
                "model.call",
                {
                    "model": getattr(client, "model", str(self.model)),
                    "in_tokens": in_tokens,
                    "out_tokens": out_tokens,
                    "cost_usd": usd,
                    "duration_s": elapsed,
                    "via": "run_monitor",
                },
                call_id=monitor_call_id,
                parent_call_id=call_id,
            )
        except Exception as e:
            summary = f"Monitor failed after {display_agent}: {type(e).__name__}: {e}"
            await self.ctx.events.emit(
                "monitor.error",
                {"agent": agent, "type": type(e).__name__, "msg": str(e)},
                call_id=monitor_call_id,
                parent_call_id=call_id,
            )

        item = {
            "agent": agent,
            "display_label": display_agent,
            "call_id": call_id,
            "status": status,
            "summary": summary,
        }
        self.summaries.append(item)
        await self.ctx.events.emit(
            "monitor.summary",
            item,
            call_id=monitor_call_id,
            parent_call_id=call_id,
        )

    async def _get_client(self) -> Any:
        if self._client is None:
            self._client = await asyncio.to_thread(self.ctx.api_client_factory, self.model)
        return self._client

    def _prompt(
        self,
        *,
        agent: str,
        agent_path: str,
        display_agent: str,
        execution_mode: str | None,
        input_json: Any,
        output_json: Any,
        status: str,
        error: dict[str, Any] | None,
    ) -> str:
        context = {
            "problem_id": self.problem_id,
            "problem": _trim_text(self.problem, 2500),
            "workflow_structure": self._workflow_structure_for_prompt(),
            "finished": {
                "agent": display_agent,
                "agent_type": _humanize_identifier(agent),
                "execution_mode": execution_mode,
                "status": status,
                "input": _trim_value(input_json, self.max_output_chars),
                "output": _trim_value(output_json, self.max_output_chars),
                "error": error,
            },
            "previous_summaries": [
                {
                    "agent": item.get("display_label") or _humanize_identifier(item.get("agent")),
                    "status": item.get("status"),
                    "summary": item.get("summary"),
                }
                for item in self.summaries
            ],
        }
        return (
            "Live monitor context follows as JSON. Summarize what the viewer should understand now.\n\n"
            f"{json.dumps(context, ensure_ascii=False, indent=2, default=str)}"
        )

    def _display_agent(self, *, agent: str, agent_path: str) -> str:
        labels = self._workflow_node_labels()
        for part in reversed([p for p in str(agent_path or "").split(".") if p]):
            if part in labels:
                return labels[part]
        if agent == "DAGWorkflow" or str(agent_path or "").endswith("DAGWorkflow"):
            return "Workflow"
        return _humanize_identifier(agent or agent_path or "Monitor update")

    def _workflow_node_labels(self) -> dict[str, str]:
        if not isinstance(self.workflow_structure, dict):
            return {}
        labels: dict[str, str] = {}
        nodes = self.workflow_structure.get("nodes")
        if not isinstance(nodes, list):
            return labels
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "").strip()
            if not node_id:
                continue
            labels[node_id] = str(node.get("label") or _humanize_identifier(node_id))
        return labels

    def _workflow_structure_for_prompt(self) -> Any:
        if not isinstance(self.workflow_structure, dict):
            return self.workflow_structure
        labels = self._workflow_node_labels()
        nodes = []
        raw_nodes = self.workflow_structure.get("nodes")
        if isinstance(raw_nodes, list):
            for node in raw_nodes:
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("id") or "")
                item: dict[str, Any] = {
                    "label": str(node.get("label") or labels.get(node_id) or _humanize_identifier(node_id)),
                    "kind": node.get("kind", "agent"),
                }
                needs = [
                    labels.get(str(dep), _humanize_identifier(dep))
                    for dep in (node.get("needs") or [])
                ]
                if needs:
                    item["after"] = needs
                nodes.append(item)
        return {
            "workflow": self.workflow_structure.get("workflow"),
            "description": self.workflow_structure.get("description", ""),
            "nodes": nodes,
            "outputs": self.workflow_structure.get("outputs") or [],
        }


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


def _one_shot_query(client: Any, messages: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    return next(iter(client.run_queries([messages], no_tqdm=True)))


def _assistant_text(conversation: list[dict[str, Any]]) -> str:
    for msg in reversed(conversation):
        if msg.get("role") == "assistant" and msg.get("type") != "cot":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
                )
    return ""


def _trim_value(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return _trim_text(text, max_chars)


def _trim_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _humanize_identifier(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Monitor update"
    text = text.rsplit(".", 1)[-1]
    return text.replace("_", " ").strip().title() or "Monitor update"


def normalize_monitor_model_spec(model: ModelSpec) -> ModelSpec:
    if not isinstance(model, str):
        return model
    raw = model.strip()
    aliases = {
        "gpt-5.6": "models/openai/gpt-56-sol",
        "gpt-5.6-sol": "models/openai/gpt-56-sol",
        "gpt-5.6-sol--max": "models/openai/gpt-56-sol-max",
        "gpt-5.6-sol-pro": "models/openai/gpt-56-sol-pro",
        "gpt-5.5": "models/openai/gpt-54-mini",
        "gpt-5.5-mini--low": "models/openai/gpt-54-mini",
        "gpt-5.5-mini-low": "models/openai/gpt-54-mini",
        "gpt-5.5-pro": "models/openai/gpt-55-pro",
        "openai/gpt-55-mini": "models/openai/gpt-54-mini",
        "models/openai/gpt-55-mini": "models/openai/gpt-54-mini",
        "gpt-5.5-pro--xhigh": "models/openai/gpt-55-pro",
    }
    if raw in aliases:
        return aliases[raw]
    if raw and "/" in raw and not raw.startswith(("models/", "/", ".")) and not raw.endswith(".yaml"):
        return f"models/{raw}"
    return raw or DEFAULT_MONITOR_MODEL


__all__ = ["DEFAULT_MONITOR_MODEL", "RunMonitor", "normalize_monitor_model_spec"]
