"""Human-in-the-loop agent.

A node that treats a human like any other (very slow) model: it is handed
the same ``inputs`` every node receives, surfaces a task, then blocks until a
human submits a response. Because it honours the standard ``Inputs -> Outputs``
contract it is interchangeable with model nodes — drop it into a solver,
referee, or hint slot and the rest of the graph is unaffected.

Mechanism (mirrors ``CLIAgent``'s ``finish`` handshake):
  - ``run()`` writes a ``*.task.json`` into ``<run>/human_inbox/`` describing
    the rendered prompt, the raw inputs, and the output fields expected back.
  - it emits a ``human.waiting`` event carrying the response path so a UI (or a
    person at a shell) knows where to drop the answer.
  - it polls for the matching ``*.response.json`` until it appears or the run's
    wallclock budget runs out, then returns the submitted values as Outputs.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from proofstack.agent import Agent


class HumanAgent(Agent):
    """YAML-configurable human-input node."""

    description: ClassVar[str] = "Hand a task to a human and wait for their response."
    execution_mode: ClassVar[str] = "human_assisted"
    # A human's answer is not reproducible and is not captured by config, so
    # A human's answer is expensive and irreplaceable, so cache it: on resume the
    # prior answer is replayed instead of re-asked. (Opt out per component with
    # cache_enabled: false if you ever want to force a fresh ask on resume.)
    cache_enabled: ClassVar[bool] = True

    POLL_INTERVAL_S: ClassVar[float] = 2.0
    HEARTBEAT_INTERVAL_S: ClassVar[float] = 30.0
    # How long to wait for a human before giving up. Independent of (and far
    # larger than) the run's compute wallclock budget — a person may take days.
    DEFAULT_HUMAN_TIMEOUT_S: ClassVar[float] = 7 * 24 * 3600.0

    PALETTE: ClassVar[dict[str, str]] = {
        "id": "human",
        "label": "Human",
        "group": "Proof Work",
        "description": "Hand the task to a human and wait for their typed answer (treats you as a node).",
        "keywords": "human in the loop person manual input wait reviewer solver",
    }

    @classmethod
    def default_component_config(cls) -> dict[str, Any]:
        return {
            "prompt": (
                "Please complete this task and submit your answer.\n\n"
                "Problem:\n{problem}\n"
            ),
            "input_schema": {"problem": "string"},
            "output_schema": {"answer_tex": "string", "status": "string"},
        }

    class Inputs(BaseModel):
        model_config = ConfigDict(extra="allow")

        workspace: str | None = Field(default=None)

    class Outputs(BaseModel):
        model_config = ConfigDict(extra="allow")

        status: str = Field(default="done")

    async def run(self, inp: BaseModel) -> BaseModel:  # type: ignore[override]
        fields = inp.model_dump(mode="json")
        prompt = _format_template(str(self.component_config.get("prompt") or ""), fields)
        output_fields = self._output_fields()

        inbox = self.ctx.root_workdir / "human_inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        stem = f"{self.name}__{self.workdir.name}"
        task_path = inbox / f"{stem}.task.json"
        response_path = inbox / f"{stem}.response.json"
        try:
            response_path.unlink()
        except FileNotFoundError:
            pass

        task = {
            "agent": self.name,
            "run_id": self.ctx.run_id,
            "prompt": prompt,
            "inputs": fields,
            "output_fields": output_fields,
            "response_path": str(response_path),
        }
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

        await self.events.emit(
            "human.waiting",
            {
                "prompt": prompt,
                "output_fields": output_fields,
                "task_path": str(task_path),
                "response_path": str(response_path),
            },
        )

        done = await self._wait_for_response(response_path)
        if done is None:
            await self.events.emit("human.timeout", {"response_path": str(response_path)})
            return self.Outputs.model_validate(
                {**self._defaults(output_fields), "status": "timeout"}
            )

        await self.events.emit("human.submitted", {"response_path": str(response_path)})
        data = {**self._defaults(output_fields), **done}
        data.setdefault("status", "done")
        return self.Outputs.model_validate(data)

    async def _wait_for_response(self, response_path: Path) -> dict[str, Any] | None:
        timeout_s = float(
            self.component_config.get("human_timeout_s") or self.DEFAULT_HUMAN_TIMEOUT_S
        )
        start = time.monotonic()
        last_heartbeat = start
        while True:
            if response_path.exists():
                try:
                    raw = json.loads(response_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    await asyncio.sleep(self.POLL_INTERVAL_S)
                    continue
                return raw if isinstance(raw, dict) else {"value": raw}
            elapsed = time.monotonic() - start
            if timeout_s > 0 and elapsed >= timeout_s:
                return None
            now = time.monotonic()
            if now - last_heartbeat >= self.HEARTBEAT_INTERVAL_S:
                last_heartbeat = now
                await self.events.emit(
                    "human.heartbeat",
                    {"waited_s": elapsed, "response_path": str(response_path)},
                )
            await asyncio.sleep(self.POLL_INTERVAL_S)
            # Human thinking time is not compute time: credit it back so it never
            # eats the run's wallclock budget.
            self.tracker.add_paused(self.POLL_INTERVAL_S)

    def _output_fields(self) -> dict[str, str]:
        schema = self.component_config.get("output_schema")
        fields: dict[str, str] = {}
        if isinstance(schema, dict):
            for key, spec in schema.items():
                if key == "workspace":
                    continue
                fields[str(key)] = spec if isinstance(spec, str) else "string"
        if not fields:
            fields = {"response": "string"}
        return fields

    def _defaults(self, output_fields: dict[str, str]) -> dict[str, Any]:
        return {field: "" for field, typ in output_fields.items() if typ == "string"}


def _format_template(template: str, fields: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in fields:
            return match.group(0)
        value = fields.get(key, "")
        return "" if value is None else str(value)

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, template)


__all__ = ["HumanAgent"]
