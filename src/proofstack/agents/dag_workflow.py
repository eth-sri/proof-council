"""Config-defined DAG workflow executor.

YAML defines nodes, dependencies, agent classes, input wiring, and a
mapped branch chain. The executor schedules ready nodes concurrently and
lets mapped branch items advance independently.
"""
from __future__ import annotations

import asyncio
import ast
import importlib
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from proofstack.agent import Agent
from proofstack.budget import BudgetExhausted, allow_budget_overrun


class DAGConfigError(ValueError):
    """Raised when a YAML DAG is malformed before execution starts."""


class DAGWorkflow(Agent):
    """Execute a workflow whose component graph is defined in YAML."""

    description: ClassVar[str] = "Execute a config-defined DAG of proofstack components."
    execution_mode: ClassVar[str] = "workflow"

    class Inputs(BaseModel):
        model_config = ConfigDict(extra="allow")

        problem: str = Field(description="Problem statement (LaTeX or plain text).")
        problem_id: str = Field(description="Stable id for output paths.")
        solution: str | None = Field(default=None, description="Optional existing proof/solution for subworkflows.")
        approach: str | None = Field(default=None, description="Optional proposed strategy for subworkflows.")
        n_approaches: int | None = Field(default=None, ge=1, le=8)
        max_usd: float | None = Field(default=None, gt=0, description="Optional run cost budget in USD.")
        max_wallclock_s: float | None = Field(default=None, gt=0, description="Optional run time budget in seconds.")

    class Outputs(BaseModel):
        model_config = ConfigDict(extra="allow")

    def __init__(self, ctx, **kw):
        super().__init__(ctx, **kw)
        self.dag: dict[str, Any] = self._load_dag()
        self._nodes_by_id: dict[str, dict[str, Any]] = {
            str(node["id"]): node for node in self.dag["nodes"]
        }
        self._agents: dict[str, Agent] = {}

    async def run(self, inp):  # type: ignore[override]
        started_at = time.monotonic()
        state: dict[str, Any] = {
            "input": inp,
            "node": {},
            "param": self.component_config.get("params") if isinstance(self.component_config.get("params"), dict) else {},
            "best_tex": None,
            "run": {
                "started_at": started_at,
                "elapsed_s": lambda: time.monotonic() - started_at,
                "budget_exhausted": False,
                "budget_error": None,
            },
            "path": {
                "node_counts": {},
                "last_node": None,
                "current_node": None,
            },
        }
        try:
            await self._run_nodes(self.dag.get("nodes") or [], state, run_on="normal")
            outputs = self._build_outputs(self.dag.get("outputs") or {}, state)
            return self.Outputs(**outputs)
        except BudgetExhausted as e:
            fallback_outputs = await self._run_budget_fallback(inp, state, e)
            if fallback_outputs is not None:
                return self.Outputs(**fallback_outputs)
            return await self._last_gasp(inp, state, e)
        except Exception as e:
            return await self._last_gasp(inp, state, e)

    async def _last_gasp(self, inp, state: dict[str, Any], error: Exception):
        await self.events.emit(
            "workflow.last_gasp",
            {"type": type(error).__name__, "msg": str(error)},
        )
        tex = state.get("best_tex") or _empty_solution(inp.problem)
        wrapped = _bare_wrap(_extract_body(str(tex)))
        path = self._stash_solution(inp.problem_id, wrapped)
        return self.Outputs(
            problem_id=inp.problem_id,
            solution_tex=path,
            compiled=False,
            pages=0,
            n_branches=len(_resolve_path("node.branches.drafts", state, default=[])),
            n_citations_kept=0,
            last_gasp=True,
            error=str(error),
        )

    async def _run_budget_fallback(
        self,
        inp,
        state: dict[str, Any],
        error: BudgetExhausted,
    ) -> dict[str, Any] | None:
        fallback_nodes = [
            node
            for node in (self.dag.get("nodes") or [])
            if _node_run_on(node) == "budget_exhausted"
        ]
        if not fallback_nodes:
            return None
        state["run"]["budget_exhausted"] = True
        state["run"]["budget_error"] = {
            "scope": error.scope,
            "limit_kind": error.limit_kind,
            "limit": error.limit,
            "used": error.used,
            "message": str(error),
        }
        await self.events.emit(
            "workflow.budget_fallback_started",
            {"nodes": [str(node["id"]) for node in fallback_nodes], "error": str(error)},
        )
        try:
            with allow_budget_overrun():
                if fallback_nodes:
                    await self._run_nodes(
                        self.dag.get("nodes") or [],
                        state,
                        completed_ids=set(state["node"]),
                        run_on="budget_exhausted",
                    )
            return self._build_outputs(self.dag.get("outputs") or {}, state)
        except Exception as fallback_error:
            await self.events.emit(
                "workflow.budget_fallback_failed",
                {"type": type(fallback_error).__name__, "msg": str(fallback_error)},
            )
            return None

    async def _run_nodes(
        self,
        nodes: list[dict[str, Any]],
        state: dict[str, Any],
        *,
        completed_ids: set[str] | None = None,
        run_on: str = "normal",
        terminal_consumers: Any = None,
        node_path_prefix: str = "",
    ) -> bool:
        completed: set[str] = set(completed_ids or ())
        pruned: set[str] = set()
        pending = {
            str(n["id"]): n
            for n in nodes
            if str(n.get("id")) not in completed and _node_run_on(n) == run_on
        }
        active_ids = set(pending)
        running: dict[asyncio.Task, str] = {}
        terminal = False
        terminal_if_ids: set[str] = set()

        try:
            while pending or running:
                pruned_now = _prune_inactive_branch_nodes(
                    pending,
                    completed,
                    pruned,
                    terminal_if_ids,
                    state,
                    nodes,
                )
                for node_id, reason in pruned_now:
                    pruned.add(node_id)
                    payload = {"node": node_id, "reason": reason}
                    if node_path_prefix:
                        payload["node_path"] = f"{node_path_prefix}{node_id}"
                    await self.events.emit(
                        "dag.node_pruned",
                        payload,
                    )
                ready = [
                    node_id
                    for node_id, node in pending.items()
                    if _deps_satisfied(node, completed, active_ids, pruned, run_on=run_on)
                ]
                for node_id in ready:
                    node = pending.pop(node_id)
                    running[
                        asyncio.create_task(
                            self._run_node(node, state, node_path_prefix=node_path_prefix)
                        )
                    ] = node_id

                if not running:
                    if terminal or pruned_now or not pending:
                        return terminal
                    unresolved = ", ".join(sorted(pending))
                    raise RuntimeError(f"DAG has no runnable nodes; unresolved: {unresolved}")

                done, _ = await asyncio.wait(
                    running.keys(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    node_id = running.pop(task)
                    state["node"][node_id] = await task
                    completed.add(node_id)
                    node = next((item for item in nodes if str(item.get("id")) == node_id), None)
                    if node and _if_active_branch_has_no_node_consumers(
                        node,
                        state,
                        pending.values(),
                        terminal_consumers=terminal_consumers,
                    ):
                        terminal = True
                        terminal_if_ids.add(node_id)
                        payload = {"node": node_id, "branch": _active_if_branch(node, state)}
                        if node_path_prefix:
                            payload["node_path"] = f"{node_path_prefix}{node_id}"
                        await self.events.emit(
                            "dag.branch_terminal",
                            payload,
                        )
            return terminal
        except Exception:
            await _cancel_and_drain(running)
            raise

    async def _run_node(
        self,
        node: dict[str, Any],
        state: dict[str, Any],
        *,
        node_path_prefix: str = "",
    ) -> Any:
        node_id = str(node["id"])
        node_path = f"{node_path_prefix}{node_id}"
        event_base = {"node": node_id}
        if node_path_prefix:
            event_base["node_path"] = node_path
        path = state.setdefault("path", {})
        counts = path.setdefault("node_counts", {})
        counts[node_id] = int(counts.get(node_id, 0) or 0) + 1
        path["current_node"] = node_id
        if not _condition(node.get("when"), _node_condition_scope(node, state)):
            await self.events.emit("dag.node_skipped", dict(event_base))
            path["last_node"] = node_id
            return _eval_value(node.get("default", {}), state)

        kind = node.get("kind", "agent")
        await self.events.emit("dag.node_started", {**event_base, "kind": kind})
        try:
            if kind == "agent":
                out = await self._run_agent_node(node, state)
            elif kind == "if_else":
                out = await self._run_if_else_node(node, state)
            elif _is_repeat_kind(kind):
                out = await self._run_loop_node(node, state, node_path=node_path)
            elif kind == "workflow_ref":
                out = await self._run_workflow_ref_node(node, state)
            elif kind == "map_chain":
                out = await self._run_map_chain(node, state)
            elif kind == "join_or_agent":
                out = await self._run_join_or_agent(node, state)
            else:
                raise ValueError(f"unknown DAG node kind {kind!r} for node {node_id}")
        except BudgetExhausted as e:
            await self.events.emit(
                "dag.node_budget_exhausted",
                {
                    **event_base,
                    "kind": kind,
                    "scope": e.scope,
                    "limit_kind": e.limit_kind,
                },
            )
            path["last_node"] = node_id
            raise
        except Exception as e:
            await self.events.emit(
                "dag.node_error",
                {
                    **event_base,
                    "kind": kind,
                    "type": type(e).__name__,
                    "msg": str(e),
                },
            )
            path["last_node"] = node_id
            raise

        best_expr = node.get("best_tex")
        if best_expr is not None:
            best = _eval_value(best_expr, {**state, "output": out})
            if best:
                state["best_tex"] = best
        await self.events.emit("dag.node_done", {**event_base, "kind": kind})
        path["last_node"] = node_id
        return out

    async def _run_agent_node(self, node: dict[str, Any], state: dict[str, Any]) -> Any:
        agent = self._agent_for(node)
        inputs = _eval_value(node.get("inputs", {}), state)
        inputs = _agent_inputs_with_workflow_defaults(agent, inputs, state)
        try:
            return await _call_with_retries(
                lambda: agent(**inputs),
                retries=int(node.get("retries", 0) or 0),
                retry_delay_s=float(node.get("retry_delay_s", 0.0) or 0.0),
            )
        except BudgetExhausted as e:
            if e.scope == "run" or not node.get("soft_fail"):
                raise
            await self.events.emit(
                "dag.node_budget_exhausted",
                {"node": node["id"], "scope": e.scope, "kind": e.limit_kind},
            )
            return _eval_value(node.get("default", {}), state)
        except Exception as e:
            if not node.get("soft_fail"):
                raise
            await self.events.emit(
                "dag.node_error",
                {"node": node["id"], "type": type(e).__name__, "msg": str(e)},
            )
            return _eval_value(node.get("default", {}), state)

    async def _run_map_chain(self, node: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        raw_items = _eval_value(node.get("foreach"), state)
        items = list(raw_items or [])
        if not items:
            items = list(node.get("foreach_default", []))

        max_parallel = int(node.get("max_parallel", len(items)) or len(items) or 1)
        semaphore = asyncio.Semaphore(max(1, max_parallel))
        tasks = [
            asyncio.create_task(
                self._run_chain_item(node, state, item, index, semaphore=semaphore)
            )
            for index, item in enumerate(items)
        ]
        item_results: list[dict[str, Any]] = []
        try:
            for done in asyncio.as_completed(tasks):
                result = await done
                if result:
                    item_results.append(result)
                    if result.get("final"):
                        state["best_tex"] = result["final"]
        except BudgetExhausted:
            await _cancel_and_drain({task: "map_item" for task in tasks})
            raise
        except Exception:
            await _cancel_and_drain({task: "map_item" for task in tasks})
            raise

        item_results.sort(key=lambda r: int(r.get("index", 0)))
        if not item_results and not node.get("allow_empty"):
            raise RuntimeError(f"{node['id']}: all mapped items failed")
        return {
            "items": item_results,
            "drafts": [r["draft"] for r in item_results if r.get("draft")],
            "finals": [r["final"] for r in item_results if r.get("final")],
        }

    async def _run_if_else_node(self, node: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        inputs = _eval_value(node.get("inputs", {}), state)
        if inputs is None:
            inputs = {}
        if not isinstance(inputs, dict):
            raise TypeError(f"{node['id']}: if_else inputs must evaluate to a mapping")
        condition_scope = {
            **state,
            "inputs": inputs,
            **{str(k): v for k, v in inputs.items() if _ID_RE.fullmatch(str(k))},
        }
        condition_spec = node.get("condition")
        passed = _condition(condition_spec, condition_scope)
        branch = str(node.get("then_label", "then") if passed else node.get("else_label", "else"))
        state.setdefault("_if_branch", {})[str(node["id"])] = "then" if passed else "else"
        output_scope = {**condition_scope, "condition": passed, "branch": branch}
        out: dict[str, Any] = {}
        outputs = node.get("outputs")
        if isinstance(outputs, dict):
            out.update(_eval_value(outputs, output_scope))
        branch_outputs = node.get("then" if passed else "else")
        if isinstance(branch_outputs, dict):
            out.update(_eval_value(branch_outputs, output_scope))
        elif "then" not in node and "else" not in node:
            out["True" if passed else "False"] = True
        out["condition"] = passed
        return out

    async def _run_loop_node(
        self,
        node: dict[str, Any],
        state: dict[str, Any],
        *,
        node_path: str = "",
    ) -> dict[str, Any]:
        node_id = str(node["id"])
        max_iterations = int(_eval_value(node.get("max_iterations", 3), state) or 0)
        if max_iterations <= 0:
            raise ValueError(f"{node_id}: loop max_iterations must be positive")
        initial_state = _eval_value(node.get("initial_state", {}), state)
        if initial_state is None:
            initial_state = {}
        if not isinstance(initial_state, dict):
            raise TypeError(f"{node_id}: loop initial_state must evaluate to a mapping")

        body = node.get("body") if isinstance(node.get("body"), dict) else {}
        body_nodes = body.get("nodes") or node.get("nodes") or []
        if not isinstance(body_nodes, list):
            raise TypeError(f"{node_id}: loop body.nodes must be a list")
        state_updates = body.get("state_updates", node.get("state_updates", {}))
        output_spec = node.get("outputs") or {
            "state": "$state",
            "iterations": "$iteration",
            "history": "$history",
        }

        loop_state: dict[str, Any] = dict(initial_state)
        history: list[dict[str, Any]] = []
        last_nodes: dict[str, Any] = {}
        iteration = 0
        reason = "max_iterations"
        for iteration in range(max_iterations):
            scope = {
                **state,
                "parent": state,
                "node": {},
                "state": loop_state,
                "inputs": loop_state,
                "iteration": iteration,
                "max_iterations": max_iterations,
                "history": history,
            }
            if not _condition(node.get("condition", True), scope):
                reason = "condition_false"
                break
            await self.events.emit("dag.loop_iteration_started", {"node": node_id, "iteration": iteration})
            terminal_branch = await self._run_nodes(
                body_nodes,
                scope,
                run_on="normal",
                terminal_consumers=state_updates,
                node_path_prefix=f"{node_path}{REPEAT_BODY_MARKER}",
            )
            updates = _eval_value(state_updates, scope)
            if updates is None:
                updates = {}
            if not isinstance(updates, dict):
                raise TypeError(f"{node_id}: loop state_updates must evaluate to a mapping")
            loop_state = {**loop_state, **updates}
            last_nodes = dict(scope.get("node") or {})
            history.append(
                {
                    "iteration": iteration,
                    "state": loop_state,
                    "node": last_nodes,
                }
            )
            best_expr = node.get("best_tex")
            if best_expr is not None:
                best = _eval_value(
                    best_expr,
                    {
                        **state,
                        "parent": state,
                        "node": last_nodes,
                        "state": loop_state,
                        "iteration": iteration,
                        "max_iterations": max_iterations,
                        "history": history,
                        "loop": {
                            "state": loop_state,
                            "history": history,
                            "iterations": len(history),
                            "reason": reason,
                        },
                    },
                )
                if best:
                    state["best_tex"] = best
            await self.events.emit("dag.loop_iteration_done", {"node": node_id, "iteration": iteration})
            if terminal_branch:
                reason = "terminal_branch"
                break
        else:
            iteration = max_iterations

        output_scope = {
            **state,
            "parent": state,
            "node": last_nodes,
            "state": loop_state,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "history": history,
            "loop": {
                "state": loop_state,
                "history": history,
                "iterations": len(history),
                "reason": reason,
            },
        }
        out = _eval_value(output_spec, output_scope)
        if not isinstance(out, dict):
            raise TypeError(f"{node_id}: loop outputs must evaluate to a mapping")
        out.setdefault("state", loop_state)
        out.setdefault("history", history)
        out.setdefault("iterations", len(history))
        out.setdefault("reason", reason)
        return out

    async def _run_workflow_ref_node(self, node: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        from proofstack.registry import load_preset

        preset_name = str(node.get("preset") or "")
        if not preset_name:
            raise ValueError(f"{node['id']}: workflow_ref node requires 'preset'")
        preset = load_preset(preset_name)
        params = _eval_value(node.get("params", {}), state) or {}
        inputs = _eval_value(node.get("inputs", {}), state) or {}
        if not isinstance(params, dict):
            raise TypeError(f"{node['id']}: workflow_ref params must evaluate to a mapping")
        if not isinstance(inputs, dict):
            raise TypeError(f"{node['id']}: workflow_ref inputs must evaluate to a mapping")
        component_overrides = _eval_value(node.get("component_overrides", {}), state) or {}
        model_overrides = _eval_value(node.get("model_overrides", {}), state) or {}
        if not isinstance(component_overrides, dict):
            raise TypeError(f"{node['id']}: component_overrides must evaluate to a mapping")
        if not isinstance(model_overrides, dict):
            raise TypeError(f"{node['id']}: model_overrides must evaluate to a mapping")

        component_configs = _deep_merge(dict(preset.component_configs), component_overrides)
        workflow_cfg = dict(component_configs.get(preset.workflow_cls.__name__, {}))
        workflow_cfg["params"] = params
        component_configs[preset.workflow_cls.__name__] = workflow_cfg
        child_ctx = replace(
            self.ctx,
            component_configs=component_configs,
            model_overrides={**self.ctx.model_overrides, **preset.model_overrides, **model_overrides},
            config_snapshot={
                **self.ctx.config_snapshot,
                "subworkflow": preset.name,
                "subworkflow_node": str(node["id"]),
                "subworkflow_params": params,
            },
        )
        workflow_inputs = _workflow_input_dict(state.get("input"))
        built_inputs = preset.build_inputs(cli_overrides={**workflow_inputs, **params, **inputs})
        workflow = preset.workflow_cls(child_ctx, name=f"{node['id']}:{preset.name}")
        out = await workflow(**built_inputs)
        if hasattr(out, "model_dump"):
            return out.model_dump(mode="python")
        if isinstance(out, dict):
            return out
        return dict(out)

    async def _run_chain_item(
        self,
        node: dict[str, Any],
        state: dict[str, Any],
        item: Any,
        index: int,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any] | None:
        async with semaphore:
            scope = {**state, "item": item, "index": index, "step": {}}
            for step in node.get("steps", ()):
                step_id = str(step["id"])
                if not _condition(step.get("when"), scope):
                    scope["step"][step_id] = _eval_value(step.get("default", {}), scope)
                    continue
                agent = self._agent_for(step, default_name=f"{node['id']}.{step_id}")
                inputs = _eval_value(step.get("inputs", {}), scope)
                inputs = _agent_inputs_with_workflow_defaults(agent, inputs, scope)
                try:
                    scope["step"][step_id] = await _call_with_retries(
                        lambda: agent(**inputs),
                        retries=int(step.get("retries", 0) or 0),
                        retry_delay_s=float(step.get("retry_delay_s", 0.0) or 0.0),
                    )
                except BudgetExhausted as e:
                    if e.scope == "run":
                        raise
                    if step.get("on_error") == "skip_item":
                        return None
                    scope["step"][step_id] = _eval_value(step.get("default", {}), scope)
                except Exception as e:
                    await self.events.emit(
                        "dag.chain_step_error",
                        {
                            "node": node["id"],
                            "step": step_id,
                            "index": index,
                            "type": type(e).__name__,
                            "msg": str(e),
                        },
                    )
                    if step.get("on_error") == "skip_item":
                        return None
                    scope["step"][step_id] = _eval_value(step.get("default", {}), scope)

            collected = _eval_value(node.get("collect", {}), scope)
        if not isinstance(collected, dict):
            raise TypeError(f"map_chain {node['id']} collect must evaluate to a mapping")
        collected["index"] = index
        return collected

    async def _run_join_or_agent(self, node: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        values = list(_eval_value(node["source"], state) or [])
        field = str(node.get("output_field", "solution"))
        if not values:
            raise RuntimeError(f"{node['id']}: no values to join")
        if len(values) == 1:
            return {field: values[0]}
        agent = self._agent_for(node)
        inputs = _eval_value(node.get("inputs", {}), {**state, "source": values})
        inputs = _agent_inputs_with_workflow_defaults(agent, inputs, state)
        return await _call_with_retries(
            lambda: agent(**inputs),
            retries=int(node.get("retries", 0) or 0),
            retry_delay_s=float(node.get("retry_delay_s", 0.0) or 0.0),
        )

    def _build_outputs(self, mapping: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, spec in mapping.items():
            if isinstance(spec, dict) and "stash_solution" in spec:
                args = _eval_value(spec["stash_solution"], {**state, "output": out})
                out[key] = self._stash_solution(args["problem_id"], args["tex"])
            else:
                out[key] = _eval_value(spec, {**state, "output": out})
        return out

    def _agent_for(self, spec: dict[str, Any], *, default_name: str | None = None) -> Agent:
        class_path = spec.get("agent") or spec.get("class")
        if not class_path:
            raise ValueError(f"node {spec.get('id')} is missing 'agent'")
        key = f"{default_name or spec.get('id')}:{class_path}"
        if key in self._agents:
            return self._agents[key]
        cls = _import_class(str(class_path))
        name = spec.get("name") or default_name
        agent = cls(self.ctx, name=name) if name else cls(self.ctx)
        self._agents[key] = agent
        return agent

    def _load_dag(self) -> dict[str, Any]:
        dag = self.component_config.get("dag")
        if not isinstance(dag, dict):
            raise ValueError("DAGWorkflow requires a component config with a 'dag' mapping")
        if not isinstance(dag.get("nodes"), list):
            raise ValueError("DAGWorkflow dag.nodes must be a list")
        if not isinstance(dag.get("outputs"), dict):
            raise ValueError("DAGWorkflow dag.outputs must be a mapping")
        _validate_dag(dag)
        return dag

    def _stash_solution(self, problem_id: str, tex_body: str) -> Path:
        solutions_dir = self.ctx.root_workdir / "solutions"
        solutions_dir.mkdir(parents=True, exist_ok=True)
        path = solutions_dir / f"{problem_id}.tex"
        path.write_text(tex_body, encoding="utf-8")
        return path


def _agent_inputs_with_workflow_defaults(
    agent: Agent,
    explicit_inputs: Any,
    state: dict[str, Any],
) -> dict[str, Any]:
    if explicit_inputs is None:
        explicit_inputs = {}
    if not isinstance(explicit_inputs, dict):
        raise TypeError(f"{agent.name}: inputs must evaluate to a mapping")
    defaults = _workflow_input_defaults_for_agent(agent, state)
    schema_defaults = _component_input_defaults_for_agent(agent)
    merged = {**schema_defaults, **defaults, **explicit_inputs}
    for key, value in list(merged.items()):
        if value is None and key in schema_defaults:
            merged[key] = schema_defaults[key]
    return merged


def _component_input_defaults_for_agent(agent: Agent) -> dict[str, Any]:
    config = getattr(agent, "component_config", {}) or {}
    raw_schema = config.get("input_schema") if isinstance(config, dict) else None
    if not isinstance(raw_schema, dict):
        return {}
    defaults: dict[str, Any] = {}
    for raw_field, raw_spec in raw_schema.items():
        field = str(raw_field)
        type_name = ""
        if isinstance(raw_spec, dict):
            if "default" in raw_spec:
                defaults[field] = raw_spec["default"]
                continue
            type_name = str(raw_spec.get("type") or "")
        else:
            type_name = str(raw_spec)
        if type_name == "string":
            defaults[field] = ""
    return defaults


def _workflow_input_defaults_for_agent(agent: Agent, state: dict[str, Any]) -> dict[str, Any]:
    workflow_inputs = _workflow_input_dict(state.get("input"))
    if not workflow_inputs:
        return {}
    input_model = getattr(agent, "Inputs", None)
    model_config = getattr(input_model, "model_config", {}) or {}
    if model_config.get("extra") == "allow":
        return workflow_inputs
    fields = getattr(input_model, "model_fields", {}) or {}
    return {key: workflow_inputs[key] for key in fields if key in workflow_inputs}


def _workflow_input_dict(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if hasattr(raw, "model_dump"):
        dumped = raw.model_dump(mode="python")
        return dict(dumped) if isinstance(dumped, dict) else {}
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _node_condition_scope(node: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    inputs = _eval_value(node.get("inputs", {}), state)
    if inputs is None:
        inputs = {}
    if not isinstance(inputs, dict):
        inputs = {}
    inputs = {**_workflow_input_dict(state.get("input")), **inputs}
    return {
        **state,
        "inputs": inputs,
        **{str(k): v for k, v in inputs.items() if _ID_RE.fullmatch(str(k))},
    }


def _eval_value(spec: Any, scope: dict[str, Any]) -> Any:
    if isinstance(spec, str) and spec.startswith("$"):
        return _resolve_path(spec[1:], scope)
    if isinstance(spec, list):
        return [_eval_value(v, scope) for v in spec]
    if isinstance(spec, dict):
        if "coalesce" in spec:
            for candidate in spec["coalesce"]:
                value = _eval_value(candidate, scope)
                if value is not None and value != "":
                    return value
            return None
        if "len" in spec:
            return len(_eval_value(spec["len"], scope) or [])
        if "add" in spec:
            values = spec["add"]
            if not isinstance(values, list):
                values = [values]
            total = 0
            for value in values:
                total += int(_eval_value(value, scope) or 0)
            return total
        if "join" in spec:
            values = _eval_value(spec["join"], scope) or []
            sep = str(_eval_value(spec.get("sep", ""), scope) or "")
            return sep.join(str(v) for v in values)
        if "format" in spec:
            template = str(spec["format"])
            fields = {
                str(k): _eval_value(v, scope)
                for k, v in (spec.get("fields") or {}).items()
            }
            return template.format(**fields)
        if "python" in spec:
            condition_inputs = spec.get("inputs")
            if isinstance(condition_inputs, dict):
                inputs = _eval_value(condition_inputs, scope)
                if not isinstance(inputs, dict):
                    inputs = {}
                scope = {
                    **scope,
                    "inputs": inputs,
                    **{str(k): v for k, v in inputs.items() if _ID_RE.fullmatch(str(k))},
                }
            return _eval_python_condition(str(spec["python"]), scope)
        if "bare_wrap" in spec:
            value = _eval_value(spec["bare_wrap"], scope)
            return _bare_wrap(_extract_body(str(value or "")))
        if "bug_report_from_findings" in spec:
            findings = _eval_value(spec["bug_report_from_findings"], scope) or []
            return _bug_report_from_findings(findings)
        if "not" in spec:
            return not bool(_eval_value(spec["not"], scope))
        return {k: _eval_value(v, scope) for k, v in spec.items()}
    return spec


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _node_run_on(node: dict[str, Any]) -> str:
    raw = str(node.get("run_on") or "normal")
    return "budget_exhausted" if raw in {"budget_exhausted", "budget", "fallback"} else "normal"


def _deps_satisfied(
    node: dict[str, Any],
    completed: set[str],
    active_ids: set[str],
    pruned: set[str],
    *,
    run_on: str,
) -> bool:
    for dep in node.get("needs", ()):
        if dep in completed:
            continue
        if dep in pruned and _dependency_refs_are_optional(node, dep):
            continue
        if run_on == "budget_exhausted" and dep not in active_ids:
            continue
        return False
    return True


def _prune_inactive_branch_nodes(
    pending: dict[str, dict[str, Any]],
    completed: set[str],
    pruned: set[str],
    terminal_if_ids: set[str],
    state: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    specs = {str(node["id"]): node for node in nodes if "id" in node}
    out: list[tuple[str, str]] = []
    changed = True
    while changed:
        changed = False
        for node_id, node in list(pending.items()):
            reason = _branch_prune_reason(node, specs, state, completed, pruned, terminal_if_ids)
            if not reason:
                continue
            pending.pop(node_id, None)
            pruned.add(node_id)
            out.append((node_id, reason))
            changed = True
    return out


def _branch_prune_reason(
    node: dict[str, Any],
    specs: dict[str, dict[str, Any]],
    state: dict[str, Any],
    completed: set[str],
    pruned: set[str],
    terminal_if_ids: set[str],
) -> str:
    for dep in node.get("needs", ()):
        dep = str(dep)
        if dep in pruned:
            if _dependency_refs_are_optional(node, dep):
                continue
            return f"dependency {dep} did not run"
        dep_node = specs.get(dep)
        if dep not in completed or not _is_if_node(dep_node):
            continue
        active = _active_if_branch(dep_node, state)
        if not active:
            continue
        refs = _node_field_refs(node, dep)
        live_fields = _if_branch_fields(dep_node, active) | _if_common_fields(dep_node)
        if dep in terminal_if_ids and not (refs & live_fields or "" in refs):
            return f"{dep}.{active} branch ended"
        inactive = "else" if active == "then" else "then"
        inactive_fields = _if_branch_fields(dep_node, inactive)
        if refs & inactive_fields and not (refs & live_fields or "" in refs):
            return f"{dep}.{inactive} branch inactive"
    return ""


def _if_active_branch_has_no_node_consumers(
    node: dict[str, Any],
    state: dict[str, Any],
    pending_nodes: Any,
    *,
    terminal_consumers: Any = None,
) -> bool:
    if not _is_if_node(node):
        return False
    node_id = str(node.get("id") or "")
    active = _active_if_branch(node, state)
    if not active:
        return False
    live_fields = _if_branch_fields(node, active) | _if_common_fields(node)
    terminal_refs = _node_field_refs(terminal_consumers, node_id)
    if terminal_refs & live_fields or "" in terminal_refs:
        return False
    for pending in pending_nodes:
        refs = _node_field_refs(pending, node_id)
        if refs & live_fields or "" in refs:
            return False
    return True


def _is_if_node(node: dict[str, Any] | None) -> bool:
    return bool(node) and str(node.get("kind")) == "if_else"


def _active_if_branch(node: dict[str, Any], state: dict[str, Any]) -> str:
    node_id = str(node.get("id") or "")
    branch = state.get("_if_branch", {}).get(node_id)
    if branch in {"then", "else"}:
        return branch
    out = state.get("node", {}).get(node_id)
    if not isinstance(out, dict):
        return ""
    then_fields = _if_branch_fields(node, "then")
    else_fields = _if_branch_fields(node, "else")
    out_fields = {str(field) for field in out}
    if then_fields & out_fields and not else_fields & out_fields:
        return "then"
    if else_fields & out_fields and not then_fields & out_fields:
        return "else"
    if "condition" in out:
        return "then" if bool(out.get("condition")) else "else"
    return ""


def _if_branch_fields(node: dict[str, Any], branch: str) -> set[str]:
    raw = node.get(branch)
    if not isinstance(raw, dict):
        if "then" not in node and "else" not in node:
            return {"True"} if branch == "then" else {"False"}
        return set()
    return {str(field) for field in raw if field}


def _if_common_fields(node: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    fields.add("condition")
    for key in ("outputs",):
        raw = node.get(key)
        if isinstance(raw, dict):
            fields.update(str(field) for field in raw if field)
    return fields


def _node_field_refs(value: Any, node_id: str) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str):
        for match in re.finditer(r"\$node\.([A-Za-z_][A-Za-z0-9_-]*)(?:\.([A-Za-z0-9_.-]+))?", value):
            if match.group(1) != node_id:
                continue
            field = (match.group(2) or "").split(".", 1)[0]
            refs.add(field)
    elif isinstance(value, list):
        for item in value:
            refs.update(_node_field_refs(item, node_id))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"id", "ui"}:
                continue
            refs.update(_node_field_refs(item, node_id))
    return refs


def _node_required_field_refs(value: Any, node_id: str) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str):
        return _node_field_refs(value, node_id)
    if isinstance(value, list):
        for item in value:
            refs.update(_node_required_field_refs(item, node_id))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"id", "ui", "coalesce"}:
                continue
            refs.update(_node_required_field_refs(item, node_id))
    return refs


def _dependency_refs_are_optional(value: Any, node_id: str) -> bool:
    return bool(_node_field_refs(value, node_id)) and not _node_required_field_refs(value, node_id)


def _condition(spec: Any, scope: dict[str, Any]) -> bool:
    if spec is None:
        return True
    if isinstance(spec, bool):
        return spec
    if isinstance(spec, str):
        return bool(_eval_value(spec, scope))
    if isinstance(spec, dict):
        condition_inputs = spec.get("inputs")
        if isinstance(condition_inputs, dict):
            inputs = _eval_value(condition_inputs, scope)
            if not isinstance(inputs, dict):
                inputs = {}
            scope = {
                **scope,
                "inputs": inputs,
                **{str(k): v for k, v in inputs.items() if _ID_RE.fullmatch(str(k))},
            }
        if "not" in spec:
            return not _condition(spec["not"], scope)
        if "all" in spec:
            return all(_condition(item, scope) for item in spec["all"])
        if "any" in spec:
            return any(_condition(item, scope) for item in spec["any"])
        if "python" in spec:
            return bool(_eval_python_condition(str(spec["python"]), scope))
        if "python_code" in spec:
            return bool(_exec_python_condition(str(spec["python_code"]), scope))
        ref = spec.get("ref")
        value = _eval_value(ref, scope) if ref is not None else None
        if "equals" in spec:
            return value == spec["equals"]
        if "not_equals" in spec:
            return value != spec["not_equals"]
        if "gt" in spec:
            return value > _eval_value(spec["gt"], scope)
        if "gte" in spec:
            return value >= _eval_value(spec["gte"], scope)
        if "lt" in spec:
            return value < _eval_value(spec["lt"], scope)
        if "lte" in spec:
            return value <= _eval_value(spec["lte"], scope)
        if "min_len" in spec:
            return len(value or []) >= int(spec["min_len"])
        if "max_len" in spec:
            return len(value or []) <= int(spec["max_len"])
        if "contains" in spec:
            return _eval_value(spec["contains"], scope) in (value or [])
        if "any_verdict" in spec:
            verdicts = {str(v) for v in spec["any_verdict"]}
            return any(_finding_verdict(item) in verdicts for item in (value or []))
        return bool(value)
    return bool(spec)


_SAFE_PYTHON_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}


def _eval_python_condition(expr: str, scope: dict[str, Any]) -> Any:
    tree = ast.parse(expr, mode="eval")
    _validate_safe_python_ast(tree)
    return eval(compile(tree, "<dag-condition>", "eval"), {"__builtins__": _SAFE_PYTHON_BUILTINS}, _python_locals(scope))


def _exec_python_condition(code: str, scope: dict[str, Any]) -> Any:
    tree = ast.parse(code, mode="exec")
    _validate_safe_python_ast(tree)
    locals_: dict[str, Any] = _python_locals(scope)
    exec(compile(tree, "<dag-condition-code>", "exec"), {"__builtins__": _SAFE_PYTHON_BUILTINS}, locals_)
    return locals_.get("result", locals_.get("condition", False))


_BANNED_PYTHON_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.ClassDef,
    ast.Delete,
    ast.FunctionDef,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Try,
    ast.While,
    ast.With,
)
_SAFE_METHOD_CALLS = {
    "get",
    "items",
    "keys",
    "values",
    "strip",
    "lower",
    "upper",
    "startswith",
    "endswith",
}


def _validate_safe_python_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, _BANNED_PYTHON_NODES):
            raise ValueError(f"unsafe Python condition construct: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("unsafe Python condition name")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr not in _SAFE_METHOD_CALLS:
                raise ValueError(f"unsafe Python condition attribute: {node.attr}")
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                if fn.id not in _SAFE_PYTHON_BUILTINS:
                    raise ValueError(f"unsafe Python condition call: {fn.id}")
            elif isinstance(fn, ast.Attribute):
                if fn.attr.startswith("__") or fn.attr not in _SAFE_METHOD_CALLS:
                    raise ValueError(f"unsafe Python condition method call: {fn.attr}")
            else:
                raise ValueError("unsafe Python condition call")


def _python_locals(scope: dict[str, Any]) -> dict[str, Any]:
    locals_: dict[str, Any] = {}
    for key, value in scope.items():
        if _ID_RE.fullmatch(str(key)):
            locals_[str(key)] = value
    run = dict(scope.get("run") or {})
    elapsed = run.get("elapsed_s")
    if callable(elapsed):
        elapsed = elapsed()
        run["elapsed_s"] = elapsed
    locals_["run"] = run
    locals_["path"] = scope.get("path") or {}
    locals_["node"] = scope.get("node") or {}
    locals_["input"] = scope.get("input")
    locals_["inputs"] = scope.get("inputs") or {}
    locals_["best_tex"] = scope.get("best_tex")
    locals_["elapsed_s"] = elapsed or 0.0
    return locals_


def _resolve_path(path: str, scope: dict[str, Any], *, default: Any = None) -> Any:
    cur: Any = scope
    parts = path.split(".")
    for index, part in enumerate(parts):
        if part == "":
            continue
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return default
        else:
            cur = getattr(cur, part, default)
            if cur is default:
                return default
        if index == len(parts) - 1 and callable(cur):
            cur = cur()
    return cur


_DOC_BODY_RE = re.compile(r"\\begin\{document\}(.*?)\\end\{document\}", re.DOTALL)


def _extract_body(full_tex: str) -> str:
    match = _DOC_BODY_RE.search(full_tex)
    if match is None:
        return full_tex.strip()
    return match.group(1).strip()


def _empty_solution(problem: str) -> str:
    safe = problem.replace("\\", "\\\\").replace("{", "{{").replace("}", "}}")
    return (
        "% Last-gasp output: no successful proof was produced.\n"
        f"\\section*{{Problem}}\n{safe}\n\n"
        "\\section*{Status}\nNo solution produced before deadline / budget.\n"
    )


def _bare_wrap(body: str) -> str:
    if r"\documentclass" in body and r"\begin{document}" in body:
        return body
    return (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{amsmath,amssymb,amsthm}\n"
        "\\usepackage{hyperref}\n"
        "\\title{Solution}\n"
        "\\date{}\n"
        "\\begin{document}\n"
        "\\maketitle\n\n"
        f"{body}\n"
        "\\end{document}\n"
    )


def _finding_verdict(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("verdict", ""))
    return str(getattr(item, "verdict", ""))


def _bug_report_from_findings(findings: list[Any]) -> str:
    lines: list[str] = []
    for index, item in enumerate(findings, 1):
        if isinstance(item, dict):
            verdict = str(item.get("verdict", "ok"))
            text = str(item.get("text", ""))
            comment = str(item.get("comment", ""))
        else:
            verdict = str(getattr(item, "verdict", "ok"))
            text = str(getattr(item, "text", ""))
            comment = str(getattr(item, "comment", ""))
        if verdict == "ok":
            continue
        lines.append(f"{index}. [{verdict.upper()}] {text}\n   {comment}")
    return "\n".join(lines) if lines else "No issues found."


def _import_class(dotted: str) -> type[Agent]:
    module_path, _, cls_name = dotted.rpartition(".")
    if not module_path or not cls_name:
        raise ValueError(f"class path must be dotted: {dotted!r}")
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    if not issubclass(cls, Agent):
        raise TypeError(f"{dotted} is not an Agent subclass")
    return cls


async def _call_with_retries(factory, *, retries: int, retry_delay_s: float) -> Any:
    attempts = max(1, retries + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await factory()
        except BudgetExhausted:
            raise
        except Exception as e:
            last_error = e
            if attempt + 1 >= attempts:
                break
            if retry_delay_s > 0:
                await asyncio.sleep(retry_delay_s)
    assert last_error is not None
    raise last_error


async def _cancel_and_drain(tasks: dict[asyncio.Task, str]) -> None:
    if not tasks:
        return
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks.keys(), return_exceptions=True)


_REF_RE = re.compile(r"\$node\.([A-Za-z_][A-Za-z0-9_-]*)")
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_REPEAT_KINDS = {"repeat"}
REPEAT_BODY_MARKER = "::body::"
_SUPPORTED_KINDS = {
    "agent",
    "if_else",
    "repeat",
    "workflow_ref",
    "map_chain",
    "join_or_agent",
}


def _is_repeat_kind(kind: Any) -> bool:
    return str(kind) in _REPEAT_KINDS


def _validate_dag(dag: dict[str, Any]) -> None:
    if not isinstance(dag, dict):
        raise DAGConfigError("dag must be a mapping")
    nodes = dag.get("nodes")
    if not isinstance(nodes, list):
        raise DAGConfigError("dag.nodes must be a list")
    outputs = dag.get("outputs")
    if not isinstance(outputs, dict):
        raise DAGConfigError("dag.outputs must be a mapping")
    ids: list[str] = []
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise DAGConfigError(f"dag.nodes[{i}] must be a mapping")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not _ID_RE.fullmatch(node_id):
            raise DAGConfigError(f"dag.nodes[{i}].id must be a valid identifier")
        ids.append(node_id)
        kind = node.get("kind", "agent")
        if kind not in _SUPPORTED_KINDS:
            raise DAGConfigError(f"{node_id}: unsupported kind {kind!r}")
        needs = node.get("needs", [])
        if not isinstance(needs, list) or not all(isinstance(n, str) for n in needs):
            raise DAGConfigError(f"{node_id}: needs must be a list of node ids")
        _validate_required_fields(node_id, kind, node)
        _validate_agent_imports(node_id, kind, node)
        if _is_repeat_kind(kind):
            _validate_loop_body(node_id, node)
        if kind == "workflow_ref":
            _validate_workflow_ref(node_id, node)

    duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
    if duplicates:
        raise DAGConfigError(f"duplicate DAG node id(s): {', '.join(duplicates)}")
    known = set(ids)
    for node in nodes:
        node_id = str(node["id"])
        for dep in node.get("needs", []):
            if dep not in known:
                raise DAGConfigError(f"{node_id}: unknown dependency {dep!r}")
    _validate_acyclic(nodes)
    _validate_node_refs(nodes, {"outputs": outputs}, known)


def _validate_required_fields(node_id: str, kind: str, node: dict[str, Any]) -> None:
    required_by_kind = {
        "agent": (),
        "if_else": ("condition",),
        "repeat": ("condition",),
        "workflow_ref": ("preset",),
        "map_chain": ("foreach", "steps", "collect"),
        "join_or_agent": ("source", "inputs"),
    }
    for field in required_by_kind[kind]:
        if field not in node:
            raise DAGConfigError(f"{node_id}: {kind} node requires {field!r}")
    if kind == "map_chain":
        steps = node.get("steps")
        if not isinstance(steps, list) or not steps:
            raise DAGConfigError(f"{node_id}: map_chain steps must be a non-empty list")
        step_ids: list[str] = []
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                raise DAGConfigError(f"{node_id}.steps[{i}] must be a mapping")
            step_id = step.get("id")
            if not isinstance(step_id, str) or not _ID_RE.fullmatch(step_id):
                raise DAGConfigError(f"{node_id}.steps[{i}].id must be a valid identifier")
            if "agent" not in step and "class" not in step:
                raise DAGConfigError(f"{node_id}.{step_id}: step requires 'agent'")
            step_ids.append(step_id)
        duplicates = sorted({step_id for step_id in step_ids if step_ids.count(step_id) > 1})
        if duplicates:
            raise DAGConfigError(
                f"{node_id}: duplicate map_chain step id(s): {', '.join(duplicates)}"
            )


def _validate_loop_body(node_id: str, node: dict[str, Any]) -> None:
    max_iterations = node.get("max_iterations", 1)
    if isinstance(max_iterations, int):
        if max_iterations <= 0:
            raise DAGConfigError(f"{node_id}: loop max_iterations must be a positive integer")
    elif not isinstance(max_iterations, (str, dict)):
        raise DAGConfigError(f"{node_id}: loop max_iterations must be a positive integer or expression")
    body = node.get("body")
    if not isinstance(body, dict):
        raise DAGConfigError(f"{node_id}: repeat node requires body mapping")
    body_nodes = body.get("nodes")
    if not isinstance(body_nodes, list) or not body_nodes:
        raise DAGConfigError(f"{node_id}: loop body.nodes must be a non-empty list")
    local_ids: list[str] = []
    for i, body_node in enumerate(body_nodes):
        if not isinstance(body_node, dict):
            raise DAGConfigError(f"{node_id}.body.nodes[{i}] must be a mapping")
        body_id = body_node.get("id")
        if not isinstance(body_id, str) or not _ID_RE.fullmatch(body_id):
            raise DAGConfigError(f"{node_id}.body.nodes[{i}].id must be a valid identifier")
        if body_id == node_id:
            raise DAGConfigError(f"{node_id}: repeat body node may not reuse the repeat node id")
        local_ids.append(body_id)
        kind = body_node.get("kind", "agent")
        if kind not in _SUPPORTED_KINDS:
            raise DAGConfigError(f"{node_id}.{body_id}: unsupported kind {kind!r}")
        needs = body_node.get("needs", [])
        if not isinstance(needs, list) or not all(isinstance(n, str) for n in needs):
            raise DAGConfigError(f"{node_id}.{body_id}: needs must be a list of node ids")
        _validate_required_fields(f"{node_id}.{body_id}", kind, body_node)
        _validate_agent_imports(f"{node_id}.{body_id}", kind, body_node)
        if _is_repeat_kind(kind):
            _validate_loop_body(f"{node_id}.{body_id}", body_node)
        if kind == "workflow_ref":
            _validate_workflow_ref(f"{node_id}.{body_id}", body_node)
    duplicates = sorted({local_id for local_id in local_ids if local_ids.count(local_id) > 1})
    if duplicates:
        raise DAGConfigError(f"{node_id}: duplicate loop body node id(s): {', '.join(duplicates)}")
    known = set(local_ids)
    for body_node in body_nodes:
        body_id = str(body_node["id"])
        for dep in body_node.get("needs", []):
            if dep not in known:
                raise DAGConfigError(f"{node_id}.{body_id}: unknown dependency {dep!r}")
    _validate_acyclic(body_nodes)
    _validate_node_refs(
        body_nodes,
        {
            "state_updates": body.get("state_updates", {}),
            "outputs": node.get("outputs", {}),
        },
        known,
    )


def _validate_workflow_ref(node_id: str, node: dict[str, Any]) -> None:
    preset = node.get("preset")
    if not isinstance(preset, str) or not preset:
        raise DAGConfigError(f"{node_id}: workflow_ref node requires a preset name")
    try:
        from proofstack.registry import load_preset

        load_preset(preset)
    except Exception as e:
        raise DAGConfigError(f"{node_id}: cannot load workflow preset {preset!r}: {e}") from e


def _validate_agent_imports(node_id: str, kind: str, node: dict[str, Any]) -> None:
    if kind in {"agent", "join_or_agent"}:
        _validate_agent_ref(node_id, node)
    if kind != "map_chain":
        return
    for step in node.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "?")
        _validate_agent_ref(f"{node_id}.{step_id}", step)


def _validate_agent_ref(label: str, spec: dict[str, Any]) -> None:
    raw = spec.get("agent") or spec.get("class")
    if not isinstance(raw, str) or not raw:
        raise DAGConfigError(f"{label}: missing agent class path")
    try:
        _import_class(raw)
    except Exception as e:
        raise DAGConfigError(f"{label}: cannot import agent {raw!r}: {e}") from e


def _validate_acyclic(nodes: list[dict[str, Any]]) -> None:
    graph = {str(node["id"]): list(node.get("needs", [])) for node in nodes}
    temp: set[str] = set()
    perm: set[str] = set()

    def visit(node_id: str, stack: list[str]) -> None:
        if node_id in perm:
            return
        if node_id in temp:
            cycle = " -> ".join([*stack, node_id])
            raise DAGConfigError(f"DAG dependency cycle: {cycle}")
        temp.add(node_id)
        for dep in graph[node_id]:
            visit(dep, [*stack, node_id])
        temp.remove(node_id)
        perm.add(node_id)

    for node_id in graph:
        visit(node_id, [])


def _validate_node_refs(
    nodes: list[dict[str, Any]],
    outputs: dict[str, Any],
    known: set[str],
) -> None:
    transitive_needs = {
        str(node["id"]): _transitive_needs(str(node["id"]), nodes)
        for node in nodes
    }
    for node in nodes:
        node_id = str(node["id"])
        refs = _find_global_node_refs(node)
        for ref in refs:
            if ref not in known:
                raise DAGConfigError(f"{node_id}: references unknown node {ref!r}")
            if ref == node_id:
                raise DAGConfigError(f"{node_id}: may not reference itself via $node.{ref}")
            if ref not in transitive_needs[node_id]:
                raise DAGConfigError(
                    f"{node_id}: references $node.{ref} but does not depend on it"
                )
    for ref in _find_node_refs(outputs):
        if ref not in known:
            raise DAGConfigError(f"outputs: references unknown node {ref!r}")


def _transitive_needs(node_id: str, nodes: list[dict[str, Any]]) -> set[str]:
    by_id = {str(node["id"]): node for node in nodes}
    out: set[str] = set()

    def collect(current: str) -> None:
        for dep in by_id[current].get("needs", []):
            if dep in out:
                continue
            out.add(dep)
            collect(dep)

    collect(node_id)
    return out


def _find_node_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str):
        refs.update(_REF_RE.findall(value))
    elif isinstance(value, list):
        for item in value:
            refs.update(_find_node_refs(item))
    elif isinstance(value, dict):
        for item in value.values():
            refs.update(_find_node_refs(item))
    return refs


def _find_global_node_refs(node: dict[str, Any]) -> set[str]:
    if not _is_repeat_kind(node.get("kind")):
        return _find_node_refs(node)
    refs: set[str] = set()
    for key in ("inputs", "initial_state", "when", "default", "best_tex"):
        if key in node:
            refs.update(_find_node_refs(node[key]))
    return refs


__all__ = ["DAGWorkflow", "DAGConfigError"]
