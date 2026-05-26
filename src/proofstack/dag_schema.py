"""Typed DAG inspection helpers for the dev dashboard."""
from __future__ import annotations

import importlib
import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from proofstack.agent import Agent
from proofstack.agents.configurable_cli import ConfigurableCLIAgent
from proofstack.agents.configurable_prompt import ConfigurablePromptAgent
from proofstack.agents.dag_workflow import DAGConfigError, _validate_dag
from proofstack.kinds.api_call import APICallAgent

_REPEAT_KINDS = {"repeat"}


def _is_repeat_kind(kind: Any) -> bool:
    return str(kind) in _REPEAT_KINDS


@dataclass
class TypedNode:
    id: str
    kind: str
    agent: str | None
    name: str | None
    component_key: str | None
    label: str
    ui: dict[str, Any]
    needs: list[str]
    input_fields: list[str]
    output_fields: list[str]
    inputs_schema: dict[str, Any]
    outputs_schema: dict[str, Any]
    config: dict[str, Any]
    component_config: dict[str, Any]


@dataclass
class TypedEdge:
    edge_kind: str
    source: str
    source_path: str
    target: str
    target_path: str
    source_schema: dict[str, Any]
    target_schema: dict[str, Any]
    status: str
    message: str


@dataclass
class DAGReport:
    ok: bool
    errors: list[str]
    nodes: list[TypedNode]
    edges: list[TypedEdge]
    warnings: list[str] = field(default_factory=list)
    workflow_inputs: dict[str, Any] = field(default_factory=dict)
    workflow_input_schema: dict[str, Any] = field(default_factory=dict)
    workflow_budget: dict[str, Any] = field(default_factory=dict)
    workflow_outputs: dict[str, Any] = field(default_factory=dict)
    workflow_output_ui: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "warnings": self.warnings,
            "workflow_inputs": self.workflow_inputs,
            "workflow_input_schema": self.workflow_input_schema,
            "workflow_budget": self.workflow_budget,
            "workflow_outputs": self.workflow_outputs,
            "workflow_output_ui": self.workflow_output_ui,
        }


_NODE_REF_RE = re.compile(r"\$node\.([A-Za-z_][A-Za-z0-9_-]*)(?:\.([A-Za-z0-9_.-]+))?")


def build_dag_report(
    dag: dict[str, Any] | None,
    *,
    component_configs: dict[str, dict[str, Any]] | None = None,
    workflow_inputs: dict[str, Any] | None = None,
    workflow_input_schema: dict[str, Any] | None = None,
    workflow_budget: dict[str, Any] | None = None,
    workflow_outputs: dict[str, Any] | None = None,
    workflow_output_ui: dict[str, Any] | None = None,
) -> DAGReport:
    if not isinstance(dag, dict):
        return DAGReport(
            ok=False,
            errors=["preset has no top-level dag mapping"],
            nodes=[],
            edges=[],
            warnings=[],
            workflow_inputs=workflow_inputs or {},
            workflow_input_schema=workflow_input_schema or {},
            workflow_budget=workflow_budget or {},
            workflow_outputs=workflow_outputs or {},
            workflow_output_ui=workflow_output_ui or {},
        )

    errors: list[str] = []
    try:
        _validate_dag(dag)
    except DAGConfigError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"DAG validation failed: {type(e).__name__}: {e}")

    component_configs = component_configs or {}
    raw_nodes = [node for node in (dag.get("nodes") or []) if isinstance(node, dict)]
    workflow_input_names = set(workflow_inputs or {}) | set(workflow_input_schema or {})
    output_schemas = {
        str(node.get("id")): _infer_outputs_schema(node, component_configs)
        for node in raw_nodes
        if node.get("id") is not None
    }
    typed_nodes: list[TypedNode] = []
    for node in raw_nodes:
        node_id = str(node.get("id", ""))
        ui = node.get("ui") if isinstance(node.get("ui"), dict) else {}
        component_name = str(node.get("name")) if node.get("name") is not None else None
        component_key = _component_key(node)
        component_config = _component_config(node, component_configs)
        outputs_schema = output_schemas.get(node_id, {})
        output_fields = sorted(outputs_schema)
        if _is_repeat_kind(node.get("kind")):
            outputs = node.get("outputs")
            output_fields = sorted(str(key) for key in outputs) if isinstance(outputs, dict) else []
        elif node.get("kind") == "if_else":
            output_fields = _if_else_public_output_fields(node)
        typed_nodes.append(
            TypedNode(
                id=node_id,
                kind=str(node.get("kind", "agent")),
                agent=_agent_path(node),
                name=component_name,
                component_key=component_key,
                label=str(ui.get("label") or node_id),
                ui=dict(ui),
                needs=[str(dep) for dep in node.get("needs", []) if isinstance(dep, str)],
                input_fields=_public_input_fields(
                    _infer_input_fields(node, component_config),
                    node,
                    workflow_input_names,
                    component_config,
                ),
                output_fields=output_fields,
                inputs_schema=_infer_inputs_schema(node),
                outputs_schema=outputs_schema,
                config=_editor_config(node, component_configs),
                component_config=dict(component_config) if isinstance(component_config, dict) else {},
            )
        )
    edges: list[TypedEdge] = []
    for node in raw_nodes:
        target = str(node.get("id", ""))
        for ref in _collect_visible_node_refs(node):
            source = ref["source"]
            source_path = ref["source_path"]
            target_path = ref["target_path"]
            source_schema = _schema_at_path(output_schemas.get(source, {}), source_path)
            target_schema = _infer_target_schema(node, target_path, component_configs)
            status, message = _compatibility(_schema_after_transform(source_schema, target_path), target_schema)
            edges.append(
                TypedEdge(
                    edge_kind="data",
                    source=source,
                    source_path=source_path,
                    target=target,
                    target_path=target_path,
                    source_schema=source_schema,
                    target_schema=target_schema,
                    status=status,
                    message=message,
                )
            )
        data_sources = {edge.source for edge in edges if edge.target == target}
        for dep in node.get("needs", []):
            if not isinstance(dep, str) or dep in data_sources:
                continue
            edges.append(
                TypedEdge(
                    edge_kind="dependency",
                    source=dep,
                    source_path="",
                    target=target,
                    target_path="needs",
                    source_schema={},
                    target_schema={},
                    status="ok",
                    message="execution dependency",
                )
            )
    for ref in _collect_node_refs(dag.get("outputs") or {}, prefix="outputs"):
        source_schema = _schema_at_path(output_schemas.get(ref["source"], {}), ref["source_path"])
        edges.append(
            TypedEdge(
                edge_kind="output",
                source=ref["source"],
                source_path=ref["source_path"],
                target="outputs",
                target_path=ref["target_path"],
                source_schema=source_schema,
                target_schema={},
                status="ok" if source_schema else "unknown",
                message="workflow output reference",
            )
        )
    return DAGReport(
        ok=not errors and all(edge.status != "error" for edge in edges),
        errors=errors,
        nodes=typed_nodes,
        edges=edges,
        warnings=_validation_warnings(typed_nodes, edges),
        workflow_inputs=workflow_inputs or {},
        workflow_input_schema=workflow_input_schema or {},
        workflow_budget=workflow_budget or {},
        workflow_outputs=workflow_outputs or {},
        workflow_output_ui=workflow_output_ui or {},
    )


def _collect_node_refs(value: Any, *, prefix: str = "") -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    if isinstance(value, str):
        for match in _NODE_REF_RE.finditer(value):
            refs.append(
                {
                    "source": match.group(1),
                    "source_path": match.group(2) or "",
                    "target_path": prefix,
                }
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            refs.extend(_collect_node_refs(item, prefix=f"{prefix}.{index}" if prefix else str(index)))
    elif isinstance(value, dict):
        for key, item in value.items():
            refs.extend(_collect_node_refs(item, prefix=f"{prefix}.{key}" if prefix else str(key)))
    return refs


def _collect_visible_node_refs(node: dict[str, Any]) -> list[dict[str, str]]:
    if not _is_repeat_kind(node.get("kind")):
        return _collect_node_refs(node)
    refs: list[dict[str, str]] = []
    for key in ("inputs", "initial_state", "when", "default", "best_tex"):
        if key in node:
            refs.extend(_collect_node_refs(node[key], prefix=key))
    return refs


def _validation_warnings(
    nodes: list[TypedNode],
    edges: list[TypedEdge],
) -> list[str]:
    warnings: list[str] = []
    warnings.extend(_unused_node_warnings(nodes, edges))
    warnings.extend(_edge_type_warnings(nodes, edges))
    return warnings


def _unused_node_warnings(
    nodes: list[TypedNode],
    edges: list[TypedEdge],
) -> list[str]:
    node_ids = {node.id for node in nodes}
    output_tied = {
        edge.source
        for edge in edges
        if edge.edge_kind == "output" and edge.source in node_ids
    }
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.edge_kind != "data":
                continue
            if edge.target in output_tied and edge.source in node_ids and edge.source not in output_tied:
                output_tied.add(edge.source)
                changed = True

    unused = [node for node in nodes if node.id not in output_tied]
    if not unused:
        return []
    names = ", ".join(_node_label(node) for node in unused)
    return [
        f"Nodes not connected to any workflow output: {names}. "
        "They likely do not affect returned results and may still cost money."
    ]


def _edge_type_warnings(nodes: list[TypedNode], edges: list[TypedEdge]) -> list[str]:
    labels = {node.id: _node_label(node) for node in nodes}
    warnings: list[str] = []
    for edge in edges:
        if edge.edge_kind != "data" or edge.status != "warning":
            continue
        source_type = _type_label(_schema_type(edge.source_schema))
        target_type = _type_label(_schema_type(edge.target_schema))
        warnings.append(
            "Type mismatch on edge: "
            f"{labels.get(edge.source, edge.source)}.{_path_label(edge.source_path)} "
            f"({source_type}) -> "
            f"{labels.get(edge.target, edge.target)}.{_path_label(edge.target_path)} "
            f"({target_type})."
        )
    return warnings


def _node_label(node: TypedNode) -> str:
    return str(node.ui.get("label") or node.label or node.id).replace("_", " ")


def _path_label(path: str) -> str:
    text = str(path or "")
    for prefix in ("inputs.", "outputs."):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text or "output"


def _type_label(type_name: str) -> str:
    return {
        "array": "list",
        "string": "str",
        "integer": "int",
        "number": "number",
        "boolean": "bool",
    }.get(type_name or "unknown", type_name or "unknown")


def _agent_path(node: dict[str, Any]) -> str | None:
    raw = node.get("agent") or node.get("class")
    return str(raw) if raw else None


def _component_key(spec: dict[str, Any]) -> str | None:
    if spec.get("name") is not None:
        return str(spec.get("name"))
    agent_path = _agent_path(spec)
    if not agent_path:
        return None
    try:
        return _import_class(agent_path).__name__
    except Exception:
        return agent_path.rpartition(".")[2] or agent_path


def _component_config(
    spec: dict[str, Any],
    component_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    agent_path = _agent_path(spec)
    out: dict[str, Any] = {}
    keys: list[str] = []
    if agent_path:
        try:
            cls = _import_class(agent_path)
            out = _class_prompt_config(cls)
            keys.extend([
                f"{cls.__module__}.{cls.__qualname__}",
                cls.__qualname__,
                cls.__name__,
            ])
        except Exception:
            pass
    if spec.get("name") is not None:
        keys.append(str(spec.get("name")))
    for key in keys:
        cfg = component_configs.get(key)
        if isinstance(cfg, dict):
            out = _merge_dicts(out, cfg)
    return out


def _class_prompt_config(cls: type[Agent]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    default_component_config = getattr(cls, "default_component_config", None)
    if callable(default_component_config):
        raw_default = default_component_config()
        if isinstance(raw_default, dict):
            cfg = _merge_dicts(cfg, raw_default)
    component_config_editor = getattr(cls, "component_config_editor", None)
    if callable(component_config_editor):
        raw_editor = component_config_editor()
        if isinstance(raw_editor, dict):
            cfg["__editor__"] = raw_editor
    hidden_inputs = getattr(cls, "HIDDEN_GRAPH_INPUTS", ())
    if hidden_inputs:
        cfg["__hidden_inputs__"] = sorted(str(field) for field in hidden_inputs)
    if issubclass(cls, APICallAgent):
        cfg["model"] = getattr(cls, "MODEL", "")
        cfg["system_prompt"] = getattr(cls, "SYSTEM_PROMPT", "") or ""
        cfg["user_prompt"] = getattr(cls, "USER_PROMPT", "") or ""
    if issubclass(cls, ConfigurableCLIAgent):
        cfg["output_schema"] = {"workspace": "string"}
    return cfg


def _editor_config(
    node: dict[str, Any],
    component_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cfg = copy.deepcopy(node)
    if isinstance(cfg.get("steps"), list):
        for step in cfg["steps"]:
            if not isinstance(step, dict):
                continue
            step["component_key"] = _component_key(step)
            step["component_config"] = _component_config(step, component_configs)
    body = cfg.get("body")
    if isinstance(body, dict) and isinstance(body.get("nodes"), list):
        for body_node in body["nodes"]:
            if not isinstance(body_node, dict):
                continue
            body_node["component_key"] = _component_key(body_node)
            body_node["component_config"] = _component_config(body_node, component_configs)
    return cfg


def _merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _if_else_public_output_fields(node: dict[str, Any]) -> list[str]:
    fields: list[str] = ["condition"]
    outputs = node.get("outputs")
    if isinstance(outputs, dict):
        fields.extend(str(key) for key in outputs)
    for branch_key in ("then", "else"):
        branch_outputs = node.get(branch_key)
        if isinstance(branch_outputs, dict):
            fields.extend(str(key) for key in branch_outputs)
    public = sorted(dict.fromkeys(field for field in fields if field))
    return public if len(public) > 1 else ["False", "True", "condition"]


def _infer_inputs_schema(node: dict[str, Any]) -> dict[str, Any]:
    agent_path = _agent_path(node)
    if not agent_path:
        return {}
    try:
        cls = _import_class(agent_path)
        schema = cls.Inputs.model_json_schema()  # type: ignore[attr-defined]
        return schema.get("properties", {})
    except Exception:
        return {}


def _infer_outputs_schema(
    node: dict[str, Any],
    component_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    kind = str(node.get("kind", "agent"))
    if kind == "map_chain":
        props: dict[str, Any] = {
            "items": {"type": "array", "items": {"type": "object"}},
            "drafts": {"type": "array", "items": {"type": "string"}},
            "finals": {"type": "array", "items": {"type": "string"}},
        }
        collect = node.get("collect") or {}
        if isinstance(collect, dict):
            for key in collect:
                props.setdefault(str(key), {"type": "string"})
        return props
    if kind == "if_else":
        props = {field: {"type": "string"} for field in _if_else_public_output_fields(node)}
        if "condition" in props:
            props["condition"] = {"type": "boolean"}
        return _apply_output_schema_overrides(node, props)
    if _is_repeat_kind(kind):
        outputs = node.get("outputs")
        if isinstance(outputs, dict):
            return {str(key): {"type": "string"} for key in outputs}
        state_props: dict[str, Any] = {}
        initial_state = node.get("initial_state")
        if isinstance(initial_state, dict):
            for key in initial_state:
                state_props.setdefault(str(key), {"type": "string"})
        body = node.get("body")
        state_updates = body.get("state_updates") if isinstance(body, dict) else node.get("state_updates")
        if isinstance(state_updates, dict):
            for key, value in state_updates.items():
                state_props.setdefault(str(key), _schema_for_state_update(value))
        props = {
            "state": {"type": "object", "properties": state_props},
            "history": {"type": "array", "items": {"type": "object"}},
            "iterations": {"type": "integer"},
            "reason": {"type": "string"},
        }
        return props
    if kind == "workflow_ref":
        try:
            from proofstack.registry import load_preset

            preset = load_preset(str(node.get("preset") or ""))
            dag = preset.component_configs.get(preset.workflow_cls.__name__, {}).get("dag")
            if isinstance(dag, dict) and isinstance(dag.get("outputs"), dict):
                return {str(key): {"type": "string"} for key in dag["outputs"]}
            props = preset.workflow_cls.Outputs.model_json_schema().get("properties", {})  # type: ignore[attr-defined]
            return dict(props) if isinstance(props, dict) else {}
        except Exception:
            return {
                "problem_id": {"type": "string"},
                "solution_tex": {"type": "string"},
                "compiled": {"type": "boolean"},
                "pages": {"type": "integer"},
            }
    if kind == "join_or_agent":
        return {str(node.get("output_field", "solution")): {"type": "string"}}
    agent_path = _agent_path(node)
    if not agent_path:
        return {}
    try:
        cls = _import_class(agent_path)
        props = cls.Outputs.model_json_schema().get("properties", {})  # type: ignore[attr-defined]
        if issubclass(cls, ConfigurablePromptAgent):
            props = {**props, **_configured_prompt_outputs(node, component_configs)}
        if issubclass(cls, ConfigurableCLIAgent):
            props = {**props, **_configured_cli_outputs(node, component_configs)}
        props.pop("raw_text", None)
        return props
    except Exception:
        return {}


def _schema_for_state_update(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, list):
        return {"type": "array", "items": {}}
    if isinstance(value, dict):
        return {"type": "object"}
    if isinstance(value, str) and re.fullmatch(r"\$node\.[A-Za-z_][A-Za-z0-9_-]*", value):
        return {"type": "object"}
    return {"type": "string"}


def _configured_prompt_outputs(
    node: dict[str, Any],
    component_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cfg = component_configs.get(str(node.get("name") or "")) or {}
    out_cfg = cfg.get("output") or {}
    props: dict[str, Any] = {}
    for field, schema in _configured_schema_fields(cfg.get("output_schema")).items():
        props[field] = schema
    if isinstance(out_cfg, dict):
        for field in (out_cfg.get("xml_lists") or {}):
            props[str(field)] = {"type": "array", "items": {"type": "string"}}
        for field in (out_cfg.get("xml_tags") or []):
            props[str(field)] = {"type": "string"}
        json_field = out_cfg.get("json_field")
        if isinstance(json_field, str):
            props[json_field] = {}
        json_tags = out_cfg.get("json_tags") or {}
        if isinstance(json_tags, dict):
            for field in json_tags:
                props[str(field)] = {}
        default_field = out_cfg.get("default_field")
        if isinstance(default_field, str):
            props.setdefault(default_field, {"type": "string"})
    return props


def _configured_cli_outputs(
    node: dict[str, Any],
    component_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cfg = component_configs.get(str(node.get("name") or "")) or {}
    props: dict[str, Any] = {"workspace": {"type": "string"}}
    for field, schema in _configured_schema_fields(cfg.get("output_schema")).items():
        props[field] = schema
    raw_files = cfg.get("output_files") or {}
    if isinstance(raw_files, dict):
        for raw_field, raw_spec in raw_files.items():
            field = str(raw_field).split(".", 1)[0]
            if "." in str(raw_field):
                props.setdefault(field, {"type": "object"})
                continue
            kind = "text"
            if isinstance(raw_spec, dict):
                kind = str(raw_spec.get("type") or "text")
            props.setdefault(field, _schema_for_cli_file_output(kind))
    raw_done = cfg.get("done_outputs") or {}
    if isinstance(raw_done, dict):
        for raw_field, raw_spec in raw_done.items():
            field = str(raw_field).split(".", 1)[0]
            if field in props:
                continue
            done_field = str(raw_spec.get("field") if isinstance(raw_spec, dict) else raw_spec)
            props[field] = _schema_for_done_output(done_field)
    return props


def _configured_schema_fields(raw_schema: Any) -> dict[str, Any]:
    if not isinstance(raw_schema, dict):
        return {}
    props: dict[str, Any] = {}
    for raw_field, raw in raw_schema.items():
        field = str(raw_field)
        if isinstance(raw, dict):
            props[field] = dict(raw)
        elif isinstance(raw, str):
            props[field] = {"type": raw}
        else:
            props[field] = {}
    return props


def _schema_for_cli_file_output(kind: str) -> dict[str, Any]:
    kind = kind.lower()
    if kind == "exists":
        return {"type": "boolean"}
    if kind == "int":
        return {"type": "integer"}
    if kind == "float":
        return {"type": "number"}
    if kind == "json":
        return {}
    return {"type": "string"}


def _schema_for_done_output(done_field: str) -> dict[str, Any]:
    if done_field in {"open_questions", "artifacts"}:
        return {"type": "array", "items": {}}
    return {"type": "string"}


def _infer_input_fields(node: dict[str, Any], component_config: dict[str, Any]) -> list[str]:
    fields: set[str] = set(_infer_inputs_schema(node))
    inputs = node.get("inputs")
    if isinstance(inputs, dict):
        fields.update(str(key) for key in inputs)
    if node.get("kind") == "map_chain":
        fields.add("foreach")
    if node.get("kind") == "if_else":
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            fields.update(str(key) for key in inputs)
    if _is_repeat_kind(node.get("kind")):
        initial_state = node.get("initial_state")
        if isinstance(initial_state, dict):
            fields.update(str(key) for key in initial_state)
    if node.get("kind") == "workflow_ref":
        fields.update(_workflow_ref_input_fields(node))
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            fields.update(str(key) for key in inputs)
    if _is_configurable_prompt_spec(node):
        fields.update(_configured_prompt_placeholders(component_config))
    if _is_configurable_cli_spec(node):
        fields.update(_configured_cli_placeholders(component_config))
    return sorted(fields)


def _public_input_fields(
    fields: list[str],
    node: dict[str, Any],
    workflow_input_names: set[str],
    component_config: dict[str, Any],
) -> list[str]:
    inputs = node.get("inputs")
    wired = inputs if isinstance(inputs, dict) else {}
    hidden = {
        str(field)
        for field in component_config.get("__hidden_inputs__", [])
        if isinstance(field, str)
    }
    out = []
    for field in fields:
        if field in hidden:
            continue
        if field == "workspace":
            out.append(field)
            continue
        if field in workflow_input_names and not _collect_node_refs(wired.get(field)):
            continue
        out.append(field)
    return out


def _is_configurable_prompt_spec(spec: dict[str, Any]) -> bool:
    agent_path = _agent_path(spec)
    if not agent_path:
        return False
    try:
        return issubclass(_import_class(agent_path), ConfigurablePromptAgent)
    except Exception:
        return False


def _is_configurable_cli_spec(spec: dict[str, Any]) -> bool:
    agent_path = _agent_path(spec)
    if not agent_path:
        return False
    try:
        return issubclass(_import_class(agent_path), ConfigurableCLIAgent)
    except Exception:
        return False


def _configured_prompt_placeholders(component_config: dict[str, Any]) -> set[str]:
    placeholders: set[str] = set()
    if not isinstance(component_config, dict):
        return placeholders
    for key in ("system_prompt", "user_prompt"):
        raw = component_config.get(key)
        if isinstance(raw, str):
            placeholders.update(_format_fields(raw))
    messages = component_config.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                placeholders.update(_format_fields(msg["content"]))
    input_schema = component_config.get("input_schema")
    if isinstance(input_schema, dict):
        placeholders.update(str(key) for key in input_schema)
    return placeholders


def _configured_cli_placeholders(component_config: dict[str, Any]) -> set[str]:
    placeholders: set[str] = {"workspace"}
    if not isinstance(component_config, dict):
        return placeholders
    input_schema = component_config.get("input_schema")
    declared_inputs = set(str(key) for key in input_schema) if isinstance(input_schema, dict) else set()
    placeholders.update(declared_inputs)
    inferred: set[str] = set()
    for key in ("prompt", "workspace_root"):
        raw = component_config.get(key)
        if isinstance(raw, str):
            inferred.update(_format_fields(raw))
    for key in ("cmd",):
        raw = component_config.get(key)
        if isinstance(raw, str):
            inferred.update(_format_fields(raw))
        elif isinstance(raw, list):
            for part in raw:
                if isinstance(part, str):
                    inferred.update(_format_fields(part))
    for key in ("env", "input_files", "bootstrap_files", "constant_outputs"):
        inferred.update(_placeholders_in_config(component_config.get(key)))
    placeholders.update(inferred & declared_inputs if declared_inputs else inferred)
    return placeholders


def _placeholders_in_config(value: Any) -> set[str]:
    placeholders: set[str] = set()
    if isinstance(value, str):
        placeholders.update(_format_fields(value))
    elif isinstance(value, list):
        for item in value:
            placeholders.update(_placeholders_in_config(item))
    elif isinstance(value, dict):
        for item in value.values():
            placeholders.update(_placeholders_in_config(item))
    return placeholders


def _workflow_ref_input_fields(node: dict[str, Any]) -> set[str]:
    preset_name = str(node.get("preset") or "").strip()
    if not preset_name:
        return set()
    try:
        from proofstack.registry import load_preset

        preset = load_preset(preset_name)
        return {str(field) for field in preset.inputs}
    except Exception:
        return set()


def _format_fields(template: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)", template)
    }


def _schema_at_path(properties: dict[str, Any], path: str) -> dict[str, Any]:
    if not path:
        return {}
    cur: Any = properties
    for part in path.split("."):
        if not part:
            continue
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, dict) and "properties" in cur and part in cur["properties"]:
            cur = cur["properties"][part]
        elif isinstance(cur, dict) and part.isdigit() and "items" in cur:
            cur = cur["items"]
        else:
            return {}
    return cur if isinstance(cur, dict) else {}


def _infer_target_schema(
    node: dict[str, Any],
    target_path: str,
    component_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    parts = target_path.split(".")
    if len(parts) >= 2 and parts[0] == "when" and parts[1] == "inputs":
        return {"type": "any"}
    if "inputs" not in parts:
        return {}
    idx = parts.index("inputs")
    if idx + 1 >= len(parts):
        return {}
    field = parts[idx + 1]
    if "steps" in parts:
        step_id = parts[parts.index("steps") + 1]
        steps = node.get("steps", [])
        if isinstance(steps, list) and step_id.isdigit():
            step_index = int(step_id)
            if 0 <= step_index < len(steps) and isinstance(steps[step_index], dict):
                return _input_field_schema(steps[step_index], field, component_configs)
        for step in steps:
            if isinstance(step, dict) and step.get("id") == step_id:
                return _input_field_schema(step, field, component_configs)
    return _input_field_schema(node, field, component_configs)


def _schema_after_transform(source_schema: dict[str, Any], target_path: str) -> dict[str, Any]:
    parts = target_path.split(".")
    if "inputs" not in parts:
        return source_schema
    idx = parts.index("inputs")
    if idx + 2 >= len(parts):
        return source_schema
    transform = parts[idx + 2]
    if transform in {"join", "bug_report_from_findings", "format"}:
        return {"type": "string"}
    if transform == "len":
        return {"type": "integer"}
    return source_schema


def _input_field_schema(
    spec: dict[str, Any],
    field: str,
    component_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    agent_path = _agent_path(spec)
    if not agent_path:
        return {}
    try:
        cls = _import_class(agent_path)
        props = cls.Inputs.model_json_schema().get("properties", {})  # type: ignore[attr-defined]
        schema = props.get(field, {})
        if schema:
            return schema
        if issubclass(cls, ConfigurablePromptAgent):
            configured = _configured_input_schema(spec, field, component_configs)
            if configured:
                return configured
            return {"type": "string"}
        if issubclass(cls, ConfigurableCLIAgent):
            configured = _configured_input_schema(spec, field, component_configs)
            if configured:
                return configured
            return {"type": "string"}
        return {}
    except Exception:
        return {}


def _configured_input_schema(
    spec: dict[str, Any],
    field: str,
    component_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cfg = component_configs.get(str(spec.get("name") or "")) or {}
    input_schema = cfg.get("input_schema")
    if not isinstance(input_schema, dict):
        return {}
    raw = input_schema.get(field)
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        return {"type": raw}
    return {}


def _apply_output_schema_overrides(node: dict[str, Any], props: dict[str, Any]) -> dict[str, Any]:
    overrides = node.get("output_schema")
    if not isinstance(overrides, dict):
        return props
    for raw_field, raw_schema in overrides.items():
        field = str(raw_field)
        if field not in props:
            continue
        current = props[field] if isinstance(props.get(field), dict) else {}
        if isinstance(raw_schema, str):
            props[field] = {**current, "description": raw_schema}
        elif isinstance(raw_schema, dict):
            allowed = {
                key: value
                for key, value in raw_schema.items()
                if key in {"type", "description", "items"}
            }
            if allowed:
                props[field] = {**current, **allowed}
    return props


def _compatibility(source: dict[str, Any], target: dict[str, Any]) -> tuple[str, str]:
    if not source or not target:
        return ("unknown", "schema unavailable")
    source_type = _schema_type(source)
    target_type = _schema_type(target)
    if not source_type or not target_type:
        return ("unknown", "schema type unavailable")
    if source_type == target_type or target_type == "any" or source_type == "any":
        return ("ok", f"{source_type} -> {target_type}")
    return ("warning", f"possible type mismatch: {source_type} -> {target_type}")


def _schema_type(schema: dict[str, Any]) -> str:
    raw = schema.get("type")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        non_null = [item for item in raw if item != "null"]
        return str(non_null[0]) if non_null else "null"
    if "anyOf" in schema:
        types = [_schema_type(item) for item in schema["anyOf"] if isinstance(item, dict)]
        types = [t for t in types if t and t != "null"]
        return types[0] if types else "any"
    return "any"


def _import_class(dotted: str) -> type[Agent]:
    module_path, _, cls_name = dotted.rpartition(".")
    if not module_path or not cls_name:
        raise ValueError(f"class path must be dotted: {dotted!r}")
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    if not issubclass(cls, Agent):
        raise TypeError(f"{dotted} is not an Agent subclass")
    return cls


__all__ = ["DAGReport", "TypedEdge", "TypedNode", "build_dag_report"]
