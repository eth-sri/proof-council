"""YAML-configurable one-shot prompt agent.

Use this when an agent is just:

  Inputs -> formatted messages -> model call -> structured text extraction.

It lets workflow configs define Solver/Improver-style components without
creating a new Python subclass for every prompt variation.
"""
from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from proofstack.context import ModelSpec
from proofstack.kinds.api_call import APICallAgent, _extract_xml_tags


class ConfigurablePromptAgent(APICallAgent):
    """Generic API-call component configured through ``components:`` YAML."""

    description: ClassVar[str] = "YAML-defined prompt/model/output parser."
    MODEL: ClassVar[ModelSpec] = "models/openai/gpt-54"
    SYSTEM_PROMPT: ClassVar[str | None] = None
    USER_PROMPT: ClassVar[str] = "{problem}"

    class Inputs(BaseModel):
        model_config = ConfigDict(extra="allow")

    class Outputs(BaseModel):
        model_config = ConfigDict(extra="allow")

        raw_text: str = Field(default="", exclude=True)

    def render_messages(self, inp: BaseModel):
        fields = inp.model_dump(mode="json")
        messages_cfg = self.component_config.get("messages")
        if isinstance(messages_cfg, list):
            messages = []
            for msg in messages_cfg:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "user"))
                content = str(msg.get("content", "")).format(**fields)
                messages.append({"role": role, "content": content})
            if messages:
                return messages
        return super().render_messages(inp)

    def extra_client_kwargs(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        tools = []
        tool_refs = self.component_config.get("tool_refs")
        if isinstance(tool_refs, list):
            from proofstack.tool_registry import resolve_tool_pairs

            tools.extend(resolve_tool_pairs([str(ref) for ref in tool_refs]))
        tools_cfg = self.component_config.get("tools")
        if isinstance(tools_cfg, list):
            for tool in tools_cfg:
                if isinstance(tool, dict):
                    tools.append((None, dict(tool)))
        if tools:
            out["tools"] = tools
        if "max_tool_calls" in self.component_config:
            out["max_tool_calls"] = self.component_config["max_tool_calls"]
        return out

    def parse_output(self, raw_text: str, inp: BaseModel) -> BaseModel:
        output_cfg = self.component_config.get("output") or {}
        if not isinstance(output_cfg, dict):
            output_cfg = {}
        if not raw_text.strip():
            return self.Outputs.model_validate(_empty_configured_output(output_cfg))

        parsed: dict[str, Any] = {}
        parsed.update(_parse_repeated_xml(raw_text, output_cfg.get("xml_lists") or {}))

        xml_tags = output_cfg.get("xml_tags") or []
        if isinstance(xml_tags, list):
            parsed.update(_extract_xml_tags(raw_text, tuple(str(tag) for tag in xml_tags)))

        json_tag = output_cfg.get("json_tag")
        if isinstance(json_tag, str):
            json_value = _parse_json_tag(
                raw_text,
                json_tag,
                default=output_cfg.get("json_default"),
            )
            if output_cfg.get("json_merge") and isinstance(json_value, dict):
                parsed.update(json_value)
            else:
                parsed[output_cfg.get("json_field", json_tag)] = json_value
        json_tags = output_cfg.get("json_tags") or {}
        if isinstance(json_tags, dict):
            defaults = output_cfg.get("json_defaults") or {}
            if not isinstance(defaults, dict):
                defaults = {}
            for field, tag in json_tags.items():
                if isinstance(tag, str):
                    parsed[str(field)] = _parse_json_tag(
                        raw_text,
                        tag,
                        default=defaults.get(str(field)),
                    )

        regex_fields = output_cfg.get("regex_fields") or {}
        if isinstance(regex_fields, dict):
            for field, pattern in regex_fields.items():
                if not isinstance(pattern, str):
                    continue
                match = re.search(pattern, raw_text, re.DOTALL)
                if match:
                    parsed[str(field)] = match.group(1).strip() if match.groups() else match.group(0).strip()

        default_field = str(output_cfg.get("default_field") or "text")
        if default_field not in parsed:
            parsed[default_field] = raw_text.strip()
        parsed["raw_text"] = raw_text
        return self.Outputs.model_validate(parsed)


def _empty_configured_output(output_cfg: dict[str, Any]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}

    xml_lists = output_cfg.get("xml_lists") or {}
    if isinstance(xml_lists, dict):
        for field in xml_lists:
            parsed[str(field)] = []

    xml_tags = output_cfg.get("xml_tags") or []
    if isinstance(xml_tags, list):
        for tag in xml_tags:
            parsed[str(tag)] = ""

    json_tag = output_cfg.get("json_tag")
    if isinstance(json_tag, str):
        parsed[str(output_cfg.get("json_field") or json_tag)] = output_cfg.get("json_default")
    json_tags = output_cfg.get("json_tags") or {}
    if isinstance(json_tags, dict):
        defaults = output_cfg.get("json_defaults") or {}
        if not isinstance(defaults, dict):
            defaults = {}
        for field in json_tags:
            parsed[str(field)] = defaults.get(str(field))

    regex_fields = output_cfg.get("regex_fields") or {}
    if isinstance(regex_fields, dict):
        for field in regex_fields:
            parsed.setdefault(str(field), "")

    default_field = str(output_cfg.get("default_field") or "text")
    parsed.setdefault(default_field, "")
    parsed["raw_text"] = ""
    return parsed


def _parse_repeated_xml(raw_text: str, config: dict[str, Any]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    if not isinstance(config, dict):
        return parsed
    for field, tag in config.items():
        if not isinstance(tag, str):
            continue
        pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)
        parsed[str(field)] = [
            match.strip()
            for match in pattern.findall(raw_text)
            if match.strip()
        ]
    return parsed


def _parse_json_tag(raw_text: str, tag: str, *, default: Any = None) -> Any:
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", raw_text, re.DOTALL)
    body = match.group(1).strip() if match else raw_text.strip()
    body = re.sub(r"^```(?:json)?", "", body).strip()
    body = re.sub(r"```$", "", body).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return default


__all__ = ["ConfigurablePromptAgent"]
