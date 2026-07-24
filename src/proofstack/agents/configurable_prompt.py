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
        messages = None
        template_text = ""
        if isinstance(messages_cfg, list):
            rendered = []
            for msg in messages_cfg:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "user"))
                content = str(msg.get("content", "")).format(**fields)
                rendered.append({"role": role, "content": content})
            if rendered:
                messages = rendered
                template_text = "\n".join(
                    str(msg.get("content") or "")
                    for msg in messages_cfg
                    if isinstance(msg, dict)
                )
        if messages is None:
            messages = super().render_messages(inp)
            template_text = "\n".join(
                str(t) for t in (self.SYSTEM_PROMPT, self.USER_PROMPT) if t
            )
        return _with_format_instruction(
            messages, self.component_config.get("output"), template_text
        )

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


def _with_format_instruction(
    messages: list[dict[str, Any]], output_cfg: Any, template_text: str = ""
) -> list[dict[str, Any]]:
    """Append generated delivery-format instructions to the final user message.

    The API-side analog of the CLI ``contract: auto`` tail: component prompts
    stay delivery-neutral (describing only the task) and each executor
    generates its own delivery instructions at runtime. Bindings the authored
    prompt template already mentions are skipped, so hand-written format prompts
    keep working unchanged. Only the template is consulted, never the rendered
    user input — an output tag echoed in a candidate answer must not suppress
    that field's instruction.
    """
    instruction = _format_instruction(output_cfg, template_text)
    if not instruction:
        return messages
    out = [dict(m) for m in messages]
    # Append to the last USER message. Never add a trailing user turn after an
    # assistant prefill (that breaks the intended prefill conversation shape);
    # fold the instruction into the nearest preceding user turn instead.
    for m in reversed(out):
        if m.get("role") == "user":
            m["content"] = f"{str(m.get('content') or '').rstrip()}\n{instruction}"
            return out
    out.append({"role": "user", "content": instruction.lstrip("\n")})
    return out


def _format_instruction(output_cfg: Any, existing_text: str) -> str:
    if not isinstance(output_cfg, dict):
        return ""
    lines: list[str] = []

    def mentioned(tag: str) -> bool:
        # a complete opening/closing tag, not a prefix: `<n>` counts, `x<n`
        # (ordinary math) and an unrelated `<n_items>` do not
        return bool(re.search(rf"</?{re.escape(tag)}\s*>", existing_text))

    for tag in output_cfg.get("xml_tags") or []:
        tag = str(tag).strip()
        if tag and not mentioned(tag):
            lines.append(f"- Wrap that output in <{tag}>...</{tag}>.")
    xml_lists = output_cfg.get("xml_lists") or {}
    if isinstance(xml_lists, dict):
        for field, tag in xml_lists.items():
            tag = str(tag).strip()
            if tag and not mentioned(tag):
                lines.append(
                    f"- Output every item of {str(field).replace('_', ' ')} in its own "
                    f"<{tag}>...</{tag}> tag (repeat the tag per item)."
                )
    json_tag = output_cfg.get("json_tag")
    if isinstance(json_tag, str) and json_tag.strip() and not mentioned(json_tag.strip()):
        lines.append(f"- Output the result as valid JSON inside <{json_tag.strip()}>...</{json_tag.strip()}>.")
    json_tags = output_cfg.get("json_tags") or {}
    if isinstance(json_tags, dict):
        for field, tag in json_tags.items():
            tag = str(tag).strip()
            if tag and not mentioned(tag):
                lines.append(
                    f"- Output {str(field).replace('_', ' ')} as valid JSON inside <{tag}>...</{tag}>."
                )
    # regex_fields are hand-authored bindings whose prompts already explain the
    # expected format; default_field needs no instruction (whole response).
    if not lines:
        return ""
    return "\n\nFORMAT YOUR ANSWER:\n" + "\n".join(lines)


__all__ = ["ConfigurablePromptAgent"]
