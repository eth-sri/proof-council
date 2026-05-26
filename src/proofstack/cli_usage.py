"""Token / cost accounting helpers for external CLI workers."""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class CodexUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    n_turns: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def merge(self, other: "CodexUsage") -> "CodexUsage":
        return CodexUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_output_tokens=self.reasoning_output_tokens + other.reasoning_output_tokens,
            n_turns=self.n_turns + other.n_turns,
        )


def parse_codex_jsonl(text: str) -> CodexUsage:
    usage = CodexUsage()
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] != "{":
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict) or ev.get("type") != "turn.completed":
            continue
        raw_usage = ev.get("usage")
        if not isinstance(raw_usage, dict):
            continue
        usage.input_tokens += int(raw_usage.get("input_tokens") or 0)
        usage.cached_input_tokens += int(raw_usage.get("cached_input_tokens") or 0)
        usage.output_tokens += int(raw_usage.get("output_tokens") or 0)
        usage.reasoning_output_tokens += int(raw_usage.get("reasoning_output_tokens") or 0)
        usage.n_turns += 1
    return usage


def cost_for_codex_usage(
    usage: CodexUsage,
    *,
    read_cost: float,
    write_cost: float,
    cache_read_cost: float | None = None,
) -> float:
    cache_rate = read_cost if cache_read_cost is None else cache_read_cost
    cached_in = max(0, usage.cached_input_tokens)
    fresh_in = max(0, usage.input_tokens - cached_in)
    out = max(0, usage.output_tokens)
    return (fresh_in * read_cost + cached_in * cache_rate + out * write_cost) / 1_000_000.0


def load_cost_rates(config_ref: str) -> dict[str, float]:
    from mathagents.config_loader import load_yaml_config

    cfg = load_yaml_config(config_ref)
    read = float(cfg["read_cost"])
    write = float(cfg["write_cost"])
    cached = cfg.get("cache_read_cost")
    cache_read = float(cached) if cached is not None else read
    return {
        "read_cost": read,
        "write_cost": write,
        "cache_read_cost": cache_read,
    }


__all__ = [
    "CodexUsage",
    "cost_for_codex_usage",
    "load_cost_rates",
    "parse_codex_jsonl",
]
