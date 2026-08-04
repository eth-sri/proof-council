"""Token / cost accounting helpers for external CLI workers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class CodexUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int | None = None
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    n_turns: int = 0
    turns: list["CodexUsage"] = field(default_factory=list, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def merge(self, other: "CodexUsage") -> "CodexUsage":
        cache_write_input_tokens = None
        if self.cache_write_input_tokens is not None or other.cache_write_input_tokens is not None:
            cache_write_input_tokens = (self.cache_write_input_tokens or 0) + (
                other.cache_write_input_tokens or 0
            )
        return CodexUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_output_tokens=self.reasoning_output_tokens + other.reasoning_output_tokens,
            n_turns=self.n_turns + other.n_turns,
            turns=[*self.turns, *other.turns],
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
        cache_write_tokens = next(
            (
                raw_usage[key]
                for key in (
                    "cache_write_input_tokens",
                    "cache_write_tokens",
                    "cache_creation_input_tokens",
                )
                if raw_usage.get(key) is not None
            ),
            None,
        )
        turn = CodexUsage(
            input_tokens=int(raw_usage.get("input_tokens") or 0),
            cached_input_tokens=int(raw_usage.get("cached_input_tokens") or 0),
            cache_write_input_tokens=(
                int(cache_write_tokens) if cache_write_tokens is not None else None
            ),
            output_tokens=int(raw_usage.get("output_tokens") or 0),
            reasoning_output_tokens=int(raw_usage.get("reasoning_output_tokens") or 0),
            n_turns=1,
        )
        usage = usage.merge(turn)
        usage.turns.append(turn)
    return usage


@dataclass
class ClaudeUsage:
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    num_turns: int = 0
    total_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def metered_tokens(self) -> int:
        # Tokens the call actually processed, against the subscription's rolling
        # window. Cache reads dominate in an agentic loop (the cached system
        # prompt + conversation is re-fed every turn), so they MUST be counted
        # or the backstop is blind to exactly the runaway it exists to catch.
        # Counted full-weight on purpose: a backstop should not undercount.
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
            + self.output_tokens
        )

    @property
    def found(self) -> bool:
        return self.num_turns > 0 or self.input_tokens > 0 or self.output_tokens > 0


def _usage_from_result_object(obj: dict) -> ClaudeUsage:
    usage = obj.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return ClaudeUsage(
        input_tokens=int(usage.get("input_tokens") or 0),
        cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        num_turns=int(obj.get("num_turns") or 0),
        total_cost_usd=float(obj.get("total_cost_usd") or 0.0),
    )


def parse_claude_json(text: str) -> ClaudeUsage:
    """Parse token usage from a ``claude -p`` run.

    Handles both output formats:

    - ``--output-format json`` / the final ``result`` event of stream-json: a
      single object whose ``usage`` is the accurate CUMULATIVE total across all
      turns (verified: its input/cache/output equal the sum of the per-turn
      usages). When present it is authoritative — we use it directly.
    - ``--output-format stream-json`` KILLED mid-run (no ``result`` event): we
      reconstruct a partial total from the per-turn ``assistant`` usages. The
      stream emits several ``assistant`` snapshots per turn (sharing a message
      id) as the message streams, so we keep the LAST usage per message id and
      sum across distinct turns — never double-counting a streamed turn. This is
      the whole point of streaming: a node killed at its timeout — the most
      expensive case — is still metered, instead of recording zero.
    """
    result_obj: dict | None = None
    per_turn: dict[str, dict] = {}
    anon_turns = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] != "{":
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        etype = ev.get("type")
        if etype == "result":
            result_obj = ev
        elif etype == "assistant":
            message = ev.get("message")
            usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict):
                mid = message.get("id")
                if not mid:
                    mid = f"_anon_{anon_turns}"
                    anon_turns += 1
                per_turn[str(mid)] = usage  # last streamed snapshot wins
        elif result_obj is None and isinstance(ev.get("usage"), dict):
            result_obj = ev  # bare single result object
    if result_obj is not None:
        return _usage_from_result_object(result_obj)
    if per_turn:
        agg = ClaudeUsage(num_turns=len(per_turn))
        for usage in per_turn.values():
            agg.input_tokens += int(usage.get("input_tokens") or 0)
            agg.cache_creation_input_tokens += int(usage.get("cache_creation_input_tokens") or 0)
            agg.cache_read_input_tokens += int(usage.get("cache_read_input_tokens") or 0)
            agg.output_tokens += int(usage.get("output_tokens") or 0)
        return agg
    return ClaudeUsage()


def cost_for_codex_usage(
    usage: CodexUsage,
    *,
    read_cost: float,
    write_cost: float,
    cache_read_cost: float | None = None,
    cache_write_cost: float | None = None,
    cache_write_tokens_in_input: bool = False,
    long_context_threshold_tokens: int | None = None,
    long_context_input_multiplier: float = 1.0,
    long_context_output_multiplier: float = 1.0,
) -> float:
    turns = usage.turns or [usage]
    return sum(
        _cost_for_codex_turn(
            turn,
            read_cost=read_cost,
            write_cost=write_cost,
            cache_read_cost=cache_read_cost,
            cache_write_cost=cache_write_cost,
            cache_write_tokens_in_input=cache_write_tokens_in_input,
            long_context_threshold_tokens=long_context_threshold_tokens,
            long_context_input_multiplier=long_context_input_multiplier,
            long_context_output_multiplier=long_context_output_multiplier,
        )
        for turn in turns
    )


def _cost_for_codex_turn(
    usage: CodexUsage,
    *,
    read_cost: float,
    write_cost: float,
    cache_read_cost: float | None,
    cache_write_cost: float | None,
    cache_write_tokens_in_input: bool,
    long_context_threshold_tokens: int | None,
    long_context_input_multiplier: float,
    long_context_output_multiplier: float,
) -> float:
    cache_rate = read_cost if cache_read_cost is None else cache_read_cost
    input_tokens = max(0, usage.input_tokens)
    cached_in = min(max(0, usage.cached_input_tokens), input_tokens)
    fresh_in = input_tokens - cached_in
    cache_write_in = usage.cache_write_input_tokens
    if cache_write_cost is None:
        cache_write_in = 0
    elif cache_write_in is None:
        # Codex JSONL currently omits cache-write counts. For configs whose
        # writes are included in input, treating fresh input as cache-written
        # avoids silently under-accounting the worker's budget.
        cache_write_in = fresh_in if cache_write_tokens_in_input else 0
    cache_write_in = max(0, cache_write_in)
    if cache_write_tokens_in_input:
        cache_write_in = min(cache_write_in, fresh_in)
        fresh_in -= cache_write_in
    out = max(0, usage.output_tokens)
    long_context = (
        long_context_threshold_tokens is not None
        and input_tokens > long_context_threshold_tokens
    )
    input_multiplier = long_context_input_multiplier if long_context else 1.0
    output_multiplier = long_context_output_multiplier if long_context else 1.0
    return (
        (
            fresh_in * read_cost
            + cached_in * cache_rate
            + cache_write_in * (cache_write_cost or 0)
        )
        * input_multiplier
        + out * write_cost * output_multiplier
    ) / 1_000_000.0


def load_cost_rates(config_ref: str) -> dict[str, float | int | bool | None]:
    from mathagents.config_loader import load_solver_config

    cfg = load_solver_config(config_ref)
    read = float(cfg["read_cost"])
    write = float(cfg["write_cost"])
    cached = cfg.get("cache_read_cost")
    cache_read = float(cached) if cached is not None else read
    cache_write_tokens_in_input = cfg.get("cache_write_tokens_in_input", False)
    if not isinstance(cache_write_tokens_in_input, bool):
        raise ValueError("cache_write_tokens_in_input must be a boolean")
    raw_cache_write = cfg.get("cache_write_cost")
    cache_write = float(raw_cache_write) if raw_cache_write is not None else None
    if cache_write_tokens_in_input and (cache_write is None or cache_write <= 0):
        raise ValueError(
            "cache_write_cost must be positive when cache_write_tokens_in_input is true"
        )
    raw_threshold = cfg.get("long_context_threshold_tokens")
    long_context_threshold = int(raw_threshold) if raw_threshold is not None else None
    if long_context_threshold is not None and long_context_threshold <= 0:
        raise ValueError("long_context_threshold_tokens must be positive")
    long_context_input_multiplier = float(cfg.get("long_context_input_multiplier", 1))
    long_context_output_multiplier = float(cfg.get("long_context_output_multiplier", 1))
    if long_context_input_multiplier <= 0 or long_context_output_multiplier <= 0:
        raise ValueError("long-context cost multipliers must be positive")
    return {
        "read_cost": read,
        "write_cost": write,
        "cache_read_cost": cache_read,
        "cache_write_cost": cache_write,
        "cache_write_tokens_in_input": cache_write_tokens_in_input,
        "long_context_threshold_tokens": long_context_threshold,
        "long_context_input_multiplier": long_context_input_multiplier,
        "long_context_output_multiplier": long_context_output_multiplier,
    }


__all__ = [
    "ClaudeUsage",
    "CodexUsage",
    "cost_for_codex_usage",
    "load_cost_rates",
    "parse_claude_json",
    "parse_codex_jsonl",
]
