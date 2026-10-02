"""Token / cost accounting helpers for external CLI workers."""
from __future__ import annotations

import json
import math
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
    total_cost_usd: float | None = None
    has_result: bool = False
    usage_reported: bool = False
    messages_identified: bool = True
    model_usage: dict[str, dict] = field(default_factory=dict, repr=False)
    turns: list["ClaudeUsage"] = field(default_factory=list, repr=False)

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
        return self.usage_reported or self.metered_tokens > 0

    @property
    def cost_estimated(self) -> bool:
        return self.total_cost_usd is None or any(
            value.get("costBasis") == "unknown" for value in self.model_usage.values()
        )


_CLAUDE_TOKEN_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)
_CLAUDE_MODEL_TOKEN_FIELDS = {
    "input_tokens": "inputTokens", "output_tokens": "outputTokens",
    "cache_creation_input_tokens": "cacheCreationInputTokens",
    "cache_read_input_tokens": "cacheReadInputTokens",
}


def _claude_token_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Claude token counts must be nonnegative integers")
    return value


def _claude_usage(raw: dict, **kwargs) -> ClaudeUsage:
    return ClaudeUsage(
        **{key: _claude_token_count(raw.get(key, 0)) for key in _CLAUDE_TOKEN_FIELDS},
        usage_reported=any(key in raw for key in _CLAUDE_TOKEN_FIELDS),
        **kwargs,
    )


def _finite_cost(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    cost = float(value)
    if not math.isfinite(cost) or cost < 0 or (positive and cost == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return cost


def _usage_from_result_object(obj: dict) -> ClaudeUsage:
    usage = obj.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    models = obj.get("modelUsage") or {}
    if not isinstance(models, dict) or any(not isinstance(value, dict) for value in models.values()):
        raise ValueError("Claude modelUsage must map models to usage records")
    if models and all(all(key in raw for key in _CLAUDE_MODEL_TOKEN_FIELDS.values())
                      for raw in models.values()):
        # Async reviewers can cause several result records, whose `usage` covers
        # only the latest segment. Complete per-model counters cover the session
        # invocation, including native reviewers; do not add the two views.
        usage = {name: sum(_claude_token_count(raw[key]) for raw in models.values())
                 for name, key in _CLAUDE_MODEL_TOKEN_FIELDS.items()}
    return _claude_usage(
        usage,
        num_turns=_claude_token_count(0 if obj.get("num_turns") is None else obj["num_turns"]),
        total_cost_usd=(
            _finite_cost(obj["total_cost_usd"], "total_cost_usd")
            if obj.get("total_cost_usd") is not None else None
        ),
        has_result=True,
        model_usage=models,
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
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = None
    lines = [text] if isinstance(document, dict) else text.splitlines()
    for line in lines:
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
                if not isinstance(mid, str) or not mid:
                    mid = f"_anon_{anon_turns}"
                    anon_turns += 1
                per_turn[str(mid)] = usage  # last streamed snapshot wins
        elif etype is None and result_obj is None and isinstance(ev.get("usage"), dict):
            result_obj = ev  # bare single result object
    if result_obj is not None:
        return _usage_from_result_object(result_obj)
    if per_turn:
        agg = ClaudeUsage(num_turns=len(per_turn), messages_identified=anon_turns == 0)
        for usage in per_turn.values():
            turn = _claude_usage(usage, num_turns=1)
            agg.turns.append(turn)
            agg.usage_reported |= turn.usage_reported
            for key in _CLAUDE_TOKEN_FIELDS:
                setattr(agg, key, getattr(agg, key) + getattr(turn, key))
        return agg
    return ClaudeUsage()


def cost_for_claude_usage(
    usage: ClaudeUsage | str,
    *,
    cost_config: str | None = None,
    expected_model: str | None = None,
    read_cost: float | None = None,
    write_cost: float | None = None,
    cache_read_cost: float | None = None,
    cache_write_cost: float | None = None,
    cache_write_tokens_in_input: bool = False,
    long_context_threshold_tokens: int | None = None,
    long_context_input_multiplier: float = 1.0,
    long_context_output_multiplier: float = 1.0,
) -> float:
    """Return reported USD, correcting explicit unknown-price fallbacks.

    Accepts parsed usage or raw Claude JSON/JSONL. A final result must report
    its cost. Interrupted streams and models explicitly marked with unknown CLI
    pricing use configured rates; unknown model identities fail closed.
    Pass ``cost_config=...`` or unpack rates loaded with
    ``load_cost_rates(..., require_cache_rates=True)``. Unknown cost raises.
    """
    if cost_config is not None:
        from mathagents.config_loader import load_solver_config
        return cost_for_claude_usage(
            usage, expected_model=expected_model or load_solver_config(cost_config).get("model"),
            **load_cost_rates(cost_config, require_cache_rates=True)
        )
    if isinstance(usage, str):
        usage = parse_claude_json(usage)
    for key in _CLAUDE_TOKEN_FIELDS:
        _claude_token_count(getattr(usage, key))
    if not usage.found:
        raise ValueError("paid Claude call produced no parseable usage record")
    if usage.total_cost_usd is not None:
        reported = _finite_cost(usage.total_cost_usd, "total_cost_usd")
        unknown = [(model, raw) for model, raw in usage.model_usage.items() if raw.get("costBasis") == "unknown"]
        fallback, corrected = 0.0, 0.0
        for model, raw in unknown:
            if not expected_model or expected_model not in {model, raw.get("canonicalModel")}:
                raise ValueError(f"no configured pricing for unknown Claude model {model}")
            if long_context_threshold_tokens is not None:
                raise ValueError("unknown CLI pricing with context tiers needs per-request reconciliation")
            if raw.get("webSearchRequests", 0):
                raise ValueError("unknown Claude web-search pricing requires reconciliation")
            # Per-model totals include native reviewers, unlike assistant-only
            # stream snapshots. Replace only the explicitly unknown-priced part.
            mapping = _CLAUDE_MODEL_TOKEN_FIELDS
            if any(key not in raw for key in mapping.values()):
                raise ValueError("unknown Claude pricing requires complete per-model token usage")
            tokens = _claude_usage({name: raw[key] for name, key in mapping.items()})
            fallback += _finite_cost(raw.get("costUSD"), "modelUsage.costUSD")
            corrected += cost_for_claude_usage(tokens, read_cost=read_cost, write_cost=write_cost,
                cache_read_cost=cache_read_cost, cache_write_cost=cache_write_cost,
                cache_write_tokens_in_input=cache_write_tokens_in_input)
        if fallback > reported + 1e-8:
            raise ValueError("Claude per-model costs exceed its terminal total")
        return max(0.0, reported - fallback) + corrected
    if usage.has_result:
        raise ValueError("Claude result is missing total_cost_usd")
    if not usage.messages_identified:
        raise ValueError("Claude stream usage needs message ids to avoid double charging")
    rates = _validate_claude_rates(
        read_cost=read_cost,
        write_cost=write_cost,
        cache_read_cost=cache_read_cost,
        cache_write_cost=cache_write_cost,
        cache_write_tokens_in_input=cache_write_tokens_in_input,
        long_context_threshold_tokens=long_context_threshold_tokens,
        long_context_input_multiplier=long_context_input_multiplier,
        long_context_output_multiplier=long_context_output_multiplier,
    )
    cost = 0.0
    for turn in usage.turns or [usage]:
        if not turn.found:
            raise ValueError("Claude message is missing token usage")
        for key in _CLAUDE_TOKEN_FIELDS:
            _claude_token_count(getattr(turn, key))
        long_context = (
            long_context_threshold_tokens is not None
            and turn.metered_tokens - turn.output_tokens > long_context_threshold_tokens
        )
        cost += (
            (
                turn.input_tokens * rates["read_cost"]
                + turn.cache_read_input_tokens * rates["cache_read_cost"]
                + turn.cache_creation_input_tokens * rates["cache_write_cost"]
            ) * (long_context_input_multiplier if long_context else 1.0)
            + turn.output_tokens * rates["write_cost"]
            * (long_context_output_multiplier if long_context else 1.0)
        ) / 1_000_000.0
    return _finite_cost(cost, "estimated Claude cost")


def _validate_claude_rates(**rates) -> dict:
    for key in (
        "read_cost", "write_cost", "cache_read_cost", "cache_write_cost",
        "long_context_input_multiplier", "long_context_output_multiplier",
    ):
        default = 1 if key.endswith("multiplier") else None
        rates[key] = _finite_cost(rates.get(key, default), key, positive=True)
    if rates.get("cache_write_tokens_in_input", False) is not False:
        raise ValueError("Claude cache writes are separate from input_tokens")
    threshold = rates.get("long_context_threshold_tokens")
    if threshold is not None and (
        isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0
    ):
        raise ValueError("long_context_threshold_tokens must be a positive integer")
    return rates


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


def load_cost_rates(
    config_ref: str, *, require_cache_rates: bool = False
) -> dict[str, float | int | bool | None]:
    """Load model pricing; paid Claude requires explicit, valid cache rates."""
    from mathagents.config_loader import load_solver_config

    cfg = load_solver_config(config_ref)
    if require_cache_rates:
        _validate_claude_rates(**cfg)
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
    "cost_for_claude_usage",
    "cost_for_codex_usage",
    "load_cost_rates",
    "parse_claude_json",
    "parse_codex_jsonl",
]
