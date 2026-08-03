"""Multi-tier cooperative budget tracking.

A BudgetSpec can be attached at run, problem, agent-family, or phase
scope. Trackers form a tree: spending money at a child scope rolls up
to all parents. Any scope crossing 100% of any limit raises
``BudgetExhausted`` from the next ``check`` call.

The check is cooperative — agents (and APICallAgent in particular) call
``ctx.budgets.check()`` before issuing the next provider request. There
is no preemption.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from pydantic import BaseModel, ConfigDict, Field


_BUDGET_OVERRUN_DEPTH: ContextVar[int] = ContextVar("budget_overrun_depth", default=0)


@contextmanager
def allow_budget_overrun() -> Iterator[None]:
    """Temporarily turn hard budget failures into warnings.

    This is intentionally narrow and is used by workflow-level fallback
    cleanup paths. Spending still accumulates; only ``check()`` stops raising
    while the context is active.
    """

    token = _BUDGET_OVERRUN_DEPTH.set(_BUDGET_OVERRUN_DEPTH.get() + 1)
    try:
        yield
    finally:
        _BUDGET_OVERRUN_DEPTH.reset(token)


class BudgetExhausted(Exception):
    """Raised when a tracker is asked to spend past its limit.

    ``scope`` identifies which tier blew the budget so the workflow's
    last-gasp handler can decide what to do.
    """

    def __init__(self, scope: str, limit_kind: str, limit: float, used: float):
        self.scope = scope
        self.limit_kind = limit_kind
        self.limit = limit
        self.used = used
        super().__init__(
            f"Budget exhausted at scope='{scope}' on '{limit_kind}': "
            f"used={used:.4f} >= limit={limit:.4f}"
        )


class SubscriptionParked(BudgetExhausted):
    """A run was paused in a resumable state (rather than failed).

    Subclasses BudgetExhausted so cooperative budget plumbing (agent.error
    events, node-level accounting) applies, but DAGWorkflow re-raises it
    instead of running the budget fallback: a park must surface as a resumable
    error, not a salvaged terminal answer.
    """

    def __init__(self, window: str, wait_s: float, used: float, allowed: float):
        self.window = window
        self.wait_s = wait_s
        self.scope = "subscription"
        self.limit_kind = f"window:{window}"
        self.limit = float(allowed)
        self.used = float(used)
        # Not BudgetExhausted.__init__: its "used >= limit" phrasing is false
        # for parks triggered by a reset time rather than our own cap.
        Exception.__init__(
            self,
            f"subscription window '{window}' has no headroom "
            f"(used+claims={used:.0f} of allowed={allowed:.0f}); projected wait "
            f"{wait_s / 3600.0:.1f}h exceeds the park threshold — run parked, "
            f"resume it after the window resets",
        )


# Env keys a run may persist into resume.json and have re-injected on resume.
# Both the writer (run_workflow._write_resume_spec) and the reader (the
# dashboard's resume route) filter on this so a hand-edited resume.json cannot
# inject arbitrary environment into the relaunched process. Empty now that the
# pacing override it once carried is gone; the filter mechanism is retained.
RESUME_ENV_ALLOWLIST: tuple[str, ...] = ()


class BudgetSpec(BaseModel):
    """Per-scope limits. ``None`` means "no limit on this dimension"."""

    model_config = ConfigDict(frozen=True)

    max_usd: float | None = None
    max_tokens: int | None = None
    max_wallclock_s: float | None = None
    max_tool_calls: int | None = None


@dataclass
class _Counters:
    usd: float = 0.0
    tokens: int = 0
    tool_calls: int = 0
    paused_s: float = 0.0
    # Reference count of waiters currently pausing this scope, plus when the
    # count last went 0 -> 1. The union of overlapping waits is computed in
    # O(1) state: while any waiter is active the whole span counts once.
    active_pauses: int = 0
    pause_started_at: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    def wallclock_s(self, now: float | None = None) -> float:
        # Time spent waiting on a human (paused_s, plus any pause still in
        # progress) does not count as compute wallclock, so a node that waits
        # hours/days for input never trips the run's wallclock budget.
        now = time.monotonic() if now is None else now
        paused = self.paused_s
        if self.active_pauses > 0:
            paused += max(0.0, now - self.pause_started_at)
        return max(0.0, now - self.started_at - paused)


@dataclass
class BudgetTracker:
    """One node in the budget tree. Spending bubbles up to ``parent``."""

    scope: str
    spec: BudgetSpec | None = None
    parent: "BudgetTracker | None" = None
    counters: _Counters = field(default_factory=_Counters)
    warned: dict[str, bool] = field(default_factory=dict)

    def add_usd(self, amount: float) -> None:
        self.counters.usd += amount
        if self.parent is not None:
            self.parent.add_usd(amount)

    def add_tokens(self, count: int) -> None:
        self.counters.tokens += count
        if self.parent is not None:
            self.parent.add_tokens(count)

    def add_tool_call(self, n: int = 1) -> None:
        self.counters.tool_calls += n
        if self.parent is not None:
            self.parent.add_tool_call(n)

    def begin_pause(self, now: float | None = None) -> None:
        """Mark a human wait as active at this scope and all parents.

        Paired with ``end_pause`` (call in ``finally``). Reference counting
        makes N overlapping waiters credit exactly the *union* of their wait
        spans at shared ancestors — never N times the wall time — in O(1)
        state. Compute that runs concurrently with a human wait is
        deliberately not re-charged: the span counts as paused. ``now`` is
        injectable for tests."""
        now = time.monotonic() if now is None else now
        if self.counters.active_pauses == 0:
            self.counters.pause_started_at = now
        self.counters.active_pauses += 1
        if self.parent is not None:
            self.parent.begin_pause(now)

    def end_pause(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.counters.active_pauses <= 0:
            # Unbalanced end (defensive): every begin_pause increments the
            # whole ancestor chain, so propagating an unmatched end would
            # decrement parent counters that were never incremented for it
            # and desync the union accounting.
            return
        self.counters.active_pauses -= 1
        if self.counters.active_pauses == 0:
            self.counters.paused_s += max(
                0.0, now - self.counters.pause_started_at
            )
        if self.parent is not None:
            self.parent.end_pause(now)

    def chain(self) -> Iterable["BudgetTracker"]:
        node: BudgetTracker | None = self
        while node is not None:
            yield node
            node = node.parent

    def check(self) -> list[tuple[str, str, float, float]]:
        """Inspect each ancestor; return any scopes that crossed the warn
        threshold. Raises ``BudgetExhausted`` if any ancestor is at 100%.

        Returns: list of ``(scope, limit_kind, used, limit)`` tuples that
        just crossed 90% (used so the caller can emit ``budget.warn``).
        """
        warnings: list[tuple[str, str, float, float]] = []
        allow_overrun = _BUDGET_OVERRUN_DEPTH.get() > 0
        for node in self.chain():
            if node.spec is None:
                continue
            for kind, used, limit in (
                ("usd", node.counters.usd, node.spec.max_usd),
                ("tokens", node.counters.tokens, node.spec.max_tokens),
                ("wallclock_s", node.counters.wallclock_s(), node.spec.max_wallclock_s),
                ("tool_calls", node.counters.tool_calls, node.spec.max_tool_calls),
            ):
                if limit is None or limit < 0:
                    continue
                exhausted = used > 0 if limit == 0 else used >= limit
                if exhausted and not allow_overrun:
                    raise BudgetExhausted(node.scope, kind, float(limit), float(used))
                if limit > 0 and used >= 0.9 * limit and not node.warned.get(kind):
                    node.warned[kind] = True
                    warnings.append((node.scope, kind, float(used), float(limit)))
        return warnings

    def remaining_wallclock_s(self) -> float | None:
        """Smallest remaining wallclock allowance across this tracker chain."""
        remaining: list[float] = []
        for node in self.chain():
            if node.spec is None or node.spec.max_wallclock_s is None:
                continue
            remaining.append(float(node.spec.max_wallclock_s) - node.counters.wallclock_s())
        if not remaining:
            return None
        return max(0.0, min(remaining))


class BudgetRegistry:
    """Holds all tracker nodes for a run; agents claim a tracker via
    ``tracker_for(name, parent=...)``. The root tracker is created with
    ``register_root(spec)`` by RunContext."""

    def __init__(self) -> None:
        self._roots: dict[str, BudgetTracker] = {}
        self._by_scope: dict[str, BudgetTracker] = {}
        self._lock = asyncio.Lock()

    def register_root(self, scope: str, spec: BudgetSpec | None) -> BudgetTracker:
        tracker = BudgetTracker(scope=scope, spec=spec, parent=None)
        self._roots[scope] = tracker
        self._by_scope[scope] = tracker
        return tracker

    def child(
        self,
        scope: str,
        *,
        parent: BudgetTracker,
        spec: BudgetSpec | None = None,
    ) -> BudgetTracker:
        tracker = BudgetTracker(scope=scope, spec=spec, parent=parent)
        self._by_scope[scope] = tracker
        return tracker

    def get(self, scope: str) -> BudgetTracker | None:
        return self._by_scope.get(scope)

    def root(self, scope: str = "run") -> BudgetTracker:
        return self._roots[scope]


__all__ = [
    "allow_budget_overrun",
    "BudgetExhausted",
    "BudgetRegistry",
    "BudgetSpec",
    "BudgetTracker",
]
