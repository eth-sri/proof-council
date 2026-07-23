"""Cross-run subscription window accounting and pacing.

Subscription (CLI) runs are limited by the provider's rolling account-wide
windows (Anthropic: 5-hour + weekly + model-class weeklies), not by USD. A
per-run BudgetTracker structurally cannot see those, so this module keeps a
small persistent store shared by every ProofCouncil process:

  ~/.proofcouncil/            (override with $PROOFCOUNCIL_HOME)
    subscription.json         settings + calibration (observed ceilings/resets)
    usage_ledger.jsonl        one line per finished CLI node: ts/provider/model/tokens
    claims.json               in-flight node claims (batch runs = many processes)
    .lock                     flock guarding all of the above

Ceilings are *estimates*: seeded from the configured plan tier, then replaced
by calibration whenever a CLI actually reports hitting a limit (the observed
usage at that instant is ground truth in our own metered-token units, which
sidesteps not knowing how the provider weights cache reads internally). The
pacer therefore starts conservative and sharpens with use; it can only see
ProofCouncil's own spend, so outside usage is invisible until a real
rate-limit hit recalibrates it.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from proofstack.budget import BudgetExhausted


DEFAULT_RECHECK_S = 60.0
DEFAULT_NODE_ESTIMATE_TOKENS = 500_000
DEFAULT_PARK_AFTER_S = 6 * 3600.0
DEFAULT_CLAIM_TTL_S = 2 * 3600.0
LEDGER_COMPACT_BYTES = 4_000_000
LEDGER_KEEP_S = 8 * 86400.0

FIVE_HOURS_S = 5 * 3600.0
ONE_WEEK_S = 7 * 86400.0

# window id -> (rolling seconds, model substring filter or None for all)
WINDOWS: dict[str, tuple[float, str | None]] = {
    "five_hour": (FIVE_HOURS_S, None),
    "weekly": (ONE_WEEK_S, None),
    "weekly_opus": (ONE_WEEK_S, "opus"),
    "weekly_fable": (ONE_WEEK_S, "fable"),
}

# Seed ceilings in OUR metered-token units (input + cache creation + cache
# read + output). Deliberately rough: cache reads dominate agentic loops and
# the provider's internal weighting is unknown, so these only bootstrap the
# pacer until calibration observes a real limit. A window absent for a plan
# means "not tracked" (no separate cap on that plan).
PLAN_SEEDS: dict[str, dict[str, int]] = {
    "pro": {
        "five_hour": 15_000_000,
        "weekly": 150_000_000,
    },
    "max_5x": {
        "five_hour": 75_000_000,
        "weekly": 750_000_000,
        "weekly_opus": 150_000_000,
        "weekly_fable": 75_000_000,
    },
    "max_20x": {
        "five_hour": 300_000_000,
        "weekly": 3_000_000_000,
        "weekly_opus": 600_000_000,
        "weekly_fable": 300_000_000,
    },
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "enabled": False,
    "provider": "claude",
    "plan": "max_5x",
    # percent of each window's (estimated) ceiling ProofCouncil may consume
    "cap_pct": {"five_hour": 50, "weekly": 30, "weekly_opus": 30, "weekly_fable": 30},
    # hard stop on TOTAL account utilization (probed % + our estimated drift);
    # protects a run from usage that happened outside ProofCouncil
    "account_cap_pct": {"five_hour": 90, "weekly": 90, "weekly_opus": 90, "weekly_fable": 90},
    # manual token-ceiling overrides per window; wins over seed AND calibration
    "ceilings": {},
    "park_after_s": DEFAULT_PARK_AFTER_S,
    "node_estimate_tokens": None,
    # command (string, run via shell) printing provider usage JSON, e.g. the
    # check-usage skill's claude-usage-api.sh; null disables probing
    "usage_probe_cmd": None,
    "probe_ttl_s": 600,
    # per-provider: {"claude": {window: {...}}, "codex": {...}} — namespaced
    # because claude and codex ceilings are on entirely different scales
    "calibration": {},
}


def _provider_calibration(settings: dict[str, Any], provider: str) -> dict[str, Any]:
    calibration = settings.get("calibration")
    if not isinstance(calibration, dict):
        return {}
    scoped = calibration.get(provider)
    if isinstance(scoped, dict):
        return scoped
    # pre-namespacing files stored claude windows at the top level
    if provider == "claude" and any(k in WINDOWS for k in calibration):
        return calibration
    return {}

PROBE_TIMEOUT_S = 60.0
# claude usage-API keys -> our window ids (null-valued keys are skipped)
_PROBE_KEY_MAP = {
    "five_hour": "five_hour",
    "seven_day": "weekly",
    "seven_day_opus": "weekly_opus",
    "seven_day_fable": "weekly_fable",
}


class SubscriptionParked(BudgetExhausted):
    """A pacing wait exceeded park_after_s; end the run in a resumable state.

    Subclasses BudgetExhausted so cooperative budget plumbing (agent.error
    events, node-level accounting) applies, but DAGWorkflow re-raises it
    instead of running the budget fallback: a park must surface as a
    resumable error, not a salvaged terminal answer.
    """

    def __init__(self, window: str, wait_s: float, used: float, allowed: float):
        self.window = window
        self.wait_s = wait_s
        self.scope = "subscription"
        self.limit_kind = f"window:{window}"
        self.limit = float(allowed)
        self.used = float(used)
        # Not BudgetExhausted.__init__: its "used >= limit" phrasing is false
        # for parks triggered by a provider reset time rather than our cap.
        Exception.__init__(
            self,
            f"subscription window '{window}' has no headroom "
            f"(used+claims={used:.0f} of allowed={allowed:.0f}); projected wait "
            f"{wait_s / 3600.0:.1f}h exceeds the park threshold — run parked, "
            f"resume it after the window resets",
        )


def subscription_home() -> Path:
    raw = os.environ.get("PROOFCOUNCIL_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".proofcouncil"


# Env keys a run may persist into resume.json and have re-injected on resume.
# Both the writer (run_workflow._write_resume_spec) and the reader (the
# dashboard's resume route) filter on this so a hand-edited resume.json cannot
# inject arbitrary environment into the relaunched process.
RESUME_ENV_ALLOWLIST: tuple[str, ...] = ("PROOFCOUNCIL_PACING",)


def pacing_env_override() -> str | None:
    raw = (os.environ.get("PROOFCOUNCIL_PACING") or "").strip().lower()
    if raw in {"on", "off"}:
        return raw
    return None


def _park_after_s(settings: dict[str, Any]) -> float:
    raw = settings.get("park_after_s")
    if raw is None:
        return DEFAULT_PARK_AFTER_S
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_PARK_AFTER_S


def _block_until(reset_at: float | None, now: float) -> float:
    """When a limit is detected, produce a block even with no known reset.

    A real limit hit (or a ~100%-utilization probe) always means "no headroom
    right now". If the provider gave a reset time we block until then; otherwise
    we block for one recheck interval so a re-probe or a re-hit refreshes it,
    instead of admitting straight back into the exhausted window (A9). A block
    already in the past is cleared later, at read time in _window_statuses.
    """
    return float(reset_at) if reset_at else now + DEFAULT_RECHECK_S


def _probe_ttl_s(settings: dict[str, Any]) -> float:
    try:
        return max(30.0, float(settings.get("probe_ttl_s") or 600))
    except (TypeError, ValueError):
        return 600.0


@dataclass
class LedgerEntry:
    ts: float
    provider: str
    model: str
    tokens: int
    run_id: str = ""


def _filter_window(
    entries: list[LedgerEntry],
    *,
    provider: str,
    seconds: float,
    model_filter: str | None,
    now: float,
) -> list[LedgerEntry]:
    cutoff = now - seconds
    return sorted(
        (
            e
            for e in entries
            if e.provider == provider
            and cutoff <= e.ts <= now
            and (model_filter is None or model_filter in e.model.lower())
        ),
        key=lambda e: e.ts,
    )


def _entries_in_block(
    entries: list[LedgerEntry],
    *,
    provider: str,
    seconds: float,
    model_filter: str | None,
    block_start: float | None,
    now: float,
) -> list[LedgerEntry]:
    """Ledger entries counting against a window: the rolling window, further
    trimmed to the current discrete provider block when its start is known."""
    win = _filter_window(
        entries, provider=provider, seconds=seconds, model_filter=model_filter, now=now
    )
    if block_start is not None:
        win = [e for e in win if e.ts >= block_start]
    return win


def _relevant_claims(
    claims: dict[str, Any], *, provider: str, model_filter: str | None, now: float
) -> Iterator[dict[str, Any]]:
    for c in claims.values():
        if (
            isinstance(c, dict)
            and float(c.get("expires_at") or 0) > now
            and c.get("provider") == provider
            and (model_filter is None or model_filter in str(c.get("model") or "").lower())
        ):
            yield c


def _claims_sum(
    claims: dict[str, Any],
    *,
    provider: str,
    model_filter: str | None,
    now: float,
) -> int:
    return sum(
        int(c.get("est_tokens") or 0)
        for c in _relevant_claims(claims, provider=provider, model_filter=model_filter, now=now)
    )


def _soonest_claim_expiry(
    claims: dict[str, Any], *, provider: str, model_filter: str | None, now: float
) -> float | None:
    """Earliest expiry among live claims counted against this window, or None.

    Claims expiring release the headroom they reserve, so no own-spend wait
    should ever project past the soonest expiry — otherwise a 30-second claim
    forces an hours-long park (A11).
    """
    expiries = [
        float(c.get("expires_at") or 0)
        for c in _relevant_claims(claims, provider=provider, model_filter=model_filter, now=now)
    ]
    return min(expiries) if expiries else None


def _iso_to_epoch(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(raw)).timestamp()
    except (TypeError, ValueError):
        return None


def parse_claude_probe_json(text: str) -> dict[str, dict[str, float | None]]:
    """Parse the check-usage probe output into {window_id: {used_pct, resets_at}}.

    Tolerates missing/null windows and unknown keys: the endpoint carries many
    plan-dependent fields and only non-null known windows are used.
    """
    try:
        raw = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, float | None]] = {}
    for key, window in _PROBE_KEY_MAP.items():
        entry = raw.get(key)
        if not isinstance(entry, dict):
            continue
        pct = entry.get("utilization")
        if pct is None:
            pct = entry.get("used_percent")
        if pct is None:
            continue
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            continue
        out[window] = {
            "used_pct": max(0.0, min(100.0, pct)),
            "resets_at": _iso_to_epoch(entry.get("resets_at")),
        }
    return out


def _walk_for_key(obj: Any, key: str) -> Iterator[dict[str, Any]]:
    if isinstance(obj, dict):
        found = obj.get(key)
        if isinstance(found, dict):
            yield found
        for value in obj.values():
            yield from _walk_for_key(value, key)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_for_key(value, key)


def parse_codex_rollout_rate_limits(
    text: str, *, now: float | None = None
) -> dict[str, dict[str, float | None]]:
    """Extract the LAST rate_limits snapshot from codex session-rollout JSONL.

    Codex reports {primary, secondary} windows with used_percent and
    window_minutes; minutes decide which of our window ids each maps to.
    The nesting around the snapshot varies across CLI versions, so we walk
    every JSON line for a "rate_limits" dict instead of pinning a schema.
    """
    if not text or "rate_limits" not in text:
        return {}
    now = now if now is not None else time.time()
    last: dict[str, Any] | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or "rate_limits" not in line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        for rl in _walk_for_key(ev, "rate_limits"):
            last = rl
    if last is None:
        return {}
    out: dict[str, dict[str, float | None]] = {}
    for part in ("primary", "secondary"):
        entry = last.get(part)
        if not isinstance(entry, dict):
            continue
        pct = entry.get("used_percent")
        if pct is None:
            continue
        try:
            minutes = float(entry.get("window_minutes") or 0)
        except (TypeError, ValueError):
            minutes = 0.0
        window = "five_hour" if 0 < minutes <= 600 else "weekly"
        resets_at = entry.get("resets_at")
        if resets_at is None and entry.get("resets_in_seconds") is not None:
            try:
                resets_at = now + float(entry["resets_in_seconds"])
            except (TypeError, ValueError):
                resets_at = None
        out[window] = {
            "used_pct": max(0.0, min(100.0, float(pct))),
            "resets_at": float(resets_at) if resets_at else None,
        }
    return out


def run_usage_probe(cmd: str) -> str:
    import subprocess

    proc = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"usage probe exited {proc.returncode}: {proc.stderr.strip()[:200]}")
    return proc.stdout


def _claim_record(
    *,
    provider: str,
    model: str,
    est_tokens: int,
    run_id: str,
    ttl_s: float,
    now: float,
) -> dict[str, Any]:
    return {
        "ts": now,
        "expires_at": now + ttl_s,
        "provider": provider,
        "model": model,
        "est_tokens": int(est_tokens),
        "run_id": run_id,
        "pid": os.getpid(),
    }


class SubscriptionStore:
    """flock-guarded persistence for settings, the usage ledger, and claims.

    All methods are synchronous; async callers go through asyncio.to_thread.
    """

    def __init__(self, home: Path | None = None):
        self.home = home or subscription_home()
        self.settings_path = self.home / "subscription.json"
        self.ledger_path = self.home / "usage_ledger.jsonl"
        self.claims_path = self.home / "claims.json"
        self.probes_path = self.home / "probes.json"
        self.lock_path = self.home / ".lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.home.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    # --- settings -----------------------------------------------------------

    def load_settings(self) -> dict[str, Any]:
        with self._locked():
            return self._read_settings()

    def _read_settings(self) -> dict[str, Any]:
        merged = json.loads(json.dumps(DEFAULT_SETTINGS))
        try:
            raw = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return merged
        if not isinstance(raw, dict):
            return merged
        for key, value in raw.items():
            if key in ("cap_pct", "ceilings", "calibration", "account_cap_pct") and isinstance(value, dict):
                merged[key] = {**merged.get(key, {}), **value}
            else:
                merged[key] = value
        return merged

    def save_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        with self._locked():
            settings = self._read_settings()
            for key, value in updates.items():
                if key not in DEFAULT_SETTINGS:
                    continue
                if key in ("cap_pct", "ceilings", "account_cap_pct") and isinstance(value, dict):
                    # merge per window; an explicit null deletes the override
                    merged = {**settings.get(key, {}), **value}
                    settings[key] = {k: v for k, v in merged.items() if v is not None}
                else:
                    settings[key] = value
            self._write_settings(settings)
            return settings

    def _write_atomic(self, path: Path, text: str) -> None:
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def _write_settings(self, settings: dict[str, Any]) -> None:
        self._write_atomic(self.settings_path, json.dumps(settings, indent=2))

    # --- usage ledger ---------------------------------------------------------

    def append_usage(
        self,
        *,
        provider: str,
        model: str,
        tokens: int,
        run_id: str = "",
        now: float | None = None,
    ) -> None:
        if tokens <= 0:
            return
        entry = {
            "ts": now if now is not None else time.time(),
            "provider": provider,
            "model": model,
            "tokens": int(tokens),
            "run_id": run_id,
        }
        with self._locked():
            with self.ledger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            self._maybe_compact(entry["ts"])

    def _maybe_compact(self, now: float) -> None:
        try:
            if self.ledger_path.stat().st_size <= LEDGER_COMPACT_BYTES:
                return
        except OSError:
            return
        cutoff = now - LEDGER_KEEP_S
        kept = [e for e in self._read_ledger() if e.ts >= cutoff]
        self._write_atomic(
            self.ledger_path,
            "".join(
                json.dumps(
                    {
                        "ts": e.ts,
                        "provider": e.provider,
                        "model": e.model,
                        "tokens": e.tokens,
                        "run_id": e.run_id,
                    }
                )
                + "\n"
                for e in kept
            ),
        )

    def _read_ledger(self) -> list[LedgerEntry]:
        entries: list[LedgerEntry] = []
        try:
            text = self.ledger_path.read_text(encoding="utf-8")
        except OSError:
            return entries
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                entries.append(
                    LedgerEntry(
                        ts=float(raw["ts"]),
                        provider=str(raw.get("provider") or ""),
                        model=str(raw.get("model") or ""),
                        tokens=int(raw.get("tokens") or 0),
                        run_id=str(raw.get("run_id") or ""),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return entries

    def snapshot(
        self,
    ) -> tuple[dict[str, Any], list[LedgerEntry], dict[str, Any], dict[str, Any]]:
        """Read settings + ledger + claims + probes under ONE lock acquisition.

        Admission decisions must be computed from a single consistent snapshot
        (and claims written within the same acquisition, see
        SubscriptionPacer.try_claim) or two concurrent nodes can both admit
        into the same last slot of headroom.
        """
        with self._locked():
            return (
                self._read_settings(),
                self._read_ledger(),
                self._read_claims(),
                self._read_probes(),
            )

    # --- usage probes -----------------------------------------------------------

    def _read_probes(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.probes_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def read_probes(self) -> dict[str, Any]:
        with self._locked():
            return self._read_probes()

    def record_probe_attempt(self, *, provider: str, error: str | None, now: float) -> None:
        with self._locked():
            probes = self._read_probes()
            meta = probes.setdefault(provider, {}).setdefault("_meta", {})
            meta["last_attempt_ts"] = now
            meta["last_error"] = error
            self._write_atomic(self.probes_path, json.dumps(probes, indent=2))

    def record_probe(
        self,
        *,
        provider: str,
        windows: dict[str, dict[str, float | None]],
        now: float,
    ) -> None:
        """Store probed window utilization and calibrate ceilings from deltas.

        Between two probes, our own metered tokens divided by the observed
        utilization change bounds the window ceiling from below (outside usage
        inflates the pct change, shrinking — never inflating — the estimate).
        This converges much faster than waiting to hit a real limit. A ~100%
        probe additionally marks the window blocked until its reset.
        """
        if not windows:
            return
        with self._locked():
            probes = self._read_probes()
            settings = self._read_settings()
            entries = self._read_ledger()
            calibration = settings.setdefault("calibration", {}).setdefault(provider, {})
            provider_probes = probes.setdefault(provider, {})
            settings_changed = False
            for window, data in windows.items():
                spec = WINDOWS.get(window)
                if spec is None:
                    continue
                seconds, model_filter = spec
                pct = float(data.get("used_pct") or 0.0)
                resets_at = data.get("resets_at")
                own_tokens = sum(
                    e.tokens
                    for e in _filter_window(
                        entries,
                        provider=provider,
                        seconds=seconds,
                        model_filter=model_filter,
                        now=now,
                    )
                )
                prior = provider_probes.get(window)
                if isinstance(prior, dict):
                    d_pct = pct - float(prior.get("used_pct") or 0.0)
                    d_own = own_tokens - int(prior.get("own_tokens") or 0)
                    # skip resets (pct dropped) and noise (< 1 point of change)
                    if d_pct >= 1.0 and d_own > 0:
                        ceiling_obs = int(100.0 * d_own / d_pct)
                        cal = calibration.get(window)
                        cal = cal if isinstance(cal, dict) else {}
                        if ceiling_obs > int(cal.get("observed_ceiling") or 0):
                            cal.update(
                                {"observed_ceiling": ceiling_obs, "ts": now, "source": "probe"}
                            )
                            calibration[window] = cal
                            settings_changed = True
                if pct >= 99.5:
                    cal = calibration.get(window)
                    cal = cal if isinstance(cal, dict) else {}
                    cal.update({"blocked_until": _block_until(resets_at, now), "ts": now})
                    calibration[window] = cal
                    settings_changed = True
                provider_probes[window] = {
                    "used_pct": pct,
                    "resets_at": resets_at,
                    "ts": now,
                    "own_tokens": own_tokens,
                }
            meta = provider_probes.setdefault("_meta", {})
            meta["last_attempt_ts"] = now
            meta["last_error"] = None
            self._write_atomic(self.probes_path, json.dumps(probes, indent=2))
            if settings_changed:
                self._write_settings(settings)

    def window_entries(
        self,
        *,
        provider: str,
        seconds: float,
        model_filter: str | None = None,
        now: float | None = None,
    ) -> list[LedgerEntry]:
        now = now if now is not None else time.time()
        with self._locked():
            entries = self._read_ledger()
        return _filter_window(
            entries, provider=provider, seconds=seconds, model_filter=model_filter, now=now
        )

    def recent_node_tokens(self, *, provider: str, n: int = 20) -> list[int]:
        with self._locked():
            entries = [e for e in self._read_ledger() if e.provider == provider]
        return [e.tokens for e in entries[-n:]]

    # --- in-flight claims -------------------------------------------------------

    def add_claim(
        self,
        *,
        provider: str,
        model: str,
        est_tokens: int,
        run_id: str = "",
        ttl_s: float = DEFAULT_CLAIM_TTL_S,
        now: float | None = None,
    ) -> str:
        now = now if now is not None else time.time()
        claim_id = uuid.uuid4().hex[:12]
        with self._locked():
            claims = self._read_claims()
            claims[claim_id] = _claim_record(
                provider=provider,
                model=model,
                est_tokens=est_tokens,
                run_id=run_id,
                ttl_s=ttl_s,
                now=now,
            )
            self._write_claims(self._prune_claims(claims, now))
        return claim_id

    def remove_claim(self, claim_id: str) -> None:
        with self._locked():
            claims = self._read_claims()
            claims.pop(claim_id, None)
            self._write_claims(claims)

    def claims_total(
        self,
        *,
        provider: str,
        model_filter: str | None = None,
        now: float | None = None,
    ) -> int:
        now = now if now is not None else time.time()
        with self._locked():
            claims = self._read_claims()
        return _claims_sum(claims, provider=provider, model_filter=model_filter, now=now)

    def _prune_claims(self, claims: dict[str, Any], now: float) -> dict[str, Any]:
        return {
            cid: c
            for cid, c in claims.items()
            if isinstance(c, dict) and float(c.get("expires_at") or 0) > now
        }

    def _read_claims(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.claims_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_claims(self, claims: dict[str, Any]) -> None:
        self._write_atomic(self.claims_path, json.dumps(claims, indent=2))

    # --- calibration -------------------------------------------------------------

    def record_rate_limit(
        self,
        *,
        provider: str,
        window: str,
        observed_ceiling: int,
        reset_at: float | None,
        now: float | None = None,
    ) -> None:
        now = now if now is not None else time.time()
        with self._locked():
            settings = self._read_settings()
            scoped = settings.setdefault("calibration", {}).setdefault(provider, {})
            prior = scoped.get(window) if isinstance(scoped.get(window), dict) else {}
            # observed usage at the moment of a real limit hit is a lower bound
            # on the true ceiling; keep the largest ever seen
            best = max(int(prior.get("observed_ceiling") or 0), int(observed_ceiling))
            scoped[window] = {
                "observed_ceiling": best,
                # a detected limit always blocks, even with no reset epoch (A9)
                "blocked_until": _block_until(reset_at, now),
                # keep the raw reset epoch: it bounds the discrete block so that,
                # once the block lapses, pre-reset usage is excluded rather than
                # re-blocking the fresh window (A2 — the CLI sibling of B3's
                # probe-sourced boundary)
                "reset_at": reset_at,
                "ts": now,
            }
            self._write_settings(settings)


# --- rate-limit detection ------------------------------------------------------


@dataclass
class RateLimitHit:
    reset_at: float | None
    window_guess: str
    excerpt: str


_LIMIT_PATTERNS = (
    # claude CLI subscription limit: "Claude AI usage limit reached|<epoch>"
    re.compile(r"usage limit reached\|(\d{10,13})", re.IGNORECASE),
    re.compile(r"usage limit reached", re.IGNORECASE),
    re.compile(r"(?:5-hour|weekly) limit reached", re.IGNORECASE),
    re.compile(r"rate.?limit(?:ed|s)?\b.{0,80}?resets?\D{0,20}(\d{10,13})", re.IGNORECASE | re.DOTALL),
)


def detect_rate_limit(text: str, *, now: float | None = None) -> RateLimitHit | None:
    """Best-effort scan of CLI output for a subscription limit signal.

    Tolerant by design: CLI error formats change between versions, so we look
    for several known shapes and degrade to "hit detected, no reset known".
    """
    # cheap pre-check: transcripts are large and every pattern contains "limit"
    if not text or "limit" not in text.lower():
        return None
    now = now if now is not None else time.time()
    for pattern in _LIMIT_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        reset_at: float | None = None
        if m.groups() and m.group(1):
            reset_at = float(m.group(1))
            if reset_at > 1e12:  # milliseconds epoch
                reset_at /= 1000.0
            if reset_at <= now or reset_at > now + 32 * 86400:
                reset_at = None  # implausible; keep the hit, drop the reset
        lowered = m.group(0).lower()
        if "weekly" in lowered:
            window_guess = "weekly"
        elif reset_at is not None:
            window_guess = "five_hour" if reset_at - now <= FIVE_HOURS_S * 1.1 else "weekly"
        else:
            window_guess = "five_hour"
        start = max(0, m.start() - 40)
        return RateLimitHit(
            reset_at=reset_at,
            window_guess=window_guess,
            excerpt=text[start : m.end() + 40].strip()[:200],
        )
    return None


# --- pacer -----------------------------------------------------------------------


@dataclass
class WindowStatus:
    window: str
    seconds: float
    model_filter: str | None
    ceiling: int  # 0 when unknown (probe-only window, e.g. codex without seeds)
    ceiling_source: str  # manual | calibrated | seed | probe-only
    allowed: int | None  # None = no own-spend cap (probe-only); 0 = hard block (0% cap)
    usage: int
    claims: int
    blocked_until: float | None
    probe_pct: float | None = None
    probe_ts: float | None = None
    resets_at: float | None = None
    account_cap_pct: float | None = None
    own_since_probe: int = 0
    # discrete-block boundary: ledger entries before it belong to a previous,
    # already-reset provider block and must not count (None = pure rolling window)
    block_start: float | None = None


@dataclass
class PacingDecision:
    admit: bool
    est_tokens: int
    blocking_window: str | None = None
    wait_s: float = 0.0
    windows: list[WindowStatus] = field(default_factory=list)


class SubscriptionPacer:
    """Admission control for subscription CLI nodes across all runs/processes.

    decide() never blocks; callers loop decide -> sleep -> decide so claims
    added by other processes are re-read every pass.
    """

    def __init__(self, store: SubscriptionStore | None = None, *, provider: str = "claude"):
        self.store = store or SubscriptionStore()
        self.provider = provider

    def gate_config(self) -> tuple[bool, float]:
        """(enabled, park_after_s) from one settings read."""
        override = pacing_env_override()
        if override == "off":
            return False, DEFAULT_PARK_AFTER_S
        settings = self.store.load_settings()
        enabled = True if override == "on" else bool(settings.get("enabled"))
        return enabled, _park_after_s(settings)

    def enabled(self) -> bool:
        return self.gate_config()[0]

    def park_after_s(self) -> float:
        return self.gate_config()[1]

    def estimate_tokens(self, settings: dict[str, Any] | None = None) -> int:
        if settings is None:
            settings, entries, _, _ = self.store.snapshot()
            recent = self._recent_tokens(entries)
        else:
            recent = self.store.recent_node_tokens(provider=self.provider)
        return self._estimate_from(settings, recent)

    def ensure_fresh_probe(self, now: float | None = None, *, force: bool = False) -> None:
        """Run the configured usage probe when the cached one has gone stale.

        Called OUTSIDE the store lock (the probe subprocess takes seconds).
        Failures are recorded and throttled by the same TTL so a broken probe
        command doesn't re-run on every admission pass; the pacer degrades to
        ledger-only accounting.
        """
        if self.provider != "claude":
            return  # codex probes arrive via session-rollout harvest, not a command
        now = now if now is not None else time.time()
        settings = self.store.load_settings()
        cmd = settings.get("usage_probe_cmd")
        if not cmd:
            return
        if not force:
            ttl_s = _probe_ttl_s(settings)
            meta = (self.store.read_probes().get(self.provider) or {}).get("_meta") or {}
            last_attempt = float(meta.get("last_attempt_ts") or 0)
            if now - last_attempt < ttl_s:
                return
        try:
            windows = parse_claude_probe_json(run_usage_probe(str(cmd)))
            if windows:
                self.store.record_probe(provider=self.provider, windows=windows, now=now)
            else:
                self.store.record_probe_attempt(
                    provider=self.provider, error="no windows parsed", now=now
                )
        except Exception as e:
            self.store.record_probe_attempt(
                provider=self.provider, error=f"{type(e).__name__}: {e}"[:200], now=now
            )

    def _recent_tokens(self, entries: list[LedgerEntry]) -> list[int]:
        return [e.tokens for e in entries if e.provider == self.provider][-20:]

    def _estimate_from(self, settings: dict[str, Any], recent: list[int]) -> int:
        manual = settings.get("node_estimate_tokens")
        if manual is not None:
            try:
                # 0 is a deliberate "reserve nothing", not "unset"
                return max(0, int(manual))
            except (TypeError, ValueError):
                pass
        if recent:
            ordered = sorted(recent)
            return max(1, ordered[len(ordered) // 2])
        return DEFAULT_NODE_ESTIMATE_TOKENS

    def _window_statuses(
        self,
        settings: dict[str, Any],
        entries: list[LedgerEntry],
        claims: dict[str, Any],
        probes: dict[str, Any],
        model: str,
        now: float,
    ) -> list[WindowStatus]:
        plan = str(settings.get("plan") or "")
        seeds = PLAN_SEEDS.get(plan, PLAN_SEEDS["max_5x"]) if self.provider == "claude" else {}
        manual_ceilings = settings.get("ceilings") or {}
        cap_pct = settings.get("cap_pct") or {}
        account_cap_pct = settings.get("account_cap_pct") or {}
        calibration = _provider_calibration(settings, self.provider)
        provider_probes = probes.get(self.provider) or {}
        model_l = (model or "").lower()
        ttl_s = _probe_ttl_s(settings)

        statuses: list[WindowStatus] = []
        for window, (seconds, model_filter) in WINDOWS.items():
            if model_filter is not None and model_filter not in model_l:
                continue
            cal = calibration.get(window) if isinstance(calibration.get(window), dict) else {}
            probe = provider_probes.get(window)
            probe = probe if isinstance(probe, dict) else None

            # The provider window is a discrete block, not a rolling one. A probe
            # reset time bounds that block: entries before it belong to a prior,
            # already-reset block. That boundary must survive past its own reset
            # (a stale probe) so pre-reset usage can't resurrect (B3) — even
            # though the probe's *utilization* goes stale once the block rolls.
            raw_resets_at = None
            if probe is not None:
                pr = probe.get("resets_at")
                raw_resets_at = float(pr) if pr else None
            # a CLI-detected limit (record_rate_limit) carries the same kind of
            # reset boundary as a probe; honour the later of the two so a 429's
            # block also survives past its reset to exclude pre-reset usage (A2)
            cal_reset = cal.get("reset_at")
            if cal_reset:
                cal_reset = float(cal_reset)
                raw_resets_at = cal_reset if raw_resets_at is None else max(raw_resets_at, cal_reset)
            probe_ts_val = float(probe.get("ts") or 0) if probe is not None else 0.0
            # a saturated probe with no reset epoch can't be refreshed for codex
            # (which only re-probes by harvesting a rollout AFTER a node runs);
            # once it ages past its TTL, drop its utilization so one node may run
            # and re-harvest, instead of parking the provider forever (A3/B1)
            no_reset_stale = (
                raw_resets_at is None and probe_ts_val > 0 and (now - probe_ts_val) > ttl_s
            )
            probe_stale = (raw_resets_at is not None and raw_resets_at <= now) or no_reset_stale
            active_probe = None if probe_stale else probe
            if raw_resets_at is None:
                block_start = None
            elif raw_resets_at > now:
                block_start = raw_resets_at - seconds  # start of the current block
            else:
                block_start = raw_resets_at  # a new block began at the reset

            manual = manual_ceilings.get(window)
            observed = int(cal.get("observed_ceiling") or 0)
            if manual:
                ceiling, source = int(manual), "manual"
            elif observed > 0:
                # calibrated from a real limit hit or probe deltas; beats seeds
                ceiling, source = observed, "calibrated"
            elif window in seeds:
                ceiling, source = int(seeds[window]), "seed"
            elif active_probe is not None:
                # no token ceiling yet, but the provider reports utilization %
                ceiling, source = 0, "probe-only"
            else:
                continue
            try:
                pct = float(cap_pct.get(window, 100))
            except (TypeError, ValueError):
                pct = 100.0
            # None = no own-spend cap (probe-only: the account gate governs it);
            # 0 = a real 0% cap that must hard-block, not silently admit (A8)
            allowed = None if ceiling <= 0 else int(ceiling * max(0.0, min(100.0, pct)) / 100.0)
            blocked_until = cal.get("blocked_until")
            blocked_until = float(blocked_until) if blocked_until else None
            if blocked_until is not None and blocked_until <= now:
                blocked_until = None
            resets_at = raw_resets_at if (raw_resets_at is not None and raw_resets_at > now) else None
            window_entries = _entries_in_block(
                entries,
                provider=self.provider,
                seconds=seconds,
                model_filter=model_filter,
                block_start=block_start,
                now=now,
            )
            probe_ts = float(active_probe.get("ts") or 0) if active_probe else None
            try:
                acct_cap = float(account_cap_pct.get(window, 90))
            except (TypeError, ValueError):
                acct_cap = 90.0
            statuses.append(
                WindowStatus(
                    window=window,
                    seconds=seconds,
                    model_filter=model_filter,
                    ceiling=ceiling,
                    ceiling_source=source,
                    allowed=allowed,
                    usage=sum(e.tokens for e in window_entries),
                    claims=_claims_sum(
                        claims, provider=self.provider, model_filter=model_filter, now=now
                    ),
                    blocked_until=blocked_until,
                    probe_pct=float(active_probe.get("used_pct")) if active_probe and active_probe.get("used_pct") is not None else None,
                    probe_ts=probe_ts,
                    resets_at=resets_at,
                    account_cap_pct=acct_cap,
                    own_since_probe=sum(
                        e.tokens for e in window_entries if probe_ts and e.ts >= probe_ts
                    ),
                    block_start=block_start,
                )
            )
        return statuses

    def _account_gate_wait(
        self, st: WindowStatus, est: int, ttl_s: float, now: float
    ) -> float | None:
        """Seconds to wait for the account-level cap, or None when it admits.

        Protects against usage that happened OUTSIDE ProofCouncil: the probed
        utilization plus our estimated drift since the probe must stay under
        account_cap_pct. Drift converts tokens to percent via the ceiling
        estimate; with no ceiling yet (probe-only window) a stale probe is
        penalized instead so we neither stall nor sail blindly past the cap.
        """
        if st.probe_pct is None or st.account_cap_pct is None:
            return None
        drift_tokens = st.own_since_probe + st.claims + est
        if st.ceiling > 0:
            drift_pct = 100.0 * drift_tokens / st.ceiling
        else:
            # probe-only window: a fresh probe already at the cap must block any
            # pending spend, since we have no ceiling to convert drift to percent
            if drift_tokens > 0 and st.probe_pct >= st.account_cap_pct:
                if st.resets_at is not None and st.resets_at > now:
                    return st.resets_at - now
                return DEFAULT_RECHECK_S
            stale = st.probe_ts is None or (now - st.probe_ts) > ttl_s * 1.5
            drift_pct = 10.0 if stale else 0.0
        if st.probe_pct + drift_pct <= st.account_cap_pct:
            return None
        if st.resets_at is not None and st.resets_at > now:
            return st.resets_at - now
        return DEFAULT_RECHECK_S  # next probe refresh may show roll-off

    def _decide_from(
        self,
        settings: dict[str, Any],
        entries: list[LedgerEntry],
        claims: dict[str, Any],
        probes: dict[str, Any],
        *,
        model: str,
        now: float,
    ) -> PacingDecision:
        est = self._estimate_from(settings, self._recent_tokens(entries))
        statuses = self._window_statuses(settings, entries, claims, probes, model, now)
        ttl_s = _probe_ttl_s(settings)

        # blocking_window must name the window that produced the LONGEST wait,
        # or parked-run diagnostics blame the wrong window
        blocking: str | None = None
        wait_s = 0.0
        for st in statuses:
            window_wait: float | None = None
            if st.blocked_until is not None:
                window_wait = max(0.0, st.blocked_until - now)
            account_wait = self._account_gate_wait(st, est, ttl_s, now)
            if account_wait is not None:
                # a short-lived claim inflates the account drift too, so bound
                # the account wait by the soonest claim expiry — else a 30s claim
                # projects to the hours-away provider reset (A4, the account-gate
                # sibling of the own-spend A11 bound below)
                claim_expiry = _soonest_claim_expiry(
                    claims, provider=self.provider, model_filter=st.model_filter, now=now
                )
                if claim_expiry is not None:
                    account_wait = min(account_wait, max(0.0, claim_expiry - now))
                window_wait = max(window_wait or 0.0, account_wait)
            # own-spend cap. allowed is None only for probe-only windows (no
            # token ceiling) — those are governed by the account gate above.
            if window_wait is None and st.allowed is not None:
                if st.allowed <= 0:
                    # a real 0% cap hard-closes the window; the untouched-window
                    # rule must not rescue it. Re-check so raising the cap (a
                    # live settings edit) admits without a restart (A8).
                    window_wait = DEFAULT_RECHECK_S
                elif st.usage + st.claims > 0 and st.usage + st.claims + est > st.allowed:
                    # untouched windows (zero usage and claims) never stall: an
                    # estimate alone must not deadlock a fresh window.
                    deficit = st.usage + st.claims + est - st.allowed
                    window_entries = _entries_in_block(
                        entries,
                        provider=self.provider,
                        seconds=st.seconds,
                        model_filter=st.model_filter,
                        block_start=st.block_start,
                        now=now,
                    )
                    cum = 0
                    window_wait = DEFAULT_RECHECK_S  # claims may free up before ledger ages out
                    for e in window_entries:
                        cum += e.tokens
                        if cum >= deficit:
                            window_wait = max(0.0, e.ts + st.seconds - now)
                            break
                    # bound the wait by the soonest event that can free headroom:
                    # a claim expiring (A11) or a discrete provider reset that
                    # frees all own-spend at once (A5/B3). Never project past
                    # either into a needless hours-long park.
                    claim_expiry = _soonest_claim_expiry(
                        claims, provider=self.provider, model_filter=st.model_filter, now=now
                    )
                    if claim_expiry is not None:
                        window_wait = min(window_wait, max(0.0, claim_expiry - now))
                    if st.resets_at is not None and st.resets_at > now:
                        window_wait = min(window_wait, st.resets_at - now)
            if window_wait is None:
                continue
            if blocking is None or window_wait > wait_s:
                blocking, wait_s = st.window, window_wait
        return PacingDecision(
            admit=blocking is None,
            est_tokens=est,
            blocking_window=blocking,
            wait_s=wait_s,
            windows=statuses,
        )

    def decide(self, *, model: str, now: float | None = None) -> PacingDecision:
        now = now if now is not None else time.time()
        settings, entries, claims, probes = self.store.snapshot()
        return self._decide_from(settings, entries, claims, probes, model=model, now=now)

    def try_claim(
        self,
        *,
        model: str,
        run_id: str = "",
        ttl_s: float = DEFAULT_CLAIM_TTL_S,
        now: float | None = None,
    ) -> tuple[str | None, PacingDecision]:
        """Atomically decide and, if admitted, register the in-flight claim.

        Read + decide + claim-write happen under ONE lock acquisition so two
        concurrent nodes (same or different process) cannot both admit into
        the same last slot of headroom.
        """
        now = now if now is not None else time.time()
        self.ensure_fresh_probe(now)  # slow subprocess; must run before the lock
        store = self.store
        with store._locked():
            settings = store._read_settings()
            entries = store._read_ledger()
            claims = store._read_claims()
            probes = store._read_probes()
            decision = self._decide_from(
                settings, entries, claims, probes, model=model, now=now
            )
            if not decision.admit:
                return None, decision
            claim_id = uuid.uuid4().hex[:12]
            claims = store._prune_claims(claims, now)
            claims[claim_id] = _claim_record(
                provider=self.provider,
                model=model,
                est_tokens=decision.est_tokens,
                run_id=run_id,
                ttl_s=ttl_s,
                now=now,
            )
            store._write_claims(claims)
        return claim_id, decision

    def release(self, claim_id: str) -> None:
        self.store.remove_claim(claim_id)

    def status(self, *, model: str = "", now: float | None = None) -> dict[str, Any]:
        """Snapshot for the dashboard: settings + per-window usage/headroom.

        The default model string matches every model-class filter so all
        tracked windows are reported.
        """
        now = now if now is not None else time.time()
        settings, entries, claims, probes = self.store.snapshot()
        statuses = self._window_statuses(
            settings, entries, claims, probes, model or "opus fable", now
        )
        meta = (probes.get(self.provider) or {}).get("_meta") or {}
        return {
            "enabled": bool(settings.get("enabled")),
            "provider": self.provider,
            "plan": settings.get("plan"),
            "park_after_s": _park_after_s(settings),
            "node_estimate_tokens": self._estimate_from(settings, self._recent_tokens(entries)),
            "probe": {
                "configured": bool(settings.get("usage_probe_cmd")),
                "last_attempt_ts": meta.get("last_attempt_ts"),
                "last_error": meta.get("last_error"),
            },
            "windows": [
                {
                    **asdict(st),
                    "headroom": None
                    if st.allowed is None
                    else max(0, st.allowed - st.usage - st.claims),
                }
                for st in statuses
            ],
        }


__all__ = [
    "DEFAULT_NODE_ESTIMATE_TOKENS",
    "DEFAULT_PARK_AFTER_S",
    "DEFAULT_RECHECK_S",
    "LedgerEntry",
    "PacingDecision",
    "PLAN_SEEDS",
    "RateLimitHit",
    "SubscriptionPacer",
    "SubscriptionParked",
    "SubscriptionStore",
    "WINDOWS",
    "WindowStatus",
    "detect_rate_limit",
    "pacing_env_override",
    "subscription_home",
]
