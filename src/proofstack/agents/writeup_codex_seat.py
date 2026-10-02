"""Subscription seat for WriteupLoop: one prompt through ``codex exec``.

The default seat (``writeup_loop._OneShotSeat``) talks to the OpenAI
Responses API and bills an API key. This module is the same seat contract
-- one user turn in, the model's final message out -- served instead by
the local codex CLI on the ChatGPT subscription: same model family, same
server-side web search, zero API dollars.

Deliberate choices a maintainer should know about:

* The binary is resolved by FULL PATH (``~/.local/bin/codex``). It is not
  on PATH in non-login shells or systemd units, so a bare ``codex`` would
  fail with "command not found" inside an unattended run.
* ``-c tools.web_search=true`` keeps the web search the API seat gets via
  its ``{"type": "web_search"}`` tool. Dropping it would change what the
  frozen prompts can do, not just how they are billed.
* ``CODEX_HOME`` is inherited, not copied. ``configurable_cli.py`` copies
  ``auth.json`` into a per-run ``.codex-home`` because its CLI workers run
  inside a sandbox that cannot see ``$HOME``; this seat runs directly on
  the host, and inheriting the real home keeps token refresh working and
  keeps session rollouts under ``~/.codex/sessions`` where the campaign's
  quota probe reads its ``rate_limits`` lines.
* ``--ignore-user-config`` by default, for two reasons. (a) Fidelity: the
  frozen, approval-tracked prompts should be the only instruction the
  model gets, and ``~/.codex/config.toml`` injects a ``personality``
  directive plus project-doc fallbacks — style instructions, on a node
  whose entire job is style. (b) Billing: that same file is where
  ``codex-billing`` sets ``model_provider = "openai_api"`` to switch to a
  key-billed provider. Ignoring it pins this seat to the built-in openai
  provider. Set ``cli_ignore_user_config: false`` to opt back into the
  user config.
* It does NOT pin the CREDENTIAL, though: auth still comes from
  ``CODEX_HOME/auth.json``, which ``codex login --api-key`` fills with an
  API key, and from the ``OPENAI_API_KEY`` / ``CODEX_API_KEY`` /
  ``CODEX_ACCESS_TOKEN`` env vars. Either would bill real dollars while
  this seat reports $0. So the login is classified with the shared
  ``classify_codex_auth`` and a non-subscription one is refused, and those
  env vars are stripped from the child — the same stance, for the same
  reason, as ``configurable_cli.py`` (codex r7 #1).
* ``--json`` so usage comes from the machine-readable ``turn.completed``
  event; the human-log ``tokens used:`` line is kept only as a fallback.
* Containment has a documented limit. Survivors are found through /proc,
  by process group+session or by the marker in their environment; a
  descendant that BOTH leaves that identity (``setsid``, or just
  ``setpgid`` -- it need not leave the session) and hides its environment
  (non-dumpable, or exec'd with the marker stripped) is identifiable by
  neither, and no /proc-based sweep can reach it. Such an escapee is then
  bounded by nothing at all: not this seat's wallclock, not the node's
  budget. Real containment for that case needs a cgroup or a pid
  namespace, which this seat deliberately does not build: it runs codex
  directly on the host so that token refresh and the session rollouts
  keep working. It is accepted because codex's own children do neither
  thing -- not because the consequence would be small.
* No background/poll equivalent exists for the CLI. ``codex exec`` blocks
  for the whole turn, so the caller's wallclock bound is enforced here by
  killing the process GROUP (codex spawns children) and reporting an
  ordinary seat failure. The tree is cleaned up on EVERY exit path, not
  only the timeout: a descendant that redirected its own output can
  outlive a clean exit, and nothing would then be bounding it. Once the
  leader has been reaped its pid number is the kernel's to reissue, so
  after a normal exit the survivors are identified per process — same
  group AND session as the seat, or carrying the seat's marker
  environment variable, which is the only evidence that survives
  ``setsid`` — rather than signalled by that number.
* The reply and the log tail are redacted before they leave this module,
  not just before they are written to disk -- the reply becomes the
  document and the next seat's prompt -- and against every credential seen
  from before the turn to after it, auth.json being watched meanwhile
  because codex refreshes its token mid turn. A reply that actually needed
  redacting is reported as a seat FAILURE rather than returned scrubbed:
  silently substituting text inside a document's mathematics is not
  something the compile gate can catch.

Every failure mode leaves as ``CodexSeatError`` (an ``Exception``), which
is exactly what WriteupLoop's degradation paths already expect from a
seat. ``asyncio.CancelledError`` is re-raised after the child is killed,
so the node's cancellation contract is unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import bisect
import contextlib
import json
import math
import os
import re
import shutil
import signal
import string
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from proofstack.cli_usage import CodexUsage, parse_codex_jsonl
from proofstack.codex_auth import (
    classify_codex_auth,
    extract_codex_auth_secrets,
    redact_codex_secrets,
)

# Stamped into the child's environment and inherited by every descendant,
# so survivors can be identified by something no recycled pid can forge.
SEAT_MARKER_ENV = "PROOFSTACK_WRITEUP_SEAT_ID"

# Auth channels codex accepts from the environment. This seat records every
# call as $0, so any of them present would be silent unbilled spend.
BLOCKED_AUTH_ENVS = ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")

# Not on PATH in non-login shells / systemd services -- resolve by path.
DEFAULT_CODEX_BIN = "~/.local/bin/codex"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "max"
DEFAULT_SANDBOX = "read-only"

# Human-log terminator, e.g. "tokens used: 12,345". Only consulted when the
# JSONL usage event is missing (older CLI, truncated stdout).
_TOKENS_USED_RE = re.compile(r"tokens\s+used:?\s*([\d,]+)", re.IGNORECASE)

_LOG_TAIL_CHARS = 2000

# Credential SHAPES, matched without knowing the value. Watching auth.json
# can only catch a credential that existed while something looked: codex
# may mint a token, echo it, and rotate it away between two polls, and no
# amount of polling closes that (codex r11 #1). These tests do not care --
# an OpenAI key or a JWT in a reply is refused whether or not it was ever
# seen on disk.
#
# Both are deliberately narrow, because a false positive costs a DISCARDED
# CANDIDATE, and a document whose every draft trips one would ship
# unimproved: `\label{task-polynomial-estimates}` contains the substring
# "sk-polynomial-estimates", which a bare `sk-\w{16,}` match happily
# claimed (codex r12 #1). So the key pattern must start at a word boundary
# and the secret part must look random (all three character classes), and
# a JWT candidate must actually base64url-decode to a JSON header with an
# "alg" -- which also catches the header encodings `eyJ` misses, e.g. the
# RFC-legal `{ "alg": ...}` -> `eyAi` (codex r12 #2).
# Three ways to be an API key, because one rule cannot both catch a real
# key and spare a LaTeX identifier:
#   RUN       `sk-` + optional short hyphenated prefixes + one UNBROKEN
#             random alphanumeric run (the classic 48-character key).
#   PREFIXED  a known provider prefix, after which separators in the body
#             are ordinary: `sk-proj-Ab3dEf9_...` never has a 16-character
#             unbroken run and so escaped the RUN rule entirely
#             (codex r14 #1).
#   LONG      any other `sk-` followed by a long separator-bearing body
#             that measures random — the catch-all for a provider format
#             nobody here has seen yet.
# Left-bounded on alphanumerics only, so `task-...` cannot start a match
# but `_sk-..._` (Markdown emphasis round a token) can (codex r13 #1).
# Every quantifier is bounded. An unbounded one scans to the end of the
# run from EVERY admissible start, which `"sk-" * 8000` turns into three
# seconds of blocked event loop (codex r16 #2). The caps are far above any
# real key (the longest project keys are ~170 characters), and a match is
# extended to the end of its run afterwards, so a longer one is still
# redacted whole.
_API_KEY_RUN_RE = re.compile(
    r"(?<![A-Za-z0-9])sk-(?:[A-Za-z0-9]{1,12}-){0,4}([A-Za-z0-9]{16,250})")
_API_KEY_PREFIXED_RE = re.compile(
    r"(?<![A-Za-z0-9])sk-(?:proj|ant|svcacct|admin|live|test|None)"
    r"-([A-Za-z0-9_-]{20,250})")
_API_KEY_LONG_RE = re.compile(r"(?<![A-Za-z0-9])sk-([A-Za-z0-9_-]{40,250})")
_RUN_ALPHABET = frozenset(string.ascii_letters + string.digits)
_BODY_ALPHABET = _RUN_ALPHABET | frozenset("-_")
_LONG_KEY_MIN_ENTROPY = 4.0
# A JOSE header is small; a first segment longer than this is not one.
_JWT_HEADER_MAX = 344
_JWT_SEGMENT_MIN = 8
# How far into a candidate a wrapper prefix is looked for.
_JWT_LEAD_SCAN = 64
# Reject a LONG body with three or more word segments. Two would spare a
# hyphenated prose identifier like `sk-DeligneMumford-compactification-...`
# at the cost of ~1 in 1000 random bodies; for a screen whose false
# positive costs a re-roll and whose false negative ships a credential,
# three is the right way round. (Real OpenAI and Anthropic keys are caught
# by RUN and PREFIXED whatever this is set to.)
_LONG_KEY_MAX_WORDS = 3
# Longer than any real key: a run past this is not one credential, so the
# catch-all rule declines it rather than measuring the whole thing.
_LONG_KEY_MAX_LEN = 512
_ACCEPT_WINDOW = 512
# Left-bounded on the WHOLE segment alphabet, for cost: every position the
# lookbehind admits is a position from which the greedy match scans the
# rest of the run, so admitting `_` and `-` left 64k of underscores taking
# 7.6s with the event loop — and cancellation — blocked throughout (codex
# r14 #3, r15 #3). A run now has exactly one admissible start. Emphasis is
# still handled: in `___token`, that one start is the first underscore, and
# ``_is_jwt`` trims the leading run off before decoding.
_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    r"\.[A-Za-z0-9_-]{8,}")

# LaTeX commands whose braces hold an identifier chosen by the author. An
# identifier can look enough like a classic key to get every candidate
# discarded (codex r13 #4, r14 P3), so a match of the WEAKEST rule is
# ignored inside one of these. Only that rule: exempting the argument
# outright let a real `sk-proj-...` key ride into the shipped document
# inside `\label{...}` (codex r15 #1). Possessive quantifiers because two
# adjacent `\s*` over `\cite` + 32k spaces took 2.5s (codex r15 #2). The
# caps bound pathological input; they are not limits on real syntax. At
# 200 and 500, a long postnote or a two-dozen-key citation lost its
# exemption and the citation KEY was then flagged (codex r17 #2).
_IDENTIFIER_ARG_RE = re.compile(
    r"\\(?:cite[a-zA-Z]*|label|eqref|pageref|autoref|nameref|[cCvV]ref"
    r"|ref|bibitem)\*?\s*+(?:\[[^]\n]{0,500}\]\s*+){0,2}"
    r"\{([^{}]{0,4000})\}")


def _is_random_looking(token: str) -> bool:
    """Upper, lower and digit all present — necessary for a real key's
    random body, and false of most hyphenated words."""
    return (any(c.isupper() for c in token) and any(c.islower() for c in token)
            and any(c.isdigit() for c in token))


def _entropy(token: str) -> float:
    """Shannon entropy per character. A random base64url body of this
    length measures about 4.3 at worst; a hyphenated phrase of the same
    length measures about 4.2 at best, so entropy alone does NOT separate
    them — it is one of two tests, not the test."""
    if not token:
        return 0.0
    counts = Counter(token)
    n = len(token)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _word_segments(token: str) -> int:
    """How many of the token's separator-delimited pieces are WORDS —
    alphabetic, at least three characters, and cased like a word (all
    lower, or one leading capital). This is what an identifier like
    `Sobolev-estimates-for-elliptic-equations-2024b` is made of. A random
    body's alphabetic pieces are mixed-case (`HdlfG`, `woGiAg`), which is
    why the case test matters: counting those as words lost 3% of random
    bodies to the word rule (codex r15 P3)."""
    return sum(1 for part in re.split(r"[-_]", token)
               if len(part) >= 3 and part.isalpha()
               and (part.islower() or (part[:1].isupper() and part[1:].islower())))


def _identifier_arg_spans(text: str) -> tuple[list[int], list[int]]:
    """Starts and ends of the cross-reference arguments in the text, as
    two parallel sorted lists — the shape ``_within`` can binary-search
    without rebuilding anything per lookup."""
    starts: list[int] = []
    ends: list[int] = []
    for m in _IDENTIFIER_ARG_RE.finditer(text):
        starts.append(m.start(1))
        ends.append(m.end(1))
    return starts, ends


def _within(spans: tuple[list[int], list[int]], start: int, end: int) -> bool:
    """Binary search, not a scan: a document with thousands of valid
    references made the per-match lookup quadratic (codex r16 #3). The
    spans come from ``finditer``, so they are sorted and disjoint."""
    starts, ends = spans
    i = bisect.bisect_right(starts, start) - 1
    return i >= 0 and end <= ends[i]


def _decodes_to_jose_header(segment: str) -> bool:
    try:
        raw = base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
        obj = json.loads(raw)
    except Exception:  # noqa: BLE001 — not base64, not JSON: not a header
        return False
    return isinstance(obj, dict) and "alg" in obj


def _is_jwt(candidate: str) -> tuple[bool, int]:
    """Whether the candidate really is a JWT — decided by DECODING its
    first segment into a JOSE header, not by the `eyJ` spelling of one
    encoding (codex r12 #2) — and where inside the candidate the token
    actually starts.

    A wrapper can be more than emphasis: ``\\label{jwt_<JWT>}`` puts a whole
    word in front, and trimming only leading `_`/`-` decoded the wrong
    header and missed it (codex r16 #1). Base64 is position-sensitive, so
    the true start has to be tried: offset 0 and every position after a
    separator, within a bounded lead. The decode is bounded too — a JOSE
    header is a couple of hundred bytes, so a longer first segment is not
    one, and neither bound lets a long run cost more than a constant."""
    head = candidate.split(".", 1)[0]
    # Offset 0; past ANY leading run of emphasis characters, however long;
    # and past each separator within a bounded lead, which is what finds
    # the token after a word like `jwt_`.
    starts = [0, len(head) - len(head.lstrip("_-"))]
    starts += [i + 1 for i, c in enumerate(head[:_JWT_LEAD_SCAN])
               if c in "_-"]
    for offset in dict.fromkeys(starts):
        segment = head[offset:]
        if (_JWT_SEGMENT_MIN <= len(segment) <= _JWT_HEADER_MAX
                and _decodes_to_jose_header(segment)):
            return True, offset
    return False, 0


def _run_end(text: str, end: int, alphabet: frozenset[str]) -> int:
    """Extend a capped match to the end of its run, so a key longer than
    the pattern's bound is still redacted whole."""
    while end < len(text) and text[end] in alphabet:
        end += 1
    return end


def _scan_keys(pattern: "re.Pattern[str]", text: str,
               alphabet: frozenset[str],
               accept: "Callable[[str], bool]",
               window: int) -> list[tuple[int, int]]:
    """Every accepted match of one key rule, each span extended to the end
    of its run.

    Driven by hand rather than ``finditer`` because of what the extension
    costs otherwise: ``finditer`` resumes at the CAPPED end, so a long run
    yields a match every few characters and each one walks the same suffix
    again — 125k of repeated key text took 2.8 seconds with the event loop
    blocked (codex r17 #1). An accepted match therefore skips its whole
    run, and the last run walked is remembered so a rejected one does not
    walk it again either. Acceptance tests the EXTENDED body, not the
    capped prefix, so a long identifier is judged on all of itself
    (codex r17 #3) — but only ``window`` characters of it are ever COPIED
    out, because building the argument was itself unbounded work and cost
    4 seconds on 4MB of repeated key text (codex r18 P3). Each rule's
    window is the most its test can actually look at, so the verdict is
    unchanged."""
    spans: list[tuple[int, int]] = []
    pos, known_from, known_to = 0, -1, -1
    while pos <= len(text):
        m = pattern.search(text, pos)
        if m is None:
            break
        if known_from <= m.start() and m.end() <= known_to:
            end = known_to
        else:
            end = _run_end(text, m.end(), alphabet)
            known_from, known_to = m.start(), end
        body_start = m.start(1)
        if accept(text[body_start:min(end, body_start + window)]):
            spans.append((m.start(), end))
            pos = max(end, m.end())
        else:
            pos = max(m.end(), m.start() + 1)
    return spans


def _is_long_key_body(body: str) -> bool:
    """LONG is the catch-all rule, so it judges the WHOLE extended body —
    and refuses a body longer than any real key, which keeps that judgment
    O(1)-bounded. Without the length bound, a rejected match re-measured
    the entropy of a 500k run every 250 characters (codex r17 #1)."""
    return (len(body) <= _LONG_KEY_MAX_LEN
            and _is_random_looking(body)
            and _entropy(body) >= _LONG_KEY_MIN_ENTROPY
            and _word_segments(body) < _LONG_KEY_MAX_WORDS)


def _has_random_prefix(body: str) -> bool:
    """RUN and PREFIXED already required their own prefix to match, so the
    whole body need not be examined — a window is enough, and is what
    keeps a rejected match cheap inside a very long run."""
    return _is_random_looking(body[:_ACCEPT_WINDOW])


def _api_key_spans(text: str) -> list[tuple[int, int]]:
    """RUN matches are dropped inside a cross-reference argument, where an
    author's identifier lives; PREFIXED and LONG never are, because no
    LaTeX context makes a real key safe to keep."""
    identifiers = _identifier_arg_spans(text)
    spans = [(start, end) for start, end in
             _scan_keys(_API_KEY_RUN_RE, text, _RUN_ALPHABET,
                        _has_random_prefix, _ACCEPT_WINDOW)
             if not _within(identifiers, start, end)]
    spans += _scan_keys(_API_KEY_PREFIXED_RE, text, _BODY_ALPHABET,
                        _has_random_prefix, _ACCEPT_WINDOW)
    # One character past the rule's own length bound is all it needs to
    # see to decline an over-long body.
    spans += _scan_keys(_API_KEY_LONG_RE, text, _BODY_ALPHABET,
                        _is_long_key_body, _LONG_KEY_MAX_LEN + 1)
    return spans


def _credential_spans(text: str) -> list[tuple[int, int]]:
    """Half-open spans of everything credential-shaped, merged and
    non-overlapping."""
    text = str(text)
    spans = _api_key_spans(text)
    for m in _JWT_RE.finditer(text):
        found, offset = _is_jwt(m.group(0))
        if found:
            spans.append((m.start() + offset, m.end()))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged

# How often auth.json is re-read while a turn runs. codex refreshes the
# ChatGPT token mid-turn; a value observed only between two refreshes is
# scrubbable only if something looked while it was there (codex r11 #1).
_AUTH_POLL_SECONDS = 2.0


class CodexSeatError(RuntimeError):
    """A codex-CLI seat call that produced no usable final message.

    ``partial`` holds the turn's parsed usage when it ran far enough to
    report any, so the caller can charge what a failing seat really spent
    (codex r9 #2). ``None`` when the turn produced no usage at all."""

    partial: "CodexSeatResult | None" = None


@dataclass
class CodexSeatResult:
    text: str
    usage: CodexUsage = field(default_factory=CodexUsage)
    returncode: int = 0
    duration_s: float = 0.0
    fallback_total_tokens: int = 0
    log_tail: str = ""

    @property
    def metered_tokens(self) -> int:
        """Tokens to charge the budget tracker. The subscription's real
        limit is throughput, so input+output is what we meter; cached
        input is included because the provider counts it against the
        window too."""
        total = self.usage.input_tokens + self.usage.output_tokens
        return total or self.fallback_total_tokens


def resolve_codex_bin(raw: str | None = None) -> str:
    candidate = Path(os.path.expanduser(raw or DEFAULT_CODEX_BIN))
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    found = shutil.which(candidate.name)
    if found:
        return found
    raise CodexSeatError(f"codex binary not found (looked at {candidate})")


def build_codex_cmd(
    *,
    codex_bin: str,
    model: str,
    reasoning_effort: str,
    workdir: Path | str,
    last_message_path: Path | str,
    web_search: bool = True,
    sandbox: str = DEFAULT_SANDBOX,
    ignore_user_config: bool = True,
) -> list[str]:
    """The exec invocation. ``-`` is the prompt argument: instructions are
    read from stdin, so a multi-megabyte guide + document never touches
    argv. Effort is quoted so codex's ``-c`` parses it as TOML string,
    matching ``configurable_cli._with_codex_reasoning_effort``."""
    cmd = [codex_bin, "exec"]
    if ignore_user_config:
        cmd.append("--ignore-user-config")
    cmd += [
        "-m",
        model,
        # Pin the provider whatever the user config says. Ignoring the user
        # config is an opt-out from config.toml's PROMPT-shaping keys, and
        # turning that opt-out off must not hand BILLING to the same file:
        # its top-level `model_provider` is where codex-billing selects the
        # key-billed provider, while this seat books every call at $0
        # (codex r8 #1). Verified on codex 0.153.4: this override beats
        # config.toml's `model_provider`, and codex rejects a config that
        # redefines the built-in "openai" id, so the file has no way back
        # to a paid route.
        "-c",
        'model_provider="openai"',
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
    ]
    if web_search:
        cmd += ["-c", "tools.web_search=true"]
    cmd += [
        "-s",
        sandbox,
        "-C",
        str(workdir),
        # The seat runs in a scratch dir, which is not a git repo.
        "--skip-git-repo-check",
        "--color",
        "never",
        "--json",
        "--output-last-message",
        str(last_message_path),
        "-",
    ]
    return cmd


def child_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """The host environment the seat hands to codex, minus the API-key auth
    channels (see ``BLOCKED_AUTH_ENVS``)."""
    merged = {**os.environ, **(env or {})}
    for key in BLOCKED_AUTH_ENVS:
        merged.pop(key, None)
    return merged


def assert_subscription_login(env: dict[str, str]) -> None:
    """Refuse to spawn unless ``CODEX_HOME`` holds a ChatGPT login.

    ``--ignore-user-config`` neutralizes codex-billing's ``model_provider``
    switch but not the credential, and ``_record_cli_usage`` books every
    call at $0 — so an API-key ``auth.json`` would spend real money
    invisibly. Raises ``CodexSeatError``, i.e. an ordinary seat failure,
    so the caller's degradation paths are unchanged (codex r7 #1)."""
    auth = _auth_path(env)
    try:
        auth_text = auth.read_text(encoding="utf-8")
    except OSError:
        auth_text = None
    kind = classify_codex_auth(auth_text)
    if kind != "subscription":
        raise CodexSeatError(
            f"codex login at {auth} classifies as '{kind}', not a ChatGPT "
            "subscription; refusing to spawn a seat that records $0")


def _auth_path(env: dict[str, str] | None = None) -> Path:
    src = (env or os.environ).get("CODEX_HOME") or "~/.codex"
    return Path(os.path.expanduser(src)) / "auth.json"


_SHAPE_REDACTION = "[redacted-credential-shaped]"


def auth_secrets(env: dict[str, str] | None = None, *,
                 additional: Iterable[str] = ()) -> tuple[str, ...]:
    """Credential values currently in ``CODEX_HOME/auth.json``, plus any
    ``additional`` ones the caller is carrying — the accumulate-across-a-
    refresh shape ``configurable_cli._refresh_transient_codex_secrets``
    uses. An unreadable or unparsable file (absent, or caught mid-rewrite
    by a token refresh) contributes nothing and raises nothing: a snapshot
    that fails is one fewer string to scrub, not a reason to lose a
    reply."""
    try:
        auth_text: str | None = _auth_path(env).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        auth_text = None
    try:
        # Parsing is guarded too, not just the read: auth.json is written
        # by another program, and a pathological one (deep nesting ->
        # RecursionError) escaping from here skipped the caller's process
        # cleanup entirely (codex r11 #8).
        return extract_codex_auth_secrets(auth_text, additional=additional)
    except Exception:  # noqa: BLE001
        return tuple(sorted({v.strip() for v in additional
                             if isinstance(v, str) and len(v.strip()) >= 8}))


def looks_like_credential(text: str) -> bool:
    """True when the text contains something SHAPED like a credential,
    whatever auth.json happens to hold right now."""
    return bool(_credential_spans(str(text)))


def scrub_credentials(text: str, secrets: Iterable[str]) -> str:
    """Known values first, then anything credential-shaped."""
    out = redact_codex_secrets(str(text), secrets)
    spans = _credential_spans(out)
    for start, end in reversed(spans):
        out = out[:start] + _SHAPE_REDACTION + out[end:]
    return out


def redact_seat_text(text: str, *, extra_secrets: Iterable[str] = (),
                     env: dict[str, str] | None = None) -> str:
    """Scrub credentials out of seat text — a run artifact, or a reply on
    its way back into the loop. Seat output quotes a codex log tail, which
    can echo one (codex r8 #6).

    Reading ``auth.json`` at redaction time is not sufficient by itself:
    codex refreshes the ChatGPT token during a long turn, so the value a
    log echoed early is no longer the value on disk, and an unreadable
    file used to return the text untouched (codex r10 #2). Callers holding
    a pre-call snapshot pass it as ``extra_secrets``; both sets are
    scrubbed, so a rotation across the call leaves neither value exposed.
    """
    return scrub_credentials(
        str(text), auth_secrets(env, additional=extra_secrets))


def _proc_fs_available() -> bool:
    return os.path.isdir("/proc/self")


def _in_seat_session(pid: int, pgid: int, *, use_proc: bool | None = None) -> bool:
    """True when ``pid`` is in the seat's process group AND its session.

    ``start_new_session=True`` made the codex leader both, so every
    descendant that did not call ``setsid`` has pgrp == sid == pgid. A
    recycled pid number would have to be BOTH again to match, which needs
    an unrelated process to have called ``setsid`` after landing on
    exactly that number."""
    try:
        if use_proc is None:
            use_proc = _proc_fs_available()
        if not use_proc:
            return os.getpgid(pid) == pgid and os.getsid(pid) == pgid
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            stat = fh.read()
        # comm (field 2) is parenthesised and may contain spaces.
        fields = stat[stat.rindex(")") + 2:].split()
        return int(fields[2]) == pgid and int(fields[3]) == pgid
    except Exception:  # noqa: BLE001 — gone, or unreadable: not ours
        return False


def _carries_marker(pid: int, marker: str, *, use_proc: bool | None = None) -> bool:
    """True when ``pid`` was exec'd with the seat's marker in its
    environment — the only evidence that survives ``setsid``, reparenting
    to init, and pid recycling. Best effort: ``/proc/<pid>/environ`` needs
    read permission on the target, which a hardened host may withhold."""
    try:
        if use_proc is None:
            use_proc = _proc_fs_available()
        if not use_proc:
            import psutil

            return psutil.Process(pid).environ().get(SEAT_MARKER_ENV) == marker
        with open(f"/proc/{pid}/environ", "rb") as fh:
            return f"{SEAT_MARKER_ENV}={marker}".encode() in fh.read()
    except Exception:  # noqa: BLE001
        return False


def _is_seat_survivor(pid: int, pgid: int, marker: str) -> bool:
    return _carries_marker(pid, marker) or _in_seat_session(pid, pgid)


def seat_survivors(pgid: int, marker: str) -> list[int]:
    """Processes still belonging to this seat call, by one process scan.

    Two independent tests, because neither alone is enough: the
    group/session test catches the ordinary descendant but not one that
    called ``setsid``, and the marker test catches that one, and one whose
    pid number was recycled, but needs permission to read its environment.
    Both are identity checks on the process itself, which is what makes
    signalling it safer than ``killpg`` on a number the kernel may by now
    have handed to somebody else (codex r11 #3).

    Residual, documented rather than claimed away: a process that acquired
    exactly the seat's old pid AND made itself the leader of a group and
    session of that number matches the first test without the marker. That
    needs the pid space to wrap between the leader's exit and this scan.
    Dropping the group/session test instead would let any descendant whose
    environment we cannot read survive, which is the likelier accident of
    the two."""
    self_pid = os.getpid()
    try:
        use_proc = _proc_fs_available()
        if use_proc:
            pids = [int(entry) for entry in os.listdir("/proc") if entry.isdigit()]
        else:
            import psutil

            pids = psutil.pids()
    except Exception:  # noqa: BLE001 -- cleanup must also tolerate psutil.Error
        return []
    found: list[int] = []
    for pid in pids:
        if pid == self_pid:
            continue
        if (_in_seat_session(pid, pgid, use_proc=use_proc)
                or _carries_marker(pid, marker, use_proc=use_proc)):
            found.append(pid)
    return found


def kill_seat_survivors(pgid: int, marker: str, passes: int = 10) -> int:
    """SIGKILL whatever is left of the seat's process tree. Returns the
    number signalled.

    Swept repeatedly until a pass finds nothing: a survivor can fork
    between the scan and its own death, and that child is in no snapshot
    taken so far (codex r11 #7); one extra pass is not enough either, since
    the same race repeats a generation down (codex r12 #8). SIGKILL cannot
    be caught, so a process already signalled forks no more and each pass
    has strictly fewer generations left to find. The pass cap is a
    termination guard, not an expectation: a descendant forking hard enough
    to outrun ten passes is the same case as one that hides its identity —
    only a cgroup would contain it.

    Each pid is re-identified immediately before it is signalled, which
    narrows (it cannot close) the window in which a pid dies and is
    reissued between being listed and being killed."""
    killed = 0
    for _ in range(max(1, passes)):
        found = seat_survivors(pgid, marker)
        if not found:
            break
        for pid in found:
            if not _is_seat_survivor(pid, pgid, marker):
                continue      # died and was reissued since the scan
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                continue
            killed += 1
    return killed


def _kill_process_group(proc: asyncio.subprocess.Process,
                        pgid: int | None = None) -> None:
    """codex spawns children; killing only the direct child can leave a
    model turn running and holding subscription quota.

    Signalled even when the leader has already exited: if codex exits while
    a descendant still holds the output pipe, ``communicate()`` times out
    with ``returncode`` already set, and the old early return then left
    that descendant alive (codex r9 #4). ``os.getpgid`` on a reaped leader
    is unreliable, so the caller passes the pgid saved at spawn —
    ``start_new_session=True`` makes the child a group leader, so it equals
    its pid."""
    target = pgid if pgid is not None else proc.pid
    try:
        os.killpg(target, signal.SIGKILL)
        return
    except Exception:  # noqa: BLE001 — no such group: fall back below
        pass
    if proc.returncode is None:
        with contextlib.suppress(Exception):
            proc.kill()


def _kill_group_if_leader_alive(proc: asyncio.subprocess.Process,
                                pgid: int) -> bool:
    """Kill the process group, but only while the transport still reports
    the leader as running.

    Once the leader has been reaped its pid — and with it the group number
    — is the kernel's to reissue, and on the timeout path that window can
    be the whole remaining turn (codex r12 #7). ``returncode is None`` is
    not a proof of unreaped: the pidfd watcher reaps first and updates the
    transport in a later callback, so a race of one callback remains
    (codex r13 P3). It closes the long window, which is the one that
    mattered; the identity-checked sweep in the finally is what actually
    stops survivors."""
    if proc.returncode is not None:
        return False
    _kill_process_group(proc, pgid)
    return True


async def _watch_auth(env: dict[str, str], seen: set[str],
                      interval: float | None = None) -> None:
    """Accumulate the credentials auth.json holds WHILE the turn runs.

    A token refreshed to B during the turn, echoed by the turn, and
    refreshed again before the turn ends is invisible to a before/after
    pair of reads: nothing ever looked while B was on disk (codex r10 #2).
    Polling is the only observation point available — codex owns the file
    and announces nothing — and it is explicitly BEST EFFORT, not a
    guarantee: a value that lives only between two polls is still missed
    (codex r11 #1). What does not depend on having seen the value is
    ``looks_like_credential``, which refuses a reply on its shape.

    Cheap (a stat, and a parse only when the file changed) and silent:
    this task never raises into the caller."""
    if interval is None:
        interval = _AUTH_POLL_SECONDS
    last: tuple[int, int] | None = None
    while True:
        try:
            st = _auth_path(env).stat()
            signature = (st.st_mtime_ns, st.st_size)
        except Exception:  # noqa: BLE001 — mid-rewrite or absent
            signature = None
        if signature is not None and signature != last:
            last = signature
            try:
                seen.update(auth_secrets(env))
            except Exception:  # noqa: BLE001 — never into the caller
                pass
        await asyncio.sleep(interval)


async def run_codex_seat(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    codex_bin: str | None = None,
    web_search: bool = True,
    sandbox: str = DEFAULT_SANDBOX,
    ignore_user_config: bool = True,
    timeout_s: float | None = None,
    env: dict[str, str] | None = None,
) -> CodexSeatResult:
    """Run one codex turn and return its final message.

    Raises ``CodexSeatError`` on a missing binary, a nonzero exit, an
    empty final message, or a timeout. Never returns a blank ``text``.
    """
    binary = resolve_codex_bin(codex_bin)
    spawn_env = child_env(env)
    assert_subscription_login(spawn_env)
    # Every credential seen from before the turn to after it: the pre-turn
    # snapshot, everything the watcher below catches mid-turn, and a final
    # read. Redacting against the post-turn auth.json alone misses the
    # value the log actually echoed (codex r10 #2, r11 #1).
    seen_secrets: set[str] = set(auth_secrets(spawn_env))
    # Inherited by every descendant, so a survivor can be recognised even
    # after it has left our process group (codex r11 #4).
    marker = uuid.uuid4().hex
    spawn_env[SEAT_MARKER_ENV] = marker
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="writeup_codex_") as tmp:
        tmp_path = Path(tmp)
        work = tmp_path / "cwd"
        work.mkdir(parents=True, exist_ok=True)
        last = tmp_path / "last-message.md"
        cmd = build_codex_cmd(
            codex_bin=binary,
            model=model,
            reasoning_effort=reasoning_effort,
            workdir=work,
            last_message_path=last,
            web_search=web_search,
            sandbox=sandbox,
            ignore_user_config=ignore_user_config,
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(work),
            env=spawn_env,
            # Own process group, so every exit path can kill the whole
            # tree rather than orphan a running turn.
            start_new_session=True,
        )
        # Saved now: start_new_session makes the child a group leader, so
        # the pgid is its pid, and it stays usable after the leader has
        # been reaped (codex r9 #4).
        pgid = proc.pid
        payload = prompt if prompt.endswith("\n") else prompt + "\n"
        watcher = asyncio.get_running_loop().create_task(
            _watch_auth(spawn_env, seen_secrets))
        try:
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(payload.encode("utf-8")),
                    timeout=timeout_s,
                )
            except (TimeoutError, asyncio.TimeoutError):
                # The fast stop, not the whole of it: the survivor sweep in
                # the finally runs either way. Guarded, because
                # communicate() also times out when the leader has already
                # exited and a detached descendant holds the pipe (codex
                # r9 #4) — and by then the pid, and with it the group
                # number, may belong to somebody else, for as long as the
                # rest of the timeout (codex r12 #7). While the leader is
                # unreaped the kernel cannot reissue the number, so that
                # is exactly when killpg is safe.
                _kill_group_if_leader_alive(proc, pgid)
                raise CodexSeatError(
                    f"codex exec exceeded its {timeout_s}s wallclock bound"
                ) from None
            except asyncio.CancelledError:
                # Same reasoning, same contract as before: kill the child,
                # then let the cancellation propagate to run().
                _kill_group_if_leader_alive(proc, pgid)
                raise
        finally:
            watcher.cancel()
            # Cleanup runs on EVERY exit path, not only timeout and
            # cancellation. A codex descendant that redirected its own
            # output does not hold the pipe, so communicate() can return
            # cleanly -- on a success OR a nonzero exit -- with that
            # descendant still running the turn, past the wallclock bound
            # and on the subscription's quota (codex r10 #4). It runs
            # FIRST, before either await below: an await that fails or
            # hangs must not be able to skip it (codex r11 #8), and the
            # sooner the scan happens the shorter the window in which the
            # leader's pid could have been reissued (codex r11 #3).
            # On hosts without /proc, psutil and POSIX session/group
            # lookups provide the same per-process identity checks.
            kill_seat_survivors(pgid, marker)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=10)
            # CancelledError too: this await is the cancellation just
            # issued above coming back. Any OTHER exception is the
            # watcher's own, and is equally not a reason to fail a turn
            # whose cleanup has already run (codex r11 #8).
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await watcher
        log = (stdout or b"").decode("utf-8", errors="replace")
        text = ""
        with contextlib.suppress(OSError):
            if last.exists():
                text = last.read_text(encoding="utf-8", errors="replace")

    duration = time.monotonic() - started
    # Scrubbed here, before anything leaves this function: the caller feeds
    # ``text`` straight back into the loop as the document and as the next
    # seat's prompt, so a credential echoed by a tool result or diagnostic
    # would otherwise travel into the referee prompt and possibly into the
    # shipped document -- redacting only the on-disk transcripts left that
    # path open (codex r10 #1). ONE secret set for both strings, taken
    # after a final read, so a value first seen while scrubbing the tail
    # also protects the reply (codex r11 #1).
    secrets = auth_secrets(spawn_env, additional=seen_secrets)
    tail = scrub_credentials(log[-_LOG_TAIL_CHARS:], secrets)
    scrubbed = scrub_credentials(text, secrets)

    # Parsed BEFORE the failure checks. A turn can report valid
    # turn.completed usage and then exit nonzero or leave no final message;
    # discarding that usage let a failing seat cost real subscription
    # throughput while recording zero tokens, so max_tokens — the
    # documented subscription backstop — stopped nothing (codex r9 #2).
    usage = parse_codex_jsonl(log)
    fallback = 0
    if usage.n_turns == 0:
        m = _TOKENS_USED_RE.search(log)
        if m:
            with contextlib.suppress(ValueError):
                fallback = int(m.group(1).replace(",", ""))
    result = CodexSeatResult(
        text=scrubbed,
        usage=usage,
        returncode=proc.returncode or 0,
        duration_s=duration,
        fallback_total_tokens=fallback,
        log_tail=tail,
    )

    def _failed(message: str) -> CodexSeatError:
        """The error carries what the turn actually spent, so the caller
        can charge it before degrading."""
        err = CodexSeatError(message)
        err.partial = result
        return err

    if proc.returncode != 0:
        raise _failed(f"codex exec exited {proc.returncode}; log tail: {tail}")
    if not text.strip():
        raise _failed(
            f"codex exec produced no final message; log tail: {tail}")
    if scrubbed != text or looks_like_credential(text):
        # The reply itself carried a credential. Returning the redacted
        # version would hand the loop a document silently altered inside
        # its mathematics -- a substitution the compile gate cannot see
        # (codex r11 #5) -- so this is reported as an ordinary seat
        # failure instead, and the caller's degradation path (re-roll,
        # or keep the previous document) decides. The scrubbed transcript
        # in the run directory is the evidence; the value is never named
        # here.
        raise _failed("codex exec reply contained a credential; candidate "
                      "discarded (see the seat transcript)")
    return result


__all__ = [
    "BLOCKED_AUTH_ENVS",
    "looks_like_credential",
    "scrub_credentials",
    "SEAT_MARKER_ENV",
    "auth_secrets",
    "DEFAULT_CODEX_BIN",
    "DEFAULT_MODEL",
    "DEFAULT_REASONING_EFFORT",
    "CodexSeatError",
    "CodexSeatResult",
    "assert_subscription_login",
    "build_codex_cmd",
    "child_env",
    "redact_seat_text",
    "kill_seat_survivors",
    "resolve_codex_bin",
    "run_codex_seat",
    "seat_survivors",
]
