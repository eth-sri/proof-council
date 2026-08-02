"""Fetch a public ChatGPT share link and reconstruct the consultation.

Uses the undocumented ``chatgpt.com/backend-api/share/<uuid>`` JSON
endpoint; it may break without notice, so every consumer must keep the
manual-paste path as a first-class fallback. Sandbox-generated files
are retrievable statelessly through the ``file_from_message`` resolver
(``resolve_shared_file`` + ``download_shared_file``); when that fails,
the operator downloads them from the chat by hand.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SHARE_ID_PATTERN = re.compile(
    r"(?:https?://chatgpt\.com/share/)?"
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)

# Cloudflare serves a challenge to non-browser UAs (2026-07-23).
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

_SANDBOX_LINK_RE = re.compile(r"sandbox:(/[^\s\"'<>]+)")
_FILE_FENCE_RE = re.compile(r"```file\s+path=(?P<path>[A-Za-z0-9._/-]+)")


def _sandbox_paths(text: str) -> list[str]:
    """Sandbox link targets, tolerating parentheses in filenames.

    Links usually appear as markdown ``[x](sandbox:/mnt/data/f.pdf)``, so a
    greedy match swallows the closing paren; trim trailing parens only while
    they are unbalanced, so ``report_(final).pdf`` survives intact."""
    out: list[str] = []
    for m in _SANDBOX_LINK_RE.finditer(text):
        path = m.group(1)
        while path.endswith(")") and path.count("(") < path.count(")"):
            path = path[:-1]
        out.append(path.rstrip(".,;:"))
    return out


class ShareFetchError(Exception):
    """Retrieval or validation failure with an operator-friendly message."""


def share_id(url_or_id: str) -> str | None:
    m = SHARE_ID_PATTERN.search(str(url_or_id).strip())
    return m.group(1) if m else None


def fetch_share(url_or_id: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
    sid = share_id(url_or_id)
    if sid is None:
        raise ShareFetchError(
            f"not a ChatGPT share URL or UUID: {str(url_or_id)[:200]!r}"
        )
    req = urllib.request.Request(
        f"https://chatgpt.com/backend-api/share/{sid}",
        headers=_share_headers(sid),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ShareFetchError(
                "share link not found (deleted, never published, or mistyped)"
            ) from e
        if e.code == 403:
            raise ShareFetchError(
                "the share endpoint refused the request (Cloudflare challenge "
                "or endpoint change) — use the manual paste fallback"
            ) from e
        raise ShareFetchError(f"share fetch failed with HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise ShareFetchError(f"network error fetching share link: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise ShareFetchError(
            "share endpoint returned non-JSON (endpoint change?) — use the "
            "manual paste fallback"
        ) from e
    except (TimeoutError, OSError, UnicodeDecodeError) as e:
        raise ShareFetchError(f"share fetch failed: {type(e).__name__}: {e}") from e
    if not isinstance(data, dict):
        raise ShareFetchError(
            "share endpoint returned an unexpected JSON shape (endpoint "
            "change?) — use the manual paste fallback"
        )
    return data


# Signed download URLs are served from OpenAI's user-content CDN only.
_ALLOWED_DOWNLOAD_HOST_SUFFIX = ".oaiusercontent.com"
MAX_SHARED_FILE_BYTES = 50 * 1024 * 1024


def _share_headers(sid: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "User-Agent": _BROWSER_UA,
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://chatgpt.com/share/{sid}",
    }


def resolve_shared_file(
    sid: str, message_id: str, sandbox_path: str, *, timeout_s: float = 30.0
) -> dict[str, Any]:
    """Resolve a ``sandbox:/mnt/data/...`` link to a temporary signed URL.

    Uses the stateless ``file_from_message`` endpoint (no auth needed for
    public shares). Undocumented — failures must fall back to asking the
    operator to download the file by hand.
    """
    name = sandbox_path.rsplit("/", 1)[-1]
    query = urllib.parse.urlencode({"file_path": sandbox_path})
    req = urllib.request.Request(
        f"https://chatgpt.com/backend-api/share/{sid}/file_from_message/"
        f"{message_id}?{query}",
        headers=_share_headers(sid),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        raise ShareFetchError(f"file resolver failed for {name!r}: HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ShareFetchError(
            f"file resolver failed for {name!r}: {type(e).__name__}"
        ) from e
    if not isinstance(payload, dict) or not payload.get("download_url"):
        raise ShareFetchError(f"file resolver returned no download URL for {name!r}")
    return payload


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise ShareFetchError("signed download URL redirected unexpectedly")


def download_shared_file(
    info: dict[str, Any],
    dest_dir: Path,
    *,
    fallback_name: str = "file",
    max_bytes: int = MAX_SHARED_FILE_BYTES,
    timeout_s: float = 60.0,
) -> Path:
    """Download a resolved shared file. The signed URL is treated as a
    secret: it is never logged, persisted, or included in error text."""
    url = str(info.get("download_url") or "")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc.endswith(
        _ALLOWED_DOWNLOAD_HOST_SUFFIX
    ):
        raise ShareFetchError("signed download URL failed the host allowlist")
    size = info.get("file_size_bytes")
    if isinstance(size, int) and size > max_bytes:
        raise ShareFetchError(
            f"generated file is too large to auto-download ({size} bytes)"
        )
    # The resolver reports file_name as a full /mnt/data/... path.
    raw_name = str(info.get("file_name") or fallback_name).rsplit("/", 1)[-1]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name).lstrip(".")
    if not name:
        name = "file"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    # Distinct sandbox paths can share a basename (/a/report.tex vs
    # /b/report.tex) — never silently overwrite an earlier download.
    counter = 1
    while dest.exists():
        counter += 1
        dest = dest_dir / f"{Path(name).stem}-{counter}{Path(name).suffix}"
    opener = urllib.request.build_opener(_RefuseRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
    try:
        with opener.open(req, timeout=timeout_s) as resp, dest.open("wb") as out:
            written = 0
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise ShareFetchError(
                        f"generated file {name!r} exceeded the download size limit"
                    )
                out.write(chunk)
    except ShareFetchError:
        dest.unlink(missing_ok=True)
        raise
    except urllib.error.HTTPError as e:
        dest.unlink(missing_ok=True)
        raise ShareFetchError(
            f"download of {name!r} failed: HTTP {e.code} (signed links expire "
            "quickly — re-fetch the share)"
        ) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        dest.unlink(missing_ok=True)
        raise ShareFetchError(
            f"download of {name!r} failed: {type(e).__name__}"
        ) from e
    return dest


def _conversation_nodes(data: dict[str, Any]) -> list[Any]:
    linear = data.get("linear_conversation")
    if isinstance(linear, list) and linear:
        return linear
    mapping = data.get("mapping")
    if not isinstance(mapping, dict):
        return []
    # Walk the active branch backwards from current_node: raw mapping order
    # is unspecified and may include abandoned/regenerated branches.
    cur = data.get("current_node")
    chain: list[Any] = []
    seen: set[str] = set()
    while isinstance(cur, str) and cur in mapping and cur not in seen:
        seen.add(cur)
        node = mapping.get(cur)
        if not isinstance(node, dict):
            break
        chain.append(node)
        cur = node.get("parent")
    if chain:
        return list(reversed(chain))
    return list(mapping.values())


def visible_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in _conversation_nodes(data):
        if not isinstance(node, dict):
            continue
        msg = node.get("message")
        if not isinstance(msg, dict):
            continue
        meta = msg.get("metadata")
        meta = meta if isinstance(meta, dict) else {}
        author = msg.get("author")
        role = author.get("role") if isinstance(author, dict) else None
        if role not in ("user", "assistant"):
            continue
        if meta.get("is_visually_hidden_from_conversation"):
            continue
        content = msg.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        text = "".join(p for p in (parts or []) if isinstance(p, str))
        if not text.strip() and not meta.get("attachments"):
            continue
        out.append(
            {
                "role": role,
                "id": msg.get("id"),
                "text": text,
                "model": meta.get("resolved_model_slug") or meta.get("model_slug"),
                "effort": meta.get("thinking_effort"),
                "attachments": list(meta.get("attachments") or []),
                "content_references": list(meta.get("content_references") or []),
            }
        )
    return out


def extract_result(data: dict[str, Any]) -> dict[str, Any]:
    msgs = visible_messages(data)
    last_user = -1
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            last_user = i
    answer_msgs = [m for m in msgs[last_user + 1 :] if m["role"] == "assistant"]
    assistant_text = "\n\n".join(
        m["text"] for m in answer_msgs if m["text"].strip()
    )

    # Slug and effort must come from the SAME, NEWEST message — walking
    # back to an older turn could verify the final answer with a progress
    # turn's metadata. A metadata-less final answer keeps slug None, which
    # triggers the cannot-verify warning downstream.
    model_slug = None
    effort = None
    source_msgs = answer_msgs or [m for m in msgs if m["role"] == "assistant"]
    if source_msgs:
        model_slug = source_msgs[-1]["model"]
        effort = source_msgs[-1]["effort"]

    # Only the answer turns (after the operator's final message) can carry
    # model-generated files worth downloading; earlier turns and citation
    # references to *uploaded* sources (e.g. a user-provided PDF the model
    # cites) must not be flagged as generated artifacts.
    sandbox_artifacts: list[str] = []
    for m in answer_msgs:
        for att in m["attachments"]:
            name = att.get("name") if isinstance(att, dict) else None
            sandbox_artifacts.append(str(name or att))
        for ref in m["content_references"]:
            if not isinstance(ref, dict):
                continue
            ref_type = str(ref.get("type") or "")
            if "file" in ref_type and "citation" not in ref_type and "cite" not in ref_type:
                sandbox_artifacts.append(str(ref.get("name") or ref_type))
    # Structured generated-file records: sandbox links paired with their own
    # message id, which is exactly what the stateless file resolver needs.
    # Keyed by path with later messages overwriting earlier ones: when the
    # model regenerates a file, the LATEST message's copy is the one to
    # download.
    by_path: dict[str, dict[str, str]] = {}
    for m in answer_msgs:
        if not m.get("id"):
            continue
        for path in _sandbox_paths(m["text"]):
            by_path[path] = {
                "message_id": str(m["id"]),
                "sandbox_path": path,
                "name": path.rsplit("/", 1)[-1],
            }
    generated_files = list(by_path.values())

    sandbox_artifacts.extend("sandbox:" + p for p in _sandbox_paths(assistant_text))
    seen: set[str] = set()
    sandbox_artifacts = [
        a for a in sandbox_artifacts if not (a in seen or seen.add(a))
    ]

    sources = _reference_sources(answer_msgs)
    if sources:
        assistant_text += (
            "\n\n### Sources cited (from the share's reference metadata) ###\n"
            + "\n".join(f"- {s}" for s in sources)
        )

    return {
        "title": data.get("title") or "Shared conversation",
        "assistant_text": assistant_text,
        "model_slug": model_slug,
        "effort": effort,
        "default_model_slug": data.get("default_model_slug"),
        "models_seen": sorted(
            {
                (m["model"], m["effort"])
                for m in msgs
                if m["role"] == "assistant" and m["model"]
            },
            key=lambda t: tuple(str(x) for x in t),
        ),
        "answer_models": sorted(
            {str(m["model"]) for m in answer_msgs if m["model"]}
        ),
        "sandbox_artifacts": sandbox_artifacts,
        "generated_files": generated_files,
        "sources": sources,
        "messages": msgs,
    }


def _reference_sources(msgs: list[dict[str, Any]]) -> list[str]:
    """Readable title+URL list from ChatGPT ``content_references`` metadata.

    The assistant text keeps only opaque citation markers; the actual
    titles/URLs live in the reference objects (top-level or nested under
    ``items``) and would otherwise be discarded.

    The list is appended to text that downstream workflow parsers treat as
    model output, so every line is sanitized: no backticks, angle brackets,
    or newlines survive — citation metadata must never be able to form a
    fenced ``file`` block, ``<council>``/``<compute_agent>``/``<ready>``
    markup, or any other control sequence."""
    sources: list[str] = []
    seen: set[str] = set()

    def _add(title: Any, url: Any) -> None:
        url = re.sub(r"[\s<>`]+", "", str(url or ""))
        if not url or not url.lower().startswith(("http://", "https://")) or url in seen:
            return
        seen.add(url)
        title = re.sub(r"[<>`]+", " ", str(title or ""))
        title = re.sub(r"\s+", " ", title).strip()
        sources.append(f"{title} — {url}" if title and title != url else url)

    for m in msgs:
        for ref in m.get("content_references") or []:
            if not isinstance(ref, dict):
                continue
            _add(ref.get("title"), ref.get("url"))
            for item in ref.get("items") or []:
                if isinstance(item, dict):
                    _add(item.get("title"), item.get("url"))
    return sources


_TOKEN_MARKER_RE = re.compile(r"\[ProofCouncil task ([A-Za-z0-9._-]+)\]")
_INSTRUCTION_FILE_TOKEN_RE = re.compile(r"^instruction_([A-Za-z0-9._-]+)\.txt$")
# Typed references ("Follow instruction_<token>.txt.") and the packet zip
# itself also bind the message to a task. Greedy with a boundary lookahead
# so a token whose agent-name part contains ".txt" is not truncated; a
# trailing sentence period is still allowed.
_INSTRUCTION_MENTION_RE = re.compile(
    r"instruction_([A-Za-z0-9._-]+)\.txt(?![A-Za-z0-9_-])"
)
_PACKET_ZIP_TOKEN_RE = re.compile(r"^([A-Za-z0-9._-]+)\.packet\.zip$")


def latest_task_tokens(result: dict[str, Any]) -> set[str]:
    """Task tokens named by the *most recent* user message that names any.

    A conversation reused for several harness tasks contains several
    markers; only the newest binding counts, so submitting the share
    against an *older* task in the same chat is refused even though that
    task's token still appears somewhere in the history. Steering messages
    that name no task never reset the binding. Tokens are recognized from
    uploaded ``instruction_<token>.txt`` / ``<token>.packet.zip`` names,
    the ``[ProofCouncil task <token>]`` marker, and typed
    ``instruction_<token>.txt`` mentions."""
    latest: set[str] = set()
    for m in result.get("messages") or []:
        if m.get("role") != "user":
            continue
        found: set[str] = set()
        for att in m.get("attachments") or []:
            name = str(
                (att.get("name") if isinstance(att, dict) else att) or ""
            )
            mt = _INSTRUCTION_FILE_TOKEN_RE.match(name)
            if mt:
                found.add(mt.group(1))
            mz = _PACKET_ZIP_TOKEN_RE.match(name)
            if mz:
                found.add(mz.group(1))
        text = str(m.get("text") or "")
        found.update(_TOKEN_MARKER_RE.findall(text))
        found.update(_INSTRUCTION_MENTION_RE.findall(text))
        if found:
            latest = found
    return latest


def _slug_matches(slug: str, expected: str) -> bool:
    """Delimiter-aware: ``gpt-5-6-sol-pro-high`` matches ``gpt-5-6-sol-pro``,
    but a short slug never matches a longer expected one (``gpt-5`` must not
    pass for ``gpt-5-6-sol-pro``)."""
    slug, expected = slug.lower(), expected.lower()
    return slug == expected or slug.startswith(expected + "-")


def validate_result(
    result: dict[str, Any],
    task: dict[str, Any],
    *,
    provided_files: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """Operator-facing warnings; an empty answer is a hard error.

    ``provided_files``: filenames already secured out-of-band (e.g.
    auto-downloaded mergeable share files) — they satisfy fenced/required
    file expectations just like inline fenced blocks."""
    text = str(result.get("assistant_text") or "")
    if not text.strip():
        raise ShareFetchError(
            "the shared conversation contains no assistant answer after your "
            "message — did you share before the model finished?"
        )
    warnings: list[str] = []

    browser = task.get("browser") or {}
    slug = result.get("model_slug") or result.get("default_model_slug")
    effort = result.get("effort")
    variants = [v for v in (browser.get("expected_model_variants") or []) if isinstance(v, dict)]
    if variants:
        # Slug/effort form valid *pairs* — validating them independently
        # would accept e.g. a Pro-capable slug at a non-Pro effort.
        slugs = [str(v.get("slug") or "") for v in variants]
        if not slug:
            warnings.append(
                "the share carried no model slug, so the model used cannot be "
                f"verified (expected one of {slugs})"
            )
        else:
            # Exact slug match wins; among prefix matches prefer the most
            # specific, so overlapping variants are order-independent.
            candidates = [
                v for v in variants if _slug_matches(str(slug), str(v.get("slug") or ""))
            ]
            matched = next(
                (v for v in candidates if str(v.get("slug") or "").lower() == str(slug).lower()),
                max(candidates, key=lambda v: len(str(v.get("slug") or "")), default=None),
            )
            if matched is None:
                warnings.append(
                    f"the conversation ran on model {slug!r}, expected one of "
                    f"{slugs} — check the model picker"
                )
            else:
                efforts = [str(e).lower() for e in (matched.get("efforts") or []) if e]
                if efforts and not effort:
                    warnings.append(
                        f"the share carried no reasoning-effort marker for model "
                        f"{slug!r} (expected one of {efforts})"
                    )
                elif efforts and str(effort).lower() not in efforts:
                    warnings.append(
                        f"the answer used reasoning effort {effort!r}, but model "
                        f"{slug!r} is expected at one of {efforts} — check the "
                        "reasoning-mode picker"
                    )
    else:
        expected_slugs = [s for s in (browser.get("expected_model_slugs") or []) if s]
        if expected_slugs:
            if not slug:
                warnings.append(
                    "the share carried no model slug, so the model used cannot be "
                    f"verified (expected one of {expected_slugs})"
                )
            elif not any(_slug_matches(str(slug), str(s)) for s in expected_slugs):
                warnings.append(
                    f"the conversation ran on model {slug!r}, expected one of "
                    f"{expected_slugs} — check the model picker"
                )
        expected_efforts = [
            str(e).lower() for e in (browser.get("expected_efforts") or []) if e
        ]
        if expected_efforts:
            if not effort:
                warnings.append(
                    "the share carried no reasoning-effort marker, so the reasoning "
                    f"mode cannot be verified (expected one of {expected_efforts})"
                )
            elif str(effort).lower() not in expected_efforts:
                warnings.append(
                    f"the answer used reasoning effort {effort!r}, expected one of "
                    f"{expected_efforts} — check the reasoning-mode picker"
                )

    token = str(task.get("task_token") or "")
    if token:
        tokens = latest_task_tokens(result)
        if not tokens:
            instruction_file = str(
                task.get("instruction_file") or f"instruction_{token}.txt"
            )
            warnings.append(
                "could not verify this share belongs to this task: no ProofCouncil "
                f"task marker found ({instruction_file} was not uploaded and no "
                "[ProofCouncil task ...] line appears in your messages) — "
                "double-check you pasted the link into the right card"
            )
        elif len(tokens) > 1:
            # Binding must be unambiguous: a message naming several tasks
            # could otherwise satisfy every one of their cards.
            warnings.append(
                "the conversation's most recent message names several "
                f"ProofCouncil tasks ({sorted(tokens)}) — the binding is "
                "ambiguous, so this answer cannot be verified as belonging "
                "to this card"
            )
        elif token not in tokens:
            warnings.append(
                "this conversation's most recent ProofCouncil task is "
                f"{sorted(tokens)}, not this card's ({token}) — the answer you are "
                "submitting belongs to that newer task"
            )

    answer_models = [str(m) for m in (result.get("answer_models") or [])]
    if len(answer_models) > 1:
        warnings.append(
            f"the answer mixes turns from multiple models ({answer_models}) — "
            "was the model switched mid-conversation?"
        )

    expected = task.get("expected") or {}
    present = set(_FILE_FENCE_RE.findall(text)) | {str(f) for f in provided_files}
    fenced = [str(f) for f in (expected.get("fenced_files") or [])]
    if fenced and not present.intersection(fenced):
        warnings.append(
            "no fenced ``file path=...`` blocks found for "
            f"{fenced} — if the model offered file downloads instead, "
            "download them from the chat and add them via the manual upload"
        )
    for required in expected.get("required_files") or []:
        if str(required) not in present:
            warnings.append(
                f"required file {str(required)!r} is missing from the answer — "
                "without it this round produces no usable output; ask the model "
                "to print it inline, or download it and add it via the upload"
            )
    for marker in expected.get("markers") or []:
        if str(marker) not in text:
            warnings.append(f"expected marker {str(marker)!r} not found in the answer")

    if result.get("sandbox_artifacts"):
        listed = ", ".join(str(a) for a in result["sandbox_artifacts"][:8])
        warnings.append(
            "the model created files in its sandbox that could not be "
            f"auto-downloaded ({listed}) — download any you need from the "
            "chat and add them via the manual upload"
        )
    return warnings


__all__ = [
    "MAX_SHARED_FILE_BYTES",
    "SHARE_ID_PATTERN",
    "ShareFetchError",
    "share_id",
    "fetch_share",
    "resolve_shared_file",
    "download_shared_file",
    "visible_messages",
    "extract_result",
    "validate_result",
    "latest_task_tokens",
]
