"""Browser-harness transport: a human runs the model call in a browser.

Substitutes an ``APICallAgent`` model call when its resolved model config
has ``api: browser``. Writes a packet (instruction.txt + attachments +
zip) into the run's ``human_inbox``, mirrors it into a fixed
``harness_packages/`` folder the operator keeps open in a file manager,
surfaces the task through the existing ``human.waiting`` dashboard flow,
and blocks (wallclock-paused) until a ``*.response.json`` appears — via
share-link fetch or manual paste, both written by the dashboard. The
response's assistant text (plus any uploaded files, synthesized as
fenced ``file path=...`` blocks) then goes through the agent's normal
``parse_output``.

The task stem is derived from the agent's resume-cache key, not the
per-call workdir: if the run crashes mid-wait, the resumed node lands on
the same task/response path, so an answer submitted in the meantime is
picked up immediately (and an existing response file is deliberately
consumed rather than cleared).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from proofstack.agents.human_agent import wait_for_response_file
from proofstack.events import new_call_id

if TYPE_CHECKING:
    from proofstack.kinds.api_call import APICallAgent

DEFAULT_HUMAN_TIMEOUT_S = 7 * 24 * 3600.0
INSTRUCTION_PREVIEW_CHARS = 800
OPERATOR_COMMENTS_FILE = "harness/operator_comments.jsonl"


def harness_packages_root() -> Path:
    env = os.environ.get("PROOFCOUNCIL_HARNESS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "harness_packages"


async def run_browser_call(
    agent: "APICallAgent", inp: BaseModel, browser_cfg: dict[str, Any]
) -> BaseModel:
    for scope, kind, used, limit in agent.tracker.check():
        await agent.events.emit(
            "budget.warn",
            {"scope": scope, "kind": kind, "used": used, "limit": limit},
        )

    # Agent names come from workflow YAML, but they end up in filesystem
    # paths that get recursively deleted — keep the stem strictly safe.
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", agent.name)
    stem = f"{safe_name}__{agent._cache_key(inp)[:12]}"

    instruction, attachments = agent.render_harness_packet(inp)
    # The token binds a share back to this exact task: it names the
    # instruction file (checked against the share's uploaded-file names) and
    # heads the instruction text (checked when the operator pastes instead
    # of uploading).
    instruction = f"[ProofCouncil task {stem}]\n\n" + instruction
    addendum = str(browser_cfg.get("instruction_addendum") or "").strip()
    if addendum:
        instruction = (
            instruction.rstrip()
            + "\n\n### Browser-harness output contract ###\n"
            + addendum
            + "\n"
        )

    instruction_file = f"instruction_{stem}.txt"
    inbox = agent.ctx.root_workdir / "human_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    packet_dir = inbox / f"{stem}.packet"
    if packet_dir.exists():
        shutil.rmtree(packet_dir)
    packet_dir.mkdir(parents=True)
    (packet_dir / instruction_file).write_text(instruction, encoding="utf-8")
    for name, body in attachments.items():
        target = packet_dir / Path(str(name)).name
        if isinstance(body, bytes):
            target.write_bytes(body)
        else:
            target.write_text(str(body), encoding="utf-8")
    zip_path = inbox / f"{stem}.packet.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(packet_dir.iterdir()):
            zf.write(f, arcname=f.name)

    copy_dir: Path | None = None
    try:
        copy_dir = harness_packages_root() / agent.ctx.run_id / stem
        if copy_dir.exists():
            shutil.rmtree(copy_dir)
        copy_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(packet_dir, copy_dir)
    except OSError:
        copy_dir = None

    response_path = inbox / f"{stem}.response.json"
    display_model = str(
        browser_cfg.get("display_model") or browser_cfg.get("model") or "browser model"
    )
    preview = instruction[:INSTRUCTION_PREVIEW_CHARS]
    if len(instruction) > INSTRUCTION_PREVIEW_CHARS:
        preview += "\n… (full instruction in the packet's instruction.txt)"
    task = {
        "agent": agent.name,
        "run_id": agent.ctx.run_id,
        "type": "browser_call",
        "task_token": stem,
        "instruction_file": instruction_file,
        "prompt": preview,
        "response_path": str(response_path),
        "packet_dir": str(packet_dir),
        "packet_zip": str(zip_path),
        "harness_copy_dir": str(copy_dir) if copy_dir else None,
        "files": [
            {"name": f.name, "bytes": f.stat().st_size}
            for f in sorted(packet_dir.iterdir())
        ],
        "browser": {
            "model": browser_cfg.get("model"),
            "display_model": display_model,
            "service": browser_cfg.get("service") or "",
            "settings_hint": browser_cfg.get("settings_hint") or "",
            "chat_url": browser_cfg.get("chat_url") or "",
            "expected_model_slugs": list(browser_cfg.get("expected_model_slugs") or []),
            "expected_efforts": list(browser_cfg.get("expected_efforts") or []),
            "expected_model_variants": list(
                browser_cfg.get("expected_model_variants") or []
            ),
        },
        "expected": agent.harness_expectations(inp),
        "output_fields": {},
    }
    task_path = inbox / f"{stem}.task.json"
    task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        (agent.workdir / "messages.json").write_text(
            json.dumps(
                [{"role": "user", "content": instruction}],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass

    waiting_payload = {
        "type": "browser_call",
        "task_path": str(task_path),
        "response_path": str(response_path),
        "packet_dir": str(packet_dir),
    }
    await agent.events.emit("human.waiting", waiting_payload)
    call_id = new_call_id()
    await agent.events.emit(
        "model.call.start",
        {"model": display_model, "via": "browser_harness"},
        call_id=call_id,
    )

    timeout_s = float(
        agent.component_config.get("human_timeout_s") or DEFAULT_HUMAN_TIMEOUT_S
    )
    start = time.monotonic()
    while True:
        response = await wait_for_response_file(
            response_path,
            events=agent.events,
            tracker=agent.tracker,
            timeout_s=timeout_s,
        )
        elapsed = time.monotonic() - start
        if response is None:
            await agent.events.emit(
                "human.timeout", {"response_path": str(response_path)}
            )
            raise RuntimeError(
                f"browser call {stem} timed out after {timeout_s:.0f}s without a response"
            )

        await agent.events.emit(
            "human.submitted", {"response_path": str(response_path)}
        )

        try:
            raw_text = str(response.get("assistant_text") or "")
            uploads = response.get("uploaded_files") or {}
            for name in sorted(uploads):
                body = _read_upload_text(agent.ctx.root_workdir, str(uploads[name]))
                if body is None:
                    await agent.events.emit(
                        "harness.upload_skipped",
                        {"name": str(name), "reason": "unreadable or binary"},
                    )
                    continue
                if "\n```" in body or body.startswith("```"):
                    await agent.events.emit(
                        "harness.upload_fence_conflict", {"name": str(name)}
                    )
                raw_text += f"\n\n```file path={Path(str(name)).name}\n{body}\n```"

            comments = str(response.get("operator_comments") or "").strip()
            try:
                (agent.workdir / "raw_response.txt").write_text(
                    raw_text, encoding="utf-8"
                )
                (agent.workdir / "share.json").write_text(
                    json.dumps(
                        {
                            "transport": response.get("transport"),
                            "share_url": response.get("share_url"),
                            "model_slug": response.get("model_slug"),
                            "effort": response.get("effort"),
                            "operator_comments": comments,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass

            out = agent.parse_output(raw_text, inp)
        except Exception as e:
            # A response that cannot be processed must not be consumed again
            # (or kill the run): move it aside, surface the error, and go
            # back to waiting so the operator can submit a corrected one.
            rejected = _move_response_aside(response_path)
            error_text = f"{type(e).__name__}: {e}"
            await agent.events.emit(
                "harness.response_rejected",
                {
                    "response_path": str(response_path),
                    "moved_to": str(rejected) if rejected else None,
                    "error": error_text,
                },
            )
            await agent.events.emit(
                "human.waiting", {**waiting_payload, "rejected_error": error_text}
            )
            continue

        # Steering comments count only once their response actually parsed:
        # a quarantined submission must not leave its comment queued.
        if comments:
            try:
                _append_operator_comment(
                    agent.ctx.root_workdir, agent.name, stem, comments
                )
            except OSError as e:
                await agent.events.emit(
                    "harness.comment_append_failed",
                    {"error": f"{type(e).__name__}: {e}"},
                )
        break

    if copy_dir is not None:
        shutil.rmtree(copy_dir, ignore_errors=True)
        try:
            copy_dir.parent.rmdir()
        except OSError:
            pass

    await agent.events.emit(
        "model.call",
        {
            "model": response.get("model_slug") or task["browser"]["model"] or display_model,
            "in_tokens": 0,
            "out_tokens": 0,
            "reasoning_tokens": 0,
            "cost_usd": 0.0,
            "duration_s": elapsed,
            "via": "browser_harness",
            "transport": response.get("transport") or "manual",
            "share_url": response.get("share_url"),
        },
        call_id=call_id,
    )
    if not raw_text.strip():
        await agent.events.emit(
            "model.empty_response",
            {
                "type": "EmptyResponse",
                "msg": f"browser call {stem} returned an empty response",
            },
            call_id=call_id,
        )
    return out


def _append_operator_comment(run_dir: Path, agent_name: str, stem: str, text: str) -> None:
    """Append a comment, deduplicated by (stem, text): a crash between
    consuming a response and persisting the cache re-appends the same
    comment on resume otherwise."""
    comments_path = run_dir / OPERATOR_COMMENTS_FILE
    comments_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for line in comments_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("stem") == stem and entry.get("text") == text:
                return
    except OSError:
        pass
    with comments_path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"agent": agent_name, "stem": stem, "text": text, "ts": time.time()},
                ensure_ascii=False,
            )
            + "\n"
        )


def _move_response_aside(response_path: Path) -> Path | None:
    for i in range(1, 100):
        rejected = response_path.with_name(
            response_path.name.replace(".response.json", f".response.rejected-{i}.json")
        )
        if not rejected.exists():
            try:
                response_path.rename(rejected)
                return rejected
            except OSError:
                return None
    return None


def _read_upload_text(run_dir: Path, rel: str) -> str | None:
    try:
        path = (run_dir / rel).resolve()
        path.relative_to(run_dir.resolve())
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def peek_operator_comments(run_dir: Path) -> tuple[str, int]:
    """Unconsumed operator comments + the offset to commit once used.

    Committing only after the consuming call succeeds keeps the drain
    idempotent across crashes: a resumed Author round re-reads the same
    comments, so its inputs (and resume-cache key) are unchanged.
    """
    comments_path = run_dir / OPERATOR_COMMENTS_FILE
    if not comments_path.exists():
        return "", 0
    offset_path = comments_path.with_suffix(".consumed")
    try:
        offset = int(offset_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        offset = 0
    try:
        data = comments_path.read_bytes()
    except OSError:
        return "", offset
    if offset >= len(data):
        return "", offset
    texts: list[str] = []
    for line in data[offset:].decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = str(entry.get("text") or "").strip()
        if text:
            texts.append(f"[{entry.get('agent') or 'operator'}] {text}")
    return "\n".join(texts), len(data)


def commit_operator_comments(run_dir: Path, offset: int) -> None:
    if offset <= 0:
        return
    comments_path = run_dir / OPERATOR_COMMENTS_FILE
    try:
        comments_path.parent.mkdir(parents=True, exist_ok=True)
        comments_path.with_suffix(".consumed").write_text(str(offset), encoding="utf-8")
    except OSError:
        pass


__all__ = [
    "run_browser_call",
    "peek_operator_comments",
    "commit_operator_comments",
    "harness_packages_root",
]
