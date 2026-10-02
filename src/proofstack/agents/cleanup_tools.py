"""Private, invocation-scoped MCP transport for the cleanup supervisor."""
from __future__ import annotations

import asyncio
import hmac
import secrets
import socket
from contextlib import asynccontextmanager, nullcontext
from typing import Literal

from mcp.server.fastmcp import FastMCP
import uvicorn

# Claude Code 2.1.280 moves an MCP result over 50,000 characters or 25,000
# tokens into a file the editor cannot open; Fable spends about one token per
# two characters of LaTeX, so a default chunk must stay far below both.
READ_CHUNK_CHARS = 30_000
MAX_READ_CHARS = 100_000
# A declared result size lifts both of Claude Code's caps, so a larger
# explicit read limit still arrives whole.
_LARGE_RESULT = {"anthropic/maxResultSizeChars": 200_000}


def _inline_report(result: dict) -> dict:
    report = result["report"]
    if len(report) <= READ_CHUNK_CHARS:
        return result
    return {**result, "report": report[:READ_CHUNK_CHARS], "report_length": len(report),
            "note": f"PARTIAL: report cut at {READ_CHUNK_CHARS} of {len(report)} characters; "
                    f"read {result['path']} with offset={READ_CHUNK_CHARS} for the rest."}


class _Bearer:
    def __init__(self, app, token):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            if not hmac.compare_digest(headers.get(b"authorization", b""), self.token):
                await send({"type": "http.response.start", "status": 401, "headers": []})
                await send({"type": "http.response.body", "body": b"Unauthorized"})
                return
        await self.app(scope, receive, send)


class _Server(uvicorn.Server):
    # The enclosing workflow owns SIGTERM/SIGINT, not a per-editor server.
    def capture_signals(self):
        return nullcontext()


@asynccontextmanager
async def cleanup_tools(*, compile_document, codex_review, read_file, write_file, list_files,
                        linear_read=None, call_timeout_s=3600, review_start=None, review_status=None, review_read=None,
                        cancel_reviews=None, progress=None):
    """Keep tool calls in the workflow process and cancel them before shutdown.

    Paid Codex jobs are owned by CleanupReviews; HTTP requests only start or
    inspect them. Cancel them before draining HTTP helpers; the enclosing
    invocation joins their shutdown and accounting after this server.
    """
    mcp = FastMCP("cleanup", stateless_http=True, json_response=True)
    pending: set[asyncio.Task] = set()
    closing = False

    async def tracked(fn, *args):
        if closing:
            raise RuntimeError("cleanup invocation has ended")
        task = asyncio.current_task()
        pending.add(task)
        try:
            result = await fn(*args)
            if progress is not None and isinstance(result, dict):
                result = {**result, "cleanup_control": progress()}
            return result
        finally:
            pending.discard(task)

    @mcp.tool()
    async def compile() -> dict:
        """Compile current answer.tex and report page count and diagnostics."""
        return await tracked(compile_document)

    @mcp.tool()
    async def review(task: str, purpose: Literal["attribution", "general"] = "general") -> dict:
        """Start a paid Codex review of a frozen snapshot; it cannot edit files.

        Use purpose=attribution for the required citation/prior-work review.
        Returns a review ID promptly, not the report. Use review_status to wait
        and retrieve it. Repeating a request reuses running/completed work;
        failed/cancelled jobs can be retried subject to the remaining budget.
        An outstanding review must be retrieved before another can start.
        """
        if review_start is not None:
            return await tracked(review_start, task, purpose)
        return _inline_report(await tracked(codex_review, task))

    if review_status is not None:
        @mcp.tool(name="review_status", meta=_LARGE_RESULT)
        async def review_result(review_id: str | None = None, wait_seconds: float = 0) -> dict:
            """List review jobs, or retrieve one by ID without paying for new inference.

            Omit review_id to recover a lost ID after a timeout. With an ID,
            wait_seconds (0-240) waits for completion without cancelling
            the worker. Use 240 while running. Completed reports remain
            readable when the reviewer budget is exhausted. Read any truncated
            report's remaining chunks using read before editing/completing.
            """
            return await tracked(review_status, review_id, wait_seconds)

    if linear_read is not None:
        @mcp.tool(name="linear_read")
        async def prefix_read() -> dict:
            """Have a cheap model read current answer.tex strictly from the start and report define-before-use.

            It reads one passage at a time, seeing only the preceding text, and
            lists every symbol, term or result used before the manuscript
            introduces it. Fast and cheap; the report is advisory and is also
            saved under reviews/.
            """
            return _inline_report(await tracked(linear_read))

    @mcp.tool(meta=_LARGE_RESULT)
    async def read(path: str, offset: int = 0, limit: int = READ_CHUNK_CHARS) -> dict:
        """Read an editorial input, answer.tex, feedback.md or a saved review.

        Returns up to `limit` characters (default 30000, at most 100000) from
        character `offset`, with the file's total `length`. Long files arrive in
        chunks: while `next_offset` is not null, call read again with
        offset=next_offset until you have the whole file.
        """
        if offset < 0 or limit < 1:
            raise ValueError("offset must be >= 0 and limit >= 1")

        async def chunk():
            result = await read_file(path)
            text = result["content"]
            end = min(len(text), offset + min(limit, MAX_READ_CHARS))
            more = end if end < len(text) else None
            out = {**result, "content": text[offset:end], "offset": offset,
                   "length": len(text), "next_offset": more}
            if more is not None:
                out["note"] = (f"PARTIAL: characters {offset}-{end} of {len(text)}; "
                               f"call read with offset={end} for the rest.")
            if review_read is not None:
                review_read(path, offset, end)
            return out
        return await tracked(chunk)

    @mcp.tool()
    async def write(path: str, content: str) -> dict:
        """Write answer.tex, feedback.md or completion.json. Other paths are read-only."""
        return await tracked(write_file, path, content)

    @mcp.tool()
    async def edit(path: str, old: str, new: str) -> dict:
        """Replace one exact occurrence in an editable file, preserving the rest."""
        async def replace():
            current = await read_file(path)
            text = current["content"]
            if not old or text.count(old) != 1:
                raise ValueError("edit must match exactly one nonempty passage")
            return await write_file(path, text.replace(old, new, 1))
        return await tracked(replace)

    @mcp.tool()
    async def files() -> list[str]:
        """List the available manuscript, context and review files."""
        return await tracked(list_files)

    token = secrets.token_urlsafe(32)
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
    except BaseException:
        sock.close()
        raise
    app = _Bearer(mcp.streamable_http_app(), f"Bearer {token}".encode())
    server = _Server(uvicorn.Config(app, log_level="error", access_log=False,
                                     timeout_graceful_shutdown=5))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("cleanup MCP server failed to start")
                await asyncio.sleep(0.01)
        yield {"mcpServers": {"cleanup": {
            "type": "http", "url": f"http://127.0.0.1:{sock.getsockname()[1]}/mcp",
            "headers": {"Authorization": f"Bearer {token}"},
            "timeout": max(1000, int(call_timeout_s * 1000)),
        }}}
    finally:
        closing = True

        async def shutdown():
            if cancel_reviews is not None:
                cancel_reviews()
            calls = list(pending)
            for call in calls:
                call.cancel()
            if calls:
                await asyncio.gather(*calls, return_exceptions=True)
            server.should_exit = True
            try:
                await asyncio.wait_for(asyncio.shield(task), 10)
            except asyncio.TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            finally:
                sock.close()

        cleanup = asyncio.create_task(shutdown())
        interrupted = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                interrupted = True
        cleanup.result()
        if interrupted:
            raise asyncio.CancelledError
