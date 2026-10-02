from __future__ import annotations

import asyncio
import json
import pytest

from proofstack.agents.cleanup_reviews import CleanupReviews
from proofstack.agents.cleanup_session import _read_file
from proofstack.agents.cleanup_tools import READ_CHUNK_CHARS


@pytest.fixture
def reviews(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "reviews").mkdir()
    state = {"snapshot": {"answer.tex": "original", "baseline.tex": "baseline"},
             "calls": [], "spent": 0, "report": "A verified attribution report", "budget": 15}

    async def execute(task, *, review_id, snapshot):
        if state["spent"] >= state["budget"]:
            raise RuntimeError("budget exhausted")
        state["calls"].append((task, snapshot))
        if "started" in state:
            state["started"].set()
        if "release" in state:
            await state["release"].wait()
        state["spent"] += 20.14
        path = f"reviews/{review_id}.md"
        (workspace / path).write_text(state["report"])
        return {"report": state["report"], "path": path, "cost_usd": 20.14}

    def make():
        return CleanupReviews(tmp_path, workspace, snapshot=lambda: dict(state["snapshot"]),
                              execute=execute, read_text=_read_file)

    return make, state


def test_timeout_then_retry_reuses_worker_and_returns_paid_report_over_budget(reviews):
    make, state = reviews

    async def run():
        state.update(started=asyncio.Event(), release=asyncio.Event())
        async with make() as jobs:
            first = await jobs.start("Verify citations", "attribution")
            await state["started"].wait()
            # HTTP waiter times out; inference belongs to the invocation.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(jobs.status(first["review_id"], 20), 0.01)
            assert not jobs.tasks[first["review_id"]].done()
            repeated = await jobs.start("Verify citations", "attribution")
            reworded = await jobs.start("Verify citations more concisely", "attribution")
            assert first["review_id"] == repeated["review_id"] == reworded["review_id"]
            assert len((await jobs.status())["reviews"]) == 1
            assert not jobs.attribution_retrieved()
            state["release"].set()
            await jobs.tasks[first["review_id"]]
            # Even a new task must recover this unread report before another call.
            pending = await jobs.start("Different task", "general")
            assert pending["review_id"] == first["review_id"]
            report = await jobs.status(first["review_id"])
            assert report["report"] == state["report"] and report["cost_usd"] == 20.14
            assert jobs.attribution_retrieved()
            assert (await jobs.status(first["review_id"]))["report"] == state["report"]
            denied = await jobs.start("Different task", "general")
            await jobs.tasks[denied["review_id"]]
            error = await jobs.status(denied["review_id"])
            assert error["status"] == "failed" and "budget exhausted" in error["error"]
            assert len(state["calls"]) == 1 and state["spent"] == 20.14

    asyncio.run(run())


def test_start_freezes_snapshot_and_completed_report_survives_new_supervisor(reviews):
    make, state = reviews

    async def run():
        async with make() as jobs:
            first = await jobs.start("Verify", "attribution")
            state["snapshot"]["answer.tex"] = "changed after launch"
            await jobs.tasks[first["review_id"]]
            assert state["calls"][0][1]["answer.tex"] == "original"
        async with make() as resumed:
            assert not resumed.attribution_retrieved()
            report = await resumed.status(first["review_id"])
            assert report["status"] == "completed" and resumed.attribution_retrieved()
            assert len(state["calls"]) == 1

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_settled_unsuccessful_review_can_be_retried_after_resume(reviews, outcome):
    make, state = reviews

    async def fail(*args, **kwargs):
        if outcome == "cancelled":
            raise asyncio.CancelledError
        raise RuntimeError("temporary provider failure")

    async def run():
        async with make() as jobs:
            jobs.execute = fail
            first = await jobs.start("Verify", "attribution")
            await asyncio.gather(jobs.tasks[first["review_id"]], return_exceptions=True)
            assert (await jobs.status(first["review_id"]))["status"] == outcome
        async with make() as resumed:
            retry = await resumed.start("Verify", "attribution")
            assert retry["review_id"] != first["review_id"]
            assert (await resumed.start("Verify", "attribution"))["review_id"] == retry["review_id"]
            await resumed.tasks[retry["review_id"]]
            assert (await resumed.status(retry["review_id"], 240))["status"] == "completed"
            assert resumed.attribution_retrieved()
            assert len(state["calls"]) == 1

    asyncio.run(run())


def test_truncated_report_requires_contiguous_read_of_remaining_chunks(reviews):
    make, state = reviews
    state["report"] = "x" * (2 * READ_CHUNK_CHARS + 100)

    async def run():
        async with make() as jobs:
            first = await jobs.start("Verify", "attribution")
            await jobs.tasks[first["review_id"]]
            report = await jobs.status(first["review_id"])
            assert len(report["report"]) == READ_CHUNK_CHARS
            assert not jobs.attribution_retrieved()
            jobs.note_read(report["path"], 2 * READ_CHUNK_CHARS, len(state["report"]))
            assert not jobs.attribution_retrieved()
            jobs.note_read(report["path"], READ_CHUNK_CHARS, 2 * READ_CHUNK_CHARS)
            jobs.note_read(report["path"], 2 * READ_CHUNK_CHARS, len(state["report"]))
            assert jobs.attribution_retrieved()

    asyncio.run(run())


def test_general_review_does_not_satisfy_attribution_gate(reviews):
    make, _ = reviews

    async def run():
        async with make() as jobs:
            job = await jobs.start("Check exposition")
            await jobs.tasks[job["review_id"]]
            await jobs.status(job["review_id"])
            assert not jobs.attribution_retrieved()

    asyncio.run(run())


def test_interrupted_durable_job_blocks_new_work_without_relaunching(reviews):
    make, state = reviews

    async def run():
        state.update(started=asyncio.Event(), release=asyncio.Event())
        async with make() as jobs:
            first = await jobs.start("Verify", "attribution")
            await state["started"].wait()
            # Loading a persisted in-flight record must not restart its inference.
            async with make() as resumed:
                assert (await resumed.status(first["review_id"]))["status"] == "interrupted"
                assert (await resumed.start("Verify", "attribution"))["status"] == "interrupted"
                with pytest.raises(RuntimeError, match="reconciliation"):
                    await resumed.start("Retry changed wording", "attribution")
            assert len(state["calls"]) == 1

    asyncio.run(run())


def test_shutdown_joins_worker_even_with_repeated_cancellation(reviews):
    make, state = reviews

    async def run():
        cancelled = asyncio.Event()
        release = asyncio.Event()
        started = asyncio.Event()

        async def worker(*args, **kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
                await release.wait()
                state["spent"] = 1.25

        jobs = make()
        jobs.execute = worker

        async def owner():
            async with jobs:
                await jobs.start("Verify", "attribution")
                await started.wait()
                await asyncio.Future()

        task = asyncio.create_task(owner())
        await started.wait()
        task.cancel()
        await cancelled.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state["spent"] == 1.25
        assert all(t.done() for t in jobs.tasks.values())
        record = json.loads(next(jobs.directory.glob("*.json")).read_text())
        assert record["status"] == "cancelled"
        with pytest.raises(RuntimeError, match="ended"):
            await jobs.start("new")

    asyncio.run(run())


def test_shutdown_before_task_starts_does_not_leave_running_record(reviews):
    make, state = reviews

    async def run():
        jobs = make()
        async with jobs:
            await jobs.start("Verify")
            for task in jobs.tasks.values():
                task.cancel()
        record = json.loads(next(jobs.directory.glob("*.json")).read_text())
        assert record["status"] == "cancelled" and not state["calls"]

    asyncio.run(run())


@pytest.mark.parametrize("wait", [-1, 241, float("nan"), float("inf")])
def test_status_wait_is_bounded(reviews, wait):
    make, _ = reviews

    async def run():
        async with make() as jobs:
            with pytest.raises(ValueError, match="between"):
                await jobs.status(wait_seconds=wait)

    asyncio.run(run())


def test_http_tools_recover_id_and_deliver_report_in_chunks(reviews):
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from proofstack.agents.cleanup_tools import cleanup_tools

    make, state = reviews
    state["report"] = "report " * 6000

    async def unused(*args):
        raise AssertionError("Must use the asynchronous review tools")

    async def run():
        state["release"] = asyncio.Event()
        jobs = make()

        async def read_file(path):
            return {"content": _read_file(jobs.workspace, path)}

        async with jobs, cleanup_tools(
            compile_document=unused, codex_review=unused, read_file=read_file,
            write_file=unused, list_files=unused, review_start=jobs.start,
            review_status=jobs.status, review_read=jobs.note_read, cancel_reviews=jobs.cancel,
        ) as config:
            server = config["mcpServers"]["cleanup"]
            async with httpx.AsyncClient(headers=server["headers"]) as client:
                async with streamable_http_client(server["url"], http_client=client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = {t.name: t for t in (await session.list_tools()).tools}
                        assert tools["review_status"].meta == {"anthropic/maxResultSizeChars": 200_000}

                        async def call(name, args):
                            result = await session.call_tool(name, args)
                            assert not result.isError, result
                            return json.loads(result.content[0].text)

                        await call("review", {"task": "Verify", "purpose": "attribution"})
                        [job] = (await call("review_status", {}))["reviews"]
                        assert job["status"] == "running"
                        repeated = await call("review", {"task": "Verify", "purpose": "attribution"})
                        assert repeated["review_id"] == job["review_id"]
                        state["release"].set()
                        report = await call("review_status", {"review_id": job["review_id"], "wait_seconds": 240})
                        assert report["status"] == "completed" and not jobs.attribution_retrieved()
                        remainder = await call("read", {"path": report["path"], "offset": READ_CHUNK_CHARS})
                        assert report["report"] + remainder["content"] == state["report"]
                        assert jobs.attribution_retrieved() and len(state["calls"]) == 1

    asyncio.run(run())
