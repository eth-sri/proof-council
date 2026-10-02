import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from proofstack.agents.ac import author as author_module
from proofstack.agents.ac.author import Author
from proofstack.context import RunContext


def test_download_deadline_covers_retry_backoff(tmp_path, monkeypatch):
    author = Author(RunContext.create(root_workdir=tmp_path, flat=True))
    bridge = Mock()
    bridge.download.side_effect = OSError("transient failure")
    monkeypatch.setattr(author_module, "_DOWNLOAD_TIMEOUT_S", 0.05)
    monkeypatch.setattr(author_module, "_DOWNLOAD_RETRY_DELAY_S", 60)

    async def scenario():
        with pytest.raises(TimeoutError):
            await author._download_completed_files(bridge, "container", provider="openai")
        assert bridge.download.call_count == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
def test_download_never_retries_a_still_running_thread(tmp_path, monkeypatch, stop):
    author = Author(RunContext.create(root_workdir=tmp_path, flat=True))
    started, release = threading.Event(), threading.Event()

    def slow_download(source):
        started.set()
        assert release.wait(5), "test did not release download thread"
        return {"answer.tex": "completed"}

    bridge = Mock(download=Mock(side_effect=slow_download))
    monkeypatch.setattr(author_module, "_DOWNLOAD_TIMEOUT_S", 0.1 if stop == "timeout" else 90)

    async def scenario():
        task = asyncio.create_task(author._download_completed_files(bridge, "container", provider="openai"))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            if stop == "cancel":
                task.cancel()
            with pytest.raises(TimeoutError if stop == "timeout" else asyncio.CancelledError):
                await task
            assert bridge.download.call_count == 1
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_cancellation_during_backoff_does_not_retry(tmp_path, monkeypatch):
    author = Author(RunContext.create(root_workdir=tmp_path, flat=True))
    bridge = Mock()
    bridge.download.side_effect = OSError("transient failure")
    monkeypatch.setattr(author_module, "_DOWNLOAD_RETRY_DELAY_S", 60)

    async def scenario():
        retry_seen = asyncio.Event()

        async def emit(*args, **kwargs):
            retry_seen.set()

        monkeypatch.setattr(author.events, "emit", emit)
        task = asyncio.create_task(author._download_completed_files(bridge, [], provider="anthropic"))
        try:
            await asyncio.wait_for(retry_seen.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert bridge.download.call_count == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_history_read_failure_does_not_skip_paid_author_download(tmp_path, monkeypatch, provider):
    from mathagents.provider_trace import ProviderTrace
    from proofstack.provider_accounting import settle_provider_usage

    ctx = RunContext.create(root_workdir=tmp_path, flat=True)
    author = Author(ctx)
    bridge = Mock(extra_upload_failures=[], uploaded=[])
    bridge.upload.return_value = ["file-synthetic"]
    bridge.render_workspace_listing.return_value = "Synthetic files"
    bridge.render_container_upload_blocks.return_value = []
    bridge.download.return_value = {"answer.tex": "New paid proof"}
    if provider == "openai":
        monkeypatch.setenv("OPENAI_API_KEY", "synthetic-not-used")
        monkeypatch.setattr("openai.OpenAI", Mock())
        monkeypatch.setattr(author_module, "ContainerFileBridge", Mock(return_value=bridge))
        monkeypatch.setattr(author, "_build_api_client_with_file_ids", Mock(return_value=SimpleNamespace(model="synthetic")))
        run = author._run_with_container_files
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-not-used")
        monkeypatch.setattr("anthropic.Anthropic", Mock())
        monkeypatch.setattr(author_module, "AnthropicContainerFileBridge", Mock(return_value=bridge))
        monkeypatch.setattr(author, "_build_anthropic_api_client_with_files", Mock(return_value=SimpleNamespace(model="synthetic")))
        run = author._run_with_anthropic_container_files

    read_text = Path.read_text

    def unreadable_events(path, *args, **kwargs):
        if path == tmp_path / "events.jsonl":
            raise OSError("synthetic history read failure")
        return read_text(path, *args, **kwargs)

    async def query(client, messages, query_fn, *, call_id):
        trace = ProviderTrace(author.workdir / "provider-attempts.jsonl", call_id=call_id)
        trace.update(("synthetic", 0), response_id="completed-response", status="completed", cost=4,
                     usage_unavailable=False, input_tokens=10, output_tokens=20)
        monkeypatch.setattr(Path, "read_text", unreadable_events)
        return 0, [
            {"type": "code_interpreter_call", "container_id": "container-synthetic", "status": "completed"},
            {"role": "assistant", "content": "<ready>true</ready>"},
        ], {"cost": 4, "input_tokens": 10, "output_tokens": 20}

    query_mock = AsyncMock(side_effect=query)
    monkeypatch.setattr(author, "_query", query_mock)

    async def scenario():
        result = await run(Author.Inputs(problem="Synthetic question", round=1, n_rounds=2, answer_tex="Old draft"))
        assert result.answer_tex == "New paid proof" and result.ready
        assert result.artifact_status == "changed"
        query_mock.assert_awaited_once()
        bridge.download.assert_called_once()
        bridge.cleanup.assert_called_once()
        assert ctx.budgets.root().counters.usd == 4
        assert ctx.budgets.root().counters.tokens == 30
        monkeypatch.setattr(Path, "read_text", read_text)
        await settle_provider_usage(ctx, ctx.budgets.root())
        await settle_provider_usage(ctx, ctx.budgets.root())
        assert ctx.budgets.root().counters.usd == 4
        assert ctx.budgets.root().counters.tokens == 30

    asyncio.run(scenario())
