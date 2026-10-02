import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import anthropic
import openai
import pytest

from proofstack.agents.ac import author as author_module
from proofstack.agents.ac.author import Author
from proofstack.agents.ac.multi_author import MultiAuthor
from proofstack.context import RunContext
from proofstack.budget import BudgetExhausted, BudgetSpec


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("author_class,delegation", [(Author, False), (MultiAuthor, False), (MultiAuthor, True)])
@pytest.mark.parametrize("result", ["changed", "unchanged", "download_failed", "download_retried"])
@pytest.mark.parametrize("budget_hit", [False, True])
def test_multi_author_real_container_result_path(tmp_path, monkeypatch, provider, author_class, delegation, result, budget_hit):
    ctx = RunContext.create(
        root_workdir=tmp_path, flat=True,
        run_budget=BudgetSpec(max_usd=1) if budget_hit else None,
        component_configs={"Author": {"delegation": {"enabled": delegation}}},
    )
    author = author_class(ctx, name="Author")
    monkeypatch.setattr(author_module, "_DOWNLOAD_RETRY_DELAY_S", 0)
    inp = Author.Inputs(problem="P", round=1, n_rounds=3, answer_tex="old manuscript")
    monkeypatch.setattr(author, "USE_CONTAINER_FILES", True)
    monkeypatch.setattr(author, "_container_file_provider_api", lambda: provider)
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-used")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-used")
    monkeypatch.setattr(openai, "OpenAI", Mock())
    monkeypatch.setattr(anthropic, "Anthropic", Mock())
    bridge = Mock(extra_upload_failures=[], uploaded=[])
    bridge.upload.return_value = ["file-proof"]
    bridge.render_workspace_listing.return_value = "answer.tex"
    bridge.render_container_upload_blocks.return_value = []
    bridge.download.return_value = {"answer.tex": "new manuscript"} if result == "changed" else {}
    if result == "download_failed":
        bridge.download.side_effect = OSError("download interrupted")
    elif result == "download_retried":
        bridge.download.side_effect = [
            OSError("transient list failure"), OSError("transient content failure"),
            {"answer.tex": "new manuscript"},
        ]
    monkeypatch.setattr(author_module, "ContainerFileBridge", Mock(return_value=bridge))
    monkeypatch.setattr(author_module, "AnthropicContainerFileBridge", Mock(return_value=bridge))
    client = SimpleNamespace(model="offline-model")
    monkeypatch.setattr(author, "_build_api_client_with_file_ids", lambda _: client)
    monkeypatch.setattr(author, "_build_anthropic_api_client_with_files", lambda: client)
    conversation = [
        {"type": "code_interpreter_call", "container_id": "container-proof", "status": "completed"},
        {"role": "assistant", "content": "<ready>true</ready>"},
    ]
    query = AsyncMock(return_value=(0, conversation, {
        "cost": 1.25, "input_tokens": 100, "output_tokens": 40, "reasoning_tokens": 20,
    }))
    monkeypatch.setattr(author, "_query", query)
    if author_class is MultiAuthor:
        monkeypatch.setattr(author, "delegation_summary", lambda: "delegation preserved")

    if budget_hit:
        with pytest.raises(BudgetExhausted) as caught:
            asyncio.run(author(**inp.model_dump()))
        out = caught.value.completed_output
        saved = list(tmp_path.rglob("completed_author.json"))
        assert len(saved) == 1
        assert json.loads(saved[0].read_text())["answer_tex"] == out.answer_tex
    else:
        out = asyncio.run(author(**inp.model_dump()))

    changed = result in ("changed", "download_retried")
    assert out.answer_tex == ("new manuscript" if changed else "old manuscript")
    assert out.artifact_status == ("changed" if changed else "failed" if result == "download_failed" else "unchanged")
    assert out.ready == (result != "download_failed")
    assert out.delegation_summary == ("delegation preserved" if author_class is MultiAuthor else "")
    assert out.via == ("container_files" if provider == "openai" else "anthropic_container_files")
    query.assert_awaited_once()
    retried = result in ("download_failed", "download_retried")
    assert bridge.download.call_count == (3 if retried else 1)
    bridge.cleanup.assert_called_once()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    calls = [e for e in events if e["kind"] == "model.call"]
    assert len(calls) == 1 and calls[0]["payload"]["cost_usd"] == 1.25
    assert calls[0]["payload"]["reasoning_tokens"] == 20
    retries = [e for e in events if e["kind"] == "ac.author.container_download_retry"]
    assert [e["payload"]["attempt"] for e in retries] == ([1, 2] if retried else [])
    assert all(e["payload"]["provider"] == provider for e in retries)
    assert any(e["kind"] == "agent.error" for e in events) == budget_hit
