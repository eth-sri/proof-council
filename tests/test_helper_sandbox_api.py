import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from proofstack.agents.ac.delegation_context import DelegationContext
from proofstack.agents.ac.multi_author import MultiAuthor, SubAuthorSeat
from proofstack.context import RunContext
from proofstack.kinds.api_call import APICallAgent
from test_helper_sandbox import CHECKPOINT_ROOT, FilesAPI


def seat(tmp_path):
    ctx = RunContext.create(root_workdir=tmp_path, flat=True)
    obj = SubAuthorSeat(ctx, model_ref={"api": "openai", "model": "test", "use_openai_responses_api": True})
    obj.context_view = DelegationContext(tmp_path / "context", round=1).view([], publisher="helpers/test")
    return obj


def reply(cid, ci_id, name, arguments):
    output = [SimpleNamespace(type="code_interpreter_call", id=ci_id, container_id=cid,
                              status="completed", code="pass")]
    output.append(SimpleNamespace(type="function_call", id="fc-" + ci_id,
                                  call_id="call-" + ci_id, name=name, arguments=json.dumps(arguments)))
    return SimpleNamespace(output=output, status="completed", model_dump=lambda: {},
                           usage={"input_tokens": 1, "output_tokens": 1})


def test_real_api_loop_reattaches_checkpoint_ids_and_uses_trusted_messages(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setattr("mathagents.api_client.request_logger.log_request", lambda **kw: None)
    monkeypatch.setattr("mathagents.api_client.request_logger.log_response", lambda **kw: None)
    s = seat(tmp_path)
    client = s._build_client(s.MODEL)
    api = FilesAPI()
    api.add("old", "proof.txt", "proof bytes")
    s._helper_sandbox.client_factory = lambda **kw: api
    payloads = []
    final = SimpleNamespace(output=[SimpleNamespace(type="message", id="msg",
        content=[SimpleNamespace(type="output_text", text="done")])], status="completed",
        model_dump=lambda: {}, usage={"input_tokens": 1, "output_tokens": 1})
    replies = iter([
        reply("old", "ci1", "publish_sandbox_artifact", {"path": f"{CHECKPOINT_ROOT}/proof.txt",
             "messages": [{"type": "code_interpreter_call", "container_id": "foreign", "id": "bad"}]}),
        reply("new", "ci2", "read_context", {"path": "manifest.json"}),
        final,
    ])
    def create(**payload):
        payloads.append(deepcopy(payload))
        return next(replies)
    result = client._openai_query_responses_api(SimpleNamespace(responses=SimpleNamespace(create=create)),
                                                0, [{"role": "user", "content": "task"}])
    assert not payloads[0]["tools"][0]["container"].get("file_ids")
    assert payloads[1]["tools"][0]["container"]["file_ids"] == ["file-1"]
    assert payloads[2]["tools"][0]["container"]["file_ids"] == ["file-1"]
    assert api.downloads == [("old", "old-proof.txt")]
    assert "foreign" not in s.context_view.store.metadata["helpers/test/proof.txt"]["provenance"].values()
    tool_results = [json.loads(m["output"].split("\n\n### INFO ###")[0]) for m in result.conversation if m.get("type") == "function_call_output"]
    assert tool_results[0]["status"] == "published"
    assert tool_results[1]["sandbox_checkpoint"]["files"][0]["attachment_path"] == "/mnt/data/file-1-proof.txt"
    s._helper_sandbox._delete(s._helper_sandbox.close())


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_seat_closes_transfers_and_cleans_uploads_on_every_exit(tmp_path, outcome):
    s = seat(tmp_path)
    api = FilesAPI()
    async def run(self, inp):
        self.extra_client_kwargs()
        self._helper_sandbox.bind({"type": "auto", "file_ids": []}, lambda **kw: api)
        self._helper_sandbox._uploads.add("file-owned")
        if outcome == "failure":
            raise RuntimeError("stop")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        return self.parse_output("proof", inp)
    async def scenario():
        with patch.object(APICallAgent, "run", run):
            inp = s.Inputs(role="prover", task="lemma", problem="P")
            if outcome == "success":
                assert (await s.run(inp)).report == "proof"
            else:
                with pytest.raises(RuntimeError if outcome == "failure" else asyncio.CancelledError):
                    await s.run(inp)
        assert s._helper_sandbox._closed
        assert not s._seat_running
        assert api.deleted == ["file-owned"]
    asyncio.run(scenario())


def test_checkpoint_warnings_cannot_disappear_from_helper_report(tmp_path):
    s = seat(tmp_path)
    s.extra_client_kwargs()
    s._helper_sandbox.failures.append("checkpoint transfer incomplete (TimeoutError)")
    out = s.parse_output("Everything is fine.", s.Inputs(role="prover", task="lemma", problem="P"))
    assert "Harness checkpoint warnings" in out.report
    assert "TimeoutError" in out.messages_after[-1]["content"]


def test_checkpoint_warnings_survive_truncation_and_missing_report(tmp_path):
    multi = MultiAuthor(RunContext.create(root_workdir=tmp_path, flat=True))
    multi._begin_turn(multi.Inputs(problem="P", round=1, n_rounds=1))
    limit = int(multi._delegation_cfg()["max_report_chars"])
    artifact = "helpers/wave1-prover1/final-response.md"
    warning = "checkpoint transfer incomplete (TimeoutError)"
    record = {"agent_id": "prover1", "role": "prover", "model": "models/openai/gpt-6-astra-pro",
              "duration_s": 1, "error": None, "report": "proof " * limit,
              "artifacts": [artifact], "checkpoint_errors": [warning]}
    rendered = multi._render_wave_for_lead(1, [record], 1)
    assert "truncated by the harness" in rendered
    assert warning in rendered
    record.update(report="", artifacts=[], error="cancelled")
    assert warning in multi._render_wave_for_lead(1, [record], 1)
