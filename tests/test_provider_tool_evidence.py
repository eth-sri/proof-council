"""Bounded poll evidence, without SDK calls or private run fixtures."""
import json

import pytest

from mathagents.provider_trace import ProviderTrace
from proofstack.agents.ac.critic import ACCritic
from proofstack.context import RunContext


@pytest.mark.parametrize("diagnostics", ["x" * 34_000, "\u03b1" * 34_000])
def test_oversized_logs_keep_checker_metadata_and_bounded_evidence(tmp_path, diagnostics):
    critic = ACCritic(RunContext.create(root_workdir=tmp_path, flat=True))
    inp = critic.Inputs(problem="Synthetic", research_notes_tex="notes")
    code = critic._notes_verification_code(inp)
    item = {"type": "code_interpreter_call", "id": "ci", "container_id": "container", "status": "completed",
            "code": code, "outputs": [{"type": "logs", "logs": json.dumps(critic._notes_metadata(inp)) + diagnostics}]}
    trace = ProviderTrace()
    trace.response(None, "request", 0, {"status": "in_progress", "output": [item]})
    kept, = trace.tool_evidence()
    assert kept["code"] == code and kept["container_id"] == "container" and kept["id"] == "ci"
    assert kept["evidence_truncated"] and len(json.dumps(kept).encode()) <= 32_000
    assert critic._notes_verified(inp, evidence=[kept])
    assert "evidence_truncated" not in item


def test_oversized_code_is_explicitly_unverifiable_not_silently_dropped():
    trace = ProviderTrace()
    trace.response(None, "request", 0, {"output": [{"type": "code_interpreter_call", "id": "ci",
        "container_id": "container", "status": "completed", "code": "x" * 40_000, "outputs": []}]})
    kept, = trace.tool_evidence()
    assert kept["id"] == "ci" and kept["code"] is None and kept["evidence_truncated"]


def test_completed_checker_is_observed_during_poll_and_receipt_survives_final_snapshot(tmp_path):
    observed = []
    receipt = {"file_id": "receipt", "content": "synthetic"}

    def capture(client, trace):
        calls = trace.tool_evidence()
        if calls and not calls[0].get("notes_verification_receipt"):
            observed.append(trace.attempts[("request", 0)]["status"])
            trace.retain_tool_receipt(calls[0], receipt)

    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl", on_response=capture)
    item = {"type": "code_interpreter_call", "id": "ci", "container_id": "container",
            "status": "completed", "code": "print(1)", "outputs": []}
    trace.response(None, "request", 0, {"status": "in_progress", "output": [item]})
    assert observed == ["in_progress"]
    assert "notes_verification_receipt" not in item
    trace.response(None, "request", 0, {"id": "response", "status": "completed", "output": [
        {**item, "code": None}, {"type": "message", "content": [{"type": "output_text", "text": "Report"}]}]})
    saved = json.loads((tmp_path / f"provider-completed-{trace.id}.json").read_text())
    assert saved["tool_evidence"][0]["notes_verification_receipt"] == receipt
    assert saved["tool_evidence"][0]["code"] == "print(1)"
    assert observed == ["in_progress"]


def test_later_poll_does_not_erase_previously_seen_tool_items():
    trace = ProviderTrace()
    for name in ("first", "second"):
        trace.response(None, "request", 0, {"status": "in_progress", "output": [
            {"type": "code_interpreter_call", "id": name, "status": "completed", "code": "print(1)"}]})
    assert [item["id"] for item in trace.tool_evidence()] == ["first", "second"]


def test_final_report_is_durable_before_artifact_capture_can_be_interrupted(tmp_path):
    def interrupted(client, trace):
        path = tmp_path / f"provider-completed-{trace.id}.json"
        assert json.loads(path.read_text())["report"] == "Paid report"
        raise KeyboardInterrupt("synthetic interruption during capture")

    trace = ProviderTrace(tmp_path / "provider-attempts.jsonl", on_response=interrupted)
    with pytest.raises(KeyboardInterrupt):
        trace.response(None, "request", 0, {"status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "Paid report"}]}]})
