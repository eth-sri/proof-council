import json

from scripts.cleanup_smoke import evidence


def test_audit_requires_successful_tool_results_and_ignores_codex_errors(tmp_path):
    records = [
        {"type": "error", "message": "Codex failed"},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "ok", "name": "Task", "input": {"subagent_type": "correctness-reviewer"}},
            {"type": "tool_use", "id": "failed", "name": "Agent", "input": {"subagent_type": "exposition-reviewer"}},
            {"type": "tool_use", "id": "launched", "name": "Agent", "input": {"subagent_type": "not-yet-done"}},
            {"type": "tool_use", "id": "pending", "name": "mcp__cleanup__review", "input": {}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "ok", "content": "checked"},
            {"type": "tool_result", "tool_use_id": "failed", "content": "error", "is_error": True},
            {"type": "tool_result", "tool_use_id": "launched", "content": "Async agent launched successfully"},
        ]}},
        {"type": "system", "subtype": "task_notification", "tool_use_id": "ok", "status": "completed", "summary": "Proof checked"},
        {"type": "system", "subtype": "task_notification", "tool_use_id": "failed", "status": "failed", "summary": "Review failed"},
        None,
    ]
    (tmp_path / "cli_stdout.log").write_text("\n".join(map(json.dumps, records)) + "\n{partial")
    observed = evidence(tmp_path)
    assert observed["completed_native_reviewers"] == ["correctness-reviewer"]
    assert observed["completed_mcp_tools"] == []
    assert observed["failed_tool_results"] == 1
    assert observed["codex_reports"] == 0


def test_successful_launch_without_final_report_is_not_a_completed_review(tmp_path):
    records = [
        {"message": {"content": [{"type": "tool_use", "id": "review", "name": "Agent",
                                  "input": {"subagent_type": "correctness-reviewer"}}]}},
        {"message": {"content": [{"type": "tool_result", "tool_use_id": "review", "content": "launched"}]}},
        {"type": "system", "subtype": "task_notification", "tool_use_id": "review", "status": "completed", "summary": ""},
    ]
    (tmp_path / "cli_stdout.log").write_text("\n".join(map(json.dumps, records)))
    assert evidence(tmp_path)["completed_native_reviewers"] == []


def test_finishing_evidence_is_scoped_to_new_invocation(tmp_path):
    first = tmp_path / "first.log"
    first.write_text(json.dumps({"type": "system", "subtype": "init", "tools": ["Task"]}))
    second = tmp_path / "second.log"
    second.write_text(json.dumps({"type": "system", "subtype": "init", "tools": ["mcp__cleanup__compile"]}))
    observed = evidence(tmp_path, paths=[second])
    assert observed["exposed_tools"] == [["mcp__cleanup__compile"]]
    assert observed["native_tool_calls"] == 0
