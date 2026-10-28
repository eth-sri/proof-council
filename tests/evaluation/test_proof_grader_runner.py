import asyncio
from dataclasses import replace
import json

import pytest

from proofstack.evaluation.proof_grader import Grade, parse_json_response


def grade_reply(score):
    return json.dumps({"score": score, "rationale": "Synthetic grading explanation.", "errors": []})


def test_flagged_references_are_skipped_without_model_calls(make_runner, proof_case):
    runner, scheme_client, grader_client = make_runner([])
    flagged = replace(proof_case, reference_status="needs_review")
    result = asyncio.run(runner.run([flagged]))[0]
    assert result["status"] == "skipped_reference_issue"
    assert result["score"] is None
    assert not scheme_client.messages
    assert not grader_client.messages
    summary = json.loads((runner.output_dir / "summary.json").read_text())
    assert summary["statuses"] == {"skipped_reference_issue": 1}


def test_real_agent_wrapper_accounts_all_independent_calls(make_runner, proof_case):
    runner, scheme_client, grader_client = make_runner(list(map(grade_reply, [0, 7, 3, 4, 6])))
    results = asyncio.run(runner.run([proof_case]))
    assert len(scheme_client.messages) == 1
    assert len(grader_client.messages) == 5
    assert results[0]["score"] == 4
    assert results[0]["sample_scores"] == [0, 7, 3, 4, 6]
    summary = json.loads((runner.output_dir / "summary.json").read_text())
    assert summary["cost_usd"] == pytest.approx(0.06)
    assert summary["tokens"] == 120
    assert len(list((runner.output_dir / "agents").glob("**/messages.json"))) == 6
    scheme_request = json.loads(scheme_client.messages[0][-1]["content"])
    assert "solution" not in scheme_request
    for messages in grader_client.messages:
        assert json.loads(messages[-1]["content"])["solution"] == proof_case.solution


def test_distinct_candidates_share_scheme_but_not_grade_samples(make_runner, proof_case):
    runner, scheme_client, grader_client = make_runner(
        list(map(grade_reply, [7, 1, 3, 4, 7, 6])), samples=3)
    cases = [proof_case, replace(proof_case, candidate_id="b", solution="Another proof.")]
    results = asyncio.run(runner.run(cases))
    assert [result["score"] for result in results] == [3, 6]
    assert len(scheme_client.messages) == 1
    assert len(grader_client.messages) == 6
    for messages in grader_client.messages:
        assert len(messages) == 2
        assert "rationale" not in json.loads(messages[-1]["content"])


@pytest.mark.parametrize("budget", [2.5, 3])
def test_budget_crossing_mid_ensemble_retains_paid_reply_without_scoring(make_runner, proof_case, budget):
    runner, scheme_client, grader_client = make_runner(
        [grade_reply(3), grade_reply(4)], samples=3, budget=budget, cost=1)
    results = asyncio.run(runner.run([proof_case, replace(proof_case, candidate_id="b")]))
    assert len(scheme_client.messages) == 1
    assert len(grader_client.messages) == 2
    assert results[0]["status"] == "budget_exhausted"
    assert results[0]["sample_scores"] == [3, 4]
    assert results[0]["score"] is None
    assert results[1]["status"] == "not_run"
    detail = json.loads((runner.output_dir / results[0]["details_file"]).read_text())
    assert len(detail["samples"]) == 2


def test_budget_crossing_on_last_sample_preserves_complete_score(make_runner, proof_case):
    runner, _, grader_client = make_runner([grade_reply(5)] * 3, samples=3, budget=4, cost=1)
    results = asyncio.run(runner.run([proof_case, replace(proof_case, candidate_id="b")]))
    assert results[0]["status"] == "graded"
    assert results[0]["score"] == 5
    assert results[1]["status"] == "not_run"
    assert len(grader_client.messages) == 3
    summary = json.loads((runner.output_dir / "summary.json").read_text())
    assert summary["cost_usd"] == 4


def test_failed_scheme_is_shared_without_regenerating(make_runner, proof_case):
    runner, scheme_client, grader_client = make_runner([], scheme_usable=False)
    results = asyncio.run(runner.run([proof_case, replace(proof_case, candidate_id="b")]))
    assert len(scheme_client.messages) == 1
    assert not grader_client.messages
    assert [result["status"] for result in results] == ["unusable_marking_scheme"] * 2


def test_malformed_sample_does_not_become_zero_or_partial_ensemble_score(make_runner, proof_case):
    runner, _, _ = make_runner([grade_reply(7), "not JSON", grade_reply(7)], samples=3)
    result = asyncio.run(runner.run([proof_case]))[0]
    assert result["status"] == "incomplete"
    assert result["score"] is None
    assert result["sample_scores"] == [7, None, 7]
    assert result["samples"][1]["raw_response"] == "not JSON"
    assert result["samples"][1]["error"]


@pytest.mark.parametrize("score", [True, "7", 7.0, -1, 8, None])
def test_grade_schema_rejects_plausible_but_invalid_scores(score):
    with pytest.raises(ValueError):
        parse_json_response(grade_reply(score), Grade)


def test_model_config_cannot_reenable_evaluator_tools(make_runner, proof_case):
    runner, scheme_client, grader_client = make_runner(
        [grade_reply(7)], samples=1,
        model_options={"tools": [(None, {"type": "web_search"})], "max_tool_calls": 12})
    assert asyncio.run(runner.run([proof_case]))[0]["status"] == "graded"
    for client in (scheme_client, grader_client):
        assert client.config["tools"] == []
        assert client.config["max_tool_calls"] == 0


def test_unknown_provider_usage_is_explicit_in_summary(make_runner, proof_case):
    runner, _, _ = make_runner([grade_reply(7)] * 5, usage={
        "cost": 0, "input_tokens": 0, "output_tokens": 0, "usage_unavailable": True,
    })
    asyncio.run(runner.run([proof_case]))
    summary = json.loads((runner.output_dir / "summary.json").read_text())
    assert summary["cost_usd"] == 0
    assert summary["usage_unavailable"] is True
    assert summary["accounting_complete"] is False
