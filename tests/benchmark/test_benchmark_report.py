import math

import pytest

from proofstack.evaluation.benchmark_report import summarize_benchmark


def solver(status="solved", *, cost=1.0, tokens=100, complete=True, runtime=2.0):
    return {"status": status, "cost_usd": cost, "tokens": tokens,
            "accounting_complete": complete, "runtime_seconds": runtime}


def test_full_credit_rate_counts_all_selected_problems():
    solver_results = [solver(), solver(), solver("timeout"), solver("missing_proof")]
    grade_results = [{"status": "graded", "score": 7}, {"status": "graded", "score": 3}]
    report = summarize_benchmark(solver_results, grade_results, selected_count=5,
                                 grading_usage={"cost_usd": 2, "tokens": 200,
                                                "accounting_complete": True},
                                 grading_runtime_seconds=6, grading_status="complete")
    assert report["full_credit_rate"] == 1 / 5
    assert report["mean_score_of_graded"] == 5
    assert report["grading_coverage"] == 2 / 5
    assert report["proof_count"] == 2
    assert report["solver_statuses"] == {"solved": 2, "timeout": 1, "missing_proof": 1, "not_run": 1}
    assert report["grading_statuses"] == {"graded": 2, "not_run": 3}
    assert report["solver_runtime_seconds"] == 8
    assert report["grading_runtime_seconds"] == 6
    assert report["usage"]["recorded_total_cost_usd"] == 6
    assert report["usage"]["recorded_total_tokens"] == 600
    assert report["usage"]["accounting_complete"] is False


def test_incomplete_grades_never_become_zero_scores():
    grades = [{"status": "graded", "score": 7}, {"status": "incomplete", "score": None},
              {"status": "error", "score": None}, {"status": "not_run", "score": None}]
    report = summarize_benchmark([solver()] * 4, grades, selected_count=4, grading_status="error")
    assert report["graded_count"] == 1
    assert report["mean_score_of_graded"] == 7
    assert report["full_credit_rate"] == 0.25
    assert report["grading_statuses"] == {"graded": 1, "incomplete": 1, "error": 1, "not_run": 1}
    assert report["usage"]["grading"]["accounting_complete"] is False


def test_no_grades_reports_none_mean_and_zero_success_rate():
    report = summarize_benchmark([solver("solver_error")], [], selected_count=1)
    assert report["mean_score_of_graded"] is None
    assert report["full_credit_rate"] == 0
    assert report["grading_coverage"] == 0
    assert report["grading_status"] == "not_run"
    assert report["usage"]["accounting_complete"] is True


def test_unknown_cost_and_tokens_preserve_known_totals():
    report = summarize_benchmark([solver(cost=None), solver(tokens=None)], [], selected_count=2,
                                 grading_usage={"cost_usd": 0.25, "tokens": None,
                                                "accounting_complete": True},
                                 grading_status="error")
    assert report["usage"]["recorded_total_cost_usd"] == 1.25
    assert report["usage"]["recorded_total_tokens"] == 100
    assert report["usage"]["solver"]["accounting_complete"] is False
    assert report["usage"]["grading"]["accounting_complete"] is False


def test_explicit_incomplete_accounting_flag_is_preserved():
    report = summarize_benchmark([solver(complete=False)], [], selected_count=1)
    assert report["usage"]["solver"]["recorded_cost_usd"] == 1
    assert report["usage"]["accounting_complete"] is False


def test_empty_selection_has_undefined_rates():
    report = summarize_benchmark([], [], selected_count=0)
    assert report["full_credit_rate"] is None
    assert report["grading_coverage"] is None
    assert report["mean_score_of_graded"] is None


@pytest.mark.parametrize("score", [None, -1, 8, True, "7", math.nan, math.inf])
def test_invalid_completed_scores_are_rejected(score):
    with pytest.raises(ValueError, match="score"):
        summarize_benchmark([solver()], [{"status": "graded", "score": score}], selected_count=1)


@pytest.mark.parametrize("selected_count", [-1, True, 1.2])
def test_invalid_selected_count_is_rejected(selected_count):
    with pytest.raises(ValueError, match="selected_count"):
        summarize_benchmark([], [], selected_count=selected_count)


def test_extra_results_are_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        summarize_benchmark([solver(), solver()], [], selected_count=1)
    with pytest.raises(ValueError, match="exceeds"):
        summarize_benchmark([], [{"status": "graded", "score": 7}], selected_count=0)
