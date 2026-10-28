"""Summarize a harness benchmark while keeping missing grades and usage explicit."""

from collections import Counter
import math
from statistics import mean


def _status_counts(results, selected_count):
    if len(results) > selected_count:
        raise ValueError("Result count exceeds selected problem count")
    counts = Counter(result["status"] for result in results)
    if len(results) < selected_count:
        counts["not_run"] += selected_count - len(results)
    return dict(sorted(counts.items()))


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return value


def _completed_scores(results):
    scores = []
    for result in results:
        if result["status"] != "graded":
            continue
        score = _number(result.get("score"), "Completed grade score")
        if score > 7:
            raise ValueError("Completed grade score must be between 0 and 7")
        scores.append(score)
    return scores


def _usage(records, *, expected_count):
    usage = {"recorded_cost_usd": 0.0, "recorded_tokens": 0,
             "accounting_complete": len(records) == expected_count}
    for record in records:
        usage["accounting_complete"] &= record.get("accounting_complete") is True
        for source, target in (("cost_usd", "recorded_cost_usd"), ("tokens", "recorded_tokens")):
            value = record.get(source)
            if value is None:
                usage["accounting_complete"] = False
            else:
                usage[target] += _number(value, source)
    return usage


def _combined_usage(solver_results, selected_count, grading_usage, grading_status):
    solver = _usage(solver_results, expected_count=selected_count)
    if grading_usage is None and grading_status == "not_run":
        grading_usage = {"cost_usd": 0, "tokens": 0, "accounting_complete": True}
    grading = _usage([grading_usage or {}], expected_count=1)
    return {
        "solver": solver, "grading": grading,
        "recorded_total_cost_usd": solver["recorded_cost_usd"] + grading["recorded_cost_usd"],
        "recorded_total_tokens": solver["recorded_tokens"] + grading["recorded_tokens"],
        "accounting_complete": solver["accounting_complete"] and grading["accounting_complete"],
    }


def summarize_benchmark(solver_results, grade_results, *, selected_count: int,
                        grading_usage: dict | None = None,
                        grading_runtime_seconds: float = 0,
                        grading_status: str = "not_run") -> dict:
    if type(selected_count) is not int or selected_count < 0:
        raise ValueError("selected_count must be a nonnegative integer")
    solver_statuses = _status_counts(solver_results, selected_count)
    grading_statuses = _status_counts(grade_results, selected_count)
    scores = _completed_scores(grade_results)
    full_credit_count = sum(score == 7 for score in scores)
    return {
        "selected_count": selected_count,
        "solver_statuses": solver_statuses, "grading_statuses": grading_statuses,
        "grading_status": grading_status,
        "proof_count": solver_statuses.get("solved", 0), "graded_count": len(scores),
        "grading_coverage": len(scores) / selected_count if selected_count else None,
        "mean_score_of_graded": mean(scores) if scores else None,
        "full_credit_count": full_credit_count,
        "full_credit_rate": full_credit_count / selected_count if selected_count else None,
        "usage": _combined_usage(solver_results, selected_count, grading_usage, grading_status),
        "solver_runtime_seconds": sum(_number(item["runtime_seconds"], "Solver runtime")
                                      for item in solver_results),
        "grading_runtime_seconds": _number(grading_runtime_seconds, "Grading runtime"),
        "metric_note": "Full-credit rate uses every selected problem as its denominator, including "
                       "failures and ungraded problems. Mean score uses completed grades only. "
                       "Missing scores are not zeros; recorded usage may be incomplete.",
    }
