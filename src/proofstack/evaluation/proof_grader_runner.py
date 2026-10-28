"""Sequential, checkpointed marking-scheme generation and ensemble grading."""

from collections import Counter
import asyncio
from statistics import mean, median

from proofstack.budget import BudgetExhausted
from proofstack.evaluation.proof_dataset import content_hash
from proofstack.evaluation.proof_grader import MarkingSchemeAgent, ProofGrader
from proofstack.evaluation.run_support import usage_report, write_json


def result_record(case):
    return {
        "problem_id": case.problem_id,
        "candidate_id": case.candidate_id,
        "solution_sha256": content_hash(case.solution),
        "context_sha256": case.context_hash(),
        "reference_status": case.reference_status,
        "reference_notes": case.reference_notes,
        "status": "not_run",
        "score": None,
        "sample_scores": [],
    }


def summarize(results, *, cost_usd, tokens):
    graded = [result["score"] for result in results if result["status"] == "graded"]
    return {
        "candidates": len(results),
        "statuses": dict(Counter(result["status"] for result in results)),
        "mean_score_of_graded": mean(graded) if graded else None,
        "full_credit_count": sum(score == 7 for score in graded),
        "graded_count": len(graded),
        "cost_usd": cost_usd,
        "tokens": tokens,
        "note": "Scores are model assessments, not formal proof verification. Missing scores are not zeros.",
    }


class ProofGraderRunner:
    def __init__(self, context, *, samples=5, marking_schemes=None):
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1 or samples % 2 == 0:
            raise ValueError("samples must be a positive odd integer")
        self.context = context
        self.samples = samples
        self.marking_schemes = dict(marking_schemes or {})
        self.unusable_schemes = {}
        self.scheme_agent = MarkingSchemeAgent(context)
        self.grader = ProofGrader(context)
        self.output_dir = context.root_workdir

    async def run(self, cases):
        results = [result_record(case) for case in cases]
        for case, result in zip(cases, results):
            if case.reference_status == "needs_review":
                result["status"] = "skipped_reference_issue"
                continue
            try:
                await self.evaluate_candidate(case, result)
            except asyncio.CancelledError:
                result.update(status="interrupted", error="Evaluation interrupted")
                self.save_candidate(result)
                raise
            except BudgetExhausted as error:
                result["error"] = str(error)
                if result["status"] != "graded":
                    result["status"] = "budget_exhausted"
                self.save_candidate(result)
                break
            except Exception as error:
                result.update(status="error", error=f"{type(error).__name__}: {error}")
                self.save_candidate(result)
                # A provider failure can have unreported cost; stop before another paid call.
                break
            finally:
                self.save_results(results)
        self.save_results(results)
        return results

    async def evaluate_candidate(self, case, result):
        result["status"] = "in_progress"
        scheme = await self.get_marking_scheme(case, result)
        if scheme is None:
            return
        result["marking_scheme_sha256"] = content_hash(scheme["marking_scheme"])
        result["samples"] = []
        inputs = {**case.reference_context(), "marking_scheme": scheme["marking_scheme"],
                  "solution": case.solution}
        for _ in range(self.samples):
            try:
                response = await self.grader(**inputs)
            except BudgetExhausted as error:
                if getattr(error, "completed_output", None) is not None:
                    self.record_grade(result, error.completed_output)
                raise
            self.record_grade(result, response)

    def record_grade(self, result, response):
        result["samples"].append(response.model_dump(mode="json"))
        result["sample_scores"].append(response.grade.score if response.grade else None)
        if len(result["samples"]) == self.samples:
            valid_scores = [score for score in result["sample_scores"] if score is not None]
            result["status"] = "graded" if len(valid_scores) == self.samples else "incomplete"
            result["score"] = median(valid_scores) if len(valid_scores) == self.samples else None
        self.save_candidate(result)

    async def get_marking_scheme(self, case, result):
        if case.problem_id in self.marking_schemes:
            return self.marking_schemes[case.problem_id]
        if case.problem_id in self.unusable_schemes:
            return self.record_scheme(case, result, self.unusable_schemes[case.problem_id])
        try:
            response = await self.scheme_agent(**case.reference_context())
        except BudgetExhausted as error:
            if getattr(error, "completed_output", None) is not None:
                self.record_scheme(case, result, error.completed_output)
            raise
        return self.record_scheme(case, result, response)

    def record_scheme(self, case, result, response):
        result["scheme_response"] = response.model_dump(mode="json")
        if response.scheme is None or not response.scheme.usable:
            self.unusable_schemes[case.problem_id] = response
            result["status"] = "unusable_marking_scheme"
            self.save_candidate(result)
            return None
        scheme = {
            "context_sha256": case.context_hash(),
            "marking_scheme": response.scheme.marking_scheme,
            "model_config": self.scheme_agent.MODEL,
            "prompt_sha256": content_hash(self.scheme_agent.SYSTEM_PROMPT),
            "raw_response": response.raw_response,
        }
        self.marking_schemes[case.problem_id] = scheme
        write_json(self.output_dir / "marking_schemes.json", self.marking_schemes)
        self.save_candidate(result)
        return scheme

    def save_candidate(self, result):
        key = content_hash([result["problem_id"], result["candidate_id"]])
        directory = self.output_dir / "grades"
        directory.mkdir(exist_ok=True)
        result["details_file"] = f"grades/{key}.json"
        write_json(directory / f"{key}.json", result)

    def save_results(self, results):
        compact = [{key: value for key, value in result.items()
                    if key not in {"samples", "scheme_response"}} for result in results]
        finished = all(result["status"] in {"graded", "incomplete", "unusable_marking_scheme",
                                             "skipped_reference_issue"} for result in results)
        usage = usage_report(self.output_dir, completed=finished)
        write_json(self.output_dir / "results.json", compact)
        write_json(self.output_dir / "summary.json",
                   {**summarize(compact, cost_usd=usage["cost_usd"], tokens=usage["tokens"]), **usage})
