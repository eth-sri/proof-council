"""Choose reproducible benchmark problems without exposing reference solutions."""

from dataclasses import dataclass
from pathlib import Path
import random

from proofstack.evaluation.proof_dataset import (
    dataset_provenance, load_reference_table, normalize_reference_row, require_text,
)


@dataclass(frozen=True)
class BenchmarkSelection:
    records: list[dict[str, str]]
    report: dict
    provenance: dict


def _validate_sampling(limit, seed):
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")


def _eligible_records(table):
    records, flagged_ids = [], set()
    for problem_id, row in table.sort_index().iterrows():
        problem = require_text(row["problem"], f"problem {problem_id}")
        reference = normalize_reference_row(row, problem_id)
        flagged = reference["reference_status"] == "needs_review"
        if flagged:
            flagged_ids.add(problem_id)
        if not flagged:
            records.append({"problem_id": problem_id, "problem": problem})
    return records, flagged_ids


def _unique_statements(records):
    unique = {}
    for record in records:
        unique.setdefault(record["problem"], record)
    return list(unique.values())


def _explicit_records(problem_ids, records, all_ids, flagged_ids):
    if not isinstance(problem_ids, (list, tuple)) or not problem_ids:
        raise ValueError("problem_ids must be a nonempty list of problem IDs")
    for problem_id in problem_ids:
        require_text(problem_id, "problem_id")
    if len(set(problem_ids)) != len(problem_ids):
        raise ValueError("Duplicate problem IDs requested")
    missing = set(problem_ids) - all_ids
    if missing:
        raise ValueError(f"Unknown problem IDs: {', '.join(sorted(missing))}")
    flagged = set(problem_ids) & flagged_ids
    if flagged:
        raise ValueError(f"Flagged references are excluded: {', '.join(sorted(flagged))}")
    by_id = {record["problem_id"]: record for record in records}
    selected = [by_id[problem_id] for problem_id in sorted(problem_ids)]
    if len(_unique_statements(selected)) != len(selected):
        raise ValueError("Requested problem IDs contain duplicate problem statements")
    return selected


def _sample_records(records, limit, seed):
    if not records:
        raise ValueError("No eligible benchmark problems remain")
    if limit is None:
        return records
    if limit > len(records):
        raise ValueError(f"Requested {limit} problems, but only {len(records)} unique eligible problems exist")
    return sorted(random.Random(seed).sample(records, limit), key=lambda item: item["problem_id"])


def select_problems(dataset_dir: Path, *, limit: int | None = 10, seed: int = 0,
                    problem_ids: list[str] | None = None) -> BenchmarkSelection:
    """Sample unique statements; explicit IDs select every named problem, ignoring limit."""
    _validate_sampling(limit, seed)
    provenance = dataset_provenance(dataset_dir)
    table = load_reference_table(dataset_dir)
    records, flagged_ids = _eligible_records(table)
    unique_records = _unique_statements(records)
    if problem_ids is None:
        selected = _sample_records(unique_records, limit, seed)
    else:
        selected = _explicit_records(problem_ids, records, set(table.index), flagged_ids)
    report = {
        "mode": "explicit" if problem_ids is not None else "all" if limit is None else "sample",
        "limit": None if problem_ids is not None else limit,
        "seed": seed,
        "source_problem_count": len(table), "eligible_problem_count": len(records),
        "eligible_unique_statement_count": len(unique_records),
        "excluded_flagged_count": len(flagged_ids),
        "duplicate_statement_count": len(records) - len(unique_records),
        "selected_problem_ids": [record["problem_id"] for record in selected],
    }
    return BenchmarkSelection(selected, report, provenance)
