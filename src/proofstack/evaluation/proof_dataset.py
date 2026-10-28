"""Load candidate proofs and join reference material only on the evaluator side."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ProofCase:
    problem_id: str
    candidate_id: str
    problem: str
    solution: str
    reference_solutions: list[str]
    reference_notes: list[str]
    reference_status: str

    def reference_context(self):
        return {"problem": self.problem, "reference_solutions": self.reference_solutions,
                "reference_notes": self.reference_notes}

    def context_hash(self):
        return content_hash(self.reference_context())


def content_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def text_list(value, label, *, nonempty=False):
    if isinstance(value, str) or not pd.api.types.is_list_like(value):
        raise ValueError(f"{label} must be a list of strings")
    result = [require_text(item, label) for item in value]
    if nonempty and not result:
        raise ValueError(f"{label} must contain a reference solution")
    return result


def normalize_reference_row(row, problem_id):
    notes = []
    for field in ("problem_issues", "reference_issues", "reference_notes"):
        notes.extend(f"{field}: {note}" for note in text_list(row[field], field))
    references = text_list(row["reference_solutions"], "reference_solutions", nonempty=True)
    status = row["reference_status"]
    if status not in {"available", "needs_review"}:
        raise ValueError(f"Unsupported reference status for {problem_id}: {status}")
    if notes or len(references) > 1:
        status = "needs_review"
    return {"reference_solutions": references, "reference_notes": notes,
            "reference_status": status}


def load_candidate_records(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
        if isinstance(records, dict):
            records = records.get("solutions")
    if not isinstance(records, list) or not records:
        raise ValueError("Solutions must be a nonempty JSON list, {'solutions': [...]}, or JSONL file")
    candidates, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each candidate must be a JSON object")
        problem_id = require_text(record.get("problem_id", record.get("id")), "problem_id")
        candidate_id = require_text(record.get("candidate_id", "default"), "candidate_id")
        solution = require_text(record.get("solution", record.get("latex")), "solution or latex")
        identity = (problem_id, candidate_id)
        if identity in seen:
            raise ValueError(f"Duplicate candidate identity: {identity}")
        seen.add(identity)
        candidates.append({"problem_id": problem_id, "candidate_id": candidate_id, "solution": solution})
    return candidates


def load_reference_table(dataset_dir: Path):
    problems = pd.read_parquet(dataset_dir / "problems.parquet")
    references = pd.read_parquet(dataset_dir / "reference_solutions.parquet")
    for name, table in (("problems", problems), ("references", references)):
        if table.problem_id.duplicated().any():
            raise ValueError(f"Duplicate problem_id in {name} table")
        for problem_id in table.problem_id:
            require_text(problem_id, f"{name}.problem_id")
    if set(problems.problem_id) != set(references.problem_id):
        raise ValueError("Problem and reference tables have different problem IDs")
    return problems.merge(references, on="problem_id", validate="one_to_one").set_index("problem_id")


def load_cases(dataset_dir: Path, candidates_path: Path):
    return _join_reference_cases(load_reference_table(dataset_dir), load_candidate_records(candidates_path))


def load_reference_cases(dataset_dir: Path, problem_ids):
    """Load reference contexts for preflight validation, without candidate proofs."""
    candidates = [{"problem_id": problem_id, "candidate_id": "default", "solution": ""}
                  for problem_id in problem_ids]
    return _join_reference_cases(load_reference_table(dataset_dir), candidates)


def _join_reference_cases(table, candidates):
    cases = []
    for candidate in candidates:
        problem_id = candidate["problem_id"]
        if problem_id not in table.index:
            raise ValueError(f"Unknown problem_id: {problem_id}")
        row = table.loc[problem_id]
        reference = normalize_reference_row(row, problem_id)
        cases.append(ProofCase(**candidate, problem=require_text(row["problem"], "problem"),
                               **reference))
    return cases


def dataset_provenance(dataset_dir: Path):
    hashes = {
        filename: hashlib.sha256((dataset_dir / filename).read_bytes()).hexdigest()
        for filename in ("problems.parquet", "reference_solutions.parquet")
    }
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for filename, digest in hashes.items():
        expected = manifest.get("outputs", {}).get(filename, {}).get("sha256")
        if expected is not None and expected != digest:
            raise ValueError(f"Dataset checksum differs from manifest: {filename}")
    return {"revision": manifest.get("revision"), "files": hashes}


def load_marking_schemes(path: Path | None, cases):
    if path is None:
        return {}
    schemes = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(schemes, dict):
        raise ValueError("Marking schemes must be an object keyed by problem_id")
    for case in cases:
        if case.problem_id not in schemes:
            raise ValueError(f"Missing frozen marking scheme: {case.problem_id}")
        record = schemes[case.problem_id]
        if not isinstance(record, dict) or record.get("context_sha256") != case.context_hash():
            raise ValueError(f"Marking scheme context mismatch: {case.problem_id}")
        require_text(record.get("marking_scheme"), "marking_scheme")
    return schemes
