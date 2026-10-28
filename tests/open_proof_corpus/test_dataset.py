"""Offline checks for reference-only extraction and stored dataset integrity."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd


DATASET_DIR = Path(__file__).resolve().parents[2] / "opc_problems"
SPEC = importlib.util.spec_from_file_location("update_dataset", DATASET_DIR / "update_dataset.py")
updater = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(updater)


def source_row(problem_id="example", **changes):
    row = {
        "problem_id": problem_id,
        "problem": "Prove the synthetic example.",
        "competition": "Synthetic",
        "level": None,
        "source": "Synthetic fixture",
        "url": None,
        "category": ["algebra"],
        "year": 2025,
        "split": "test",
        "_source_split": "generic",
        "ground_truth_solution": "A synthetic reference proof.",
        "problem_issue": None,
        "ground_truth_issue": None,
        "ground_truth_note": None,
    }
    return row | changes


class ExtractionTests(unittest.TestCase):
    def test_keeps_only_ids_with_a_nonempty_author_reference(self):
        rows = pd.DataFrame([
            source_row("keep", ground_truth_solution=None),
            source_row("keep"),
            source_row("missing", ground_truth_solution=None),
            source_row("blank", ground_truth_solution=" \n\t"),
        ])
        problems, references = updater.build_tables(rows)
        self.assertEqual(problems["problem_id"].tolist(), ["keep"])
        self.assertEqual(references["problem_id"].tolist(), ["keep"])
        self.assertTrue(references["has_reference_solution"].all())
        self.assertEqual(updater.dataset_counts(rows, references)["excluded_without_reference"], 2)

    def test_aggregates_all_occurrences_before_filtering(self):
        rows = pd.DataFrame([
            source_row(ground_truth_solution=" First proof.\n"),
            source_row(ground_truth_solution="Second proof.", _source_split="test"),
            source_row(ground_truth_solution=None, ground_truth_note="Check this reference."),
            source_row(ground_truth_solution=" First proof.\n", problem_issue="Check the statement."),
        ])
        problems, references = updater.build_tables(rows)
        reference = references.iloc[0]
        self.assertEqual(reference["reference_solutions"], [" First proof.\n", "Second proof."])
        self.assertEqual(reference["reference_notes"], ["Check this reference."])
        self.assertEqual(reference["problem_issues"], ["Check the statement."])
        self.assertEqual(reference["reference_status"], "needs_review")
        self.assertEqual(problems.iloc[0]["source_splits"], ["generic", "test"])
        self.assertEqual(problems.iloc[0]["annotated_splits"], ["test"])

    def test_generated_solution_cannot_supply_a_missing_reference(self):
        rows = pd.DataFrame([
            source_row("keep"),
            source_row("discard", ground_truth_solution=None, solution="A generated proof.", score=[7]),
        ])
        problems, references = updater.build_tables(rows)
        self.assertEqual(problems["problem_id"].tolist(), ["keep"])
        for frame in (problems, references):
            self.assertNotIn("solution", frame.columns)
            self.assertNotIn("score", frame.columns)

    def test_nullable_year_and_deduplicated_categories(self):
        rows = pd.DataFrame([
            source_row(year=None, category=["algebra", "geometry"]),
            source_row(year=None, category=["algebra"]),
        ])
        problems, _ = updater.build_tables(rows)
        self.assertEqual(str(problems["year"].dtype), "Int64")
        self.assertTrue(pd.isna(problems.iloc[0]["year"]))
        self.assertEqual(problems.iloc[0]["category"], ["algebra", "geometry"])

    def test_missing_columns_have_an_actionable_error(self):
        rows = pd.DataFrame([source_row()]).drop(columns="ground_truth_solution")
        with self.assertRaisesRegex(ValueError, "Missing upstream columns: ground_truth_solution"):
            updater.build_tables(rows)

    def test_conflicting_statements_are_rejected(self):
        rows = pd.DataFrame([source_row(), source_row(problem="A different statement.")])
        with self.assertRaisesRegex(ValueError, "Conflicting problem for example"):
            updater.build_tables(rows)

    def test_invalid_required_fields_and_years_are_rejected(self):
        invalid_rows = [
            source_row(problem_id=" "),
            source_row(problem=None),
            source_row(year=2025.5),
            source_row(category="algebra"),
        ]
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                updater.build_tables(pd.DataFrame([row]))

    def test_empty_or_reference_free_release_cannot_replace_existing_data(self):
        for rows in (
            pd.DataFrame(columns=updater.SOURCE_COLUMNS + ["_source_split"]),
            pd.DataFrame([source_row(ground_truth_solution=None)]),
        ):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                updater.build_tables(rows)

    def test_written_manifest_matches_tables_and_file_hashes(self):
        rows = pd.DataFrame([source_row(), source_row("excluded", ground_truth_solution=None)])
        problems, references = updater.build_tables(rows)
        manifest = updater.build_manifest({"sha": "synthetic"}, "main", rows, references, [])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            updater.write_dataset(directory, problems, references, manifest)
            stored = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(stored["counts"]["source_problems"], 2)
            self.assertEqual(stored["counts"]["excluded_without_reference"], 1)
            for filename, metadata in stored["outputs"].items():
                contents = (directory / filename).read_bytes()
                self.assertEqual(metadata["sha256"], hashlib.sha256(contents).hexdigest())
                self.assertEqual(metadata["bytes"], len(contents))
                self.assertEqual(metadata["rows"], len(pd.read_parquet(directory / filename)))


class SnapshotTests(unittest.TestCase):
    def test_stored_tables_have_matching_ids_and_nonempty_references(self):
        problems = pd.read_parquet(DATASET_DIR / "problems.parquet")
        references = pd.read_parquet(DATASET_DIR / "reference_solutions.parquet")
        self.assertTrue(problems["problem_id"].is_unique)
        self.assertTrue(references["problem_id"].is_unique)
        self.assertEqual(problems["problem_id"].tolist(), references["problem_id"].tolist())
        self.assertTrue(references["has_reference_solution"].all())
        self.assertTrue(references["reference_solutions"].map(len).gt(0).all())
        self.assertTrue(references["reference_status"].isin(["available", "needs_review"]).all())

    def test_stored_manifest_matches_the_current_snapshot(self):
        manifest = json.loads((DATASET_DIR / "manifest.json").read_text())
        counts = manifest["counts"]
        self.assertEqual(counts["source_problems"], counts["problems"] + counts["excluded_without_reference"])
        self.assertEqual(counts["problems"], counts["problems_with_reference"])
        for filename, metadata in manifest["outputs"].items():
            contents = (DATASET_DIR / filename).read_bytes()
            self.assertEqual(metadata["sha256"], hashlib.sha256(contents).hexdigest())
            self.assertEqual(metadata["bytes"], len(contents))
            self.assertEqual(metadata["rows"], counts["problems"])
        references = pd.read_parquet(DATASET_DIR / "reference_solutions.parquet")
        self.assertEqual(counts["reference_texts"], int(references["reference_solutions"].map(len).sum()))
        self.assertEqual(counts["reference_status"], references["reference_status"].value_counts().to_dict())


if __name__ == "__main__":
    unittest.main()
