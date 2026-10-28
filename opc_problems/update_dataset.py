# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2,<3", "pyarrow>=18"]
# ///
"""Extract public OPC problems that have at least one author reference solution."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fnmatch
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request

import pandas as pd


DATASET = "INSAIT-Institute/OPC"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET}"
TEXT_METADATA = ("competition", "level", "source", "url")
REFERENCE_FIELDS = {
    "reference_solutions": "ground_truth_solution",
    "problem_issues": "problem_issue",
    "reference_issues": "ground_truth_issue",
    "reference_notes": "ground_truth_note",
}
SOURCE_COLUMNS = [
    "problem_id", "problem", *TEXT_METADATA, "category", "year", "split",
    *REFERENCE_FIELDS.values(),
]
OUTPUT_NAMES = ("problems.parquet", "reference_solutions.parquet", "manifest.json")


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ProofCouncil-OPC-downloader/2"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def unique_texts(values):
    return sorted({value for value in values if isinstance(value, str) and value.strip()})


def consistent_text(values, problem_id, field):
    variants = unique_texts(values)
    if len(variants) > 1:
        raise ValueError(f"Conflicting {field} for {problem_id}; inspect the upstream change")
    return variants[0] if variants else None


def validate_source_rows(rows):
    missing_columns = set(SOURCE_COLUMNS + ["_source_split"]) - set(rows.columns)
    if missing_columns:
        raise ValueError(f"Missing upstream columns: {', '.join(sorted(missing_columns))}")
    if rows.empty:
        raise ValueError("The upstream dataset is empty")
    for field in ("problem_id", "problem"):
        valid = rows[field].map(lambda value: isinstance(value, str) and bool(value.strip()))
        if not valid.all():
            raise ValueError(f"Missing or invalid {field} in upstream rows")


def consistent_year(values, problem_id):
    years = values.dropna().unique()
    if not len(years):
        return None
    try:
        year = int(years[0])
        valid = len(years) == 1 and float(years[0]) == year
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"Conflicting or invalid year for {problem_id}")
    return year


def combined_categories(values, problem_id):
    categories = []
    for category_list in values:
        if category_list is None:
            continue
        if isinstance(category_list, str) or not pd.api.types.is_list_like(category_list):
            raise ValueError(f"Invalid category list for {problem_id}")
        categories.extend(category_list)
    return unique_texts(categories)


def problem_record(problem_id, rows):
    record = {"problem_id": problem_id}
    for field in ("problem", *TEXT_METADATA):
        record[field] = consistent_text(rows[field], problem_id, field)
    record["year"] = consistent_year(rows["year"], problem_id)
    record["category"] = combined_categories(rows["category"], problem_id)
    record["source_splits"] = unique_texts(rows["_source_split"])
    record["annotated_splits"] = unique_texts(rows["split"])
    return record


def reference_record(problem_id, rows):
    record = {"problem_id": problem_id}
    for target, source in REFERENCE_FIELDS.items():
        record[target] = unique_texts(rows[source])
    if not record["reference_solutions"]:
        return None
    needs_review = (
        any(record[field] for field in ("problem_issues", "reference_issues", "reference_notes"))
        or len(record["reference_solutions"]) > 1
    )
    record["has_reference_solution"] = True
    record["reference_status"] = "needs_review" if needs_review else "available"
    return record


def build_tables(rows):
    validate_source_rows(rows)
    problem_records, reference_records = [], []
    for problem_id, group in rows.groupby("problem_id", sort=True):
        reference = reference_record(problem_id, group)
        if reference is None:
            continue
        problem_records.append(problem_record(problem_id, group))
        reference_records.append(reference)
    if not problem_records:
        raise ValueError("The upstream dataset has no problems with reference solutions")

    problems = pd.DataFrame(problem_records)
    problems["year"] = problems["year"].astype("Int64")
    for field in ("problem_id", "problem", *TEXT_METADATA):
        problems[field] = problems[field].astype("string")
    references = pd.DataFrame(reference_records)
    for field in ("problem_id", "reference_status"):
        references[field] = references[field].astype("string")
    return problems, references


def source_files(info):
    configs = info["cardData"]["configs"]
    if len(configs) != 1 or configs[0]["config_name"] != "default":
        raise ValueError("Upstream configurations changed; review which configurations to include")
    available = {item["rfilename"] for item in info["siblings"] if item["rfilename"].endswith(".parquet")}
    files = {}
    for entry in configs[0]["data_files"]:
        patterns = entry["path"] if isinstance(entry["path"], list) else [entry["path"]]
        matches = {path for path in available if any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)}
        if not matches:
            raise ValueError(f"No files found for split {entry['split']}")
        for path in matches:
            if path in files:
                raise ValueError(f"Overlapping upstream split definitions for {path}")
            files[path] = entry["split"]
    if set(files) != available:
        raise ValueError("Unmapped upstream parquet files; review the dataset configuration")
    return sorted(files.items())


def load_release_info(revision):
    encoded_revision = urllib.parse.quote(revision, safe="")
    url = f"https://huggingface.co/api/datasets/{DATASET}/revision/{encoded_revision}"
    info = json.loads(fetch(url))
    if info["cardData"].get("license") != "cc-by-nc-sa-4.0":
        raise ValueError("The upstream license changed; update the local attribution before refreshing")
    return info


def load_source_rows(info):
    frames, sources, counts = [], [], Counter()
    for path, split in source_files(info):
        print(f"Downloading {path} at {info['sha'][:12]}...", flush=True)
        data = fetch(f"{DATASET_URL}/resolve/{info['sha']}/{path}")
        frame = pd.read_parquet(BytesIO(data), columns=SOURCE_COLUMNS, engine="pyarrow")
        frame["_source_split"] = split
        frames.append(frame)
        counts[split] += len(frame)
        sources.append({
            "path": path, "split": split, "rows": len(frame), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    expected = {
        item["name"]: item["num_examples"]
        for item in info["cardData"]["dataset_info"]["splits"]
    }
    if dict(counts) != expected:
        raise ValueError(f"Row counts differ from upstream metadata: {dict(counts)} != {expected}")
    return pd.concat(frames, ignore_index=True), sources


def dataset_counts(rows, references):
    source_problems = rows["problem_id"].nunique()
    return {
        "source_rows": len(rows),
        "source_problems": int(source_problems),
        "excluded_without_reference": int(source_problems - len(references)),
        "problems": len(references),
        "problems_with_reference": len(references),
        "reference_texts": int(references["reference_solutions"].map(len).sum()),
        "reference_status": references["reference_status"].value_counts().sort_index().to_dict(),
    }


def build_manifest(info, revision, rows, references, sources):
    return {
        "schema_version": 2,
        "dataset": DATASET,
        "source_url": DATASET_URL,
        "revision": info["sha"],
        "requested_revision": revision,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "license": "CC-BY-NC-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "reference_source_field": "ground_truth_solution",
        "transformations": (
            "Deduplicate by problem_id; keep only problems with a nonempty author reference; "
            "preserve exact texts and all reference variants; aggregate metadata and issue notes; "
            "exclude generated attempts and grades."
        ),
        "counts": dataset_counts(rows, references),
        "sources": sources,
        "outputs": {},
    }


def write_dataset(output_dir, problems, references, manifest):
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".opc-update-", dir=output_dir) as temporary:
        staging = Path(temporary)
        for name, frame in zip(OUTPUT_NAMES[:2], (problems, references), strict=True):
            path = staging / name
            frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
            manifest["outputs"][name] = {
                "rows": len(frame), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for name in OUTPUT_NAMES:
            os.replace(staging / name, output_dir / name)


def update(output_dir, revision):
    info = load_release_info(revision)
    rows, sources = load_source_rows(info)
    problems, references = build_tables(rows)
    manifest = build_manifest(info, revision, rows, references, sources)
    write_dataset(output_dir, problems, references, manifest)
    print(json.dumps(manifest["counts"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Upstream branch, tag, or commit (default: main)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    update(args.output_dir, args.revision)


if __name__ == "__main__":
    main()
