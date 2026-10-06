"""Finalize the manually reviewed PlantDoc Candidate-v2 training manifest.

This step is deliberately separate from the automated image audit. It applies
the committed conservative quarantine decision, verifies the exact uploaded
audit evidence, and emits a deterministic training manifest plus review
summary. It never adds PlantDoc test images and never relabels source data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


REQUIRED_FIELDS = [
    "split",
    "relative_path",
    "source_class",
    "target_label",
    "file_sha256",
    "pixel_sha256",
    "dhash64",
    "width",
    "height",
    "file_bytes",
    "include",
    "exclusion_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument(
        "--quarantine",
        type=Path,
        default=Path(__file__).with_name("plantdoc_manual_quarantine.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def require_string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise RuntimeError(f"{name} must be a non-empty string list.")
    if len(value) != len(set(value)):
        raise RuntimeError(f"{name} contains duplicate paths.")
    return value


def load_clean_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REQUIRED_FIELDS:
            raise RuntimeError(
                f"Unexpected clean manifest columns: {reader.fieldnames}; expected {REQUIRED_FIELDS}."
            )
        rows = list(reader)
    return REQUIRED_FIELDS, rows


def validate_clean_rows(rows: list[dict[str, str]]) -> None:
    paths: set[str] = set()
    file_hashes: set[str] = set()
    pixel_hashes: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        path = row["relative_path"]
        if row["split"] != "train":
            raise RuntimeError(f"Non-training row at line {row_number}: {path}")
        if row["include"].lower() != "true" or row["exclusion_reason"]:
            raise RuntimeError(f"Unclean row at line {row_number}: {path}")
        if path in paths:
            raise RuntimeError(f"Duplicate relative_path in clean manifest: {path}")
        if row["file_sha256"] in file_hashes:
            raise RuntimeError(f"Duplicate file hash remains in clean manifest: {path}")
        if row["pixel_sha256"] in pixel_hashes:
            raise RuntimeError(f"Duplicate pixel hash remains in clean manifest: {path}")
        paths.add(path)
        file_hashes.add(row["file_sha256"])
        pixel_hashes.add(row["pixel_sha256"])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    audit_summary = read_json(args.audit_summary)
    quarantine = read_json(args.quarantine)

    expected_audit_sha = quarantine.get("audit_summary_sha256")
    expected_manifest_sha = quarantine.get("clean_manifest_sha256")
    if sha256(args.audit_summary) != expected_audit_sha:
        raise RuntimeError("audit_summary.json does not match the manually reviewed audit evidence.")
    if sha256(args.clean_manifest) != expected_manifest_sha:
        raise RuntimeError("clean_train_manifest.csv does not match the manually reviewed audit evidence.")
    if audit_summary.get("status") != "review_required":
        raise RuntimeError("Expected the source audit to have status review_required.")
    if audit_summary.get("actual_revision") != quarantine.get("dataset_revision"):
        raise RuntimeError("Dataset revision differs between the audit and quarantine decision.")
    if audit_summary.get("decode_errors") != 0:
        raise RuntimeError("Cannot finalize a manifest with image decode errors.")

    fieldnames, rows = load_clean_manifest(args.clean_manifest)
    validate_clean_rows(rows)
    expected_clean = quarantine.get("expected_clean_manifest_rows")
    if len(rows) != expected_clean or audit_summary.get("clean_train_images") != expected_clean:
        raise RuntimeError(
            f"Expected {expected_clean} clean rows, received {len(rows)}; audit summary disagrees or changed."
        )

    additional = require_string_list(quarantine.get("additional_quarantine_paths"), "additional_quarantine_paths")
    already_removed = require_string_list(
        quarantine.get("already_removed_by_exact_audit"), "already_removed_by_exact_audit"
    )
    clean_paths = {row["relative_path"] for row in rows}
    missing_additional = sorted(set(additional) - clean_paths)
    unexpectedly_present = sorted(set(already_removed) & clean_paths)
    if missing_additional:
        raise RuntimeError(f"Reviewed quarantine paths missing from clean manifest: {missing_additional}")
    if unexpectedly_present:
        raise RuntimeError(f"Exact-audit exclusions unexpectedly remain: {unexpectedly_present}")

    quarantine_paths = set(additional)
    reviewed_rows = [row for row in rows if row["relative_path"] not in quarantine_paths]
    expected_reviewed = quarantine.get("expected_reviewed_manifest_rows")
    if len(reviewed_rows) != expected_reviewed:
        raise RuntimeError(f"Expected {expected_reviewed} reviewed rows, received {len(reviewed_rows)}.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_path = output_dir / "reviewed_train_manifest.csv"
    write_csv(reviewed_path, fieldnames, reviewed_rows)

    before_counts = Counter(row["target_label"] for row in rows)
    after_counts = Counter(row["target_label"] for row in reviewed_rows)
    summary = {
        "status": "reviewed_train_manifest_generated",
        "production_approved": False,
        "dataset_revision": quarantine["dataset_revision"],
        "audit_summary_sha256": expected_audit_sha,
        "clean_manifest_sha256": expected_manifest_sha,
        "quarantine_decision_sha256": sha256(args.quarantine),
        "reviewed_manifest_sha256": sha256(reviewed_path),
        "clean_manifest_rows": len(rows),
        "exact_audit_exclusions": audit_summary["excluded_train_images"],
        "already_removed_review_paths": len(already_removed),
        "additional_manual_quarantine": len(additional),
        "reviewed_manifest_rows": len(reviewed_rows),
        "class_counts_before_review": dict(sorted(before_counts.items())),
        "class_counts_after_review": dict(sorted(after_counts.items())),
        "training_authorized": True,
        "authorization_scope": "Candidate-v2 experimentation only; not production approval.",
        "test_policy": "PlantDoc test is consumed and remains evaluation-only.",
    }
    summary_path = output_dir / "manifest_review.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved reviewed manifest to {reviewed_path}")
    print(f"Saved review evidence to {summary_path}")


if __name__ == "__main__":
    main()
