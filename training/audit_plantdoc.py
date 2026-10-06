"""Audit PlantDoc before it is used as Candidate-v2 training data.

This script does not train a model. It validates the pinned repository,
decodes every train/test image, verifies the explicit class mapping, detects
exact pixel duplicates, and flags perceptually similar images for manual
review. PlantDoc test remains evaluation-only and is never added to the clean
training manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    split: str
    relative_path: str
    source_class: str
    target_label: str
    file_sha256: str
    pixel_sha256: str
    dhash64: str
    width: int
    height: int
    file_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_revision(path: Path) -> str | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=candidate,
                capture_output=True,
                check=False,
                text=True,
            )
            if completed.returncode == 0:
                return completed.stdout.strip()
    return None


def decoded_fingerprints(path: Path) -> tuple[str, str, int, int]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
    width, height = image.size
    pixel_digest = hashlib.sha256()
    pixel_digest.update(f"RGB:{width}x{height}:".encode("ascii"))
    pixel_digest.update(np.asarray(image, dtype=np.uint8).tobytes())

    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    dhash = 0
    for value in bits.reshape(-1):
        dhash = (dhash << 1) | int(value)
    return pixel_digest.hexdigest(), f"{dhash:016x}", width, height


def load_mapping(path: Path) -> tuple[dict[str, str], dict[str, object]]:
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("classes"), dict):
        raise RuntimeError("Mapping must contain a classes object.")
    classes = value["classes"]
    if not classes or not all(isinstance(key, str) and isinstance(target, str) for key, target in classes.items()):
        raise RuntimeError("Every mapping entry must map a source directory to a target label.")
    dataset = value.get("dataset")
    return classes, dataset if isinstance(dataset, dict) else {}


def discover_images(
    dataset_root: Path,
    classes: dict[str, str],
) -> tuple[list[tuple[str, Path, str, str]], list[str]]:
    discovered: list[tuple[str, Path, str, str]] = []
    unknown_classes: list[str] = []
    for split in ("train", "test"):
        split_root = dataset_root / split
        if not split_root.is_dir():
            raise RuntimeError(f"Missing required PlantDoc split directory: {split_root}")
        for directory in sorted(path for path in split_root.iterdir() if path.is_dir()):
            if directory.name not in classes:
                unknown_classes.append(f"{split}/{directory.name}")
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    discovered.append((split, path, directory.name, classes[directory.name]))
    return discovered, unknown_classes


def audit_images(
    dataset_root: Path,
    discovered: list[tuple[str, Path, str, str]],
) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    records: list[ImageRecord] = []
    decode_errors: list[dict[str, str]] = []
    for split, path, source_class, target_label in discovered:
        relative_path = str(path.relative_to(dataset_root))
        try:
            pixel_hash, dhash, width, height = decoded_fingerprints(path)
            records.append(
                ImageRecord(
                    split=split,
                    relative_path=relative_path,
                    source_class=source_class,
                    target_label=target_label,
                    file_sha256=file_sha256(path),
                    pixel_sha256=pixel_hash,
                    dhash64=dhash,
                    width=width,
                    height=height,
                    file_bytes=path.stat().st_size,
                )
            )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            decode_errors.append({"split": split, "relative_path": relative_path, "error": str(exc)})
    return records, decode_errors


def exact_duplicate_rows(records: list[ImageRecord]) -> tuple[list[dict[str, object]], dict[str, str]]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.pixel_sha256].append(record)

    rows: list[dict[str, object]] = []
    exclusions: dict[str, str] = {}
    group_number = 0
    for pixel_hash, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        group_number += 1
        splits = {member.split for member in members}
        labels = {member.target_label for member in members}
        cross_split = len(splits) > 1
        label_conflict = len(labels) > 1
        ordered = sorted(members, key=lambda item: item.relative_path)

        if cross_split:
            for member in ordered:
                if member.split == "train":
                    exclusions[member.relative_path] = "exact_pixel_duplicate_of_test"
        elif label_conflict:
            for member in ordered:
                if member.split == "train":
                    exclusions[member.relative_path] = "exact_pixel_duplicate_label_conflict"
        else:
            train_members = [member for member in ordered if member.split == "train"]
            for member in train_members[1:]:
                exclusions[member.relative_path] = "redundant_exact_train_duplicate"

        for member in ordered:
            rows.append(
                {
                    "group_id": group_number,
                    "pixel_sha256": pixel_hash,
                    "cross_split": cross_split,
                    "label_conflict": label_conflict,
                    **asdict(member),
                }
            )
    return rows, exclusions


def near_duplicate_rows(records: list[ImageRecord], maximum_distance: int) -> list[dict[str, object]]:
    if not 0 <= maximum_distance <= 64:
        raise RuntimeError("near-duplicate-distance must be between 0 and 64.")
    values = [int(record.dhash64, 16) for record in records]
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            right = records[right_index]
            if left.pixel_sha256 == right.pixel_sha256:
                continue
            distance = (values[left_index] ^ values[right_index]).bit_count()
            if distance > maximum_distance:
                continue
            rows.append(
                {
                    "hamming_distance": distance,
                    "cross_split": left.split != right.split,
                    "label_match": left.target_label == right.target_label,
                    "left_split": left.split,
                    "left_path": left.relative_path,
                    "left_label": left.target_label,
                    "right_split": right.split,
                    "right_path": right.relative_path,
                    "right_label": right.target_label,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            not bool(row["cross_split"]),
            int(row["hamming_distance"]),
            str(row["left_path"]),
            str(row["right_path"]),
        ),
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    classes, dataset_metadata = load_mapping(args.mapping)

    expected_revision = dataset_metadata.get("revision")
    actual_revision = repository_revision(dataset_root)
    if expected_revision and actual_revision != expected_revision:
        raise RuntimeError(
            f"Dataset revision mismatch: expected {expected_revision}, received {actual_revision or 'unknown'}."
        )

    discovered, unknown_classes = discover_images(dataset_root, classes)
    if unknown_classes:
        raise RuntimeError(f"Unmapped PlantDoc class directories: {unknown_classes}")
    if not discovered:
        raise RuntimeError("No supported PlantDoc images were discovered.")

    records, decode_errors = audit_images(dataset_root, discovered)
    exact_rows, exclusions = exact_duplicate_rows(records)
    near_rows = near_duplicate_rows(records, args.near_duplicate_distance)

    manifest_rows: list[dict[str, object]] = []
    clean_rows: list[dict[str, object]] = []
    for record in records:
        if record.split != "train":
            continue
        reason = exclusions.get(record.relative_path, "")
        row = {**asdict(record), "include": not reason, "exclusion_reason": reason}
        manifest_rows.append(row)
        if not reason:
            clean_rows.append(row)

    class_counts = Counter((record.split, record.source_class, record.target_label) for record in records)
    class_rows = [
        {"split": split, "source_class": source, "target_label": target, "decoded_images": count}
        for (split, source, target), count in sorted(class_counts.items())
    ]
    cross_split_near = [row for row in near_rows if bool(row["cross_split"])]
    conflicting_near = [row for row in near_rows if not bool(row["label_match"])]

    summary = {
        "status": "review_required",
        "dataset": dataset_metadata,
        "dataset_root": str(dataset_root),
        "audited_splits": ["train", "test"],
        "expected_revision": expected_revision,
        "actual_revision": actual_revision,
        "near_duplicate_hamming_distance": args.near_duplicate_distance,
        "discovered_images": len(discovered),
        "decoded_images": len(records),
        "decode_errors": len(decode_errors),
        "train_decoded": sum(record.split == "train" for record in records),
        "test_decoded": sum(record.split == "test" for record in records),
        "exact_duplicate_groups": len({row["group_id"] for row in exact_rows}),
        "exact_duplicate_members": len(exact_rows),
        "excluded_train_images": len(exclusions),
        "clean_train_images": len(clean_rows),
        "near_duplicate_pairs": len(near_rows),
        "cross_split_near_duplicate_pairs": len(cross_split_near),
        "different_label_near_duplicate_pairs": len(conflicting_near),
        "warning": (
            "The clean manifest removes only decoded exact duplicates. Near-duplicate pairs and labels require "
            "manual review before Candidate-v2 training. PlantDoc test is evaluation-only."
        ),
    }

    write_json(output_dir / "audit_summary.json", summary)
    write_csv(output_dir / "decode_errors.csv", ["split", "relative_path", "error"], decode_errors)
    write_csv(
        output_dir / "class_counts.csv",
        ["split", "source_class", "target_label", "decoded_images"],
        class_rows,
    )
    duplicate_fields = [
        "group_id",
        "cross_split",
        "label_conflict",
        *ImageRecord.__dataclass_fields__.keys(),
    ]
    write_csv(output_dir / "exact_duplicate_groups.csv", duplicate_fields, exact_rows)
    near_fields = [
        "hamming_distance",
        "cross_split",
        "label_match",
        "left_split",
        "left_path",
        "left_label",
        "right_split",
        "right_path",
        "right_label",
    ]
    write_csv(output_dir / "near_duplicate_pairs.csv", near_fields, near_rows)
    write_csv(output_dir / "cross_split_near_duplicate_pairs.csv", near_fields, cross_split_near)
    write_csv(output_dir / "different_label_near_duplicate_pairs.csv", near_fields, conflicting_near)
    manifest_fields = [*ImageRecord.__dataclass_fields__.keys(), "include", "exclusion_reason"]
    write_csv(output_dir / "train_manifest.csv", manifest_fields, manifest_rows)
    write_csv(output_dir / "clean_train_manifest.csv", manifest_fields, clean_rows)
    print(json.dumps(summary, indent=2))
    print(f"Saved PlantDoc audit evidence to {output_dir}")


if __name__ == "__main__":
    main()
