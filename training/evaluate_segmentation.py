#!/usr/bin/env python3
"""Validate paired masks and evaluate offline leaf/lesion predictions.

This command is deliberately separate from the live API.  A segmentation
candidate must pass the frozen test gate before its values can be exposed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.measurement import measure_affected_area


REQUIRED_COLUMNS = {
    "sample_id", "image_path", "leaf_mask_path", "lesion_mask_path",
    "crop", "classifier_label", "group_id", "split", "source_url",
    "license_id", "rights_status", "attribution",
}
VALID_SPLITS = {"train", "validation", "test"}
APPROVED_RIGHTS_STATUS = "approved_for_project"


class SegmentationEvidenceError(ValueError):
    """Raised when evidence cannot support a valid evaluation."""


@dataclass(frozen=True)
class SegmentationRecord:
    sample_id: str
    image_path: Path
    leaf_mask_path: Path
    lesion_mask_path: Path
    crop: str
    classifier_label: str
    group_id: str
    split: str
    source_url: str
    license_id: str
    rights_status: str
    attribution: str


@dataclass(frozen=True)
class SampleMetrics:
    sample_id: str
    crop: str
    classifier_label: str
    leaf_dice: float
    leaf_iou: float
    lesion_dice: float
    lesion_iou: float
    true_affected_percent: float
    predicted_affected_percent: float
    absolute_area_error_pp: float


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(path: Path) -> list[SegmentationRecord]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise SegmentationEvidenceError(
                f"manifest is missing columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)

    if not rows:
        raise SegmentationEvidenceError("manifest has no samples")

    base = path.parent.resolve()
    records: list[SegmentationRecord] = []
    seen_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        values = {key: (row.get(key) or "").strip() for key in REQUIRED_COLUMNS}
        empty = sorted(key for key, value in values.items() if not value)
        if empty:
            raise SegmentationEvidenceError(
                f"row {row_number} has empty fields: {', '.join(empty)}"
            )
        if values["sample_id"] in seen_ids:
            raise SegmentationEvidenceError(f"duplicate sample_id: {values['sample_id']}")
        seen_ids.add(values["sample_id"])
        if values["split"] not in VALID_SPLITS:
            raise SegmentationEvidenceError(
                f"row {row_number} has invalid split: {values['split']}"
            )
        if values["rights_status"] != APPROVED_RIGHTS_STATUS:
            raise SegmentationEvidenceError(
                f"row {row_number} is not approved for project use"
            )
        prior_split = group_splits.setdefault(values["group_id"], values["split"])
        if prior_split != values["split"]:
            raise SegmentationEvidenceError(
                f"group_id {values['group_id']} leaks across {prior_split} and {values['split']}"
            )
        records.append(SegmentationRecord(
            sample_id=values["sample_id"],
            image_path=_resolve(base, values["image_path"]),
            leaf_mask_path=_resolve(base, values["leaf_mask_path"]),
            lesion_mask_path=_resolve(base, values["lesion_mask_path"]),
            crop=values["crop"], classifier_label=values["classifier_label"],
            group_id=values["group_id"], split=values["split"],
            source_url=values["source_url"], license_id=values["license_id"],
            rights_status=values["rights_status"], attribution=values["attribution"],
        ))
    return records


def _ground_truth_mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise SegmentationEvidenceError(f"missing mask: {path}")
    with Image.open(path) as image:
        if image.size != expected_size:
            raise SegmentationEvidenceError(
                f"mask {path} size {image.size} does not match image size {expected_size}"
            )
        mask = np.asarray(image.convert("L"))
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 255}):
        raise SegmentationEvidenceError(f"ground-truth mask is not binary 0/255: {path}")
    return mask == 255


def validate_record(record: SegmentationRecord) -> tuple[np.ndarray, np.ndarray]:
    if not record.image_path.is_file():
        raise SegmentationEvidenceError(f"missing image: {record.image_path}")
    with Image.open(record.image_path) as image:
        image.verify()
        size = image.size
    leaf = _ground_truth_mask(record.leaf_mask_path, size)
    lesion = _ground_truth_mask(record.lesion_mask_path, size)
    outside = int(np.count_nonzero(lesion & ~leaf))
    if outside:
        raise SegmentationEvidenceError(
            f"{record.sample_id} has {outside} lesion pixels outside the leaf mask"
        )
    if not np.any(leaf):
        raise SegmentationEvidenceError(f"{record.sample_id} has an empty leaf mask")
    return leaf, lesion


def _prediction_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    if mask.shape != expected_shape:
        raise SegmentationEvidenceError(
            f"prediction {path} shape {mask.shape} does not match {expected_shape}"
        )
    return mask >= 128


def _overlap(predicted: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(predicted & expected))
    predicted_count = int(np.count_nonzero(predicted))
    expected_count = int(np.count_nonzero(expected))
    denominator = predicted_count + expected_count
    union = predicted_count + expected_count - intersection
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    iou = 1.0 if union == 0 else intersection / union
    return dice, iou


def evaluate(
    records: list[SegmentationRecord], predictions: Path, split: str = "test",
    minimum_crop_samples: int = 20,
) -> tuple[list[SampleMetrics], dict[str, object]]:
    selected = [record for record in records if record.split == split]
    if not selected:
        raise SegmentationEvidenceError(f"manifest has no {split} samples")

    results: list[SampleMetrics] = []
    for record in selected:
        true_leaf, true_lesion = validate_record(record)
        predicted_leaf_path = predictions / "leaf" / f"{record.sample_id}.png"
        predicted_lesion_path = predictions / "lesion" / f"{record.sample_id}.png"
        if not predicted_leaf_path.is_file() or not predicted_lesion_path.is_file():
            continue
        predicted_leaf = _prediction_mask(predicted_leaf_path, true_leaf.shape)
        predicted_lesion = _prediction_mask(predicted_lesion_path, true_leaf.shape)
        if not np.any(predicted_leaf):
            continue
        predicted_lesion &= predicted_leaf
        leaf_dice, leaf_iou = _overlap(predicted_leaf, true_leaf)
        lesion_dice, lesion_iou = _overlap(predicted_lesion, true_lesion)
        true_area = measure_affected_area(true_leaf, true_lesion).affected_area_percent
        predicted_area = measure_affected_area(
            predicted_leaf, predicted_lesion
        ).affected_area_percent
        results.append(SampleMetrics(
            sample_id=record.sample_id, crop=record.crop,
            classifier_label=record.classifier_label,
            leaf_dice=leaf_dice, leaf_iou=leaf_iou,
            lesion_dice=lesion_dice, lesion_iou=lesion_iou,
            true_affected_percent=true_area,
            predicted_affected_percent=predicted_area,
            absolute_area_error_pp=abs(predicted_area - true_area),
        ))

    coverage = len(results) / len(selected)
    if not results:
        return [], {
            "status": "failed", "reason": "no_predictions", "samples": len(selected),
            "evaluated": 0, "coverage": 0.0, "passed": False,
        }
    errors = np.asarray([item.absolute_area_error_pp for item in results])
    crops = sorted({record.crop for record in selected})
    per_crop: dict[str, dict[str, float | int]] = {}
    for crop in crops:
        crop_samples = sum(record.crop == crop for record in selected)
        crop_results = [item for item in results if item.crop == crop]
        crop_errors = np.asarray([item.absolute_area_error_pp for item in crop_results])
        per_crop[crop] = {
            "samples": crop_samples,
            "evaluated": len(crop_results),
            "coverage": len(crop_results) / crop_samples,
            "mean_leaf_dice": (
                float(np.mean([item.leaf_dice for item in crop_results]))
                if crop_results else 0.0
            ),
            "mean_lesion_dice": (
                float(np.mean([item.lesion_dice for item in crop_results]))
                if crop_results else 0.0
            ),
            "affected_area_mae_pp": (
                float(np.mean(crop_errors)) if crop_results else 0.0
            ),
        }
    minimum_test_crop_samples = min(item["samples"] for item in per_crop.values())
    minimum_crop_coverage = min(item["coverage"] for item in per_crop.values())
    summary: dict[str, object] = {
        "status": "evaluated", "samples": len(selected), "evaluated": len(results),
        "coverage": coverage,
        "mean_leaf_dice": float(np.mean([item.leaf_dice for item in results])),
        "mean_leaf_iou": float(np.mean([item.leaf_iou for item in results])),
        "mean_lesion_dice": float(np.mean([item.lesion_dice for item in results])),
        "mean_lesion_iou": float(np.mean([item.lesion_iou for item in results])),
        "affected_area_mae_pp": float(np.mean(errors)),
        "affected_area_p95_error_pp": float(np.percentile(errors, 95)),
        "minimum_required_crop_samples": minimum_crop_samples,
        "minimum_test_crop_samples": minimum_test_crop_samples,
        "minimum_crop_coverage": minimum_crop_coverage,
        "per_crop": per_crop,
    }
    summary["passed"] = bool(
        coverage >= 0.90
        and summary["mean_leaf_dice"] >= 0.95
        and summary["mean_lesion_dice"] >= 0.75
        and summary["affected_area_mae_pp"] <= 5.0
        and summary["affected_area_p95_error_pp"] <= 15.0
        and minimum_test_crop_samples >= minimum_crop_samples
        and minimum_crop_coverage >= 0.90
    )
    return results, summary


def write_evidence(output_dir: Path, results: list[SampleMetrics], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "segmentation_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "segmentation_per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        fields = list(SampleMetrics.__dataclass_fields__)
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(item) for item in results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    parser.add_argument("--minimum-crop-samples", type=int, default=20)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    results, summary = evaluate(
        records, args.predictions, args.split, args.minimum_crop_samples
    )
    write_evidence(args.output_dir, results, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
