"""Evaluate a fixed ONNX candidate on field or unsupported images.

Ground-truth directory names are used only for scoring. Model inputs are always
decoded image pixels; filenames and directories are never passed to inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.metrics import accuracy_score, f1_score


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Sample:
    path: Path
    relative_path: str
    source_class: str | None
    expected_label: str | None


@dataclass(frozen=True)
class Result:
    relative_path: str
    source_class: str | None
    expected_label: str | None
    predicted_label: str
    confidence: float
    accepted: bool
    quality_ok: bool
    width: int
    height: int
    sharpness: float
    correct: bool | None
    top5: list[dict[str, float | str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("field", "ood"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_candidate(model_path: Path, labels_path: Path, manifest_path: Path) -> tuple[list[str], dict[str, object]]:
    manifest = read_json(manifest_path)
    labels = read_json(labels_path)
    if not isinstance(manifest, dict):
        raise RuntimeError("candidate_manifest.json must contain an object.")
    if manifest.get("production_approved") is not False or manifest.get("candidate_only") is not True:
        raise RuntimeError("Evaluation requires the unchanged candidate-only manifest.")
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
        raise RuntimeError("labels.json must contain a non-empty string array.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("Candidate manifest does not contain artifact hashes.")
    for path in (model_path, labels_path):
        expected = artifacts.get(path.name)
        if not isinstance(expected, str):
            raise RuntimeError(f"Candidate manifest has no hash for {path.name}.")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {path.name}; refusing evaluation.")
    return labels, manifest


def repository_revision(path: Path) -> str | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=candidate,
                capture_output=True,
                check=True,
                text=True,
            )
            return completed.stdout.strip()
    return None


def field_samples(dataset_root: Path, mapping_path: Path, labels: list[str]) -> tuple[list[Sample], dict[str, object]]:
    mapping = read_json(mapping_path)
    if not isinstance(mapping, dict) or not isinstance(mapping.get("classes"), dict):
        raise RuntimeError("Field mapping must contain a classes object.")
    classes = mapping["classes"]
    directories = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    unknown = [path.name for path in directories if path.name not in classes]
    if unknown:
        raise RuntimeError(f"Unmapped field-dataset classes: {unknown}")
    invalid_targets = sorted({classes[path.name] for path in directories} - set(labels))
    if invalid_targets:
        raise RuntimeError(f"Mappings refer to labels absent from labels.json: {invalid_targets}")
    expected_revision = mapping.get("dataset", {}).get("revision") if isinstance(mapping.get("dataset"), dict) else None
    actual_revision = repository_revision(dataset_root)
    if expected_revision and actual_revision != expected_revision:
        raise RuntimeError(
            f"Dataset revision mismatch: expected {expected_revision}, received {actual_revision or 'unknown'}."
        )
    samples: list[Sample] = []
    for directory in directories:
        expected = classes[directory.name]
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(Sample(path, str(path.relative_to(dataset_root)), directory.name, expected))
    if not samples:
        raise RuntimeError("No supported image files were found in the field dataset.")
    dataset_metadata = mapping.get("dataset")
    return samples, dataset_metadata if isinstance(dataset_metadata, dict) else {}


def ood_samples(dataset_root: Path) -> list[Sample]:
    samples = [
        Sample(path, str(path.relative_to(dataset_root)), None, None)
        for path in sorted(dataset_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not samples:
        raise RuntimeError("No supported image files were found in the OOD directory.")
    return samples


def preprocess(path: Path) -> tuple[np.ndarray, int, int, float]:
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(str(exc)) from exc
    width, height = image.size
    gray = np.asarray(image.resize((256, 256)).convert("L"), dtype=np.float32)
    sharpness = float((np.diff(gray, axis=1).var() + np.diff(gray, axis=0).var()) / 2)
    fitted = ImageOps.fit(image, (224, 224))
    tensor = np.asarray(fitted, dtype=np.float32) / 255.0
    tensor = (tensor - MEAN) / STD
    return np.transpose(tensor, (2, 0, 1)).astype(np.float32), width, height, sharpness


def batches(values: list[tuple[Sample, np.ndarray, int, int, float]], size: int) -> Iterable[list[tuple[Sample, np.ndarray, int, int, float]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def infer(
    samples: list[Sample],
    model_path: Path,
    labels: list[str],
    threshold: float,
    batch_size: int,
) -> tuple[list[Result], list[dict[str, str]]]:
    if not 0 < threshold <= 1:
        raise RuntimeError("Acceptance threshold must be in (0, 1].")
    prepared: list[tuple[Sample, np.ndarray, int, int, float]] = []
    decode_errors: list[dict[str, str]] = []
    for sample in samples:
        try:
            tensor, width, height, sharpness = preprocess(sample.path)
            prepared.append((sample, tensor, width, height, sharpness))
        except ValueError as exc:
            decode_errors.append({"path": sample.relative_path, "error": str(exc)})
    if not prepared:
        raise RuntimeError("Every discovered image failed decoding.")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    results: list[Result] = []
    for batch in batches(prepared, batch_size):
        inputs = np.stack([row[1] for row in batch])
        logits = np.asarray(session.run(None, {input_name: inputs})[0], dtype=np.float64)
        if logits.ndim != 2 or logits.shape != (len(batch), len(labels)):
            raise RuntimeError(
                f"Unexpected ONNX output shape {logits.shape}; expected {(len(batch), len(labels))}."
            )
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        for row, sample_probabilities in zip(batch, probabilities):
            sample, _, width, height, sharpness = row
            ranking = np.argsort(sample_probabilities)[::-1][:5]
            prediction_index = int(ranking[0])
            confidence = float(sample_probabilities[prediction_index])
            predicted_label = labels[prediction_index]
            quality_ok = min(width, height) >= 224 and sharpness >= 25
            accepted = quality_ok and confidence >= threshold
            results.append(
                Result(
                    relative_path=sample.relative_path,
                    source_class=sample.source_class,
                    expected_label=sample.expected_label,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    accepted=accepted,
                    quality_ok=quality_ok,
                    width=width,
                    height=height,
                    sharpness=sharpness,
                    correct=predicted_label == sample.expected_label if sample.expected_label else None,
                    top5=[{"label": labels[int(index)], "confidence": float(sample_probabilities[int(index)])} for index in ranking],
                )
            )
    return results, decode_errors


def write_predictions(results: list[Result], path: Path) -> None:
    fields = [
        "relative_path", "source_class", "expected_label", "predicted_label", "confidence",
        "accepted", "quality_ok", "width", "height", "sharpness", "correct", "top5",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["top5"] = json.dumps(row["top5"], separators=(",", ":"))
            writer.writerow(row)


def confidence_summary(results: list[Result]) -> dict[str, float]:
    values = np.asarray([result.confidence for result in results], dtype=np.float64)
    return {f"p{percentile}": float(np.quantile(values, percentile / 100)) for percentile in (10, 25, 50, 75, 90, 95, 99)}


def save_field_confusion(results: list[Result], labels: list[str], output_dir: Path) -> None:
    expected_labels = [label for label in labels if any(row.expected_label == label for row in results)]
    matrix = np.zeros((len(expected_labels), len(labels)), dtype=np.int64)
    expected_index = {label: index for index, label in enumerate(expected_labels)}
    predicted_index = {label: index for index, label in enumerate(labels)}
    for result in results:
        matrix[expected_index[str(result.expected_label)], predicted_index[result.predicted_label]] += 1
    with (output_dir / "field_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(expected_labels, matrix):
            writer.writerow([label, *row.tolist()])
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=np.float64),
        where=matrix.sum(axis=1, keepdims=True) != 0,
    )
    figure, axis = plt.subplots(figsize=(20, 14))
    rendered = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    figure.colorbar(rendered, ax=axis, fraction=0.025, pad=0.02)
    axis.set(
        title="PlantDoc field evaluation (all top-1 predictions, row-normalized)",
        xlabel="Predicted PlantVillage class",
        ylabel="PlantDoc mapped class",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(expected_labels)),
        xticklabels=labels,
        yticklabels=expected_labels,
    )
    plt.setp(axis.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(axis.get_yticklabels(), fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "field_confusion_matrix.png", dpi=180)
    plt.close(figure)


def field_metrics(
    results: list[Result],
    decode_errors: list[dict[str, str]],
    labels: list[str],
    manifest: dict[str, object],
    threshold: float,
    dataset_metadata: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    targets = [str(result.expected_label) for result in results]
    predictions = [result.predicted_label for result in results]
    accepted = [result for result in results if result.accepted]
    correct_accepted = sum(bool(result.correct) for result in accepted)
    per_class_rows: list[dict[str, object]] = []
    for label in labels:
        rows = [result for result in results if result.expected_label == label]
        if not rows:
            continue
        accepted_rows = [result for result in rows if result.accepted]
        per_class_rows.append(
            {
                "class": label,
                "support": len(rows),
                "accuracy": sum(bool(result.correct) for result in rows) / len(rows),
                "accepted": len(accepted_rows),
                "coverage": len(accepted_rows) / len(rows),
                "accepted_accuracy": (
                    sum(bool(result.correct) for result in accepted_rows) / len(accepted_rows)
                    if accepted_rows else None
                ),
                "quality_rejected": sum(not result.quality_ok for result in rows),
            }
        )
    with (output_dir / "field_per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)
    errors = sorted((row for row in results if not row.correct), key=lambda row: row.confidence, reverse=True)
    write_predictions(errors, output_dir / "field_top_confident_errors.csv")
    save_field_confusion(results, labels, output_dir)
    return {
        "production_approved": False,
        "candidate_model_version": manifest.get("model_version"),
        "evaluation_gate": "independent_field_images",
        "dataset": dataset_metadata,
        "threshold_fixed_before_evaluation": threshold,
        "discovered_images": len(results) + len(decode_errors),
        "evaluated_images": len(results),
        "decode_errors": decode_errors,
        "top1_accuracy_without_rejection": float(accuracy_score(targets, predictions)),
        "macro_f1_without_rejection": float(
            f1_score(targets, predictions, labels=sorted(set(targets)), average="macro", zero_division=0)
        ),
        "accepted": len(accepted),
        "rejected": len(results) - len(accepted),
        "coverage": len(accepted) / len(results),
        "accepted_accuracy": correct_accepted / len(accepted) if accepted else None,
        "quality_rejected": sum(not result.quality_ok for result in results),
        "threshold_rejected_after_quality": sum(result.quality_ok and not result.accepted for result in results),
        "confidence_quantiles": confidence_summary(results),
        "errors_without_rejection": sum(not bool(result.correct) for result in results),
        "accepted_errors": sum(result.accepted and not bool(result.correct) for result in results),
        "required_decision": "Manual failure review and predefined field/OOD gate assessment",
    }


def ood_metrics(
    results: list[Result],
    decode_errors: list[dict[str, str]],
    manifest: dict[str, object],
    threshold: float,
) -> dict[str, object]:
    accepted = [result for result in results if result.accepted]
    return {
        "production_approved": False,
        "candidate_model_version": manifest.get("model_version"),
        "evaluation_gate": "realistic_unsupported_images",
        "threshold_fixed_before_evaluation": threshold,
        "discovered_images": len(results) + len(decode_errors),
        "evaluated_images": len(results),
        "decode_errors": decode_errors,
        "accepted": len(accepted),
        "rejected": len(results) - len(accepted),
        "false_acceptance_rate": len(accepted) / len(results),
        "quality_rejected": sum(not result.quality_ok for result in results),
        "threshold_rejected_after_quality": sum(result.quality_ok and not result.accepted for result in results),
        "confidence_quantiles": confidence_summary(results),
        "accepted_prediction_counts": dict(Counter(result.predicted_label for result in accepted).most_common()),
        "required_decision": "Manual inspection of every accepted unsupported image",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels, manifest = verify_candidate(args.model, args.labels, args.manifest)
    manifest_threshold = manifest.get("acceptance_threshold")
    threshold = args.threshold if args.threshold is not None else manifest_threshold
    if not isinstance(threshold, (float, int)):
        raise RuntimeError("No numeric threshold was supplied or found in the candidate manifest.")
    if args.mode == "field":
        if args.mapping is None:
            raise RuntimeError("--mapping is required in field mode.")
        samples, dataset_metadata = field_samples(args.dataset_root, args.mapping, labels)
    else:
        samples = ood_samples(args.dataset_root)
        dataset_metadata = {}
    results, decode_errors = infer(samples, args.model, labels, float(threshold), args.batch_size)
    write_predictions(results, args.output_dir / f"{args.mode}_predictions.csv")
    if args.mode == "field":
        metrics = field_metrics(
            results, decode_errors, labels, manifest, float(threshold), dataset_metadata, args.output_dir
        )
        metrics_path = args.output_dir / "field_metrics.json"
    else:
        metrics = ood_metrics(results, decode_errors, manifest, float(threshold))
        metrics_path = args.output_dir / "ood_metrics.json"
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved evaluation evidence to {args.output_dir}")


if __name__ == "__main__":
    main()
