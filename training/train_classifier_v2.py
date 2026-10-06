"""Train Candidate v2 with controlled and reviewed field-domain images.

Candidate v2 starts from the rejected Candidate-v1 classifier weights, then
fine-tunes on a fixed mixture of leakage-controlled PlantVillage training data
and the manually reviewed PlantDoc training pool. PlantDoc test is never read
by this script. The result is always a non-production candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision
from datasets import DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset as TorchDataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm.auto import tqdm

from train_classifier import (
    MEAN,
    STD,
    CalibratedModel,
    LeafDataset,
    calibration_metrics,
    collect_logits,
    evaluate_split,
    group_split,
    make_class_weights,
    ood_confidences,
    published_split_overlap,
    save_per_class_metrics,
    seed_everything,
    selective_metrics,
    sha256,
    verify_onnx_export,
    write_json,
)


PLANTDOC_REVISION = "5467f6012d78d1c446145d5f582da6096f852ae8"
MANIFEST_FIELDS = [
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


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str
    plantdoc_root: str
    plantdoc_manifest: str
    plantdoc_review: str
    initial_checkpoint: str
    initial_labels: str
    initial_candidate_manifest: str
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    plantdoc_sample_fraction: float = 0.25
    num_workers: int = 2
    seed: int = 7386
    pv_val_fold: int = 0
    pv_calibration_fold: int = 1
    pv_test_fold: int = 2
    pv_split_folds: int = 10
    plantdoc_val_fold: int = 0
    plantdoc_calibration_fold: int = 1
    plantdoc_split_folds: int = 10
    near_duplicate_distance: int = 4
    ood_limit: int = 10_000
    resume: bool = True
    preflight_only: bool = False


class PlantDocDataset(TorchDataset[tuple[Tensor, int]]):
    def __init__(
        self,
        root: Path,
        rows: list[dict[str, str]],
        indices: np.ndarray,
        class_to_index: dict[str, int],
        transform: transforms.Compose,
    ):
        self.root = root
        self.rows = rows
        self.indices = np.asarray(indices, dtype=np.int64)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.rows[int(self.indices[index])]
        with Image.open(self.root / row["relative_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        return self.transform(image), self.class_to_index[row["target_label"]]


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plantdoc-root", required=True)
    parser.add_argument("--plantdoc-manifest", required=True)
    parser.add_argument("--plantdoc-review", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--initial-labels", required=True)
    parser.add_argument("--initial-candidate-manifest", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--plantdoc-sample-fraction", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7386)
    parser.add_argument("--pv-val-fold", type=int, default=0)
    parser.add_argument("--pv-calibration-fold", type=int, default=1)
    parser.add_argument("--pv-test-fold", type=int, default=2)
    parser.add_argument("--pv-split-folds", type=int, default=10)
    parser.add_argument("--plantdoc-val-fold", type=int, default=0)
    parser.add_argument("--plantdoc-calibration-fold", type=int, default=1)
    parser.add_argument("--plantdoc-split-folds", type=int, default=10)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    parser.add_argument("--ood-limit", type=int, default=10_000)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--preflight-only", action="store_true")
    return TrainConfig(**vars(parser.parse_args()))


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


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
            return completed.stdout.strip() if completed.returncode == 0 else None
    return None


def require_clean_repository(path: Path) -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=path,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Could not verify repository cleanliness at {path}.")
    if completed.stdout.strip():
        raise RuntimeError("PlantDoc repository has modified tracked files; use a clean pinned checkout.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plantdoc_inputs(
    config: TrainConfig,
    verify_image_hashes: bool,
) -> tuple[list[dict[str, str]], dict[str, object], list[str], dict[str, object], dict[str, str]]:
    root = Path(config.plantdoc_root).resolve()
    manifest_path = Path(config.plantdoc_manifest).resolve()
    review_path = Path(config.plantdoc_review).resolve()
    labels_path = Path(config.initial_labels).resolve()
    candidate_path = Path(config.initial_candidate_manifest).resolve()
    checkpoint_path = Path(config.initial_checkpoint).resolve()

    actual_revision = repository_revision(root)
    if actual_revision != PLANTDOC_REVISION:
        raise RuntimeError(
            f"PlantDoc revision mismatch: expected {PLANTDOC_REVISION}, received {actual_revision or 'unknown'}."
        )
    require_clean_repository(root)
    review = read_json(review_path)
    if not isinstance(review, dict):
        raise RuntimeError("PlantDoc manifest review must be a JSON object.")
    if review.get("production_approved") is not False or review.get("training_authorized") is not True:
        raise RuntimeError("PlantDoc review does not authorize Candidate-v2 experimentation.")
    if review.get("dataset_revision") != PLANTDOC_REVISION:
        raise RuntimeError("PlantDoc review records a different dataset revision.")
    manifest_hash = file_sha256(manifest_path)
    if manifest_hash != review.get("reviewed_manifest_sha256"):
        raise RuntimeError("Reviewed PlantDoc manifest SHA-256 does not match manifest_review.json.")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise RuntimeError(f"Unexpected PlantDoc manifest columns: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != review.get("reviewed_manifest_rows"):
        raise RuntimeError("PlantDoc manifest row count differs from manifest_review.json.")

    paths: set[str] = set()
    file_hashes: set[str] = set()
    pixel_hashes: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        relative = Path(row["relative_path"])
        if row["split"] != "train" or relative.parts[:1] != ("train",):
            raise RuntimeError(f"Non-training PlantDoc row at line {row_number}: {relative}")
        if row["include"].lower() != "true" or row["exclusion_reason"]:
            raise RuntimeError(f"Excluded PlantDoc row reached reviewed manifest at line {row_number}.")
        if row["relative_path"] in paths or row["file_sha256"] in file_hashes or row["pixel_sha256"] in pixel_hashes:
            raise RuntimeError(f"Duplicate survives in reviewed PlantDoc manifest at line {row_number}.")
        image_path = (root / relative).resolve()
        if root not in image_path.parents or not image_path.is_file():
            raise RuntimeError(f"Missing or unsafe PlantDoc image path: {relative}")
        if verify_image_hashes and file_sha256(image_path) != row["file_sha256"]:
            raise RuntimeError(f"PlantDoc file SHA-256 mismatch: {relative}")
        paths.add(row["relative_path"])
        file_hashes.add(row["file_sha256"])
        pixel_hashes.add(row["pixel_sha256"])

    labels = read_json(labels_path)
    if not isinstance(labels, list) or len(labels) != 38 or not all(isinstance(item, str) for item in labels):
        raise RuntimeError("Candidate-v1 labels.json must contain the 38 ordered class names.")
    unknown_targets = sorted({row["target_label"] for row in rows} - set(labels))
    if unknown_targets:
        raise RuntimeError(f"PlantDoc manifest contains targets outside Candidate-v1 labels: {unknown_targets}")
    candidate = read_json(candidate_path)
    if not isinstance(candidate, dict) or candidate.get("production_approved") is not False:
        raise RuntimeError("Initial candidate manifest is missing or incorrectly marked production-approved.")
    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, dict) or artifacts.get("labels.json") != file_sha256(labels_path):
        raise RuntimeError("Initial labels do not match the Candidate-v1 manifest.")

    input_hashes = {
        "plantdoc_manifest": manifest_hash,
        "plantdoc_review": file_sha256(review_path),
        "initial_checkpoint": file_sha256(checkpoint_path),
        "initial_labels": file_sha256(labels_path),
        "initial_candidate_manifest": file_sha256(candidate_path),
    }
    return rows, review, labels, candidate, input_hashes


def perceptual_groups(rows: list[dict[str, str]], maximum_distance: int) -> np.ndarray:
    if not 0 <= maximum_distance <= 64:
        raise RuntimeError("near-duplicate-distance must be between 0 and 64.")
    values = [int(row["dhash64"], 16) for row in rows]
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if (values[left] ^ values[right]).bit_count() <= maximum_distance:
                union(left, right)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        members[find(index)].append(index)
    ordered_roots = sorted(members, key=lambda root: min(rows[index]["relative_path"] for index in members[root]))
    root_names = {root: f"pd-near-{number:04d}" for number, root in enumerate(ordered_roots, start=1)}
    groups = np.asarray([root_names[find(index)] for index in range(len(rows))], dtype=str)

    group_labels: dict[str, set[str]] = defaultdict(set)
    for group, row in zip(groups, rows):
        group_labels[str(group)].add(row["target_label"])
    conflicts = {group: sorted(labels) for group, labels in group_labels.items() if len(labels) > 1}
    if conflicts:
        raise RuntimeError(f"Reviewed PlantDoc manifest still has cross-label near-duplicate groups: {conflicts}")
    return groups


def split_plantdoc(
    rows: list[dict[str, str]],
    groups: np.ndarray,
    config: TrainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([row["target_label"] for row in rows], dtype=str)
    if config.plantdoc_split_folds < 3:
        raise RuntimeError("PlantDoc requires at least three folds.")
    splitter = StratifiedGroupKFold(
        n_splits=config.plantdoc_split_folds,
        shuffle=True,
        random_state=config.seed,
    )
    splits = list(splitter.split(np.zeros(len(rows)), labels, groups))
    val_fold = config.plantdoc_val_fold % len(splits)
    calibration_fold = config.plantdoc_calibration_fold % len(splits)
    if val_fold == calibration_fold:
        raise RuntimeError("PlantDoc validation and calibration folds must differ.")
    validation_indices = splits[val_fold][1]
    calibration_indices = splits[calibration_fold][1]
    held_out = np.concatenate([validation_indices, calibration_indices])
    train_indices = np.setdiff1d(np.arange(len(rows)), held_out, assume_unique=False)

    split_groups = [set(groups[indices]) for indices in (train_indices, validation_indices, calibration_indices)]
    for left in range(len(split_groups)):
        for right in range(left + 1, len(split_groups)):
            if split_groups[left].intersection(split_groups[right]):
                raise RuntimeError("Perceptual PlantDoc group leakage detected across Candidate-v2 splits.")
    missing_train = sorted(set(labels) - set(labels[train_indices]))
    if missing_train:
        raise RuntimeError(f"PlantDoc training split is missing mapped classes: {missing_train}")
    return train_indices, validation_indices, calibration_indices


def write_plantdoc_split_manifest(
    path: Path,
    rows: list[dict[str, str]],
    groups: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    calibration_indices: np.ndarray,
) -> None:
    assignments: dict[int, str] = {}
    for name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("calibration", calibration_indices),
    ):
        assignments.update({int(index): name for index in indices})
    fields = [*MANIFEST_FIELDS, "candidate_v2_split", "perceptual_group"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({**row, "candidate_v2_split": assignments[index], "perceptual_group": groups[index]})


def training_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(22),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.25),
            transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.20, hue=0.04),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_transform, eval_transform


def build_candidate_model(class_count: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
    return model


def load_initial_weights(model: nn.Module, path: Path) -> None:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict) and "model_state" in value:
        value = value["model_state"]
    if not isinstance(value, dict):
        raise RuntimeError("Initial checkpoint does not contain a model state dictionary.")
    model.load_state_dict(value, strict=True)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    examples = 0
    for images, labels in tqdm(loader, desc="mixed-domain train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * images.size(0)
        examples += images.size(0)
    return total_loss / examples


def macro_f1_for_present_classes(logits: Tensor, labels: Tensor) -> float:
    targets = labels.numpy()
    predictions = logits.argmax(dim=1).numpy()
    return float(
        f1_score(
            targets,
            predictions,
            labels=np.unique(targets),
            average="macro",
            zero_division=0,
        )
    )


def fit_balanced_temperature(
    pv_logits: Tensor,
    pv_labels: Tensor,
    field_logits: Tensor,
    field_labels: Tensor,
) -> float:
    log_temperature = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=80, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = 0.5 * nn.functional.cross_entropy(pv_logits / temperature, pv_labels)
        loss += 0.5 * nn.functional.cross_entropy(field_logits / temperature, field_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def wilson_lower(correct: int, total: int, z: float = 1.959963984540054) -> float:
    if total == 0:
        return 0.0
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
    return (centre - margin) / denominator


def acceptance_at(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accepted = confidence >= threshold
    accepted_count = int(accepted.sum())
    correct = int((predictions[accepted] == labels[accepted]).sum()) if accepted_count else 0
    return {
        "coverage": float(accepted.mean()),
        "accepted_count": accepted_count,
        "correct_count": correct,
        "accepted_accuracy": correct / accepted_count if accepted_count else 0.0,
        "accepted_accuracy_wilson_lower_95": wilson_lower(correct, accepted_count),
    }


def choose_threshold(
    pv_probabilities: np.ndarray,
    pv_labels: np.ndarray,
    field_probabilities: np.ndarray,
    field_labels: np.ndarray,
    ood_confidence: np.ndarray,
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    evaluated: list[dict[str, object]] = []
    for threshold in np.linspace(0.40, 0.99, 119):
        pv = acceptance_at(pv_probabilities, pv_labels, float(threshold))
        field = acceptance_at(field_probabilities, field_labels, float(threshold))
        false_acceptance = float((ood_confidence >= threshold).mean())
        row = {
            "threshold": float(threshold),
            "plantvillage_calibration": pv,
            "plantdoc_calibration": field,
            "cifar100_false_acceptance": false_acceptance,
        }
        evaluated.append(row)
        if (
            pv["coverage"] >= 0.50
            and pv["accepted_accuracy"] >= 0.95
            and field["coverage"] >= 0.30
            and field["accepted_accuracy"] >= 0.95
            and field["accepted_accuracy_wilson_lower_95"] >= 0.90
            and false_acceptance <= 0.01
        ):
            candidates.append(row)
    if candidates:
        selected = max(candidates, key=lambda row: float(row["plantdoc_calibration"]["coverage"]))
        return {"status": "provisional_mixed_domain_calibration_gate", **selected}

    fallback_threshold = float(np.clip(max(0.72, np.quantile(ood_confidence, 0.99)), 0.72, 0.99))
    closest = min(evaluated, key=lambda row: abs(float(row["threshold"]) - fallback_threshold))
    return {
        "status": "failed_provisional_mixed_domain_gate",
        "note": "No threshold satisfied every predefined controlled, field-calibration, and CIFAR-100 constraint.",
        **closest,
    }


def field_metrics(logits: Tensor, labels: Tensor, class_names: list[str], temperature: float) -> dict[str, object]:
    probabilities = torch.softmax(logits / temperature, dim=1).numpy()
    targets = labels.numpy()
    predictions = probabilities.argmax(axis=1)
    present = sorted(set(map(int, targets)))
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1_present_classes": float(
            f1_score(targets, predictions, labels=present, average="macro", zero_division=0)
        ),
        "present_class_count": len(present),
        "present_classes": [class_names[index] for index in present],
        "per_class": report,
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=list(range(len(class_names))),
        ).tolist(),
        "probabilities": probabilities,
        "targets": targets,
    }


def save_confusion(
    matrix: list[list[int]],
    class_names: list[str],
    csv_path: Path,
    image_path: Path,
    title: str,
) -> None:
    values = np.asarray(matrix, dtype=np.int64)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *class_names])
        for class_name, row in zip(class_names, values):
            writer.writerow([class_name, *row.tolist()])
    row_totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, row_totals, out=np.zeros_like(values, dtype=float), where=row_totals != 0)
    figure, axis = plt.subplots(figsize=(18, 16))
    rendered = axis.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(rendered, ax=axis, fraction=0.03, pad=0.02, label="Fraction of actual class")
    axis.set(
        title=title,
        xlabel="Predicted class",
        ylabel="Actual class",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=90, ha="center", fontsize=7)
    plt.setp(axis.get_yticklabels(), fontsize=7)
    figure.tight_layout()
    figure.savefig(image_path, dpi=180)
    plt.close(figure)


def save_history(history: list[dict[str, float | int]], path: Path) -> None:
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], marker="o")
    axes[0].set(title="Mixed-domain training loss", xlabel="Epoch", ylabel="Weighted cross-entropy")
    axes[1].plot(epochs, [float(row["pv_validation_macro_f1"]) for row in history], marker="o", label="PlantVillage")
    axes[1].plot(epochs, [float(row["plantdoc_validation_macro_f1"]) for row in history], marker="o", label="PlantDoc internal")
    axes[1].plot(epochs, [float(row["selection_score"]) for row in history], marker="o", label="Mean selection score")
    axes[1].set(title="Model-selection validation", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    config = parse_args()
    if not 0.05 <= config.plantdoc_sample_fraction <= 0.50:
        raise RuntimeError("plantdoc-sample-fraction must be between 0.05 and 0.50.")
    seed_everything(config.seed)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = output_dir / "input_preflight.json"

    previous_preflight = read_json(preflight_path) if preflight_path.exists() else None
    verify_images = config.preflight_only or not isinstance(previous_preflight, dict)
    rows, review, initial_labels, initial_candidate, input_hashes = load_plantdoc_inputs(config, verify_images)
    groups = perceptual_groups(rows, config.near_duplicate_distance)
    pd_train_indices, pd_validation_indices, pd_calibration_indices = split_plantdoc(rows, groups, config)
    split_manifest_path = output_dir / "plantdoc_candidate_v2_splits.csv"
    write_plantdoc_split_manifest(
        split_manifest_path,
        rows,
        groups,
        pd_train_indices,
        pd_validation_indices,
        pd_calibration_indices,
    )
    split_hash = file_sha256(split_manifest_path)
    preflight = {
        "status": "passed",
        "production_approved": False,
        "dataset_revision": PLANTDOC_REVISION,
        "verified_all_plantdoc_file_hashes": verify_images,
        "input_hashes": input_hashes,
        "plantdoc_split_manifest_sha256": split_hash,
        "plantdoc_rows": len(rows),
        "perceptual_groups": len(set(groups)),
        "plantdoc_train_rows": len(pd_train_indices),
        "plantdoc_validation_rows": len(pd_validation_indices),
        "plantdoc_calibration_rows": len(pd_calibration_indices),
        "plantdoc_train_classes": len({rows[index]["target_label"] for index in pd_train_indices}),
        "plantdoc_validation_classes": len({rows[index]["target_label"] for index in pd_validation_indices}),
        "plantdoc_calibration_classes": len({rows[index]["target_label"] for index in pd_calibration_indices}),
        "plantdoc_test_accessed": False,
    }
    if not verify_images:
        expected = dict(preflight)
        expected["verified_all_plantdoc_file_hashes"] = True
        if previous_preflight != expected:
            raise RuntimeError("Existing preflight evidence does not match the current Candidate-v2 inputs.")
        preflight = expected
    initial_model = build_candidate_model(len(initial_labels))
    load_initial_weights(initial_model, Path(config.initial_checkpoint))
    write_json(preflight_path, preflight)
    if config.preflight_only:
        print(json.dumps(preflight, indent=2))
        print("Candidate-v2 preflight passed. No training was performed.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Candidate-v2 training requires a CUDA runtime. Run --preflight-only on CPU.")
    print(f"Using {device}")

    dataset_info = HfApi().dataset_info("mohanty/PlantVillage")
    dataset_revision = dataset_info.sha
    loader_path = hf_hub_download(
        repo_id="mohanty/PlantVillage",
        filename="plant_village.py",
        repo_type="dataset",
        revision=dataset_revision,
    )
    data: DatasetDict = load_dataset(loader_path, "default", trust_remote_code=True)
    source_train = data["train"]
    source_test = data["test"]
    source_overlap = published_split_overlap(source_train, source_test)
    source = concatenate_datasets([source_train, source_test])
    class_names = list(source.features["label"].names)
    if class_names != initial_labels:
        raise RuntimeError("Candidate-v1 label order differs from the pinned PlantVillage loader.")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    unknown_targets = sorted({row["target_label"] for row in rows} - set(class_names))
    if unknown_targets:
        raise RuntimeError(f"PlantDoc manifest contains unmapped target labels: {unknown_targets}")

    pv_train_indices, pv_validation_indices, pv_calibration_indices, pv_test_indices = group_split(
        source,
        config.seed,
        config.pv_val_fold,
        config.pv_calibration_fold,
        config.pv_test_fold,
        config.pv_split_folds,
    )
    train_transform, eval_transform = training_transforms()
    pv_train_dataset = LeafDataset(source, pv_train_indices, train_transform)
    pv_validation_dataset = LeafDataset(source, pv_validation_indices, eval_transform)
    pv_calibration_dataset = LeafDataset(source, pv_calibration_indices, eval_transform)
    pv_test_dataset = LeafDataset(source, pv_test_indices, eval_transform)
    plantdoc_root = Path(config.plantdoc_root).resolve()
    pd_train_dataset = PlantDocDataset(plantdoc_root, rows, pd_train_indices, class_to_index, train_transform)
    pd_validation_dataset = PlantDocDataset(plantdoc_root, rows, pd_validation_indices, class_to_index, eval_transform)
    pd_calibration_dataset = PlantDocDataset(plantdoc_root, rows, pd_calibration_indices, class_to_index, eval_transform)

    mixed_train_dataset = ConcatDataset([pv_train_dataset, pd_train_dataset])
    pv_mass = (1.0 - config.plantdoc_sample_fraction) / len(pv_train_dataset)
    pd_mass = config.plantdoc_sample_fraction / len(pd_train_dataset)
    sample_weights = torch.tensor(
        [pv_mass] * len(pv_train_dataset) + [pd_mass] * len(pd_train_dataset),
        dtype=torch.double,
    )
    train_generator = torch.Generator().manual_seed(config.seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(pv_train_dataset),
        replacement=True,
        generator=train_generator,
    )
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(mixed_train_dataset, sampler=sampler, **loader_options)
    pv_validation_loader = DataLoader(pv_validation_dataset, shuffle=False, **loader_options)
    pv_calibration_loader = DataLoader(pv_calibration_dataset, shuffle=False, **loader_options)
    pv_test_loader = DataLoader(pv_test_dataset, shuffle=False, **loader_options)
    pd_validation_loader = DataLoader(pd_validation_dataset, shuffle=False, **loader_options)
    pd_calibration_loader = DataLoader(pd_calibration_dataset, shuffle=False, **loader_options)

    model = initial_model.to(device)
    pv_training_labels = np.asarray(source.select(pv_train_indices.tolist())["label"], dtype=np.int64)
    criterion = nn.CrossEntropyLoss(
        weight=make_class_weights(pv_training_labels, len(class_names), device),
        label_smoothing=0.04,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_path = output_dir / "best_model.pt"
    checkpoint_path = output_dir / "last_checkpoint.pt"
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    baseline_validation: dict[str, float]
    best_score: float

    if config.resume and checkpoint_path.exists():
        if not best_path.exists():
            raise RuntimeError("Resume checkpoint exists but best_model.pt is missing; use a new output directory.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        immutable_keys = (
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "plantdoc_sample_fraction",
            "seed",
            "pv_val_fold",
            "pv_calibration_fold",
            "pv_test_fold",
            "pv_split_folds",
            "plantdoc_val_fold",
            "plantdoc_calibration_fold",
            "plantdoc_split_folds",
            "near_duplicate_distance",
        )
        checkpoint_config = checkpoint.get("config", {})
        mismatches = [key for key in immutable_keys if checkpoint_config.get(key) != getattr(config, key)]
        if checkpoint.get("input_hashes") != input_hashes or checkpoint.get("split_hash") != split_hash:
            mismatches.append("input_evidence")
        if mismatches:
            raise RuntimeError(
                f"Resume checkpoint differs for: {', '.join(mismatches)}. Use a new output directory."
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        history = checkpoint["history"]
        best_score = float(checkpoint["best_score"])
        baseline_validation = checkpoint["baseline_validation"]
        start_epoch = int(checkpoint["completed_epoch"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
        train_generator.set_state(checkpoint["train_generator_state"])
        print(f"Resuming after epoch {start_epoch} from {checkpoint_path}")
    else:
        pv_baseline_logits, pv_baseline_labels = collect_logits(model, pv_validation_loader, device)
        pd_baseline_logits, pd_baseline_labels = collect_logits(model, pd_validation_loader, device)
        baseline_pv_f1 = macro_f1_for_present_classes(pv_baseline_logits, pv_baseline_labels)
        baseline_pd_f1 = macro_f1_for_present_classes(pd_baseline_logits, pd_baseline_labels)
        baseline_score = 0.5 * (baseline_pv_f1 + baseline_pd_f1)
        baseline_validation = {
            "plantvillage_macro_f1": baseline_pv_f1,
            "plantdoc_macro_f1_present_classes": baseline_pd_f1,
            "selection_score": baseline_score,
        }
        best_score = baseline_score
        torch.save(model.state_dict(), best_path)

    for epoch in range(start_epoch, config.epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        pv_logits, pv_labels = collect_logits(model, pv_validation_loader, device)
        pd_logits, pd_labels = collect_logits(model, pd_validation_loader, device)
        pv_f1 = macro_f1_for_present_classes(pv_logits, pv_labels)
        pd_f1 = macro_f1_for_present_classes(pd_logits, pd_labels)
        selection_score = 0.5 * (pv_f1 + pd_f1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss,
                "pv_validation_accuracy": float(accuracy_score(pv_labels.numpy(), pv_logits.argmax(dim=1).numpy())),
                "pv_validation_macro_f1": pv_f1,
                "plantdoc_validation_accuracy": float(
                    accuracy_score(pd_labels.numpy(), pd_logits.argmax(dim=1).numpy())
                ),
                "plantdoc_validation_macro_f1": pd_f1,
                "selection_score": selection_score,
            }
        )
        print(history[-1])
        if selection_score > best_score:
            best_score = selection_score
            torch.save(model.state_dict(), best_path)
        scheduler.step()
        torch.save(
            {
                "completed_epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
                "best_score": best_score,
                "baseline_validation": baseline_validation,
                "config": asdict(config),
                "input_hashes": input_hashes,
                "split_hash": split_hash,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": torch.cuda.get_rng_state_all(),
                "train_generator_state": train_generator.get_state(),
            },
            checkpoint_path,
        )

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    pv_validation_logits, pv_validation_labels = collect_logits(model, pv_validation_loader, device)
    pd_validation_logits, pd_validation_labels = collect_logits(model, pd_validation_loader, device)
    pv_calibration_logits, pv_calibration_labels = collect_logits(model, pv_calibration_loader, device)
    pd_calibration_logits, pd_calibration_labels = collect_logits(model, pd_calibration_loader, device)
    temperature = fit_balanced_temperature(
        pv_calibration_logits,
        pv_calibration_labels,
        pd_calibration_logits,
        pd_calibration_labels,
    )
    pv_test_logits, pv_test_labels = collect_logits(model, pv_test_loader, device)

    pv_validation_metrics = evaluate_split(pv_validation_logits, pv_validation_labels, class_names, temperature)
    pv_calibration_metrics = evaluate_split(pv_calibration_logits, pv_calibration_labels, class_names, temperature)
    pv_test_metrics = evaluate_split(pv_test_logits, pv_test_labels, class_names, temperature)
    pd_validation_metrics = field_metrics(pd_validation_logits, pd_validation_labels, class_names, temperature)
    pd_calibration_metrics = field_metrics(pd_calibration_logits, pd_calibration_labels, class_names, temperature)
    calibrated_model = CalibratedModel(model, temperature).to(device).eval()
    ood_confidence = ood_confidences(
        calibrated_model,
        eval_transform,
        device,
        config.batch_size,
        config.num_workers,
        config.ood_limit,
    )
    threshold = choose_threshold(
        pv_calibration_metrics["probabilities"],
        pv_calibration_metrics["targets"],
        pd_calibration_metrics["probabilities"],
        pd_calibration_metrics["targets"],
        ood_confidence,
    )
    threshold_value = float(threshold["threshold"])
    pv_test_selective = selective_metrics(
        pv_test_metrics["probabilities"],
        pv_test_metrics["targets"],
        threshold_value,
    )
    balanced_calibration = {
        "objective": "Equal mean NLL weight for PlantVillage and reviewed PlantDoc calibration splits.",
        "temperature": temperature,
        "plantvillage": calibration_metrics(pv_calibration_logits, pv_calibration_labels, temperature),
        "plantdoc": calibration_metrics(pd_calibration_logits, pd_calibration_labels, temperature),
    }

    for result in (
        pv_validation_metrics,
        pv_calibration_metrics,
        pv_test_metrics,
        pd_validation_metrics,
        pd_calibration_metrics,
    ):
        result.pop("probabilities")
        result.pop("targets")

    version = datetime.now(timezone.utc).strftime("mixed-pv-plantdoc-mnv3-%Y%m%dT%H%M%SZ")
    write_json(output_dir / "labels.json", class_names)
    metrics = {
        "model_version": version,
        "production_approved": False,
        "approval_blockers": [
            "Candidate v2 has not passed a different untouched field-photo gate.",
            "Realistic unsupported-crop and non-leaf rejection has not passed the predefined gate.",
            "Disease-area segmentation and severity have not passed validation.",
            "Disease information and recommendations have not completed agricultural review.",
        ],
        "config": asdict(config),
        "software": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "onnxruntime": ort.__version__,
        },
        "initial_candidate": {
            "model_version": initial_candidate.get("model_version"),
            "production_approved": initial_candidate.get("production_approved"),
            "input_hashes": input_hashes,
        },
        "plantvillage": {
            "name": "mohanty/PlantVillage",
            "revision": dataset_revision,
            "classes": len(class_names),
            "source_examples": len(source),
            "published_split_leaf_group_overlap": source_overlap,
            "train_examples": len(pv_train_dataset),
            "validation_examples": len(pv_validation_dataset),
            "calibration_examples": len(pv_calibration_dataset),
            "controlled_regression_examples": len(pv_test_dataset),
            "leaf_group_overlap_across_all_splits": 0,
            "training_class_counts": dict(sorted(Counter(pv_training_labels.tolist()).items())),
            "warning": "This locked fold was already reported for Candidate v1 and is a controlled regression benchmark, not a new untouched test set.",
        },
        "plantdoc": {
            "name": "Cropped-PlantDoc classification dataset",
            "revision": PLANTDOC_REVISION,
            "license": "CC-BY-4.0",
            "reviewed_manifest_sha256": review["reviewed_manifest_sha256"],
            "split_manifest_sha256": split_hash,
            "reviewed_examples": len(rows),
            "perceptual_groups": len(set(groups)),
            "train_examples": len(pd_train_dataset),
            "validation_examples": len(pd_validation_dataset),
            "calibration_examples": len(pd_calibration_dataset),
            "test_accessed": False,
            "warning": "Validation is used for model selection and calibration is used for temperature/threshold fitting; neither is an independent approval set.",
        },
        "mixed_training": {
            "samples_per_epoch": len(pv_train_dataset),
            "target_plantvillage_fraction": 1.0 - config.plantdoc_sample_fraction,
            "target_plantdoc_fraction": config.plantdoc_sample_fraction,
        },
        "baseline_validation": baseline_validation,
        "best_selection_score": best_score,
        "calibration": balanced_calibration,
        "acceptance_threshold": threshold,
        "plantvillage_validation": pv_validation_metrics,
        "plantdoc_internal_validation": pd_validation_metrics,
        "plantdoc_internal_calibration": pd_calibration_metrics,
        "plantvillage_controlled_regression": pv_test_metrics,
        "plantvillage_controlled_regression_selective": pv_test_selective,
        "ood_sanity_check": {
            "dataset": "CIFAR-100 test images (provisional non-leaf proxy only)",
            "examples": int(ood_confidence.size),
            "confidence_quantiles": {
                "p50": float(np.quantile(ood_confidence, 0.50)),
                "p90": float(np.quantile(ood_confidence, 0.90)),
                "p95": float(np.quantile(ood_confidence, 0.95)),
                "p99": float(np.quantile(ood_confidence, 0.99)),
            },
            "false_acceptance_at_selected_threshold": float((ood_confidence >= threshold_value).mean()),
        },
        "history": history,
    }

    save_history(history, output_dir / "training_history.png")
    save_confusion(
        pv_test_metrics["confusion_matrix"],
        class_names,
        output_dir / "plantvillage_regression_confusion_matrix.csv",
        output_dir / "plantvillage_regression_confusion_matrix.png",
        "Candidate v2 PlantVillage controlled regression benchmark (row-normalized)",
    )
    save_per_class_metrics(
        pv_test_metrics["per_class"],
        class_names,
        output_dir / "plantvillage_regression_per_class_metrics.csv",
    )
    save_confusion(
        pd_validation_metrics["confusion_matrix"],
        class_names,
        output_dir / "plantdoc_validation_confusion_matrix.csv",
        output_dir / "plantdoc_validation_confusion_matrix.png",
        "Candidate v2 PlantDoc internal validation (row-normalized; not approval evidence)",
    )
    save_per_class_metrics(
        pd_validation_metrics["per_class"],
        class_names,
        output_dir / "plantdoc_validation_per_class_metrics.csv",
    )

    dummy = torch.zeros(1, 3, 224, 224, device=device)
    onnx_path = output_dir / "crop_classifier.onnx"
    torch.onnx.export(
        calibrated_model,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    parity_images, _ = next(iter(pv_test_loader))
    metrics["onnx_verification"] = verify_onnx_export(
        calibrated_model,
        parity_images[:8],
        onnx_path,
        device,
    )
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    candidate_manifest = {
        "model_version": version,
        "production_approved": False,
        "candidate_only": True,
        "acceptance_threshold": threshold_value,
        "artifacts": {
            "crop_classifier.onnx": sha256(onnx_path),
            "labels.json": sha256(output_dir / "labels.json"),
            "metrics.json": sha256(metrics_path),
        },
        "required_next_gate": "A different untouched field-photo set and a realistic unsupported-image suite.",
    }
    write_json(output_dir / "candidate_manifest.json", candidate_manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "model_version": version,
                "best_selection_score": best_score,
                "threshold": threshold,
                "plantvillage_regression_accuracy": pv_test_metrics["accuracy"],
                "plantvillage_regression_macro_f1": pv_test_metrics["macro_f1"],
                "production_approved": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
