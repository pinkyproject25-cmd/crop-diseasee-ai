"""Train and evaluate a PlantVillage classifier without filename leakage.

This produces candidate artifacts only. A candidate is not production-approved until
the field-image, unsupported-image, severity, and agricultural review gates described
in docs/IMPLEMENTATION_STATUS.md are complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
import torch
import torchvision
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import datasets as vision_datasets
from torchvision import models, transforms
from tqdm.auto import tqdm


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str
    epochs: int = 12
    warmup_epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 7386
    val_fold: int = 0
    calibration_fold: int = 1
    test_fold: int = 2
    split_folds: int = 10
    ood_limit: int = 10_000
    resume: bool = True


class LeafDataset(TorchDataset[tuple[Tensor, int]]):
    def __init__(self, source: Dataset, indices: Iterable[int], transform: transforms.Compose):
        self.source = source
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.source[int(self.indices[index])]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        return self.transform(image.convert("RGB")), int(row["label"])


class CalibratedModel(nn.Module):
    def __init__(self, model: nn.Module, temperature: float):
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image) / self.temperature.clamp_min(0.05)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/classifier")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7386)
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--calibration-fold", type=int, default=1)
    parser.add_argument("--test-fold", type=int, default=2)
    parser.add_argument("--split-folds", type=int, default=10)
    parser.add_argument("--ood-limit", type=int, default=10_000)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    return TrainConfig(**vars(parser.parse_args()))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.72, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(18),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.16, hue=0.03),
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


def group_split(
    source: Dataset,
    seed: int,
    validation_fold: int,
    calibration_fold: int,
    test_fold: int,
    split_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(source["label"], dtype=np.int64)
    groups = np.asarray(source["leaf_id"], dtype=str)
    if np.any(groups == "unknown"):
        raise RuntimeError("PlantVillage rows without leaf_id cannot be used for leakage-safe splitting.")
    group_labels: dict[str, int] = {}
    for group, label in zip(groups, labels):
        previous = group_labels.setdefault(group, int(label))
        if previous != int(label):
            raise RuntimeError(f"Leaf group {group!r} appears under more than one class.")

    if split_folds < 4:
        raise RuntimeError("At least four folds are required for train, validation, calibration, and test partitions.")
    splitter = StratifiedGroupKFold(n_splits=split_folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(labels.shape[0]), labels, groups))
    validation_fold %= len(splits)
    calibration_fold %= len(splits)
    test_fold %= len(splits)
    if len({validation_fold, calibration_fold, test_fold}) != 3:
        raise RuntimeError("Validation, calibration, and test folds must be different.")

    validation_indices = splits[validation_fold][1]
    calibration_indices = splits[calibration_fold][1]
    test_indices = splits[test_fold][1]
    held_out = np.concatenate([validation_indices, calibration_indices, test_indices])
    train_indices = np.setdiff1d(np.arange(labels.shape[0]), held_out, assume_unique=False)

    index_sets = [set(groups[indices]) for indices in (train_indices, validation_indices, calibration_indices, test_indices)]
    for left in range(len(index_sets)):
        for right in range(left + 1, len(index_sets)):
            if index_sets[left].intersection(index_sets[right]):
                raise RuntimeError("Leaf-group leakage detected across custom dataset splits.")
    expected_classes = set(range(int(labels.max()) + 1))
    for name, indices in zip(("training", "validation", "calibration", "test"), (train_indices, validation_indices, calibration_indices, test_indices)):
        missing = expected_classes.difference(map(int, labels[indices]))
        if missing:
            raise RuntimeError(f"The {name} split is missing class indices: {sorted(missing)}")
    return train_indices, validation_indices, calibration_indices, test_indices


def published_split_overlap(source_train: Dataset, source_test: Dataset) -> int:
    train_groups = set(map(str, source_train["leaf_id"]))
    test_groups = set(map(str, source_test["leaf_id"]))
    if "unknown" in train_groups or "unknown" in test_groups:
        raise RuntimeError("PlantVillage rows without leaf_id cannot be used for independent evaluation.")
    return len(train_groups.intersection(test_groups))


def build_model(class_count: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
    return model


def make_class_weights(labels: np.ndarray, class_count: int, device: torch.device) -> Tensor:
    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("Every advertised class must be present in the training fold.")
    weights = counts.sum() / (class_count * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def set_feature_training(model: nn.Module, enabled: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = enabled


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
    for images, labels in tqdm(loader, desc="train", leave=False):
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
    return total_loss / len(loader.dataset)


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor]:
    # Calibration optimizes a temperature parameter against these logits.
    # Tensors created in inference mode cannot later be used in an autograd
    # graph, even when the logits themselves are treated as constants.
    model.eval()
    logits: list[Tensor] = []
    labels: list[Tensor] = []
    for images, batch_labels in tqdm(loader, desc="evaluate", leave=False):
        logits.append(model(images.to(device)).cpu())
        labels.append(batch_labels.cpu())
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits: Tensor, labels: Tensor) -> float:
    log_temperature = nn.Parameter(torch.zeros(1))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=80, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if not mask.any():
            continue
        bin_accuracy = (predictions[mask] == labels[mask]).mean()
        ece += mask.mean() * abs(float(bin_accuracy) - float(confidence[mask].mean()))
    return float(ece)


@torch.inference_mode()
def ood_confidences(
    model: nn.Module,
    eval_transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
    workers: int,
    limit: int,
) -> np.ndarray:
    dataset = vision_datasets.CIFAR100(root=".cache/ood", train=False, download=True, transform=eval_transform)
    if limit < len(dataset):
        generator = torch.Generator().manual_seed(7386)
        dataset, _ = torch.utils.data.random_split(dataset, [limit, len(dataset) - limit], generator=generator)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    model.eval()
    values: list[np.ndarray] = []
    for images, _ in tqdm(loader, desc="OOD sanity check", leave=False):
        probabilities = torch.softmax(model(images.to(device)), dim=1)
        values.append(probabilities.max(dim=1).values.cpu().numpy())
    return np.concatenate(values)


def choose_provisional_threshold(
    validation_probabilities: np.ndarray,
    validation_labels: np.ndarray,
    ood_confidence: np.ndarray,
) -> dict[str, float | str]:
    confidence = validation_probabilities.max(axis=1)
    predictions = validation_probabilities.argmax(axis=1)
    candidates: list[dict[str, float]] = []
    for threshold in np.linspace(0.40, 0.99, 119):
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        accepted_precision = float((predictions[accepted] == validation_labels[accepted]).mean()) if accepted.any() else 0.0
        false_acceptance = float((ood_confidence >= threshold).mean())
        if coverage >= 0.50 and accepted_precision >= 0.95 and false_acceptance <= 0.01:
            candidates.append(
                {
                    "threshold": float(threshold),
                    "validation_coverage": coverage,
                    "accepted_precision": accepted_precision,
                    "cifar100_false_acceptance": false_acceptance,
                }
            )
    if not candidates:
        fallback = float(np.clip(max(0.72, np.quantile(ood_confidence, 0.99)), 0.72, 0.99))
        return {
            "status": "failed_provisional_gate",
            "threshold": fallback,
            "validation_coverage": float((confidence >= fallback).mean()),
            "accepted_precision": 0.0,
            "cifar100_false_acceptance": float((ood_confidence >= fallback).mean()),
        }
    best = max(candidates, key=lambda item: item["validation_coverage"])
    return {"status": "provisional_ood_sanity_check", **best}


def evaluate_split(logits: Tensor, labels: Tensor, class_names: list[str], temperature: float) -> dict[str, object]:
    probabilities = torch.softmax(logits / temperature, dim=1).numpy()
    targets = labels.numpy()
    predictions = probabilities.argmax(axis=1)
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
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "ece": expected_calibration_error(probabilities, targets),
        "per_class": report,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(len(class_names)))).tolist(),
        "probabilities": probabilities,
        "targets": targets,
    }


def calibration_metrics(logits: Tensor, labels: Tensor, temperature: float) -> dict[str, float]:
    targets = labels.numpy()
    raw_probabilities = torch.softmax(logits, dim=1).numpy()
    calibrated_probabilities = torch.softmax(logits / temperature, dim=1).numpy()
    raw_nll = float(nn.functional.cross_entropy(logits, labels))
    calibrated_nll = float(nn.functional.cross_entropy(logits / temperature, labels))
    return {
        "temperature": temperature,
        "raw_nll": raw_nll,
        "calibrated_nll": calibrated_nll,
        "raw_ece": expected_calibration_error(raw_probabilities, targets),
        "calibrated_ece": expected_calibration_error(calibrated_probabilities, targets),
    }


def selective_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accepted = confidence >= threshold
    accepted_count = int(accepted.sum())
    return {
        "threshold": threshold,
        "coverage": float(accepted.mean()),
        "accepted_count": accepted_count,
        "rejected_count": int((~accepted).sum()),
        "accepted_accuracy": float((predictions[accepted] == labels[accepted]).mean()) if accepted_count else 0.0,
    }


def save_confusion_matrix(
    matrix: list[list[int]],
    class_names: list[str],
    csv_path: Path,
    image_path: Path,
) -> None:
    values = np.asarray(matrix, dtype=np.int64)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *class_names])
        for class_name, row in zip(class_names, values):
            writer.writerow([class_name, *row.tolist()])

    row_totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, row_totals, out=np.zeros_like(values, dtype=np.float64), where=row_totals != 0)
    figure, axis = plt.subplots(figsize=(18, 16))
    rendered = axis.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(rendered, ax=axis, fraction=0.03, pad=0.02, label="Fraction of actual class")
    axis.set(
        title="PlantVillage custom group-independent test confusion matrix (row-normalized)",
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


def save_per_class_metrics(per_class: dict[str, object], class_names: list[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "precision", "recall", "f1-score", "support"])
        writer.writeheader()
        for class_name in class_names:
            row = per_class[class_name]
            if not isinstance(row, dict):
                raise RuntimeError(f"Missing per-class metrics for {class_name}.")
            writer.writerow({"class": class_name, **{key: row[key] for key in writer.fieldnames[1:]}})


def save_training_history(history: list[dict[str, float | int]], path: Path) -> None:
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], marker="o")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Weighted cross-entropy")
    axes[1].plot(epochs, [float(row["validation_accuracy"]) for row in history], marker="o", label="Accuracy")
    axes[1].plot(epochs, [float(row["validation_macro_f1"]) for row in history], marker="o", label="Macro-F1")
    axes[1].set(title="Model-selection validation", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def verify_onnx_export(model: nn.Module, images: Tensor, path: Path, device: torch.device) -> dict[str, float | int]:
    model.eval()
    with torch.inference_mode():
        torch_logits = model(images.to(device)).cpu().numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_logits = np.asarray(session.run(None, {input_name: images.cpu().numpy()})[0])
    max_absolute_error = float(np.max(np.abs(torch_logits - onnx_logits)))
    agreement = int((torch_logits.argmax(axis=1) == onnx_logits.argmax(axis=1)).sum())
    if agreement != images.shape[0] or max_absolute_error > 1e-4:
        raise RuntimeError(
            f"ONNX parity check failed: {agreement}/{images.shape[0]} classes agree, max error {max_absolute_error}."
        )
    return {
        "examples": int(images.shape[0]),
        "top1_agreement": agreement,
        "max_absolute_logit_error": max_absolute_error,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Datasets 4.x no longer executes Hub dataset scripts, and the repository's
    # mixed-case name prevents automatic discovery of plant_village.py. Fetch
    # the reviewed loader explicitly so image, label, and leaf_id are preserved.
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
    source_split_overlap = published_split_overlap(source_train, source_test)
    source = concatenate_datasets([source_train, source_test])
    label_feature = source.features["label"]
    class_names = list(label_feature.names)
    if len(class_names) != 38:
        raise RuntimeError(f"Expected 38 PlantVillage classes; received {len(class_names)}.")

    train_indices, validation_indices, calibration_indices, test_indices = group_split(
        source,
        config.seed,
        config.val_fold,
        config.calibration_fold,
        config.test_fold,
        config.split_folds,
    )
    train_transform, eval_transform = image_transforms()
    train_dataset = LeafDataset(source, train_indices, train_transform)
    validation_dataset = LeafDataset(source, validation_indices, eval_transform)
    calibration_dataset = LeafDataset(source, calibration_indices, eval_transform)
    test_dataset = LeafDataset(source, test_indices, eval_transform)
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=train_generator, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    calibration_loader = DataLoader(calibration_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = build_model(len(class_names)).to(device)
    set_feature_training(model, False)
    labels_for_weights = np.asarray(source.select(train_indices.tolist())["label"], dtype=np.int64)
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(labels_for_weights, len(class_names), device), label_smoothing=0.04)
    optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    best_macro_f1 = -math.inf
    best_path = output_dir / "best_model.pt"
    checkpoint_path = output_dir / "last_checkpoint.pt"
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    features_unfrozen = False

    if config.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint.get("config", {})
        immutable_keys = ("epochs", "warmup_epochs", "batch_size", "learning_rate", "weight_decay", "seed", "val_fold", "calibration_fold", "test_fold", "split_folds")
        mismatches = [key for key in immutable_keys if checkpoint_config.get(key) != getattr(config, key)]
        if mismatches:
            raise RuntimeError(f"Resume checkpoint configuration differs for: {', '.join(mismatches)}. Use a new output directory or --no-resume.")
        features_unfrozen = bool(checkpoint["features_unfrozen"])
        set_feature_training(model, features_unfrozen)
        model.load_state_dict(checkpoint["model_state"])
        phase_learning_rate = config.learning_rate / 3 if features_unfrozen else config.learning_rate
        optimizer = torch.optim.AdamW(
            model.parameters() if features_unfrozen else filter(lambda parameter: parameter.requires_grad, model.parameters()),
            lr=phase_learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.epochs - config.warmup_epochs) if features_unfrozen else max(1, config.epochs),
        )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        history = checkpoint["history"]
        best_macro_f1 = float(checkpoint["best_macro_f1"])
        start_epoch = int(checkpoint["completed_epoch"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
        train_generator.set_state(checkpoint["train_generator_state"])
        print(f"Resuming after epoch {start_epoch} from {checkpoint_path}")

    for epoch in range(start_epoch, config.epochs):
        if epoch == config.warmup_epochs and not features_unfrozen:
            set_feature_training(model, True)
            features_unfrozen = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate / 3, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs - epoch))
        loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        validation_logits, validation_labels = collect_logits(model, validation_loader, device)
        validation_predictions = validation_logits.argmax(dim=1).numpy()
        macro_f1 = float(f1_score(validation_labels.numpy(), validation_predictions, average="macro"))
        accuracy = float(accuracy_score(validation_labels.numpy(), validation_predictions))
        history.append({"epoch": epoch + 1, "train_loss": loss, "validation_accuracy": accuracy, "validation_macro_f1": macro_f1})
        print(history[-1])
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), best_path)
        scheduler.step()
        torch.save(
            {
                "completed_epoch": epoch + 1,
                "features_unfrozen": features_unfrozen,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
                "best_macro_f1": best_macro_f1,
                "config": asdict(config),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "train_generator_state": train_generator.get_state(),
            },
            checkpoint_path,
        )

    if not best_path.exists():
        raise RuntimeError("No best-model checkpoint exists; training did not complete a validation epoch.")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    validation_logits, validation_labels = collect_logits(model, validation_loader, device)
    calibration_logits, calibration_labels = collect_logits(model, calibration_loader, device)
    temperature = fit_temperature(calibration_logits, calibration_labels)
    test_logits, test_labels = collect_logits(model, test_loader, device)
    validation_metrics = evaluate_split(validation_logits, validation_labels, class_names, temperature)
    calibration_split_metrics = evaluate_split(calibration_logits, calibration_labels, class_names, temperature)
    test_metrics = evaluate_split(test_logits, test_labels, class_names, temperature)
    calibrated = CalibratedModel(model, temperature).to(device).eval()
    ood_confidence = ood_confidences(calibrated, eval_transform, device, config.batch_size, config.num_workers, config.ood_limit)
    threshold = choose_provisional_threshold(
        calibration_split_metrics["probabilities"],
        calibration_split_metrics["targets"],
        ood_confidence,
    )
    threshold_value = float(threshold["threshold"])
    independent_test_selective = selective_metrics(
        test_metrics["probabilities"],
        test_metrics["targets"],
        threshold_value,
    )
    calibration_evidence = calibration_metrics(calibration_logits, calibration_labels, temperature)
    validation_metrics.pop("probabilities")
    validation_metrics.pop("targets")
    calibration_split_metrics.pop("probabilities")
    calibration_split_metrics.pop("targets")
    test_metrics.pop("probabilities")
    test_metrics.pop("targets")

    version = datetime.now(timezone.utc).strftime("plantvillage-mnv3-%Y%m%dT%H%M%SZ")
    write_json(output_dir / "labels.json", class_names)
    metrics = {
        "model_version": version,
        "production_approved": False,
        "approval_blockers": [
            "Independent field-photo evaluation has not passed.",
            "Unsupported crop and non-leaf rejection has not been validated beyond the CIFAR-100 sanity check.",
            "Disease-area segmentation and severity have not passed validation.",
            "Disease information and recommendations have not completed agricultural review.",
        ],
        "config": asdict(config),
        "software": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "onnxruntime": ort.__version__,
        },
        "dataset": {
            "name": "mohanty/PlantVillage",
            "loader_configuration": "default (color)",
            "revision": dataset_revision,
            "license": dataset_info.card_data.get("license") if dataset_info.card_data else None,
            "classes": len(class_names),
            "source_examples": len(source),
            "source_train_examples": len(source_train),
            "source_test_examples": len(source_test),
            "published_split_leaf_group_overlap": source_split_overlap,
            "custom_split_folds": config.split_folds,
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            "calibration_examples": len(calibration_dataset),
            "test_examples": len(test_dataset),
            "leaf_group_overlap_across_all_splits": 0,
            "training_class_counts": dict(sorted(Counter(labels_for_weights.tolist()).items())),
        },
        "calibration": calibration_evidence,
        "acceptance_threshold": threshold,
        "independent_test_selective_metrics": independent_test_selective,
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
        "validation": validation_metrics,
        "calibration_split": calibration_split_metrics,
        "test": test_metrics,
        "history": history,
    }

    save_training_history(history, output_dir / "training_history.png")
    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        output_dir / "test_confusion_matrix.csv",
        output_dir / "test_confusion_matrix.png",
    )
    save_per_class_metrics(test_metrics["per_class"], class_names, output_dir / "test_per_class_metrics.csv")

    dummy = torch.zeros(1, 3, 224, 224, device=device)
    onnx_path = output_dir / "crop_classifier.onnx"
    torch.onnx.export(
        calibrated,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    parity_images, _ = next(iter(test_loader))
    metrics["onnx_verification"] = verify_onnx_export(calibrated, parity_images[:8], onnx_path, device)
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    manifest = {
        "model_version": version,
        "production_approved": False,
        "candidate_only": True,
        "acceptance_threshold": threshold_value,
        "artifacts": {
            "crop_classifier.onnx": sha256(onnx_path),
            "labels.json": sha256(output_dir / "labels.json"),
            "metrics.json": sha256(metrics_path),
        },
        "required_next_gate": "Independent field-photo and realistic unsupported-image evaluation",
    }
    write_json(output_dir / "candidate_manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "model_version": version, "threshold": threshold, "test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
