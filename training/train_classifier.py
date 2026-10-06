"""Train and evaluate a PlantVillage classifier without filename leakage.

This produces candidate artifacts only. A candidate is not production-approved until
the field-image, unsupported-image, severity, and agricultural review gates described
in docs/IMPLEMENTATION_STATUS.md are complete.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
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
    ood_limit: int = 10_000


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
    parser.add_argument("--ood-limit", type=int, default=10_000)
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


def group_split(source: Dataset, seed: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(source["label"], dtype=np.int64)
    groups = np.asarray(source["leaf_id"], dtype=str)
    if np.any(groups == "unknown"):
        raise RuntimeError("PlantVillage rows without leaf_id cannot be used for leakage-safe splitting.")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(labels.shape[0]), labels, groups))
    train_indices, validation_indices = splits[fold % len(splits)]
    overlap = set(groups[train_indices]).intersection(groups[validation_indices])
    if overlap:
        raise RuntimeError(f"Leaf-group leakage detected for {len(overlap)} groups.")
    return train_indices, validation_indices


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


@torch.inference_mode()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor]:
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
    loader_path = hf_hub_download(
        repo_id="mohanty/PlantVillage",
        filename="plant_village.py",
        repo_type="dataset",
    )
    data: DatasetDict = load_dataset(loader_path, "default", trust_remote_code=True)
    source_train = data["train"]
    source_test = data["test"]
    label_feature = source_train.features["label"]
    class_names = list(label_feature.names)
    if len(class_names) != 38:
        raise RuntimeError(f"Expected 38 PlantVillage classes; received {len(class_names)}.")

    train_indices, validation_indices = group_split(source_train, config.seed, config.val_fold)
    train_transform, eval_transform = image_transforms()
    train_dataset = LeafDataset(source_train, train_indices, train_transform)
    validation_dataset = LeafDataset(source_train, validation_indices, eval_transform)
    test_dataset = LeafDataset(source_test, range(len(source_test)), eval_transform)
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = build_model(len(class_names)).to(device)
    set_feature_training(model, False)
    labels_for_weights = np.asarray(source_train.select(train_indices.tolist())["label"], dtype=np.int64)
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(labels_for_weights, len(class_names), device), label_smoothing=0.04)
    optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    best_macro_f1 = -math.inf
    best_path = output_dir / "best_model.pt"
    history: list[dict[str, float | int]] = []

    for epoch in range(config.epochs):
        if epoch == config.warmup_epochs:
            set_feature_training(model, True)
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

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    validation_logits, validation_labels = collect_logits(model, validation_loader, device)
    temperature = fit_temperature(validation_logits, validation_labels)
    test_logits, test_labels = collect_logits(model, test_loader, device)
    validation_metrics = evaluate_split(validation_logits, validation_labels, class_names, temperature)
    test_metrics = evaluate_split(test_logits, test_labels, class_names, temperature)
    calibrated = CalibratedModel(model, temperature).to(device).eval()
    ood_confidence = ood_confidences(calibrated, eval_transform, device, config.batch_size, config.num_workers, config.ood_limit)
    threshold = choose_provisional_threshold(
        validation_metrics.pop("probabilities"),
        validation_metrics.pop("targets"),
        ood_confidence,
    )
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
        "dataset": {
            "name": "mohanty/PlantVillage",
            "configuration": "color",
            "classes": len(class_names),
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            "test_examples": len(test_dataset),
            "leaf_group_overlap": 0,
            "training_class_counts": dict(sorted(Counter(labels_for_weights.tolist()).items())),
        },
        "temperature": temperature,
        "acceptance_threshold": threshold,
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
    }
    write_json(output_dir / "metrics.json", metrics)

    dummy = torch.zeros(1, 3, 224, 224, device=device)
    torch.onnx.export(
        calibrated,
        dummy,
        output_dir / "crop_classifier.onnx",
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(json.dumps({"output_dir": str(output_dir), "model_version": version, "threshold": threshold, "test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
