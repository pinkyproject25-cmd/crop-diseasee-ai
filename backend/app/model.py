import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import settings
from .schemas import Prediction


class ModelUnavailableError(RuntimeError):
    pass


class InvalidImageError(ValueError):
    pass


@dataclass
class ModelOutput:
    predictions: list[Prediction]
    width: int
    height: int
    sharpness: float


class CropClassifier:
    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._labels: list[str] = []

    @property
    def ready(self) -> bool:
        return self._session is not None and bool(self._labels)

    def load(self) -> None:
        self._labels = []
        self._session = None
        required_paths = (settings.model_path, settings.labels_path, settings.model_manifest_path)
        if not all(path.exists() for path in required_paths):
            return

        if self._sha256(settings.model_manifest_path) != settings.model_manifest_sha256:
            raise RuntimeError("Candidate manifest SHA-256 verification failed")

        manifest = json.loads(settings.model_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise RuntimeError("candidate_manifest.json must contain a JSON object")
        if manifest.get("model_version") != settings.model_version:
            raise RuntimeError("Candidate model version does not match MODEL_VERSION")
        if manifest.get("production_approved") is not False or manifest.get("candidate_only") is not True:
            raise RuntimeError("Only the recorded experimental Candidate-v2 manifest is accepted")
        if float(manifest.get("acceptance_threshold", -1)) != settings.acceptance_threshold:
            raise RuntimeError("Candidate threshold does not match ACCEPTANCE_THRESHOLD")

        model_hash = self._sha256(settings.model_path)
        labels_hash = self._sha256(settings.labels_path)
        manifest_artifacts = manifest.get("artifacts", {})
        if model_hash != settings.model_sha256 or manifest_artifacts.get("crop_classifier.onnx") != model_hash:
            raise RuntimeError("Candidate ONNX SHA-256 verification failed")
        if labels_hash != settings.labels_sha256 or manifest_artifacts.get("labels.json") != labels_hash:
            raise RuntimeError("Candidate labels SHA-256 verification failed")

        labels = json.loads(settings.labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise RuntimeError("labels.json must contain a JSON array of class names")
        if len(labels) != 38 or len(set(labels)) != 38:
            raise RuntimeError("Candidate-v2 must contain exactly 38 unique ordered labels")
        session = ort.InferenceSession(str(settings.model_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "image" or inputs[0].shape[1:] != [3, settings.input_size, settings.input_size]:
            raise RuntimeError("Candidate ONNX input must have shape [batch, 3, 224, 224]")
        if len(outputs) != 1 or outputs[0].name != "logits" or outputs[0].shape[1:] != [38]:
            raise RuntimeError("Candidate ONNX output must have shape [batch, 38]")
        self._labels = labels
        self._session = session

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _decode(image_bytes: bytes) -> Image.Image:
        try:
            image = Image.open(BytesIO(image_bytes))
            image.verify()
            image = Image.open(BytesIO(image_bytes))
            return ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("The uploaded file is not a readable image.") from exc

    @staticmethod
    def _sharpness(image: Image.Image) -> float:
        gray = np.asarray(image.resize((256, 256)).convert("L"), dtype=np.float32)
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        return float((gx.var() + gy.var()) / 2)

    def predict(self, image_bytes: bytes) -> ModelOutput:
        if not self.ready or self._session is None:
            raise ModelUnavailableError("The validated crop model is not installed.")
        image = self._decode(image_bytes)
        sharpness = self._sharpness(image)
        resized = ImageOps.fit(image, (settings.input_size, settings.input_size))
        tensor = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...].astype(np.float32)
        input_name = self._session.get_inputs()[0].name
        logits = np.asarray(self._session.run(None, {input_name: tensor})[0]).reshape(-1)
        probabilities = np.exp(logits - logits.max())
        probabilities = probabilities / probabilities.sum()
        indices = np.argsort(probabilities)[::-1][:5]
        predictions = [
            Prediction(label=self._labels[int(index)], probability=float(probabilities[int(index)]))
            for index in indices
        ]
        return ModelOutput(
            predictions=predictions,
            width=image.width,
            height=image.height,
            sharpness=sharpness,
        )


classifier = CropClassifier()
