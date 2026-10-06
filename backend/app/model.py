import json
from dataclasses import dataclass
from io import BytesIO

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
        if not settings.model_path.exists() or not settings.labels_path.exists():
            return
        labels = json.loads(settings.labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise RuntimeError("labels.json must contain a JSON array of class names")
        session = ort.InferenceSession(str(settings.model_path), providers=["CPUExecutionProvider"])
        self._labels = labels
        self._session = session

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
