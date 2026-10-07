"""Run with: python -m unittest discover -s backend/tests -p test_knowledge.py"""

import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.knowledge import ENTRIES, get_knowledge
from app.main import app
from app.model import ModelOutput
from app.schemas import Prediction


COVERED_LABELS = {
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Pepper,_bell___Bacterial_spot",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
}


def image_payload() -> bytes:
    payload = BytesIO()
    Image.new("RGB", (300, 300), "green").save(payload, format="PNG")
    return payload.getvalue()


def output(label: str, probability: float = 0.95) -> ModelOutput:
    return ModelOutput(
        predictions=[Prediction(label=label, probability=probability)],
        width=300,
        height=300,
        sharpness=100.0,
    )


class KnowledgeTests(unittest.TestCase):
    def test_only_reviewed_exact_labels_return_content(self):
        self.assertEqual(set(ENTRIES), COVERED_LABELS)
        self.assertEqual(len(COVERED_LABELS), 26)
        self.assertIsNone(get_knowledge("Apple___healthy"))
        self.assertIsNone(get_knowledge("Tomato___healthy"))
        self.assertIsNone(get_knowledge("Apple___Apple_scab_other"))

    def test_every_entry_has_traceable_general_content(self):
        for label, entry in ENTRIES.items():
            with self.subTest(label=label):
                self.assertTrue(entry.typical_symptoms)
                self.assertTrue(entry.causes)
                self.assertTrue(entry.recommendations)
                self.assertTrue(entry.source_url.startswith("https://"))
                self.assertNotIn("observed", " ".join(entry.typical_symptoms).lower())

class KnowledgeApiTests(unittest.TestCase):
    def analyze(self, mocked_output: ModelOutput) -> dict:
        with (
            patch("app.main.classifier.predict", return_value=mocked_output),
            patch("app.main.fetch_weather", new=AsyncMock(return_value=None)),
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/analyze",
                files={"image": ("renamed-file.png", image_payload(), "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_exact_covered_disease_receives_source_linked_general_knowledge(self) -> None:
        report = self.analyze(output("Apple___Cedar_apple_rust"))

        self.assertEqual(report["disease"], "Cedar apple rust")
        self.assertEqual(report["observedSymptoms"], [])
        self.assertTrue(report["typicalSymptoms"])
        self.assertTrue(report["causes"])
        self.assertTrue(report["recommendations"])
        self.assertEqual(len(report["knowledgeSources"]), 1)
        self.assertIn("extension.umn.edu", report["knowledgeSources"][0]["url"])

    def test_healthy_report_never_receives_disease_knowledge(self) -> None:
        report = self.analyze(output("Apple___healthy"))

        self.assertEqual(report["state"], "healthy")
        self.assertEqual(report["disease"], None)
        self.assertEqual(report["typicalSymptoms"], [])
        self.assertEqual(report["causes"], [])
        self.assertEqual(len(report["recommendations"]), 3)
        self.assertEqual(report["knowledgeSources"], [])

    def test_low_confidence_report_never_receives_disease_knowledge(self) -> None:
        report = self.analyze(output("Apple___Cedar_apple_rust", probability=0.5))

        self.assertEqual(report["state"], "unknown")
        self.assertEqual(report["typicalSymptoms"], [])
        self.assertEqual(report["causes"], [])
        self.assertEqual(report["knowledgeSources"], [])


if __name__ == "__main__":
    unittest.main()
