from __future__ import annotations

import os
import re
import unittest
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageFilter

from app.classification import parse_combined_label
from app.main import app
from app.model import classifier


FIXTURE_DIR = Path(os.environ.get("STAGE1_FIXTURE_DIR", ""))
FIXTURES_AVAILABLE = all(
    (FIXTURE_DIR / filename).is_file()
    for filename in ("diseased-apple-scab.jpg", "healthy-grape.jpg", "low-confidence.jpg")
)


def image_bytes(path: Path) -> bytes:
    return path.read_bytes()


class LabelParsingTests(unittest.TestCase):
    def test_diseased_label(self) -> None:
        parsed = parse_combined_label("Corn_(maize)___Northern_Leaf_Blight")
        self.assertEqual((parsed.crop, parsed.condition, parsed.disease, parsed.state),
                         ("Corn (maize)", "Diseased", "Northern Leaf Blight", "diseased"))

    def test_healthy_label(self) -> None:
        parsed = parse_combined_label("Grape___healthy")
        self.assertEqual((parsed.crop, parsed.condition, parsed.disease, parsed.state),
                         ("Grape", "Healthy", None, "healthy"))


class FrontendBackendContractTests(unittest.TestCase):
    def test_analysis_report_fields_exist_in_frontend_contract(self) -> None:
        backend_fields = set(app.openapi()["components"]["schemas"]["AnalysisReport"]["properties"])
        typescript = (Path(__file__).parents[2] / "src" / "types.ts").read_text(encoding="utf-8")
        interface = re.search(r"export interface AnalysisReport \{(.*?)\n\}", typescript, re.S)
        self.assertIsNotNone(interface)
        frontend_fields = set(re.findall(r"^\s*([A-Za-z][A-Za-z0-9]*)\??:", interface.group(1), re.M))
        self.assertEqual(backend_fields - frontend_fields, set())


@unittest.skipUnless(FIXTURES_AVAILABLE, "set STAGE1_FIXTURE_DIR to the pinned PlantDoc fixtures")
class CandidateV2HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)

    def analyze(self, payload: bytes, filename: str = "leaf.jpg"):
        return self.client.post("/api/v1/analyze", files={"image": (filename, payload, "image/jpeg")})

    def assert_missing_layers(self, report: dict) -> None:
        self.assertIsNone(report["diseaseRate"])
        self.assertIsNone(report["severity"])
        self.assertIsNone(report["healthScore"])
        self.assertEqual(report["observedSymptoms"], [])

    def test_accepted_diseased_report(self) -> None:
        response = self.analyze(image_bytes(FIXTURE_DIR / "diseased-apple-scab.jpg"))
        self.assertEqual(response.status_code, 200)
        report = response.json()
        self.assertEqual((report["state"], report["crop"], report["condition"], report["disease"]),
                         ("diseased", "Apple", "Diseased", "Apple scab"))
        self.assertGreater(report["confidence"], 0.998)
        self.assertEqual(report["topPredictions"][0]["label"], "Apple___Apple_scab")
        self.assertEqual(len(report["topPredictions"]), 5)
        self.assertEqual(report["modelStatus"], "experimental")
        if report["diseaseRate"] is None:
            self.assert_missing_layers(report)
        else:
            self.assertEqual(report["healthScore"], round(100 - report["diseaseRate"], 1))
            self.assertIn(report["severity"], ("Low", "Medium", "High"))
            self.assertTrue(report["observedSymptoms"])
        self.assertTrue(report["typicalSymptoms"])
        self.assertTrue(report["causes"])
        self.assertTrue(report["recommendations"])
        self.assertEqual(report["knowledgeSources"][0]["url"],
                         "https://ipm.ucanr.edu/agriculture/apple/apple-scab/")

    def test_accepted_healthy_report(self) -> None:
        response = self.analyze(image_bytes(FIXTURE_DIR / "healthy-grape.jpg"))
        self.assertEqual(response.status_code, 200)
        report = response.json()
        self.assertEqual((report["state"], report["crop"], report["condition"], report["disease"]),
                         ("healthy", "Grape", "Healthy", None))
        self.assertGreater(report["confidence"], 0.9999)
        self.assert_missing_layers(report)
        self.assertEqual(report["typicalSymptoms"], [])
        self.assertEqual(report["causes"], [])
        self.assertEqual(len(report["recommendations"]), 3)
        self.assertEqual(report["knowledgeSources"], [])

    def test_low_confidence_returns_unknown(self) -> None:
        response = self.analyze(image_bytes(FIXTURE_DIR / "low-confidence.jpg"))
        self.assertEqual(response.status_code, 200)
        report = response.json()
        self.assertEqual(report["state"], "unknown")
        self.assertLess(report["confidence"], 0.845)
        self.assertIn("confidence threshold", report["uncertaintyReason"])
        self.assertEqual(report["knowledgeSources"], [])

    def test_blurred_image_returns_unknown(self) -> None:
        source = Image.open(FIXTURE_DIR / "healthy-grape.jpg").convert("RGB").resize((512, 320))
        blurred = source.filter(ImageFilter.GaussianBlur(radius=24))
        payload = BytesIO()
        blurred.save(payload, format="JPEG", quality=90)
        response = self.analyze(payload.getvalue(), "blurred.jpg")
        self.assertEqual(response.status_code, 200)
        report = response.json()
        self.assertEqual(report["state"], "unknown")
        self.assertIn("too small or blurred", report["uncertaintyReason"])

    def test_filename_does_not_change_prediction(self) -> None:
        payload = image_bytes(FIXTURE_DIR / "diseased-apple-scab.jpg")
        first = self.analyze(payload, "apple_scab.jpg").json()
        second = self.analyze(payload, "healthy_tomato.jpg").json()
        keys = ("state", "crop", "condition", "disease", "confidence", "topPredictions")
        self.assertEqual({key: first[key] for key in keys}, {key: second[key] for key in keys})

    def test_model_unavailable_returns_503(self) -> None:
        saved_session, saved_labels = classifier._session, classifier._labels
        classifier._session, classifier._labels = None, []
        try:
            response = self.analyze(image_bytes(FIXTURE_DIR / "healthy-grape.jpg"))
        finally:
            classifier._session, classifier._labels = saved_session, saved_labels
        self.assertEqual(response.status_code, 503)
        self.assertIn("model is not available", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
