"""Tests for the offline paired-mask evidence gate."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.evaluate_segmentation import (
    SegmentationEvidenceError,
    evaluate,
    load_manifest,
)


FIELDS = [
    "sample_id", "image_path", "leaf_mask_path", "lesion_mask_path", "crop",
    "classifier_label", "group_id", "split", "source_url", "license_id",
    "rights_status", "attribution",
]


class SegmentationEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for directory in ("images", "leaf", "lesion", "predictions/leaf", "predictions/lesion"):
            (self.root / directory).mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_sample(self, sample_id: str = "sample-1", split: str = "test") -> Path:
        Image.new("RGB", (4, 4), "green").save(self.root / "images" / f"{sample_id}.png")
        leaf = np.zeros((4, 4), dtype=np.uint8)
        leaf[1:3, 1:3] = 255
        lesion = np.zeros((4, 4), dtype=np.uint8)
        lesion[1, 1] = 255
        Image.fromarray(leaf).save(self.root / "leaf" / f"{sample_id}.png")
        Image.fromarray(lesion).save(self.root / "lesion" / f"{sample_id}.png")
        manifest = self.root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow({
                "sample_id": sample_id,
                "image_path": f"images/{sample_id}.png",
                "leaf_mask_path": f"leaf/{sample_id}.png",
                "lesion_mask_path": f"lesion/{sample_id}.png",
                "crop": "Apple", "classifier_label": "Apple___Apple_scab",
                "group_id": "photo-1", "split": split,
                "source_url": "https://example.invalid/photo-1",
                "license_id": "OWNER-CONSENT",
                "rights_status": "approved_for_project",
                "attribution": "Test fixture",
            })
        return manifest

    def test_perfect_predictions_pass_the_engineering_gate(self) -> None:
        manifest = self.write_sample()
        for kind in ("leaf", "lesion"):
            source = self.root / kind / "sample-1.png"
            Image.open(source).save(self.root / "predictions" / kind / "sample-1.png")

        results, summary = evaluate(
            load_manifest(manifest), self.root / "predictions", minimum_crop_samples=1
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].true_affected_percent, 25.0)
        self.assertEqual(results[0].predicted_affected_percent, 25.0)
        self.assertTrue(summary["passed"])

    def test_missing_predictions_are_counted_as_abstentions(self) -> None:
        manifest = self.write_sample()

        results, summary = evaluate(load_manifest(manifest), self.root / "predictions")

        self.assertEqual(results, [])
        self.assertEqual(summary["coverage"], 0.0)
        self.assertFalse(summary["passed"])

    def test_gate_checks_each_crop_instead_of_hiding_abstentions(self) -> None:
        manifest = self.write_sample()
        with manifest.open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        second = dict(
            rows[0], sample_id="sample-2", crop="Grape",
            classifier_label="Grape___Black_rot", group_id="photo-2",
            image_path="images/sample-1.png", leaf_mask_path="leaf/sample-1.png",
            lesion_mask_path="lesion/sample-1.png",
        )
        with manifest.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows([rows[0], second])
        for kind in ("leaf", "lesion"):
            with Image.open(self.root / kind / "sample-1.png") as source:
                source.save(self.root / "predictions" / kind / "sample-1.png")

        _, summary = evaluate(
            load_manifest(manifest), self.root / "predictions",
            minimum_crop_samples=1,
        )

        self.assertEqual(summary["per_crop"]["Grape"]["coverage"], 0.0)
        self.assertFalse(summary["passed"])

    def test_unapproved_rights_are_rejected(self) -> None:
        manifest = self.write_sample()
        text = manifest.read_text(encoding="utf-8").replace(
            "approved_for_project", "unknown"
        )
        manifest.write_text(text, encoding="utf-8")

        with self.assertRaisesRegex(SegmentationEvidenceError, "not approved"):
            load_manifest(manifest)

    def test_group_leakage_is_rejected(self) -> None:
        manifest = self.write_sample()
        with manifest.open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        duplicate = dict(rows[0], sample_id="sample-2", split="train")
        with manifest.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows([rows[0], duplicate])

        with self.assertRaisesRegex(SegmentationEvidenceError, "leaks"):
            load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
