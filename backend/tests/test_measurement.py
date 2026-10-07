"""Tests for evaluation-only visible-leaf affected-area arithmetic."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.measurement import MeasurementUnavailableError, measure_affected_area


class AffectedAreaMeasurementTests(unittest.TestCase):
    def test_uses_visible_leaf_as_denominator(self) -> None:
        leaf = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.uint8)
        lesion = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.uint8)

        result = measure_affected_area(leaf, lesion)

        self.assertEqual(result.leaf_pixels, 4)
        self.assertEqual(result.lesion_pixels_inside_leaf, 1)
        self.assertEqual(result.affected_area_percent, 25.0)

    def test_excludes_and_reports_lesion_pixels_outside_leaf(self) -> None:
        leaf = np.array([[1, 1, 0], [1, 1, 0]], dtype=bool)
        lesion = np.array([[1, 0, 1], [0, 0, 1]], dtype=bool)

        result = measure_affected_area(leaf, lesion)

        self.assertEqual(result.lesion_pixels_inside_leaf, 1)
        self.assertEqual(result.lesion_pixels_outside_leaf, 2)
        self.assertEqual(result.affected_area_percent, 25.0)

    def test_rejects_empty_leaf_mask(self) -> None:
        with self.assertRaisesRegex(MeasurementUnavailableError, "visible leaf"):
            measure_affected_area(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaisesRegex(MeasurementUnavailableError, "same shape"):
            measure_affected_area(np.ones((2, 2)), np.ones((3, 3)))

    def test_rejects_non_binary_or_non_finite_masks(self) -> None:
        for bad_mask in (np.array([[0.0, 0.5]]), np.array([[0.0, np.nan]])):
            with self.subTest(mask=bad_mask):
                with self.assertRaises(MeasurementUnavailableError):
                    measure_affected_area(np.ones_like(bad_mask), bad_mask)


if __name__ == "__main__":
    unittest.main()
