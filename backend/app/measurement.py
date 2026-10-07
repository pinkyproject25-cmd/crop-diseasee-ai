"""Evaluation-only visible-leaf affected-area arithmetic.

This module does not segment an uploaded image.  It turns two already validated,
pixel-aligned binary masks into the documented photographed-leaf measurement.
The live API must keep returning ``None`` until separate leaf and lesion models
pass the evaluation gate.
"""

from dataclasses import dataclass

import numpy as np


class MeasurementUnavailableError(ValueError):
    """Raised when reviewed masks cannot support an affected-area value."""


@dataclass(frozen=True)
class AffectedAreaMeasurement:
    leaf_pixels: int
    lesion_pixels_inside_leaf: int
    lesion_pixels_outside_leaf: int
    affected_area_percent: float


def _binary_mask(value: np.ndarray, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2:
        raise MeasurementUnavailableError(f"{name} must be a two-dimensional mask")
    if not np.issubdtype(mask.dtype, np.bool_) and not np.issubdtype(mask.dtype, np.number):
        raise MeasurementUnavailableError(f"{name} must contain binary numeric values")
    if not np.all(np.isfinite(mask)) or not np.all((mask == 0) | (mask == 1)):
        raise MeasurementUnavailableError(f"{name} must contain only 0 and 1")
    return mask.astype(bool, copy=False)


def measure_affected_area(leaf_mask: np.ndarray, lesion_mask: np.ndarray) -> AffectedAreaMeasurement:
    """Measure lesion pixels inside a visible-leaf mask.

    The denominator is the visible photographed leaf, not the whole image,
    plant, or field.  Lesion pixels outside the leaf are excluded and reported
    so an evaluator can detect mask-alignment or segmentation problems.
    """

    leaf = _binary_mask(leaf_mask, "leaf_mask")
    lesion = _binary_mask(lesion_mask, "lesion_mask")
    if leaf.shape != lesion.shape:
        raise MeasurementUnavailableError("leaf_mask and lesion_mask must have the same shape")

    leaf_pixels = int(np.count_nonzero(leaf))
    if leaf_pixels == 0:
        raise MeasurementUnavailableError("leaf_mask does not contain a visible leaf")

    inside = int(np.count_nonzero(lesion & leaf))
    outside = int(np.count_nonzero(lesion & ~leaf))
    return AffectedAreaMeasurement(
        leaf_pixels=leaf_pixels,
        lesion_pixels_inside_leaf=inside,
        lesion_pixels_outside_leaf=outside,
        affected_area_percent=100.0 * inside / leaf_pixels,
    )
