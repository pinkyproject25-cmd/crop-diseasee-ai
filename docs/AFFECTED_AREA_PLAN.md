# Stage 2 affected-area measurement

Status: evaluation arithmetic implemented; segmentation data/model not approved;
live output remains unavailable.

## Defined output

For a supported, sufficiently clear photograph of one visible leaf:

`affected_area_percent = 100 × lesion pixels inside visible leaf / visible leaf pixels`

This is the affected portion of the photographed leaf only. It is not disease
prevalence in the plant or field. Classifier confidence is never substituted
for affected area. If the leaf or lesion cannot be segmented reliably, the API
must return `null` and the UI must show **Unable to estimate**.

## Implemented intermediate capability

`backend/app/measurement.py` implements the formula for two already validated,
pixel-aligned binary masks. It validates dimensionality, shape, binary values,
finite values, and a non-empty leaf denominator. Lesion pixels outside the leaf
are excluded from the percentage and reported for evaluation diagnostics.

The helper is intentionally not connected to `/api/v1/analyze`. Unit tests
cover the denominator, out-of-leaf pixels, empty masks, mismatched shapes, and
non-binary/non-finite masks. This is arithmetic only; it does not infer masks.

## Dataset evidence checked on 2026-10-07

- [PlantSeg v7](https://zenodo.org/records/17719108) supplies disease-region
  masks. The accompanying Scientific Data paper states CC BY-NC 4.0, but the
  Zenodo record displays no licence value. More importantly, PlantSeg does not
  establish paired whole-leaf masks for the same images, so it cannot provide
  this project's leaf-area denominator by itself.
- [Plant leaves image segmentation dataset](https://zenodo.org/records/14707857)
  contains beet and rye image/mask collections for leaf and disease
  segmentation. Its directory description shows separate task collections,
  its crops are outside this classifier's 14 groups, and the Zenodo licence
  field is blank. It is not approved for this project.
- [CropAndWeedAndLeaf](https://zenodo.org/records/20116408) supplies leaf-instance
  masks, including maize, squash, potato, and soybean, but no disease-region
  masks. Separate datasets cannot be combined as though their masks annotate
  the same photograph.

## Gate before live integration

Acquire or create rights-cleared, same-image leaf and lesion masks for relevant
supported crop/disease classes. Freeze group-disjoint train/validation/test
partitions and predefined acceptance thresholds. Record leaf and lesion
Dice/IoU, absolute affected-area error in percentage points, per-crop results,
and abstention coverage on untouched field photographs.

Severity thresholds and health-score semantics need their own documented and
validated rules. Until those gates pass, `diseaseRate`, `severity`, and
`healthScore` remain `null` in live reports.
