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
  masks and its Scientific Data paper explicitly states CC BY-NC 4.0. The same
  paper also explicitly says leaf boundaries are not annotated. PlantSeg is
  therefore a legally plausible non-commercial academic starting point for
  lesion masks, but it cannot provide the leaf-area denominator by itself.
- [PlantVillage Apple Synthetic Segmentation Dataset](https://zenodo.org/records/18659728)
  supplies 75 apple images with manually drawn disease masks, but the Zenodo
  record's licence field is blank. It must not be used until the depositor or a
  rights owner supplies explicit reuse terms.
- [LDD grape disease instance segmentation](https://zenodo.org/records/10573036)
  has potentially useful field annotations, but its files are access-restricted
  and limited to non-commercial use. Access and exact mask semantics would have
  to be approved before use.
- [Plant leaves image segmentation dataset](https://zenodo.org/records/14707857)
  contains beet and rye image/mask collections for leaf and disease
  segmentation. Its directory description shows separate task collections,
  its crops are outside this classifier's 14 groups, and the Zenodo licence
  field is blank. It is not approved for this project.
- [CropAndWeedAndLeaf](https://zenodo.org/records/20116408) supplies leaf-instance
  masks, including maize, squash, potato, and soybean, but no disease-region
  masks. Separate datasets cannot be combined as though their masks annotate
  the same photograph.

## Fastest rights-safe annotation path

1. Use a non-commercial PlantSeg subset only after recording CC BY-NC 4.0,
   source URL, attribution, archive checksum, and an exact mapping to a
   Candidate-v2 crop/disease label.
2. Preserve PlantSeg's lesion mask and annotate the *whole visible leaf* on the
   same image in local open-source CVAT. Do not trace a different photo or infer
   a leaf mask from another dataset.
3. Have a second reviewer inspect every leaf mask and at least 20% of lesion
   masks. Record disagreements and corrections.
4. Keep source/near-duplicate groups in one split. Do not use the final test
   split for model selection or threshold tuning.
5. For crops/classes not covered with suitable rights, collect project-owned
   photographs under a written consent/reuse statement and annotate both masks.

`training/segmentation_manifest.example.csv` defines the evidence fields. The
validator refuses empty rights metadata, anything not marked
`approved_for_project`, group leakage, mismatched dimensions, non-binary masks,
lesions outside the leaf, and empty leaf masks.

## Implemented offline evaluation

`training/evaluate_segmentation.py` evaluates paired leaf and lesion prediction
PNGs against an untouched manifest split and writes machine-readable summary
and per-sample evidence. Missing or empty leaf predictions are abstentions, not
zero-area measurements. The current predefined engineering gate is:

- at least 90% test coverage overall and for every reported crop;
- mean leaf Dice at least 0.95;
- mean lesion Dice at least 0.75;
- affected-area mean absolute error at most 5 percentage points; and
- 95th-percentile affected-area absolute error at most 15 percentage points.

Before approval, the untouched test set must also contain at least 20 images per
reported crop. The evaluator records per-crop metrics, and results must also be
inspected per disease. These are
prototype engineering criteria, not evidence that a percentage predicts yield
loss or whole-plant severity.

## Gate before live integration

Acquire or create rights-cleared, same-image leaf and lesion masks for relevant
supported crop/disease classes, then train a segmentation candidate and run the
frozen gate above. The evaluator records leaf and lesion Dice/IoU, absolute
affected-area error in percentage points, and abstention coverage.

Severity thresholds and health-score semantics need their own documented and
validated rules. Until those gates pass, `diseaseRate`, `severity`, and
`healthScore` remain `null` in live reports.
