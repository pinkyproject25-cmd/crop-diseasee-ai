# Stage 2 measurement checkpoint — 2026-10-07

## Live report state

- Source-linked knowledge for all 26 diseased Candidate-v2 labels is merged to
  `main` at `ab3214a66420289245081f8b6a96cb69fda7c83d` and to Render's
  `deploy-pinky-prototype` branch at
  `c54adc5021be61b0601b5a6393e0f40bd8aee1f6`.
- Pinky Render reported the Candidate-v2 model ready. A renamed fixed Apple
  scab fixture returned Apple scab at confidence `0.9980962873` with its UC IPM
  entry. A pinned PlantDoc Cedar-rust test image returned Cedar apple rust at
  confidence `0.9549486637` with its University of Minnesota Extension entry.
- The live responses still return `null` for affected area (`diseaseRate`),
  severity, and health score. This is intentional and scientifically required.

## Measurement work preserved on this branch

Implementation commit `b7d9c9f7cd522219c1622e3e9da331166ce77aa9`
adds an offline, evaluation-only paired-mask validator and scorer. It rejects
unapproved rights metadata, group leakage, invalid or mismatched masks, missing
leaf area, and lesion pixels outside the leaf. Missing predictions count as
abstentions. It records overlap, affected-area error, overall coverage, and
per-crop evidence against a predefined prototype gate.

The evaluator is not connected to the API and does not make the live result
look complete before validation.

## Verification

- Backend: 24/24 tests passed from `backend` with the real Candidate-v2
  artifacts and fixed PlantDoc fixtures.
- Frontend: `npm run check` and `npm run build` passed. Vite emitted only the
  existing bundle-size warning.
- Live Render: `/health` returned `model_ready: true` with version
  `mixed-pv-plantdoc-mnv3-20261006T194137Z`.
- Live report APIs: Apple scab and Cedar apple rust accepted-result knowledge
  paths passed. Unaccepted Cedar-rust field photos remained Unknown.

## Evidence blocker

No rights-cleared dataset has yet been verified to provide both a whole-leaf
mask and a lesion mask for the same realistic supported-crop images. PlantSeg
is CC BY-NC 4.0 and supplies lesion masks, but its paper explicitly says leaf
boundaries are not annotated. The 75-image synthetic Apple mask dataset has no
licence displayed. The investigated grape instance-segmentation data is
restricted access and its exact semantics still require inspection.

Consequently, no segmentation model has been trained or approved, and severity
or health-score rules have not been activated.

## Exact next action

Create a rights-recorded PlantSeg subset mapped to Candidate-v2 labels, annotate
whole-visible-leaf masks on those same images in local CVAT, independently
review them, freeze group-disjoint splits, then train one segmentation candidate
and run `training/evaluate_segmentation.py` on the untouched test split. Only a
candidate passing the recorded gate can be considered for API integration.
