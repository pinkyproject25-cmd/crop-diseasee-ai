# Implementation status

## Completed

- Isolated public GitHub repository `pinkyproject25-cmd/crop-diseasee-ai` and durable project specification.
- React + Vite + TypeScript frontend.
- Seven routes: Home, Analyze, Result, Dashboard, History, Supported Crops, and About.
- Upload and mobile camera capture input with preview, format checks, and a 10 MB client limit.
- Strict API client with no fake/demo prediction fallback.
- Diseased, healthy, and unknown result layouts.
- Separate fields for confidence, visible disease rate, severity, and health score.
- Top-prediction chart using original probabilities.
- Browser-local report history, reopen, delete, and clear flows.
- Local dashboard with healthy, diseased, and unknown summaries.
- Language selection and server-generated speech hooks, including immediate Stop. End-to-end provider and fluent-language verification are still pending.
- FastAPI backend contract, image decoding, filename-independent pixel preprocessing, basic photo-quality rejection, model threshold handling, weather integration, Azure translation, and Azure speech.
- Docker packaging for the API.
- Generated files, local secrets, caches, and unreviewed model binaries are excluded from Git.
- Frontend TypeScript checking and the Vite production build pass locally.
- Backend smoke checks pass for health, unsupported media, and deliberate missing-model refusal.
- The training pipeline creates fresh leaf-group-independent train/validation/calibration/test partitions, writes resumable checkpoints, exports per-class/confusion-matrix evidence, and verifies ONNX parity.
- Candidate v1 completed controlled evaluation but failed the predefined
  independent PlantDoc field gate (`9.75%` coverage and `52.17%` accepted
  accuracy at threshold `0.99`). It is not approved or installed.
- The pinned PlantDoc train/test audit decoded all 2,578 images with no errors,
  automatically removed 20 exact training duplicates/conflicts, and produced
  2,322 clean training rows. Manual review conservatively quarantined 44 more
  rows implicated in cross-split or conflicting-label near duplicates, leaving
  2,278 reviewed field-training images for Candidate-v2 experimentation.
- Candidate-v2 training is implemented with hash-checked inputs, fixed
  mixed-domain sampling, perceptual-group-safe PlantDoc partitions, resumable
  checkpoints, balanced-domain calibration, and explicit non-production
  outputs. Its CPU preflight must pass before GPU training begins.

## Scientific gates before the application can claim real production analysis

- Implement, train, and export a mixed-domain Candidate-v2 classifier using
  only the reviewed PlantDoc training manifest plus leakage-controlled
  PlantVillage training data.
- Reconfirm class-wise performance on leakage-independent controlled and field
  splits; PlantDoc test is now a consumed benchmark, not a tuning set.
- Pass a different untouched field-photo gate and a source-documented
  unsupported/out-of-distribution gate.
- Calibrate the acceptance threshold using the dedicated calibration split and confirm selective performance on the independent test set.
- Obtain segmentation/severity annotations and validate visible affected-area measurement.
- Add reviewed, cited disease information for every production-supported class.
- Validate Telugu and Hindi agricultural terminology with a fluent reviewer.

## Current deliberate behavior

When the model files are absent, the API returns HTTP 503 and states that no prediction was generated. When a candidate class is returned without the reviewed information and severity layers, the API also refuses to construct an incomplete disease report.

This parallel instance has no Vercel project and no confirmed Render service. Model readiness remains false until the scientific gates above are completed. Future deployments must use new resources associated only with `pinkyproject25-cmd/crop-diseasee-ai`.

This behavior prevents the polished interface from being mistaken for a completed AI system.
