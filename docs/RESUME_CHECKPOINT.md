# Crop Disease AI Parallel — Resume Checkpoint

Saved: 2026-10-06 (Asia/Kolkata)

Writable repository: **pinkyproject25-cmd/crop-diseasee-ai only**

## Parallel resources

- Repository: https://github.com/pinkyproject25-cmd/crop-diseasee-ai
- Fresh Colab entry point: https://colab.research.google.com/github/pinkyproject25-cmd/crop-diseasee-ai/blob/main/notebooks/train_classifier_colab.ipynb
- Vercel: no parallel project created.
- Render: no parallel service confirmed or created.

Never use the original project's repository, Colab notebook, Vercel project, Render service, environment variables, or deployment settings.

## Completed

- Verified the isolated GitHub repository and its only branch/commit before takeover.
- Built the responsive React, Vite, and TypeScript frontend locally.
- Implemented Home, Analyze, Result, Dashboard, History, Supported Crops, and About pages.
- Implemented image upload and camera capture, analysis preview, localStorage history, farmer-friendly gauges and charts, and responsive navigation.
- Verified the FastAPI backend locally with image-byte preprocessing, ONNX inference hooks, weather integration, translation/TTS hooks, and strict refusal when no validated model is available.
- Added a hosted classifier-training pipeline using MobileNetV3 Small, group-aware splitting, augmentation, class weights, calibration, out-of-distribution checks, ONNX export, and explicit production blockers.
- Replaced the inherited notebook with an isolated parallel Colab workflow that clones only this repository and stores resumable output under `MyDrive/CropDiseaseAIParallel`.
- Corrected the PlantVillage loader for current Colab behavior by pinning `datasets==3.6.0`, downloading the reviewed repository loader with `hf_hub_download`, and loading its `default` configuration with `trust_remote_code=True`.
- Separated model selection, calibration, and testing into distinct custom leaf-group folds; added source-partition overlap auditing, resumable epoch checkpoints, dataset revision recording, calibration before/after evidence, independent-test selective metrics, per-class CSV, confusion-matrix CSV/PNG, training curves, candidate manifest hashes, and ONNX parity verification.
- Verified locally that the frontend builds, the backend smoke tests pass, the training script compiles, and the notebook is valid JSON.
- Completed all 12 parallel T4 training epochs and the full post-training
  evaluation for candidate `plantvillage-mnv3-20261006T022813Z`.
- Recorded group-independent test accuracy `0.9933456562`, macro-F1
  `0.9915131174`, and ECE `0.0025011350`; `production_approved` remains false.
- Verified the provisional `0.99` threshold on the independent test set:
  `94.60%` coverage and `99.88%` accepted accuracy. CIFAR-100 false acceptance
  was `0.71%`, which is only a preliminary OOD sanity check.
- Verified reported ONNX parity on eight examples (8/8 top-1 agreement,
  maximum absolute logit error `3.8147e-05`) and matched the uploaded metrics
  file to the SHA-256 recorded in the candidate manifest.
- Added `training/evaluate_candidate.py`, the pinned PlantDoc mapping, and
  `docs/FIELD_EVALUATION.md` for the next external field/OOD gate. The evaluator
  verifies candidate hashes, mirrors production preprocessing/quality checks,
  and fixes the acceptance threshold before examining field results.
- Field pass criteria were fixed before execution (30% coverage, 95% accepted
  accuracy, 90% Wilson lower bound); the realistic OOD gate requires at least
  500 documented images and no more than 1% false acceptance.
- Evaluated candidate `plantvillage-mnv3-20261006T022813Z` on all 236 images in
  the pinned official PlantDoc test split. Top-1 accuracy was `25.42%` and
  macro-F1 was `24.61%` before rejection.
- Candidate v1 failed the predefined field gate: `9.75%` coverage, `52.17%`
  accepted accuracy (12/23), and `32.96%` accepted-accuracy Wilson lower bound.
  All 11 accepted errors were manually reviewed; two possible source-label
  anomalies do not alter the failure decision.
- Added `training/audit_plantdoc.py` and `docs/CANDIDATE_V2.md` for the next
  CPU-only step. The audit verifies the pinned dataset, checks decoding and
  mappings, removes only exact leakage/duplicates, flags perceptual similarity,
  and keeps PlantDoc test out of the generated training manifest.
- Completed the pinned PlantDoc audit: 2,578/2,578 images decoded, no errors,
  20 exact-integrity training exclusions, and 2,322 initially clean rows.
- Completed visual review of every cross-split and conflicting-label
  near-duplicate pair. The committed conservative quarantine removes 44 more
  rows (with one reviewed path already removed by the exact audit), yielding a
  2,278-row Candidate-v2 PlantDoc training manifest.
- Added `training/finalize_plantdoc_manifest.py` and the hash-pinned manual
  quarantine decision. The finalizer validates the uploaded evidence and keeps
  `production_approved` false.

## Current parallel behavior

The parallel application is not deployed. Candidate v1 is explicitly rejected
for backend integration after failing the independent field gate. No diagnostic
model has been approved or installed. The local API deliberately refuses to
invent a crop or disease prediction, which remains the intended behavior.

## Parallel Colab state

- The parallel T4 run completed. Its candidate artifacts and resumable
  checkpoints are under `MyDrive/CropDiseaseAIParallel/classifier_candidate_v1`;
  the archive is `MyDrive/CropDiseaseAIParallel/crop-disease-classifier-candidate-v1.zip`.
- The repository notebook mounts Drive, verifies the Git origin, and resumes
  from the persistent checkpoint directory.
- The first new T4 attempt exposed 227 overlapping `leaf_id` groups in the loader's published train/test partitions and correctly stopped before training; the pipeline now creates a fresh 10-fold group-independent split.
- A post-training temperature-calibration failure caused by inference tensors
  was fixed in commit `54e8c7d9b2d7fcfe601fab3b9c54affa8711bb67`;
  the resumed run completed without repeating training.
- The original project's earlier T4 runtime and interrupted epoch are historical only and must not be resumed or modified.

## Remaining work

1. Pull the latest parallel commit in the existing CPU runtime and run the
   manifest finalizer against `plantdoc_audit_v1`; retain its two output files.
2. Implement and run a fixed mixed-domain Candidate-v2 recipe using the
   reviewed PlantDoc train manifest and leakage-controlled PlantVillage data.
3. Recalibrate candidate v2 without tuning on the now-consumed PlantDoc test
   split, then evaluate a source-documented realistic unsupported/non-leaf suite.
4. Keep `production_approved` false and do not install candidate v1.
5. Review the 36 controlled-test errors, especially visually similar tomato diseases, and gather more evidence for low-support classes such as potato healthy.
6. Implement and validate disease-area/severity estimation.
7. Complete and agriculturally review the symptoms, causes, and recommendations knowledge base.
8. Complete working English, Telugu, and Hindi translation and speech.
9. Create isolated Vercel and Render resources only after a scientific candidate is approved; install only an approved model and run full end-to-end verification.

## Integrity rules

- Analyze actual image bytes, never filenames.
- Never fabricate a crop, disease, severity, confidence, symptom, cause, or recommendation.
- Return Undefined/Unknown for unsupported or uncertain images.
- Treat model output as decision support, not a definitive agricultural diagnosis.

## Repository state

- Takeover base commit: `fedbbebac0173f59c09fafc84eb854a23902c860`.
- Training-readiness milestone commit: `684ad9593a6c2eb5e4cf9164488c887b4ca40c06`.
- Leakage-free custom split correction: `01da124048f59ff55b3ee1fb7b2de1765ebf1e76`.
- Temperature-calibration fix: `54e8c7d9b2d7fcfe601fab3b9c54affa8711bb67`.
- PlantDoc field-gate evidence and candidate-v1 rejection:
  `ed1682e75853f6c180b16ee517ba7b17f10956d3`.
- Candidate-v2 PlantDoc integrity audit:
  `1fb6e3b923fe9d80e073dc5c100dd2bc2634775e`.
- Reviewed PlantDoc manifest finalization:
  `85d2a36b241ebe377185d381ed4b6c76bd784062`.
- Detailed progress: `docs/PARALLEL_PROGRESS.md`.
