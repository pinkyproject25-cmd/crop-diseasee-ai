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

## Current parallel behavior

The parallel application is not deployed. No diagnostic model has been approved or installed. The local API deliberately refuses to invent a crop or disease prediction, which is the intended safe behavior until the model passes the required tests.

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

1. Run and review the pinned official PlantDoc test evaluation, then evaluate a source-documented realistic unsupported/non-leaf suite.
2. Review the 36 controlled-test errors, especially visually similar tomato diseases, and gather more evidence for low-support classes such as potato healthy.
3. Keep `production_approved` false until all gates pass; do not install a candidate merely because controlled PlantVillage metrics are high.
4. Implement and validate disease-area/severity estimation.
5. Complete and agriculturally review the symptoms, causes, and recommendations knowledge base.
6. Complete working English, Telugu, and Hindi translation and speech.
7. Create isolated Vercel and Render resources only after the scientific candidate is reviewed; install only an approved model and run full end-to-end verification.

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
- Detailed progress: `docs/PARALLEL_PROGRESS.md`.
