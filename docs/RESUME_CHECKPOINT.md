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
- Separated the model-selection validation fold from the calibration fold, added an explicit published train/test leaf-group overlap check, resumable epoch checkpoints, dataset revision recording, calibration before/after evidence, independent-test selective metrics, per-class CSV, confusion-matrix CSV/PNG, training curves, candidate manifest hashes, and ONNX parity verification.
- Verified locally that the frontend builds, the backend smoke tests pass, the training script compiles, and the notebook is valid JSON.

## Current parallel behavior

The parallel application is not deployed. No diagnostic model has been approved or installed. The local API deliberately refuses to invent a crop or disease prediction, which is the intended safe behavior until the model passes the required tests.

## Parallel Colab state

- No new parallel Colab runtime has been started yet.
- The repository notebook requests a new GPU runtime, mounts Drive, verifies the Git origin, and resumes from the persistent checkpoint directory.
- Metrics remain unavailable until that new run completes.
- The original project's earlier T4 runtime and interrupted epoch are historical only and must not be resumed or modified.

## Remaining work

1. Open the repository notebook in a completely new Colab GPU runtime and finish/resume candidate training from the new parallel Drive folder.
2. Review validation, test, calibration, confusion-matrix, and out-of-distribution evidence.
3. Evaluate field-image behavior with PlantDoc/PlantSeg-style data and unsupported/non-leaf images.
4. Keep `production_approved` false until all gates pass; do not install a candidate merely because controlled PlantVillage metrics are high.
5. Implement and validate disease-area/severity estimation.
6. Complete and agriculturally review the symptoms, causes, and recommendations knowledge base.
7. Complete working English, Telugu, and Hindi translation and speech.
8. Create isolated Vercel and Render resources only after the scientific candidate is reviewed; install only an approved model and run full end-to-end verification.

## Integrity rules

- Analyze actual image bytes, never filenames.
- Never fabricate a crop, disease, severity, confidence, symptom, cause, or recommendation.
- Return Undefined/Unknown for unsupported or uncertain images.
- Treat model output as decision support, not a definitive agricultural diagnosis.

## Repository state

- Takeover base commit: `fedbbebac0173f59c09fafc84eb854a23902c860`.
- Training-readiness milestone commit: `331e4d46d85685bd828be889ef8e271c90aa89a0`.
- Detailed progress: `docs/PARALLEL_PROGRESS.md`.
