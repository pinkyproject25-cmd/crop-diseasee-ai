# Crop Disease AI — Resume Checkpoint

Saved: 2026-10-06 (Asia/Kolkata)

Resume phrase: **RUN NOW**

## Online resources

- Colab training notebook: https://colab.research.google.com/drive/1yxpNiNnhRlrcmVYC4HyjAdSvNNboRaQX

## Completed

- Created the private GitHub repository and populated it with the application.
- Built and deployed the responsive React, Vite, and TypeScript frontend.
- Implemented Home, Analyze, Result, Dashboard, History, Supported Crops, and About pages.
- Implemented image upload and camera capture, analysis preview, localStorage history, farmer-friendly gauges and charts, and responsive navigation.
- Deployed a FastAPI backend on Render with image-byte preprocessing, ONNX inference hooks, weather integration, translation/TTS hooks, and strict refusal when no validated model is available.
- Verified an end-to-end upload against the production API. The request reached the backend and the UI correctly displayed that no prediction was generated because a validated model was not installed.
- Added a hosted classifier-training pipeline using MobileNetV3 Small, group-aware splitting, augmentation, class weights, calibration, out-of-distribution checks, ONNX export, and explicit production blockers.
- Added a Colab notebook for the hosted training workflow.
- Corrected the PlantVillage loader for current Colab behavior by pinning `datasets==3.6.0`, downloading the reviewed repository loader with `hf_hub_download`, and loading its `default` configuration with `trust_remote_code=True`.
- Verified locally that the training script compiles and that the generated notebook is valid JSON with the corrected dependency and loader code.

## Current production behavior

The public application is online, but no diagnostic model has been approved or installed. It deliberately refuses to invent a crop or disease prediction. This is the intended safe behavior until the model passes the required tests.

## Colab state

- A T4 GPU runtime was connected.
- The official PlantVillage archive and pretrained MobileNetV3 weights downloaded successfully.
- The corrected loader produced 43,596 training examples and 10,709 test examples.
- A training run was intentionally interrupted during epoch 1 at the user's request to pause work.
- The Colab runtime remains connected and idle. Its cache may survive for a while, but it must not be treated as durable storage. If the runtime disconnects, rerun the corrected notebook from the top.

## Remaining work

1. Rerun the corrected Colab notebook cleanly and finish candidate training.
2. Review validation, test, calibration, confusion-matrix, and out-of-distribution evidence.
3. Evaluate field-image behavior with PlantDoc/PlantSeg-style data and unsupported/non-leaf images.
4. Keep `production_approved` false until all gates pass; do not install a candidate merely because controlled PlantVillage metrics are high.
5. Implement and validate disease-area/severity estimation.
6. Complete and agriculturally review the symptoms, causes, and recommendations knowledge base.
7. Complete working English, Telugu, and Hindi translation and speech.
8. Install only an approved model in the Render service and run full production end-to-end verification.

## Integrity rules

- Analyze actual image bytes, never filenames.
- Never fabricate a crop, disease, severity, confidence, symptom, cause, or recommendation.
- Return Undefined/Unknown for unsupported or uncertain images.
- Treat model output as decision support, not a definitive agricultural diagnosis.

## Repository state before this checkpoint

- Previous `main` commit: `0597e5a8f701f9d28f1c955b49a70b5fe84609b4`
- This checkpoint commit adds the loader corrections and this resume document.
