# Parallel-instance progress

Updated: 2026-10-06 (Asia/Kolkata)

## Isolation boundary

- Sole writable repository: `pinkyproject25-cmd/crop-diseasee-ai`.
- Takeover base: `main` at `fedbbebac0173f59c09fafc84eb854a23902c860`.
- The original repository and its Vercel, Render, and Colab resources are read-only/off-limits and were not modified.
- Local Git `origin` was verified as `https://github.com/pinkyproject25-cmd/crop-diseasee-ai.git` before editing.

## Completed in this milestone

- Inspected every repository file, frontend/backend architecture, deployment configuration, and the full Colab notebook.
- Verified the connected GitHub identity owns the parallel repository and has write/admin access.
- Verified that no Vercel project in the connected account is linked to the parallel repository.
- Detected one Render workspace but did not select it, inspect services, create a service, or change any setting.
- Confirmed there are no classifier artifacts or completed training metrics in the repository.
- Separated model-selection validation from the temperature/threshold calibration split.
- Audited the loader's published PlantVillage partitions and found 227 overlapping physical `leaf_id` groups during the first parallel Colab attempt.
- Replaced reliance on those source partitions with a fresh 10-fold stratified group split: seven training folds plus distinct model-selection validation, calibration, and untouched test folds.
- Added persistent epoch checkpoint/resume support suitable for Google Drive.
- Added dataset revision and software-version recording.
- Added calibration before/after NLL and ECE evidence, independent-test selective metrics, OOD confidence summaries, per-class CSV, confusion-matrix CSV/PNG, and training curves.
- Added ONNX/PyTorch parity verification and a hashed candidate manifest that remains `production_approved: false`.
- Replaced the inherited Colab notebook with an isolated notebook that clones only the parallel repository and writes to `MyDrive/CropDiseaseAIParallel`.
- Renamed the undeployed Render Blueprint service to `crop-diseasee-ai-parallel-api`.
- Corrected stale documentation that previously described the original project's deployments and Colab as this clone's state.
- Completed all 12 classifier-training epochs in the parallel Colab runtime; the
  epoch-12 checkpoint and training history are persisted in Google Drive.
- Corrected a post-training calibration failure by changing `collect_logits`
  from `torch.inference_mode()` to `torch.no_grad()`. This preserves
  gradient-free model evaluation while allowing LBFGS to optimize the
  temperature parameter against the collected logits.
- Ran the fixed candidate once on all 236 images in the official pinned
  PlantDoc test split and preserved the complete evidence under the isolated
  Drive directory `field_evaluation_plantdoc_v1`.
- Reviewed all 11 field errors that passed the `0.99` threshold. Two corn
  examples warrant independent label review, but even crediting both to the
  model cannot change the failed gate decision.
- Added a CPU-only Candidate-v2 PlantDoc audit that verifies the pinned source,
  decodes every image, detects exact pixel duplication/leakage, flags perceptual
  near-duplicates, and generates a review-required training manifest without
  ever adding PlantDoc test images to training.
- Completed that audit on all 2,578 pinned PlantDoc images with zero decode
  errors. Exact-integrity rules excluded 20 training images and left 2,322
  initially clean rows.
- Visually reviewed all three cross-split near-duplicate pairs and all 23
  different-label near-duplicate pairs. Every pair represented the same or a
  near-identical source photograph, so no disputed labels were guessed or
  repaired. A conservative committed quarantine removes 44 additional rows
  (one of 45 flagged paths was already excluded), leaving 2,278 reviewed
  PlantDoc training images.
- Added a hash-pinned manifest finalizer that refuses changed audit evidence,
  enforces train-only/unique clean rows, applies the reviewed quarantine, and
  records `production_approved: false` in its review evidence.

## Verification

- `npm ci`: passed.
- `npm run check`: passed.
- `npm run build`: passed; Vite reports a non-blocking large-bundle warning (approximately 669 KB minified JavaScript).
- Python backend and training source compilation: passed.
- Colab notebook JSON validation: passed.
- Backend smoke checks: `/health` 200 with `model_ready=false`; unsupported MIME 415; analysis without a model 503 with no fabricated prediction.
- New parallel T4 runtime and Drive checkpoint directory: started successfully.
- First training attempt: stopped before epoch 1 by the strict leakage check after detecting 227 overlapping source-partition groups; no model or metrics were produced.
- Full 12-epoch GPU training: passed. Final model-selection validation accuracy
  was `0.9929798633` and macro-F1 was `0.9908847217`.
- The first post-training evaluation attempt stopped during temperature fitting
  because inference tensors cannot participate in an autograd graph. The fix is
  committed; rerunning the same command resumes after epoch 12, so no training
  epochs need to be repeated.
- Resumed from the Drive-backed epoch-12 checkpoint after the calibration fix;
  calibration, independent testing, CIFAR-100 OOD sanity checking, artifact
  generation, and ONNX parity verification completed successfully.
- Candidate `plantvillage-mnv3-20261006T022813Z` achieved test accuracy
  `0.9933456562`, test macro-F1 `0.9915131174`, and test ECE
  `0.0025011350` on 5,410 group-independent PlantVillage examples.
- Temperature scaling reduced calibration-split ECE from `0.1345514788` to
  `0.0023865518` with temperature `0.4217035472`.
- The provisional `0.99` acceptance threshold retained `94.60%` of the test
  set with `99.88%` accepted accuracy. CIFAR-100 false acceptance was `0.71%`.
- ONNX parity passed on eight examples with 8/8 top-1 agreement and maximum
  absolute logit error `3.8147e-05`.
- The uploaded `metrics.json` SHA-256 matches its candidate manifest. The ONNX
  and labels hashes remain recorded in the manifest but were not independently
  re-hashed outside Colab.
- Verified the official Cropped-PlantDoc repository, pinned revision
  `5467f6012d78d1c446145d5f582da6096f852ae8`, and CC BY 4.0 license.
- Added a candidate-integrity-checked field/OOD evaluator that reproduces the
  backend's pixel preprocessing and quality gate, never uses filenames for
  inference, refuses unmapped classes, and keeps the `0.99` threshold fixed.
- Added an explicit mapping covering every class in PlantDoc's official test
  split plus its train-only tomato spider-mite class.
- Predefined the field gate before viewing results: at least 30% coverage, 95%
  accepted accuracy, and a 90% lower bound for its 95% Wilson interval. The OOD
  gate requires at least 500 documented images and at most 1% false acceptance.
- PlantDoc field evaluation: 236/236 images decoded; top-1 accuracy before
  rejection was `25.42%` and macro-F1 was `24.61%`.
- At the fixed `0.99` threshold, 23 images were accepted (coverage `9.75%`),
  12 were correct (accepted accuracy `52.17%`), and the accepted-accuracy 95%
  Wilson lower bound was `32.96%`. The predefined field gate failed.
- Uploaded audit evidence verification: `audit_summary.json` SHA-256
  `23056c43a866bd8d0c4590c07641863371776612d2c8fb9fd9846c6fb0181d01`;
  `clean_train_manifest.csv` SHA-256
  `d2fd5a3846bb7d053e87330c32a943d3ca69b16850e1227b34c99e712a88a631`.
- Manifest finalizer integration check passed: 2,322 clean input rows, 44
  additional quarantines, 2,278 output rows, 28 represented target classes,
  and deterministic reviewed-manifest SHA-256
  `66e8326295550f5e8453182dbf0bfcddf7af7d06a3a8e9954c3e5f9322ee2b62`.

## Deployment state

- Vercel: no parallel project exists.
- Render: no parallel service has been confirmed or created.
- Model: absent; no candidate or production model is installed.
- `render.yaml`: isolated future service name prepared, but not deployed.

## Known problems and scientific blockers

- High controlled-dataset performance did not transfer to the independent
  field set. Candidate v1 failed the PlantDoc field gate and is rejected for
  backend integration in its current form.
- The 36 independent-test errors are concentrated in visually similar tomato
  diseases. The largest pairs are tomato late blight to early blight (5) and
  tomato spider mites to target spot (5).
- Potato healthy has only 16 independent-test examples and the lowest per-class
  F1 (`0.9375`), so its controlled-set estimate has high sampling uncertainty.
- CIFAR-100 is only a provisional non-leaf OOD sanity check.
- The `0.99` confidence threshold is not field-calibrated: 11 of 23 accepted
  PlantDoc predictions were wrong under the published labels.
- PlantDoc test has now been consumed for candidate-v1 evaluation and must not
  be used for threshold tuning. A different untouched field set is required
  for a future approval decision.
- Realistic unsupported/non-leaf evaluation remains pending, but it cannot
  rescue candidate v1's failed field gate.
- Severity/visible affected-area estimation lacks a validated segmentation dataset/model.
- Disease symptoms, causes, and recommendations lack a complete cited agricultural review.
- English/Telugu/Hindi translation and actual cloud speech have not been validated end to end.
- The frontend currently stores the selected full-resolution data URL as a history thumbnail and does not fully translate UI/chart labels.
- Weather has no manual-town fallback, and its saved timestamp handling requires review.

## Next actions

1. Run the committed finalizer against the Drive-backed audit evidence and
   retain `reviewed_train_manifest.csv` plus `manifest_review.json`.
2. Implement and run the predefined mixed-domain Candidate-v2 training recipe;
   keep PlantDoc test excluded from training, model selection, and calibration.
3. Refit calibration and rejection using only candidate-v2 calibration data;
   do not tune against the consumed PlantDoc test results.
4. Assemble a source-documented realistic unsupported/non-leaf suite and
   measure false acceptance under the separately predefined OOD gate.
5. Keep every candidate out of the backend until both independent field and
   realistic OOD gates pass.

## Relevant commit

- Training-readiness milestone: `684ad9593a6c2eb5e4cf9164488c887b4ca40c06`.
- Leakage-free custom split correction: `01da124048f59ff55b3ee1fb7b2de1765ebf1e76`.
- Temperature-calibration fix: `54e8c7d9b2d7fcfe601fab3b9c54affa8711bb67`.
- PlantDoc field-gate evidence and candidate-v1 rejection:
  `ed1682e75853f6c180b16ee517ba7b17f10956d3`.
- Candidate-v2 PlantDoc integrity audit:
  `1fb6e3b923fe9d80e073dc5c100dd2bc2634775e`.
- Reviewed PlantDoc manifest finalization:
  `85d2a36b241ebe377185d381ed4b6c76bd784062`.
