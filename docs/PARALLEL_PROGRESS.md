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

## Verification

- `npm ci`: passed.
- `npm run check`: passed.
- `npm run build`: passed; Vite reports a non-blocking large-bundle warning (approximately 669 KB minified JavaScript).
- Python backend and training source compilation: passed.
- Colab notebook JSON validation: passed.
- Backend smoke checks: `/health` 200 with `model_ready=false`; unsupported MIME 415; analysis without a model 503 with no fabricated prediction.
- New parallel T4 runtime and Drive checkpoint directory: started successfully.
- First training attempt: stopped before epoch 1 by the strict leakage check after detecting 227 overlapping source-partition groups; no model or metrics were produced.
- Full GPU training and ONNX parity execution: pending rerun with the corrected custom split.

## Deployment state

- Vercel: no parallel project exists.
- Render: no parallel service has been confirmed or created.
- Model: absent; no candidate or production model is installed.
- `render.yaml`: isolated future service name prepared, but not deployed.

## Known problems and scientific blockers

- No completed classifier run or measured accuracy/macro-F1 exists yet.
- CIFAR-100 is only a provisional non-leaf OOD sanity check.
- PlantDoc or another licensed field-image evaluation set must be mapped and evaluated independently.
- The rejection threshold cannot be approved until realistic unsupported and field inputs are measured.
- Severity/visible affected-area estimation lacks a validated segmentation dataset/model.
- Disease symptoms, causes, and recommendations lack a complete cited agricultural review.
- English/Telugu/Hindi translation and actual cloud speech have not been validated end to end.
- The frontend currently stores the selected full-resolution data URL as a history thumbnail and does not fully translate UI/chart labels.
- Weather has no manual-town fallback, and its saved timestamp handling requires review.

## Next actions

1. Commit and push this training-readiness milestone only to the parallel repository.
2. Start a completely new Colab GPU runtime from the repository notebook.
3. Finish the candidate run and preserve Drive checkpoints/artifacts.
4. Review the confusion matrix, per-class metrics, macro-F1, calibration, selected threshold, test selective metrics, and ONNX parity evidence.
5. Design and run the licensed field/OOD evaluation before any production approval or model installation.

## Relevant commit

- Training-readiness milestone: `684ad9593a6c2eb5e4cf9164488c887b4ca40c06`.
- Leakage-free custom split correction: `01da124048f59ff55b3ee1fb7b2de1765ebf1e76`.
