# Candidate v2 preparation

Candidate v1 is not eligible for backend integration because it failed the
predefined PlantDoc field gate. Candidate v2 must address field-domain
generalization without tuning against the consumed PlantDoc test results.

## Stage 1: PlantDoc data audit (CPU only)

The first step is an integrity audit, not training. The audit:

- verifies the official repository revision;
- requires an explicit mapping for every class directory;
- decodes every train and test image;
- hashes source files and decoded RGB pixels;
- excludes exact training duplicates of test images;
- excludes exact same-pixel label conflicts;
- removes redundant exact copies within the training split;
- flags perceptually similar pairs for manual review; and
- never adds PlantDoc test images to the training manifest.

Use a fresh Colab **CPU** runtime. Mount the parallel Drive and run:

```bash
cd /content
git clone https://github.com/pinkyproject25-cmd/crop-diseasee-ai.git crop-diseasee-ai-parallel
cd /content/crop-diseasee-ai-parallel

git remote -v
git rev-parse HEAD

cd /content
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git
cd /content/PlantDoc-Dataset
git checkout 5467f6012d78d1c446145d5f582da6096f852ae8

cd /content/crop-diseasee-ai-parallel
pip install -q pillow==11.2.1 numpy==2.2.5

python training/audit_plantdoc.py \
  --dataset-root /content/PlantDoc-Dataset \
  --mapping training/plantdoc_mapping.json \
  --output-dir /content/drive/MyDrive/CropDiseaseAIParallel/plantdoc_audit_v1 \
  --near-duplicate-distance 4
```

The command deliberately reports `status: review_required`. A generated clean
manifest is not authorization to train until the duplicate and label review is
recorded.

Expected evidence:

- `audit_summary.json`
- `decode_errors.csv`
- `class_counts.csv`
- `exact_duplicate_groups.csv`
- `near_duplicate_pairs.csv`
- `cross_split_near_duplicate_pairs.csv`
- `different_label_near_duplicate_pairs.csv`
- `train_manifest.csv`
- `clean_train_manifest.csv`

## Completed audit and manual review

The pinned audit completed on 2,578 decodable images (2,342 train and 236
test), with no decode errors. It found 17 exact duplicate groups containing 34
images. The automatic integrity rules excluded 20 training images and produced
2,322 initially clean training rows.

The perceptual review then examined all three cross-split near-duplicate pairs
and all 23 different-label near-duplicate pairs. Each reviewed pair contained
the same or a near-identical source photograph. The conservative decision is:

- remove every training-side cross-split near duplicate;
- remove both training sides of every conflicting-label near duplicate;
- never guess or repair a disputed source label; and
- leave PlantDoc test entirely outside training.

This identified 45 unique training paths. One was already removed by the exact
audit, so the committed manual quarantine removes 44 additional rows. The
reviewed Candidate-v2 PlantDoc training manifest therefore contains 2,278
images. This authorizes controlled Candidate-v2 experimentation only; it does
not approve a model for production.

After pulling the review commit in the same CPU runtime, finalize the manifest:

```bash
cd /content/crop-diseasee-ai-parallel
git pull --ff-only origin main

python training/finalize_plantdoc_manifest.py \
  --audit-summary /content/drive/MyDrive/CropDiseaseAIParallel/plantdoc_audit_v1/audit_summary.json \
  --clean-manifest /content/drive/MyDrive/CropDiseaseAIParallel/plantdoc_audit_v1/clean_train_manifest.csv \
  --output-dir /content/drive/MyDrive/CropDiseaseAIParallel/plantdoc_audit_v1
```

Expected final output:

- `reviewed_train_manifest.csv` with 2,278 rows;
- `manifest_review.json` with `production_approved: false`; and
- `training_authorized: true`, scoped only to Candidate-v2 experimentation.

The finalizer verifies the SHA-256 hashes of the reviewed audit inputs before
writing anything, so a changed or accidentally substituted audit will stop
instead of silently producing a different dataset.

## Candidate-v2 constraints

- PlantDoc `test` is a consumed benchmark and remains evaluation-only.
- Do not select an epoch, threshold, augmentation, or architecture by optimizing
  against PlantDoc test predictions.
- PlantDoc `train` may be considered only after the audit and manual review.
- Candidate v2 needs separately defined model-selection and calibration data.
- A different licensed, untouched field-photo set is required for the final
  approval gate.
- The model remains `production_approved: false` until both field and realistic
  unsupported-image gates pass.

## Stage 2: fixed mixed-domain experiment

`training/train_classifier_v2.py` fixes the experiment before any Candidate-v2
results are viewed:

- initialize from Candidate v1 `best_model.pt`, recording its SHA-256;
- train the full MobileNetV3 Small network for 8 epochs;
- use batch size 64, learning rate `1e-4`, weight decay `1e-4`, and seed 7386;
- sample 75% PlantVillage and 25% reviewed PlantDoc images per epoch;
- keep PlantVillage leaf groups separated with the existing locked folds;
- connect PlantDoc images with dHash distance at most 4 into indivisible groups;
- split PlantDoc into 1,822 training, 228 model-selection validation, and 228
  calibration images with no perceptual-group overlap;
- select the checkpoint by the equal mean of PlantVillage and PlantDoc
  validation macro-F1, retaining Candidate v1 as an epoch-zero baseline;
- fit one temperature with equal NLL weight for the two calibration domains;
  and
- export an ONNX candidate that remains `production_approved: false`.

PlantDoc has only two reviewed examples for its mapped tomato spider-mite
class. Both remain in training, so the PlantDoc validation and calibration
splits each cover 27 of 28 mapped classes. The trainer records this limitation;
it does not fabricate support.

The PlantVillage fold previously called the independent test fold was already
reported for Candidate v1. Candidate v2 therefore treats it only as a locked
controlled regression benchmark, not as a new untouched test set.

The provisional threshold constraints are also fixed before training:

- PlantVillage calibration coverage at least 50% and accepted accuracy at
  least 95%;
- PlantDoc calibration coverage at least 30%, accepted accuracy at least 95%,
  and 95% Wilson lower bound at least 90%; and
- CIFAR-100 false acceptance at most 1%.

Passing those constraints still cannot approve Candidate v2. A different
licensed, untouched field-photo dataset and a realistic agricultural OOD suite
remain mandatory.

### CPU preflight

Run this before switching to a T4. It verifies every reviewed PlantDoc file
hash, Candidate-v1 inputs, the pinned dataset revision, and all split groups.
It performs no training.

```bash
cd /content/crop-diseasee-ai-parallel
git pull --ff-only origin main
pip install -q -r training/requirements.txt

python training/train_classifier_v2.py \
  --output-dir /content/drive/MyDrive/CropDiseaseAIParallel/classifier_candidate_v2 \
  --plantdoc-root /content/PlantDoc-Dataset \
  --plantdoc-manifest /content/drive/MyDrive/CropDiseaseAIParallel/plantdoc_audit_v1/reviewed_train_manifest.csv \
  --plantdoc-review /content/drive/MyDrive/CropDiseaseAIParallel/plantdoc_audit_v1/manifest_review.json \
  --initial-checkpoint /content/drive/MyDrive/CropDiseaseAIParallel/classifier_candidate_v1/best_model.pt \
  --initial-labels /content/drive/MyDrive/CropDiseaseAIParallel/classifier_candidate_v1/labels.json \
  --initial-candidate-manifest /content/drive/MyDrive/CropDiseaseAIParallel/classifier_candidate_v1/candidate_manifest.json \
  --preflight-only
```

The expected preflight split is 2,278 rows in 2,251 perceptual groups: 1,822
training, 228 validation, and 228 calibration. Keep the generated
`input_preflight.json` and `plantdoc_candidate_v2_splits.csv` in Drive.
