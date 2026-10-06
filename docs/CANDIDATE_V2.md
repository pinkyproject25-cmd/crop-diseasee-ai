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

The next code milestone is a mixed-domain Candidate-v2 trainer using the 2,278
reviewed PlantDoc training images together with PlantVillage training data.
Model selection and calibration must remain independent of the consumed
PlantDoc test split. The GPU configuration and epoch schedule will be committed
before Candidate-v2 training begins.
