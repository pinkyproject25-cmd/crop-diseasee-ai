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

The GPU configuration and epoch schedule will be fixed only after the audit
establishes the usable field-training counts. This prevents choosing a training
recipe without knowing the effective data size and leakage risk.
