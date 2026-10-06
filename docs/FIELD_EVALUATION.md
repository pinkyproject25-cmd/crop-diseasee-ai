# Candidate field and unsupported-image evaluation

This gate evaluates the fixed candidate without retraining, tuning, or changing
its `0.99` acceptance threshold. Passing the controlled PlantVillage test is not
evidence that field photographs are reliable.

## PlantDoc source and scope

- Official repository: `https://github.com/pratikkayal/PlantDoc-Dataset`
- Pinned revision: `5467f6012d78d1c446145d5f582da6096f852ae8`
- License: CC BY 4.0; preserve attribution when sharing derived evidence.
- Evaluation split: official `test` directory only.
- Class mapping: `training/plantdoc_mapping.json`.

The mapping is explicit and is used only for scoring. Inference receives decoded
pixels only. A directory or filename is never an input feature. An unexpected
class directory, artifact hash mismatch, or dataset revision mismatch stops the
evaluation instead of silently dropping data.

## Colab field run

Use the same parallel Colab runtime/Drive. The candidate remains in Drive and is
not copied into the backend.

```bash
cd /content/crop-diseasee-ai-parallel
git pull --ff-only origin main

cd /content
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git
cd PlantDoc-Dataset
git checkout 5467f6012d78d1c446145d5f582da6096f852ae8

cd /content/crop-diseasee-ai-parallel
python training/evaluate_candidate.py \
  --mode field \
  --model /content/drive/MyDrive/CropDiseaseAIParallel/classifier_candidate_v1/crop_classifier.onnx \
  --labels /content/drive/MyDrive/CropDiseaseAIParallel/classifier_candidate_v1/labels.json \
  --manifest /content/drive/MyDrive/CropDiseaseAIParallel/classifier_candidate_v1/candidate_manifest.json \
  --dataset-root /content/PlantDoc-Dataset/test \
  --mapping training/plantdoc_mapping.json \
  --output-dir /content/drive/MyDrive/CropDiseaseAIParallel/field_evaluation_plantdoc_v1
```

Review `field_metrics.json`, `field_per_class_metrics.csv`, the confusion matrix,
and every row in `field_top_confident_errors.csv`. Do not tune the threshold on
this test split.

The field gate is fixed before inspecting results: at least 30% coverage, at
least 95% accepted accuracy, and a 95% Wilson lower confidence bound of at least
90% for accepted accuracy. Passing this gate still does not grant production
approval.

## Candidate v1 result (2026-10-06)

Candidate `plantvillage-mnv3-20261006T022813Z` was evaluated once against the
pinned official PlantDoc test split with the unchanged `0.99` threshold.

| Measure | Result | Predefined requirement |
| --- | ---: | ---: |
| Evaluated images | 236 | n/a |
| Top-1 accuracy before rejection | 25.42% | Diagnostic only |
| Macro-F1 before rejection | 24.61% | Diagnostic only |
| Accepted images | 23 | n/a |
| Coverage | 9.75% | at least 30% |
| Accepted accuracy | 52.17% (12/23) | at least 95% |
| Accepted-accuracy 95% Wilson lower bound | 32.96% | at least 90% |
| Quality rejections | 14 | n/a |
| Confidence rejections after quality | 199 | n/a |

The predefined field gate **failed**. The candidate must not be installed in
the backend or marked production-approved. Confidence learned from the
controlled PlantVillage domain did not transfer reliably to field images: 11
of the 23 predictions accepted at `0.99` were wrong under the published
PlantDoc directory labels.

All 11 accepted errors were visually reviewed against the source images. Two
files placed in `Corn leaf blight` (`2013Corn_GrayLeafSpot_0815_0003.JPG.jpg`
and `corn-gray-leaf-spot-f4.jpg`) have filenames and visible lesions that
warrant independent ground-truth review because the model predicted gray leaf
spot. They were not silently relabeled. Even treating both as correct would
produce only 60.87% accepted accuracy (14/23), so label uncertainty cannot
change the failed gate decision.

The generated evidence remains in the isolated Drive directory
`MyDrive/CropDiseaseAIParallel/field_evaluation_plantdoc_v1`. Review-copy
SHA-256 values:

- `field_metrics.json`: `af0f72f17611b6dd5c430d12b167c66fa00cdb9faa965a1a7df351db7077a81d`
- `field_per_class_metrics.csv`: `a37b15a15869af426a0d4ed955e833be90f8189b7c6025923e8e1419c18b0895`
- `field_top_confident_errors.csv`: `6da02492d6f7d6bd6d88fdfdb2d3a1da238e44f8f8e218c996a12676888b49a4`
- `field_predictions.csv`: `7c015dba1d2f2dc67d94013338cde9915717d44f168f5678ac768e2b4a15322c`
- `field_confusion_matrix.png`: `7c1825d7a4f8a4028b9f931800a7b85b0be7fc317667a3caddd9634f9a5918ac`

PlantDoc's test split is now a consumed evaluation set for this development
line. It may continue to be reported as a fixed benchmark, but it must not be
used to tune thresholds or make image-by-image training decisions. A revised
candidate needs field-domain training data plus a different untouched field
set for its final approval gate.

## Realistic unsupported-image run

Place a separately sourced, license-compatible suite under a Drive directory.
It should include non-leaf objects, soil, tools, hands, animals, indoor scenes,
food, damaged leaves outside the supported crops, and unsupported crop species.
Keep a source/permission inventory for every image collection.

```bash
cd /content/crop-diseasee-ai-parallel
python training/evaluate_candidate.py \
  --mode ood \
  --model /content/drive/MyDrive/CropDiseaseAIParallel/classifier_candidate_v1/crop_classifier.onnx \
  --labels /content/drive/MyDrive/CropDiseaseAIParallel/classifier_candidate_v1/labels.json \
  --manifest /content/drive/MyDrive/CropDiseaseAIParallel/classifier_candidate_v1/candidate_manifest.json \
  --dataset-root /content/drive/MyDrive/CropDiseaseAIParallel/realistic_ood_v1 \
  --output-dir /content/drive/MyDrive/CropDiseaseAIParallel/ood_evaluation_v1
```

Every accepted unsupported image must be inspected. The candidate remains
`production_approved: false` until field and realistic OOD criteria are defined,
measured, reviewed, and recorded.

The predefined OOD gate requires at least 500 source-documented images and no
more than 1% false acceptance under the full production quality-plus-confidence
decision. Suite composition and all accepted examples must also be reviewed.
