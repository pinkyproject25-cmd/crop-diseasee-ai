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
