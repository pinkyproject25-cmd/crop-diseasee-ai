# Candidate v2 evaluation

Evaluated: 2026-10-07 (Asia/Kolkata)

Model: `mixed-pv-plantdoc-mnv3-20261006T194137Z`

Status: **candidate only; production_approved remains false**

## Provenance

- Architecture: MobileNetV3 Small.
- Output classes: 38 PlantVillage crop/disease classes across 14 crop groups.
- PlantVillage revision: `9e97599868962bd0079b8db4b7f1efa9185fa1e7`.
- PlantDoc revision: `5467f6012d78d1c446145d5f582da6096f852ae8`.
- Reviewed PlantDoc training manifest: 2,278 images.
- Candidate-v2 training: eight fixed mixed-domain fine-tuning epochs, using 75% PlantVillage and 25% reviewed PlantDoc sampling.
- Selected checkpoint: epoch 8 by the predefined equal-domain validation macro-F1 score.
- Candidate metrics SHA-256: `82175598c6c89f05541424960a23ce5b4c64aeb96a9b2ae810df6e793092abd5`.
- Field metrics SHA-256: `42e93abccdeed8d22730c97be8353969638984b50077140efdca969ecbcbfdc5`.

The official PlantDoc test split was previously viewed for Candidate v1. Candidate-v2 training did not read it, but this result is treated as a comparative held-out benchmark, not a new untouched production-approval gate.

## Training and controlled results

- PlantVillage validation: 99.19% accuracy, 98.85% macro-F1.
- PlantVillage controlled regression benchmark: 99.15% accuracy, 98.73% macro-F1.
- PlantDoc internal validation: 60.70% accuracy, 57.80% macro-F1 across 27 present classes.
- PlantDoc internal calibration: 61.78% accuracy, 59.86% macro-F1 across 27 present classes.
- ONNX verification: 8/8 top-1 agreement; maximum absolute logit error `1.7643e-05`.
- CIFAR-100 provisional false acceptance at threshold 0.845: 0.97%.

Compared with the Candidate-v1 PlantDoc internal baseline, Candidate-v2 raised internal field macro-F1 from 16.32% to 57.80% while retaining high PlantVillage performance.

## PlantDoc test comparison

All 236 pinned test images decoded successfully.

| Metric | Candidate v1 | Candidate v2 |
|---|---:|---:|
| Top-1 accuracy without rejection | 25.42% | 57.20% |
| Macro-F1 without rejection | 24.61% | 56.30% |
| Coverage at fixed threshold | 9.75% | 36.02% |
| Accepted accuracy | 52.17% | 85.88% |
| Accepted-accuracy Wilson lower 95% | 32.96% | 76.93% |
| Accepted errors | 11 | 12 |

Candidate-v2 accepted 85 images at its fixed threshold of 0.845: 73 were correct and 12 were incorrect. It rejected 151 images, including 14 quality rejections.

## Gate decision

The predefined field gate required all of:

- coverage at least 30%;
- accepted accuracy at least 95%; and
- accepted-accuracy Wilson lower 95% at least 90%.

Candidate-v2 passed only the coverage requirement. The field gate therefore **failed**. Raising the threshold after viewing this test would be test-set tuning and is not permitted. Diagnostic checks also show high-confidence mistakes, so confidence alone does not resolve the remaining domain problem.

Prominent accepted confusions include:

- corn Northern Leaf Blight versus corn Gray Leaf Spot;
- potato Early Blight versus potato Late Blight;
- tomato Bacterial Spot versus tomato Septoria Leaf Spot;
- visually similar healthy leaves across crops.

Some PlantDoc filenames suggest possible source-label anomalies, but they are not silently relabeled and do not change the failure decision.

## Appropriate use

Candidate-v2 is suitable for a **controlled B.Tech academic prototype demonstration** when all of the following are enforced:

- clearly label the model experimental and not production approved;
- present results as decision support, not a definitive agricultural diagnosis;
- return Unknown/Undefined below the fixed threshold or when image quality fails;
- do not claim reliable real-world performance across all 14 crops;
- do not infer labels from filenames;
- do not fabricate severity, symptoms, causes, or treatment;
- use carefully documented demonstration images and disclose the field benchmark results.

It is not scientifically justified for unrestricted farmer-facing deployment, automatic treatment decisions, or production approval.

## Remaining scientific work

1. Evaluate a different untouched licensed field-photo dataset.
2. Build a realistic unsupported-crop, non-leaf, and non-plant OOD suite.
3. Improve weak field classes with additional audited data, especially confusing corn, potato, pepper, and tomato diseases.
4. Validate affected-area segmentation and severity separately.
5. Complete agricultural review of symptoms, causes, and recommendations.
6. Keep deployment and backend model integration blocked unless an explicit non-production academic-demo mode is designed and approved.
