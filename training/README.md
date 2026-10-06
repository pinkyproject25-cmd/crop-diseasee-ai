# Online model training

The classifier pipeline is designed for a hosted GPU runtime such as Google Colab. It downloads the official `mohanty/PlantVillage` color data through the repository's reviewed `plant_village.py` loader and pins the dataset revision in the run metrics. A live audit found overlapping physical `leaf_id` groups in the loader's published train/test partitions, so the pipeline combines those source partitions and creates a fresh 10-fold stratified group split. Separate folds are reserved for model-selection validation, calibration, and untouched testing; the remaining seven folds are used for training. It fine-tunes MobileNetV3 Small, fits temperature and the provisional threshold only on the calibration subset, evaluates the selected threshold on the untouched custom test subset, runs a basic CIFAR-100 rejection sanity check, and exports ONNX. The pipeline pins Hugging Face Datasets 3.6 because Datasets 4.x no longer runs repository loader scripts.

```bash
pip install -r training/requirements.txt
python training/train_classifier.py --output-dir artifacts/classifier
```

Every completed epoch writes `last_checkpoint.pt`; rerunning the same command resumes automatically. Use a new output directory for a separate experiment. `--no-resume` intentionally starts at epoch 1 but does not delete existing files.

Expected candidate artifacts:

- `crop_classifier.onnx`
- `labels.json`
- `metrics.json`
- `candidate_manifest.json`
- `test_per_class_metrics.csv`
- `test_confusion_matrix.csv`
- `test_confusion_matrix.png`
- `training_history.png`
- `best_model.pt`
- `last_checkpoint.pt`

The output is deliberately marked `production_approved: false`. PlantVillage is largely a controlled-background dataset. A strong PlantVillage test result cannot establish field performance, unsupported-image rejection, or disease severity. CIFAR-100 is only a provisional non-leaf sanity check, not a realistic agricultural OOD benchmark. Do not place these artifacts in `backend/models` until the blockers listed in `metrics.json` have been resolved and recorded.

Dataset sources used by the validation plan:

- PlantVillage: 14 crops, 38 healthy/disease classes, leaf-group metadata.
- PlantDoc: independent field photographs for external classification evaluation where labels can be mapped safely.
- PlantSeg: in-the-wild lesion masks for the separate severity pipeline.

The next gate is implemented by `evaluate_candidate.py`. See
`docs/FIELD_EVALUATION.md` for the pinned PlantDoc test procedure, explicit class
mapping, artifact-integrity checks, and realistic unsupported-image workflow.

Candidate v1 failed that field gate. Before any candidate-v2 training, run
`audit_plantdoc.py` against the pinned PlantDoc repository. It validates image
decoding and mappings, detects exact pixel duplicates and train/test leakage,
flags perceptual near-duplicates, and writes a reviewable training manifest.
See `docs/CANDIDATE_V2.md` for the CPU-only Colab procedure. The generated clean
manifest is not approved for training until its review artifacts are examined.

After review, `finalize_plantdoc_manifest.py` produces the hash-pinned
`reviewed_train_manifest.csv`. Candidate v2 uses a separate entry point:

```bash
python training/train_classifier_v2.py --help
```

Run its documented `--preflight-only` command on CPU before allocating a GPU.
The preflight validates every PlantDoc source-file hash, the Candidate-v1
initialization inputs, and deterministic perceptual-group splits. The fixed
experiment uses 1,822 reviewed PlantDoc training images, 228 internal
model-selection images, and 228 calibration images. It never reads PlantDoc
test. Candidate-v2 artifacts remain non-production and require a different
untouched field-photo gate.
