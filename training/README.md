# Online model training

The classifier pipeline is designed for a hosted GPU runtime such as Google Colab. It downloads the official `mohanty/PlantVillage` color data through the repository's reviewed `plant_village.py` loader, preserves the dataset's leaf-group-independent test split, creates a group-aware validation fold from `leaf_id`, fine-tunes MobileNetV3 Small, calibrates its temperature, runs a basic CIFAR-100 rejection sanity check, and exports ONNX. The pipeline pins Hugging Face Datasets 3.6 because Datasets 4.x no longer runs repository loader scripts.

```bash
pip install -r training/requirements.txt
python training/train_classifier.py --output-dir artifacts/classifier
```

Expected candidate artifacts:

- `crop_classifier.onnx`
- `labels.json`
- `metrics.json`

The output is deliberately marked `production_approved: false`. PlantVillage is largely a controlled-background dataset. A strong PlantVillage test result cannot establish field performance, unsupported-image rejection, or disease severity. Do not place these artifacts in `backend/models` until the blockers listed in `metrics.json` have been resolved and recorded.

Dataset sources used by the validation plan:

- PlantVillage: 14 crops, 38 healthy/disease classes, leaf-group metadata.
- PlantDoc: independent field photographs for external classification evaluation where labels can be mapped safely.
- PlantSeg: in-the-wild lesion masks for the separate severity pipeline.
