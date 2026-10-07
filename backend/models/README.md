# Candidate-v2 model artifacts

The experimental Candidate-v2 binary is intentionally excluded from this public
Git repository. Keep the authoritative files in the project owner's access-
controlled Drive folder, download them into a private local directory, then run:

```bash
python backend/install_candidate_v2.py /path/to/private/classifier_candidate_v2
```

The installer verifies the fixed model version, experimental status, label
count/order (by SHA-256), and every runtime artifact before copying:

- `crop_classifier.onnx`
- `labels.json`
- `candidate_manifest.json`

The same verification runs again when the API starts. The API remains
unavailable until all three exact files exist. Do not substitute ImageNet
weights, random weights, filename rules, or placeholder predictions.

This delivery method keeps the public repository reproducible without making
the model binary public. The owner controls Drive permissions and can revoke or
replace access without rewriting Git history.
