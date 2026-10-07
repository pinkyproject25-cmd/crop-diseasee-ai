# Stage 1 academic-prototype checkpoint

Recorded: 2026-10-07 UTC

## Scope and isolation

- Repository: `pinkyproject25-cmd/crop-diseasee-ai` only.
- Source `main` inspected at `900f0a3f31f436f134c5839efcad83a7946f13e5`.
- Stage 1 implementation commit: `c973ec081ae4b301989ae4cecc9306422daf1082`.
- The original `Ashish7386/crop-disease-ai` repository and all original deployments were not modified or accessed for writes.
- No deployment or paid resource was created. `production_approved` remains `false`.

## Verified Candidate-v2 artifacts

Authoritative source: the owner's access-controlled
`MyDrive/CropDiseaseAIParallel/classifier_candidate_v2/` folder.

| Artifact | SHA-256 |
|---|---|
| `crop_classifier.onnx` | `864dd9f77c01e6b4f77acfc1e0446b49c39855f31d5aed65c0d2a9a3b9ad4bb7` |
| `labels.json` | `16925c0cb74cd5be219eb9e8cf87f0fa4f5c255f49b8db7c865c55050cb7d307` |
| `candidate_manifest.json` | `357efbe591ca6e960acb5bf79e1211d164ec63bbfbd10244be40924db84d8c15` |
| `metrics.json` | `82175598c6c89f05541424960a23ce5b4c64aeb96a9b2ae810df6e793092abd5` |

The manifest and metrics confirm:

- model version `mixed-pv-plantdoc-mnv3-20261006T194137Z`;
- MobileNetV3-Small, 38 unique ordered combined labels across 14 crop groups;
- dynamic-batch ONNX input `image` with shape `[batch, 3, 224, 224]`;
- ONNX output `logits` with shape `[batch, 38]`;
- temperature `T=0.9030987620353699` embedded by exporting the calibrated wrapper;
- fixed acceptance threshold `0.845`; and
- experimental candidate only, with `production_approved: false`.

The recorded export parity evidence remains 8/8 top-1 agreement with maximum
absolute logit error `1.7642974853515625e-05`. The Stage 1 runtime also loaded
the exact hashed ONNX and reproduced its saved PlantDoc labels/confidences on
the pinned fixtures. The backend applies softmax directly to the exported
outputs and does **not** apply temperature scaling again.

## Implemented behavior

- Startup verifies the exact ONNX, labels, and manifest hashes, candidate
  version/status, threshold, 38-label uniqueness, and ONNX input/output contract.
- An accepted combined label is parsed into real `crop`, `condition`, and
  `disease` fields. Healthy labels return `disease: null`.
- `confidence` is the actual top-1 probability and `topPredictions` contains the
  five original, non-renormalized combined-label probabilities.
- Image-byte decoding, EXIF transpose, RGB conversion, ImageNet normalization,
  quality rejection, threshold rejection, and technical 503 behavior remain in place.
- Accepted reports explicitly return `null` for affected area, severity, and
  health score. Symptoms, causes, and recommendations are empty rather than fabricated.
- The result UI labels the model experimental/not production approved and
  renders unavailable measurement/knowledge fields honestly.
- The public repository excludes runtime artifacts. The owner downloads the
  three runtime files from private Drive and uses
  `backend/install_candidate_v2.py`, which verifies every hash before copying.

## Local verification results

Pinned PlantDoc revision: `5467f6012d78d1c446145d5f582da6096f852ae8`
(CC-BY-4.0). Fixture paths and SHA-256 values are fixed in
`backend/tests/fetch_stage1_fixtures.py`.

| Test | Actual result |
|---|---|
| Healthy field image | `Grape`, `Healthy`, confidence `0.9999256134`, HTTP 200 |
| Diseased field image | `Apple`, `Diseased`, `Apple scab`, confidence `0.9980962873`, HTTP 200 |
| Low-confidence field image | `Unknown`, confidence `0.1253407896`, HTTP 200 |
| Derived blurred image | `Unknown`; quality reason returned, HTTP 200 |
| Renamed-image test | Identical state/crop/condition/disease/confidence/top-five for identical bytes |
| Model unavailable | Honest HTTP 503; no prediction generated |
| Backend response/frontend schema | All backend report fields exist in the TypeScript report contract |
| Backend automated suite | 9/9 tests passed |
| TypeScript check | Passed |
| Frontend production build | Passed; existing bundle-size warning only |
| Artifact installer | Passed and reproduced all three expected hashes |

## Current deployment state

- Not deployed in this stage.
- No Vercel or Render configuration/resource was changed remotely.
- Runtime model files were not committed to Git.

## Known limitations

- Candidate-v2 failed its predefined PlantDoc production field gate: field
  accuracy `57.20%`; accepted accuracy `85.88%` at `36.02%` coverage.
- CIFAR-100 is only a provisional non-leaf OOD proxy; a realistic agricultural
  unsupported-image suite is still absent.
- Affected-area segmentation, severity, and health score are not implemented.
- Symptoms, causes, and recommendations have not completed agricultural review.
- Translation and speech can operate only when their external providers are
  configured; they do not create missing agricultural content.
- Frontend build emits a non-blocking JavaScript chunk-size warning.

## Remaining work after Stage 1

For the agreed academic prototype, the genuine classifier report path is now
implemented and locally verified. Later stages may add separately validated
affected-area/severity estimation and reviewed agricultural knowledge. Broader
field/OOD research is required before any production claim or unrestricted use.
