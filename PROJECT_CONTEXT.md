# Crop Disease AI — current project context

Checkpoint: 7 October 2026, 13:25 IST. This file records the Pinky-owned academic prototype after the result badge removal. It reconciles the earlier project conversations with the repository and the observed live tests; it is not evidence of production readiness.

## Ownership and resources

- **Writable project:** `pinkyproject25-cmd/crop-diseasee-ai`. Its `main` branch drives the Pinky Vercel frontend; `deploy-pinky-prototype` drives the Pinky Render backend.
- **Frontend:** https://crop-diseasee-ai.vercel.app/ . **Backend:** https://crop-diseasee-ai.onrender.com/ (`GET /health`, `POST /api/v1/analyze`). The public site loaded during this checkpoint; the user recently obtained live results. The Vercel project could not be inspected through this session's connector, which is scoped to another account.
- **Private model delivery:** `pinkyproject25-cmd/crop-diseasee-ai-artifacts` supplies hash-pinned model files to the Render build via `backend/fetch_model_artifacts.py`, `MODEL_ARTIFACT_TOKEN`, and `MODEL_ARTIFACT_REF`. The ONNX binary is not in the public application repository. Do not put the token into Git or this document.
- **Original/reference only:** `Ashish7386/crop-disease-ai` and its hosting, credentials, Colab, and other resources. No synchronization or deletion is authorized. The replacement belongs to the Pinky account.
- **No new paid service** is authorized; the running prototype needs no Supabase account. It has no user login or server-side database.

## Final intended experience and architecture

A user uploads or captures a crop-leaf image and receives one of three reports: **healthy**, **diseased**, or **Unknown**. A diseased report includes crop, disease, model confidence, top-five predictions, typical symptoms, known causes, source-linked next steps, and a clearly identified visual-area measurement when possible. Unknown must not invent a diagnosis. Healthy must not invent disease symptoms.

The seven routes are Home, Analyze, Result, Dashboard, History, Supported Crops, and About. React/Vite/TypeScript and browser storage provide the frontend, FastAPI the API, and ONNX Runtime the crop classifier. Open-Meteo weather and Azure translation/speech have integrations; full multilingual and speech validation remains outstanding. History and dashboard summaries are stored in the current browser, not in a cloud database.

## Current implementation and model

- Candidate-v2 is a genuine image-byte classifier, with 38 ordered outputs for selected conditions across 14 target crops; it does not use filenames to predict. Image-quality rejection and the fixed **0.845** confidence threshold produce Unknown when the input is not accepted. Missing or mismatched model files result in HTTP 503.
- Model version: `mixed-pv-plantdoc-mnv3-20261006T194137Z`. ONNX SHA-256: `864dd9f77c01e6b4f77acfc1e0446b49c39855f31d5aed65c0d2a9a3b9ad4bb7`. Labels SHA-256: `16925c0cb74cd5be219eb9e8cf87f0fa4f5c255f49b8db7c865c55050cb7d307`. Temperature scaling is embedded in ONNX. The manifest still says `production_approved: false`; the visible result badge was removed at the user's request, without altering that flag.
- Its measured field limitations remain: PlantDoc raw accuracy **57.20%**, accepted accuracy **85.88%** at **36.02% coverage** in the earlier evaluation. These figures do not guarantee accuracy on new photos; high-confidence crop errors were observed in the ad hoc review batch.
- Source-linked general knowledge covers all 26 diseased labels, conditional on accepted classification. It is typical guidance, **not** a claim that those symptoms were visually observed. Some generic healthy-care recommendations are possible. Regional agricultural review remains pending.
- The classifier has no lesion/leaf segmentation masks. The optional **user-assisted** tool lets a user outline a visible leaf and mark affected pixels on the actual displayed photo. Affected area = marked pixels inside leaf / outlined visible-leaf pixels × 100. Prototype bands: Low <10%, Medium 10–<30%, High ≥30%; visually unaffected score = 100 − marked area. These are user marks and demonstration rules, not autonomous AI severity or whole-plant health. Automatic color estimation may abstain on field photos, leaving those fields unavailable until the user marks an accepted diseased photo. Healthy/Unknown do not receive invented disease-area figures.
- An offline paired-mask evaluator exists on `stage2-measurement-evaluation` (draft PR #3), but no validated segmentation model was trained or deployed. That work is separate from the working user-assisted prototype.

## Live evidence and limitations

- Earlier live API smoke tests: Apple scab accepted at ~0.9981; Cedar apple rust at ~0.9549; `/health` returned `model_ready: true` on 7 October. Recent user screenshots show Grape Black rot accepted at ~96% with 10.8% **user-marked** area, Medium severity, and 89.2% visual score; Potato Early blight at ~98% with 41.2% user-marked area, High severity, and 58.8% visual score.
- In an ad hoc 17-photo Codespace batch, 5 were accepted and 12 returned Unknown. Potato, Corn Common rust, and Grape Black rot gave plausible non-Apple accepted outputs subject to manual ground-truth confirmation. A file named `rice.jpg` was accepted as Corn Northern Leaf Blight and `orange.jpg` as Apple Cedar apple rust; if those filenames accurately identify the crops, these are high-confidence misclassifications. Do not show them as successes. An apple-fruit image and several field leaves returned Unknown; one image with 89% top probability was rejected by the separate size/blur gate.
- Several screenshots show HTTP 200 `analyze` requests ending in Unknown at 74% and 49%; that is intended rejection, not an API outage. A transient generic frontend “Analysis failed” message appeared while multiple images were tested. The frontend storage ordering and quota fallback were then changed on `main`; the user subsequently showed working Grape and Potato results. The precise cause of the earlier failure was not independently proven.
- The result banner text “Experimental Candidate-v2 · not production approved” has been removed from `src/App.tsx` on Pinky `main`. Other prototype explanations remain; this UI edit did not approve the model.
- Last known frontend type check and production build passed; the longstanding bundle-size warning is non-blocking. Earlier real-model backend suites passed. The latest badge-only change did not touch the backend.

## Persistence and next step

- The code is stored in Pinky GitHub, with frontend changes verified in remote `main`; the public Vercel site loads. Render stays independent of the Codespace and runs its own deployed branch with hash-verified private artifacts. Closing this chat or Codespace does not stop those deployments. A free Render service may sleep and require a cold start.
- **Browser data is different:** History is `localStorage` on that browser/device only, and the active report and draft image use `sessionStorage`. Closing a tab can remove the active report; History should still be available on the same browser unless site data is cleared, private browsing ends, or quota fallback prevents a save. There is no cross-device/cloud sync or report backup.
- The user had a modified `backend/.dockerignore` in Codespace that was not included in the badge-only commit. Its exact diff has not been reviewed; inspect it before discarding or committing it. `review_samples/` and local `.patch` files used during manual testing are not part of the deployment.
- **Immediate approved step:** confirm the last Vercel deployment shows the badge removed, then demonstrate an accepted Apple, Potato, and Grape photo plus an Unknown case using vetted originals. Explain user-assisted marking accurately. Check `git status --short` and `git diff -- backend/.dockerignore` in Pinky Codespace before closing it. No model retraining, threshold lowering, paid resource, or removal of original Ashish assets is approved.

## Documentation conflicts to keep visible

`README.md` and `docs/IMPLEMENTATION_STATUS.md` still say the Pinky instance is not deployed, and the latter describes knowledge/area fields as entirely unavailable. `docs/AFFECTED_AREA_PLAN.md` says area remains null until trained segmentation; this is true for **autonomous segmentation**, but the later `docs/USER_ASSISTED_REPORT.md` and live site add a separately labeled manual marking path. These files reflect earlier checkpoints and should be updated rather than treated as current deployment facts. The original architecture preferred a fine-tuned EfficientNet-B0; the integrated Candidate-v2 is MobileNetV3-based. The experimental status and field gate remain unchanged despite the requested removal of the result badge.
