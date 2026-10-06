# Crop Disease AI

Crop Disease AI is a responsive crop-leaf screening application for 14 target crops. It accepts an uploaded or captured image, sends the decoded image to a specialist inference API, and presents one of three explicit states: healthy, diseased, or unknown.

This repository is the isolated parallel instance owned by `pinkyproject25-cmd`. It must not be connected to, deployed over, or synchronized with `Ashish7386/crop-disease-ai` or that project's Vercel, Render, or Colab resources.

## Current implementation status

The responsive seven-page frontend and production API contract are implemented. The API refuses to generate a prediction until validated ONNX model artifacts are installed. This is intentional: filename rules, randomized results, generic ImageNet weights, and placeholder scores are prohibited.

The remaining scientific gate is model development and validation:

1. Train and evaluate the 38-class PlantVillage baseline with leaf-group-aware splits.
2. Add licensed data for baseline gaps and independent field photographs.
3. Calibrate confidence and validate the unsupported-image rejection rule.
4. Train or obtain separately labeled leaf/lesion segmentation data for disease rate and severity.
5. Add reviewed, sourced disease knowledge for every advertised production class.

See [PROJECT_SPEC.md](docs/PROJECT_SPEC.md) and [IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md).

## Frontend

Requirements: Node.js 22+.

```bash
npm install
npm run dev
```

Set `VITE_API_BASE_URL` to the deployed FastAPI origin. The frontend does not fall back to sample predictions when this variable is missing.

## Backend

Requirements: Python 3.12.

```bash
cd backend
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Copy `.env.example` to `.env`, then configure it. Secrets must be stored in the deployment platform and must never be committed.

The required production artifacts are:

- `backend/models/crop_classifier.onnx`
- `backend/models/labels.json`

The API health endpoint reports whether the model is loaded:

```text
GET /health
```

## Deployment

- Weather: Open-Meteo current conditions.
- Translation and speech: Azure Translator and Azure AI Speech, using server-side credentials.
- History: browser `localStorage`; no account or database.

This parallel instance is not deployed yet. Any future frontend and API must use newly created Vercel and Render resources dedicated to this repository. Until an independently reviewed model is approved and installed, the API remains intentionally not prediction-ready and returns a technical-unavailable response instead of inventing a result.

For a new hosted GPU run, open the repository notebook in a fresh Colab runtime. It stores resumable candidate checkpoints under `MyDrive/CropDiseaseAIParallel` and keeps `production_approved` false:

https://colab.research.google.com/github/pinkyproject25-cmd/crop-diseasee-ai/blob/main/notebooks/train_classifier_colab.ipynb

## Privacy

Compact reports and small thumbnails are stored only in the current browser. Image retention by the production API must be documented before public launch. The current API performs in-memory request processing and contains no persistence layer.
