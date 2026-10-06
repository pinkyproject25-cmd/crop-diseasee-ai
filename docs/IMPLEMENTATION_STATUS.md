# Implementation status

## Completed

- Private online project and durable project specification.
- React + Vite + TypeScript frontend.
- Seven routes: Home, Analyze, Result, Dashboard, History, Supported Crops, and About.
- Upload and mobile camera capture input with preview, format checks, and a 10 MB client limit.
- Strict API client with no fake/demo prediction fallback.
- Diseased, healthy, and unknown result layouts.
- Separate fields for confidence, visible disease rate, severity, and health score.
- Top-prediction chart using original probabilities.
- Browser-local report history, reopen, delete, and clear flows.
- Local dashboard with healthy, diseased, and unknown summaries.
- Language selection and server-generated speech controls, including immediate Stop.
- FastAPI backend contract, image decoding, filename-independent pixel preprocessing, basic photo-quality rejection, model threshold handling, weather integration, Azure translation, and Azure speech.
- Docker packaging for the API.
- Private GitHub source repository with generated files, local secrets, caches, and unreviewed model binaries excluded.
- Git-linked Vercel production frontend with direct-route fallback verified for every application page.
- Render FastAPI service deployed in Singapore with automatic deploys from `main`.
- Production frontend configured with the deployed API origin.

## Scientific gates before the application can claim real production analysis

- Train and export the crop classifier.
- Measure class-wise performance on a leaf-group-independent split.
- Evaluate field photographs and unsupported/out-of-distribution images.
- Calibrate the acceptance threshold using validation data.
- Obtain segmentation/severity annotations and validate visible affected-area measurement.
- Add reviewed, cited disease information for every production-supported class.
- Validate Telugu and Hindi agricultural terminology with a fluent reviewer.

## Current deliberate behavior

When the model files are absent, the API returns HTTP 503 and states that no prediction was generated. When a candidate class is returned without the reviewed information and severity layers, the API also refuses to construct an incomplete disease report.

The hosted API process and `/health` endpoint are live. Model readiness remains false until the scientific gates above are completed.

This behavior prevents the polished interface from being mistaken for a completed AI system.
