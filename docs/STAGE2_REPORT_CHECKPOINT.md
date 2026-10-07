# Stage 2 result-report checkpoint

Date: 2026-10-07 (Asia/Kolkata)

## Scope and ownership

- Writable repository: `pinkyproject25-cmd/crop-diseasee-ai` only.
- Authenticated GitHub login was rechecked as `pinkyproject25-cmd`; repository
  permissions reported `push: true` and `admin: true`.
- No Ashish-owned repository, deployment, credential, or artifact was changed.
- Candidate-v2 remains experimental and `production_approved: false`.

## Completed

- Commit `a3e4bc4` expands exact-label, source-linked English knowledge from 3
  deployed entries to all 26 diseased Candidate-v2 labels.
- Knowledge is added only for an accepted diseased classification. Healthy and
  Unknown responses keep empty knowledge arrays. `observedSymptoms` stays
  empty because the classifier does not localize symptoms.
- Recommendations avoid pesticide products/doses and direct users to local
  agricultural confirmation.
- Evaluation-only affected-area arithmetic now defines lesion pixels inside
  the visible photographed leaf divided by visible-leaf pixels. It is not wired
  to the API because no rights-cleared, same-image leaf/lesion mask dataset has
  passed the documented evaluation gate.
- Fixed the test-fixture availability check so running from the repository root
  no longer mistakes the current directory for a fixture directory.

## Verification

- Candidate-v2 local artifact hashes remain:
  - ONNX: `864dd9f77c01e6b4f77acfc1e0446b49c39855f31d5aed65c0d2a9a3b9ad4bb7`
  - labels: `16925c0cb74cd5be219eb9e8cf87f0fa4f5c255f49b8db7c865c55050cb7d307`
- Local labels: 38 unique outputs, 26 diseased labels, exact set equality with
  the 26 knowledge keys.
- Backend: 19/19 tests passed from `backend/` with pinned licensed fixtures and
  real Candidate-v2 artifacts. Cases include accepted healthy/diseased,
  low-confidence Unknown, blurred Unknown, filename rename invariance,
  model-unavailable 503, API/frontend contract, exact knowledge selection, and
  affected-area mask arithmetic.
- Frontend: `npm run check` and `npm run build` passed. Vite reported only the
  existing large-chunk warning.

## Deployment state at checkpoint creation

- Pinky frontend: `https://crop-diseasee-ai.vercel.app` (last verified before
  this batch on main commit `dfab869`).
- Pinky backend: `https://crop-diseasee-ai.onrender.com` (existing Render
  service tracks `deploy-pinky-prototype`; last verified live commit before
  this batch was `d7e9236`).
- The changes in `a3e4bc4` were not yet live when this checkpoint text was
  written. Update this section only after the existing deployments are tested.

## Remaining limitations and next step

- Affected area, severity, and health score remain `null` and must not be
  inferred from classifier confidence.
- Knowledge requires regional/agricultural review and translation validation.
- Candidate-v2 retains its measured field limitations; it is not production
  approved.
- Next: deploy the tested knowledge-only backend change through the existing
  Pinky deployment branch, smoke-test live health and accepted/Unknown results,
  then obtain or create rights-cleared paired leaf/lesion masks before any live
  affected-area integration.
