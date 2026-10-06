Build and deploy a complete web application called **Crop Disease AI**. Work through implementation, testing, and deployment—not just planning or generating sample code.

I want all development, model training, testing, and hosting to happen online using cloud environments and my available connected plugins. My computer should only need a browser.

Our previous project approach used GitHub, cloud development, Vercel, and Render. Follow that approach where appropriate, while inspecting which capabilities and connected accounts are actually available in this Codex session. Create separate resources for this project.

**1. Work autonomously within the project scope.**

First inspect the available workspace, repository access, connected plugins, deployment services, and cloud compute capabilities.

Use connected plugins, APIs, and supported tools where available. Read applicable project instructions and skills. Check current official documentation when configuring external services.

Create a new repository named `crop-disease-ai` if possible, preferably private initially. Keep code, requirements, setup instructions, model documentation, and progress in the repository.

Make reasonable implementation decisions and continue without repeatedly asking me to approve routine coding steps. Ask concise questions only for missing access, credentials, a consequential ambiguity, or a new billable resource. Explain the exact blocker and continue independent work wherever possible.

Do not assume a connected plugin or account is available. Do not modify my other projects. Obtain approval before starting new paid subscriptions or billable resources, with the proposed cost explained.

**2. Use this architecture unless you identify a concrete reason to change it.**

- Frontend: React, Vite, TypeScript, and Tailwind CSS.
- Backend: Python and FastAPI.
- Disease classification: a specialist crop-image model, preferably an EfficientNet-B0 model fine-tuned using PyTorch.
- Production inference: exported model with ONNX Runtime if compatible and beneficial after benchmarking.
- Severity estimation: a separate validated segmentation or severity model.
- Charts: Recharts or similarly lightweight components.
- History: localStorage.
- Translation: reviewed language resources and Azure Translator where needed.
- Speech: Azure AI Speech or another documented cloud TTS service with verified English, Telugu, and Hindi support.
- Weather: Open-Meteo or a suitable documented weather API.
- Code and version control: GitHub.
- Development and testing: Codex Cloud.
- GPU training, when needed: Google Colab or another accessible hosted GPU environment.
- Hosting: Vercel for the frontend and Render for the backend.

Do not assume Codex Cloud provides GPU training resources. Prepare cloud training notebooks and use an accessible cloud runtime. If training access is missing, explain that blocker.

A server database is unnecessary for the requested browser-local history. Keep the architecture simple.

**3. Keep the application focused and farmer-friendly.**

Support these crops:

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

Build these seven pages:

| Page | Required functionality |
|---|---|
| Home | Upload Image, Take Photo, clear instructions, and Analyze |
| Analyze | Preview the uploaded/captured image, replace or retake it, and run analysis |
| Result | Complete conditional AI report, weather, visualizations, language selection, Listen, and Stop |
| Dashboard | Simple summaries and visualizations derived from saved analyses |
| History | Save, reopen, and delete previous analyses using localStorage |
| Supported Crops | Show the 14 crops, exact supported diseases, and coverage limitations |
| About | Explain the application, how analysis works, privacy, and limitations |

Use a clean agricultural visual style, readable text, large touch targets, clear icons, and responsive layouts. Show severity using words and icons as well as color.

Do not add login, accounts, marketplace, chat, payments, admin screens, or unrelated features.

**4. Implement real image analysis and honest coverage.**

The model must analyze the decoded visual image. Exclude the filename, extension-based crop guesses, upload metadata, and user interface labels from disease inference.

For example, a diseased grape photograph renamed `tomato.jpg` must still be evaluated visually.

Never use random results, filename matching, hardcoded predictions, mock API responses, or placeholder scores in the actual application. A pretrained architecture with generic weights is not a trained crop-disease classifier.

Begin with an AI feasibility milestone before polishing the full interface:

- Identify the model, dataset, licensing, and exact crop–disease taxonomy.
- Run real inference on representative images.
- Evaluate healthy samples, diseased samples, unrelated images, and unsupported plants.
- Verify filename independence.
- Identify missing training categories and uncertain cases.

PlantVillage is a useful starting point, but its standard taxonomy has important gaps:

- Blueberry, Raspberry, and Soybean have healthy classes only.
- Orange and Squash lack healthy classes.
- Pepper coverage is specifically bell pepper.
- The dataset covers selected conditions, not every disease affecting the 14 crops.

Verify these details against the dataset version used. Obtain suitable licensed, verified data for missing categories. Do not advertise disease coverage that has not been implemented and evaluated.

Add image-quality checks and a supported-leaf/unsupported-image rejection approach. Assess confidence and unfamiliar-image behavior using held-out examples; a high classification score alone does not prove an image belongs to a supported category.

Keep images of the same physical leaf and duplicate/augmented versions within the same dataset partition. Use separate training, validation, and test data. Include independent field photographs where possible.

Report measured performance, including per-class results, confusion matrices, uncertainty behavior, and known limitations. Do not invent accuracy claims.

If considering an existing specialist model or external inference API, verify its license, actual coverage of the required crops, response semantics, costs, and performance before adopting it. Keep the integration replaceable.

**5. Implement the three result states exactly.**

For an accepted diseased crop, display:

- Crop name.
- Predicted disease name.
- Plant condition.
- Disease rate, when validly measured.
- Severity: Low, Medium, or High, when validly assessed.
- AI confidence.
- At least three relevant symptom descriptions.
- At least three established causes or contributing factors.
- At least three practical recommendations.
- Today’s location-based weather.
- Simple health, severity, confidence, and top-prediction visualizations.

Separate **symptoms observed in the image** from **typical symptoms to check**. If only one symptom is visibly supported, do not invent two additional observations.

Label causes as established disease causes or possible contributing factors. A leaf image cannot prove the farm’s watering history, soil conditions, or exact infection source.

For an accepted healthy crop, display:

- Crop name.
- Healthy status, described as no supported disease detected.
- A health score only when supported by a documented measurement.
- AI confidence.
- At least three crop-care recommendations.
- Disease: None.
- Severity: None.
- Symptoms: None.
- Causes: None.
- Disease rate: None.

For an unsupported or uncertain image, display:

- Crop: Undefined.
- Disease: Undefined.
- Condition: Unknown.
- Disease rate: Undefined.
- Severity: Undefined.

Explain why analysis is uncertain when possible and suggest a clearer photograph or another leaf. Do not show a confident disease diagnosis or disease-specific treatment in this state.

Keep API/service failures separate from unknown-image results. An unavailable model must produce a clear technical error and retry option.

**6. Define scores honestly and implement genuine visualizations.**

AI confidence, disease severity, and crop health are different measurements.

Define “disease rate” as the **estimated percentage of visible photographed leaf area affected**, where a validated measurement is available. Do not present it as infection prevalence across the entire crop or field.

Use annotated lesion/leaf masks or verified severity labels to build and evaluate the severity component. Map Low/Medium/High using documented, justified thresholds; use disease-specific interpretation where necessary.

Do not calculate disease rate from model confidence. Do not automatically assign a healthy prediction a 100/100 health score. Do not treat a Grad-CAM heatmap as a validated lesion mask.

A visual health score may represent estimated visibly unaffected leaf area, but label its scope clearly. It must not imply whole-plant health or yield prediction.

If a score cannot be supported, display “Unable to estimate” and explain the limitation. Do not fill the field with an arbitrary number merely to complete a chart. Treat missing severity coverage as an outstanding requirement.

Implement:

- A visual leaf-health gauge where a valid score exists.
- A Low/Medium/High severity indicator.
- An AI confidence bar.
- A top-prediction chart using actual model outputs.
- An optional history health-score chart using valid saved measurements.

Do not renormalize only the top predictions to create misleading percentages. Missing measurements must remain missing rather than becoming zero.

Dashboard charts must reflect this browser’s actual saved reports. Clearly distinguish healthy, diseased, and unknown results. Do not imply that these scans represent regional disease prevalence.

**7. Make English, Telugu, and Hindi translation and audio functional.**

On the Result page provide:

**English | తెలుగు | हिन्दी | 🔊 Listen | Stop**

Selecting a language must translate the complete report, including labels, condition, disease descriptions, symptoms, causes, recommendations, uncertainty explanations, weather descriptions, and chart labels.

Maintain the same underlying prediction and numerical values when languages change. Translation must not rerun diagnosis or invent recommendations.

Use verified cloud text-to-speech support. Example Azure voice identifiers to confirm against current documentation are:

- English: `en-IN-NeerjaNeural`.
- Telugu: `te-IN-ShrutiNeural`.
- Hindi: `hi-IN-SwaraNeural`.

Do not rely exclusively on browser speech synthesis, because device voice availability varies.

Listen must speak the complete meaningful report in the selected language. Handle long reports through ordered audio chunks if necessary. Stop must immediately stop playback and cancel queued audio. Changing language must stop the previous playback.

Show audio loading, playing, stopped, and failure states. Allow retry after failure. Audio failure must not prevent the user from reading the report.

Use UTF-8 and fonts that correctly display Telugu and Devanagari. Review agricultural terminology and preserve numbers, units, and scientific names correctly.

Keep speech and translation credentials on the server. Document the configured provider and verify all three languages with real generated audio.

**8. Implement camera, weather, history, and error handling properly.**

Support image upload and camera capture from compatible mobile and desktop browsers. Request camera permission only when the user selects Take Photo. Prefer the rear camera on mobile where supported.

Provide preview, retake, replace, and upload fallback. Handle denied permission, missing cameras, unreadable images, unsupported formats, excessive file sizes, and poor-quality photographs.

Validate image contents server-side, normalize orientation, and apply consistent preprocessing. Use HTTPS in deployment.

For weather, request optional location permission or let the user select a town manually. Do not assume the photograph identifies the location.

Display temperature, humidity, weather conditions, relevant rain information, location, and timestamp from the actual provider. Explain unavailable weather clearly and allow disease analysis to continue. Weather must not be presented as proof of disease causation.

Save compact analysis reports, timestamps, model versions, and small thumbnails in localStorage. Avoid saving full-resolution images or audio there. Handle quota errors and unavailable storage gracefully.

History must survive refresh in the same browser, reopen saved reports, and support deleting individual records or clearing history with confirmation. Explain that browser history does not synchronize between devices and can disappear when browser data is cleared.

Clearly distinguish historical weather saved with a report from weather fetched today.

Use a reviewed disease-information resource with source links for symptoms, causes, and recommendations. Avoid unverified chemical prescriptions or location-inappropriate pesticide dosages.

**9. Complete meaningful verification and cloud deployment.**

Test and document:

- The same image under different filenames produces the same prediction.
- Representative healthy and diseased examples for every advertised class.
- Unsupported plants, non-leaf objects, unfamiliar conditions, and blurred images.
- Healthy results show None for all required disease-related fields.
- Unknown results use the required Undefined/Unknown values.
- Classification confidence does not drive disease-rate or severity values.
- English, Telugu, and Hindi text and actual audio work.
- Stop and language switching correctly interrupt playback.
- Camera capture, upload fallback, and permission errors.
- History persistence, reopening, deletion, and storage failure.
- Weather location handling and provider failure.
- Backend/model failure without fabricated results.
- Mobile layouts, keyboard access, and readable charts.
- Production frontend-to-backend integration.

Use real labeled test images for AI evaluation. Distinguish automated browser simulations from actual device verification, and disclose anything that remains untested.

Configure secrets through appropriate environment or secret settings. Never expose credentials in frontend bundles, source control, logs, or screenshots. Add reasonable upload limits, request timeouts, rate limits, and CORS configuration for the deployed frontend.

Benchmark backend memory and inference latency before selecting hosting resources. Avoid downloading or loading the model on every request. Prefer affordable resources within the approved budget, but do not silently provision paid services.

Deploy using available connected Vercel and Render capabilities, or explain a concrete access blocker. Verify the public application using real API responses after deployment.

**10. Deliver a usable project and a truthful completion report.**

Provide:

- Repository URL.
- Live frontend URL and backend health endpoint.
- A working application with the seven required pages.
- Actual model artifacts or a fully configured, documented inference integration.
- Cloud training/evaluation notebooks and reproducible instructions.
- Exact supported crop–condition list.
- Model and dataset licenses and attribution.
- Model version, measured evaluation results, and limitations.
- Required environment-variable names and setup instructions without secret values.
- A concise README explaining how to run, train, test, and redeploy entirely online.
- Verification results and an explicit list of any remaining blockers.

Keep durable requirements and progress in repository files so future Codex sessions can continue from the actual project state.

Do not describe a polished interface as a completed project if inference, voice, required coverage, or deployment is still missing. Continue implementing authorized work until the requested outcome is achieved or a genuine external blocker remains.

Begin now by inspecting available access and cloud resources, saving the requirements, and implementing the real-image AI feasibility milestone.