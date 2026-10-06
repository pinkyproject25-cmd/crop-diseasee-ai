from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .config import settings
from .language import report_to_speech_text, synthesize_speech, translate_report
from .model import InvalidImageError, ModelUnavailableError, classifier
from .schemas import AnalysisReport, SpeechRequest, TranslateRequest
from .weather import fetch_weather


@asynccontextmanager
async def lifespan(_: FastAPI):
    classifier.load()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_ready": classifier.ready,
        "model_version": settings.model_version,
    }


@app.post("/api/v1/analyze", response_model=AnalysisReport)
async def analyze(
    image: UploadFile = File(...),
    latitude: float | None = Form(default=None),
    longitude: float | None = Form(default=None),
) -> AnalysisReport:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP image.")
    image_bytes = await image.read(settings.max_image_bytes + 1)
    if len(image_bytes) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="The image exceeds the 10 MB upload limit.")
    try:
        output = classifier.predict(image_bytes)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail="The validated AI model is not available. No prediction was generated.") from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    top = output.predictions[0]
    weather = await fetch_weather(latitude, longitude)
    now = datetime.now(timezone.utc)

    if min(output.width, output.height) < 224 or output.sharpness < 25:
        return AnalysisReport(
            id=str(uuid4()), createdAt=now, modelVersion=settings.model_version,
            state="unknown", crop=None, disease=None, condition="Unknown",
            confidence=top.probability, diseaseRate=None, severity=None, healthScore=None,
            observedSymptoms=[], typicalSymptoms=[], causes=[],
            recommendations=["Retake the photo in good light.", "Keep one leaf sharp and close to the camera.", "Try both sides of the leaf."],
            uncertaintyReason="The image is too small or blurred for reliable analysis.",
            topPredictions=output.predictions, weather=weather,
        )

    if top.probability < settings.acceptance_threshold:
        return AnalysisReport(
            id=str(uuid4()), createdAt=now, modelVersion=settings.model_version,
            state="unknown", crop=None, disease=None, condition="Unknown",
            confidence=top.probability, diseaseRate=None, severity=None, healthScore=None,
            observedSymptoms=[], typicalSymptoms=[], causes=[],
            recommendations=["Use one leaf in natural light.", "Move closer while keeping the leaf in focus.", "Consult a local expert if symptoms are progressing."],
            uncertaintyReason="The image did not meet the validated confidence threshold for a supported class.",
            topPredictions=output.predictions, weather=weather,
        )

    raise HTTPException(
        status_code=503,
        detail=(
            "The classifier returned a supported candidate, but the reviewed disease-information "
            "and severity layers are not installed. No complete report was generated."
        ),
    )


@app.post("/api/v1/reports/translate", response_model=AnalysisReport)
async def translate(payload: TranslateRequest) -> AnalysisReport:
    try:
        return await translate_report(payload.report, payload.language)
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=502, detail="Translation provider request failed.") from exc


@app.post("/api/v1/reports/speech")
async def speech(payload: SpeechRequest) -> Response:
    try:
        report = payload.report
        if payload.language != report.language:
            report = await translate_report(report, payload.language)
        audio = await synthesize_speech(report_to_speech_text(report, payload.language), payload.language)
        return Response(content=audio, media_type="audio/mpeg")
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=502, detail="Speech provider request failed.") from exc
