from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


Language = Literal["en", "te", "hi"]


class Prediction(BaseModel):
    label: str
    probability: float = Field(ge=0, le=1)


class WeatherSnapshot(BaseModel):
    location: str
    observedAt: datetime
    temperatureC: float | None
    humidityPercent: float | None
    precipitationMm: float | None
    description: str


class AnalysisReport(BaseModel):
    id: str
    createdAt: datetime
    modelVersion: str
    state: Literal["healthy", "diseased", "unknown"]
    crop: str | None
    disease: str | None
    condition: str
    confidence: float | None
    diseaseRate: float | None
    severity: Literal["Low", "Medium", "High"] | None
    healthScore: float | None
    observedSymptoms: list[str]
    typicalSymptoms: list[str]
    causes: list[str]
    recommendations: list[str]
    uncertaintyReason: str | None
    topPredictions: list[Prediction]
    weather: WeatherSnapshot | None
    language: Language = "en"


class TranslateRequest(BaseModel):
    report: AnalysisReport
    language: Language


class SpeechRequest(BaseModel):
    report: AnalysisReport
    language: Language
