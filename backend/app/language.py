import html
import uuid

import httpx
from fastapi import HTTPException

from .config import settings
from .schemas import AnalysisReport, Language


VOICE_BY_LANGUAGE: dict[Language, tuple[str, str]] = {
    "en": ("en-IN", "en-IN-NeerjaNeural"),
    "te": ("te-IN", "te-IN-ShrutiNeural"),
    "hi": ("hi-IN", "hi-IN-SwaraNeural"),
}


async def translate_texts(texts: list[str], language: Language) -> list[str]:
    if language == "en":
        return texts
    if not settings.azure_translator_key:
        raise HTTPException(status_code=503, detail="Translation service is not configured.")
    url = f"{settings.azure_translator_endpoint.rstrip('/')}/translate"
    headers = {
        "Ocp-Apim-Subscription-Key": settings.azure_translator_key,
        "Content-Type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }
    if settings.azure_translator_region:
        headers["Ocp-Apim-Subscription-Region"] = settings.azure_translator_region
    params = {"api-version": "3.0", "from": "en", "to": language}
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(url, params=params, headers=headers, json=[{"text": text} for text in texts])
        response.raise_for_status()
    return [item["translations"][0]["text"] for item in response.json()]


async def translate_report(report: AnalysisReport, language: Language) -> AnalysisReport:
    if language == "en":
        return report.model_copy(update={"language": "en"})
    scalar_values = [
        report.condition,
        report.crop or "",
        report.disease or "",
        report.uncertaintyReason or "",
        report.weather.description if report.weather else "",
    ]
    list_values = report.observedSymptoms + report.typicalSymptoms + report.causes + report.recommendations
    translated = await translate_texts(scalar_values + list_values, language)
    cursor = len(scalar_values)
    lengths = [
        len(report.observedSymptoms),
        len(report.typicalSymptoms),
        len(report.causes),
        len(report.recommendations),
    ]
    groups: list[list[str]] = []
    for length in lengths:
        groups.append(translated[cursor : cursor + length])
        cursor += length
    weather = report.weather.model_copy(update={"description": translated[4]}) if report.weather else None
    return report.model_copy(
        update={
            "condition": translated[0],
            "crop": translated[1] or None,
            "disease": translated[2] or None,
            "uncertaintyReason": translated[3] or None,
            "weather": weather,
            "observedSymptoms": groups[0],
            "typicalSymptoms": groups[1],
            "causes": groups[2],
            "recommendations": groups[3],
            "language": language,
        }
    )


SPEECH_LABELS: dict[Language, dict[str, str]] = {
    "en": {
        "crop": "Crop", "condition": "Condition", "disease": "Disease",
        "confidence": "AI confidence", "severity": "Severity",
        "rate": "Visible disease rate", "symptoms": "Symptoms to check",
        "causes": "Known causes or contributors", "recommendations": "Recommendations",
        "weather": "Weather at", "degrees": "degrees Celsius", "humidity": "percent humidity",
        "undefined": "Undefined", "none": "None", "percent": "percent",
    },
    "te": {
        "crop": "పంట", "condition": "స్థితి", "disease": "వ్యాధి",
        "confidence": "ఏఐ నమ్మకం", "severity": "తీవ్రత",
        "rate": "ఆకులో కనిపించే వ్యాధి శాతం", "symptoms": "గమనించాల్సిన లక్షణాలు",
        "causes": "తెలిసిన కారణాలు లేదా దోహదకారకాలు", "recommendations": "సిఫార్సులు",
        "weather": "వాతావరణం, ప్రదేశం", "degrees": "డిగ్రీల సెల్సియస్", "humidity": "శాతం తేమ",
        "undefined": "నిర్వచించబడలేదు", "none": "లేదు", "percent": "శాతం",
    },
    "hi": {
        "crop": "फसल", "condition": "स्थिति", "disease": "रोग",
        "confidence": "एआई विश्वास", "severity": "गंभीरता",
        "rate": "पत्ती का दिखाई देने वाला रोग प्रतिशत", "symptoms": "जाँचने योग्य लक्षण",
        "causes": "ज्ञात कारण या सहायक कारक", "recommendations": "सिफारिशें",
        "weather": "मौसम, स्थान", "degrees": "डिग्री सेल्सियस", "humidity": "प्रतिशत आर्द्रता",
        "undefined": "अपरिभाषित", "none": "कोई नहीं", "percent": "प्रतिशत",
    },
}


def report_to_speech_text(report: AnalysisReport, language: Language) -> str:
    labels = SPEECH_LABELS[language]
    values = [
        f"{labels['crop']}: {report.crop or labels['undefined']}.",
        f"{labels['condition']}: {report.condition}.",
        f"{labels['disease']}: {report.disease or labels['none']}.",
        f"{labels['confidence']}: {round(report.confidence * 100)} {labels['percent']}."
        if report.confidence is not None else f"{labels['confidence']}: {labels['undefined']}.",
    ]
    if report.severity:
        values.append(f"{labels['severity']}: {report.severity}.")
    if report.diseaseRate is not None:
        values.append(f"{labels['rate']}: {report.diseaseRate} {labels['percent']}.")
    if report.uncertaintyReason:
        values.append(report.uncertaintyReason)
    if report.typicalSymptoms:
        values.append(f"{labels['symptoms']}: " + "; ".join(report.typicalSymptoms) + ".")
    if report.causes:
        values.append(f"{labels['causes']}: " + "; ".join(report.causes) + ".")
    if report.recommendations:
        values.append(f"{labels['recommendations']}: " + "; ".join(report.recommendations) + ".")
    if report.weather:
        values.append(
            f"{labels['weather']} {report.weather.location}: {report.weather.description}, "
            f"{report.weather.temperatureC} {labels['degrees']}, "
            f"{report.weather.humidityPercent} {labels['humidity']}."
        )
    return " ".join(values)


async def synthesize_speech(text: str, language: Language) -> bytes:
    if not settings.azure_speech_key or not settings.azure_speech_region:
        raise HTTPException(status_code=503, detail="Speech service is not configured.")
    locale, voice = VOICE_BY_LANGUAGE[language]
    url = f"https://{settings.azure_speech_region}.tts.speech.microsoft.com/cognitiveservices/v1"
    escaped = html.escape(text)
    ssml = f"<speak version='1.0' xml:lang='{locale}'><voice name='{voice}'>{escaped}</voice></speak>"
    headers = {
        "Ocp-Apim-Subscription-Key": settings.azure_speech_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
        "User-Agent": "CropDiseaseAI",
    }
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.post(url, headers=headers, content=ssml.encode("utf-8"))
        response.raise_for_status()
        return response.content
