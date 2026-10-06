from datetime import datetime, timezone

import httpx

from .schemas import WeatherSnapshot


WEATHER_LABELS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    80: "Rain showers",
    81: "Rain showers",
    82: "Heavy rain showers",
    95: "Thunderstorm",
}


async def fetch_weather(latitude: float | None, longitude: float | None) -> WeatherSnapshot | None:
    if latitude is None or longitude is None:
        return None
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
            response.raise_for_status()
            data = response.json()
            current = data["current"]
            return WeatherSnapshot(
                location=f"{latitude:.3f}, {longitude:.3f}",
                observedAt=datetime.fromisoformat(current["time"]).replace(tzinfo=timezone.utc),
                temperatureC=current.get("temperature_2m"),
                humidityPercent=current.get("relative_humidity_2m"),
                precipitationMm=current.get("precipitation"),
                description=WEATHER_LABELS.get(current.get("weather_code"), "Weather conditions"),
            )
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return None
