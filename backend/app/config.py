from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Crop Disease AI API"
    model_path: Path = Path("models/crop_classifier.onnx")
    labels_path: Path = Path("models/labels.json")
    model_version: str = "not-configured"
    input_size: int = 224
    acceptance_threshold: float = 0.72
    max_image_bytes: int = 10 * 1024 * 1024
    allowed_origins: str = "http://localhost:5173"

    azure_speech_key: str | None = None
    azure_speech_region: str | None = None
    azure_translator_key: str | None = None
    azure_translator_region: str | None = None
    azure_translator_endpoint: str = "https://api.cognitive.microsofttranslator.com"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


settings = Settings()
