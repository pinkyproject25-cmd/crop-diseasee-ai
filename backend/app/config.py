from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Crop Disease AI API"
    model_path: Path = Path("models/crop_classifier.onnx")
    labels_path: Path = Path("models/labels.json")
    model_manifest_path: Path = Path("models/candidate_manifest.json")
    model_version: str = "mixed-pv-plantdoc-mnv3-20261006T194137Z"
    model_sha256: str = "864dd9f77c01e6b4f77acfc1e0446b49c39855f31d5aed65c0d2a9a3b9ad4bb7"
    labels_sha256: str = "16925c0cb74cd5be219eb9e8cf87f0fa4f5c255f49b8db7c865c55050cb7d307"
    model_manifest_sha256: str = "357efbe591ca6e960acb5bf79e1211d164ec63bbfbd10244be40924db84d8c15"
    input_size: int = 224
    acceptance_threshold: float = 0.845
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
