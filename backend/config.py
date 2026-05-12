"""Application configuration for the Digitalisierung System."""
from __future__ import annotations
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Locate .env — prefer CWD, fallback to project root (parent of backend/)
_env_path = Path(".env")
if not _env_path.exists():
    _project_env = Path(__file__).resolve().parent.parent / ".env"
    if _project_env.exists():
        _env_path = _project_env


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_env_path),
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    APP_NAME: str = Field(default="Digitalisierung")
    DEBUG: bool = Field(default=False)
    VLM_MODEL_ID: str = Field(default="surya-ocr")
    LLM_MODEL_PATH: str = Field(default="models/llama-3.1-8b-instruct.Q4_K_M.gguf")
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    LLM_N_THREADS: int = Field(default=8)
    LLM_N_BATCH: int = Field(default=512)
    LLM_N_CTX: int = Field(default=8192)
    RAM_THRESHOLD: float = Field(default=0.80, ge=0.0, le=1.0)
    MAX_CONCURRENT_REQUESTS: int = Field(default=1, ge=1)
    LEGIBILITY_THRESHOLD: float = Field(default=0.70, ge=0.0, le=1.0)
    CONFIDENCE_THRESHOLD: float = Field(default=0.70, ge=0.0, le=1.0)
    CHROMA_PERSIST_DIR: str = Field(default="./chroma_data")
    MAX_FEW_SHOT_CORRECTIONS: int = Field(default=5, ge=0)
    AUDIT_LOG_PATH: str = Field(default="./logs/audit.jsonl")
    AUDIT_LOG_ROTATE_SIZE_MB: int = Field(default=100)
    AUDIT_LOG_MAX_ARCHIVES: int = Field(default=10)
    VLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    UPLOAD_DIR: str = Field(default="./uploads")
    SURYA_DEVICE: str = Field(default="")
    SURYA_DETECTOR_BATCH_SIZE: int = Field(default=12, ge=1)
    SURYA_RECOGNITION_BATCH_SIZE: int = Field(default=96, ge=1)
    SURYA_LAYOUT_BATCH_SIZE: int = Field(default=12, ge=1)


# -- Module-level constants --
APP_VERSION: str = "1.0.0"
UPLOAD_DIR: str = "./uploads"
ALLOWED_IMAGE_TYPES: set[str] = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/tiff",
    "image/bmp",
    "application/pdf",
    "text/plain",
    "text/csv",
    "text/tab-separated-values",
    "text/markdown",
    "text/html",
    "application/json",
    "application/xml",
    "text/xml",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

# Singleton
settings = Settings()


def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return settings
