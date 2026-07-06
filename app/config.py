"""
Central configuration — reads from environment variables / .env file.
Load with:  from app.config import settings
"""
from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"  # Groq model ID
    llm_temperature: float = 0.2

    # ── Paths ─────────────────────────────────────────────────────────────────
    model_path: Path = BASE_DIR / "models" / "xgboost_model.joblib"
    encoders_path: Path = BASE_DIR / "models" / "encoders.joblib"
    listings_csv: Path = BASE_DIR / "data" / "listings.csv"
    faiss_index_path: Path = BASE_DIR / "vector_store" / "faiss_index"
    policies_dir: Path = BASE_DIR / "data" / "policies"

    # ── RAG ───────────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_k: int = 4

    # ── API ───────────────────────────────────────────────────────────────────
    app_title: str = "Airbnb AI Service API"
    app_version: str = "1.0.0"
    cors_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
