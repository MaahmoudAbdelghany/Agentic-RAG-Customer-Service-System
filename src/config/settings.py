"""Application settings loaded from environment variables using Pydantic Settings.

All configuration is centralized here. Values are loaded from a `.env` file
at the project root, with sensible defaults for local development.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve the project root (three levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Central configuration for the Agentic RAG Customer Service System."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── LLM (Groq) ──────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 1024

    # ── Embeddings ───────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # ── ChromaDB ─────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "customer_service_kb"

    # ── RAG Chunking ─────────────────────────────────────────────────────
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    # ── Retrieval ────────────────────────────────────────────────────────
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

    # ── Company / Branding ───────────────────────────────────────────────
    COMPANY_NAME: str = "TechStyle"
    DEFAULT_LANGUAGE: str = "en"  # "en" or "ar"

    # ── Knowledge Base ───────────────────────────────────────────────────
    KNOWLEDGE_BASE_DIR: str = str(PROJECT_ROOT / "data" / "knowledge_base")
    MOCK_ORDERS_PATH: str = str(PROJECT_ROOT / "data" / "mock_orders.json")

    # ── Analytics (SQLite) ───────────────────────────────────────────────
    ANALYTICS_DB_PATH: str = str(PROJECT_ROOT / "analytics.db")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of the application settings.

    Using ``@lru_cache`` ensures the ``.env`` file is read only once and the
    same ``Settings`` object is reused across the entire application lifetime.
    """
    return Settings()
