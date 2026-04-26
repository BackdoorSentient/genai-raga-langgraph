# app/config/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "GenAI RAGA System"
    ENV: str = "dev"                        # dev | staging | prod

    # ── Data paths ───────────────────────────────────────────────────────────
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    VECTOR_DB_PATH: str = "data/vector_store"

    # ── LLM Provider ─────────────────────────────────────────────────────────
    LLM_PROVIDER: str = "ollama"            # ollama | openai | azure

    # ── Ollama ───────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct"

    # ── OpenAI ───────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ── Azure OpenAI ─────────────────────────────────────────────────────────
    AZURE_OPENAI_API_KEY: str | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_DEPLOYMENT: str | None = None

    # ── Embeddings ───────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Chunking ─────────────────────────────────────────────────────────────
    # Fix 4: was hardcoded as 500/100 inside ingestion.py
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # ── Retrieval ────────────────────────────────────────────────────────────
    # Fix 4: was hardcoded as k=4 inside vector_store.py and tool_node.py
    RETRIEVAL_K: int = 4

    # ── RAGA control ─────────────────────────────────────────────────────────
    # Fix 4: was hardcoded as 2/3 inside routes.py state initialization
    MAX_RETRIES: int = 2
    AGENTIC_MAX_RETRIES: int = 3

    # Fix 4: was hardcoded as 20 inside routes.py
    TIMEOUT_SECONDS: int = 20

    # ── LLM request timeout ──────────────────────────────────────────────────
    # Fix 4: was hardcoded as 120.0 inside ollama_client.py
    LLM_TIMEOUT: float = 120.0

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()