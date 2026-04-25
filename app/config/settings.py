# app/config/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # App
    APP_NAME: str = "GenAI RAGA System"
    ENV: str = "dev"

    # Data paths
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    VECTOR_DB_PATH: str = "data/vector_store"

    # LLM Provider
    LLM_PROVIDER: str = "ollama"  # options: ollama | openai | azure

    # Ollama
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    # OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"

    # OpenAI
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Azure OpenAI
    AZURE_OPENAI_API_KEY: str | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_DEPLOYMENT: str | None = None

    # Vector Store / FAISS
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"
        extra = "ignore"  # prevents crashes on extra env vars

# cached getter
@lru_cache
def get_settings() -> Settings:
    return Settings()

settings=get_settings()
