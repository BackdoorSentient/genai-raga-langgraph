from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "GenAI RAGA System"
    ENV: str = "dev"

    # LLM Provider
    LLM_PROVIDER: str = "ollama"  # ollama | openai | azure

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"

    # OpenAI
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Azure OpenAI
    AZURE_OPENAI_API_KEY: str | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_DEPLOYMENT: str | None = None

    # Vector Store
    VECTOR_DB_PATH: str = "data/vector_store"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings():
    return Settings()
