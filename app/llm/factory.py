from app.llm.ollama_client import OllamaClient
from app.config.settings import settings

def get_llm():
    """
    LLM Factory
    Decides which LLM implementation to return
    """
    if settings.LLM_PROVIDER == "ollama":
        return OllamaClient(model=settings.OLLAMA_MODEL)

    raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
