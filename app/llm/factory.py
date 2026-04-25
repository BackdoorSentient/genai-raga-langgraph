# app/llm/factory.py
from app.config.settings import settings
from app.llm.base import BaseLLM


def get_llm(role: str | None = None) -> BaseLLM:
    """
    Return the configured LLM client.

    role: optional hint — "planner", "critic", "executor"
          (reserved for future per-role model config)

    Single entry point for ALL LLM access in the system.
    Swap provider by changing LLM_PROVIDER in .env — no code changes needed.
    """

    if settings.LLM_PROVIDER == "ollama":
        from app.llm.ollama_client import OllamaClient
        return OllamaClient(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

    if settings.LLM_PROVIDER == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")

        from app.llm.base import BaseLLM

        class _OpenAIClient(BaseLLM):
            def __init__(self):
                self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

            def generate(self, prompt: str) -> str:
                resp = self._client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content.strip()

        return _OpenAIClient()

    if settings.LLM_PROVIDER == "azure":
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )
        if not all([
            settings.AZURE_OPENAI_API_KEY,
            settings.AZURE_OPENAI_ENDPOINT,
            settings.AZURE_OPENAI_DEPLOYMENT,
        ]):
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT and "
                "AZURE_OPENAI_DEPLOYMENT must all be set in .env"
            )

        from app.llm.base import BaseLLM

        class _AzureClient(BaseLLM):
            def __init__(self):
                self._client = AzureOpenAI(
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    api_version="2024-02-01",
                )

            def generate(self, prompt: str) -> str:
                resp = self._client.chat.completions.create(
                    model=settings.AZURE_OPENAI_DEPLOYMENT,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content.strip()

        return _AzureClient()

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER!r}. "
        "Choose from: ollama, openai, azure"
    )