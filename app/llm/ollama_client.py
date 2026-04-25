# app/llm/ollama_client.py
import httpx
from app.llm.base import BaseLLM
from app.config.settings import settings


class OllamaClient(BaseLLM):

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        # Read from settings if not explicitly passed
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                res = client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url} — is it running?"
            ) from exc

        if res.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {res.status_code}: {res.text[:200]}"
            )

        return (res.json().get("response") or "").strip()

    def with_model(self, model: str) -> "OllamaClient":
        return OllamaClient(model=model, base_url=self.base_url, timeout=self.timeout)


# Single shared instance — reads model + base_url from settings automatically
ollama_llm = OllamaClient()