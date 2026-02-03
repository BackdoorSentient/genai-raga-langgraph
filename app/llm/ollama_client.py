import httpx
from app.llm.base import BaseLLM

class OllamaClient(BaseLLM):
    def __init__(self, model: str = "llama3"):
        self.model = model
        self.base_url = "http://127.0.0.1:11434"

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        with httpx.Client(timeout=120.0, trust_env=False) as client:
            res = client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )

        if res.status_code != 200:
            raise Exception(f"Ollama error: {res.text}")

        data = res.json()
        return (data.get("response") or "").strip()

    def with_model(self, model: str) -> "OllamaClient":
        return OllamaClient(model=model)

ollama_llm = OllamaClient(model="llama3")