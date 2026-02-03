import httpx
from app.llm.base import BaseLLM
import asyncio

# class OllamaClient(BaseLLM):
#     def __init__(self, model: str = "llama3"):
#         self.model = model
#         self.base_url = "http://127.0.0.1:11434"

#     async def generate(self, prompt: str) -> str:
#         payload = {
#             "model": self.model,
#             "prompt": prompt,
#             "stream": False
#         }

#         async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
#             res = await client.post(
#                 f"{self.base_url}/api/generate",
#                 json=payload
#             )

#             if res.status_code != 200:
#                 raise Exception(f"Ollama error: {res.text}")

#             data = res.json()
#             return (data.get("response") or "").strip()
        
#     def generate_sync(self, prompt: str) -> str:
#         try:
#             loop = asyncio.get_running_loop()
#         except RuntimeError:
#             loop = None

#         if loop and loop.is_running():
#             # running inside event loop (FastAPI, Jupyter)
#             return loop.run_until_complete(self.generate(prompt))
#         else:
#             return asyncio.run(self.generate(prompt))

# ollama_llm = OllamaClient(model="llama3")
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

ollama_llm = OllamaClient(model="llama3")