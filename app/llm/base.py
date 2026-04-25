# app/llm/base.py
from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Primary method — all implementations must define this."""
        pass

    def invoke(self, prompt: str) -> str:
        """
        Alias for generate().
        RAG/RAGA graph nodes use llm.invoke() via LangChain convention.
        This lets OllamaClient drop in anywhere LangChain Ollama was used.
        """
        return self.generate(prompt)