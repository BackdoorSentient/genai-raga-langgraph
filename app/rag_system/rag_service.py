from app.rag_system.vector_store import VectorStore
from app.rag_system.llm_clients import get_llm

vector_store = VectorStore()
llm = get_llm()


def initialize_rag(documents):
    vector_store.build(documents)


def get_answer(query: str) -> str:
    docs = vector_store.similarity_search(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
You are a RAG-based AI system.

Context:
{context}

Question:
{query}

Answer:
"""

    return llm(prompt)
