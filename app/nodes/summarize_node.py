from app.agent.state import AgentState
from typing import Any, List
from app.llm.ollama_client import ollama_llm


SYSTEM_PROMPT = """
You are a grounded AI assistant.
Answer ONLY using the provided documents.
If the answer cannot be found, say so clearly.
Cite sources explicitly.
"""


def summarize_node(state: AgentState) -> AgentState:
    documents = state.get("documents", [])
    query = state.get("query", "")

    if not documents:
        state["answer"] = "No relevant information found in the knowledge base."
        state["grounded"] = False
        state["confidence"] = 0.0
        state["citations"] = []
        return state

    context_chunks: List[str] = []
    citations: List[str] = []

    for i, doc in enumerate(documents):
        context_chunks.append(doc.page_content)
        # if "source" in doc.metadata:
        #     citations.append(doc.metadata["source"])
        # else:
        #     citations.append(f"doc_{i}")
        citations.append(
            doc.metadata.get("source", f"doc_{i}")
        )

    context = "\n\n".join(context_chunks)

    prompt = f"""
{SYSTEM_PROMPT}

Question:
{query}

Documents:
{context}

Answer:
"""

    # response = llm.invoke(prompt)

    # answer = response.strip()
    answer = ollama_llm.generate(prompt)

    # Simple grounding heuristic
    grounded = "I don't know" not in answer.lower()

    confidence = min(0.95, 0.5 + (len(documents) * 0.1))

    state["answer"] = answer
    state["grounded"] = grounded
    state["confidence"] = round(confidence, 2)
    state["citations"] = list(set(citations))

    state.setdefault("steps", []).append(
        "SummarizeNode → Generated grounded answer"
    )

    return state
