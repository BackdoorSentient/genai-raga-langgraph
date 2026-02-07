from app.agent.state import AgentState
from typing import List
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

    # -------------------------
    # No documents → exit early
    # -------------------------
    if not documents:
        state["answer"] = "No relevant information found in the knowledge base."
        state["grounded"] = False
        state["confidence"] = 0.0
        state["citations"] = []

        # IMPORTANT: advance plan step even on failure
        state["current_step"] = state.get("current_step", 0) + 1

        return state

    # context_chunks: List[str] = []
    # citations: List[str] = []
    context = "\n\n".join(d.page_content for d in documents)
    citations = list(set(
        d.metadata.get("source", "unknown") for d in documents
    ))

    # for i, doc in enumerate(documents):
    #     context_chunks.append(doc.page_content)
    #     citations.append(
    #         doc.metadata.get("source", f"doc_{i}")
    #     )

    # context = "\n\n".join(context_chunks)

    prompt = f"""
{SYSTEM_PROMPT}

Question:
{query}

Documents:
{context}

Answer:
"""

    answer = ollama_llm.generate(prompt)

    # -------------------------
    # Grounding heuristic
    # -------------------------
    # grounded = "i don't know" not in answer.lower()
    # confidence = min(0.95, 0.5 + (len(documents) * 0.1))

    # -------------------------
    # Update state
    # -------------------------
    state["answer"] = answer
    state["grounded"] = "i don't know" not in answer.lower()
    # state["grounded"] = grounded
    state["confidence"] = min(0.95, 0.5 + len(documents) * 0.1)
    # state["confidence"] = round(confidence, 2)
    state["citations"] = citations
    # state["citations"] = list(set(citations))

    # state.setdefault("steps", []).append(
    #     "SummarizeNode → Generated grounded answer"
    # )
    state.setdefault("steps", []).append("SummarizeNode → answer generated")

    # -------------------------
    # Advance plan step (CRITICAL FIX)
    # -------------------------
    state["current_step"] = state.get("current_step", 0) + 1

    return state
