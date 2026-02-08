from app.agent.state import AgentState
from app.llm.ollama_client import ollama_llm


SYSTEM_PROMPT = """
You are answering a factual question using ONLY the provided documents.

RULES:
- Use ONLY information present in the documents
- Do NOT add external knowledge
- If multiple facts are present, COMBINE them into a clear explanation
- Answer fully, not minimally
- 2–4 sentences if possible
- If information is missing, say so clearly
"""


def summarize_node(state: AgentState) -> AgentState:
    documents = state.get("documents", [])
    query = state.get("query", "")

    if not documents:
        state["answer"] = "No grounded information found."
        state["grounded"] = False
        state["confidence"] = 0.0
        return state

    context_chunks = [
        d.get("content", "")
        for d in documents
        if isinstance(d, dict) and d.get("content")
    ]

    if not context_chunks:
        state["answer"] = "No usable grounded content found."
        state["grounded"] = False
        state["confidence"] = 0.0
        return state

    context = "\n\n".join(context_chunks)

    prompt = f"""
{SYSTEM_PROMPT}

Question:
{query}

Documents:
{context}

Answer (definition + explanation):
"""

    answer = ollama_llm.generate(prompt)

    state["answer"] = answer
    state["grounded"] = not any(
        phrase in answer.lower()
        for phrase in ["i don't know", "not found", "no information"]
    )
    state["confidence"] = min(0.95, 0.5 + len(documents) * 0.1)

    state.setdefault("steps", []).append(
        "SummarizeNode → answer generated"
    )

    return state
