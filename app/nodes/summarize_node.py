# app/nodes/summarize_node.py
from app.agent.state import AgentState
from app.llm.ollama_client import ollama_llm
from app.schema import Document


SYSTEM_PROMPT = """
You are a factual question answering assistant.

RULES:
- Answer ONLY using the provided documents
- Use ALL relevant information from the documents to construct a complete answer
- For questions about people, use any available information from web sources — LinkedIn, news articles, company pages, social media bios are all valid sources
- If the documents contain partial information, combine them into the best possible answer
- If the documents contain truly no relevant information at all — respond with exactly: "I don't have reliable information about this."
- Be specific and factual — 3-5 sentences
"""


def summarize_node(state: AgentState) -> AgentState:
    documents = state.get("documents", [])
    query = state.get("query", "")

    if not documents:
        state["answer"] = "I don't have reliable information about this."
        state["grounded"] = False
        state["confidence"] = 0.0
        return state

    context_chunks = []
    for d in documents:
        if isinstance(d, Document):
            if d.content:
                context_chunks.append(f"[Source: {d.source}]\n{d.content}")
        elif isinstance(d, dict):
            text = d.get("content", "")
            if text:
                context_chunks.append(text)

    if not context_chunks:
        state["answer"] = "I don't have reliable information about this."
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

Answer:
"""

    try:
        answer = ollama_llm.generate(prompt)
    except Exception as exc:
        state["answer"] = "I don't have reliable information about this."
        state["grounded"] = False
        state["confidence"] = 0.0
        state.setdefault("observations", []).append({
            "node": "summarize",
            "error": f"LLM unavailable: {exc}"
        })
        state.setdefault("steps", []).append(
            "SummarizeNode → LLM failed, returning fallback"
        )
        return state

    state["answer"] = answer
    state["grounded"] = not any(
        phrase in answer.lower()
        for phrase in [
            "i don't know",
            "not found",
            "no information",
            "i don't have reliable information",
        ]
    )
    state["confidence"] = min(0.95, 0.5 + len(documents) * 0.1)

    state.setdefault("steps", []).append(
        "SummarizeNode → answer generated"
    )

    return state