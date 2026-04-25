# app/nodes/summarize_node.py
from app.agent.state import AgentState
from app.llm.ollama_client import ollama_llm
from app.schema import Document


SYSTEM_PROMPT = """
You are a factual question answering assistant.

RULES:
- Answer ONLY using the provided documents
- If the documents do not contain a clear, direct, authoritative answer to the question — respond with exactly: "I don't have reliable information about this."
- Do NOT speculate or combine loosely related facts to construct an answer
- Do NOT treat social media follower counts, unrelated mentions, or tangential references as authoritative answers
- Do NOT answer questions about specific private individuals unless a clearly authoritative source is in the documents
- 2-4 sentences maximum when you do have an answer
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
                context_chunks.append(d.content)
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

    # ── Issue 11: wrap LLM call, never crash on Ollama failure ───────────────
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
    # ─────────────────────────────────────────────────────────────────────────

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