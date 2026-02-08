from app.agent.state import AgentState
from app.llm.ollama_client import ollama_llm


SYSTEM_PROMPT = """
Answer ONLY from the documents.
If missing, say so clearly.
"""


def summarize_node(state: AgentState) -> AgentState:
    documents = state.get("documents", [])
    query = state.get("query", "")

    if not documents:
        state["answer"] = "No grounded information found."
        state["grounded"] = False
        state["confidence"] = 0.0
        return state

    context = "\n\n".join(d.page_content for d in documents)

    prompt = f"""
{SYSTEM_PROMPT}

Question:
{query}

Documents:
{context}

Answer:
"""

    answer = ollama_llm.generate(prompt)

    state["answer"] = answer
    state["grounded"] = "i don't know" not in answer.lower()
    state["confidence"] = min(0.95, 0.5 + len(documents) * 0.1)

    state.setdefault("steps", []).append(
        "SummarizeNode → answer generated"
    )

    return state
