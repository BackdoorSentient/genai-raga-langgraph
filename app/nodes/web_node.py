from app.agent.state import AgentState
from typing import List, Any


def web_node(state: AgentState) -> AgentState:
    query = state.get("refined_query") or state.get("query")
    if not query:
        raise ValueError("WebNode → query missing")

    # TODO: replace with real web search
    web_documents: List[Any] = []

    # ✅ ACCUMULATE instead of overwrite
    existing_docs = state.get("documents", [])
    state["documents"] = existing_docs + web_documents

    state["used_web"] = True

    state.setdefault("observations", []).append({
        "node": "web",
        "documents_found": len(web_documents),
        "total_documents": len(state["documents"])
    })

    state.setdefault("steps", []).append(
        f"WebNode → {len(web_documents)} docs (total={len(state['documents'])})"
    )

    return state
