from app.agent.state import AgentState
from typing import List, Dict, Any
from ddgs import DDGS

def web_node(state: AgentState) -> AgentState:
    query = state.get("refined_query") or state.get("query")
    if not query:
        raise ValueError("WebNode → query missing")

    web_documents: List[Dict[str, Any]] = []

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)

        for r in results:
            web_documents.append({
                "content": r.get("body", ""),
                "source": r.get("href", ""),
                "origin": "web"
            })

    # ✅ ACCUMULATE (never overwrite)
    existing_docs = state.get("documents", [])
    state["documents"] = existing_docs + web_documents

    state["used_web"] = True

    # ---- Citations ----
    state.setdefault("citations", [])
    for doc in web_documents:
        if doc["source"]:
            state["citations"].append(doc["source"])

    # ---- Steps (append-only) ----
    state.setdefault("steps", []).append(
        f"WebNode → {len(web_documents)} web docs (total={len(state['documents'])})"
    )

    # ---- Observability ----
    state.setdefault("observations", []).append({
        "node": "web",
        "documents_found": len(web_documents),
        "total_documents": len(state["documents"]),
        "source": "web"
    })

    return state

