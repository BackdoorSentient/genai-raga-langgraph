# app/nodes/web_node.py
from typing import List
from ddgs import DDGS
from app.agent.state import AgentState
from app.schema import Document


def web_node(state: AgentState) -> AgentState:
    query = state.get("refined_query") or state.get("query")
    if not query:
        raise ValueError("WebNode → query missing")

    web_documents: List[Document] = []

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=8)
            for r in results:
                web_documents.append(
                    Document(
                        content=r.get("body", ""),
                        source=r.get("href", ""),
                        origin="web",
                    )
                )
    except Exception as exc:
        state.setdefault("observations", []).append({
            "node": "web",
            "error": f"DuckDuckGo search failed: {exc}"
        })

    existing_docs = state.get("documents", [])
    state["documents"] = existing_docs + web_documents
    state["used_web"] = True

    state.setdefault("citations", [])
    for doc in web_documents:
        if doc.source:
            state["citations"].append(doc.source)

    state.setdefault("steps", []).append(
        f"WebNode → {len(web_documents)} web docs (total={len(state['documents'])})"
    )

    state.setdefault("observations", []).append({
        "node": "web",
        "documents_found": len(web_documents),
        "total_documents": len(state["documents"]),
        "source": "web",
    })

    return state