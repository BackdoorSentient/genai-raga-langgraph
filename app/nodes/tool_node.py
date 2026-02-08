from app.agent.state import AgentState
from typing import Any, List

def tool_node(state: AgentState, retriever: Any) -> AgentState:
    query = state.get("refined_query") or state.get("query")
    if not query:
        raise ValueError("ToolNode → query missing")

    retrieved_docs = retriever.search(query)

    normalized_docs = []
    for doc in retrieved_docs:
        normalized_docs.append({
            "content": getattr(doc, "page_content", str(doc))[:1000],
            "source": getattr(doc, "metadata", {}).get("source", "vector_store"),
            "origin": "vector"
        })

    existing_docs = state.get("documents", [])
    state["documents"] = existing_docs + normalized_docs

    state.setdefault("citations", [])
    for d in normalized_docs:
        state["citations"].append(d["source"])

    state["used_vector"] = True

    state.setdefault("steps", []).append(
        f"ToolNode → {len(normalized_docs)} vector docs (total={len(state['documents'])})"
    )

    state.setdefault("observations", []).append({
        "node": "tool",
        "documents_found": len(normalized_docs),
        "total_documents": len(state["documents"]),
        "source": "vector"
    })

    return state
