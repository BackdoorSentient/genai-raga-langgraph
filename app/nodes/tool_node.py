from app.agent.state import AgentState
from typing import Any, List


def tool_node(state: AgentState, retriever: Any) -> AgentState:
    query = state.get("refined_query") or state.get("query")
    if not query:
        raise ValueError("ToolNode → query missing")

    documents: List[Any] = retriever.search(query)

    # ✅ ACCUMULATE documents instead of overwriting
    existing_docs = state.get("documents", [])
    state["documents"] = existing_docs + documents

    state["used_vector"] = True

    state.setdefault("observations", []).append({
        "node": "tool",
        "documents_found": len(documents),
        "total_documents": len(state["documents"])
    })

    state.setdefault("steps", []).append(
        f"ToolNode → {len(documents)} docs (total={len(state['documents'])})"
    )

    return state
