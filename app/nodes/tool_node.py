# app/nodes/tool_node.py
from app.agent.state import AgentState
from typing import Any, List


def tool_node(state: AgentState, retriever: Any) -> AgentState:
    """
    ToolNode executes vector store retrieval.
    This is the ONLY place where document search happens.
    """

    query = state.get("refined_query") or state.get("query")
    # current_step = state.get("current_step", 0)

    if not query:
        raise ValueError("ToolNode → query is missing")

    # Execute vector retrieval
    documents: List[Any] = retriever.search(query)

    # Update state
    state["documents"] = documents

    # Log structured observation
    # observation = {
    #     "step": current_step,
    #     "tool": "vector_store",
    #     "query": query,
    #     "documents_found": len(documents),
    # }

    # state.setdefault("observations", []).append(observation)
    state.setdefault("observations", []).append({
        "node": "tool",
        "query": query,
        "documents_found": len(documents)
    })

    state.setdefault("steps", []).append(
        f"ToolNode → Retrieved {len(documents)} documents"
    )
    state["current_step"] = state.get("current_step", 0) + 1

    return state
