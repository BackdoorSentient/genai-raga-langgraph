from app.agent.state import AgentState
from typing import List, Any


def tool_node(state: AgentState, retriever: Any) -> AgentState:
    """
    Executes retrieval tool based on current plan step.
    """

    query = state.get("refined_query") or state.get("query")
    current_step = state.get("current_step", 0)

    # Execute retrieval
    documents: List[Any] = retriever.search(query)

    # Update state
    state["documents"] = documents

    # Log observation
    observation = {
        "step": current_step,
        "tool": "vector_retriever",
        "query": query,
        "documents_found": len(documents)
    }

    state.setdefault("observations", []).append(observation)

    state.setdefault("steps", []).append(
        f"ToolNode → Retrieved {len(documents)} documents"
    )

    return state
