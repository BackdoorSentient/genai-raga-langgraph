from app.agent.state import AgentState


def rag_node(state: AgentState) -> AgentState:
    """
    RAGNode reasons over retrieved documents.
    Does NOT retrieve or route.
    """

    documents = state.get("documents", [])
    state.setdefault("steps", [])

    if not documents:
        state["steps"].append("RAGNode → no documents to reason over")
        return state

    # Lightweight evidence struct for downstream nodes
    evidence = []
    for doc in documents:
        evidence.append({
            "content": getattr(doc, "page_content", str(doc))[:500],
            "source": getattr(doc, "metadata", {}).get("source", "vector"),
        })

    state["evidence"] = evidence

    state["steps"].append(
        f"RAGNode → prepared evidence from {len(documents)} documents"
    )

    return state
