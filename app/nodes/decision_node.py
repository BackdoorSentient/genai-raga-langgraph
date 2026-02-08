from app.agent.state import AgentState


def decision_node(state: AgentState) -> AgentState:
    """
    DecisionNode is now PURELY ANALYTICAL.
    It does NOT control routing.
    RouterNode is the single routing authority.
    """

    state.setdefault("steps", [])

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    documents = state.get("documents", [])

    if not plan:
        state["steps"].append("DecisionNode → no plan found")
        return state

    if current_step >= len(plan):
        state["steps"].append("DecisionNode → plan exhausted")
        return state

    step_text = plan[current_step].lower()

    intent = "analysis"
    if any(k in step_text for k in [
        "search",
        "retrieve",
        "locate",
        "document",
        "grounded",
        "rag",
    ]):
        intent = "retrieval"

    state.setdefault("observations", []).append({
        "node": "decision",
        "step": current_step + 1,
        "intent": intent,
        "documents_available": len(documents),
    })

    state["steps"].append(
        f"DecisionNode → step {current_step + 1} intent={intent}"
    )

    return state
