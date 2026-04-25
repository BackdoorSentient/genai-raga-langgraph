# app/agent/router_node.py
from app.agent.state import AgentState


def router_node(state: AgentState) -> AgentState:
    """
    Single routing authority.
    Guarantees termination.
    Preserves documents across retries.
    """

    phase = state.get("phase", "retrieve")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if phase == "retrieve":

        # Step 1 — always try vector first
        if not state.get("used_vector"):
            state["used_vector"] = True
            state["next_node"] = "tool"
            return state

        # Step 2 — always try web second, regardless of whether vector found anything
        if not state.get("used_web"):
            state["used_web"] = True
            state["next_node"] = "web"
            return state

        # Step 3 — both retrieval sources exhausted, move to summarize
        # Even if documents is empty, summarize will handle it gracefully
        state["phase"] = "summarize"
        state["next_node"] = "summarize"
        return state

    if phase == "summarize":
        state["phase"] = "critic"
        state["next_node"] = "critic"
        return state

    if phase == "critic":

        if state.get("grounded"):
            state["phase"] = "end"
            state["next_node"] = "__end__"
            return state

        if retry_count < max_retries:
            state["retry_count"] = retry_count + 1
            state["used_vector"] = False
            state["used_web"] = False
            state["phase"] = "retrieve"
            state["next_node"] = "tool"
            return state

        state["phase"] = "end"
        state["next_node"] = "__end__"
        return state

    state["next_node"] = "__end__"
    return state