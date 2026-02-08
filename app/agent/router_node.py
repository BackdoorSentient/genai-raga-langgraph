from app.agent.state import AgentState

def router_node(state: AgentState) -> AgentState:
    """
    Single routing authority.
    Guarantees termination.
    Preserves documents across retries.
    """

    phase = state.get("phase", "retrieve")
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if phase == "retrieve":

        if current_step >= len(plan):

            if not state.get("documents"):
                state["phase"] = "retrieve"
                state["next_node"] = "tool"
                return state

            state["phase"] = "summarize"
            state["next_node"] = "summarize"
            return state

        if not state.get("used_vector"):
            state["used_vector"] = True
            state["current_step"] = current_step + 1
            state["next_node"] = "tool"
            return state

        if not state.get("used_web"):
            state["used_web"] = True
            state["current_step"] = current_step + 1
            state["next_node"] = "web"
            return state

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
            state["current_step"] = 0
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
