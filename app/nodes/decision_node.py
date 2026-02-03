from app.agent.state import AgentState

CONFIDENCE_THRESHOLD = 0.6


# def decision_node(state: AgentState) -> AgentState:
#     """
#     Decide whether to accept the answer, retry, or stop execution.
#     """

#     retry_count = state.get("retry_count", 0)
#     max_retries = state.get("max_retries", 2)
#     grounded = state.get("grounded", False)
#     confidence = state.get("confidence", 0.0)

#     # Default decision
#     decision = "retry"

#     # Accept if answer is grounded and confident
#     if grounded and confidence >= CONFIDENCE_THRESHOLD:
#         decision = "accept"

#     # Stop if retries exhausted
#     elif retry_count >= max_retries:
#         decision = "stop"

#     # Update state
#     state["critic_decision"] = decision

#     # Track reasoning step
#     state.setdefault("steps", []).append(
#         f"DecisionNode → {decision.upper()} "
#         f"(grounded={grounded}, confidence={confidence}, retries={retry_count})"
#     )

#     return state

# def decision_node(state: AgentState) -> AgentState:
#     """
#     Decide next execution path (rag or tool).
#     """

#     # example logic (you already have this elsewhere)
#     state["next_node"] = "rag"   # or "tool"

#     state.setdefault("steps", []).append(
#         "DecisionNode → routed to " + state["next_node"]
#     )

#     return state

def decision_node(state: AgentState) -> AgentState:
    """
    Decide which execution path to take next (rag or tool).
    """

    # Example logic (yours may differ)
    if state.get("use_tool"):
        state["next_node"] = "tool"
    else:
        state["next_node"] = "rag"

    state.setdefault("steps", []).append(
        f"DecisionNode → routed to {state['next_node']}"
    )

    return state