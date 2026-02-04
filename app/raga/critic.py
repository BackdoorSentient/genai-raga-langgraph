import time
from app.raga.state import RAGAState


def critic_node(state: RAGAState) -> RAGAState:
    state.setdefault("steps", [])
    state.setdefault("retry_reason", None)
    state.setdefault("terminate", False)

    # ---- ACCEPT FIRST ----
    if state.get("grounded") is True:
        state["steps"].append("Critic: Answer accepted")
        state["terminate"] = True
        state["retry_reason"] = None
        return state

    # ---- TIMEOUT ----
    start_time = state.get("start_time")
    timeout = state.get("timeout_seconds")

    if start_time and timeout:
        if time.time() - start_time > timeout:
            state["steps"].append("Critic: Timeout exceeded")
            state["retry_reason"] = "timeout"
            state["terminate"] = True
            return state

    # ---- MAX RETRIES ----
    if state.get("retry_count", 0) >= state.get("max_retries", 0):
        state["steps"].append("Critic: Max retries reached")
        state["retry_reason"] = "max_retries"
        state["terminate"] = True
        return state

    # ---- RETRY ----
    state["steps"].append("Critic: Retry triggered")
    state["retry_reason"] = "not_grounded"
    state["terminate"] = False

    return state
