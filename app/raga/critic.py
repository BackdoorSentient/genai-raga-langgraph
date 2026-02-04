import time


def critic_node(state):
    now = time.time()

    state.setdefault("steps", [])
    state.setdefault("retry_count", 0)
    state.setdefault("max_retries", 2)

    start_time = state.get("start_time", now)
    timeout_seconds = state.get("timeout_seconds", 10.0)

    # ---- Timeout ----
    if now - start_time > timeout_seconds:
        state["retry_reason"] = "timeout"
        state["terminate"] = True
        state["steps"].append("Timeout exceeded")
        return state

    # ---- Success ----
    if state.get("grounded"):
        state["retry_reason"] = None
        state["terminate"] = True
        state["steps"].append("Answer accepted")
        return state

    # ---- Retry limit ----
    if state["retry_count"] >= state["max_retries"]:
        state["retry_reason"] = "max_retries"
        state["terminate"] = True
        state["steps"].append("Max retries exceeded")
        return state

    # ---- Retry ----
    state["retry_count"] += 1
    state["retry_reason"] = "not_grounded"
    state["terminate"] = False
    state["steps"].append(
        f"Retrying (reason={state['retry_reason']})"
    )

    # Reset refinement to avoid compounding
    state.pop("refined_query", None)

    return state
