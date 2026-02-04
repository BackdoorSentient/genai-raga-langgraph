import time
from app.raga.state import RAGAState


def critic_node(state: RAGAState) -> RAGAState:
    t0 = time.time()

    state.setdefault("steps", [])
    state.setdefault("retry_reason", None)
    state.setdefault("terminate", False)

    # ---- ACCEPT FIRST ----
    if state.get("grounded") is True:
        state["steps"].append("Critic: Answer accepted")
        state["terminate"] = True

    # ---- TIMEOUT ----
    elif time.time() - state["start_time"] > state["timeout_seconds"]:
        state["steps"].append("Critic: Timeout exceeded")
        state["retry_reason"] = "timeout"
        state["terminate"] = True

    # ---- MAX RETRIES ----
    elif state["retry_count"] >= state["max_retries"]:
        state["steps"].append("Critic: Max retries reached")
        state["retry_reason"] = "max_retries"
        state["terminate"] = True

    # ---- RETRY ----
    else:
        state["steps"].append("Critic: Retry triggered")
        state["retry_reason"] = "not_grounded"

    # ---- QUALITY SCORE ----
    doc_count = len(state.get("documents", []))
    coverage = len(state.get("citations", [])) / doc_count if doc_count else 0.0

    state["quality_score"] = round(
        state.get("confidence", 0.0)
        * (1 if state.get("grounded") else 0)
        * coverage,
        3
    )

    latency = round((time.time() - t0) * 1000, 2)
    state.setdefault("timeline", []).append({
        "node": "critic",
        "latency_ms": latency,
        "decision": "end" if state["terminate"] else "retry"
    })

    return state
