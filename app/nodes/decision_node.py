# from app.agent.state import AgentState


# def decision_node(state: AgentState) -> AgentState:

#     # -------------------------
#     # Defensive defaults
#     # -------------------------
#     state.setdefault("next_node", "summarize")
#     state.setdefault("steps", [])

#     plan = state.get("plan", [])
#     current_step = state.get("current_step", 0)
#     documents = state.get("documents", [])
#     web_attempted = state.get("web_attempted", False)
#     # query = state.get("query", "").lower()

#     # -------------------------
#     # Safety: no plan or plan finished
#     # -------------------------
#     if not plan or current_step >= len(plan):
#         state["next_node"] = "summarize"
#         state["steps"].append(
#             "DecisionNode → plan finished"
#         )
#         return state

#     step_text = plan[current_step].lower()

#     # -------------------------
#     # Intent-based routing
#     # -------------------------
#     if any(k in step_text for k in [
#         "search",
#         "retrieve",
#         "retrieval",
#         "locate",
#         "document",
#         "documents",
#         "grounded",
#         "rag",
#     ]):
#         next_node = "tool"

#     elif any(k in step_text for k in [
#         "tool",
#         "web",
#         "api"
#         # "call",
#     ]):
#         next_node = "tool"

#     else:
#         # analyze / identify / extract / synthesize
#         next_node = "summarize"

#     # -------------------------
#     # Query-aware fallback
#     # -------------------------
#     # Prevent empty summaries for factual questions
#     if next_node == "summarize" and any(
#         k in query for k in ["what", "define", "explain", "who"]
#         # k in query for k in ["what", "define", "explain", "who", "when"]
#     ):
#         next_node = "tool"

#     # -------------------------
#     # Set routing decision
#     # -------------------------
#     state["next_node"] = next_node

#     # state["steps"].append(
#     #     f"DecisionNode → step {current_step + 1}/{len(plan)} "
#     #     f"→ {next_node} | '{plan[current_step]}'"
#     # )
#     state["steps"].append(
#         f"DecisionNode → step {current_step + 1} → {next_node}"
#     )

#     return state
from app.agent.state import AgentState


def decision_node(state: AgentState) -> AgentState:
    state.setdefault("steps", [])
    state.setdefault("next_node", "summarize")

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    documents = state.get("documents", [])
    web_attempted = state.get("web_attempted", False)

    # -------------------------
    # Step-based routing
    # -------------------------
    if not plan or current_step >= len(plan):
        state["next_node"] = "summarize"
        state["steps"].append("DecisionNode → plan finished")
        return state

    step_text = plan[current_step].lower()
    # if plan and current_step < len(plan):
    #     step_text = plan[current_step].lower()

    if any(k in step_text for k in [
        "identify",
        "locate",
        "search",
        "retrieve",
        "document",
        "documents",
        "grounded",
        "knowledge",
        "rag",
    ]):
        state["next_node"] = "tool"
        state["steps"].append(
            f"DecisionNode → step {current_step + 1} → tool"
        )
        return state

    # -------------------------
    # Fallback logic
    # -------------------------
    # If RAG returned nothing → try web
    if not documents and not web_attempted:
        state["next_node"] = "web"
        state["web_attempted"] = True
        state["steps"].append("DecisionNode → RAG empty → fallback to web")
        return state

    # -------------------------
    # Otherwise summarize
    # -------------------------
    state["next_node"] = "summarize"
    state["steps"].append(
        f"DecisionNode → step {current_step + 1} → summarize"
    )
    return state
