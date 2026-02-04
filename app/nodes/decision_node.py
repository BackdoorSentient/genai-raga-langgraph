# app/nodes/decision_node.py
from app.agent.state import AgentState


def decision_node(state: AgentState) -> AgentState:
    """
    Decide next execution path based on planner output.
    Planner plan = list[str]
    """

    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    # Safety: no plan or plan finished
    if not plan or current_step >= len(plan):
        state["next_node"] = "summarize"
        state.setdefault("steps", []).append(
            "DecisionNode → plan finished, routing to summarize"
        )
        return state

    step_text = plan[current_step].lower()

    # -------------------------
    # Heuristic routing
    # -------------------------
    if any(k in step_text for k in ["search", "retrieve", "document", "rag"]):
        next_node = "rag"
    elif any(k in step_text for k in ["tool", "web", "api"]):
        next_node = "tool"
    else:
        next_node = "summarize"

    # Advance plan step
    state["current_step"] = current_step + 1
    state["next_node"] = next_node

    state.setdefault("steps", []).append(
        f"DecisionNode → step {current_step + 1}/{len(plan)} → {next_node}"
    )

    return state
