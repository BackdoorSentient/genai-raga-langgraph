# app/nodes/rag_node.py
from app.agent.state import AgentState


# def rag_node(state: AgentState) -> AgentState:
#     """
#     Agentic RAG orchestration node.
#     Retrieval is already handled by ToolNode.
#     This node only advances the plan.
#     """

#     current_step = state.get("current_step", 0)
#     state["current_step"] = current_step + 1

#     state.setdefault("steps", []).append(
#         f"RAGNode → Advanced to step {state['current_step']}"
#     )

#     return state
def rag_node(state: AgentState) -> AgentState:
    state.setdefault("steps", []).append(
        "RAGNode → reasoning over documents"
    )
    return state
