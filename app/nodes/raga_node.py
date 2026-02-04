# app/nodes/raga_node.py
from app.agent.state import AgentState

def rag_node(state: AgentState) -> AgentState:
    """
    RAG node in agentic flow.
    Retrieval is handled by tool_node.
    This node only advances the graph.
    """
    state.setdefault("steps", []).append(
        "RAGNode → Skipped (Agentic flow)"
    )
    return state