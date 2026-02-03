from langgraph.graph import StateGraph, START, END
from app.nodes.raga_node import rag_node
from app.nodes.action_node import action_node
from app.nodes.decision_node import decision_node
from app.workflows.state_schema import RAGAState


def build_raga_graph():
    graph = StateGraph(RAGAState)

    # Nodes
    graph.add_node("decision", decision_node)
    graph.add_node("rag", rag_node)
    graph.add_node("action", action_node)

    # Start → Decision
    graph.add_edge(START, "decision")

    # Decision → conditional routing
    graph.add_conditional_edges(
        "decision",
        lambda state: state["next_node"],  # value set in decision_node
        {
            "rag": "rag",
            "tool": "action",  # or tool_node if you rename it later
        }
    )

    # Execution → END
    graph.add_edge("rag", END)
    graph.add_edge("action", END)

    return graph.compile()
