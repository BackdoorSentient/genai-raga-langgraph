from langgraph.graph import StateGraph, START, END
from app.nodes.raga_node import rag_node
from app.nodes.action_node import action_node
from app.workflows.state_schema import RAGAState

def build_raga_graph():
    graph = StateGraph(RAGAState)

    graph.add_node("rag", rag_node)
    graph.add_node("action", action_node)

    graph.add_edge(START, "rag")
    graph.add_edge("rag", "action")
    graph.add_edge("action", END)

    return graph.compile()
