# app/workflows/raga_graph.py
from langgraph.graph import StateGraph, START, END
from app.nodes.raga_node import build_raga_node
from app.raga.state import RAGAState

def build_raga_graph(llm, vector_store):
    graph = StateGraph(RAGAState)

    graph.add_node(
        "raga",
        build_raga_node(llm, vector_store)
    )

    graph.add_edge(START, "raga")
    graph.add_edge("raga", END)

    return graph.compile()

