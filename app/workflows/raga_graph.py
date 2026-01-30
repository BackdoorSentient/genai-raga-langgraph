from langgraph import Graph
from app.nodes.raga_node import RAGANode
from app.nodes.action_node import ActionNode


def build_raga_graph() -> Graph:
    graph = Graph()

    graph.add_node(RAGANode(name="raga_node"))
    graph.add_node(ActionNode(name="action_node"))

    graph.add_edge("raga_node", "action_node")

    return graph
