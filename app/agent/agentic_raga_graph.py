# app/workflows/agentic_raga_graph.py

from langgraph.graph import StateGraph, END

# Existing RAGA nodes (UNCHANGED)
from app.nodes.decision_node import decision_node
from app.nodes.rag_node import rag_node
from app.nodes.tool_node import tool_node
from app.nodes.summarize_node import summarize_node

# New Agentic nodes
from app.agent.planner import planner_node
from app.agent.critic import critic_node

from app.agent.state import AgentState

from functools import partial
from app.nodes.web_node import web_node


def build_agentic_raga_graph(vector_store):

    graph = StateGraph(AgentState)

    # ------------------
    # Register nodes
    # ------------------

    graph.add_node("planner", planner_node)

    graph.add_node("decision", decision_node)
    # FIX: inject vector store into tool node
    graph.add_node(
        "tool",
        partial(tool_node, retriever=vector_store)
    )
    graph.add_node("web", web_node)
    # graph.add_node("rag", rag_node)
    # graph.add_node("tool", tool_node)
    graph.add_node("summarize", summarize_node)

    graph.add_node("critic", critic_node)

    # ------------------
    # Entry point
    # ------------------

    graph.set_entry_point("planner")

    # ------------------
    # Planner → Decision
    # ------------------

    graph.add_edge("planner", "decision")

    # ------------------
    # Existing RAGA routing
    # ------------------

    graph.add_conditional_edges(
        "decision",
        lambda s: s["next_node"],
        {
            # "rag": "rag",
            "tool": "tool",
            "web": "web",
            "summarize": "summarize",
        }
    )

    graph.add_edge("tool", "decision")
    # graph.add_edge("rag", "summarize")
    # graph.add_edge("summarize", "critic")
    graph.add_edge("web", "decision")

    

    # ------------------
    # Summarize → Critic
    # ------------------

    graph.add_edge("summarize", "critic")

    # ------------------
    # Critic decision
    # ------------------

    graph.add_conditional_edges(
        "critic",
        lambda s: s["critic_decision"],
        {
            "retry": "planner",
            "accept": END
        }
    )

    return graph.compile()
