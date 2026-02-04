# app/workflows/agentic_raga_graph.py

from langgraph.graph import StateGraph, END

# Existing RAGA nodes (UNCHANGED)
from app.nodes.decision_node import decision_node
from app.nodes.raga_node import rag_node
from app.nodes.tool_node import tool_node
from app.nodes.summarize_node import summarize_node

# New Agentic nodes
from app.agent.planner import planner_node
from app.agent.critic import critic_node

from app.agent.state import AgentState


def build_agentic_raga_graph():
    """
    Agentic RAGA v2

    Flow:
    Goal
      → Planner
      → Decision
      → RAG / Tool
      → Summarize
      → Critic
      → Retry (Planner) OR End
    """

    graph = StateGraph(AgentState)

    # ------------------
    # Register nodes
    # ------------------

    graph.add_node("planner", planner_node)

    graph.add_node("decision", decision_node)
    graph.add_node("rag", rag_node)
    graph.add_node("tool", tool_node)
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
        lambda state: state["next_node"],
        {
            "rag": "rag",
            "tool": "tool",
            "summarize": "summarize"
        }
    )

    graph.add_edge("rag", "summarize")
    graph.add_edge("tool", "summarize")

    # ------------------
    # Summarize → Critic
    # ------------------

    graph.add_edge("summarize", "critic")

    # ------------------
    # Critic decision
    # ------------------

    graph.add_conditional_edges(
        "critic",
        lambda state: state["critic_decision"],
        {
            "retry": "planner",
            "accept": END
        }
    )

    return graph.compile()
