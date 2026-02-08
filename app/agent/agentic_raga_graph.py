from langgraph.graph import StateGraph, END
from functools import partial

from app.agent.state import AgentState
from app.agent.planner import planner_node
from app.agent.router_node import router_node
from app.nodes.tool_node import tool_node
from app.nodes.web_node import web_node
from app.nodes.summarize_node import summarize_node
from app.agent.critic import critic_node


def build_agentic_raga_graph(vector_store):

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("router", router_node)
    graph.add_node("tool", partial(tool_node, retriever=vector_store))
    graph.add_node("web", web_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "router")

    graph.add_conditional_edges(
        "router",
        lambda s: s["next_node"],
        {
            "tool": "tool",
            "web": "web",
            "summarize": "summarize",
            "critic": "critic",
            "__end__": END
        }
    )

    graph.add_edge("tool", "router")
    graph.add_edge("web", "router")
    graph.add_edge("summarize", "router")
    graph.add_edge("critic", "router")

    return graph.compile()
