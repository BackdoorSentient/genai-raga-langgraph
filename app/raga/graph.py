from langgraph.graph import StateGraph, END
from app.raga.state import RAGAState
from app.raga.nodes import (
    refine_query,
    retrieve_docs,
    generate_answer,
    validate_answer
)
from app.raga.critic import critic_node


def build_raga_graph(llm, vector_store):
    graph = StateGraph(RAGAState)

    graph.add_node("refine", lambda s: refine_query(s, llm))
    graph.add_node("retrieve", lambda s: retrieve_docs(s, vector_store))
    graph.add_node("generate", lambda s: generate_answer(s, llm))
    graph.add_node("validate", lambda s: validate_answer(s, llm))
    graph.add_node("critic", critic_node)

    graph.set_entry_point("refine")

    graph.add_edge("refine", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "critic")

    # ✅ Router must return STRING, not dict
    def route(state: RAGAState):
        return END if state.get("terminate") else "refine"

    graph.add_conditional_edges(
        "critic",
        route,
        {
            "refine": "refine",
            END: END,
        },
    )

    return graph.compile()
