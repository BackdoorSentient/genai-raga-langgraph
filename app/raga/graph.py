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

    # ---------------- Nodes ----------------
    graph.add_node("refine", lambda s: refine_query(s, llm))
    graph.add_node("retrieve", lambda s: retrieve_docs(s, vector_store))
    graph.add_node("generate", lambda s: generate_answer(s, llm))
    graph.add_node("validate", lambda s: validate_answer(s, llm))
    graph.add_node("critic", critic_node)

    # ---------------- Entry ----------------
    graph.set_entry_point("refine")

    # ---------------- Flow ----------------
    graph.add_edge("refine", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "critic")

    # ---------------- Router ----------------
    def critic_router(state: RAGAState):
        """
        Decide whether to retry or terminate.
        MUST return either a node name OR END.
        """
        if state.get("terminate"):
            return END
        return "refine"

    graph.add_conditional_edges(
        "critic",
        critic_router,
        {
            "refine": "refine",
            END: END
        }
    )

    return graph.compile()
