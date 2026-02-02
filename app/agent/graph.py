from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import (
    refine_query,
    retrieve_docs,
    generate_answer,
    validate_answer
)


def build_rag_graph(llm, vector_store):

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("refine", lambda s: refine_query(s, llm))
    graph.add_node("retrieve", lambda s: retrieve_docs(s, vector_store))
    graph.add_node("generate", lambda s: generate_answer(s, llm))
    graph.add_node("validate", lambda s: validate_answer(s, llm))

    # Linear flow
    graph.set_entry_point("refine")
    graph.add_edge("refine", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")

    # Conditional routing after validation
    def validation_router(state: AgentState):
        if state["grounded"]:
            return END

        if state["insufficient_context"]:
            return END

        if state["retry_count"] >= state["max_retries"]:
            return END

        return "refine"

    graph.add_conditional_edges(
        "validate",
        validation_router
    )

    return graph.compile()
