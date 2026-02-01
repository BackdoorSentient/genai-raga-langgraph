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

    # Wrap nodes with dependencies
    graph.add_node("refine", lambda s: refine_query(s, llm))
    graph.add_node("retrieve", lambda s: retrieve_docs(s, vector_store))
    graph.add_node("generate", lambda s: generate_answer(s, llm))
    graph.add_node("validate", lambda s: validate_answer(s, llm))

    # Flow
    graph.set_entry_point("refine")
    graph.add_edge("refine", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")

    # Conditional branching
    # def validation_router(state: AgentState):
    #     return "end" if state["grounded"] else "refine"

    # graph.add_conditional_edges(
    #     "validate",
    #     validation_router,
    #     {
    #         "refine": "refine",
    #         "end": END
    #     }
    # )

    # return graph.compile()
    def validation_router(state: AgentState):
        if state["grounded"]:
            return "end"
    
        if state["retry_count"] >= state["max_retries"]:
            return "end"   # exit with best answer
    
        return "refine"
