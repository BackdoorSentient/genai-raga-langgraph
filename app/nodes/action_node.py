from app.workflows.state_schema import RAGAState

def action_node(state: RAGAState) -> RAGAState:
    result = f"Action executed on answer: {state['answer']}"
    return {
        "query": state["query"],
        "answer": state["answer"],
        "action_result": result
    }
