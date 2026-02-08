from app.raga.state import RAGAState
from app.agent.nodes import (
    refine_query,
    retrieve_docs,
    generate_answer,
    validate_answer,
)

def build_raga_node(llm, vector_store):
    def raga_node(state: RAGAState) -> RAGAState:
        state.setdefault("steps", [])
        state.setdefault("retry_count", 0)
        state.setdefault("max_retries", 2)
        state.setdefault("grounded", False)

        while state["retry_count"] <= state["max_retries"]:
            state["steps"].append(
                f"RAGA attempt {state['retry_count'] + 1}"
            )

            state = refine_query(state, llm)

            state = retrieve_docs(state, vector_store)

            state = generate_answer(state, llm)

            state = validate_answer(state, llm)

            if state["grounded"]:
                state["steps"].append("Answer accepted (grounded)")
                break

            state["steps"].append("Answer not grounded, retrying")

        return state

    return raga_node
