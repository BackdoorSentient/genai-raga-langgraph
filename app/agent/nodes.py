# from langchain.schema import Document
# from app.agent.document import Document
from app.agent.state import AgentState

# ---------------- QUERY REFINER ----------------
def refine_query(state: AgentState, llm) -> AgentState:
    query = state["query"]

    refined = llm.invoke(
        f"Rewrite this query for better document retrieval:\n{query}"
    )

    return {
        **state,
        # "refined_query": refined.content
        "refined_query": refined
    }

# ---------------- RETRIEVER ----------------
def retrieve_docs(state: AgentState, vector_store) -> AgentState:
    query = state.get("refined_query") or state["query"]

    docs = vector_store.search(query, k=4)

    return {
        **state,
        "documents": docs
    }

# ---------------- GENERATOR ----------------
def generate_answer(state: AgentState, llm) -> AgentState:
    context = "\n\n".join(
        doc.page_content for doc in state["documents"]
    )

    prompt = f"""
    Use ONLY the context below to answer.

    Context:
    {context}

    Question:
    {state["query"]}
    """

    answer = llm.invoke(prompt)

    return {
        **state,
        # "answer": answer.content
        "answer": answer
    }

# ---------------- VALIDATOR ----------------
def validate_answer(state: AgentState, llm) -> AgentState:
    validation_prompt = f"""
    You are a strict fact checker.

    CONTEXT:
    {state["documents"]}

    ANSWER:
    {state["answer"]}

    Question:
    {state["query"]}

    Is the answer fully supported by the context?
    Reply ONLY YES or NO.
    """

    result = llm.invoke(validation_prompt)
    grounded = "YES" in result.upper()

    retry_count = state.get("retry_count", 0) + 1

    return {
        **state,
        "grounded": grounded,
        "retry_count": retry_count
    }
