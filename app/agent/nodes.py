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
        "refined_query": refined.content
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
        "answer": answer.content
    }

# ---------------- VALIDATOR ----------------
def validate_answer(state: AgentState, llm) -> AgentState:
    validation_prompt = f"""
    Context:
    {state["documents"]}

    Answer:
    {state["answer"]}

    Is the answer fully grounded in the context?
    Reply ONLY YES or NO.
    """

    result = llm.invoke(validation_prompt)

    grounded = "YES" in result.content.upper()

    return {
        **state,
        "grounded": grounded
    }
