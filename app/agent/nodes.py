# from langchain.schema import Document
# from app.agent.document import Document
from app.agent.state import AgentState

# ---------------- QUERY REFINER ----------------
def refine_query(state: AgentState, llm) -> AgentState:
    refined = llm.invoke(
        f"Rewrite this query for better document retrieval:\n{state['query']}"
    )

    state["steps"].append("Query refined")

    return {
        **state,
        "refined_query": refined,
        "retry_count": state["retry_count"] + 1
    }

# ---------------- RETRIEVER ----------------
def retrieve_docs(state: AgentState, vector_store) -> AgentState:
    query = state["refined_query"] or state["query"]
    docs = vector_store.search(query, k=4)

    state["steps"].append(f"Retrieved {len(docs)} documents")

    return {**state, "documents": docs}

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

    state["steps"].append("Answer generated")

    return {
        **state,
        # "answer": answer.content
        "answer": answer
    }

# ---------------- VALIDATOR ----------------
def validate_answer(state: AgentState, llm) -> AgentState:
    prompt = f"""
    Context:
    {state['documents']}

    Answer:
    {state['answer']}

    1. Is the answer grounded in the context?
    2. Give confidence score between 0 and 1.

    Respond as:
    GROUNDED: YES/NO
    CONFIDENCE: <number>
    """

    result = llm.invoke(prompt)

    grounded = "YES" in result.upper()

    try:
        confidence = float(result.split("CONFIDENCE:")[1].strip())
    except:
        confidence = 0.0

    state["steps"].append("Answer validated")

    return {
        **state,
        "grounded": grounded,
        "confidence": confidence
    }
