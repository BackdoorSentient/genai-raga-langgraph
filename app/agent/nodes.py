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
        "refined_query": refined
    }

# ---------------- RETRIEVER ----------------
def retrieve_docs(state: AgentState, vector_store) -> AgentState:
    query = state["refined_query"] or state["query"]
    docs = vector_store.search(query, k=4)
    sources = [doc.source for doc in docs]

    state["steps"].append(f"Retrieved {len(docs)} documents")

    return {**state, "documents": docs, "citations":sources}

# ---------------- GENERATOR ----------------
def generate_answer(state: AgentState, llm) -> AgentState:
    if not state["documents"]:
        return {
            **state,
            "answer": "I could not find relevant information in the documents.",
            "citations": []
        }

    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = f"""
    Answer the question using ONLY the context.
    If unsure, say you are unsure.

    Context:
    {context}

    Question:
    {state["query"]}
    """

    answer = llm.invoke(prompt)

    state["steps"].append("Answer generated")

    return {
        **state,
        "answer": answer,
        "citations": [f"doc_{i}" for i in range(len(state["citations"]))]
    }


# ---------------- VALIDATOR ----------------
def validate_answer(state: AgentState, llm) -> AgentState:
    context = "\n\n".join(
        doc.page_content for doc in state["documents"]
    )

    prompt = f"""
    Context:
    {context}

    Answer:
    {state['answer']}

    Is the answer strictly grounded in the context?
    Reply ONLY with:
    GROUNDED: YES or NO
    CONFIDENCE: <0 to 1>
    """

    result = llm.invoke(prompt)

    grounded = "GROUNDED: YES" in result.upper()

    try:
        confidence = float(
            result.upper().split("CONFIDENCE:")[1].strip()
        )
    except:
        confidence = 0.0

    #increment retry only if not grounded
    retry_count = state["retry_count"] + (0 if grounded else 1)

    state["steps"].append("Answer validated")

    return {
        **state,
        "grounded": grounded,
        "confidence": confidence,
        "retry_count": retry_count,
        "citations":state['citations']
    }
