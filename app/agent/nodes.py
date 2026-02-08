from app.agent.state import AgentState

def refine_query(state: AgentState, llm) -> AgentState:
    refined = llm.invoke(
        f"Rewrite this query for better document retrieval:\n{state['query']}"
    )

    state["steps"].append("Query refined")

    return {
        **state,
        "refined_query": refined
    }

def retrieve_docs(state: AgentState, vector_store) -> AgentState:
    query = state["refined_query"] or state["query"]
    docs = vector_store.search(query, k=4)
    sources = [doc.source for doc in docs]

    state["steps"].append(f"Retrieved {len(docs)} documents")

    return {**state, "documents": docs, "citations":sources}

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
        "citations": state["citations"]
    }

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

    if not grounded:
        confidence = 0.0

    retry_count = state["retry_count"] + (0 if grounded else 1)

    state["steps"].append("Answer validated")

    return {
        **state,
        "grounded": grounded,
        "confidence": confidence,
        "retry_count": retry_count,
        "citations":state['citations']
    }
