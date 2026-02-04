import time


def refine_query(state, llm):
    refined = llm.invoke(
        f"Rewrite this query for better document retrieval:\n{state['query']}"
    )

    state["steps"].append("Query refined")
    return {**state, "refined_query": refined}


def retrieve_docs(state, vector_store):
    query = state.get("refined_query") or state["query"]
    docs = vector_store.search(query, k=4)
    sources = [doc.source for doc in docs]

    state["steps"].append(f"Retrieved {len(docs)} documents")

    return {
        **state,
        "documents": docs,
        "citations": sources
    }


def generate_answer(state, llm):
    if not state["documents"]:
        state["steps"].append("No documents found")
        return {
            **state,
            "answer": "I could not find relevant information.",
            "citations": []
        }

    context = "\n\n".join(d.page_content for d in state["documents"])

    prompt = f"""
    Answer ONLY using the context below.
    If unsure, say you are unsure.

    Context:
    {context}

    Question:
    {state["query"]}
    """

    answer = llm.invoke(prompt)
    state["steps"].append("Answer generated")

    return {**state, "answer": answer}


def validate_answer(state, llm):
    context = "\n\n".join(d.page_content for d in state["documents"])

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

    grounded = "YES" in result.upper()

    try:
        confidence = float(result.split("CONFIDENCE:")[1].strip())
    except:
        confidence = 0.0

    state["steps"].append("Answer validated")

    return {
        **state,
        "grounded": grounded,
        "confidence": confidence if grounded else 0.0
    }
