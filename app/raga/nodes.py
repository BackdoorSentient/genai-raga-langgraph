import time
from app.raga.state import RAGAState

def refine_query(state: RAGAState, llm) -> RAGAState:
    t0 = time.time()

    refined = llm.invoke(
        f"Rewrite this query for better document retrieval:\n{state['query']}"
    )

    latency = round((time.time() - t0) * 1000, 2)

    state.setdefault("steps", []).append("Query refined")
    state.setdefault("timeline", []).append({
        "node": "refine",
        "latency_ms": latency
    })

    state["refined_query"] = refined
    return state


# -------- RETRIEVER --------
def retrieve_docs(state: RAGAState, vector_store) -> RAGAState:
    t0 = time.time()

    query = state.get("refined_query") or state["query"]
    docs = vector_store.search(query, k=4)

    latency = round((time.time() - t0) * 1000, 2)

    state.setdefault("steps", []).append(f"Retrieved {len(docs)} documents")
    state.setdefault("timeline", []).append({
        "node": "retrieve",
        "latency_ms": latency
    })

    state["documents"] = docs
    state["citations"] = [doc.source for doc in docs]
    return state


# -------- GENERATOR --------
def generate_answer(state: RAGAState, llm) -> RAGAState:
    t0 = time.time()

    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = f"""
    Answer using ONLY the context.
    If unsure, say you are unsure.

    Context:
    {context}

    Question:
    {state['query']}
    """

    answer = llm.invoke(prompt)

    latency = round((time.time() - t0) * 1000, 2)

    state.setdefault("steps", []).append("Answer generated")
    state.setdefault("timeline", []).append({
        "node": "generate",
        "latency_ms": latency
    })

    state["answer"] = answer
    return state


# -------- VALIDATOR --------
def validate_answer(state: RAGAState, llm) -> RAGAState:
    t0 = time.time()

    context = "\n\n".join(doc.page_content for doc in state["documents"])

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
        confidence = float(result.upper().split("CONFIDENCE:")[1].strip())
    except:
        confidence = 0.0

    latency = round((time.time() - t0) * 1000, 2)

    state.setdefault("steps", []).append("Answer validated")
    state.setdefault("timeline", []).append({
        "node": "validate",
        "latency_ms": latency
    })

    state["grounded"] = grounded
    state["confidence"] = confidence if grounded else 0.0
    state["retry_count"] = state.get("retry_count", 0) + (0 if grounded else 1)

    return state
