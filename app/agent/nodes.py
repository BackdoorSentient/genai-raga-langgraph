from app.agent.state import AgentState


def token_overlap(answer: str, context: str) -> float:
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())

    if not answer_tokens:
        return 0.0

    return len(answer_tokens & context_tokens) / len(answer_tokens)


# ---------------- QUERY REFINER ----------------
def refine_query(state: AgentState, llm) -> AgentState:
    result = llm.invoke(
        f"Rewrite this query for better document retrieval:\n{state['query']}"
    )

    refined_query = (
        result.content if hasattr(result, "content") else str(result)
    )

    state["steps"].append("Query refined")

    return {
        **state,
        "refined_query": refined_query
    }


# ---------------- RETRIEVER ----------------
def retrieve_docs(state: AgentState, vector_store) -> AgentState:
    query = state.get("refined_query") or state["query"]

    base_k = 4
    k = base_k + (state["retry_count"] * 2)

    docs = vector_store.search(query, k=k)
    sources = [doc.source for doc in docs]

    state["steps"].append(f"Retrieved {len(docs)} documents (k={k})")

    return {
        **state,
        "documents": docs,
        "citations": sources
    }


# ---------------- GENERATOR ----------------
def generate_answer(state: AgentState, llm) -> AgentState:
    if not state["documents"]:
        state["steps"].append("No documents retrieved")

        return {
            **state,
            "answer": "Insufficient context to answer the question."
        }

    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "I am unsure".

Context:
{context}

Question:
{state["query"]}
"""

    result = llm.invoke(prompt)
    answer = result.content if hasattr(result, "content") else str(result)

    state["steps"].append("Answer generated")

    return {
        **state,
        "answer": answer
    }


# ---------------- VALIDATOR ----------------
def validate_answer(state: AgentState, llm) -> AgentState:
    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = f"""
Context:
{context}

Answer:
{state['answer']}

Decide:
1. Is the answer grounded in the context?
2. Provide a confidence score between 0 and 1.

Format:
GROUNDED: YES or NO
CONFIDENCE: <number>
"""

    result = llm.invoke(prompt)
    result_text = (
        result.content.upper()
        if hasattr(result, "content")
        else str(result).upper()
    )

    grounded = "GROUNDED: YES" in result_text

    try:
        confidence = float(result_text.split("CONFIDENCE:")[1].strip())
    except Exception:
        confidence = 0.0

    insufficient_context = grounded and confidence < 0.2

    retry_count = state["retry_count"]
    if not grounded and not insufficient_context:
        retry_count += 1

    state["steps"].append(
        f"Validated | grounded={grounded} | confidence={confidence}"
    )

    return {
        **state,
        "grounded": grounded,
        "confidence": confidence,
        "retry_count": retry_count,
        "insufficient_context": insufficient_context
    }
