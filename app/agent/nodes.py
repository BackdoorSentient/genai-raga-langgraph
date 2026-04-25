# app/agent/nodes.py
from app.agent.state import AgentState


def refine_query(state: AgentState, llm) -> AgentState:
    # ── Issue B: wrap LLM call ────────────────────────────────────────────────
    try:
        refined = llm.invoke(
            f"Rewrite this query for better document retrieval:\n{state['query']}"
        )
    except Exception as exc:
        refined = state["query"]    # fallback: use original query unchanged
        state.setdefault("observations", []).append({
            "node": "refine_query",
            "error": f"LLM unavailable: {exc}",
            "fallback": "original query used"
        })
    # ─────────────────────────────────────────────────────────────────────────

    state["steps"].append("Query refined")

    return {
        **state,
        "refined_query": refined,
    }


def retrieve_docs(state: AgentState, vector_store) -> AgentState:
    query = state["refined_query"] or state["query"]
    docs = vector_store.search(query, k=4)
    sources = [doc.source for doc in docs]

    state["steps"].append(f"Retrieved {len(docs)} documents")

    return {**state, "documents": docs, "citations": sources}


def generate_answer(state: AgentState, llm) -> AgentState:
    if not state["documents"]:
        return {
            **state,
            "answer": "I could not find relevant information in the documents.",
            "citations": [],
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

    # ── Issue B: wrap LLM call ────────────────────────────────────────────────
    try:
        answer = llm.invoke(prompt)
    except Exception as exc:
        answer = "I could not generate an answer due to a system error."
        state.setdefault("observations", []).append({
            "node": "generate_answer",
            "error": f"LLM unavailable: {exc}"
        })
    # ─────────────────────────────────────────────────────────────────────────

    state["steps"].append("Answer generated")

    return {
        **state,
        "answer": answer,
        "citations": state["citations"],
    }


def validate_answer(state: AgentState, llm) -> AgentState:
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

    # ── Issue B: wrap LLM call ────────────────────────────────────────────────
    try:
        result = llm.invoke(prompt)
    except Exception as exc:
        state.setdefault("observations", []).append({
            "node": "validate_answer",
            "error": f"LLM unavailable: {exc}"
        })
        state["steps"].append("Answer validated — LLM failed, defaulting grounded=False")
        return {
            **state,
            "grounded": False,
            "confidence": 0.0,
            # ── Issue A: NO retry_count here — router_node owns it ────────────
        }
    # ─────────────────────────────────────────────────────────────────────────

    grounded = "GROUNDED: YES" in result.upper()

    try:
        confidence = float(result.upper().split("CONFIDENCE:")[1].strip())
    except Exception:
        confidence = 0.0

    if not grounded:
        confidence = 0.0

    state["steps"].append("Answer validated")

    return {
        **state,
        "grounded": grounded,
        "confidence": confidence,
        # ── Issue A: retry_count removed — router_node is the single authority
        #    that increments it. Having it here too caused premature exit. ─────
    }