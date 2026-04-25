# app/raga/nodes.py
import time
from app.raga.state import RAGAState


def refine_query(state: RAGAState, llm) -> RAGAState:
    t0 = time.time()

    # ── Issue 11: wrap LLM call ───────────────────────────────────────────────
    try:
        refined = llm.invoke(
            f"Rewrite this query for better document retrieval:\n{state['query']}"
        )
    except Exception as exc:
        refined = state["query"]   # fallback: use original query unchanged
        state.setdefault("observations", []).append({
            "node": "refine_query",
            "error": f"LLM unavailable: {exc}"
        })
    # ─────────────────────────────────────────────────────────────────────────

    latency = round((time.time() - t0) * 1000, 2)

    state.setdefault("steps", []).append("Query refined")
    state.setdefault("timeline", []).append({
        "node": "refine",
        "latency_ms": latency
    })

    state["refined_query"] = refined
    return state


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


def generate_answer(state: RAGAState, llm) -> RAGAState:
    t0 = time.time()

    docs = state.get("documents", [])

    if not docs:
        state["answer"] = "I don't have reliable information about this."
        state.setdefault("steps", []).append("Answer generation skipped — no documents")
        return state

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
    Answer using ONLY the context.
    If the context does not contain a clear answer, say: "I don't have reliable information about this."

    Context:
    {context}

    Question:
    {state['query']}
    """

    # ── Issue 11: wrap LLM call ───────────────────────────────────────────────
    try:
        answer = llm.invoke(prompt)
    except Exception as exc:
        answer = "I don't have reliable information about this."
        state.setdefault("observations", []).append({
            "node": "generate_answer",
            "error": f"LLM unavailable: {exc}"
        })
    # ─────────────────────────────────────────────────────────────────────────

    latency = round((time.time() - t0) * 1000, 2)

    state.setdefault("steps", []).append("Answer generated")
    state.setdefault("timeline", []).append({
        "node": "generate",
        "latency_ms": latency
    })

    state["answer"] = answer
    return state


def validate_answer(state: RAGAState, llm) -> RAGAState:
    t0 = time.time()

    docs = state.get("documents", [])

    if not docs:
        state["grounded"] = False
        state["confidence"] = 0.0
        # ── Issue 5: NO retry_count increment — raga critic handles it ───────
        return state

    context = "\n\n".join(doc.page_content for doc in docs)

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

    # ── Issue 11: wrap LLM call ───────────────────────────────────────────────
    try:
        result = llm.invoke(prompt)
    except Exception as exc:
        state["grounded"] = False
        state["confidence"] = 0.0
        state.setdefault("observations", []).append({
            "node": "validate_answer",
            "error": f"LLM unavailable: {exc}"
        })
        # ── Issue 5: NO retry_count increment here ────────────────────────────
        return state
    # ─────────────────────────────────────────────────────────────────────────

    grounded = "GROUNDED: YES" in result.upper()
    try:
        confidence = float(result.upper().split("CONFIDENCE:")[1].strip())
    except Exception:
        confidence = 0.0

    latency = round((time.time() - t0) * 1000, 2)

    state.setdefault("steps", []).append("Answer validated")
    state.setdefault("timeline", []).append({
        "node": "validate",
        "latency_ms": latency
    })

    state["grounded"] = grounded
    state["confidence"] = confidence if grounded else 0.0
    # ── Issue 5: retry_count removed — raga/critic.py owns the increment ─────

    return state