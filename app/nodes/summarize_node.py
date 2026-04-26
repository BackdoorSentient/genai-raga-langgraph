# app/nodes/summarize_node.py
from app.agent.state import AgentState
from app.llm.factory import get_llm
from app.schema import Document


SYSTEM_PROMPT = """
You are a factual question answering assistant.

RULES:
- Answer ONLY using the provided documents
- Use ALL relevant information from the documents to construct a complete answer
- For questions about people, use any available information from web sources — LinkedIn, news articles, company pages, social media bios are all valid sources
- If the documents contain partial information, combine them into the best possible answer
- If the documents contain truly no relevant information at all — respond with exactly: "I don't have reliable information about this."
- Be specific and factual — 3-5 sentences
"""


def _compute_confidence(answer: str, documents: list) -> float:
    """
    Fix 5: Compute confidence based on actual answer-document overlap.
    Old method: min(0.95, 0.5 + len(docs) * 0.1) — based on doc count, meaningless.
    New method: fraction of documents that share meaningful content with the answer.

    Checks how many documents contain at least one 3-word sequence
    that also appears in the answer. More overlap = higher confidence.
    """
    if not documents or not answer:
        return 0.0

    answer_lower = answer.lower()

    # Build 3-word ngrams from the answer
    answer_words = answer_lower.split()
    answer_ngrams = set()
    for i in range(len(answer_words) - 2):
        ngram = " ".join(answer_words[i:i + 3])
        answer_ngrams.add(ngram)

    if not answer_ngrams:
        return 0.0

    matching_docs = 0
    for d in documents:
        content = ""
        if isinstance(d, Document):
            content = d.content.lower()
        elif isinstance(d, dict):
            content = d.get("content", "").lower()

        if not content:
            continue

        # Check if any 3-word ngram from the answer appears in this document
        doc_words = content.split()
        doc_ngrams = set()
        for i in range(len(doc_words) - 2):
            ngram = " ".join(doc_words[i:i + 3])
            doc_ngrams.add(ngram)

        if answer_ngrams & doc_ngrams:    # intersection — shared ngrams
            matching_docs += 1

    confidence = matching_docs / len(documents)
    return round(min(0.95, confidence), 2)


def summarize_node(state: AgentState) -> AgentState:
    documents = state.get("documents", [])
    query = state.get("query", "")

    if not documents:
        state["answer"] = "I don't have reliable information about this."
        state["grounded"] = False
        state["confidence"] = 0.0
        return state

    context_chunks = []
    for d in documents:
        if isinstance(d, Document):
            if d.content:
                context_chunks.append(f"[Source: {d.source}]\n{d.content}")
        elif isinstance(d, dict):
            text = d.get("content", "")
            if text:
                context_chunks.append(text)

    if not context_chunks:
        state["answer"] = "I don't have reliable information about this."
        state["grounded"] = False
        state["confidence"] = 0.0
        return state

    context = "\n\n".join(context_chunks)

    prompt = f"""
{SYSTEM_PROMPT}

Question:
{query}

Documents:
{context}

Answer:
"""

    try:
        llm = get_llm()
        answer = llm.generate(prompt)
    except Exception as exc:
        state["answer"] = "I don't have reliable information about this."
        state["grounded"] = False
        state["confidence"] = 0.0
        state.setdefault("observations", []).append({
            "node": "summarize",
            "error": f"LLM unavailable: {exc}"
        })
        state.setdefault("steps", []).append(
            "SummarizeNode → LLM failed, returning fallback"
        )
        return state

    state["answer"] = answer
    state["grounded"] = not any(
        phrase in answer.lower()
        for phrase in [
            "i don't know",
            "not found",
            "no information",
            "i don't have reliable information",
        ]
    )

    # Fix 5: actual groundedness check — not doc count heuristic
    state["confidence"] = _compute_confidence(answer, documents)

    state.setdefault("steps", []).append(
        "SummarizeNode → answer generated"
    )

    return state