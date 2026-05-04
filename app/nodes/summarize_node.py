# app/nodes/summarize_node.py
from app.agent.state import AgentState
from app.llm.factory import get_llm
from app.schema import Document


SYSTEM_PROMPT = """
You are a research assistant that compiles comprehensive answers from web search results.

RULES:
- Use ALL relevant information from the provided documents
- Compile everything into a thorough, well-structured answer
- For person queries: include name, profession, employer, location, education, social media, projects, achievements
- For factual queries: include current data, context, and any relevant details
- For news/events: include what happened, when, who was involved, and current status
- Combine information from multiple sources into one coherent answer
- If sources conflict, mention both versions
- Cite which sources provided key facts
- Be comprehensive — give the user everything the search found
- If truly nothing relevant was found, clearly state that
"""


def _compute_confidence(answer: str, documents: list) -> float:
    if not documents or not answer:
        return 0.0

    answer_lower = answer.lower()
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

        doc_words = content.split()
        doc_ngrams = set()
        for i in range(len(doc_words) - 2):
            ngram = " ".join(doc_words[i:i + 3])
            doc_ngrams.add(ngram)

        if answer_ngrams & doc_ngrams:
            matching_docs += 1

    return round(min(0.95, matching_docs / len(documents)), 2)


def summarize_node(state: AgentState) -> AgentState:
    documents = state.get("documents", [])
    query = state.get("query", "")

    if not documents:
        state["answer"] = "No information found for this query."
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
        state["answer"] = "No information found for this query."
        state["grounded"] = False
        state["confidence"] = 0.0
        return state

    context = "\n\n".join(context_chunks)

    prompt = f"""
{SYSTEM_PROMPT}

Question:
{query}

Search Results:
{context}

Comprehensive Answer:
"""

    try:
        llm = get_llm()
        answer = llm.generate(prompt)
    except Exception as exc:
        state["answer"] = "No information found for this query."
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
    state["grounded"] = True
    state["confidence"] = _compute_confidence(answer, documents)

    state.setdefault("steps", []).append(
        "SummarizeNode → answer generated"
    )

    return state