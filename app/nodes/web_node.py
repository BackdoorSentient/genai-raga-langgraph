# app/nodes/web_node.py
from typing import List
from ddgs import DDGS
from app.agent.state import AgentState
from app.schema import Document


def web_node(state: AgentState) -> AgentState:
    # Use planner-generated search queries if available
    # Fall back to raw query if planner didn't run
    search_queries = state.get("search_queries")

    if not search_queries:
        query = state.get("refined_query") or state.get("query")
        if not query:
            raise ValueError("WebNode → query missing")
        search_queries = [query]

    web_documents: List[Document] = []
    seen_urls: set = set()

    try:
        with DDGS() as ddgs:
            for search_query in search_queries:
                try:
                    results = ddgs.text(search_query, max_results=5)

                    for r in results:
                        href = r.get("href", "")
                        body = r.get("body", "")

                        # Skip duplicates
                        if href in seen_urls:
                            continue
                        seen_urls.add(href)

                        if body:
                            web_documents.append(
                                Document(
                                    content=body,
                                    source=href,
                                    origin="web",
                                )
                            )

                except Exception:
                    # One query failing doesn't stop the others
                    continue

    except Exception as exc:
        state.setdefault("observations", []).append({
            "node": "web",
            "error": f"DuckDuckGo search failed: {exc}"
        })

    existing_docs = state.get("documents", [])
    state["documents"] = existing_docs + web_documents
    state["used_web"] = True

    state.setdefault("citations", [])
    for doc in web_documents:
        if doc.source:
            state["citations"].append(doc.source)

    state.setdefault("steps", []).append(
        f"WebNode → {len(web_documents)} web docs from {len(search_queries)} queries "
        f"(total={len(state['documents'])})"
    )

    state.setdefault("observations", []).append({
        "node": "web",
        "queries_run": len(search_queries),
        "documents_found": len(web_documents),
        "total_documents": len(state["documents"]),
        "source": "web",
    })

    return state