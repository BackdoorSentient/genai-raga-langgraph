from app.agent.state import AgentState
from typing import List
from duckduckgo_search import DDGS


def web_node(state: AgentState) -> AgentState:
    query = state.get("refined_query") or state.get("query")

    if not query:
        raise ValueError("WebNode → query missing")

    results: List[dict] = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append({
                "content": r.get("body", ""),
                "source": r.get("href", "")
            })

    # Convert web results to pseudo-documents
    documents = []
    for r in results:
        documents.append(
            type(
                "WebDoc",
                (),
                {
                    "page_content": r["content"],
                    "metadata": {"source": r["source"]}
                }
            )()
        )

    state["documents"] = documents

    state.setdefault("observations", []).append({
        "node": "web",
        "query": query,
        "results": len(documents)
    })

    state.setdefault("steps", []).append(
        f"WebNode → retrieved {len(documents)} web results"
    )

    return state
