# app/agent/planner.py
from app.llm.factory import get_llm
from app.agent.state import AgentState
from app.utils.json_utils import extract_json


PLANNER_PROMPT = """
You are a search query planning agent.

User question:
{goal}

Your job is to generate multiple specific search queries that together will find 
the most complete and accurate answer to this question.

Think about:
- What is the user really asking?
- What are different angles to search this topic?
- What specific sources would have this information?

STRICT RULES:
- Output MUST be valid JSON
- NO markdown, NO explanations
- Start with '{{' and end with '}}'
- Generate 4 to 6 search queries
- Each query should target a different angle or source

Return ONLY this structure:

{{
  "search_queries": [
    "specific search query 1",
    "specific search query 2",
    "specific search query 3",
    "specific search query 4"
  ]
}}

Examples:

For "who is Elon Musk?":
{{"search_queries": ["Elon Musk biography", "Elon Musk Tesla SpaceX CEO", "Elon Musk net worth 2026", "Elon Musk latest news"]}}

For "gold price today":
{{"search_queries": ["gold price today USD", "gold rate today INR", "live gold spot price", "gold price per gram today"]}}

For "how does RAG work?":
{{"search_queries": ["retrieval augmented generation explained", "RAG architecture how it works", "RAG vs fine tuning", "RAG implementation guide"]}}
"""

FALLBACK_QUERIES = [
    "{query}",
    "{query} explained",
    "{query} latest",
]


def planner_node(state: AgentState) -> AgentState:
    goal = state.get("goal") or state.get("query")
    if not goal:
        raise ValueError("Planner → goal missing")

    try:
        llm = get_llm()
        response = llm.generate(PLANNER_PROMPT.format(goal=goal))
    except Exception as exc:
        query = state.get("query", "")
        state["plan"] = [q.format(query=query) for q in FALLBACK_QUERIES]
        state["search_queries"] = state["plan"]
        state["current_step"] = 0
        state["phase"] = "retrieve"
        state["used_vector"] = False
        state["used_web"] = False
        state.setdefault("observations", []).append({
            "node": "planner",
            "error": f"LLM unavailable: {exc}",
            "fallback": True
        })
        state.setdefault("steps", []).append(
            f"Planner → LLM failed, using fallback queries"
        )
        return state

    try:
        parsed = extract_json(response)
        queries = parsed.get("search_queries")

        if not isinstance(queries, list) or not all(
            isinstance(q, str) and q.strip() for q in queries
        ):
            raise ValueError("Invalid search_queries format")

    except Exception as e:
        query = state.get("query", "")
        queries = [q.format(query=query) for q in FALLBACK_QUERIES]
        state.setdefault("observations", []).append({
            "node": "planner",
            "error": str(e),
            "raw_output": response[:200]
        })

    state["plan"] = queries
    state["search_queries"] = queries          # ← pass to web_node
    state["current_step"] = 0
    state["phase"] = "retrieve"
    state["used_vector"] = False
    state["used_web"] = False

    state.setdefault("steps", []).append(
        f"Planner → {len(queries)} search queries generated"
    )

    return state