from typing import TypedDict, List, Any, Dict

class AgentState(TypedDict, total=False):
    # ----- Existing RAGA -----
    query: str
    refined_query: str
    documents: List[Any]
    answer: str
    grounded: bool
    retry_count: int
    max_retries: int
    confidence: float
    citations: List[str]
    steps: List[str]

    # ----- Agentic RAGA v2 -----
    goal: str
    plan: List[Dict[str, Any]]
    current_step: int
    observations: List[Any]
    critic_decision: str
