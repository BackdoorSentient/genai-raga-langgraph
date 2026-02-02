from typing import TypedDict, List, Any

class AgentState(TypedDict, total=False):
    query: str
    refined_query: str
    documents: List[Any]
    answer: str
    grounded: bool
    retry_count: int
    max_retries: int
    steps: List[str]
    confidence: float
    citations: List[str]
