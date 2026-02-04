from typing import TypedDict, List, Any, Optional


class RAGAState(TypedDict, total=False):
    # Core
    query: str
    refined_query: str

    # RAG
    documents: List[Any]
    answer: str
    citations: List[str]

    # Validation
    grounded: bool
    confidence: float
    retry_reason: Optional[str]

    # Retry
    retry_count: int
    max_retries: int

    # Observability
    steps: List[str]
    start_time: float
    timeout_seconds: float
