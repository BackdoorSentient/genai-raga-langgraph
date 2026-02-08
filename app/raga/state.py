from typing import TypedDict, List, Any, Optional


class RAGAState(TypedDict, total=False):
    #core
    query: str
    refined_query: str
    documents: List[Any]
    answer: str

    #validation
    grounded: bool
    confidence: float
    citations: List[str]

    # retry or control
    retry_count: int
    max_retries: int
    retry_reason: Optional[str]
    terminate: bool

    start_time: float
    timeout_seconds: int

    #observability
    steps: List[str]
    timeline: List[dict]

    #quality
    quality_score: float
