from typing import TypedDict, List, Any, Optional


class RAGAState(TypedDict, total=False):
    # ---- Core ----
    query: str
    refined_query: str
    documents: List[Any]
    answer: str

    # ---- Validation ----
    grounded: bool
    confidence: float
    citations: List[str]

    # ---- Retry / Control ----
    retry_count: int
    max_retries: int
    retry_reason: Optional[str]
    terminate: bool

    # ---- Timing ----
    start_time: float
    timeout_seconds: int

    # ---- Observability ----
    steps: List[str]
    timeline: List[dict]

    # ---- Quality ----
    quality_score: float
