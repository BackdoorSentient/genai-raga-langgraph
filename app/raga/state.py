from typing import TypedDict, List, Any, Optional


class RAGAState(TypedDict, total=False):
    # ---- Core ----
    query: str
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

    # ---- Time ----
    start_time: float
    timeout_seconds: int

    # ---- Tracing ----
    steps: List[str]
