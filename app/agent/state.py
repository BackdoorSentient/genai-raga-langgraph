from typing import List, Optional, TypedDict, Any


class AgentState(TypedDict):
    # Input
    query: str

    # Query refinement
    refined_query: Optional[str]

    # Retrieval
    documents: List[Any]
    citations: List[str]

    # Generation
    answer: Optional[str]

    # Validation
    grounded: bool
    confidence: float
    insufficient_context: bool

    # Control flow
    retry_count: int
    max_retries: int

    # Debug / tracing
    steps: List[str]
