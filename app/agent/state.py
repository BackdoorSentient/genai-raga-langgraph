from typing import TypedDict, List, Any, Literal


class AgentState(TypedDict, total=False):
    # -------- Core --------
    query: str
    refined_query: str
    answer: str
    documents: List[Any]
    grounded: bool
    confidence: float
    citations: List[str]

    # -------- Planning --------
    goal: str
    plan: List[str]
    current_step: int

    # -------- Control --------
    phase: Literal["retrieve", "summarize", "critic", "end"]
    next_node: str

    # -------- Retrieval flags --------
    used_vector: bool
    used_web: bool

    # -------- Retry --------
    retry_count: int
    max_retries: int

    # -------- Observability --------
    observations: List[Any]
    steps: List[str]

    # -------- Critic --------
    critic_decision: Literal["accept", "retry"]
    critic_reason: str
