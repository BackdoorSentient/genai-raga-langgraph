# app/agent/state.py
from typing import TypedDict, List, Any, Optional


class AgentState(TypedDict, total=False):
    # Core
    query: str
    goal: str
    refined_query: str

    # Planning
    plan: List[str]
    search_queries: List[str]          # ← new: planner-generated search queries
    current_step: int

    # Documents
    documents: List[Any]
    citations: List[str]

    # Answer
    answer: str
    grounded: bool
    confidence: float

    # Retry control
    retry_count: int
    max_retries: int
    phase: str
    used_vector: bool
    used_web: bool

    # Critic
    critic_decision: str
    critic_reason: str

    # Observability
    steps: List[str]
    observations: List[dict]