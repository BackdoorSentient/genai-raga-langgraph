# app/workflows/state_schema.py
from typing import TypedDict, List, Any, Dict


class RAGAState(TypedDict, total=False):
    # -------- Core --------
    query: str
    next_node: str

    # -------- Agentic --------
    goal: str
    plan: List[Dict[str, Any]]
    current_step: int
    observations: List[Any]
    critic_decision: str

    # -------- RAG --------
    documents: List[Any]
    answer: str
    grounded: bool
    confidence: float
    citations: List[str]

    # -------- Retry --------
    retry_count: int
    max_retries: int
