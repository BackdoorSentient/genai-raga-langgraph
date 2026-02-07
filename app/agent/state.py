from typing import TypedDict, List, Any, Dict, Literal

# class AgentState(TypedDict, total=False):
#     # ----- Existing RAGA -----
#     query: str
#     refined_query: str
#     documents: List[Any]
#     answer: str
#     grounded: bool
#     retry_count: int
#     max_retries: int
#     confidence: float
#     citations: List[str]
#     steps: List[str]

#     # ----- Agentic RAGA v2 -----
#     goal: str
#     plan: List[Dict[str, Any]]
#     current_step: int
#     observations: List[Any]
#     critic_decision: str


class AgentState(TypedDict, total=False):
    # ----- Core RAGA -----
    query: str
    refined_query: str
    documents: List[Any]
    answer: str
    grounded: bool
    confidence: float
    citations: List[str]
    steps: List[str]

    # ----- Retry control -----
    retry_count: int
    max_retries: int

    # ----- Routing -----
    # next_node: Literal["rag", "tool"]
    next_node: Literal["rag", "tool", "summarize"]

    # ----- Agentic RAGA v2 -----
    goal: str
    # plan: List[Dict[str, Any]]
    plan: List[str]
    current_step: int
    observations: List[Any]

    # ----- Critic -----
    critic_decision: Literal["retry", "accept"]
    critic_reason: str

    web_attempted: bool