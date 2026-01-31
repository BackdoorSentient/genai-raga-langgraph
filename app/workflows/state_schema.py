from typing import TypedDict, List


class RAGAState(TypedDict):
    query: str
    answer: str
    action_result: str
    sources: List[str]
