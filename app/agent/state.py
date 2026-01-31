from typing import TypedDict, List
# from langchain_community.document_loaders import Document
from app.agent.document import Document

class AgentState(TypedDict):
    query: str
    refined_query: str
    documents: List[Document]
    answer: str
    grounded: bool
