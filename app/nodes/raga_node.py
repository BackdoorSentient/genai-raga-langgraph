from langgraph import Node
from app.rag_system.rag_service import get_answer


class RAGANode(Node):
    """
    LangGraph node that wraps the RAG system
    """

    def run(self, query: str) -> str:
        return get_answer(query)
