from app.rag_system.rag_service import get_rag_answer
from app.workflows.state_schema import RAGAState

async def rag_node(state: RAGAState) -> RAGAState:
    result = await get_rag_answer(state["query"])

    return {
        "query": state["query"],
        "answer": result["answer"],
        "action_result": "",
        "sources": result.get("sources", [])
    }
