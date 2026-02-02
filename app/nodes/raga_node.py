from app.rag_system.rag_service import get_rag_answer
from app.workflows.state_schema import RAGAState
import asyncio

def rag_node(state: RAGAState) -> RAGAState:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result = loop.run_until_complete(
            get_rag_answer(state["query"])
        )
    else:
        result = asyncio.run(
            get_rag_answer(state["query"])
        )

    return {
        **state,   
        "query": state["query"],
        "answer": result["answer"],
        "action_result": "",
        "sources": result.get("sources", [])
    }
