# app/state.py
from typing import Any

app_state: dict[str, Any] = {
    "rag_pipeline": None,
    "rag_agent": None,
    "raga_agent": None,
    "agentic_raga_agent": None,
    "vector_store": None,
}


async def initialize_pipelines(state: dict):
    from app.rag_system.vector_store import VectorStore
    from app.rag_system.ingestion import load_and_chunk_docs
    from app.rag_system.rag_pipeline import RAGPipeline
    from app.agent.graph import build_rag_graph
    from app.raga.graph import build_raga_graph
    from app.agent.agentic_raga_graph import build_agentic_raga_graph
    from app.llm.factory import get_llm
    from app.config.settings import get_settings

    settings = get_settings()

    if state["vector_store"] is None:
        state["vector_store"] = VectorStore()

    documents = load_and_chunk_docs(settings.RAW_DATA_DIR)
    state["vector_store"].build_or_load(documents)

    llm = get_llm()

    state["rag_pipeline"] = RAGPipeline(state["vector_store"])
    state["rag_agent"] = build_rag_graph(llm, state["vector_store"])
    state["raga_agent"] = build_raga_graph(llm, state["vector_store"])
    state["agentic_raga_agent"] = build_agentic_raga_graph(state["vector_store"])