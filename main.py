# app/main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from app.utils.ollama import is_ollama_running
from app.rag_system.vector_store import VectorStore
from app.config.settings import get_settings
from app.agent.graph import build_rag_graph

# Import LangGraph RAGA
# from app.agent.graph import StateGraph, END, CompiledGraph
from langchain_community.llms.ollama import Ollama
from app.agent.state import AgentState
from typing import Any

settings = get_settings()
app = FastAPI(title=settings.APP_NAME)

# Globals
vector_store = VectorStore()
# rag_agent: "Graph" | None = None  # type: ignore
# rag_agent= "Graph"
rag_agent: Any = None  # type hint fixed

# Initialize LLM client
llm = Ollama(
    model=settings.OLLAMA_MODEL,
    base_url=settings.OLLAMA_BASE_URL
)


async def initialize_raga_pipeline():
    """
    Helper to load documents, build vector store, and initialize RAGA agent.
    """
    global rag_agent

    # Load documents
    from app.rag_system.ingestion import load_and_chunk_docs
    documents = load_and_chunk_docs(settings.RAW_DATA_DIR)

    # Build vector store (FAISS)
    vector_store.build_or_load(documents)

    # Build RAGA LangGraph
    rag_agent = build_rag_graph(llm, vector_store)


# @app.on_event("startup")
# async def startup_event():
#     """
#     Startup event: load documents, build vector store, and init RAGA agent.
#     """
#     try:
#         await initialize_raga_pipeline()
#         print("✅ RAGA pipeline initialized successfully")
#     except Exception as e:
#         print(f"Startup failed: {e}")

#     if not is_ollama_running():
#         print("WARNING: Ollama is NOT running. Start with `ollama serve`.")
@app.on_event("startup")
async def startup_event():
    """
     Startup event: load documents, build vector store, and init RAGA agent.
     """
    global rag_agent

    try:
        # 1. Build / load vector store
        vector_store.build_or_load()

        # 2. Create LLM
        llm = Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )

        # 3. Build agentic RAG graph
        rag_agent = build_rag_graph(llm, vector_store)

        print("RAGA agent initialized successfully")

    except Exception as e:
        print(f"Failed to initialize RAGA agent: {e}")

    if not is_ollama_running():
        print("Ollama is NOT running. Start with `ollama serve`.")

@app.post("/reload")
async def reload_raga_pipeline():
    """
    Manually reload documents, rebuild vector store, and reinitialize RAGA agent.
    """
    global rag_agent

    try:
        # 1. Build / load vector store
        vector_store.build_or_load()

        # 2. Create LLM
        llm = Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )

        # 3. Build agentic RAG graph
        rag_agent = build_rag_graph(llm, vector_store)

        print("RAGA agent initialized successfully")

    except Exception as e:
        print(f"Failed to initialize RAGA agent: {e}")

    if not is_ollama_running():
        print("Ollama is NOT running. Start with `ollama serve`.")

# @app.post("/reload")
# async def reload_raga_pipeline():
#     """
#     Manually reload documents, rebuild vector store, and reinitialize RAGA agent.
#     """
#     try:
#         await initialize_raga_pipeline()
#         return {"status": "RAGA pipeline reloaded successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to reload: {e}")


@app.get("/ollama/health")
def ollama_health():
    return {
        "ollama_running": is_ollama_running(),
        "ollama_url": settings.OLLAMA_BASE_URL
    }


# @app.post("/raga")
# async def raga_query(query: str):
#     """
#     Ask a question to the RAGA agentic pipeline
#     """
#     if not rag_agent:
#         raise HTTPException(503, "RAGA pipeline not initialized")

#     if not is_ollama_running():
#         raise HTTPException(
#             status_code=503,
#             detail="Ollama is not running. Start it using `ollama serve`."
#         )

#     # Initial state
#     state: AgentState = {
#         "query": query,
#         "refined_query": "",
#         "documents": [],
#         "answer": "",
#         "grounded": False
#     }

#     # Run agent
#     result = rag_agent.invoke(state)

#     return {
#         "query": query,
#         "answer": result["answer"],
#         "grounded": result["grounded"]
#     }
@app.post("/raga")
async def raga_query(query: str):
    
    if not rag_agent:
        raise HTTPException(503, "RAGA pipeline not initialized")

    if not is_ollama_running():
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it using `ollama serve`."
        )

    # Initial agent state
    state: AgentState = {
        "query": query,
        "refined_query": "",
        "documents": [],
        "answer": "",
        "grounded": False,
        "retry_count": 0,
        "max_retries": 2
    }

    result = rag_agent.invoke(state)

    return {
        "query": query,
        "answer": result["answer"],
        "grounded": result["grounded"],
        "retries_used": result["retry_count"]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "raga_ready": rag_agent is not None,
        "vector_store_loaded": vector_store.db is not None,
        "ollama_running": is_ollama_running()
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
