
import uvicorn
from fastapi import FastAPI, HTTPException
from typing import Any

from langchain_community.llms.ollama import Ollama

from app.utils.ollama import is_ollama_running
from app.rag_system.vector_store import VectorStore
from app.rag_system.ingestion import load_and_chunk_docs
from app.config.settings import get_settings
from app.agent.graph import build_rag_graph
from app.agent.state import AgentState

from app.rag_system.rag_pipeline import RAGPipeline

settings = get_settings()
app = FastAPI(title=settings.APP_NAME)

vector_store = VectorStore()
rag_agent: Any = None

rag_pipeline: RAGPipeline | None = None
rag_agent: Any = None

llm = Ollama(
    model=settings.OLLAMA_MODEL,
    base_url=settings.OLLAMA_BASE_URL
)

# Initialization helper
async def initialize_pipelines():
    global rag_pipeline, rag_agent

    # Load & chunk documents
    documents = load_and_chunk_docs(settings.RAW_DATA_DIR)

    # Build vector store
    vector_store.build_or_load(documents)

    # Initialize classic RAG
    rag_pipeline = RAGPipeline(vector_store)

    # Initialize agentic RAG (LangGraph)
    rag_agent = build_rag_graph(llm, vector_store)

# Startup
@app.on_event("startup")
async def startup_event():
    try:
        await initialize_pipelines()
        print("RAG + RAGA initialized successfully")
    except Exception as e:
        print(f"Startup failed: {e}")

    if not is_ollama_running():
        print("Ollama is NOT running. Start with `ollama serve`.")

# Reload API
@app.post("/reload")
async def reload_pipelines():
    try:
        await initialize_pipelines()
        return {"status": "RAG and RAGA pipelines reloaded"}
    except Exception as e:
        raise HTTPException(500, str(e))
    
@app.post("/rag")
async def rag_query(question: str):
    if not rag_pipeline:
        raise HTTPException(503, "RAG pipeline not initialized")

    if not is_ollama_running():
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running"
        )

    return await rag_pipeline.ask(question)

# RAGA Query API
@app.post("/raga")
async def raga_query(query: str):
    if not rag_agent:
        raise HTTPException(503, "RAGA pipeline not initialized")

    if not is_ollama_running():
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it using `ollama serve`."
        )

    state: AgentState = {
        "query": query,
        "refined_query": "",
        "documents": [],
        "answer": "",
        "grounded": False,
        "retry_count": 0,
        "max_retries": 2,
        "steps": [],
        "confidence": 0.0
    }

    result = rag_agent.invoke(state)

    return {
        "query": query,
        "answer": result["answer"],
        "grounded": result["grounded"],
        "retries_used": result["retry_count"],
        "agent_steps": result["steps"]
    }

# Health APIs
@app.get("/health")
def health():
    return {
        "status": "ok",
        "raga_ready": rag_agent is not None,
        "vector_store_loaded": vector_store.db is not None,
        "ollama_running": is_ollama_running()
    }

@app.get("/ollama/health")
def ollama_health():
    return {
        "ollama_running": is_ollama_running(),
        "ollama_url": settings.OLLAMA_BASE_URL
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
