# main.py
import uvicorn
import time
from typing import Any

from fastapi import FastAPI, HTTPException

from app.utils.ollama import is_ollama_running
from app.rag_system.vector_store import VectorStore
from app.rag_system.ingestion import load_and_chunk_docs
from app.config.settings import get_settings
from app.llm.factory import get_llm                        # ← single LLM entry point

from app.rag_system.rag_pipeline import RAGPipeline
from app.agent.graph import build_rag_graph
from app.raga.graph import build_raga_graph
from app.raga.state import RAGAState
from app.agent.agentic_raga_graph import build_agentic_raga_graph
from app.agent.state import AgentState

settings = get_settings()
app = FastAPI(title=settings.APP_NAME)

vector_store = VectorStore()

rag_pipeline: RAGPipeline | None = None
rag_agent: Any = None
raga_agent: Any = None
agentic_raga_agent: Any = None


async def initialize_pipelines():
    global rag_pipeline, rag_agent, raga_agent, agentic_raga_agent

    documents = load_and_chunk_docs(settings.RAW_DATA_DIR)
    vector_store.build_or_load(documents)

    llm = get_llm()                                        # ← one call, reads .env

    rag_pipeline = RAGPipeline(vector_store)
    rag_agent = build_rag_graph(llm, vector_store)
    raga_agent = build_raga_graph(llm, vector_store)
    agentic_raga_agent = build_agentic_raga_graph(vector_store)


@app.on_event("startup")
async def startup_event():
    try:
        await initialize_pipelines()
        print("RAG + RAGA initialized successfully")
    except Exception as e:
        print(f"Startup failed: {e}")

    if not is_ollama_running():
        print("Ollama is NOT running. Start with `ollama serve`.")


@app.post("/reload")
async def reload_pipelines():
    try:
        await initialize_pipelines()
        return {"status": "RAG and RAGA pipelines reloaded"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/rag")
def rag_query(question: str):
    if not rag_pipeline:
        raise HTTPException(503, "RAG pipeline not initialized")
    if not is_ollama_running():
        raise HTTPException(503, "Ollama is not running")
    return rag_pipeline.ask(question)


@app.post("/raga")
async def raga_query(query: str):
    if not raga_agent:
        raise HTTPException(503, "RAGA not initialized")

    state: RAGAState = {
        "query": query,
        "documents": [],
        "citations": [],
        "steps": [],
        "timeline": [],
        "retry_count": 0,
        "max_retries": 2,
        "start_time": time.time(),
        "timeout_seconds": 20,
        "terminate": False,
    }

    t0 = time.time()
    result = raga_agent.invoke(state, config={"recursion_limit": 20})
    result["total_latency_ms"] = round((time.time() - t0) * 1000, 2)

    return result


@app.post("/agentic-raga")
async def agentic_raga_query(query: str):
    if not agentic_raga_agent:
        raise HTTPException(503, "Agentic RAGA not initialized")
    if not is_ollama_running():
        raise HTTPException(503, "Ollama not running")

    state: AgentState = {
        "query": query,
        "goal": "Answer the query using grounded documents",
        "plan": [],
        "current_step": 0,
        "observations": [],
        "critic_decision": "",
        "documents": [],
        "answer": "",
        "retry_count": 0,
        "max_retries": 3,
        "confidence": 0.0,
        "citations": [],
        "steps": [],
        "used_vector": False,
        "used_web": False,
        "phase": "retrieve",
        "grounded": False,
    }

    result = agentic_raga_agent.invoke(state)

    return {
        "query": query,
        "answer": result["answer"],
        "grounded": result.get("grounded", False),
        "confidence": result.get("confidence", 0.0),
        "citations": list(set(result.get("citations", []))),
        "sources_used": list(
            {d.origin for d in result.get("documents", [])}
        ),
        "steps": result.get("steps", []),
        "critic_decision": result.get("critic_decision"),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "raga_ready": rag_agent is not None,
        "vector_store_loaded": vector_store.db is not None,
        "ollama_running": is_ollama_running(),
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.OLLAMA_MODEL,
    }


@app.get("/ollama/health")
def ollama_health():
    return {
        "ollama_running": is_ollama_running(),
        "ollama_url": settings.OLLAMA_BASE_URL,
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )