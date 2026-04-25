# app/api/routes.py
import time
from fastapi import APIRouter, HTTPException

from app.utils.ollama import is_ollama_running
from app.config.settings import get_settings
from app.raga.state import RAGAState
from app.agent.state import AgentState

settings = get_settings()
router = APIRouter()


# ── RAG ───────────────────────────────────────────────────────────────────────

@router.post("/rag", tags=["RAG"])
def rag_query(question: str):
    from app.state import app_state

    if not app_state["rag_pipeline"]:
        raise HTTPException(503, "RAG pipeline not initialized")
    if not is_ollama_running():
        raise HTTPException(503, "Ollama is not running")

    return app_state["rag_pipeline"].ask(question)


# ── RAGA ──────────────────────────────────────────────────────────────────────

@router.post("/raga", tags=["RAGA"])
async def raga_query(query: str):
    from app.state import app_state

    if not app_state["raga_agent"]:
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
    result = app_state["raga_agent"].invoke(
        state, config={"recursion_limit": 20}
    )
    result["total_latency_ms"] = round((time.time() - t0) * 1000, 2)

    return result


# ── Agentic RAGA ──────────────────────────────────────────────────────────────

@router.post("/agentic-raga", tags=["Agentic RAGA"])
async def agentic_raga_query(query: str):
    from app.state import app_state

    if not app_state["agentic_raga_agent"]:
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

    result = app_state["agentic_raga_agent"].invoke(state)

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


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", tags=["Health"])
def health():
    from app.state import app_state

    return {
        "status": "ok",
        "rag_ready": app_state["rag_agent"] is not None,
        "vector_store_loaded": app_state["vector_store"] is not None
        and app_state["vector_store"].db is not None,
        "ollama_running": is_ollama_running(),
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.OLLAMA_MODEL,
    }


@router.get("/ollama/health", tags=["Health"])
def ollama_health():
    return {
        "ollama_running": is_ollama_running(),
        "ollama_url": settings.OLLAMA_BASE_URL,
    }


@router.post("/reload", tags=["Health"])
async def reload_pipelines():
    from app.state import app_state, initialize_pipelines
    try:
        await initialize_pipelines(app_state)
        return {"status": "All pipelines reloaded successfully"}
    except Exception as e:
        raise HTTPException(500, str(e))