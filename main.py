# app/main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from app.utils.ollama import is_ollama_running
from app.rag_system.ingestion import load_and_chunk_docs
from app.rag_system.vector_store import VectorStore
from app.rag_system.rag_pipeline import RAGPipeline
from app.config.settings import get_settings

settings = get_settings()

app = FastAPI(title=settings.APP_NAME)

vector_store = VectorStore()
rag_pipeline: RAGPipeline | None = None


async def initialize_rag_pipeline():
    """
    Helper to load documents, build vector store, and initialize RAG pipeline.
    """
    global rag_pipeline

    # Load PDFs and chunk
    documents = load_and_chunk_docs(settings.RAW_DATA_DIR)

    # Build or load FAISS index
    vector_store.build_or_load(documents)

    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(vector_store)


@app.on_event("startup")
async def startup_event():
    """
    Startup event: load documents, build vector store, and init RAG pipeline.
    """
    try:
        await initialize_rag_pipeline()
        print("RAG pipeline initialized successfully")
    except Exception as e:
        print(f"Startup failed: {e}")

    if not is_ollama_running():
        print("WARNING: Ollama is NOT running. Start with `ollama serve`.")


@app.post("/reload")
async def reload_rag_pipeline():
    """
    Manually reload documents, rebuild vector store, and reinitialize RAG pipeline.
    """
    try:
        await initialize_rag_pipeline()
        return {"status": "RAG pipeline reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload: {e}")


@app.get("/ollama/health")
def ollama_health():
    return {
        "ollama_running": is_ollama_running(),
        "ollama_url": settings.OLLAMA_BASE_URL
    }


@app.get("/ask")
async def ask(question: str):
    """
    Ask a question to the RAG pipeline
    """
    if not rag_pipeline:
        raise HTTPException(503, "RAG pipeline not initialized")

    if not is_ollama_running():
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it using `ollama serve`."
        )

    return await rag_pipeline.ask(question)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_ready": rag_pipeline is not None,
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
