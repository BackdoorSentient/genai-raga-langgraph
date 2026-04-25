# main.py
import uvicorn
from fastapi import FastAPI

from app.config.settings import get_settings
from app.utils.ollama import is_ollama_running
from app.state import app_state, initialize_pipelines
from app.api.routes import router

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="GenAI RAGA System — RAG, RAGA, and Agentic RAGA pipelines",
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    try:
        await initialize_pipelines(app_state)
        print("✅ All pipelines initialized successfully")
    except Exception as e:
        print(f"❌ Startup failed: {e}")

    if not is_ollama_running():
        print("⚠️  Ollama is NOT running. Start with `ollama serve`.")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )