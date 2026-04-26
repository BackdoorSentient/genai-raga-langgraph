# app/utils/ollama.py
import httpx                                   

from app.config.settings import get_settings

settings = get_settings()


def is_ollama_running() -> bool:
    try:
        with httpx.Client(timeout=2, trust_env=False) as client:
            res = client.get(settings.OLLAMA_BASE_URL)  
        return res.status_code == 200
    except Exception:
        return False