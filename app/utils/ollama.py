import requests

OLLAMA_URL = "http://127.0.0.1:11434"

def is_ollama_running() -> bool:
    try:
        res = requests.get(OLLAMA_URL, timeout=2)
        return res.status_code == 200
    except Exception:
        return False
