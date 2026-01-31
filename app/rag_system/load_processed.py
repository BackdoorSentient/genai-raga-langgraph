import json
from pathlib import Path
from typing import List, Dict


def load_processed_docs(processed_dir: str) -> List[Dict]:
    documents = []

    for file in Path(processed_dir).glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            documents.extend(json.load(f))

    if not documents:
        raise ValueError("No processed documents found")

    return documents
