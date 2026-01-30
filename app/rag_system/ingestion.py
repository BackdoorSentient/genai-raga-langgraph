from pathlib import Path
from typing import List


def load_documents(data_dir: str) -> List[str]:
    documents = []

    for file in Path(data_dir).glob("*.txt"):
        documents.append(file.read_text(encoding="utf-8"))

    return documents
