# app/rag_system/ingestion.py
from pathlib import Path
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.config.settings import settings          # ← Fix 4: reads from settings


def load_and_chunk_docs(data_dir: str) -> List[Dict]:
    documents = []
    pdf_files = list(Path(data_dir).glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDFs found in {data_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,           # ← Fix 4: was hardcoded 500
        chunk_overlap=settings.CHUNK_OVERLAP      # ← Fix 4: was hardcoded 100
    )

    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            documents.append({
                "text": chunk.page_content,
                "metadata": {
                    "source": pdf.name,
                    "page": chunk.metadata.get("page", None)
                }
            })

    if not documents:
        raise ValueError(f"No PDFs found or PDFs are empty in {data_dir}")

    return documents