# app/rag_system/vector_store.py
import os
from typing import List, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config.settings import settings
from app.schema import Document          # ← was: from app.agent.document import AgentDocument


class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        self.db: FAISS | None = None

    def build_or_load(self, documents: List[Dict]):
        texts = [d["text"] for d in documents if d.get("text")]
        metadatas = [d.get("metadata", {}) for d in documents if d.get("text")]

        if not texts:
            raise ValueError(
                "No documents found to build FAISS index. "
                "Ensure documents contain non-empty 'text'."
            )

        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)

        if os.path.exists(os.path.join(settings.VECTOR_DB_PATH, "index.faiss")):
            self.db = FAISS.load_local(
                settings.VECTOR_DB_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("Loaded existing FAISS index.")
        else:
            self.db = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
            )
            self.db.save_local(settings.VECTOR_DB_PATH)
            print("Built new FAISS index and saved locally.")

    def search(self, query: str, k: int = 4) -> List[Document]:
        if not self.db:
            raise RuntimeError("Vector store not initialized. Call build_or_load() first.")

        results = self.db.similarity_search(query, k=k)

        return [
            Document(                                   # ← was: AgentDocument(page_content=..., metadata=..., source=...)
                content=r.page_content,                # ← unified field name
                source=r.metadata.get("source", "unknown"),
                origin="vector",
                page=r.metadata.get("page"),
            )
            for r in results
        ]