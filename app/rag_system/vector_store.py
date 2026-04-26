# app/rag_system/vector_store.py
import os
from typing import List, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config.settings import settings
from app.schema import Document


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
            # Fix 6: only allow pickle deserialization in dev
            # In prod, this is a security risk — tampered index files
            # can execute arbitrary code via pickle
            allow_pickle = settings.ENV != "prod"

            if not allow_pickle:
                raise RuntimeError(
                    "FAISS index found but pickle deserialization is disabled in prod. "
                    "Rebuild the index from source documents or set ENV=dev."
                )

            self.db = FAISS.load_local(
                settings.VECTOR_DB_PATH,
                self.embeddings,
                allow_dangerous_deserialization=allow_pickle,  # ← Fix 6
            )
            print(f"Loaded existing FAISS index (ENV={settings.ENV}).")
        else:
            self.db = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
            )
            self.db.save_local(settings.VECTOR_DB_PATH)
            print("Built new FAISS index and saved locally.")

    def search(self, query: str, k: int | None = None) -> List[Document]:
        if not self.db:
            raise RuntimeError("Vector store not initialized. Call build_or_load() first.")

        k = k or settings.RETRIEVAL_K              # ← Fix 4: was hardcoded 4

        results_with_scores = self.db.similarity_search_with_score(query, k=k)

        return [
            Document(
                content=r.page_content,
                source=r.metadata.get("source", "unknown"),
                origin="vector",
                page=r.metadata.get("page"),
            )
            for r, score in results_with_scores
            if score < 1.2
        ]