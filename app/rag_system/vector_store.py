import os
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.config.settings import settings
from app.agent.document import Document

class VectorStore:
    def __init__(self):
        # initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
        self.db: FAISS | None = None

    def build_or_load(self, documents: List[Dict]):
        """
        Build or load vector store using FAISS.
        """
        texts = [d["text"] for d in documents if d.get("text")]
        metadatas = [d.get("metadata", {}) for d in documents if d.get("text")]

        if not texts:
            raise ValueError(
                "No documents found to build FAISS index. "
                "Ensure documents contain non-empty 'text'."
            )

        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)

        # Check if FAISS index exists
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

    # def search(self, query: str, k: int = 4):
    #     if not self.db:
    #         raise RuntimeError("Vector store not initialized. Call build_or_load() first.")
    #     return self.db.similarity_search(query, k=k)
    def search(self, query: str, k: int = 4):
        if not self.db:
            raise RuntimeError("Vector store not initialized. Call build_or_load() first.")

        results = self.db.similarity_search(query, k=k)
        # Wrap results into our custom Document
        return [Document(page_content=r.page_content, metadata=r.metadata) for r in results]
