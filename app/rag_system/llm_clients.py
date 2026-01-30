from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List


class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.db = None

    def build(self, documents: List[str]):
        self.db = FAISS.from_texts(documents, self.embeddings)

    def similarity_search(self, query: str, k: int = 4):
        if not self.db:
            raise RuntimeError("Vector store not initialized")
        return self.db.similarity_search(query, k=k)
