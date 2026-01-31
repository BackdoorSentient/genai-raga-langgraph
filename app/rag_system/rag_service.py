from app.rag_system.vector_store import VectorStore
from app.rag_system.ingestion import load_and_chunk_docs
from app.rag_system.rag_pipeline import RAGPipeline

vector_store = VectorStore()
rag_pipeline: RAGPipeline | None = None


async def initialize_rag(data_dir: str):
    global rag_pipeline

    documents = load_and_chunk_docs(data_dir)
    vector_store.build(documents)

    rag_pipeline = RAGPipeline(vector_store)
    print("RAG pipeline initialized")


async def get_rag_answer(question: str):
    if not rag_pipeline:
        raise RuntimeError("RAG pipeline not initialized")

    return await rag_pipeline.ask(question)
