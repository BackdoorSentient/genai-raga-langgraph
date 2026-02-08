from app.rag_system.rag_service import get_rag_answer


def test_raga_response():
    response = get_rag_answer("What is RAG?")
    assert response is not None
