from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def load_and_chunk_docs(data_dir: str) -> List[Dict]:
    documents = []
    pdf_files = Path(data_dir).glob("*.pdf")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
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
