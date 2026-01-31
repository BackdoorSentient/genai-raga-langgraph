from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def load_and_chunk_docs(data_dir: str) -> List[Dict]:
    documents = []
    pdf_files = list(Path(data_dir).glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {data_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()  # returns List[Document]
        chunks = splitter.split_documents(pages)  # List[Document]

        for chunk in chunks:
            documents.append({
                "text": chunk.page_content,
                "metadata": chunk.metadata
            })

    return documents
