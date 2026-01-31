# genai-raga-langgraph
GenAI RAGA system built with LangGraph, FastAPI, and vector databases—combining Retrieval-Augmented Generation with actionable, multi-step AI workflows

# GenAI RAGA System (RAG + Actions)

🚀 **GenAI RAGA System** is a production-ready AI architecture that combines  
**Retrieval-Augmented Generation (RAG)** with **action-driven, multi-step workflows** using **LangGraph**.

This project demonstrates how traditional RAG systems can be extended into **RAGA (RAG + Actions)** pipelines, enabling structured reasoning, orchestration, and future agentic behavior.

---

## 🔍 What is RAGA?

**RAGA (Retrieval-Augmented Generation + Actions)** goes beyond answering questions.

It allows AI systems to:
- Retrieve relevant knowledge (RAG)
- Reason over results
- Perform **actions** such as chaining steps, triggering tools, or executing workflows

This project uses **LangGraph** to orchestrate these steps in a clean, extensible way.

---

## ✨ Key Features

- ✅ Retrieval-Augmented Generation (RAG)
- ✅ LangGraph-based workflow orchestration
- ✅ Action-ready architecture (RAG → Action → Next Step)
- ✅ Multi-LLM support (Ollama, OpenAI, Azure)
- ✅ FastAPI backend for API access
- ✅ Modular, reusable codebase
- ✅ Designed for extension into Agentic AI

---

## 🏗️ Architecture Overview

Client / User
|
v
FastAPI API Layer
|
v
LangGraph Workflow Engine
|
v
+----------------------+

RAGA Workflow
1. RAG Node
- Retrieve docs
- Generate answer

2. Action Nodes
- Optional steps
- Post-processing
+----------------------+
 |
 v
 
### Flow Explanation
1. User sends a query via API
2. FastAPI triggers a LangGraph workflow
3. RAG node retrieves relevant context from vector DB
4. LLM generates an answer
5. (Optional) Action nodes process results further
6. Final response returned to client

---

## 📁 Project Structure

genai-raga-langgraph/
├── app/
│ ├── nodes/
│ │ └── raga_node.py # LangGraph node wrapping RAG logic
│ ├── rag_system/
│ │ ├── llm_clients.py # LLM providers (Ollama/OpenAI/Azure)
│ │ ├── vector_store.py # Vector DB integration
│ │ └── rag_service.py # Core RAG query logic
│ ├── config/
│ │ └── settings.py # Environment & LLM configs
│ └── main.py # FastAPI app + LangGraph execution
├── tests/
│ └── test_raga.py
├── requirements.txt
└── README.md


---

## ⚙️ Tech Stack

- **Python**
- **FastAPI**
- **LangGraph**
- **LangChain**
- **FAISS / Vector Databases**
- **Ollama / OpenAI / Azure OpenAI**

---

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/genai-raga-langgraph.git
cd genai-raga-langgraph

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

```
### Ollama server

Start Ollama (required)
ollama serve

Verify:

```bash
curl http://127.0.0.1:11434
```


## ▶️ Running the Application

```bash
uvicorn app.main:app --reload

Server runs at:
http://127.0.0.1:8000
```