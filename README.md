# genai-raga-langgraph

GenAI RAGA system built with LangGraph, FastAPI, and vector databases—combining Retrieval‑Augmented Generation with actionable, multi‑step AI workflows

**Author:** Aniket Waichal

---

# GenAI RAGA System (RAG + Actions)

🚀 **GenAI RAGA System** is a production‑ready AI architecture that demonstrates the **evolution of modern GenAI systems** — from **classic RAG**, to **RAGA (RAG + Actions)**, and into **early Agentic AI workflows**.

Before diving into RAGA and Agentic concepts, it is important to understand the foundation: **Retrieval‑Augmented Generation (RAG)**.

---

## 🧩 What is RAG? (Foundation)

**Retrieval‑Augmented Generation (RAG)** is a technique that enhances Large Language Models by grounding responses in external knowledge sources.

In a standard RAG system:

* User queries are embedded and searched against a **vector database**
* Relevant documents are retrieved
* Retrieved context is injected into the LLM prompt
* The LLM generates a grounded response

**Why RAG matters:**

* Reduces hallucinations
* Enables domain‑specific knowledge
* Keeps models stateless while data stays external

This project includes a **pure RAG pipeline** as a baseline, which then evolves into RAGA and Agentic systems.

---

## 🔍 What is RAGA?

**RAGA (Retrieval‑Augmented Generation + Actions)** goes beyond answering questions.

It allows AI systems to:

* Retrieve relevant knowledge from vector databases (RAG)
* Reason over retrieved results using LLMs
* Perform **actions** such as routing, validation, summarization, tool usage, and multi‑step orchestration

This project uses **LangGraph** to orchestrate these steps as an explicit graph of nodes, making the system interpretable, debuggable, and extensible.

---

## ✨ Key Features

* ✅ Agentic AI foundations (planner, decision routing, stateful graphs)

* ✅ Retrieval‑Augmented Generation (RAG)

* ✅ LangGraph‑based workflow orchestration

* ✅ RAGA architecture (RAG → Reason → Action → Validate)

* ✅ Action nodes (decision, tool, summarization, validation)

* ✅ Multi‑LLM support (Ollama‑first, extensible to OpenAI/Azure)

* ✅ FastAPI backend for API access

* ✅ Modular, production‑ready code structure

* ✅ Designed for Agentic AI and workflow‑driven systems

---

## 🏗️ Architecture Overview

This project intentionally exposes **RAG, RAGA, and Agentic systems as separate API flows** via FastAPI. This separation highlights architectural evolution and keeps each system independently testable and extensible.

**High‑level Flow**

Client / User
↓
FastAPI API Layer
↓
LangGraph Workflow Engine
↓
RAGA Workflow Nodes

---

### 🧠 RAGA & Agentic Workflow (LangGraph)

This system is **not a single-pass RAG pipeline**. It is an **early Agentic AI architecture** where decisions, actions, and state transitions are explicitly modeled.

1. **Planner / Decision Node**

   * Interprets user intent
   * Decides next steps in the workflow

2. **RAG Node**

   * Retrieves relevant documents from vector store
   * Generates grounded answers using LLM + context

3. **Action Nodes**

   * Tool execution
   * Post‑processing
   * Query refinement
   * Summarization

4. **Validation Node**

   * Checks grounding against retrieved context
   * Calculates confidence / overlap metrics

5. **Agentic Control Loop**

   * Shared `AgentState` passed across nodes
   * Planner & decision nodes choose next actions dynamically
   * Enables future multi-agent or looping behaviors

6. **Final Response Node**

   * Returns structured output to API client

---

### 🔁 Flow Explanation

The FastAPI layer exposes **three distinct systems**:

1. **RAG API** – Classic retrieval + generation

2. **RAGA API** – RAG extended with actions

3. **Agentic API** – Decision‑driven, stateful workflows

4. User sends a query via REST API

5. FastAPI triggers the LangGraph execution

6. Planner node decides the workflow path

7. RAG node retrieves documents and generates an answer

8. Optional action nodes refine, validate, or summarize

9. Final structured response is returned to the client

---

## 📁 Project Structure

The structure below reflects the **actual repository layout**, exactly as implemented, showing clear separation between **RAG**, **RAGA**, and **Agentic workflows**:

```
genai-raga-langgraph/
├── app/
│   ├── actions/
│   │   ├── base_action.py        # Base class for actions
│   │   └── example_action.py     # Sample action implementation
│   │
│   ├── agent/
│   │   ├── agentic_raga_graph.py # Agentic RAGA LangGraph
│   │   ├── critic.py             # Critic / evaluation logic
│   │   ├── document.py           # Agent document abstraction
│   │   ├── executor.py           # Agent execution controller
│   │   ├── graph.py              # Core LangGraph wiring
│   │   ├── nodes.py              # Shared agent nodes
│   │   ├── planner.py            # Planner / decision logic
│   │   └── state.py              # Agent state definition
│   │
│   ├── api/
│   │   └── routes.py             # FastAPI route definitions
│   │
│   ├── config/
│   │   └── settings.py           # Application & LLM settings
│   │
│   ├── llm/
│   │   ├── base.py               # Base LLM interface
│   │   ├── factory.py            # LLM factory
│   │   └── ollama_client.py      # Ollama client
│   │
│   ├── nodes/
│   │   ├── action_node.py        # Action execution node
│   │   ├── decision_node.py      # Decision / routing node
│   │   ├── raga_node.py          # RAGA node
│   │   ├── summarize_node.py     # Summarization node
│   │   └── tool_node.py          # Tool execution node
│   │
│   ├── rag_system/
│   │   ├── ingestion.py          # Document ingestion
│   │   ├── llm_clients.py        # RAG-specific LLM clients
│   │   ├── load_processed.py     # Load processed vectors
│   │   ├── rag_pipeline.py       # Classic RAG pipeline
│   │   ├── rag_service.py        # RAG service layer
│   │   └── vector_store.py       # Vector database logic
│   │
│   ├── utils/
│   │   └── ollama.py             # Ollama health checks
│   │
│   └── workflows/
│       ├── raga_graph.py         # RAGA workflow graph
│       └── state_schema.py       # Workflow state schema
│
├── data/                          # Documents / datasets
├── scripts/                       # Utility scripts
├── tests/
│   └── test_raga.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── main.py                        # FastAPI entry point
├── LICENSE
└── README.md
```

---

## ⚙️ Tech Stack

* **Python**
* **FastAPI**
* **LangGraph**
* **LangChain**
* **FAISS / Vector Databases**
* **Ollama (local LLMs)**
* *(Extensible to OpenAI / Azure OpenAI)*

---

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/genai-raga-langgraph.git
cd genai-raga-langgraph
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🦙 Ollama Server (Required)

Start Ollama:

```bash
ollama serve
```

Verify:

```bash
curl http://127.0.0.1:11434
```

Ensure the required model is pulled:

```bash
#RAG and RAGA
ollama pull llama3
# Planner
ollama pull qwen2.5:7b-instruct
# Executor / Answer generation
ollama pull qwen2.5:14b-instruct
# Validator / reasoning
ollama pull qwen2.5:3b-instruct

```

---

## ▶️ Running the Application

```bash
uvicorn app.main:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

---

## 🎯 Project Goal

This project intentionally goes beyond classic RAG and introduces **Agentic AI concepts** such as:

* Explicit planning and decision nodes
* Stateful graph-based execution
* Action selection instead of fixed pipelines

It demonstrates how RAG systems evolve into **agentic, reasoning-driven architectures**.

This project is designed to:

* Demonstrate **RAGA (RAG + Actions)** in practice
* Showcase **LangGraph‑based orchestration**
* Serve as a strong **portfolio‑grade GenAI system**
* Act as a foundation for **Agentic AI workflows**

---

## 🔮 Future Enhancements

* Fully autonomous agent loops

* Multi-agent coordination

* Tool planning & execution graphs

* Long-term memory & reflection

* Hybrid search (BM25 + Vector)

* Production observability & tracing

* Multi‑agent collaboration

* Tool‑calling via function schemas

* Memory & long‑term state

* Hybrid search (BM25 + Vector)

* Production observability & tracing

---

⭐ If you find this project useful, consider starring the repository.
