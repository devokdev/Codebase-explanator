# Codebase Intelligence Engine (AI-Powered Code Understanding & RAG)

[![Tech Stack](https://img.shields.io/badge/Tech-Python%20%7C%20FastAPI%20%7C%20PostgreSQL%20%7C%20FAISS%20%7C%20Docker%20%7C%20React-blue.svg)](#tech-stack)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

An AI-powered repository intelligence platform utilizing semantic code retrieval, FastAPI microservices, and a decoupled PostgreSQL + FAISS metadata architecture.

---

## ⚡ Highlights & Key Metrics

* **Tech Stack:** Python 3.11, FastAPI, PostgreSQL (SQLAlchemy), FAISS, Sentence-Transformers, Docker & Docker Compose, React.js (Vite).
* **Decoupled Hybrid Storage:** Employs in-memory FAISS `IndexFlatIP` vector index for sub-millisecond dense cosine similarity search paired alongside PostgreSQL relational tables for code chunk hydration, AST structure, and query audit logs.
* **Fine-Tuning & Local LLM Support:** Integrated with a locally quantized / PEFT LoRA model pipeline with seamless Ollama (`tinyllama`) runtime fallback.
* **Full Multi-Container Orchestration:** Ready-to-deploy multi-service setup encompassing the Frontend, Backend, PostgreSQL database, and Ollama inference engine.

---

## 🏗️ Architecture Overview

```text
                                [ User / React Frontend ]
                                            │
                                            ▼
                                [ FastAPI Microservice ]
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
          ┌─────────────────────┐                       ┌─────────────────────┐
          │  FAISS Vector Store │                       │ PostgreSQL Database │
          │ (Dense Similarity)  │                       │ (Relational Chunks) │
          └──────────┬──────────┘                       └──────────┬──────────┘
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            ▼
                               [ Grounded Prompt Engine ]
                                            │
                                            ▼
                            [ Local LLM / Ollama Runtime ]
```

---

## 📂 Project Structure

```text
Codebase-explanator/
├── backend/
│   ├── __init__.py           # Package setup & environment isolation guards
│   ├── main.py               # FastAPI application & API routing
│   ├── database.py           # PostgreSQL models (CodeChunkModel, QueryLogModel)
│   ├── ingestion.py          # Git cloning, AST parsing, chunking, indexing pipeline
│   ├── chunking.py           # Intelligent code chunking (Python & JS)
│   ├── embeddings.py         # SentenceTransformer caching & vectorization
│   ├── retrieval.py          # FAISS vector store & PostgreSQL sync/load
│   ├── llm.py                # Fine-tuned model inference & Ollama fallback
│   ├── progress.py           # Ingestion job status store
│   ├── utils.py              # File system, AST, & JSON helpers
│   └── evaluation.py         # Retrieval vs No-Retrieval benchmarking
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # React UI with Ingestion & Chat workflow
│   │   ├── main.jsx          # React DOM root
│   │   └── styles.css        # Responsive styling & status indicators
│   ├── package.json          # Frontend dependencies
│   └── vite.config.js        # Vite build & host configuration
│
├── data/
│   ├── faiss_index/          # Persisted FAISS vector index
│   └── repos/                # Ingested repository cache
│
├── Dockerfile.backend        # FastAPI containerization
├── Dockerfile.frontend       # Multi-stage Nginx React containerization
├── docker-compose.yml        # PostgreSQL, Ollama, Backend, & Frontend orchestration
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🚀 Quickstart

### Method 1: Docker (Recommended)

Run the entire platform (PostgreSQL, Ollama, Backend, and Frontend) with one command:

```bash
# 1. Start all containers
docker compose up --build -d

# 2. Pull the default model inside the Ollama container (first time only)
docker compose exec ollama ollama pull tinyllama
```

* **Frontend Web App:** [http://localhost:5173](http://localhost:5173)
* **FastAPI Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **PostgreSQL:** `localhost:5432` (`user: postgres`, `password: postgrespassword`, `db: codebase_rag`)

---

### Method 2: Local Python & Node Environment

#### 1. Start Ollama
```bash
ollama pull tinyllama
ollama serve
```

#### 2. Start Backend
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows (or 'source .venv/bin/activate' on Unix)

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

#### 3. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🔌 API Endpoints

### 1. Ingest a Codebase
`POST /ingest` or `POST /ingest/start`
```json
{
  "source": "https://github.com/tiangolo/fastapi"
}
```
*Supports both GitHub URLs and local absolute directory paths.*

### 2. Query Repository
`POST /query`
```json
{
  "query": "Where is the authentication logic implemented?",
  "top_k": 5
}
```

**Example Response:**
```json
{
  "answer": "Authentication is implemented in app/security.py within the AuthService.login method...",
  "relevant_files": [
    "app/security.py",
    "app/routes/auth.py"
  ],
  "snippets": [
    {
      "file_path": "app/security.py",
      "name": "AuthService.login",
      "type": "method",
      "line_start": 18,
      "line_end": 42,
      "code": "def login(self, username, password): ...",
      "score": 3.42
    }
  ],
  "repo_summary": "High-level summary of the codebase architecture..."
}
```

---

## 🧠 Fine-Tuning Dataset & Benchmarking

To optimize procedural diagnostic explanations, the platform utilizes specialized fine-tuning pairs:
* **Dataset source:** Derived from academic `code-rag-bench/stackoverflow-posts` benchmark mappings.
* **Volume:** 2,000 paired software queries and step-by-step diagnostic walkthroughs.
* **Evaluation Benchmark:** Compare retrieval-backed answers against non-retrieval baseline:
  ```bash
  python -m backend.evaluation --source "https://github.com/tiangolo/fastapi" --query "How are routes registered?"
  ```

---

## 📄 License
This project is open-source under the MIT License.
