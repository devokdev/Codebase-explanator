# AI-Powered Codebase Understanding with RAG

This project is a complete local Retrieval-Augmented Generation system for understanding software repositories with natural language. It ingests a GitHub repository URL or a local folder, extracts Python and JavaScript functions/classes/methods, stores them in FAISS, and answers grounded questions through a FastAPI backend and React frontend.

## Features

- Ingest from GitHub URL or local folder path
- Parse Python and JavaScript code
- Extract functions, classes, and methods with line references
- Chunk large code blocks intelligently
- Embed code using `sentence-transformers`
- Store vectors in FAISS and metadata in JSON
- Query with natural language through a FastAPI API
- Ground answers with file names, symbol names, and snippets
- Local Ollama-powered response generation with a safe fallback if the model is unavailable
- Optional evaluation script to compare retrieval vs no retrieval

## Project Structure

```text
project-root/
|
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── llm.py
│   ├── utils.py
│   └── evaluation.py
|
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       └── styles.css
|
├── data/
│   ├── faiss_index/
│   ├── metadata.json
│   └── repos/
|
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Backend

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn backend.main:app --reload
```

The API will start at `http://127.0.0.1:8000`.

Before starting the backend, make sure Ollama is running locally and the configured model is installed:

```bash
ollama pull tinyllama
ollama serve
```

### 2. Frontend

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

The frontend will start at `http://127.0.0.1:5173`.

## API Usage

### Ingest a repository

`POST /ingest`

```json
{
  "source": "https://github.com/tiangolo/fastapi"
}
```

Or:

```json
{
  "source": "C:/Users/yourname/projects/my-repo"
}
```

### Query the codebase

`POST /query`

```json
{
  "query": "Where is the authentication logic implemented?",
  "top_k": 5
}
```

Example response:

```json
{
  "answer": "Authentication appears in app/security.py within the AuthService.login method...",
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
      "code": "def login(self, username, password): ..."
    }
  ]
}
```

## Prompt Template

The backend uses this exact prompt template when calling the LLM:

```text
You are a code analysis assistant.

Context:
{retrieved_code_chunks}

User Query:
{query}

Instructions:

* Explain clearly
* Mention file names
* Mention function/class names
* If unsure, say 'Not found in codebase'

Answer:
```

## Evaluation
 
 You can compare retrieval-backed answers against empty-context answers:
 
 ```bash
 python -m backend.evaluation --source "https://github.com/tiangolo/fastapi" --query "How are routes registered?"
 ```

## Fine-Tuning Dataset

To optimize localized diagnostic generation, the platform uses a dedicated training dataset.

- **Source:** Extracted dynamically from the academic `code-rag-bench/stackoverflow-posts` benchmark mappings.
- **Volume:** 2,000 independent software query parameters.
- **Role:** Instructs general foundational models (`tinyllama`) to bypass generic summaries in favor of full procedural code walkthroughs.

### Data Format Snapshot
```json
{
  "prompt": "You are a code analysis assistant.\n\nContext:\n\n\nUser Query:\nHow to convert Decimal to Double in C#?...",
  "completion": "Opacity requires a double, not a decimal value...",
  "meta": {
    "source": "CodeRAG-StackOverflow"
  }
}
```

## Notes

- Supported languages today: Python and JavaScript
- Vector store is written to `data/faiss_index/code.index`
- Metadata is written to `data/metadata.json`
- GitHub repository cloning requires `git` to be installed
- If Ollama is not running or the configured model is unavailable, the app still runs with a grounded fallback answer generator
