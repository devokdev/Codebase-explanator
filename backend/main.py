from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .embeddings import EmbeddingService
from .ingestion import IngestionService
from .llm import LLMService
from .progress import JobStore
from .retrieval import VectorStore
from .utils import ensure_directory, logger, read_json


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ensure_directory(BASE_DIR / "data")
REPOS_DIR = ensure_directory(DATA_DIR / "repos")
FAISS_DIR = ensure_directory(DATA_DIR / "faiss_index")
JOBS_DIR = ensure_directory(DATA_DIR / "jobs")
INDEX_PATH = FAISS_DIR / "code.index"
METADATA_PATH = DATA_DIR / "metadata.json"
REPO_CONTEXT_PATH = DATA_DIR / "repo_context.json"

embedding_service = EmbeddingService()
vector_store = VectorStore(INDEX_PATH, METADATA_PATH)
llm_service = LLMService()
job_store = JobStore(JOBS_DIR)
ingestion_service = IngestionService(
    REPOS_DIR,
    INDEX_PATH,
    METADATA_PATH,
    REPO_CONTEXT_PATH,
    embedding_service,
    vector_store,
    llm_service,
)

app = FastAPI(title="Codebase Understanding RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    source: str = Field(..., description="GitHub repository URL or local folder path")


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    relevant_files: List[str]
    snippets: List[Dict]
    repo_summary: str | None = None


class IngestStartResponse(BaseModel):
    job_id: str


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest_codebase(payload: IngestRequest) -> Dict:
    try:
        return ingestion_service.ingest(payload.source)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest/start", response_model=IngestStartResponse)
def start_ingestion(payload: IngestRequest) -> IngestStartResponse:
    job_id = job_store.create("ingest")
    job_store.update(job_id, status="running", message="Starting ingestion...", progress=1)

    def run_job() -> None:
        try:
            result = ingestion_service.ingest(payload.source, progress_callback=lambda **fields: job_store.update(job_id, status="running", **fields))
            job_store.finish(job_id, result)
        except Exception as exc:
            logger.exception("Ingestion failed")
            job_store.fail(job_id, str(exc))

    threading.Thread(target=run_job, daemon=True).start()
    return IngestStartResponse(job_id=job_id)


@app.get("/ingest/status/{job_id}")
def get_ingest_status(job_id: str) -> Dict:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    return job


@app.post("/query", response_model=QueryResponse)
def query_codebase(payload: QueryRequest) -> QueryResponse:
    try:
        if not vector_store.is_ready():
            raise FileNotFoundError("Index not found. Please ingest a repository first.")

        query_embedding = embedding_service.embed_query(payload.query)
        search_results = vector_store.hybrid_search(payload.query, query_embedding, top_k=payload.top_k)
        chunks = [result[1] for result in search_results]
        repo_context = read_json(REPO_CONTEXT_PATH, default={})
        answer = llm_service.answer_query(payload.query, chunks, repo_context)
        relevant_files = sorted({chunk["file_path"] for chunk in chunks})
        snippets = [
            {
                "file_path": chunk["file_path"],
                "name": chunk["name"],
                "type": chunk["type"],
                "line_start": chunk.get("line_start"),
                "line_end": chunk.get("line_end"),
                "code": chunk["code"],
                "score": round(score, 4),
            }
            for score, chunk in search_results
        ]
        return QueryResponse(
            answer=answer,
            relevant_files=relevant_files,
            snippets=snippets,
            repo_summary=repo_context.get("repo_summary"),
        )
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
