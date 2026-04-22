from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from .embeddings import EmbeddingService
from .ingestion import IngestionService
from .llm import LLMService
from .retrieval import VectorStore
from .utils import ensure_directory


def run_evaluation(source: str, query: str) -> None:
    load_dotenv()
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = ensure_directory(base_dir / "data")
    repos_dir = ensure_directory(data_dir / "repos")
    faiss_dir = ensure_directory(data_dir / "faiss_index")
    repo_context_path = data_dir / "repo_context.json"

    embedding_service = EmbeddingService()
    vector_store = VectorStore(faiss_dir / "code.index", data_dir / "metadata.json")
    llm_service = LLMService()
    ingestion_service = IngestionService(
        repos_root=repos_dir,
        index_path=faiss_dir / "code.index",
        metadata_path=data_dir / "metadata.json",
        repo_context_path=repo_context_path,
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
    )

    repo_context = ingestion_service.ingest(source)
    results = vector_store.search(embedding_service.embed_query(query), top_k=5)
    retrieved = [item[1] for item in results]

    print("=== With Retrieval ===")
    print(llm_service.answer_query(query, retrieved, repo_context))
    print()
    print("=== Without Retrieval ===")
    print(llm_service.answer_query(query, [], repo_context))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG vs no retrieval.")
    parser.add_argument("--source", required=True, help="GitHub URL or local path")
    parser.add_argument("--query", required=True, help="Natural language question")
    args = parser.parse_args()
    run_evaluation(args.source, args.query)
