from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

from .chunking import extract_code_chunks
from .embeddings import EmbeddingService
from .llm import LLMService
from .retrieval import VectorStore
from .utils import (
    detect_language,
    iter_code_files,
    logger,
    read_text_file,
    readme_excerpt,
    relative_to_root,
    resolve_source_path,
    write_json,
)


class IngestionService:
    def __init__(
        self,
        repos_root: Path,
        index_path: Path,
        metadata_path: Path,
        repo_context_path: Path,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_service: LLMService,
    ) -> None:
        self.repos_root = repos_root
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.repo_context_path = repo_context_path
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service

    def ingest(self, source: str, progress_callback: Callable[..., None] | None = None) -> Dict:
        def report(message: str, progress: int) -> None:
            if progress_callback is not None:
                progress_callback(message=message, progress=progress)

        report("Resolving repository source...", 5)
        source_root = resolve_source_path(source, self.repos_root)
        logger.info("Starting ingestion for %s", source_root)

        all_chunks: List[Dict] = []
        indexed_files = 0
        file_records: List[Dict] = []
        code_files = list(iter_code_files(source_root))

        if not code_files:
            raise ValueError("No supported code files were found in the repository.")

        report(f"Found {len(code_files)} code files. Extracting symbols...", 15)

        for index, file_path in enumerate(code_files, start=1):
            language = detect_language(file_path)
            if not language:
                continue

            try:
                code = read_text_file(file_path)
                relative_path = relative_to_root(file_path, source_root)
                chunks = extract_code_chunks(code, relative_path, language)
                all_chunks.extend(chunks)
                file_records.append(
                    {
                        "file_path": relative_path,
                        "language": language,
                        "code": code,
                        "symbols": [chunk["name"] for chunk in chunks[:12]],
                    }
                )
                indexed_files += 1
                progress = 15 + int((index / max(len(code_files), 1)) * 35)
                report(f"Parsed {index}/{len(code_files)} files and extracted code chunks...", progress)
            except Exception as exc:
                logger.warning("Failed to process %s: %s", file_path, exc)

        if not all_chunks:
            raise ValueError("No supported Python or JavaScript functions/classes were extracted.")

        report(f"Building embeddings for {len(all_chunks)} chunks...", 55)
        texts = [self._chunk_to_embedding_text(chunk) for chunk in all_chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        report("Saving vector index...", 70)
        self.vector_store.save(embeddings, all_chunks)
        report(f"Generating repository summary and {len(file_records)} file summaries with the local model...", 80)
        summary_bundle = self.llm_service.summarize_repository_bundle(
            repo_name=source_root.name,
            file_records=file_records,
            readme_excerpt=readme_excerpt(source_root),
        )
        file_summaries = summary_bundle["file_summaries"]
        repo_summary = summary_bundle["repo_summary"]
        repo_context = {
            "source_root": str(source_root),
            "repo_name": source_root.name,
            "repo_summary": repo_summary,
            "file_summaries": file_summaries,
        }
        report("Writing repository understanding cache...", 92)
        write_json(self.repo_context_path, repo_context)

        logger.info("Ingestion completed with %s files and %s chunks", indexed_files, len(all_chunks))
        report("Ingestion complete.", 100)
        return {
            "status": "success",
            "source_root": str(source_root),
            "files_indexed": indexed_files,
            "chunks_indexed": len(all_chunks),
            "repo_summary": repo_summary,
            "file_summaries": file_summaries,
        }

    @staticmethod
    def _chunk_to_embedding_text(chunk: Dict) -> str:
        return "\n".join(
            [
                f"File: {chunk['file_path']}",
                f"Type: {chunk['type']}",
                f"Name: {chunk['name']}",
                chunk["code"],
            ]
        )
