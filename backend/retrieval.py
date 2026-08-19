from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from .utils import ensure_directory, read_json, write_json


class VectorStore:
    def __init__(self, index_path: Path, metadata_path: Path) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: List[Dict] = []

    def save(self, embeddings: np.ndarray, metadata: List[Dict], repo_source: str = "") -> None:
        ensure_directory(self.index_path.parent)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, str(self.index_path))
        write_json(self.metadata_path, metadata)
        self.index = index
        self.metadata = metadata

        # Persist chunks to PostgreSQL relational store
        try:
            from .database import SessionLocal, CodeChunkModel, init_db
            init_db()
            db = SessionLocal()
            # Clear old records for clean sync
            db.query(CodeChunkModel).delete()
            for idx, item in enumerate(metadata):
                record = CodeChunkModel(
                    chunk_id=idx,
                    repo_source=repo_source,
                    file_path=item.get("file_path", ""),
                    name=item.get("name", ""),
                    chunk_type=item.get("type", "snippet"),
                    line_start=item.get("line_start"),
                    line_end=item.get("line_end"),
                    code=item.get("code", "")
                )
                db.add(record)
            db.commit()
            db.close()
        except Exception as e:
            print(f"PostgreSQL sync skipped/noted: {e}")

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError("FAISS index not found. Run ingestion first.")
        self.index = faiss.read_index(str(self.index_path))
        
        # Try to hydrate from PostgreSQL first; fallback to JSON
        try:
            from .database import SessionLocal, CodeChunkModel
            db = SessionLocal()
            records = db.query(CodeChunkModel).order_by(CodeChunkModel.chunk_id).all()
            if records:
                self.metadata = [
                    {
                        "file_path": r.file_path,
                        "name": r.name,
                        "type": r.chunk_type,
                        "line_start": r.line_start,
                        "line_end": r.line_end,
                        "code": r.code
                    }
                    for r in records
                ]
            else:
                self.metadata = read_json(self.metadata_path, default=[])
            db.close()
        except Exception:
            self.metadata = read_json(self.metadata_path, default=[])

    def is_ready(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict]]:
        if self.index is None:
            self.load()

        assert self.index is not None
        query_vector = np.asarray([query_embedding], dtype="float32")
        scores, indices = self.index.search(query_vector, top_k)

        results: List[Tuple[float, Dict]] = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            results.append((float(score), self.metadata[index]))
        return results

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        terms = set()
        for term in re.findall(r"[A-Za-z0-9_]+", query.lower()):
            if len(term) <= 2:
                continue
            terms.add(term)
            if term.endswith("ed") and len(term) > 4:
                terms.add(term[:-2])
            if term.endswith("ing") and len(term) > 5:
                terms.add(term[:-3])
            if term.endswith("s") and len(term) > 4:
                terms.add(term[:-1])
            if term.startswith("embedd") or term.startswith("embed"):
                terms.update({"embed", "embeds", "embedded", "embedding", "embeddings"})
            if term in {"vector", "vectors", "store", "stored"}:
                terms.update({"faiss", "index", "indexflatip", "vector_store"})
        return terms

    @staticmethod
    def _lexical_score(query_terms: set[str], chunk: Dict) -> float:
        path_name = f"{chunk['file_path']} {chunk['name']} {chunk['type']}".lower()
        code = chunk["code"][:1800].lower()
        score = 0.0

        for term in query_terms:
            if term in path_name:
                score += 0.45
            if term in code:
                score += 0.16

        return score

    def hybrid_search(self, query: str, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict]]:
        if self.index is None:
            self.load()

        vector_results = self.search(query_embedding, top_k=max(top_k * 4, 8))
        query_terms = self._query_terms(query)
        rescored_by_key: Dict[tuple[str, str, int | None, int | None], Tuple[float, Dict]] = {}

        for vector_score, chunk in vector_results:
            chunk_text = f"{chunk['file_path']} {chunk['name']} {chunk['type']} {chunk['code'][:1200]}".lower()
            overlap = self._lexical_score(query_terms, chunk)
            exact_dataset_bonus = 2.0 if "dataset" in query_terms and "dataset" in chunk_text else 0.0
            final_score = float(vector_score) + overlap + exact_dataset_bonus
            key = (chunk["file_path"], chunk["name"], chunk.get("line_start"), chunk.get("line_end"))
            rescored_by_key[key] = (final_score, chunk)

        for chunk in self.metadata:
            lexical_score = self._lexical_score(query_terms, chunk)
            if lexical_score <= 0:
                continue
            key = (chunk["file_path"], chunk["name"], chunk.get("line_start"), chunk.get("line_end"))
            current = rescored_by_key.get(key)
            final_score = 0.35 + lexical_score
            if current is None or final_score > current[0]:
                rescored_by_key[key] = (final_score, chunk)

        rescored = list(rescored_by_key.values())
        rescored.sort(key=lambda item: item[0], reverse=True)
        return rescored[:top_k]
