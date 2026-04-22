from __future__ import annotations

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

    def save(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        ensure_directory(self.index_path.parent)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, str(self.index_path))
        write_json(self.metadata_path, metadata)
        self.index = index
        self.metadata = metadata

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError("FAISS index not found. Run ingestion first.")
        self.index = faiss.read_index(str(self.index_path))
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

    def hybrid_search(self, query: str, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict]]:
        if self.index is None:
            self.load()

        vector_results = self.search(query_embedding, top_k=max(top_k * 4, 8))
        query_terms = {term.lower() for term in query.split() if len(term) > 2}
        rescored: List[Tuple[float, Dict]] = []

        for vector_score, chunk in vector_results:
            haystack = f"{chunk['file_path']} {chunk['name']} {chunk['type']} {chunk['code'][:1200]}".lower()
            overlap = sum(1 for term in query_terms if term in haystack)
            exact_dataset_bonus = 2.0 if "dataset" in query_terms and "dataset" in haystack else 0.0
            final_score = float(vector_score) + (0.18 * overlap) + exact_dataset_bonus
            rescored.append((final_score, chunk))

        rescored.sort(key=lambda item: item[0], reverse=True)
        return rescored[:top_k]
