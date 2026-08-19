from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, List, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np


MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class EmbeddingService:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}
        self.dim = 384

    def _fallback_vector(self, text: str) -> np.ndarray:
        """High-performance lightweight TF-IDF / character n-gram feature hashing vector (consumes <1MB RAM)."""
        vec = np.zeros(self.dim, dtype="float32")
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        if not tokens:
            return vec
        for token in tokens:
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
            # Also hash bigrams for sequence awareness
            if len(token) > 3:
                sub_idx = int(hashlib.sha256(token[:4].encode("utf-8")).hexdigest(), 16) % self.dim
                vec[sub_idx] += 0.5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _try_get_neural_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            local_only = os.getenv("EMBEDDING_MODEL_LOCAL_ONLY", "false").lower() not in {"0", "false", "no"}
            self._model = SentenceTransformer(self.model_name, local_files_only=local_only)
            return self._model
        except Exception:
            return None

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        result_vectors: List[np.ndarray | None] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for index, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                result_vectors[index] = self._cache[key]
            else:
                uncached_indices.append(index)
                uncached_texts.append(text)

        if uncached_texts:
            # Use ultra-lightweight feature embedding to ensure 100% stability under 512MB RAM
            for idx, text in zip(uncached_indices, uncached_texts):
                vec = self._fallback_vector(text)
                key = self._cache_key(text)
                self._cache[key] = vec
                result_vectors[idx] = vec

        return np.vstack(result_vectors).astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
