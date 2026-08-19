from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class EmbeddingService:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            local_only = os.getenv("EMBEDDING_MODEL_LOCAL_ONLY", "false").lower() not in {"0", "false", "no"}
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=local_only)
            except Exception:
                # Fallback to online download if local files are missing
                self._model = SentenceTransformer(self.model_name, local_files_only=False)
        return self._model

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        uncached: List[Tuple[int, str]] = []
        result_vectors: List[np.ndarray | None] = [None] * len(texts)

        for index, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                result_vectors[index] = self._cache[key]
            else:
                uncached.append((index, text))

        if uncached:
            embeddings = self.model.encode(
                [text for _, text in uncached],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for (index, text), vector in zip(uncached, embeddings):
                key = self._cache_key(text)
                self._cache[key] = vector.astype("float32")
                result_vectors[index] = self._cache[key]

        return np.vstack(result_vectors).astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
