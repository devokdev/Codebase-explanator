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
        local_only = os.getenv("EMBEDDING_MODEL_LOCAL_ONLY", "true").lower() not in {"0", "false", "no"}
        try:
            self.model = SentenceTransformer(model_name, local_files_only=local_only)
        except Exception as exc:
            raise RuntimeError(
                "Embedding model is not available locally. "
                "Set EMBEDDING_MODEL to a local path or pre-download the model before starting the app."
            ) from exc
        self._cache: Dict[str, np.ndarray] = {}

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
