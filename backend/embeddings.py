from __future__ import annotations

import hashlib
import os
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

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            local_only = os.getenv("EMBEDDING_MODEL_LOCAL_ONLY", "false").lower() not in {"0", "false", "no"}
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=local_only)
            except Exception:
                self._model = SentenceTransformer(self.model_name, local_files_only=False)
        return self._model

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        import gc
        import torch

        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        result_vectors: List[np.ndarray | None] = [None] * len(texts)

        for index, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                result_vectors[index] = self._cache[key]
            else:
                uncached_indices.append(index)
                uncached_texts.append(text)

        if uncached_texts:
            with torch.no_grad():
                for start_idx in range(0, len(uncached_texts), batch_size):
                    end_idx = start_idx + batch_size
                    batch_texts = uncached_texts[start_idx:end_idx]
                    batch_indices = uncached_indices[start_idx:end_idx]

                    embeddings = self.model.encode(
                        batch_texts,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        batch_size=batch_size,
                    )
                    for idx, vector in zip(batch_indices, embeddings):
                        key = self._cache_key(texts[idx])
                        self._cache[key] = vector.astype("float32")
                        result_vectors[idx] = self._cache[key]

                    # Trigger memory cleanup between batches
                    gc.collect()

        return np.vstack(result_vectors).astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query], batch_size=1)[0]
