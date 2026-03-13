"""Embedding utilities powered by sentence-transformers."""

from __future__ import annotations
from typing import Iterable
import numpy as np
from sentence_transformers import SentenceTransformer

from config.config import get_settings


_EMBEDDING_MODEL: SentenceTransformer | None = None


def load_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model for reuse across requests."""

    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is None:
        settings = get_settings()
        _EMBEDDING_MODEL = SentenceTransformer(settings.embedding_model_name)

    return _EMBEDDING_MODEL


def create_embeddings(text: str | Iterable[str]) -> np.ndarray:
    """Create normalized embeddings for one or many text inputs."""

    model = load_embedding_model()

    if isinstance(text, str):
        vector = model.encode(text, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)

    text_list = list(text)
    if not text_list:
        return np.empty((0, 0), dtype=np.float32)

    vectors = model.encode(text_list, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)
