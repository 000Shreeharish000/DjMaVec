"""Lightweight helpers that delegate to the package's embedding utilities.

This module avoids loading the sentence-transformers model at import time and
delegates to `djmavec.embeddings` which handles lazy loading and the
`DJMAVEC_FAKE_EMBEDDINGS` test mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .embeddings import embed_text, embed_batch
from .utils import cosine_similarity as _cosine_similarity


def get_embedding(text: str) -> np.ndarray:
    """Return an embedding for a single text using the package embedding API."""
    return embed_text(text)


def get_embeddings(texts: list[str]) -> np.ndarray:  # pragma: no cover - thin wrapper
    """Return embeddings for a batch of texts."""
    return embed_batch(texts)


def get_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Delegates to the normalized cosine implementation in `utils`.
    """
    return _cosine_similarity(vec1, vec2)


if __name__ == "__main__":
    # Small demo that uses fake embeddings by default so it is safe offline.
    import os

    os.environ.setdefault("DJMAVEC_FAKE_EMBEDDINGS", "1")
    a = get_embedding("Hello world!")
    b = get_embedding("Hi there!")
    print("Similarity:", get_similarity(a, b))
