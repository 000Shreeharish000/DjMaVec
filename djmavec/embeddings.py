"""Embedding utilities using SentenceTransformer.

This module lazily loads the 'all-MiniLM-L6-v2' model from sentence-transformers
to generate sentence embeddings. To keep tests lightweight and avoid downloading
the model, set the environment variable DJMAVEC_FAKE_EMBEDDINGS=1 to generate
deterministic pseudo-embeddings instead.
"""

from __future__ import annotations

import hashlib
import os
from typing import Iterable, List

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # Known output dimension for MiniLM-L6-v2

_MODEL = None  # Lazy-loaded global


def _fake_vector_for_text(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Generate a deterministic pseudo-embedding for a given text.

    Uses SHA256 hash to seed a PRNG for reproducible vectors of a fixed size.
    """

    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0, 1, size=(dim,)).astype(np.float32)
    # Normalize to unit length for stability
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _fake_vectors_for_batch(texts: Iterable[str], dim: int = EMBED_DIM) -> np.ndarray:
    return np.vstack([_fake_vector_for_text(t, dim) for t in texts])


def _get_model():
    """Lazily load the SentenceTransformer model.

    Avoids import cost and model download unless real embeddings are requested.
    """

    global _MODEL
    if _MODEL is None:
        # Import here to avoid hard dependency at import time
        from sentence_transformers import SentenceTransformer  # type: ignore

        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def embed_text(text: str) -> np.ndarray:
    """Embed a single text into a vector.

    Respects DJMAVEC_FAKE_EMBEDDINGS to bypass model download for tests.
    Returns a 1D numpy array of shape (EMBED_DIM,).
    """

    if os.getenv("DJMAVEC_FAKE_EMBEDDINGS"):
        return _fake_vector_for_text(text)
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)  # type: ignore
    return np.asarray(vec, dtype=np.float32)


def embed_batch(texts: List[str]) -> np.ndarray:
    """Embed a batch of texts into a matrix of shape (n, EMBED_DIM)."""

    if os.getenv("DJMAVEC_FAKE_EMBEDDINGS"):
        return _fake_vectors_for_batch(texts)
    model = _get_model()
    mat = model.encode(texts, normalize_embeddings=True)  # type: ignore
    return np.asarray(mat, dtype=np.float32)
