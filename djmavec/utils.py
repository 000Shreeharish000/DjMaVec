"""Utility helpers for vector math and serialization."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Return unit-norm vector; if zero vector, return as-is."""

    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    return v


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two 1D vectors."""

    a = normalize_vector(a)
    b = normalize_vector(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b))


def serialize_vector(v: np.ndarray) -> List[float]:
    """Convert vector to JSON-serializable list of floats."""

    arr = np.asarray(v, dtype=np.float32)
    return arr.tolist()


def parse_vector(values: Iterable[float]) -> np.ndarray:
    """Parse iterable of floats into numpy array with dtype float32."""

    return np.asarray(list(values), dtype=np.float32)
