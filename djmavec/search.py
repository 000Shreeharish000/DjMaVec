"""Semantic search helpers and Django model mixin.

VectorSearchMixin: provides in-memory cosine similarity ranking utilities.
VectorModel: Django model mixin that auto-embeds configured text fields on save
and stores the embedding into a MariaVectorField.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .embeddings import embed_text
from .utils import cosine_similarity, normalize_vector


class VectorSearchMixin:
    """Mixin offering in-memory cosine similarity search.

    This is database-agnostic and operates on Python sequences of vectors.
    Suitable for quick ranking and unit tests without a DB connection.
    """

    @staticmethod
    def rank_by_similarity(query_vec: np.ndarray, vectors: Sequence[np.ndarray]) -> List[Tuple[int, float]]:
        """Return list of (index, score) sorted by descending cosine similarity."""

        q = normalize_vector(query_vec)
        scores = [(i, cosine_similarity(q, v)) for i, v in enumerate(vectors)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class VectorModel(VectorSearchMixin):
    """Django model mixin to auto-generate embeddings.

    Contract for subclasses (Django models):
    - Define attribute/vector field name in class variable VECTOR_FIELD (str)
    - Define source text field(s) in TEXT_FIELDS (List[str] or str)
    - Optionally define EMBED_DIM (int); used by MariaVectorField(dim=...)
    - Provide an attribute matching VECTOR_FIELD storing the embedding

    Behavior:
    - On save(), concatenates text fields with newlines and embeds.
    - Stores resulting vector into VECTOR_FIELD.
    """

    VECTOR_FIELD: str = "embedding"
    TEXT_FIELDS: Sequence[str] | str = ()

    def _get_text_for_embedding(self) -> str:
        fields = self.TEXT_FIELDS
        if isinstance(fields, str):
            fields = [fields]
        parts: List[str] = []
        for name in fields:
            parts.append(str(getattr(self, name, "")))
        return "\n".join(parts).strip()

    def generate_embedding(self) -> np.ndarray:
        text = self._get_text_for_embedding()
        return embed_text(text)

    def save(self, *args, **kwargs):  # type: ignore[override]
        # Defer importing django to avoid hard dependency when importing package
        from django.db import models  # type: ignore

        if not isinstance(self, models.Model):  # safety
            return super().save(*args, **kwargs)  # type: ignore

        vec = self.generate_embedding()
        setattr(self, self.VECTOR_FIELD, vec)
        return super().save(*args, **kwargs)  # type: ignore
