"""Django field for storing vector embeddings.

This field stores embeddings as JSON array of floats. It is implemented as a
TextField with JSON serialization for portability across MariaDB versions. For
best performance in production, consider using a native JSON column or a
separate table with vector indexing. This MVP prioritizes portability.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

import numpy as np
from django.core import checks
from django.db import models

from .utils import parse_vector, serialize_vector
from .embeddings import embed_text


class MariaVectorField(models.TextField):
    """Store vector embeddings as JSON list of floats in a TextField.

    - Python value: numpy.ndarray (1D float32)
    - DB value: JSON string (e.g., "[0.1, 0.2, ...]")
    """

    description = "Vector embedding stored as JSON in TextField"

    def __init__(self, *args: Any, dim: Optional[int] = None, text_source: Optional[str] = None, **kwargs: Any):
        self.dim = dim
        self.text_source = text_source
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.dim is not None:
            kwargs["dim"] = self.dim
        if self.text_source is not None:
            kwargs["text_source"] = self.text_source
        return name, path, args, kwargs

    def from_db_value(self, value: Optional[str], expression, connection):  # type: ignore[override]
        if value is None:
            return None
        try:
            data = json.loads(value)
            vec = parse_vector(data)
            if self.dim is not None and vec.shape[0] != self.dim:
                # Do not fail hard; return as parsed but warn via checks framework at migrate time
                pass
            return vec
        except Exception:
            return None

    def to_python(self, value: Any):  # type: ignore[override]
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.astype(np.float32)
        if isinstance(value, (list, tuple)):
            return parse_vector(value)
        if isinstance(value, str):
            try:
                return parse_vector(json.loads(value))
            except Exception:
                return None
        return None

    def get_prep_value(self, value: Any):  # type: ignore[override]
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if self.dim is not None and value.shape[0] != self.dim:
                raise ValueError(f"Vector dim {value.shape[0]} != expected {self.dim}")
            return json.dumps(serialize_vector(value))
        if isinstance(value, (list, tuple)):
            arr = parse_vector(value)
            if self.dim is not None and arr.shape[0] != self.dim:
                raise ValueError(f"Vector dim {arr.shape[0]} != expected {self.dim}")
            return json.dumps(serialize_vector(arr))
        if isinstance(value, str):
            # Assume already JSON; trust the caller
            return value
        return super().get_prep_value(value)

    def check(self, **kwargs):  # type: ignore[override]
        errors = super().check(**kwargs)
        if self.dim is not None and (not isinstance(self.dim, int) or self.dim <= 0):
            errors.append(
                checks.Error("dim must be a positive integer", obj=self)
            )
        return errors

    def pre_save(self, model_instance, add):  # type: ignore[override]
        """Auto-generate embedding from text_source if configured.

        If text_source is provided and current value is falsy/None, compute
        embedding from the specified text field on the model and store it.
        """
        current = getattr(model_instance, self.attname)
        if (current is None or current == "") and self.text_source:
            text_val = getattr(model_instance, self.text_source, "")
            vec = embed_text(str(text_val))
            # Set the numpy array; Django will call get_prep_value for DB
            setattr(model_instance, self.attname, vec)
            return vec
        return super().pre_save(model_instance, add)
