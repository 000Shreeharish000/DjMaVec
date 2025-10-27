"""djmavec public API.

Minimal Django + MariaDB vector search helpers.

Exports (lazy):
- VectorModel: A Django model mixin that auto-embeds configured text fields
- MariaVectorField: Custom field to store vector embeddings (as JSON array)
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["VectorModel", "MariaVectorField"]


def __getattr__(name: str) -> Any:  # PEP 562 lazy export
	if name == "VectorModel":
		return import_module(".search", __name__).__dict__["VectorModel"]
	if name == "MariaVectorField":
		# This import may fail if Django isn't installed; raise a helpful error
		try:
			return import_module(".fields", __name__).__dict__["MariaVectorField"]
		except ModuleNotFoundError as e:
			if "django" in str(e):
				raise ImportError(
					"MariaVectorField requires Django. Please install Django to use this feature."
				) from e
			raise
	raise AttributeError(f"module 'djmavec' has no attribute {name!r}")

