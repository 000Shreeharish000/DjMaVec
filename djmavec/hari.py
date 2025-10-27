"""Tiny demo script for exploring embeddings.

This script uses the package embedding helpers and defaults to the fake
embeddings mode so it runs without downloading a model. Run as:

	python -m djmavec.hari

or

	python djmavec/hari.py

"""
from __future__ import annotations

import os
from typing import List

# Avoid downloading the real model for quick demos/tests
os.environ.setdefault("DJMAVEC_FAKE_EMBEDDINGS", "1")

try:
	# Prefer package import when executed as a module
	from djmavec.embeddings import embed_batch
except Exception:
	# Fallback if executed as a script from the repo root
	from .embeddings import embed_batch  # type: ignore


def main(texts: List[str] | None = None) -> None:
	if texts is None:
		texts = ["Hello world", "How are you?", "VectorModel works!"]
	mat = embed_batch(texts)
	print("Embedded matrix shape:", getattr(mat, "shape", None))
	print(mat)


if __name__ == "__main__":
	main()

