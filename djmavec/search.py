# djmavec/search.py
"""
Search utilities: compute cosine similarity against stored embeddings (db.json).
Exposes search_similar(query_text, top_k, model)
"""

from typing import List, Dict, Any
import numpy as np
from .storage import load_all_records, save_record
from .vector_model import VectorModel

def _cosine_sim(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def search_similar(query: str, top_k: int = 5, model: VectorModel = None) -> List[Dict[str, Any]]:
    """
    Returns top_k records sorted by similarity score.
    Each result has: {"id", "text", "score"}
    """
    if model is None:
        model = VectorModel()  # default model (will use local fallback if needed)
    qvec = model.encode([query])[0]
    records = load_all_records()
    scored = []
    for r in records:
        emb = r.get("embedding")
        if not emb:
            continue
        score = _cosine_sim(qvec, emb)
        scored.append({"id": r.get("id"), "text": r.get("text"), "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
