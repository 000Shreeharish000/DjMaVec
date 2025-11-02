# djmavec/vector_model.py
"""
VectorModel: uses a sentence-transformers model to create embeddings.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        model_name - HuggingFace sentence-transformers model name.
        """
        self.model_name = model_name
        self._model = None
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Failed to load SentenceTransformer model: {e}")
            print("Please ensure 'sentence-transformers' is installed (`pip install sentence-transformers`)")
            raise

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode list of texts into list of vectors (list of floats).
        Always returns Python-lists (JSON serializable).
        """
        if not self._model:
            raise RuntimeError("Model is not loaded.")
            
        if not isinstance(texts, list):
            texts = [texts]

        # returns numpy array
        vecs = self._model.encode(texts, show_progress_bar=False)
        # convert to nested lists
        return [list(map(float, v)) for v in np.array(vecs)]

