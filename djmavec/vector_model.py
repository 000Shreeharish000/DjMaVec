# vector_model.py

from sentence_transformers import SentenceTransformer
import numpy as np

# Load a small, fast embedding model locally
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def get_embedding(text: str) -> np.ndarray:
    """
    Generate a vector embedding for a given text.
    
    Args:
        text (str): The input text.
    
    Returns:
        np.ndarray: Embedding vector as a NumPy array.
    """
    embedding = model.encode(text)
    return embedding

def get_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1, vec2 (np.ndarray): Embedding vectors.
    
    Returns:
        float: Cosine similarity score between -1 and 1.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

if __name__ == "__main__":
    # Quick test
    text1 = "Hello world!"
    text2 = "Hi there!"
    vec1 = get_embedding(text1)
    vec2 = get_embedding(text2)
    print("Similarity:", get_similarity(vec1, vec2))
