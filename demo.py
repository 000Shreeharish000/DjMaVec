# demo.py
"""
Quick demo: Analyzes semantic similarity between a vocabulary of words.
Usage:
    python demo.py
"""

from djmavec.vector_model import VectorModel
from djmavec.search import _cosine_sim # Import the private cosine similarity function
import numpy as np

def run_word_similarity_demo():
    """
    Encodes a vocabulary of words and calculates similarity scores between them.
    """
    print("Initializing model (may download model on first run)...")
    try:
        model = VectorModel()
    except Exception as e:
        print(f"Could not initialize model: {e}")
        return

    vocabulary = [
        'happy', 'joyful', 'miserable', 'enthusiastic', 'angry', 
        'ecstatic', 'sad', 'content'
    ]
    print(f"Vocabulary: {vocabulary}\n")

    print("Encoding vocabulary...")
    embeddings = model.encode(vocabulary)
    
    # Create a dictionary mapping words to their vector embeddings
    word_to_embedding = dict(zip(vocabulary, embeddings))

    # --- Part 1: Ranking for a specific query ('happy') ---
    query_word = 'happy'
    query_embedding = word_to_embedding[query_word]

    print(f"Ranking for query '{query_word}':\n")
    
    scores = []
    for word, embedding in word_to_embedding.items():
        score = _cosine_sim(query_embedding, embedding)
        scores.append((word, score))
    
    # Sort by score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    
    for word, score in scores:
        print(f"{score:.4f} : {word}")

    # --- Part 2: Nearest neighbors for each word ---
    print("\n----------------------------------------")
    print("Nearest neighbors (top 3) for each word:")
    print("----------------------------------------\n")

    all_neighbors = {}
    for word_a, embedding_a in word_to_embedding.items():
        neighbor_scores = []
        for word_b, embedding_b in word_to_embedding.items():
            if word_a == word_b:
                continue # Don't compare a word to itself
            score = _cosine_sim(embedding_a, embedding_b)
            neighbor_scores.append((word_b, score))
        
        # Sort neighbors by score
        neighbor_scores.sort(key=lambda x: x[1], reverse=True)
        all_neighbors[word_a] = neighbor_scores[:3]

    # Print the results in the desired format
    for word, neighbors in all_neighbors.items():
        print(f"'{word}': {neighbors}")


if __name__ == "__main__":
    run_word_similarity_demo()
