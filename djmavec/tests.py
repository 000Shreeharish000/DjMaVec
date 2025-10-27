import os
import unittest

import numpy as np

# Ensure tests do not attempt to download models
os.environ.setdefault("DJMAVEC_FAKE_EMBEDDINGS", "1")

from .embeddings import EMBED_DIM, embed_text, embed_batch
from .search import VectorSearchMixin


class TestEmbeddings(unittest.TestCase):
    def test_embed_text_shape(self):
        v = embed_text("hello world")
        self.assertEqual(v.shape, (EMBED_DIM,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_embed_batch_shapes(self):
        texts = ["a", "b", "c"]
        M = embed_batch(texts)
        self.assertEqual(M.shape, (len(texts), EMBED_DIM))


class TestSearch(unittest.TestCase):
    def test_rank_by_similarity(self):
        q = embed_text("cat")
        vecs = [embed_text("cat"), embed_text("dog"), embed_text("banana")]
        ranking = VectorSearchMixin.rank_by_similarity(q, vecs)
        # Best should be index 0 ("cat") with highest similarity
        self.assertEqual(ranking[0][0], 0)
        self.assertGreater(ranking[0][1], ranking[-1][1])


if __name__ == "__main__":
    unittest.main()
