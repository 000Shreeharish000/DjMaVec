# djmavec

Minimal, professional Django + MariaDB vector search toolkit.

- Store embeddings in MariaDB using a custom field
- Auto-generate embeddings on save via a model mixin
- Perform cosine similarity search in-memory for quick ranking

This MVP uses SentenceTransformer "all-MiniLM-L6-v2" for embeddings. Tests can run without downloading the model by enabling a deterministic fake-embedding mode.

## Install

Editable install for local development:

```powershell
# From the repo root
pip install -e .
```

This installs dependencies: django, numpy, sentence-transformers.

## Quick start

Define a Django model that embeds text on save and stores the vector in MariaDB.

```python
# models.py
from django.db import models
from djmavec import VectorModel, MariaVectorField

class Article(VectorModel, models.Model):
    title = models.CharField(max_length=200)
    body = models.TextField()

    # Where to store the embedding
    embedding = MariaVectorField(dim=384)

    # Configure which text fields to use for embedding
    TEXT_FIELDS = ["title", "body"]
    VECTOR_FIELD = "embedding"
```

Usage:

```python
# Creating an article auto-generates the embedding on save
article = Article(title="Hello", body="Vector search with Django + MariaDB")
article.save()

# Rank a few vectors for a query (in-memory ranking helper)
from djmavec.embeddings import embed_text
q = embed_text("Django vector search")
vecs = [article.embedding]  # add more as needed
ranking = Article.rank_by_similarity(q, vecs)
```

## Testing

Run unit tests (DB-free):

```powershell
$env:DJMAVEC_FAKE_EMBEDDINGS = "1"
python -m unittest djmavec/tests.py
```

The fake mode avoids downloading the sentence-transformers model and yields deterministic unit-norm vectors.

## Notes

- Embeddings are stored as JSON in a TextField for broad MariaDB compatibility.
- For production-scale vector search, consider specialized vector indexes or external services.
- No API keys required. Hooking external embedding APIs can be added later.

## License

MIT