DjMaVec: AI-Powered Vector Search for Django + MariaDB

DjMaVec is a next-generation Python library that integrates MariaDB’s new vector search engine directly into the Django ORM.
It allows developers to easily add AI-powered semantic search, recommendation systems, and similarity-based queries to their Django apps — without leaving the ORM.

 In simple terms: DjMaVec helps Django developers bring AI intelligence to their data — using MariaDB instead of relying on external vector databases like Pinecone or PostgreSQL extensions.

Why DjMaVec?

Django developers have long depended on PostgreSQL — but PostgreSQL’s vector support (via pgvector) requires external setup, manual integration, and non-native syntax.

DjMaVec changes that.
It gives MariaDB the AI edge, combining simplicity, speed, and semantic understanding — all within Django’s natural workflow.

 With DjMaVec, Django developers can now build AI-native applications natively on MariaDB, not just store data — they can understand it.

 Core Features

1) Seamless Django Integration
Add AI vector fields directly into your Django models — just like CharField or TextField.
Full ORM support with VectorField and VectorManager.

2) Automatic AI Embeddings
Automatically generate embeddings for any text field using a lightweight transformer model (all-MiniLM-L6-v2).
No manual model loading or API calls needed.

3) MariaDB Vector Storage
Stores and indexes embeddings in MariaDB’s native vector format — efficient, SQL-compatible, and scalable.

4) Semantic Search API
Perform top-K similarity searches using:

```python
Article.vectors.search("deep learning in education", top_k=5)
```

and get results ranked by meaning, not keywords.

5) Hybrid Query Support (coming soon)
Combine Django filters with semantic queries:

```python
Article.vectors.filter(category="AI").search("healthcare robots")
```

6) Admin Integration (Planned)
Visualize embeddings and similarity results directly from the Django admin panel.

7) Extensible Architecture
Easily switch to custom embedding models or APIs like OpenAI or Hugging Face.

Requirements

Python 3.8+

Django 3.2+

MariaDB 11+ (with vector support)

Libraries: numpy, sentence-transformers, mysql-connector-python

Installation
```bash
pip install -e .
```

(Once published on PyPI: pip install djmavec)

How to Run the Demo

The included demo showcases core embedding and similarity search — end to end.

Start MariaDB

```bash
mariadb --version
```

Ensure it’s running and accessible with vector support.

Install dependencies

```bash
pip install -r requirements.txt
```

Run the demo

```bash
python demo.py
```

Output example:

Encoding sentences...
Saving embeddings to MariaDB...
Searching for: "I feel depressed"
Match: "I am feeling sad and down today" (score=0.89)

Django Integration (In Progress)

DjMaVec is being developed to seamlessly integrate into your Django ORM workflow.

# models.py
from django.db import models
from djmavec.models import VectorField, VectorManager

class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    embedding = VectorField(dim=384)

    objects = models.Manager()
    vectors = VectorManager()

# views.py
query = "machine learning education"
similar_articles = Article.vectors.search(query, top_k=5)

for article in similar_articles:
    print(f"{article.title} → {article.similarity_score:.2f}")

Why Django Developers Will Shift to MariaDB
| Feature | PostgreSQL + pgvector | DjMaVec (MariaDB) |
|---|---:|:---|
| ORM Integration | ❌ Manual setup | ✅ Native Django ORM |
| External Dependencies | 🧩 pgvector plugin | 🚫 None |
| SQL Syntax | Complex | Simple, Pythonic |
| Setup Time | Hours | Minutes ⚡ |
| Cost | High (external APIs, hosting) | Free & open-source |
| Target Audience | ML Engineers | Django Developers |

DjMaVec democratizes AI for Django developers — no external services, no complex setup, just MariaDB and your existing ORM.

Future Roadmap :

✅ Vector embeddings for text

🔄 Image & audio embedding support

🧩 Full Django ORM integration (VectorField, VectorManager)

⚡ MariaDB native vector SQL support

🛠 Admin panel visualization tools

📦 PyPI public release

☁️ Cloud-ready container templates (Docker + MariaDB Vector)

🏁 Vision
