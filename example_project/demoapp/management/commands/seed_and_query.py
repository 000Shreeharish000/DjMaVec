from django.core.management.base import BaseCommand
from django.core.management import call_command
from demoapp.models import Article
from djmavec.vector_model import VectorModel

class Command(BaseCommand):
    help = 'Seeds sample articles, computes embeddings, and runs a query using VectorManager'

    def handle(self, *args, **options):
        # Run migrations
        self.stdout.write('Applying migrations...')
        call_command('migrate', run_syncdb=True, verbosity=0)

        # Seed data
        examples = [
            ("I am feeling sad and down today."),
            ("What a beautiful sunny day! I love biking."),
            ("I feel depressed and low on energy."),
            ("Python programming and Django are fun to learn."),
            ("I am happy and joyful because I got a new job.")
        ]
        titles = [
            'sadness', 'sunny', 'depressed', 'django', 'happy'
        ]

        self.stdout.write('Seeding example articles...')
        # Remove existing rows
        Article.objects.all().delete()

        model = VectorModel()
        embs = model.encode(examples)
        for title, text, emb in zip(titles, examples, embs):
            Article.objects.create(title=title, content=text, embedding=emb)

        self.stdout.write('Done. Running sample search...')
        query = 'I am feeling depressed'
        results = Article.vectors.search(query, top_k=3)
        self.stdout.write(f"Query: '{query}'")
        for r in results:
            self.stdout.write(f"  {r.title} (score={getattr(r, 'similarity_score', 0):.4f}) - {r.content}")
