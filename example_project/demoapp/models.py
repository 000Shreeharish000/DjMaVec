from django.db import models
from djmavec.models import VectorField
from djmavec.manager import VectorManager

class Article(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    embedding = VectorField(dim=384, null=True, blank=True)

    objects = models.Manager()
    vectors = VectorManager()

    def __str__(self):
        return self.title
