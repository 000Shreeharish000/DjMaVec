# djmavec/models.py
from django.db import models

class VectorField(models.JSONField):
    """
    A custom Django field to store vector embeddings.
    It's based on JSONField for broad database compatibility,
    specifically targeting MariaDB's JSON support for storing arrays of floats.
    """
    def __init__(self, *args, **kwargs):
        # The dimension of the vector, e.g., 384 for all-MiniLM-L6-v2
        self.dim = kwargs.pop('dim', None)
        if self.dim is None:
            raise ValueError("VectorField requires a 'dim' argument.")
        
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs['dim'] = self.dim
        return name, path, args, kwargs

    def db_type(self, connection):
        """
        For MariaDB/MySQL, JSON is appropriate.
        Future versions could return 'VECTOR' for databases with native vector types.
        """
        return 'json'
