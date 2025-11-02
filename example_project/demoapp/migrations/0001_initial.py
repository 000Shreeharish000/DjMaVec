from django.db import migrations, models
import djmavec.models

class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='Article',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('content', models.TextField()),
                ('embedding', djmavec.models.VectorField(dim=384, null=True, blank=True)),
            ],
        ),
    ]
