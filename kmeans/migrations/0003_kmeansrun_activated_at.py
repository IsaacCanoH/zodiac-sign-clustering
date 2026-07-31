import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('kmeans', '0002_kmeansrun_cluster_comparison_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='kmeansrun',
            name='activated_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterModelOptions(
            name='kmeansrun',
            options={'ordering': ('-activated_at', '-created_at')},
        ),
    ]
