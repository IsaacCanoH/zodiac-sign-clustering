from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [('kmeans', '0005_kmeansrun_retraining_metadata')]

    operations = [
        migrations.AlterField(
            model_name='kmeansrun',
            name='dataset',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='kmeans_runs',
                to='datasets.dataset',
            ),
        ),
    ]
