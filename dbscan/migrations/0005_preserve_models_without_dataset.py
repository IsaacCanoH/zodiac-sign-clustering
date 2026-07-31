from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [('dbscan', '0004_dbscanrun_retraining_metadata')]

    operations = [
        migrations.AlterField(
            model_name='dbscanrun',
            name='dataset',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='dbscan_runs',
                to='datasets.dataset',
            ),
        ),
    ]
