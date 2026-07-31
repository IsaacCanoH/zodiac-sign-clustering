from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [('kmeans', '0004_kmeansrun_dataset_identity')]
    operations = [
        migrations.AddField(model_name='kmeansrun', name='name', field=models.CharField(blank=True, max_length=150)),
        migrations.AddField(model_name='kmeansrun', name='topic', field=models.CharField(blank=True, max_length=150)),
        migrations.AddField(model_name='kmeansrun', name='description', field=models.TextField(blank=True)),
        migrations.AddField(model_name='kmeansrun', name='version', field=models.PositiveIntegerField(default=1)),
        migrations.AddField(model_name='kmeansrun', name='source_row_count', field=models.PositiveIntegerField(default=0)),
        migrations.AddField(model_name='kmeansrun', name='new_record_count', field=models.PositiveIntegerField(default=0)),
        migrations.AddField(model_name='kmeansrun', name='dataset_schema_fingerprint', field=models.CharField(blank=True, db_index=True, max_length=64)),
        migrations.AddField(model_name='kmeansrun', name='training_config_fingerprint', field=models.CharField(blank=True, db_index=True, max_length=64)),
        migrations.AddField(model_name='kmeansrun', name='preprocessing_state', field=models.JSONField(default=dict)),
        migrations.AddField(model_name='kmeansrun', name='estimator_state', field=models.JSONField(default=dict)),
        migrations.AddField(model_name='kmeansrun', name='library_versions', field=models.JSONField(default=dict)),
        migrations.AddField(model_name='kmeansrun', name='change_summary', field=models.JSONField(default=dict)),
        migrations.AddField(model_name='kmeansrun', name='parent_run', field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='retrained_versions', to='kmeans.kmeansrun')),
    ]
