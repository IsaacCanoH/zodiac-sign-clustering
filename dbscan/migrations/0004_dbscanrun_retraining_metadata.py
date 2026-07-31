from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [('dbscan', '0003_dbscanrun_dataset_identity')]
    operations = [
        migrations.AddField(model_name='dbscanrun', name='name', field=models.CharField(blank=True, max_length=150)),
        migrations.AddField(model_name='dbscanrun', name='topic', field=models.CharField(blank=True, max_length=150)),
        migrations.AddField(model_name='dbscanrun', name='description', field=models.TextField(blank=True)),
        migrations.AddField(model_name='dbscanrun', name='version', field=models.PositiveIntegerField(default=1)),
        migrations.AddField(model_name='dbscanrun', name='source_row_count', field=models.PositiveIntegerField(default=0)),
        migrations.AddField(model_name='dbscanrun', name='new_record_count', field=models.PositiveIntegerField(default=0)),
        migrations.AddField(model_name='dbscanrun', name='dataset_schema_fingerprint', field=models.CharField(blank=True, db_index=True, max_length=64)),
        migrations.AddField(model_name='dbscanrun', name='training_config_fingerprint', field=models.CharField(blank=True, db_index=True, max_length=64)),
        migrations.AddField(model_name='dbscanrun', name='preprocessing_state', field=models.JSONField(default=dict)),
        migrations.AddField(model_name='dbscanrun', name='estimator_state', field=models.JSONField(default=dict)),
        migrations.AddField(model_name='dbscanrun', name='library_versions', field=models.JSONField(default=dict)),
        migrations.AddField(model_name='dbscanrun', name='change_summary', field=models.JSONField(default=dict)),
        migrations.AddField(model_name='dbscanrun', name='parent_run', field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='retrained_versions', to='dbscan.dbscanrun')),
    ]
