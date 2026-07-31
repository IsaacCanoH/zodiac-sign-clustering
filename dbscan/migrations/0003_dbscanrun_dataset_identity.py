import hashlib
import json

from django.db import migrations, models


def dataset_fingerprint(dataset):
    payload = {'columns': dataset.columns, 'records': dataset.records}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def populate_dataset_identity(apps, schema_editor):
    DBSCANRun = apps.get_model('dbscan', 'DBSCANRun')
    for run in DBSCANRun.objects.select_related('dataset'):
        run.dataset_fingerprint = dataset_fingerprint(run.dataset)
        run.dataset_source_name = run.dataset.source_name
        run.save(
            update_fields=('dataset_fingerprint', 'dataset_source_name')
        )


class Migration(migrations.Migration):

    dependencies = [
        ('dbscan', '0002_dbscanrun_activated_at'),
    ]

    operations = [
        migrations.AddField(
            model_name='dbscanrun',
            name='dataset_fingerprint',
            field=models.CharField(blank=True, db_index=True, max_length=64),
        ),
        migrations.AddField(
            model_name='dbscanrun',
            name='dataset_source_name',
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.RunPython(
            populate_dataset_identity,
            migrations.RunPython.noop,
        ),
        migrations.AlterField(
            model_name='dbscanrun',
            name='dataset_fingerprint',
            field=models.CharField(db_index=True, max_length=64),
        ),
        migrations.AlterField(
            model_name='dbscanrun',
            name='dataset_source_name',
            field=models.CharField(max_length=255),
        ),
    ]
