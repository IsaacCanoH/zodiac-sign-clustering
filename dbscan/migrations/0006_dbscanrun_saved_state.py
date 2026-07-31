from django.db import migrations, models


def mark_existing_as_saved(apps, schema_editor):
    apps.get_model('dbscan', 'DBSCANRun').objects.update(is_saved=True)


class Migration(migrations.Migration):
    dependencies = [('dbscan', '0005_preserve_models_without_dataset')]
    operations = [
        migrations.AddField(
            model_name='dbscanrun', name='is_saved',
            field=models.BooleanField(db_index=True, default=True),
        ),
        migrations.AddField(
            model_name='dbscanrun', name='saved_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.RunPython(mark_existing_as_saved, migrations.RunPython.noop),
    ]
