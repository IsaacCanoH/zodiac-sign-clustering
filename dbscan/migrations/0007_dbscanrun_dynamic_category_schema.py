from django.db import migrations, models


def backfill_category_column(apps, schema_editor):
    model = apps.get_model('dbscan', 'DBSCANRun')
    model.objects.filter(category_filter__gt='').update(category_column='categoria')


class Migration(migrations.Migration):
    dependencies = [('dbscan', '0006_dbscanrun_saved_state')]
    operations = [
        migrations.AddField(
            model_name='dbscanrun', name='category_column',
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.AddField(
            model_name='dbscanrun', name='schema_profile',
            field=models.JSONField(default=dict),
        ),
        migrations.RunPython(backfill_category_column, migrations.RunPython.noop),
    ]
