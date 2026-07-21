from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [('dashboard', '0001_initial')]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.CreateModel(
                    name='Dataset',
                    fields=[
                        (
                            'id',
                            models.BigAutoField(
                                auto_created=True,
                                primary_key=True,
                                serialize=False,
                                verbose_name='ID',
                            ),
                        ),
                        ('source_name', models.CharField(max_length=255)),
                        ('columns', models.JSONField()),
                        ('records', models.JSONField()),
                        ('uploaded_at', models.DateTimeField(auto_now=True)),
                    ],
                    options={'db_table': 'dashboard_dataset'},
                ),
            ],
            database_operations=[],
        ),
    ]
