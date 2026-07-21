from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('dashboard', '0001_initial'),
        ('datasets', '0001_initial'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[migrations.DeleteModel(name='Dataset')],
            database_operations=[],
        ),
    ]
