import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dbscan', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='dbscanrun',
            name='activated_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterModelOptions(
            name='dbscanrun',
            options={'ordering': ('-activated_at', '-created_at')},
        ),
    ]
