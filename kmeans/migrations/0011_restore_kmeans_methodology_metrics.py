from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('kmeans', '0010_remove_kmeansrun_cluster_category_association_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='kmeansrun', name='results_by_k',
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name='kmeansrun', name='recommended_k_silhouette',
            field=models.PositiveSmallIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='kmeansrun', name='recommended_k_elbow',
            field=models.PositiveSmallIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='kmeansrun', name='selected_k',
            field=models.PositiveSmallIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='kmeansrun', name='external_metrics',
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name='kmeansrun', name='contingency_matrix',
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name='kmeansrun', name='cluster_category_association',
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name='kmeansrun', name='quality_warnings',
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name='kmeansrun', name='stability_metrics',
            field=models.JSONField(default=dict),
        ),
    ]
