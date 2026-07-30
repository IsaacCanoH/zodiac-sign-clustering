from django.db import models

from datasets.models import Dataset


class KMeansRun(models.Model):
    """Persist the latest information produced by a K-Means training run."""

    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='kmeans_runs',
    )
    cluster_count = models.PositiveSmallIntegerField()
    selected_columns = models.JSONField()
    assignments = models.JSONField()
    centroids = models.JSONField()
    cluster_sizes = models.JSONField()
    imputed_values = models.JSONField(default=dict)
    sample_count = models.PositiveIntegerField()
    inertia = models.FloatField()
    silhouette = models.FloatField(null=True, blank=True)
    silhouette_sample_count = models.PositiveIntegerField(default=0)
    comparison_column = models.CharField(max_length=255, blank=True)
    comparison_values = models.JSONField(default=list)
    cluster_comparison = models.JSONField(default=list)
    comparison_valid_count = models.PositiveIntegerField(default=0)
    overall_match_percentage = models.FloatField(null=True, blank=True)
    category_filter = models.CharField(max_length=255, blank=True)
    category_label = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('-created_at',)

    def __str__(self):
        return f'K-Means ({self.cluster_count} grupos) - {self.created_at:%d/%m/%Y}'
