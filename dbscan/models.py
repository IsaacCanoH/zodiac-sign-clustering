from django.db import models

from datasets.models import Dataset


class DBSCANRun(models.Model):
    """Persist the latest information produced by a DBSCAN training run."""

    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='dbscan_runs',
    )
    selected_columns = models.JSONField()
    assignments = models.JSONField()
    cluster_sizes = models.JSONField()
    imputed_values = models.JSONField(default=dict)
    sample_count = models.PositiveIntegerField()
    # Clusters found (excluding noise cluster -1)
    cluster_count = models.PositiveSmallIntegerField()
    # Points labelled as noise (cluster = -1)
    noise_count = models.PositiveIntegerField(default=0)
    # DBSCAN hyperparameters
    epsilon = models.FloatField()
    min_samples = models.PositiveSmallIntegerField()
    silhouette = models.FloatField(null=True, blank=True)
    silhouette_sample_count = models.PositiveIntegerField(default=0)
    # Optional comparison column
    comparison_column = models.CharField(max_length=255, blank=True)
    comparison_values = models.JSONField(default=list)
    cluster_comparison = models.JSONField(default=list)
    comparison_valid_count = models.PositiveIntegerField(default=0)
    overall_match_percentage = models.FloatField(null=True, blank=True)
    # Category filter applied before training
    category_filter = models.CharField(max_length=255, blank=True)
    category_label = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ('-created_at',)

    def __str__(self):
        return (
            f'DBSCAN (ε={self.epsilon}, min={self.min_samples}, '
            f'{self.cluster_count} clusters) - {self.created_at:%d/%m/%Y}'
        )
