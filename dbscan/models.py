from django.db import models
from django.utils import timezone

from datasets.models import Dataset


class DBSCANRun(models.Model):
    """Persist the latest information produced by a DBSCAN training run."""

    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='dbscan_runs',
    )
    dataset_fingerprint = models.CharField(max_length=64, db_index=True)
    dataset_source_name = models.CharField(max_length=255)
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
    # Dimensionality Reduction
    use_pca = models.BooleanField(default=False)
    pca_components = models.PositiveSmallIntegerField(null=True, blank=True)

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
    category_column = models.CharField(max_length=255, blank=True)
    schema_profile = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    activated_at = models.DateTimeField(default=timezone.now)
    name = models.CharField(max_length=150, blank=True)
    topic = models.CharField(max_length=150, blank=True)
    description = models.TextField(blank=True)
    parent_run = models.ForeignKey(
        'self', null=True, blank=True, on_delete=models.SET_NULL,
        related_name='retrained_versions',
    )
    version = models.PositiveIntegerField(default=1)
    source_row_count = models.PositiveIntegerField(default=0)
    new_record_count = models.PositiveIntegerField(default=0)
    dataset_schema_fingerprint = models.CharField(
        max_length=64, blank=True, db_index=True
    )
    training_config_fingerprint = models.CharField(
        max_length=64, blank=True, db_index=True
    )
    preprocessing_state = models.JSONField(default=dict)
    estimator_state = models.JSONField(default=dict)
    library_versions = models.JSONField(default=dict)
    change_summary = models.JSONField(default=dict)
    is_saved = models.BooleanField(default=True, db_index=True)
    saved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ('-activated_at', '-created_at')

    def __str__(self):
        return (
            f'DBSCAN (ε={self.epsilon}, min={self.min_samples}, '
            f'{self.cluster_count} clusters) - {self.created_at:%d/%m/%Y}'
        )
