from django.db import models
from django.utils import timezone

from datasets.models import Dataset


class HierarchicalRun(models.Model):
    """Persist the latest information produced by a Hierarchical Clustering run."""

    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='hierarchical_runs',
    )
    dataset_fingerprint = models.CharField(max_length=64, default='', db_index=True)
    dataset_source_name = models.CharField(max_length=255, default='')

    # Algorithm hyperparameters
    n_clusters = models.PositiveSmallIntegerField(default=3)
    linkage = models.CharField(max_length=50, default='ward')
    affinity = models.CharField(max_length=50, default='euclidean')
    scaling_method = models.CharField(max_length=50, default='standard')

    # Training results
    selected_columns = models.JSONField(default=list)
    assignments = models.JSONField(default=list)
    cluster_sizes = models.JSONField(default=dict)
    imputed_values = models.JSONField(default=dict)
    sample_count = models.PositiveIntegerField(default=0)
    cluster_count = models.PositiveSmallIntegerField(default=0)
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

    # Metadata for saving
    name = models.CharField(max_length=150, blank=True)
    topic = models.CharField(max_length=150, blank=True)
    description = models.TextField(blank=True)

    # Versioning / retraining lineage
    parent_run = models.ForeignKey(
        'self', null=True, blank=True, on_delete=models.SET_NULL,
        related_name='retrained_versions',
    )
    version = models.PositiveIntegerField(default=1)
    source_row_count = models.PositiveIntegerField(default=0)
    new_record_count = models.PositiveIntegerField(default=0)

    # Fingerprints for compatibility checks
    dataset_schema_fingerprint = models.CharField(
        max_length=64, blank=True, db_index=True,
    )
    training_config_fingerprint = models.CharField(
        max_length=64, blank=True, db_index=True,
    )

    # Reproducibility state
    preprocessing_state = models.JSONField(default=dict)
    estimator_state = models.JSONField(default=dict)
    library_versions = models.JSONField(default=dict)
    change_summary = models.JSONField(default=dict)

    # Saved-model lifecycle
    is_saved = models.BooleanField(default=True, db_index=True)
    saved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ('-activated_at', '-created_at')

    def __str__(self):
        return (
            f'Jerárquico ({self.n_clusters} clusters, {self.linkage}) '
            f'- {self.created_at:%d/%m/%Y}'
        )