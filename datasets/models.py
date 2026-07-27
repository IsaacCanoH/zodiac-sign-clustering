from django.db import models


class Dataset(models.Model):
    """The single dataset currently available to the application."""

    source_name = models.CharField(max_length=255)
    columns = models.JSONField()
    records = models.JSONField()
    uploaded_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'dashboard_dataset'

    @property
    def row_count(self):
        return len(self.records)

    @property
    def column_count(self):
        return len(self.columns)

    def __str__(self):
        return self.source_name


class EquivalenceConfiguration(models.Model):
    """Reusable mapping from quantitative values to qualitative labels."""

    name = models.CharField(max_length=150)
    mapping = models.JSONField()
    possible_values = models.JSONField()
    conversion_type = models.CharField(
        max_length=30, default='quantitative_to_qualitative'
    )
    source_dataset_name = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ('name',)

    def __str__(self):
        return self.name


class DatasetEquivalenceApplication(models.Model):
    """Columns where a reusable configuration is active for a dataset."""

    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='equivalence_applications',
    )
    configuration = models.ForeignKey(
        EquivalenceConfiguration,
        on_delete=models.CASCADE,
        related_name='applications',
    )
    columns = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=('dataset', 'configuration'),
                name='unique_dataset_equivalence_application',
            )
        ]
