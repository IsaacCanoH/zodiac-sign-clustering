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
