from django.db import models
from datasets.models import Dataset

class HierarchicalRun(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='hierarchical_runs')
    
    n_clusters = models.IntegerField(default=3)
    linkage = models.CharField(max_length=50, default='ward')
    affinity = models.CharField(max_length=50, default='euclidean')
    scaling_method = models.CharField(max_length=50, default='standard')
    selected_columns = models.JSONField(default=list)
    
    # Nuevos campos para igualar a K-Means/DBSCAN
    assignments = models.JSONField(default=list)
    cluster_sizes = models.JSONField(default=dict)
    
    comparison_column = models.CharField(max_length=255, null=True, blank=True)
    comparison_values = models.JSONField(default=list)
    cluster_comparison = models.JSONField(default=list)
    comparison_valid_count = models.IntegerField(default=0)
    overall_match_percentage = models.FloatField(null=True, blank=True)
    
    cluster_count = models.IntegerField(default=0)
    sample_count = models.IntegerField(default=0)
    silhouette_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']