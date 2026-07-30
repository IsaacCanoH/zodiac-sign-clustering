from django.contrib import admin

from .models import KMeansRun


@admin.register(KMeansRun)
class KMeansRunAdmin(admin.ModelAdmin):
    list_display = (
        'dataset',
        'cluster_count',
        'sample_count',
        'silhouette',
        'created_at',
    )
    readonly_fields = ('created_at',)
