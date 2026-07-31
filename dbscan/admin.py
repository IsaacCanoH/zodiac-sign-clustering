from django.contrib import admin

from .models import DBSCANRun


@admin.register(DBSCANRun)
class DBSCANRunAdmin(admin.ModelAdmin):
    list_display = ('dataset', 'cluster_count', 'noise_count', 'epsilon', 'min_samples', 'silhouette', 'created_at')
    list_filter = ('dataset',)
    readonly_fields = ('created_at',)
