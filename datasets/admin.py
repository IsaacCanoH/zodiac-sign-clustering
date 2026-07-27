from django.contrib import admin

from .models import (
    Dataset,
    DatasetEquivalenceApplication,
    EquivalenceConfiguration,
)


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('source_name', 'uploaded_at')
    readonly_fields = ('uploaded_at',)


@admin.register(EquivalenceConfiguration)
class EquivalenceConfigurationAdmin(admin.ModelAdmin):
    list_display = ('name', 'source_dataset_name', 'updated_at')
    search_fields = ('name', 'source_dataset_name')
    readonly_fields = ('created_at', 'updated_at')


@admin.register(DatasetEquivalenceApplication)
class DatasetEquivalenceApplicationAdmin(admin.ModelAdmin):
    list_display = ('configuration', 'dataset', 'updated_at')
    readonly_fields = ('created_at', 'updated_at')
