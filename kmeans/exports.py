import json

from django.utils import timezone

from .models import KMeansRun

MODEL_TYPE = 'kmeans'
FORMAT_VERSION = '1.0'


def export_kmeans_run(run):
    """Serialize a KMeansRun to a JSON-serializable dict."""
    return {
        'type': MODEL_TYPE,
        'version': FORMAT_VERSION,
        'cluster_count': run.cluster_count,
        'selected_columns': run.selected_columns,
        'assignments': run.assignments,
        'centroids': run.centroids,
        'cluster_sizes': run.cluster_sizes,
        'imputed_values': run.imputed_values,
        'sample_count': run.sample_count,
        'inertia': run.inertia,
        'silhouette': run.silhouette,
        'silhouette_sample_count': run.silhouette_sample_count,
        'comparison_column': run.comparison_column,
        'comparison_values': run.comparison_values,
        'cluster_comparison': run.cluster_comparison,
        'comparison_valid_count': run.comparison_valid_count,
        'overall_match_percentage': run.overall_match_percentage,
        'category_filter': run.category_filter,
        'category_label': run.category_label,
        'exported_at': timezone.now().isoformat(),
    }


def import_kmeans_run(dataset, data):
    """Re-create a KMeansRun from a previously exported dict."""
    if data.get('type') != MODEL_TYPE:
        raise ValueError('El archivo no corresponde a un modelo K-Means.')
    if data.get('version') != FORMAT_VERSION:
        raise ValueError('Versión de archivo no compatible.')
    return KMeansRun.objects.create(
        dataset=dataset,
        cluster_count=data['cluster_count'],
        selected_columns=data['selected_columns'],
        assignments=data['assignments'],
        centroids=data['centroids'],
        cluster_sizes=data['cluster_sizes'],
        imputed_values=data.get('imputed_values', {}),
        sample_count=data['sample_count'],
        inertia=data['inertia'],
        silhouette=data.get('silhouette'),
        silhouette_sample_count=data.get('silhouette_sample_count', 0),
        comparison_column=data.get('comparison_column', ''),
        comparison_values=data.get('comparison_values', []),
        cluster_comparison=data.get('cluster_comparison', []),
        comparison_valid_count=data.get('comparison_valid_count', 0),
        overall_match_percentage=data.get('overall_match_percentage'),
        category_filter=data.get('category_filter', ''),
        category_label=data.get('category_label', ''),
    )
