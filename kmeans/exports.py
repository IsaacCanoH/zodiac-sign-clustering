from django.db import transaction
from django.utils import timezone

from datasets.model_validation import (
    dataset_fingerprint,
    require_integer,
    require_list,
    require_mapping,
    require_number,
    validate_assignments,
    validate_cluster_sizes,
    validate_dataset_identity,
    validate_optional_metadata,
    validate_selected_columns,
)

from .models import KMeansRun

MODEL_TYPE = 'kmeans'
FORMAT_VERSION = '2.0'


def export_kmeans_run(run):
    """Serialize a K-Means run together with its dataset identity."""
    return {
        'type': MODEL_TYPE,
        'version': FORMAT_VERSION,
        'dataset_fingerprint': run.dataset_fingerprint,
        'dataset_source_name': run.dataset_source_name,
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
    """Validate and restore a K-Means run for the exact source dataset."""
    require_mapping(data)
    if data.get('type') != MODEL_TYPE:
        raise ValueError('El archivo no corresponde a un modelo K-Means.')
    if data.get('version') != FORMAT_VERSION:
        raise ValueError(
            'Versión de archivo no compatible. Exporta nuevamente el modelo '
            'desde la versión actual de la aplicación.'
        )

    validate_dataset_identity(dataset, data)
    selected_columns = validate_selected_columns(dataset, data.get('selected_columns'))
    sample_count = require_integer(data.get('sample_count'), 'sample_count', 2)
    cluster_count = require_integer(data.get('cluster_count'), 'cluster_count', 2)
    if cluster_count >= sample_count:
        raise ValueError('K-Means requiere menos clusters que registros.')
    assignments, labels = validate_assignments(
        dataset,
        data.get('assignments'),
        sample_count,
        cluster_count,
        allow_noise=False,
    )
    cluster_sizes = validate_cluster_sizes(data.get('cluster_sizes'), labels)

    centroids = require_list(data.get('centroids'), 'centroids')
    if len(centroids) != cluster_count:
        raise ValueError('La cantidad de centroides no coincide con los clusters.')
    for expected_cluster, centroid in enumerate(centroids, start=1):
        require_mapping(centroid, 'Cada centroide debe ser un objeto JSON.')
        if centroid.get('cluster') != expected_cluster:
            raise ValueError('Los centroides contienen una etiqueta inválida.')
        values = require_list(centroid.get('values'), 'centroid.values')
        if len(values) != len(selected_columns):
            raise ValueError(
                'Las dimensiones de los centroides no coinciden con las columnas.'
            )
        for value in values:
            require_number(value, 'centroid.values')

    inertia = require_number(data.get('inertia'), 'inertia', minimum=0)
    silhouette = require_number(
        data.get('silhouette'),
        'silhouette',
        minimum=-1,
        maximum=1,
        nullable=True,
    )
    silhouette_sample_count = require_integer(
        data.get('silhouette_sample_count', 0),
        'silhouette_sample_count',
    )
    if silhouette_sample_count > sample_count:
        raise ValueError('La muestra de silueta supera los registros del modelo.')
    validate_optional_metadata(dataset, data, selected_columns)

    with transaction.atomic():
        return KMeansRun.objects.create(
            dataset=dataset,
            dataset_fingerprint=dataset_fingerprint(dataset),
            dataset_source_name=dataset.source_name,
            cluster_count=cluster_count,
            selected_columns=selected_columns,
            assignments=assignments,
            centroids=centroids,
            cluster_sizes=cluster_sizes,
            imputed_values=data.get('imputed_values', {}),
            sample_count=sample_count,
            inertia=inertia,
            silhouette=silhouette,
            silhouette_sample_count=silhouette_sample_count,
            comparison_column=data.get('comparison_column', ''),
            comparison_values=data.get('comparison_values', []),
            cluster_comparison=data.get('cluster_comparison', []),
            comparison_valid_count=data.get('comparison_valid_count', 0),
            overall_match_percentage=data.get('overall_match_percentage'),
            category_filter=data.get('category_filter', ''),
            category_label=data.get('category_label', ''),
        )
