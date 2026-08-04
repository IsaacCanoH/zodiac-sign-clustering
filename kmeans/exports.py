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
        'category_column': run.category_column,
        'schema_profile': run.schema_profile,
        'exported_at': timezone.now().isoformat(),
        'name': run.name,
        'topic': run.topic,
        'description': run.description,
        'model_version': run.version,
        'source_row_count': run.source_row_count,
        'new_record_count': run.new_record_count,
        'dataset_schema_fingerprint': run.dataset_schema_fingerprint,
        'training_config_fingerprint': run.training_config_fingerprint,
        'preprocessing_state': run.preprocessing_state,
        'estimator_state': run.estimator_state,
        'library_versions': run.library_versions,
        'change_summary': run.change_summary,
        'trained_at': run.estimator_state.get(
            'trained_at', run.created_at.isoformat()
        ),
        'results_by_k': run.results_by_k,
        'recommended_k_silhouette': run.recommended_k_silhouette,
        'recommended_k_elbow': run.recommended_k_elbow,
        'selected_k': run.selected_k,
        'external_metrics': run.external_metrics,
        'contingency_matrix': run.contingency_matrix,
        'cluster_category_association': run.cluster_category_association,
        'quality_warnings': run.quality_warnings,
        'stability_metrics': run.stability_metrics,
    }


def import_kmeans_run(dataset, data, *, allow_compatible=False):
    """Validate and restore a K-Means run for the exact source dataset."""
    require_mapping(data)
    if data.get('type') != MODEL_TYPE:
        raise ValueError('El archivo no corresponde a un modelo K-Means.')
    if data.get('version') != FORMAT_VERSION:
        raise ValueError(
            'Versión de archivo no compatible. Exporta nuevamente el modelo '
            'desde la versión actual de la aplicación.'
        )

    exact_dataset = validate_dataset_identity(
        dataset, data, allow_compatible=allow_compatible
    )
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
    estimator_state = require_mapping(
        data.get('estimator_state', {}),
        'El estado del estimador debe ser un objeto.',
    ).copy()
    if data.get('trained_at'):
        estimator_state['trained_at'] = data['trained_at']

    with transaction.atomic():
        return KMeansRun.objects.create(
            dataset=dataset,
            dataset_fingerprint=(
                dataset_fingerprint(dataset)
                if exact_dataset else data['dataset_fingerprint']
            ),
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
            category_column=data.get('category_column', ''),
            schema_profile=data.get('schema_profile', {}),
            name=str(data.get('name', ''))[:150],
            topic=str(data.get('topic', ''))[:150],
            description=str(data.get('description', '')),
            version=require_integer(data.get('model_version', 1), 'model_version', 1),
            source_row_count=require_integer(
                data.get('source_row_count', len(dataset.records)),
                'source_row_count',
            ),
            new_record_count=require_integer(
                data.get('new_record_count', 0), 'new_record_count'
            ),
            dataset_schema_fingerprint=str(
                data.get('dataset_schema_fingerprint', '')
            )[:64],
            training_config_fingerprint=str(
                data.get('training_config_fingerprint', '')
            )[:64],
            preprocessing_state=require_mapping(
                data.get('preprocessing_state', {}),
                'El estado de preprocesamiento debe ser un objeto.',
            ),
            estimator_state=estimator_state,
            library_versions=require_mapping(
                data.get('library_versions', {}),
                'Las versiones deben ser un objeto.',
            ),
            change_summary=require_mapping(
                data.get('change_summary', {}),
                'El resumen de cambios debe ser un objeto.',
            ),
            results_by_k=require_list(data.get('results_by_k', []), 'results_by_k'),
            recommended_k_silhouette=data.get('recommended_k_silhouette'),
            recommended_k_elbow=data.get('recommended_k_elbow'),
            selected_k=data.get('selected_k', cluster_count),
            external_metrics=require_mapping(
                data.get('external_metrics', {}),
                'Las métricas externas deben ser un objeto.',
            ),
            contingency_matrix=require_mapping(
                data.get('contingency_matrix', {}),
                'La matriz de contingencia debe ser un objeto.',
            ),
            cluster_category_association=require_list(
                data.get('cluster_category_association', []),
                'cluster_category_association',
            ),
            quality_warnings=require_list(
                data.get('quality_warnings', []), 'quality_warnings'
            ),
            stability_metrics=require_mapping(
                data.get('stability_metrics', {}),
                'Las métricas de estabilidad deben ser un objeto.',
            ),
            is_saved=True,
            saved_at=timezone.now(),
        )
