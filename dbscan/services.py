import math
import statistics
import unicodedata
from collections import Counter
from decimal import Decimal

import numpy as np
import sklearn
from django.core.paginator import Paginator
from django.utils import timezone
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from datasets.equivalences import canonical_number
from datasets.model_validation import (
    build_change_summary,
    build_schema_profile,
    dataset_fingerprint,
    dataset_schema_fingerprint,
    training_config_fingerprint,
)
from datasets.services import filter_dataset_by_category

from .models import DBSCANRun


RESULT_PAGE_SIZE = 50
RANDOM_STATE = 42
MAX_SILHOUETTE_SAMPLES = 2000
MAX_CHART_POINTS = 2000
IDENTIFIER_NAMES = {
    'id',
    'identificador',
    'codigo',
    'folio',
    'matricula',
    'telefono',
    'celular',
    'cuenta',
}
NULL_VALUES = {'', 'nan', 'null', 'none'}

# Default hyperparameter suggestions shown in the UI
DEFAULT_EPSILON = 0.5
DEFAULT_MIN_SAMPLES = 5


class DBSCANTrainingError(Exception):
    """A user-correctable error detected before or during training."""


# ---------------------------------------------------------------------------
# Shared helpers (mirror of kmeans/services.py utilities to keep modules
# independent; they operate on the same concepts so the logic is identical)
# ---------------------------------------------------------------------------

def _normalize_name(value):
    decomposed = unicodedata.normalize('NFKD', str(value))
    plain = ''.join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return ' '.join(plain.casefold().replace('_', ' ').split())


def _is_identifier(column):
    words = set(_normalize_name(column).split())
    return bool(words.intersection(IDENTIFIER_NAMES))


def _is_null(value):
    return str(value).strip().casefold() in NULL_VALUES


def _as_number(value):
    if _is_null(value):
        return None
    try:
        canonical = canonical_number(value)
    except (TypeError, ValueError):
        raise ValueError from None
    number = float(Decimal(canonical)) if canonical is not None else None
    if number is not None and not math.isfinite(number):
        raise ValueError
    return number


def detect_numeric_columns(dataset, records):
    """Return numeric, non-constant features suitable for DBSCAN."""
    columns = []
    for column in dataset.columns:
        if _is_identifier(column):
            continue

        values = []
        null_count = 0
        numeric = True
        for record in records:
            try:
                value = _as_number(record.get(column, ''))
            except ValueError:
                numeric = False
                break
            if value is None:
                null_count += 1
            else:
                values.append(value)

        if numeric and values and len(set(values)) > 1:
            columns.append(
                {
                    'name': column,
                    'valid_count': len(values),
                    'null_count': null_count,
                    'unique_count': len(set(values)),
                }
            )
    return columns


def detect_categorical_columns(dataset, records):
    """Return repeated-value columns that can describe the generated clusters."""
    columns = []
    for column in dataset.columns:
        if _is_identifier(column):
            continue

        values = [
            str(record.get(column, '')).strip()
            for record in records
            if not _is_null(record.get(column, ''))
        ]
        normalized_values = {value.casefold() for value in values}
        unique_count = len(normalized_values)
        if len(values) >= 2 and 1 < unique_count < len(values):
            columns.append(
                {
                    'name': column,
                    'valid_count': len(values),
                    'null_count': len(records) - len(values),
                    'unique_count': unique_count,
                }
            )
    return columns


def _filtered_rows(dataset, requested_category, requested_category_column=None):
    category_filter = filter_dataset_by_category(
        dataset, requested_category, requested_category_column
    )
    selected_category = category_filter['selected_category']
    category_column = category_filter['category_column']

    rows = []
    for row_number, record in enumerate(dataset.records, start=1):
        if selected_category:
            category = str(record.get(category_column, '')).strip().casefold()
            if category != selected_category:
                continue
        rows.append((row_number, record))
    return category_filter, rows


def build_dbscan_training_setup(
    dataset, requested_category=None, requested_category_column=None
):
    if not dataset:
        return {
            'dbscan_numeric_columns': [],
            'dbscan_categorical_columns': [],
            'dbscan_sample_count': 0,
            'dbscan_can_train': False,
            'dbscan_default_epsilon': DEFAULT_EPSILON,
            'dbscan_default_min_samples': DEFAULT_MIN_SAMPLES,
        }

    category_filter, rows = _filtered_rows(
        dataset, requested_category, requested_category_column
    )
    records = [record for _, record in rows]
    numeric_columns = detect_numeric_columns(dataset, records)
    categorical_columns = detect_categorical_columns(dataset, records)
    return {
        'dbscan_numeric_columns': numeric_columns,
        'dbscan_categorical_columns': categorical_columns,
        'dbscan_sample_count': len(records),
        'dbscan_can_train': bool(numeric_columns) and len(records) >= 2,
        'dbscan_training_category': category_filter['selected_category'],
        'dbscan_training_category_label': category_filter['selected_category_label'],
        'category_column': category_filter['category_column'],
        'dbscan_default_epsilon': DEFAULT_EPSILON,
        'dbscan_default_min_samples': DEFAULT_MIN_SAMPLES,
    }


def _build_matrix(rows, selected_columns):
    matrix = []
    for _, record in rows:
        matrix.append(
            [_as_number(record.get(column, '')) for column in selected_columns]
        )

    imputed_values = {}
    for column_index, column in enumerate(selected_columns):
        available = [
            row[column_index]
            for row in matrix
            if row[column_index] is not None
        ]
        if not available:
            raise DBSCANTrainingError(
                f'La columna "{column}" no contiene valores numéricos válidos.'
            )
        median = float(statistics.median(available))
        missing_count = 0
        for row in matrix:
            if row[column_index] is None:
                row[column_index] = median
                missing_count += 1
        if missing_count:
            imputed_values[column] = {
                'count': missing_count,
                'median': round(median, 6),
            }
    return np.asarray(matrix, dtype=float), imputed_values


def _comparison_summary(rows, labels, comparison_column, cluster_labels):
    """Build a per-cluster comparison summary (noise cluster excluded)."""
    category_labels = {}
    # Only include valid clusters (not noise = -1)
    valid_clusters = sorted(c for c in cluster_labels if c != -1)
    cluster_counters = {cluster: Counter() for cluster in valid_clusters}
    valid_count = 0
    for (_, record), cluster in zip(rows, labels, strict=True):
        if cluster == -1:
            continue
        raw_value = record.get(comparison_column, '')
        if _is_null(raw_value):
            continue
        display_value = str(raw_value).strip()
        normalized_value = display_value.casefold()
        category_labels.setdefault(normalized_value, display_value)
        cluster_counters[cluster][normalized_value] += 1
        valid_count += 1

    summaries = []
    total_matches = 0
    for cluster in valid_clusters:
        counter = cluster_counters[cluster]
        compared_count = sum(counter.values())
        if not counter:
            predominant_category = 'Sin datos'
            predominant_count = 0
            match_percentage = None
        else:
            predominant_count = max(counter.values())
            predominant_values = sorted(
                (
                    category_labels[value]
                    for value, count in counter.items()
                    if count == predominant_count
                ),
                key=str.casefold,
            )
            predominant_category = ' / '.join(predominant_values)
            if len(predominant_values) > 1:
                predominant_category += ' (empate)'
            match_percentage = round(
                predominant_count * 100 / compared_count,
                2,
            )
            total_matches += predominant_count
        cluster_size = list(labels).count(cluster)
        summaries.append(
            {
                'cluster': cluster,
                'record_count': cluster_size,
                'compared_count': compared_count,
                'predominant_category': predominant_category,
                'predominant_count': predominant_count,
                'match_percentage': match_percentage,
            }
        )

    overall_match = (
        round(total_matches * 100 / valid_count, 2)
        if valid_count
        else None
    )
    return {
        'values': sorted(category_labels.values(), key=str.casefold),
        'summaries': summaries,
        'valid_count': valid_count,
        'overall_match_percentage': overall_match,
    }


def train_dbscan(
    dataset,
    selected_columns,
    epsilon,
    min_samples,
    requested_category=None,
    comparison_column='',
    requested_category_column=None,
    name='',
    topic='',
    description='',
    parent_run=None,
    save_immediately=True,
):
    category_filter, rows = _filtered_rows(
        dataset, requested_category, requested_category_column
    )
    records = [record for _, record in rows]
    allowed_columns = {
        column['name'] for column in detect_numeric_columns(dataset, records)
    }
    allowed_comparisons = {
        column['name']
        for column in detect_categorical_columns(dataset, records)
    }
    selected_columns = list(dict.fromkeys(selected_columns))

    if not selected_columns:
        raise DBSCANTrainingError('Selecciona al menos una columna.')
    if any(column not in allowed_columns for column in selected_columns):
        raise DBSCANTrainingError(
            'Una o más columnas no son compatibles con DBSCAN.'
        )
    if comparison_column and comparison_column not in allowed_comparisons:
        raise DBSCANTrainingError(
            'La columna de comparación no es categórica o no está disponible.'
        )
    if comparison_column and comparison_column in selected_columns:
        raise DBSCANTrainingError(
            'La columna de comparación no puede utilizarse para entrenar.'
        )
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise DBSCANTrainingError('El valor de ε (epsilon) debe ser mayor a 0.')
    if min_samples < 2:
        raise DBSCANTrainingError(
            'El mínimo de muestras debe ser al menos 2.'
        )
    if len(rows) < 2:
        raise DBSCANTrainingError(
            'Se requieren al menos 2 registros para ejecutar DBSCAN.'
        )

    matrix, imputed_values = _build_matrix(rows, selected_columns)

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)

    estimator = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean')
    try:
        raw_labels = estimator.fit_predict(scaled_matrix)
    except (TypeError, ValueError) as error:
        raise DBSCANTrainingError(
            'No fue posible ejecutar DBSCAN con los parámetros seleccionados.'
        ) from error

    labels = raw_labels.tolist()
    unique_labels = set(labels)
    valid_clusters = sorted(c for c in unique_labels if c != -1)
    cluster_count = len(valid_clusters)
    noise_count = labels.count(-1)

    if cluster_count == 0:
        raise DBSCANTrainingError(
            'DBSCAN no encontró ningún cluster con los parámetros actuales. '
            'Intenta aumentar ε o reducir el mínimo de muestras.'
        )

    # Re-label clusters as 1-based for display (noise stays as -1)
    label_mapping = {original: idx + 1 for idx, original in enumerate(valid_clusters)}
    display_labels = [
        label_mapping[label] if label != -1 else -1
        for label in labels
    ]
    display_valid_clusters = list(range(1, cluster_count + 1))

    assignments = []
    for (row_number, _), cluster in zip(rows, display_labels, strict=True):
        assignments.append(
            {
                'row_number': row_number,
                'cluster': cluster,
            }
        )

    cluster_sizes = {
        str(cluster): display_labels.count(cluster)
        for cluster in display_valid_clusters
    }
    if noise_count > 0:
        cluster_sizes['-1'] = noise_count

    # Silhouette score requires at least 2 distinct non-noise labels
    non_noise_mask = [label != -1 for label in display_labels]
    non_noise_labels = [label for label, keep in zip(display_labels, non_noise_mask) if keep]
    non_noise_matrix = scaled_matrix[non_noise_mask]
    silhouette_sample_count = 0
    score = None
    if len(set(non_noise_labels)) > 1 and len(non_noise_labels) > 1:
        silhouette_sample_count = min(
            len(non_noise_labels),
            MAX_SILHOUETTE_SAMPLES,
        )
        score = float(
            silhouette_score(
                non_noise_matrix,
                non_noise_labels,
                sample_size=(
                    silhouette_sample_count
                    if silhouette_sample_count < len(non_noise_labels)
                    else None
                ),
                random_state=RANDOM_STATE,
            )
        )

    comparison = (
        _comparison_summary(
            rows, display_labels, comparison_column, display_valid_clusters
        )
        if comparison_column
        else {
            'values': [],
            'summaries': [],
            'valid_count': 0,
            'overall_match_percentage': None,
        }
    )

    change_summary = {}
    if parent_run:
        change_summary = build_change_summary(
            parent_run, assignments, len(rows),
            {
                'previous': {
                    'silhouette': parent_run.silhouette,
                    'noise_count': parent_run.noise_count,
                    'cluster_count': parent_run.cluster_count,
                },
                'current': {
                    'silhouette': score, 'noise_count': noise_count,
                    'cluster_count': cluster_count,
                },
            },
        )
    core_indices = getattr(estimator, 'core_sample_indices_', np.asarray([], dtype=int))
    dataset.dbscan_runs.filter(is_saved=False).delete()
    return DBSCANRun.objects.create(
        dataset=dataset,
        dataset_fingerprint=dataset_fingerprint(dataset),
        dataset_source_name=dataset.source_name,
        selected_columns=selected_columns,
        assignments=assignments,
        cluster_sizes=cluster_sizes,
        imputed_values=imputed_values,
        sample_count=len(rows),
        cluster_count=cluster_count,
        noise_count=noise_count,
        epsilon=epsilon,
        min_samples=min_samples,
        silhouette=score,
        silhouette_sample_count=silhouette_sample_count,
        comparison_column=comparison_column,
        comparison_values=comparison['values'],
        cluster_comparison=comparison['summaries'],
        comparison_valid_count=comparison['valid_count'],
        overall_match_percentage=comparison['overall_match_percentage'],
        category_filter=category_filter['selected_category'],
        category_label=category_filter['selected_category_label'],
        category_column=category_filter['category_column'] or '',
        name=(name or (parent_run.name if parent_run else '') or 'Modelo DBSCAN'),
        topic=topic or (parent_run.topic if parent_run else ''),
        description=description or (parent_run.description if parent_run else ''),
        parent_run=parent_run,
        version=(parent_run.version + 1 if parent_run else 1),
        source_row_count=len(dataset.records),
        new_record_count=change_summary.get('new_record_count', 0),
        dataset_schema_fingerprint=dataset_schema_fingerprint(dataset),
        training_config_fingerprint=training_config_fingerprint(
            'dbscan', selected_columns,
            category_filter=category_filter['selected_category'],
            comparison_column=comparison_column,
            epsilon=epsilon, min_samples=min_samples,
            category_column=category_filter['category_column'] or '',
        ),
        schema_profile=build_schema_profile(dataset),
        preprocessing_state={
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'variance': scaler.var_.tolist(),
            'imputed_values': imputed_values,
        },
        estimator_state={
            'core_sample_indices': core_indices.tolist(),
            'components': getattr(
                estimator, 'components_', np.empty((0, len(selected_columns)))
            ).tolist(),
            'parameters': {'epsilon': epsilon, 'min_samples': min_samples},
            'strategy': 'full_refit',
        },
        library_versions={'scikit_learn': sklearn.__version__, 'numpy': np.__version__},
        change_summary=change_summary,
        is_saved=save_immediately,
        saved_at=(timezone.now() if save_immediately else None),
    )


def clear_dbscan_runs(dataset):
    """Discard provisional DBSCAN results while preserving saved models."""
    dataset.dbscan_runs.filter(
        dataset_fingerprint=dataset_fingerprint(dataset),
        is_saved=False,
    ).delete()


def _silhouette_interpretation(score):
    if score is None:
        return {
            'label': 'No disponible',
            'description': (
                'Se necesita más de un cluster con registros no considerados ruido.'
            ),
            'class': 'secondary',
        }
    if score >= 0.71:
        return {
            'label': 'Separación fuerte',
            'description': 'Los clusters están claramente separados entre sí.',
            'class': 'success',
        }
    if score >= 0.51:
        return {
            'label': 'Separación adecuada',
            'description': 'Los clusters presentan una estructura diferenciada.',
            'class': 'primary',
        }
    if score >= 0.26:
        return {
            'label': 'Separación débil',
            'description': 'Existe agrupamiento, aunque algunos registros se solapan.',
            'class': 'warning',
        }
    if score >= 0:
        return {
            'label': 'Clusters poco definidos',
            'description': 'La separación es baja y debe interpretarse con cautela.',
            'class': 'warning',
        }
    return {
        'label': 'Agrupamiento deficiente',
        'description': 'La estructura encontrada presenta asignaciones poco coherentes.',
        'class': 'danger',
    }


def _result_matrix(dataset, run):
    rows = [
        (
            assignment['row_number'],
            dataset.records[assignment['row_number'] - 1],
        )
        for assignment in run.assignments
    ]
    matrix, _ = _build_matrix(rows, run.selected_columns)
    return matrix


def _chart_context(run, matrix):
    feature_count = len(run.selected_columns)
    if feature_count == 1:
        coordinates = np.column_stack((matrix[:, 0], np.zeros(len(matrix))))
        x_label = run.selected_columns[0]
        y_label = ''
        method = 'Valores originales de la variable seleccionada.'
        projected = False
    elif feature_count == 2:
        coordinates = matrix[:, :2]
        x_label, y_label = run.selected_columns
        method = 'Valores originales de las dos variables seleccionadas.'
        projected = False
    else:
        scaled_matrix = StandardScaler().fit_transform(matrix)
        pca = PCA(n_components=2)
        coordinates = pca.fit_transform(scaled_matrix)
        explained = pca.explained_variance_ratio_ * 100
        x_label = f'Componente principal 1 ({explained[0]:.1f}%)'
        y_label = f'Componente principal 2 ({explained[1]:.1f}%)'
        method = (
            'Proyección PCA calculada con las variables estandarizadas; conserva '
            f'{explained.sum():.1f}% de la variación.'
        )
        projected = True

    displayed_indices = np.arange(len(coordinates))
    if len(displayed_indices) > MAX_CHART_POINTS:
        displayed_indices = np.linspace(
            0,
            len(displayed_indices) - 1,
            MAX_CHART_POINTS,
            dtype=int,
        )

    groups = []
    for cluster in [-1, *range(1, run.cluster_count + 1)]:
        points = []
        for index in displayed_indices:
            assignment = run.assignments[int(index)]
            if assignment['cluster'] != cluster:
                continue
            points.append(
                {
                    'x': round(float(coordinates[index, 0]), 6),
                    'y': round(float(coordinates[index, 1]), 6),
                    'row': assignment['row_number'],
                }
            )
        if points:
            groups.append(
                {
                    'cluster': cluster,
                    'label': 'Ruido' if cluster == -1 else f'Cluster {cluster}',
                    'points': points,
                }
            )

    return {
        'x_label': x_label,
        'y_label': y_label,
        'method': method,
        'projected': projected,
        'displayed_count': len(displayed_indices),
        'total_count': len(coordinates),
        'groups': groups,
    }


def build_dbscan_results_context(dataset, page_number=None):
    run = (
        dataset.dbscan_runs.filter(
            dataset_fingerprint=dataset_fingerprint(dataset)
        ).first()
        if dataset
        else None
    )
    if not run:
        return {
            'dbscan_run': None,
            'dbscan_result_page': None,
            'dbscan_result_rows': [],
            'dbscan_cluster_summaries': [],
        }

    paginator = Paginator(run.assignments, RESULT_PAGE_SIZE)
    page = paginator.get_page(page_number)
    result_rows = []
    for assignment in page.object_list:
        row_index = assignment.get('row_number', 0) - 1
        if row_index < 0 or row_index >= len(dataset.records):
            continue
        record = dataset.records[row_index]
        result_rows.append(
            {
                **assignment,
                'values': [
                    record.get(column, '') for column in run.selected_columns
                ],
                'comparison_value': (
                    record.get(run.comparison_column, '')
                    if run.comparison_column
                    else ''
                ),
            }
        )

    # Summaries for valid clusters only (noise shown separately)
    valid_clusters = range(1, run.cluster_count + 1)
    cluster_summaries = []
    for cluster in valid_clusters:
        row_numbers = [
            assignment['row_number']
            for assignment in run.assignments
            if assignment['cluster'] == cluster
        ]
        cluster_summaries.append(
            {
                'cluster': cluster,
                'size': len(row_numbers),
                'row_numbers': row_numbers[:50],
                'remaining_rows': max(len(row_numbers) - 50, 0),
            }
        )

    matrix = _result_matrix(dataset, run)
    return {
        'dbscan_run': run,
        'dbscan_quality': _silhouette_interpretation(run.silhouette),
        'dbscan_chart': _chart_context(run, matrix),
        'dbscan_result_page': page,
        'dbscan_result_rows': result_rows,
        'dbscan_result_page_range': paginator.get_elided_page_range(
            page.number, on_each_side=2, on_ends=1
        ),
        'dbscan_cluster_summaries': cluster_summaries,
    }
