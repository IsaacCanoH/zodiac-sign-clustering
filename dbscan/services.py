import math
import statistics
import unicodedata
from collections import Counter
from decimal import Decimal

import numpy as np
from django.core.paginator import Paginator
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from datasets.equivalences import canonical_number
from datasets.services import filter_dataset_by_category

from .models import DBSCANRun


RESULT_PAGE_SIZE = 50
RANDOM_STATE = 42
MAX_SILHOUETTE_SAMPLES = 2000
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


def _filtered_rows(dataset, requested_category):
    category_filter = filter_dataset_by_category(dataset, requested_category)
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


def build_dbscan_training_setup(dataset, requested_category=None):
    if not dataset:
        return {
            'dbscan_numeric_columns': [],
            'dbscan_categorical_columns': [],
            'dbscan_sample_count': 0,
            'dbscan_can_train': False,
            'dbscan_default_epsilon': DEFAULT_EPSILON,
            'dbscan_default_min_samples': DEFAULT_MIN_SAMPLES,
        }

    category_filter, rows = _filtered_rows(dataset, requested_category)
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
):
    category_filter, rows = _filtered_rows(dataset, requested_category)
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
    if epsilon <= 0:
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
    raw_labels = estimator.fit_predict(scaled_matrix)

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
    silhouette_sample_count = min(len(non_noise_labels), MAX_SILHOUETTE_SAMPLES)
    score = None
    if len(set(non_noise_labels)) > 1 and len(non_noise_labels) > 1:
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

    return DBSCANRun.objects.create(
        dataset=dataset,
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
    )


def clear_dbscan_runs(dataset):
    """Remove every DBSCAN result associated with the active dataset."""
    dataset.dbscan_runs.all().delete()


def build_dbscan_results_context(dataset, page_number=None):
    run = dataset.dbscan_runs.first() if dataset else None
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
        record = dataset.records[assignment['row_number'] - 1]
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

    return {
        'dbscan_run': run,
        'dbscan_result_page': page,
        'dbscan_result_rows': result_rows,
        'dbscan_result_page_range': paginator.get_elided_page_range(
            page.number, on_each_side=2, on_ends=1
        ),
        'dbscan_cluster_summaries': cluster_summaries,
    }
