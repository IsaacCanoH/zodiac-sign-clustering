import math
import statistics
import unicodedata
from collections import Counter
from decimal import Decimal

import numpy as np
from django.core.paginator import Paginator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from datasets.equivalences import canonical_number
from datasets.services import filter_dataset_by_category

from .models import KMeansRun


MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
RESULT_PAGE_SIZE = 25
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


class KMeansTrainingError(Exception):
    """A user-correctable error detected before or during training."""


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
    """Return numeric, non-constant features suitable for K-Means."""
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


def build_training_setup(dataset, requested_category=None):
    if not dataset:
        return {
            'numeric_columns': [],
            'categorical_columns': [],
            'sample_count': 0,
            'max_clusters': 0,
            'can_train': False,
        }

    category_filter, rows = _filtered_rows(dataset, requested_category)
    records = [record for _, record in rows]
    numeric_columns = detect_numeric_columns(dataset, records)
    categorical_columns = detect_categorical_columns(dataset, records)
    max_clusters = min(MAX_CLUSTERS, max(len(records) - 1, 0))
    return {
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'sample_count': len(records),
        'max_clusters': max_clusters,
        'cluster_options': range(MIN_CLUSTERS, max_clusters + 1),
        'can_train': bool(numeric_columns) and max_clusters >= MIN_CLUSTERS,
        'training_category': category_filter['selected_category'],
        'training_category_label': category_filter['selected_category_label'],
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
            raise KMeansTrainingError(
                f'La columna “{column}” no contiene valores numéricos válidos.'
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


def _stable_cluster_labels(labels, centers):
    ordered_clusters = sorted(
        range(len(centers)),
        key=lambda cluster: tuple(float(value) for value in centers[cluster]),
    )
    label_mapping = {
        original_label: stable_label
        for stable_label, original_label in enumerate(ordered_clusters, start=1)
    }
    stable_labels = [label_mapping[int(label)] for label in labels]
    stable_centers = [
        centers[original_label] for original_label in ordered_clusters
    ]
    return stable_labels, stable_centers


def _comparison_summary(rows, labels, comparison_column, cluster_count):
    category_labels = {}
    cluster_counters = {
        cluster: Counter() for cluster in range(1, cluster_count + 1)
    }
    valid_count = 0
    for (_, record), cluster in zip(rows, labels, strict=True):
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
    for cluster in range(1, cluster_count + 1):
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
        summaries.append(
            {
                'cluster': cluster,
                'record_count': labels.count(cluster),
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


def train_kmeans(
    dataset,
    selected_columns,
    cluster_count,
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
        raise KMeansTrainingError('Selecciona al menos una columna.')
    if any(column not in allowed_columns for column in selected_columns):
        raise KMeansTrainingError(
            'Una o más columnas no son compatibles con K-Means.'
        )
    if comparison_column and comparison_column not in allowed_comparisons:
        raise KMeansTrainingError(
            'La columna de comparación no es categórica o no está disponible.'
        )
    if comparison_column and comparison_column in selected_columns:
        raise KMeansTrainingError(
            'La columna de comparación no puede utilizarse para entrenar.'
        )
    if cluster_count < MIN_CLUSTERS:
        raise KMeansTrainingError('Selecciona al menos 2 grupos.')
    if cluster_count >= len(rows):
        raise KMeansTrainingError(
            'La cantidad de grupos debe ser menor que la cantidad de registros.'
        )

    matrix, imputed_values = _build_matrix(rows, selected_columns)
    if len(np.unique(matrix, axis=0)) < cluster_count:
        raise KMeansTrainingError(
            'No existen suficientes combinaciones diferentes para formar '
            f'{cluster_count} grupos.'
        )

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)
    estimator = KMeans(
        n_clusters=cluster_count,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=RANDOM_STATE,
        algorithm='lloyd',
    )
    labels = estimator.fit_predict(scaled_matrix)
    original_centers = scaler.inverse_transform(estimator.cluster_centers_)
    labels, original_centers = _stable_cluster_labels(labels, original_centers)

    assignments = []
    for (row_number, _), cluster in zip(rows, labels, strict=True):
        assignments.append(
            {
                'row_number': row_number,
                'cluster': cluster,
            }
        )

    cluster_sizes = {
        str(cluster): labels.count(cluster)
        for cluster in range(1, cluster_count + 1)
    }
    centroids = [
        {
            'cluster': cluster,
            'values': [round(float(value), 6) for value in center],
        }
        for cluster, center in enumerate(original_centers, start=1)
    ]
    silhouette_sample_count = min(len(rows), MAX_SILHOUETTE_SAMPLES)
    score = None
    if len(set(labels)) > 1:
        score = float(
            silhouette_score(
                scaled_matrix,
                labels,
                sample_size=(
                    silhouette_sample_count
                    if silhouette_sample_count < len(rows)
                    else None
                ),
                random_state=RANDOM_STATE,
            )
        )

    comparison = (
        _comparison_summary(rows, labels, comparison_column, cluster_count)
        if comparison_column
        else {
            'values': [],
            'summaries': [],
            'valid_count': 0,
            'overall_match_percentage': None,
        }
    )

    return KMeansRun.objects.create(
        dataset=dataset,
        cluster_count=cluster_count,
        selected_columns=selected_columns,
        assignments=assignments,
        centroids=centroids,
        cluster_sizes=cluster_sizes,
        imputed_values=imputed_values,
        sample_count=len(rows),
        inertia=float(estimator.inertia_),
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


def clear_kmeans_runs(dataset):
    """Remove every K-Means result associated with the active dataset."""
    dataset.kmeans_runs.all().delete()


def build_results_context(dataset, page_number=None):
    run = dataset.kmeans_runs.first() if dataset else None
    if not run:
        return {
            'kmeans_run': None,
            'kmeans_result_page': None,
            'kmeans_cluster_summaries': [],
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
    cluster_summaries = []
    for cluster in range(1, run.cluster_count + 1):
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
        'kmeans_run': run,
        'kmeans_result_page': page,
        'kmeans_result_rows': result_rows,
        'kmeans_result_page_range': paginator.get_elided_page_range(
            page.number, on_each_side=2, on_ends=1
        ),
        'kmeans_cluster_summaries': cluster_summaries,
    }
