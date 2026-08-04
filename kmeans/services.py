import math
import statistics
import time
import unicodedata
import hashlib
import json
from collections import Counter
from decimal import Decimal

import numpy as np
import sklearn
from django.core.paginator import Paginator
from django.db import transaction
from django.utils import timezone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)
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

from .models import KMeansRun


MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
RESULT_PAGE_SIZE = 25
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


def build_training_setup(
    dataset, requested_category=None, requested_category_column=None
):
    if not dataset:
        return {
            'numeric_columns': [],
            'categorical_columns': [],
            'sample_count': 0,
            'max_clusters': 0,
            'can_train': False,
        }

    category_filter, rows = _filtered_rows(
        dataset, requested_category, requested_category_column
    )
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
        'category_column': category_filter['category_column'],
    }


def _build_matrix(rows, selected_columns):
    matrix = []
    for _, record in rows:
        matrix.append(
            [_as_number(record.get(column, '')) for column in selected_columns]
        )

    imputed_values = {}
    fill_values = {}
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
        fill_values[column] = median
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
    return np.asarray(matrix, dtype=float), imputed_values, fill_values


def _fit_estimator(matrix, cluster_count, *, init='k-means++', n_init=10):
    return KMeans(
        n_clusters=cluster_count,
        init=init,
        n_init=n_init,
        max_iter=300,
        random_state=RANDOM_STATE,
        algorithm='lloyd',
    ).fit(matrix)


def _elbow_candidate(results):
    """Find the point furthest from the line joining the inertia endpoints."""
    if len(results) < 3:
        return None
    points = np.asarray([[item['k'], item['inertia']] for item in results], dtype=float)
    ranges = points.max(axis=0) - points.min(axis=0)
    if np.any(ranges == 0):
        return None
    points = (points - points.min(axis=0)) / ranges
    start, end = points[0], points[-1]
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length == 0:
        return None
    offsets = points - start
    distances = np.abs(
        direction[0] * offsets[:, 1] - direction[1] * offsets[:, 0]
    ) / length
    index = int(np.argmax(distances[1:-1])) + 1
    if distances[index] <= 0:
        return None
    return int(results[index]['k'])


def analyze_kmeans_candidates(
    dataset, selected_columns, requested_category=None,
    requested_category_column=None, max_k=None,
):
    """Evaluate temporary K-Means candidates before the definitive training."""
    category_filter, rows = _filtered_rows(
        dataset, requested_category, requested_category_column
    )
    records = [record for _, record in rows]
    allowed = {item['name'] for item in detect_numeric_columns(dataset, records)}
    selected_columns = list(dict.fromkeys(selected_columns))
    if not selected_columns or any(column not in allowed for column in selected_columns):
        raise KMeansTrainingError('Selecciona variables numéricas compatibles.')
    matrix, imputed_values, _ = _build_matrix(rows, selected_columns)
    distinct_count = len(np.unique(matrix, axis=0))
    candidate_max = min(
        int(max_k or MAX_CLUSTERS), MAX_CLUSTERS,
        len(rows) - 1, distinct_count,
    )
    if candidate_max < MIN_CLUSTERS:
        raise KMeansTrainingError(
            'No existen suficientes registros distintos para comparar clusters.'
        )
    scaled = StandardScaler().fit_transform(matrix)
    results = []
    for k in range(1, candidate_max + 1):
        estimator = _fit_estimator(scaled, k)
        labels = estimator.labels_
        sizes = np.bincount(labels, minlength=k)
        row = {
            'k': k,
            'inertia': round(float(estimator.inertia_), 6),
            'silhouette': None,
            'davies_bouldin': None,
            'calinski_harabasz': None,
            'smallest_cluster': int(sizes.min()),
            'smallest_cluster_percentage': round(float(sizes.min() * 100 / len(rows)), 2),
            'iterations': int(estimator.n_iter_),
            'warnings': [],
        }
        if k >= 2:
            sample_size = min(len(rows), MAX_SILHOUETTE_SAMPLES)
            row['silhouette'] = round(float(silhouette_score(
                scaled, labels,
                sample_size=sample_size if sample_size < len(rows) else None,
                random_state=RANDOM_STATE,
            )), 6)
            row['davies_bouldin'] = round(float(davies_bouldin_score(scaled, labels)), 6)
            row['calinski_harabasz'] = round(
                float(calinski_harabasz_score(scaled, labels)), 6
            )
            if row['smallest_cluster_percentage'] < 2:
                row['warnings'].append('Cluster menor al 2% de los registros.')
            if row['silhouette'] < 0.25:
                row['warnings'].append('Separación interna baja.')
        results.append(row)
    eligible = [item for item in results if item['k'] >= 2]
    best_silhouette = max(item['silhouette'] for item in eligible)
    recommended_silhouette = min(
        item['k'] for item in eligible
        if item['silhouette'] >= best_silhouette - 0.01
    )
    recommended_elbow = _elbow_candidate(results)
    agreement = recommended_elbow == recommended_silhouette
    if best_silhouette >= 0.5 and agreement:
        confidence = 'Alta'
    elif best_silhouette >= 0.25:
        confidence = 'Moderada'
    else:
        confidence = 'Baja'
    explanation = (
        f'La mejor separación corresponde a k={recommended_silhouette} '
        f'(silueta {best_silhouette:.4f}).'
    )
    if recommended_elbow:
        explanation += f' El método del codo sugiere k={recommended_elbow}.'
    if not agreement and recommended_elbow:
        explanation += ' Los criterios no coinciden; conviene revisar ambas alternativas.'
    if best_silhouette < 0.25:
        explanation += ' La estructura encontrada es débil y K-Means puede no ser adecuado.'
    return {
        'selected_columns': selected_columns,
        'sample_count': len(rows),
        'feature_count': len(selected_columns),
        'results': results,
        'recommended_k_silhouette': recommended_silhouette,
        'recommended_k_elbow': recommended_elbow,
        'recommended_k': recommended_silhouette,
        'confidence': confidence,
        'explanation': explanation,
        'imputed_values': imputed_values,
        'category_filter': category_filter['selected_category'],
        'category_column': category_filter['category_column'] or '',
    }


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


@transaction.atomic
def train_kmeans(
    dataset,
    selected_columns,
    cluster_count,
    requested_category=None,
    comparison_column='',
    requested_category_column=None,
    name='',
    topic='',
    description='',
    parent_run=None,
    save_immediately=True,
    candidate_analysis=None,
):
    started_at = time.perf_counter()
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

    matrix, imputed_values, fill_values = _build_matrix(rows, selected_columns)
    if len(np.unique(matrix, axis=0)) < cluster_count:
        raise KMeansTrainingError(
            'No existen suficientes combinaciones diferentes para formar '
            f'{cluster_count} grupos.'
        )

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(matrix)
    estimator_options = {
        'n_clusters': cluster_count,
        'n_init': 10,
        'max_iter': 300,
        'random_state': RANDOM_STATE,
        'algorithm': 'lloyd',
    }
    if parent_run is not None:
        if parent_run.cluster_count != cluster_count:
            raise KMeansTrainingError(
                'El número de clusters no coincide con el modelo original.'
            )
        initial_original = np.asarray(
            [item['values'] for item in parent_run.centroids], dtype=float
        )
        estimator_options['init'] = scaler.transform(initial_original)
        estimator_options['n_init'] = 1
    else:
        estimator_options['init'] = 'k-means++'
    estimator = KMeans(
        **estimator_options,
    )
    labels = estimator.fit_predict(scaled_matrix)
    if parent_run is not None:
        fresh_estimator = _fit_estimator(scaled_matrix, cluster_count)
        if fresh_estimator.inertia_ < estimator.inertia_ - 1e-9:
            estimator = fresh_estimator
            labels = estimator.labels_
            estimator_options['init'] = 'k-means++_selected_over_previous'
            estimator_options['n_init'] = 10
    original_centers = scaler.inverse_transform(estimator.cluster_centers_)
    labels, original_centers = _stable_cluster_labels(labels, original_centers)

    assignments = []
    identity_column = next((
        column for column in dataset.columns
        if _is_identifier(column)
        and all(not _is_null(record.get(column, '')) for _, record in rows)
        and len({str(record.get(column, '')).strip() for _, record in rows}) == len(rows)
    ), None)
    identity_occurrences = Counter()
    for (row_number, record), cluster in zip(rows, labels, strict=True):
        identity_payload = (
            {identity_column: str(record.get(identity_column, '')).strip()}
            if identity_column else {
                column: str(record.get(column, '')).strip()
                for column in selected_columns
            }
        )
        fingerprint = hashlib.sha256(json.dumps(
            identity_payload, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')).hexdigest()
        identity_occurrences[fingerprint] += 1
        assignments.append(
            {
                'row_number': row_number,
                'cluster': cluster,
                'row_identity': f'{fingerprint}:{identity_occurrences[fingerprint]}',
            }
        )

    cluster_sizes = {
        str(cluster): labels.count(cluster)
        for cluster in range(1, cluster_count + 1)
    }
    centroids = [
        {
            'cluster': cluster,
            'values': [float(value) for value in center],
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

    external_metrics = {}
    contingency = {}
    if comparison_column and comparison['valid_count']:
        true_values = []
        predicted_values = []
        display_labels = {}
        for (_, record), cluster in zip(rows, labels, strict=True):
            raw_value = record.get(comparison_column, '')
            if _is_null(raw_value):
                continue
            normalized = str(raw_value).strip().casefold()
            display_labels.setdefault(normalized, str(raw_value).strip())
            true_values.append(normalized)
            predicted_values.append(cluster)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
            true_values, predicted_values
        )
        external_metrics = {
            'purity': comparison['overall_match_percentage'],
            'adjusted_rand': round(float(adjusted_rand_score(
                true_values, predicted_values
            )), 6),
            'normalized_mutual_information': round(float(
                normalized_mutual_info_score(true_values, predicted_values)
            ), 6),
            'homogeneity': round(float(homogeneity), 6),
            'completeness': round(float(completeness), 6),
            'v_measure': round(float(v_measure), 6),
        }
        categories = sorted(set(true_values), key=lambda item: display_labels[item].casefold())
        matrix_values = [
            [sum(
                truth == category and prediction == cluster
                for truth, prediction in zip(true_values, predicted_values, strict=True)
            ) for cluster in range(1, cluster_count + 1)]
            for category in categories
        ]
        contingency = {
            'categories': [display_labels[item] for item in categories],
            'clusters': list(range(1, cluster_count + 1)),
            'values': matrix_values,
        }

    quality_warnings = []
    smallest = min(cluster_sizes.values())
    if smallest * 100 / len(rows) < 2:
        quality_warnings.append({
            'code': 'small_cluster', 'level': 'warning',
            'message': 'Al menos un cluster contiene menos del 2% de los registros.',
        })
    missing_total = sum(item['count'] for item in imputed_values.values())
    if missing_total and missing_total / matrix.size >= 0.1:
        quality_warnings.append({
            'code': 'high_imputation', 'level': 'warning',
            'message': 'Se imputó al menos el 10% de los valores utilizados.',
        })
    if score is not None and score < 0.25:
        quality_warnings.append({
            'code': 'weak_structure', 'level': 'danger',
            'message': 'La silueta indica una estructura de clusters poco definida.',
        })
    if estimator.n_iter_ >= estimator_options['max_iter']:
        quality_warnings.append({
            'code': 'iteration_limit', 'level': 'warning',
            'message': 'El entrenamiento alcanzó el límite de iteraciones.',
        })

    stability_scores = []
    for seed in (7, 19, 31, 53, 71):
        repeated = KMeans(
            n_clusters=cluster_count, init='k-means++', n_init=1,
            max_iter=300, random_state=seed, algorithm='lloyd',
        ).fit_predict(scaled_matrix)
        stability_scores.append(float(adjusted_rand_score(estimator.labels_, repeated)))
    stability_metrics = {
        'method': 'ARI frente a 5 inicializaciones independientes',
        'runs': len(stability_scores),
        'mean_adjusted_rand': round(float(np.mean(stability_scores)), 6),
        'minimum_adjusted_rand': round(float(np.min(stability_scores)), 6),
    }
    if stability_metrics['mean_adjusted_rand'] < 0.8:
        quality_warnings.append({
            'code': 'unstable_solution', 'level': 'warning',
            'message': 'La agrupación cambia de forma relevante entre inicializaciones.',
        })

    change_summary = {}
    if parent_run:
        change_summary = build_change_summary(
            parent_run, assignments, len(rows),
            {
                'previous': {
                    'silhouette': parent_run.silhouette,
                    'inertia': parent_run.inertia,
                },
                'current': {'silhouette': score, 'inertia': float(estimator.inertia_)},
            },
        )
        shifts = []
        for old, new in zip(parent_run.centroids, centroids, strict=True):
            shifts.append(round(float(np.linalg.norm(
                np.asarray(old['values']) - np.asarray(new['values'])
            )), 6))
        change_summary['centroid_shifts'] = shifts

    dataset.kmeans_runs.filter(is_saved=False).delete()
    return KMeansRun.objects.create(
        dataset=dataset,
        dataset_fingerprint=dataset_fingerprint(dataset),
        dataset_source_name=dataset.source_name,
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
        category_column=category_filter['category_column'] or '',
        name=(name or (parent_run.name if parent_run else '') or 'Modelo K-Means'),
        topic=topic or (parent_run.topic if parent_run else ''),
        description=description or (parent_run.description if parent_run else ''),
        parent_run=parent_run,
        version=(parent_run.version + 1 if parent_run else 1),
        source_row_count=len(dataset.records),
        new_record_count=change_summary.get('new_record_count', 0),
        dataset_schema_fingerprint=dataset_schema_fingerprint(dataset),
        training_config_fingerprint=training_config_fingerprint(
            'kmeans', selected_columns,
            category_filter=category_filter['selected_category'],
            comparison_column=comparison_column,
            cluster_count=cluster_count,
            category_column=category_filter['category_column'] or '',
        ),
        schema_profile=build_schema_profile(dataset),
        preprocessing_state={
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'variance': scaler.var_.tolist(),
            'imputed_values': imputed_values,
            'fill_values': fill_values,
            'feature_order': selected_columns,
        },
        estimator_state={
            'trained_at': timezone.now().isoformat(),
            'normalized_centroids': scaler.transform(
                np.asarray(original_centers, dtype=float)
            ).tolist(),
            'iterations': int(estimator.n_iter_),
            'internal_metrics': {
                'silhouette': score,
                'davies_bouldin': float(davies_bouldin_score(
                    scaled_matrix, estimator.labels_
                )),
                'calinski_harabasz': float(calinski_harabasz_score(
                    scaled_matrix, estimator.labels_
                )),
            },
            'parameters': {
                'cluster_count': cluster_count, 'init': (
                    'previous_centroids'
                    if parent_run and not isinstance(estimator_options['init'], str)
                    else estimator_options['init']
                ), 'n_init': estimator_options['n_init'],
                'max_iter': 300, 'tol': float(estimator.tol),
                'algorithm': 'lloyd', 'random_state': RANDOM_STATE,
            },
            'training_duration_seconds': round(time.perf_counter() - started_at, 6),
            'row_identity_source': identity_column or 'training_feature_fingerprint',
        },
        library_versions={'scikit_learn': sklearn.__version__, 'numpy': np.__version__},
        change_summary=change_summary,
        results_by_k=(candidate_analysis or {}).get('results', []),
        recommended_k_silhouette=(candidate_analysis or {}).get(
            'recommended_k_silhouette'
        ),
        recommended_k_elbow=(candidate_analysis or {}).get('recommended_k_elbow'),
        selected_k=cluster_count,
        external_metrics=external_metrics,
        contingency_matrix=contingency,
        cluster_category_association=comparison['summaries'],
        quality_warnings=quality_warnings,
        stability_metrics=stability_metrics,
        is_saved=save_immediately,
        saved_at=(timezone.now() if save_immediately else None),
    )


def clear_kmeans_runs(dataset):
    """Discard provisional K-Means results while preserving saved models."""
    dataset.kmeans_runs.filter(
        dataset_fingerprint=dataset_fingerprint(dataset),
        is_saved=False,
    ).delete()


def predict_kmeans(run, records):
    """Apply a saved fitted model without recalculating its preprocessing."""
    state = run.preprocessing_state or {}
    means = np.asarray(state.get('mean', []), dtype=float)
    scales = np.asarray(state.get('scale', []), dtype=float)
    fill_values = state.get('fill_values', {})
    centers = np.asarray(
        (run.estimator_state or {}).get('normalized_centroids', []), dtype=float
    )
    feature_count = len(run.selected_columns)
    if (
        means.shape != (feature_count,) or scales.shape != (feature_count,)
        or centers.shape != (run.cluster_count, feature_count)
    ):
        raise KMeansTrainingError(
            'El modelo no contiene un estado de preprocesamiento reutilizable.'
        )
    matrix = []
    for record in records:
        row = []
        for column in run.selected_columns:
            try:
                value = _as_number(record.get(column, ''))
            except ValueError:
                raise KMeansTrainingError(
                    f'La columna "{column}" contiene un valor no numérico.'
                ) from None
            if value is None:
                if column not in fill_values:
                    raise KMeansTrainingError(
                        f'El modelo no contiene una mediana para "{column}".'
                    )
                value = float(fill_values[column])
            row.append(value)
        matrix.append(row)
    safe_scales = np.where(scales == 0, 1.0, scales)
    normalized = (np.asarray(matrix, dtype=float) - means) / safe_scales
    distances = np.linalg.norm(
        normalized[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
    )
    labels = np.argmin(distances, axis=1)
    return [
        {
            'cluster': int(label) + 1,
            'distance': round(float(distances[index, label]), 6),
        }
        for index, label in enumerate(labels)
    ]


def _silhouette_interpretation(score):
    if score is None:
        return {
            'label': 'No disponible',
            'description': (
                'No fue posible calcular la separación interna de los clusters.'
            ),
            'class': 'secondary',
        }
    if score >= 0.71:
        label = 'Separación fuerte'
        description = 'Los clusters están claramente separados entre sí.'
        style = 'success'
    elif score >= 0.51:
        label = 'Separación adecuada'
        description = 'Los clusters presentan una estructura diferenciada.'
        style = 'primary'
    elif score >= 0.26:
        label = 'Separación débil'
        description = (
            'Existe agrupamiento, aunque algunos registros pueden solaparse.'
        )
        style = 'warning'
    elif score >= 0:
        label = 'Clusters poco definidos'
        description = (
            'La separación encontrada es baja y debe interpretarse con cautela.'
        )
        style = 'warning'
    else:
        label = 'Agrupamiento deficiente'
        description = (
            'Varios registros podrían estar asignados a un cluster inadecuado.'
        )
        style = 'danger'
    return {'label': label, 'description': description, 'class': style}


def _result_matrix(dataset, run):
    rows = [
        (
            assignment['row_number'],
            dataset.records[assignment['row_number'] - 1],
        )
        for assignment in run.assignments
    ]
    matrix, _, _ = _build_matrix(rows, run.selected_columns)
    return matrix


def _cluster_profiles(run, matrix):
    means = matrix.mean(axis=0)
    deviations = matrix.std(axis=0)
    profiles = []
    for centroid in run.centroids:
        values = np.asarray(centroid['values'], dtype=float)
        comparable_deviations = np.divide(
            values - means,
            deviations,
            out=np.zeros_like(values),
            where=deviations > 0,
        )
        feature_index = int(np.argmax(np.abs(comparable_deviations)))
        difference = comparable_deviations[feature_index]
        if abs(difference) < 0.25:
            characteristic = 'Valores cercanos al promedio general'
        else:
            direction = 'por encima' if difference > 0 else 'por debajo'
            characteristic = (
                f'{run.selected_columns[feature_index]} {direction} del promedio '
                f'({values[feature_index]:.2f} frente a '
                f'{means[feature_index]:.2f})'
            )
        size = int(run.cluster_sizes.get(str(centroid['cluster']), 0))
        ranked_features = sorted(
            range(len(values)),
            key=lambda index: abs(comparable_deviations[index]),
            reverse=True,
        )[:3]
        highlights = []
        for index in ranked_features:
            deviation = comparable_deviations[index]
            if abs(deviation) < 0.25:
                continue
            highlights.append({
                'name': run.selected_columns[index],
                'direction': 'por encima' if deviation > 0 else 'por debajo',
                'value': round(float(values[index]), 4),
                'mean': round(float(means[index]), 4),
            })
        member_indices = [
            index for index, assignment in enumerate(run.assignments)
            if assignment['cluster'] == centroid['cluster']
        ]
        preprocessing = run.preprocessing_state or {}
        scales = np.asarray(preprocessing.get('scale', deviations), dtype=float)
        safe_scales = np.where(scales == 0, 1.0, scales)
        member_distances = np.linalg.norm(
            (matrix[member_indices] - values) / safe_scales, axis=1
        ) if member_indices else np.asarray([])
        profiles.append(
            {
                'cluster': centroid['cluster'],
                'size': size,
                'percentage': round(size * 100 / run.sample_count, 2),
                'characteristic': characteristic,
                'highlights': highlights,
                'centroid_values': [
                    {'name': name, 'value': round(float(value), 4)}
                    for name, value in zip(run.selected_columns, values, strict=True)
                ],
                'average_distance': (
                    round(float(member_distances.mean()), 4)
                    if len(member_distances) else None
                ),
            }
        )
    return profiles


def _chart_context(run, matrix):
    feature_count = len(run.selected_columns)
    centers = np.asarray(
        [centroid['values'] for centroid in run.centroids],
        dtype=float,
    )
    preprocessing = run.preprocessing_state or {}
    means = np.asarray(preprocessing.get('mean', []), dtype=float)
    scales = np.asarray(preprocessing.get('scale', []), dtype=float)
    if means.shape != (feature_count,) or scales.shape != (feature_count,):
        distance_scaler = StandardScaler().fit(matrix)
        means = distance_scaler.mean_
        scales = distance_scaler.scale_
    safe_scales = np.where(scales == 0, 1.0, scales)
    normalized_matrix = (matrix - means) / safe_scales
    normalized_centers = (centers - means) / safe_scales
    distances_to_centers = np.linalg.norm(
        normalized_matrix[:, np.newaxis, :] - normalized_centers[np.newaxis, :, :],
        axis=2,
    )
    if feature_count == 1:
        coordinates = np.column_stack((matrix[:, 0], np.zeros(len(matrix))))
        center_coordinates = np.column_stack(
            (centers[:, 0], np.zeros(len(centers)))
        )
        x_label = run.selected_columns[0]
        y_label = ''
        method = 'Valores originales de la variable seleccionada.'
        projected = False
    elif feature_count == 2:
        coordinates = matrix[:, :2]
        center_coordinates = centers[:, :2]
        x_label, y_label = run.selected_columns
        method = 'Valores originales de las dos variables seleccionadas.'
        projected = False
    else:
        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(matrix)
        pca = PCA(n_components=2)
        coordinates = pca.fit_transform(scaled_matrix)
        center_coordinates = pca.transform(scaler.transform(centers))
        explained = pca.explained_variance_ratio_ * 100
        x_label = f'Componente principal 1 ({explained[0]:.1f}%)'
        y_label = f'Componente principal 2 ({explained[1]:.1f}%)'
        method = (
            'Proyección PCA calculada con las variables estandarizadas del '
            f'entrenamiento; conserva {explained.sum():.1f}% de la variación.'
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

    clusters = []
    for cluster in range(1, run.cluster_count + 1):
        points = []
        for index in displayed_indices:
            assignment = run.assignments[int(index)]
            if assignment['cluster'] != cluster:
                continue
            point_distances = distances_to_centers[index]
            ordered_centers = np.argsort(point_distances)
            assigned_index = assignment['cluster'] - 1
            second_index = next(
                int(center_index)
                for center_index in ordered_centers
                if int(center_index) != assigned_index
            )
            assigned_distance = float(point_distances[assigned_index])
            second_distance = float(point_distances[second_index])
            relative_margin = (
                (second_distance - assigned_distance) / assigned_distance
                if assigned_distance > 0
                else float('inf')
            )
            if relative_margin <= 0.05:
                assignment_confidence = 'Asignación muy ajustada'
            elif relative_margin <= 0.15:
                assignment_confidence = 'Cerca de la frontera'
            else:
                assignment_confidence = 'Asignación clara'
            points.append(
                {
                    'x': round(float(coordinates[index, 0]), 6),
                    'y': round(float(coordinates[index, 1]), 6),
                    'row': assignment['row_number'],
                    'cluster': assignment['cluster'],
                    'centroid_distance': round(assigned_distance, 4),
                    'second_cluster': second_index + 1,
                    'second_centroid_distance': round(second_distance, 4),
                    'distance_difference': round(
                        second_distance - assigned_distance, 4
                    ),
                    'assignment_confidence': assignment_confidence,
                }
            )
        clusters.append({'cluster': cluster, 'points': points})

    return {
        'x_label': x_label,
        'y_label': y_label,
        'method': method,
        'projected': projected,
        'displayed_count': len(displayed_indices),
        'total_count': len(coordinates),
        'clusters': clusters,
        'centroids': [
            {
                'cluster': centroid['cluster'],
                'x': round(float(center_coordinates[index, 0]), 6),
                'y': round(float(center_coordinates[index, 1]), 6),
            }
            for index, centroid in enumerate(run.centroids)
        ],
    }


def build_results_context(dataset, page_number=None):
    run = (
        dataset.kmeans_runs.filter(
            dataset_fingerprint=dataset_fingerprint(dataset)
        ).first()
        if dataset
        else None
    )
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

    matrix = _result_matrix(dataset, run)
    contingency = run.contingency_matrix or {}
    contingency_rows = [
        {'category': category, 'values': values, 'total': sum(values)}
        for category, values in zip(
            contingency.get('categories', []),
            contingency.get('values', []),
            strict=True,
        )
    ]
    return {
        'kmeans_run': run,
        'kmeans_quality': _silhouette_interpretation(run.silhouette),
        'kmeans_chart': _chart_context(run, matrix),
        'kmeans_cluster_profiles': _cluster_profiles(run, matrix),
        'kmeans_result_page': page,
        'kmeans_result_rows': result_rows,
        'kmeans_result_page_range': paginator.get_elided_page_range(
            page.number, on_each_side=2, on_ends=1
        ),
        'kmeans_cluster_summaries': cluster_summaries,
        'kmeans_selected_candidate': next(
            (item for item in run.results_by_k if item.get('k') == run.cluster_count),
            None,
        ),
        'kmeans_contingency_rows': contingency_rows,
    }
