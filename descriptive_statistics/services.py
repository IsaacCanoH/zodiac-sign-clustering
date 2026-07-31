import math
import re
import statistics
import unicodedata
from collections import Counter
from decimal import Decimal

from datasets.equivalences import canonical_number
from datasets.services import filter_dataset_by_category

from .interpretations import build_statistical_interpretation


IDENTIFIER_TERMS = {
    'id',
    'codigo',
    'matricula',
    'folio',
    'telefono',
    'celular',
    'cuenta',
    'clave',
    'nombre',
    'correo',
    'email',
    'direccion',
    'codigo postal',
    'numero de cuenta',
}
NULL_VALUES = {'', 'nan', 'null', 'none'}
MAX_CHART_CATEGORIES = 15
MAX_SCATTER_POINTS = 500


def _normalized_name(name):
    decomposed = unicodedata.normalize('NFKD', str(name))
    without_accents = ''.join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r'[^a-z0-9]+', ' ', without_accents.casefold()).strip()


def _is_identifier_name(name):
    normalized = _normalized_name(name)
    words = set(normalized.split())
    return 'id' in words or any(
        term in normalized for term in IDENTIFIER_TERMS if term != 'id'
    )


def _is_null(value):
    return str(value).strip().casefold() in NULL_VALUES


def _numeric_values(records, column):
    values = []
    invalid_count = 0
    null_count = 0
    for record in records:
        raw_value = record.get(column, '')
        if _is_null(raw_value):
            null_count += 1
            continue
        try:
            normalized = canonical_number(raw_value)
        except ValueError:
            invalid_count += 1
            continue
        if normalized is None:
            null_count += 1
        else:
            values.append(float(Decimal(normalized)))
    return values, null_count, invalid_count


def detect_statistical_columns(dataset, records):
    """Return useful quantitative and qualitative columns, excluding identifiers."""
    columns = []
    record_count = len(records)
    for column in dataset.columns:
        if _is_identifier_name(column):
            continue

        valid_raw_values = [
            record.get(column, '')
            for record in records
            if not _is_null(record.get(column, ''))
        ]
        if not valid_raw_values:
            continue

        numeric_values, null_count, invalid_count = _numeric_values(records, column)
        if invalid_count == 0 and numeric_values:
            columns.append(
                {
                    'name': column,
                    'kind': 'quantitative',
                    'label': 'Cuantitativa',
                    'valid_count': len(numeric_values),
                    'null_count': null_count,
                    'minimum': min(numeric_values),
                    'maximum': max(numeric_values),
                }
            )
            continue

        normalized_values = [str(value).strip() for value in valid_raw_values]
        unique_ratio = len(set(normalized_values)) / len(normalized_values)
        looks_unique = len(normalized_values) > 10 and unique_ratio >= 0.90
        if not looks_unique:
            columns.append(
                {
                    'name': column,
                    'kind': 'qualitative',
                    'label': 'Cualitativa',
                    'valid_count': len(normalized_values),
                    'null_count': record_count - len(normalized_values),
                }
            )
    return columns


def _round_number(value):
    if value is None:
        return None
    return round(value, 4)


def _frequency_rows(values, label_mapping=None):
    counter = Counter(values)
    total = len(values)
    rows = []
    for value, frequency in sorted(counter.items(), key=lambda item: Decimal(item[0])):
        rows.append(
            {
                'value': value,
                'label': (label_mapping or {}).get(value, ''),
                'frequency': frequency,
                'relative_frequency': round(frequency / total, 4),
                'percentage': round(frequency * 100 / total, 2),
                'cumulative_frequency': 0,
            }
        )
    cumulative = 0
    for row in rows:
        cumulative += row['frequency']
        row['cumulative_frequency'] = cumulative
    return rows


def _mapping_for_column(dataset, column):
    applications = dataset.equivalence_applications.select_related(
        'configuration'
    ).all()
    for application in applications:
        if column in application.columns:
            return application.configuration.mapping
    return {}


def _grouped_frequency_rows(values):
    """Group high-cardinality numeric data using Sturges' rule."""
    minimum = min(values)
    maximum = max(values)
    bin_count = min(math.ceil(1 + math.log2(len(values))), MAX_CHART_CATEGORIES)
    width = (maximum - minimum) / bin_count
    frequencies = [0] * bin_count

    for value in values:
        index = min(int((value - minimum) / width), bin_count - 1)
        frequencies[index] += 1

    rows = []
    cumulative = 0
    total = len(values)
    for index, frequency in enumerate(frequencies):
        lower_limit = _round_number(minimum + index * width)
        upper_limit = _round_number(minimum + (index + 1) * width)
        closing_bracket = ']' if index == bin_count - 1 else ')'
        cumulative += frequency
        rows.append(
            {
                'value': f'[{lower_limit} – {upper_limit}{closing_bracket}',
                'label': '',
                'frequency': frequency,
                'relative_frequency': round(frequency / total, 4),
                'percentage': round(frequency * 100 / total, 2),
                'cumulative_frequency': cumulative,
            }
        )
    return rows


def analyze_quantitative_column(dataset, records, column):
    values, null_count, invalid_count = _numeric_values(records, column)
    if not values or invalid_count:
        return None

    canonical_values = [
        canonical_number(record.get(column, ''))
        for record in records
        if not _is_null(record.get(column, ''))
    ]
    label_mapping = _mapping_for_column(dataset, column)
    should_group = (
        not label_mapping
        and len(set(canonical_values)) > MAX_CHART_CATEGORIES
        and min(values) != max(values)
    )
    frequency_rows = (
        _grouped_frequency_rows(values)
        if should_group
        else _frequency_rows(canonical_values, label_mapping)
    )
    modes = statistics.multimode(values)
    maximum_frequency = max(Counter(values).values())
    mode_display = (
        'Sin moda'
        if maximum_frequency == 1
        else ', '.join(str(_round_number(value)) for value in modes)
    )

    analysis = {
        'kind': 'quantitative',
        'column': column,
        'valid_count': len(values),
        'null_count': null_count,
        'metrics': {
            'mean': _round_number(statistics.mean(values)),
            'mode': mode_display,
            'median': _round_number(statistics.median(values)),
            'minimum': _round_number(min(values)),
            'maximum': _round_number(max(values)),
            'range': _round_number(max(values) - min(values)),
            'variance': _round_number(statistics.pvariance(values)),
            'standard_deviation': _round_number(statistics.pstdev(values)),
        },
        'frequency_rows': frequency_rows,
        'grouped_frequencies': should_group,
        'chart': {
            'labels': [row['label'] or row['value'] for row in frequency_rows],
            'frequencies': [row['frequency'] for row in frequency_rows],
            'grouped': should_group,
        },
    }
    analysis['interpretation'] = build_statistical_interpretation(analysis)
    return analysis


def analyze_qualitative_column(records, column):
    values = [
        str(record.get(column, '')).strip()
        for record in records
        if not _is_null(record.get(column, ''))
    ]
    if not values:
        return None
    counter = Counter(values)
    maximum_frequency = max(counter.values())
    modes = sorted(
        value for value, frequency in counter.items() if frequency == maximum_frequency
    )
    frequency_rows = []
    cumulative = 0
    for value, frequency in sorted(
        counter.items(), key=lambda item: (-item[1], item[0].casefold())
    ):
        cumulative += frequency
        frequency_rows.append(
            {
                'value': value,
                'frequency': frequency,
                'relative_frequency': round(frequency / len(values), 4),
                'percentage': round(frequency * 100 / len(values), 2),
                'cumulative_frequency': cumulative,
            }
        )
    analysis = {
        'kind': 'qualitative',
        'column': column,
        'valid_count': len(values),
        'null_count': len(records) - len(values),
        'mode': ', '.join(modes),
        'modes': modes,
        'mode_frequency': maximum_frequency,
        'mode_percentage': round(maximum_frequency * 100 / len(values), 2),
        'unique_count': len(counter),
        'frequency_rows': frequency_rows,
        'chart': {
            'labels': [row['value'] for row in frequency_rows],
            'frequencies': [row['frequency'] for row in frequency_rows],
        },
    }
    analysis['interpretation'] = build_statistical_interpretation(analysis)
    return analysis


def _comparison_candidates(columns, selected_column):
    selected = next(
        (column for column in columns if column['name'] == selected_column), None
    )
    if not selected or selected['kind'] != 'quantitative':
        return []
    selected_span = selected['maximum'] - selected['minimum']
    candidates = []
    for column in columns:
        if column['kind'] != 'quantitative' or column['name'] == selected_column:
            continue
        candidate_span = column['maximum'] - column['minimum']
        if selected_span == candidate_span == 0:
            suggested = True
        elif not selected_span or not candidate_span:
            suggested = False
        else:
            ratio = candidate_span / selected_span
            suggested = 0.20 <= ratio <= 5
        candidates.append({**column, 'suggested': suggested})
    return sorted(candidates, key=lambda item: (not item['suggested'], item['name']))


def _scatter_analysis(records, first_column, second_column):
    points = []
    for record in records:
        try:
            first = canonical_number(record.get(first_column, ''))
            second = canonical_number(record.get(second_column, ''))
        except ValueError:
            continue
        if first is None or second is None:
            continue
        points.append({'x': float(Decimal(first)), 'y': float(Decimal(second))})
    if not points:
        return None

    first_values = [point['x'] for point in points]
    second_values = [point['y'] for point in points]
    if len(points) < 2 or statistics.pstdev(first_values) == 0 or statistics.pstdev(
        second_values
    ) == 0:
        correlation = None
    else:
        correlation = statistics.correlation(first_values, second_values)
    return {
        'first_column': first_column,
        'second_column': second_column,
        'points': points[:MAX_SCATTER_POINTS],
        'pair_count': len(points),
        'correlation': _round_number(correlation),
    }


def build_statistics_context(request, dataset):
    if not dataset:
        return {
            'statistics_columns': [],
            'statistics_analysis': None,
            'statistics_chart_data': {},
        }

    category_filter = filter_dataset_by_category(
        dataset, request.GET.get('category'),
        request.GET.get('category_column'),
    )
    records = category_filter['filtered_records']
    columns = detect_statistical_columns(dataset, records)
    column_by_name = {column['name']: column for column in columns}
    selected_column = request.GET.get('stats_column', '')
    selected_metadata = column_by_name.get(selected_column)
    analysis = None
    if selected_metadata:
        if selected_metadata['kind'] == 'quantitative':
            analysis = analyze_quantitative_column(dataset, records, selected_column)
        else:
            analysis = analyze_qualitative_column(records, selected_column)

    comparison_candidates = _comparison_candidates(columns, selected_column)
    selected_comparison = request.GET.get('compare_column', '')
    allowed_comparisons = {
        candidate['name'] for candidate in comparison_candidates
    }
    scatter = (
        _scatter_analysis(records, selected_column, selected_comparison)
        if selected_comparison in allowed_comparisons
        else None
    )

    chart_data = {
        'analysis': analysis,
        'scatter': scatter,
    }
    return {
        'statistics_columns': columns,
        'statistics_analysis': analysis,
        'statistics_chart_data': chart_data,
        'selected_statistics_column': selected_column,
        'comparison_candidates': comparison_candidates,
        'selected_comparison_column': selected_comparison,
        'scatter_analysis': scatter,
    }
