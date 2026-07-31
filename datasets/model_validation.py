import hashlib
import json
import math

from .equivalences import canonical_number


def dataset_fingerprint(dataset):
    payload = {'columns': dataset.columns, 'records': dataset.records}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint_payload(payload):
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def dataset_schema_fingerprint(dataset):
    """Identify a reusable schema independently from its rows and file name."""
    return _fingerprint_payload({'columns': dataset.columns})


def build_schema_profile(dataset):
    """Describe column types and quality for compatibility checks."""
    profile = {}
    for column in dataset.columns:
        values = [
            record.get(column, '') for record in dataset.records
            if str(record.get(column, '')).strip()
        ]
        numeric_count = 0
        for value in values:
            try:
                numeric_count += canonical_number(value) is not None
            except ValueError:
                pass
        numeric_ratio = numeric_count / len(values) if values else 0
        profile[column] = {
            'kind': 'numeric' if values and numeric_ratio >= 0.9 else 'categorical',
            'numeric_ratio': round(numeric_ratio, 4),
            'valid_count': len(values),
            'null_count': len(dataset.records) - len(values),
            'unique_count': len({
                str(value).strip().casefold() for value in values
            }),
        }
    return {'columns': list(dataset.columns), 'types': profile}


def training_config_fingerprint(
    algorithm, selected_columns, category_filter='', comparison_column='',
    **parameters,
):
    return _fingerprint_payload({
        'algorithm': algorithm,
        'selected_columns': list(selected_columns),
        'category_filter': category_filter or '',
        'comparison_column': comparison_column or '',
        'parameters': parameters,
    })


def model_compatibility(
    dataset, run, requested_category=None, requested_category_column=None
):
    """Explain exact, structural and filter compatibility for model reuse."""
    if not dataset:
        return {'compatible': False, 'exact': False, 'reasons': ['No hay un dataset cargado.']}
    reasons = []
    schema_hash = dataset_schema_fingerprint(dataset)
    saved_schema_hash = run.dataset_schema_fingerprint or _fingerprint_payload(
        {'columns': list(run.selected_columns)}
    )
    # Legacy runs have no full schema hash, so validate required columns directly.
    if run.dataset_schema_fingerprint:
        if saved_schema_hash != schema_hash:
            reasons.append('La estructura de columnas es diferente.')
    elif any(column not in dataset.columns for column in run.selected_columns):
        reasons.append('Faltan columnas utilizadas por el modelo.')
    if run.comparison_column and run.comparison_column not in dataset.columns:
        reasons.append('Falta la columna de comparación.')
    current_profile = build_schema_profile(dataset)
    saved_types = (run.schema_profile or {}).get('types', {})
    for column in run.selected_columns:
        if column in current_profile['types']:
            expected = saved_types.get(column, {}).get('kind', 'numeric')
            actual = current_profile['types'][column]['kind']
            if expected != actual:
                reasons.append(
                    f'La columna "{column}" cambió de tipo {expected} a {actual}.'
                )
    if requested_category is not None:
        normalized = str(requested_category or '').strip().casefold()
        if normalized != (run.category_filter or '').casefold():
            reasons.append('El filtro actual no coincide con el filtro del modelo.')
    saved_category_column = run.category_column or (
        'categoria' if run.category_filter else ''
    )
    if requested_category_column is not None and (
        requested_category_column or ''
    ) != saved_category_column:
        reasons.append(
            'La columna categórica actual no coincide con la del modelo.'
        )
    if run.category_filter:
        category_column = saved_category_column
        if category_column not in dataset.columns:
            reasons.append('Falta la columna categórica utilizada por el modelo.')
        values = {
            str(record.get(category_column, '')).strip().casefold()
            for record in dataset.records
        } if category_column in dataset.columns else set()
        if run.category_filter.casefold() not in values:
            reasons.append('La categoría del modelo no existe en el dataset actual.')
    return {
        'compatible': not reasons,
        'exact': run.dataset_fingerprint == dataset_fingerprint(dataset),
        'reasons': reasons,
    }


def build_change_summary(parent, assignments, sample_count, metrics):
    old = {item['row_number']: item['cluster'] for item in parent.assignments}
    new = {item['row_number']: item['cluster'] for item in assignments}
    shared = sorted(set(old) & set(new))
    changed = sum(old[row] != new[row] for row in shared)
    return {
        'previous_sample_count': parent.sample_count,
        'current_sample_count': sample_count,
        'new_record_count': max(0, sample_count - parent.sample_count),
        'shared_record_count': len(shared),
        'changed_cluster_count': changed,
        'changed_cluster_percentage': (
            round(changed * 100 / len(shared), 2) if shared else None
        ),
        'previous_metrics': metrics['previous'],
        'current_metrics': metrics['current'],
    }


def require_mapping(value, message='El archivo debe contener un objeto JSON.'):
    if not isinstance(value, dict):
        raise ValueError(message)
    return value


def require_list(value, field):
    if not isinstance(value, list):
        raise ValueError(f'El campo "{field}" debe ser una lista.')
    return value


def require_integer(value, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f'El campo "{field}" debe ser un entero mayor o igual a {minimum}.'
        )
    return value


def require_number(value, field, minimum=None, maximum=None, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'El campo "{field}" debe ser numérico.')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'El campo "{field}" debe ser un número finito.')
    if minimum is not None and number < minimum:
        raise ValueError(f'El campo "{field}" debe ser mayor o igual a {minimum}.')
    if maximum is not None and number > maximum:
        raise ValueError(f'El campo "{field}" debe ser menor o igual a {maximum}.')
    return number


def validate_dataset_identity(dataset, data, allow_compatible=False):
    if data.get('dataset_fingerprint') != dataset_fingerprint(dataset):
        if allow_compatible:
            saved_profile = data.get('schema_profile')
            if not isinstance(saved_profile, dict) or not saved_profile.get('types'):
                raise ValueError(
                    'El modelo no incluye un perfil de esquema y solo puede '
                    'restaurarse con el dataset exacto.'
                )
            current = build_schema_profile(dataset)
            if saved_profile.get('columns') != current['columns']:
                raise ValueError(
                    'El esquema del dataset actual no coincide con el modelo.'
                )
            for column in data.get('selected_columns', []):
                expected = saved_profile['types'].get(column, {}).get('kind')
                actual = current['types'].get(column, {}).get('kind')
                if expected != actual:
                    raise ValueError(
                        f'La columna "{column}" cambió de tipo '
                        f'{expected} a {actual}.'
                    )
            category_column = data.get('category_column', '')
            if category_column and category_column not in dataset.columns:
                raise ValueError(
                    'Falta la columna categórica utilizada por el modelo.'
                )
            return False
        raise ValueError(
            'El modelo fue generado con un conjunto de datos diferente. '
            'Carga exactamente el mismo archivo para poder importarlo.'
        )
    return True


def validate_selected_columns(dataset, value):
    columns = require_list(value, 'selected_columns')
    if not columns or any(not isinstance(column, str) or not column for column in columns):
        raise ValueError('El modelo debe contener al menos una columna válida.')
    if len(columns) != len(set(columns)):
        raise ValueError('El modelo contiene columnas de entrenamiento duplicadas.')
    if any(column not in dataset.columns for column in columns):
        raise ValueError(
            'El modelo utiliza columnas que no existen en el conjunto de datos actual.'
        )
    return columns


def validate_assignments(
    dataset,
    value,
    sample_count,
    cluster_count,
    *,
    allow_noise,
):
    assignments = require_list(value, 'assignments')
    if len(assignments) != sample_count:
        raise ValueError(
            'La cantidad de asignaciones no coincide con los registros del modelo.'
        )

    row_numbers = set()
    labels = []
    for assignment in assignments:
        require_mapping(assignment, 'Cada asignación debe ser un objeto JSON.')
        row_number = require_integer(assignment.get('row_number'), 'row_number', 1)
        cluster = require_integer(
            assignment.get('cluster'),
            'cluster',
            -1 if allow_noise else 1,
        )
        if row_number > len(dataset.records):
            raise ValueError(
                'El modelo contiene una fila que no existe en el conjunto de datos.'
            )
        if row_number in row_numbers:
            raise ValueError('El modelo contiene asignaciones de fila duplicadas.')
        if cluster == 0 or cluster > cluster_count or (cluster == -1 and not allow_noise):
            raise ValueError('El modelo contiene una etiqueta de cluster inválida.')
        row_numbers.add(row_number)
        labels.append(cluster)

    expected_clusters = set(range(1, cluster_count + 1))
    if {label for label in labels if label != -1} != expected_clusters:
        raise ValueError(
            'Las asignaciones no coinciden con los clusters declarados.'
        )
    return assignments, labels


def validate_cluster_sizes(value, labels):
    sizes = require_mapping(
        value,
        'El campo "cluster_sizes" debe ser un objeto JSON.',
    )
    expected = {}
    for label in labels:
        key = str(label)
        expected[key] = expected.get(key, 0) + 1
    normalized = {}
    for key, count in sizes.items():
        if not isinstance(key, str):
            raise ValueError('Las claves de "cluster_sizes" deben ser texto.')
        normalized[key] = require_integer(count, f'cluster_sizes.{key}', 1)
    if normalized != expected:
        raise ValueError('Los tamaños de cluster no coinciden con las asignaciones.')
    return sizes


def validate_optional_metadata(dataset, data, selected_columns):
    comparison_column = data.get('comparison_column', '')
    if not isinstance(comparison_column, str):
        raise ValueError('El campo "comparison_column" debe ser texto.')
    if comparison_column and comparison_column not in dataset.columns:
        raise ValueError('La columna de comparación no existe en el dataset actual.')
    if comparison_column in selected_columns:
        raise ValueError(
            'La columna de comparación no puede ser una variable de entrenamiento.'
        )

    require_list(data.get('comparison_values', []), 'comparison_values')
    require_list(data.get('cluster_comparison', []), 'cluster_comparison')
    require_mapping(
        data.get('imputed_values', {}),
        'El campo "imputed_values" debe ser un objeto JSON.',
    )
    require_integer(
        data.get('comparison_valid_count', 0),
        'comparison_valid_count',
    )
    require_number(
        data.get('overall_match_percentage'),
        'overall_match_percentage',
        minimum=0,
        maximum=100,
        nullable=True,
    )
    for field in ('category_filter', 'category_label', 'category_column'):
        if not isinstance(data.get(field, ''), str):
            raise ValueError(f'El campo "{field}" debe ser texto.')
