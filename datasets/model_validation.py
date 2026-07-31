import hashlib
import json
import math


def dataset_fingerprint(dataset):
    payload = {'columns': dataset.columns, 'records': dataset.records}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


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


def validate_dataset_identity(dataset, data):
    if data.get('dataset_fingerprint') != dataset_fingerprint(dataset):
        raise ValueError(
            'El modelo fue generado con un conjunto de datos diferente. '
            'Carga exactamente el mismo archivo para poder importarlo.'
        )


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
    for field in ('category_filter', 'category_label'):
        if not isinstance(data.get(field, ''), str):
            raise ValueError(f'El campo "{field}" debe ser texto.')
