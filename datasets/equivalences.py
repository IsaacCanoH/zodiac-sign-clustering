import re
import unicodedata
from decimal import Decimal, InvalidOperation


MIN_UNIQUE_VALUES = 1
MAX_UNIQUE_VALUES = 15
IDENTIFIER_UNIQUE_RATIO = Decimal('0.90')
NULL_TEXT_VALUES = {'', 'nan', 'null', 'none'}
IDENTIFIER_TERMS = (
    'codigo postal',
    'numero de cuenta',
    'matricula',
    'telefono',
    'codigo',
    'folio',
    'cuenta',
)
DATE_TERMS = ('fecha', 'date', 'hora', 'timestamp')
VALID_REPRESENTATIONS = {'original', 'quantitative', 'qualitative'}


def canonical_number(value):
    """Return a stable decimal string or None for null values."""
    text = str(value).strip()
    if text.casefold() in NULL_TEXT_VALUES:
        return None
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        raise ValueError(f'El valor "{text}" no es numérico.')
    if not number.is_finite():
        return None
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), 'f').rstrip('0').rstrip('.')


def _normalized_name(name):
    decomposed = unicodedata.normalize('NFKD', name)
    without_accents = ''.join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r'[^a-z0-9]+', ' ', without_accents.casefold()).strip()


def _looks_like_identifier(name, unique_count, valid_count):
    normalized_name = _normalized_name(name)
    words = set(normalized_name.split())
    name_signal = 'id' in words or any(
        term in normalized_name for term in IDENTIFIER_TERMS
    )
    unique_ratio = Decimal(unique_count) / Decimal(valid_count) if valid_count else 0
    uniqueness_signal = unique_count > MAX_UNIQUE_VALUES and (
        unique_ratio >= IDENTIFIER_UNIQUE_RATIO
    )
    return name_signal or uniqueness_signal


def analyze_numeric_columns(dataset):
    """Analyze every column and return safe numeric-column metadata."""
    analyses = []

    for column in dataset.columns:
        numeric_values = []
        null_count = 0
        invalid_count = 0
        integer_count = 0

        for record in dataset.records:
            raw_value = record.get(column, '')
            try:
                value = canonical_number(raw_value)
            except ValueError:
                invalid_count += 1
                continue
            if value is None:
                null_count += 1
                continue
            numeric_values.append(value)
            if Decimal(value) == Decimal(value).to_integral_value():
                integer_count += 1

        valid_count = len(numeric_values)
        if not valid_count or invalid_count:
            continue

        unique_values = sorted(set(numeric_values), key=Decimal)
        integer_ratio = Decimal(integer_count) / Decimal(valid_count)
        possible_identifier = _looks_like_identifier(
            column, len(unique_values), valid_count
        )
        normalized_column_words = set(_normalized_name(column).split())
        possible_date = any(term in normalized_column_words for term in DATE_TERMS)
        reduced_cardinality = (
            MIN_UNIQUE_VALUES <= len(unique_values) <= MAX_UNIQUE_VALUES
        )
        recommended = (
            integer_ratio >= Decimal('0.90')
            and reduced_cardinality
            and not possible_identifier
            and not possible_date
        )

        if possible_identifier:
            reason = 'Posible identificador'
        elif possible_date:
            reason = 'Posible fecha'
        elif not reduced_cardinality:
            reason = 'Variable numérica con muchos valores'
        elif integer_ratio < Decimal('0.90'):
            reason = 'Variable numérica continua'
        else:
            reason = 'Escala numérica recomendada'

        analyses.append(
            {
                'name': column,
                'type': 'Entero' if integer_count == valid_count else 'Decimal',
                'valid_count': valid_count,
                'null_count': null_count,
                'unique_values': unique_values,
                'recommended': recommended,
                'possible_identifier': possible_identifier,
                'reason': reason,
            }
        )

    return analyses


def validate_equivalence_payload(dataset, payload):
    """Validate user-owned semantics and column compatibility."""
    errors = {}
    name = str(payload.get('name', '')).strip()
    if not name:
        errors['name'] = 'Escribe un nombre para la configuración.'

    equivalences = payload.get('equivalences')
    if not isinstance(equivalences, list) or not equivalences:
        errors['equivalences'] = 'Agrega al menos una equivalencia.'
        equivalences = []

    mapping = {}
    row_errors = {}
    for index, equivalence in enumerate(equivalences):
        if not isinstance(equivalence, dict):
            row_errors[str(index)] = 'La equivalencia no es válida.'
            continue
        try:
            value = canonical_number(equivalence.get('value', ''))
        except ValueError:
            row_errors[str(index)] = 'El valor cuantitativo debe ser numérico.'
            continue
        label = str(equivalence.get('label', '')).strip()
        if value is None:
            row_errors[str(index)] = 'Escribe un valor cuantitativo.'
        elif value in mapping:
            row_errors[str(index)] = f'El valor {value} está repetido.'
        elif not label:
            row_errors[str(index)] = 'Escribe el significado cualitativo.'
        else:
            mapping[value] = label
    if row_errors:
        errors['equivalence_rows'] = row_errors

    columns = payload.get('columns')
    if not isinstance(columns, list) or not columns:
        errors['columns'] = 'Selecciona al menos una columna.'
        columns = []

    analyses = {item['name']: item for item in analyze_numeric_columns(dataset)}
    missing_by_column = {}
    valid_columns = []
    valid_column_names = [
        column for column in columns if isinstance(column, str)
    ]
    if len(valid_column_names) != len(columns):
        errors['columns'] = 'La selección de columnas no es válida.'
    for column in dict.fromkeys(valid_column_names):
        analysis = analyses.get(column)
        if not analysis:
            missing_by_column[column] = ['La columna no es numérica.']
            continue
        missing = [
            value for value in analysis['unique_values'] if value not in mapping
        ]
        if missing:
            missing_by_column[column] = missing
        else:
            valid_columns.append(column)
    if missing_by_column:
        errors['incompatible_columns'] = missing_by_column

    cleaned = {
        'name': name,
        'mapping': mapping,
        'possible_values': sorted(mapping, key=Decimal),
        'columns': valid_columns,
    }
    return cleaned, errors


def transform_records(dataset, records, representation):
    """Create a derived view while preserving every original record."""
    if representation not in VALID_REPRESENTATIONS:
        representation = 'original'

    applications = dataset.equivalence_applications.select_related(
        'configuration'
    ).all()
    mappings_by_column = {}
    for application in applications:
        for column in application.columns:
            if column in dataset.columns:
                mappings_by_column[column] = application.configuration.mapping

    if representation in {'original', 'quantitative'} or not mappings_by_column:
        active_representation = representation if mappings_by_column else 'original'
        return dataset.columns, [record.copy() for record in records], active_representation

    transformed_records = []
    for record in records:
        transformed = {}
        for column in dataset.columns:
            raw_value = record.get(column, '')
            mapping = mappings_by_column.get(column)
            if not mapping:
                transformed[column] = raw_value
                continue
            try:
                numeric_value = canonical_number(raw_value)
            except ValueError:
                numeric_value = None
            qualitative_value = (
                mapping.get(numeric_value, raw_value)
                if numeric_value is not None
                else raw_value
            )
            transformed[column] = qualitative_value
        transformed_records.append(transformed)

    return dataset.columns, transformed_records, representation
