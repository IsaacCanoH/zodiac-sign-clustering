import re
import unicodedata

from django.core.paginator import Paginator
from django.db import transaction

from .equivalences import analyze_numeric_columns, transform_records
from .models import Dataset, EquivalenceConfiguration


PAGE_SIZE = 25
CATEGORY_COLUMN = 'categoria'
CATEGORY_NAME_PRIORITY = {
    'categoria': 0,
    'categorias': 0,
    'category': 0,
    'categories': 0,
    'etiqueta': 1,
    'etiquetas': 1,
    'label': 1,
    'labels': 1,
    'clase': 2,
    'clases': 2,
    'class': 2,
    'classes': 2,
}


def replace_dataset(cleaned_data):
    """Atomically replace the application's active dataset."""
    with transaction.atomic():
        dataset = Dataset.objects.update_or_create(
            pk=1,
            defaults={
                'source_name': cleaned_data['file'].name,
                'columns': cleaned_data['columns'],
                'records': cleaned_data['records'],
            },
        )[0]
        dataset.equivalence_applications.all().delete()
        return dataset


def remove_dataset():
    """Remove the active dataset, if one exists."""
    Dataset.objects.filter(pk=1).delete()


def _normalize_category(value):
    return str(value).strip().casefold()


def _normalized_name_words(value):
    """Return accent/case-insensitive words from a column name."""
    normalized = unicodedata.normalize('NFKD', str(value).casefold())
    without_accents = ''.join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    return re.findall(r'[a-z0-9]+', without_accents)


def _category_name_priority(column):
    matches = (
        CATEGORY_NAME_PRIORITY[word]
        for word in _normalized_name_words(column)
        if word in CATEGORY_NAME_PRIORITY
    )
    return min(matches, default=None)


def _category_columns(dataset):
    """Return low-cardinality columns suitable for a user-selected filter."""
    candidates = []
    record_count = max(len(dataset.records), 1)
    for column in dataset.columns:
        values = {
            _normalize_category(record.get(column, ''))
            for record in dataset.records
        }
        values.discard('')
        unique_count = len(values)
        name_priority = _category_name_priority(column)
        is_legacy_category = name_priority == 0
        if unique_count <= 50 and (
            (unique_count >= 2 and unique_count < record_count)
            or (is_legacy_category and unique_count >= 1)
        ):
            candidates.append({
                'name': column,
                'unique_count': unique_count,
                'label': column,
                'suggested': name_priority is not None,
                '_name_priority': name_priority,
            })
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            item['_name_priority'] is None,
            item['_name_priority'] if item['_name_priority'] is not None else 99,
            unicodedata.normalize('NFKD', item['name'].casefold()),
        ),
    )
    for item in sorted_candidates:
        item.pop('_name_priority')
    return sorted_candidates


def _build_category_options(records, category_column):
    normalized_categories = {
        _normalize_category(record.get(category_column, '')) for record in records
    }
    normalized_categories.discard('')
    return [
        {'value': category, 'label': category.capitalize()}
        for category in sorted(normalized_categories)
    ]


def filter_dataset_by_category(
    dataset, requested_category=None, requested_category_column=None
):
    """Return category metadata and records matching a valid category."""
    category_columns = _category_columns(dataset)
    allowed_columns = {item['name'] for item in category_columns}
    category_column = (
        requested_category_column
        if requested_category_column in allowed_columns
        else (category_columns[0]['name'] if category_columns else None)
    )
    category_options = (
        _build_category_options(dataset.records, category_column)
        if category_column
        else []
    )
    available_categories = {option['value'] for option in category_options}
    normalized_request = _normalize_category(requested_category or '')
    selected_category = (
        normalized_request if normalized_request in available_categories else ''
    )

    filtered_records = dataset.records
    if selected_category:
        filtered_records = [
            record
            for record in dataset.records
            if _normalize_category(record.get(category_column, ''))
            == selected_category
        ]

    return {
        'category_column': category_column,
        'category_columns': category_columns,
        'category_options': category_options,
        'selected_category': selected_category,
        'selected_category_label': selected_category.capitalize(),
        'filtered_records': filtered_records,
    }


def build_dataset_context(
    page_number,
    requested_category=None,
    requested_category_column=None,
    requested_representation='original',
):
    """Build the table component context without leaking pagination to dashboard."""
    dataset = Dataset.objects.filter(pk=1).first()
    if not dataset:
        return {
            'dataset': None,
            'page_obj': None,
            'table_rows': [],
            'category_options': [],
        }

    category_filter = filter_dataset_by_category(
        dataset, requested_category, requested_category_column
    )
    filtered_records = category_filter.pop('filtered_records')

    display_columns, display_records, representation = transform_records(
        dataset, filtered_records, requested_representation
    )
    paginator = Paginator(display_records, PAGE_SIZE)
    page_obj = paginator.get_page(page_number)
    first_row_number = page_obj.start_index()
    table_rows = [
        {
            'number': first_row_number + index,
            'values': [record.get(column, '') for column in display_columns],
        }
        for index, record in enumerate(page_obj.object_list)
    ]

    return {
        'dataset': dataset,
        **category_filter,
        'display_columns': display_columns,
        'representation': representation,
        'has_equivalence_applications': dataset.equivalence_applications.exists(),
        'page_obj': page_obj,
        'table_rows': table_rows,
        'page_range': paginator.get_elided_page_range(
            page_obj.number, on_each_side=2, on_ends=1
        ),
    }


def build_equivalence_context(dataset):
    if not dataset:
        return {'numeric_columns': [], 'configurations': []}

    applications = {
        application.configuration_id: application
        for application in dataset.equivalence_applications.all()
    }
    configurations = []
    for configuration in EquivalenceConfiguration.objects.all():
        application = applications.get(configuration.pk)
        configurations.append(
            {
                'id': configuration.pk,
                'name': configuration.name,
                'mapping': configuration.mapping,
                'possible_values': configuration.possible_values,
                'columns': application.columns if application else [],
                'source_dataset_name': configuration.source_dataset_name,
                'applied': bool(application),
            }
        )

    return {
        'numeric_columns': analyze_numeric_columns(dataset),
        'configurations': configurations,
    }
