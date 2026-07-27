from django.core.paginator import Paginator
from django.db import transaction

from .equivalences import analyze_numeric_columns, transform_records
from .models import Dataset, EquivalenceConfiguration


PAGE_SIZE = 25
CATEGORY_COLUMN = 'categoria'


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


def _find_category_column(columns):
    return next(
        (column for column in columns if column.strip().casefold() == CATEGORY_COLUMN),
        None,
    )


def _build_category_options(records, category_column):
    normalized_categories = {
        _normalize_category(record.get(category_column, '')) for record in records
    }
    normalized_categories.discard('')
    return [
        {'value': category, 'label': category.capitalize()}
        for category in sorted(normalized_categories)
    ]


def filter_dataset_by_category(dataset, requested_category=None):
    """Return category metadata and records matching a valid category."""
    category_column = _find_category_column(dataset.columns)
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
        'category_options': category_options,
        'selected_category': selected_category,
        'selected_category_label': selected_category.capitalize(),
        'filtered_records': filtered_records,
    }


def build_dataset_context(
    page_number,
    requested_category=None,
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

    category_filter = filter_dataset_by_category(dataset, requested_category)
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
