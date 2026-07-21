from django.core.paginator import Paginator
from django.db import transaction

from .models import Dataset


PAGE_SIZE = 25


def replace_dataset(cleaned_data):
    """Atomically replace the application's active dataset."""
    with transaction.atomic():
        return Dataset.objects.update_or_create(
            pk=1,
            defaults={
                'source_name': cleaned_data['file'].name,
                'columns': cleaned_data['columns'],
                'records': cleaned_data['records'],
            },
        )[0]


def remove_dataset():
    """Remove the active dataset, if one exists."""
    Dataset.objects.filter(pk=1).delete()


def build_dataset_context(page_number):
    """Build the table component context without leaking pagination to dashboard."""
    dataset = Dataset.objects.filter(pk=1).first()
    if not dataset:
        return {'dataset': None, 'page_obj': None, 'table_rows': []}

    paginator = Paginator(dataset.records, PAGE_SIZE)
    page_obj = paginator.get_page(page_number)
    first_row_number = page_obj.start_index()
    table_rows = [
        {'number': first_row_number + index, 'values': list(record.values())}
        for index, record in enumerate(page_obj.object_list)
    ]

    return {
        'dataset': dataset,
        'page_obj': page_obj,
        'table_rows': table_rows,
        'page_range': paginator.get_elided_page_range(
            page_obj.number, on_each_side=2, on_ends=1
        ),
    }
