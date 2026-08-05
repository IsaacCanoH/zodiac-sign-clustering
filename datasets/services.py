import re
import unicodedata

from django.core.paginator import Paginator
from django.db import transaction
import random
import statistics
from datetime import datetime, timedelta

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
ELEMENT_FILTER_PREFIX_PRIORITY = {
    'elemento': 3,
    'elementos': 3,
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
    words = _normalized_name_words(column)
    matches = [
        CATEGORY_NAME_PRIORITY[word]
        for word in words
        if word in CATEGORY_NAME_PRIORITY
    ]
    # “Elemento” es significativo únicamente como nombre inicial de la
    # columna; así no se sugieren campos genéricos como “tipo de elemento”.
    if words and words[0] in ELEMENT_FILTER_PREFIX_PRIORITY:
        matches.append(ELEMENT_FILTER_PREFIX_PRIORITY[words[0]])
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


def get_numeric_value(val):
    try:
        return float(val)
    except:
        return None

def generate_synthetic_records(dataset, num_records_to_generate, pivot_column='', noise_level=0.5):
    """Generate synthetic records based on the existing dataset and append them."""
    if not dataset or not dataset.records:
        return False

    records = dataset.records
    columns = dataset.columns

    numeric_columns = []
    categorical_columns = []
    for col in columns:
        if col.startswith("Marca") or col.startswith("Tipo"):
            continue
        if col.startswith("1. ") or col.startswith("Categoría") or col == pivot_column:
            categorical_columns.append(col)
            continue
        numeric_columns.append(col)

    if pivot_column and pivot_column in columns:
        sign_column = pivot_column
    else:
        # Default to the first categorical column or 'All' if none exists
        if categorical_columns:
            sign_column = categorical_columns[0]
        else:
            sign_column = None

    if not sign_column:
        records_by_sign = {'All': records}
    else:
        records_by_sign = {}
        for r in records:
            sign = r.get(sign_column, 'Unknown')
            if sign not in records_by_sign:
                records_by_sign[sign] = []
            records_by_sign[sign].append(r)

    # Calculate statistics for numeric columns
    numeric_stats = {}
    for col in numeric_columns:
        vals = []
        is_all_ints = True
        for r in records:
            v = get_numeric_value(r.get(col))
            if v is not None:
                vals.append(v)
                if not isinstance(v, int) and not v.is_integer():
                    is_all_ints = False
        if vals:
            min_val = min(vals)
            max_val = max(vals)
            std_dev = statistics.stdev(vals) if len(vals) > 1 else 0.0
            numeric_stats[col] = {
                'min': min_val,
                'max': max_val,
                'std_dev': std_dev,
                'is_int': is_all_ints
            }

    # Calculate frequencies for categorical columns
    category_frequencies = {}
    for col in categorical_columns:
        counts = {}
        for r in records:
            val = r.get(col)
            if val is not None:
                counts[val] = counts.get(val, 0) + 1
        category_frequencies[col] = counts

    generated_records = []
    for i in range(num_records_to_generate):
        sign = random.choice(list(records_by_sign.keys()))
        template = random.choice(records_by_sign[sign])
        
        new_record = template.copy()
        
        for col in numeric_columns:
            val = get_numeric_value(template.get(col))
            if val is not None and col in numeric_stats:
                stats = numeric_stats[col]
                # Inject proportional Gaussian noise
                noise = random.gauss(0, stats['std_dev'] * noise_level)
                new_val = val + noise
                # Clamp to original min/max
                new_val = max(stats['min'], min(stats['max'], new_val))
                
                if stats['is_int']:
                    new_record[col] = str(int(round(new_val)))
                else:
                    new_record[col] = str(round(new_val, 4))

        for col in categorical_columns:
            if col == sign_column or col.startswith("Categoría") or col.startswith("2. "):
                continue
            if random.random() < 0.10 and category_frequencies.get(col):
                choices = list(category_frequencies[col].keys())
                weights = list(category_frequencies[col].values())
                new_record[col] = random.choices(choices, weights=weights, k=1)[0]
                
        new_record["Marca temporal"] = (datetime.now() + timedelta(minutes=i)).isoformat()
        new_record["Tipo de registro"] = "Sintético (Pruebas)"
        
        generated_records.append(new_record)

    # Append to the original records and save
    dataset.records.extend(generated_records)
    dataset.save()
    return True
