from django.core.paginator import Paginator

from datasets.model_validation import dataset_fingerprint


PAGE_SIZE = 25
ALGORITHMS = {
    'kmeans': {
        'label': 'K-Means',
        'related_name': 'kmeans_runs',
        'cluster_column': 'Cluster K-Means',
    },
}


def _active_run(dataset, algorithm):
    if not dataset or algorithm not in ALGORITHMS:
        return None
    return getattr(
        dataset,
        ALGORITHMS[algorithm]['related_name'],
    ).filter(
        dataset_fingerprint=dataset_fingerprint(dataset)
    ).first()


def resolve_algorithm(dataset, requested_algorithm=None):
    if requested_algorithm in ALGORITHMS and _active_run(
        dataset,
        requested_algorithm,
    ):
        return requested_algorithm
    for algorithm in ALGORITHMS:
        if _active_run(dataset, algorithm):
            return algorithm
    return 'kmeans'


def _cluster_label(cluster):
    return 'Ruido' if cluster == -1 else f'Cluster {cluster}'


def build_classified_records(dataset, algorithm, requested_cluster=None):
    run = _active_run(dataset, algorithm)
    if not run:
        return {
            'run': None,
            'records': [],
            'cluster_options': [],
            'selected_cluster': '',
        }

    cluster_values = sorted(
        {assignment['cluster'] for assignment in run.assignments},
        key=lambda value: (value == -1, value),
    )
    cluster_options = [
        {'value': str(cluster), 'label': _cluster_label(cluster)}
        for cluster in cluster_values
    ]
    allowed_clusters = {option['value'] for option in cluster_options}
    requested_cluster = (
        str(requested_cluster)
        if requested_cluster not in (None, '')
        else ''
    )
    selected_cluster = (
        requested_cluster
        if requested_cluster in allowed_clusters
        else ''
    )

    records = []
    for assignment in run.assignments:
        if selected_cluster and str(assignment['cluster']) != selected_cluster:
            continue
        row_number = assignment['row_number']
        row_index = row_number - 1
        if row_index < 0 or row_index >= len(dataset.records):
            continue
        records.append(
            {
                'row_number': row_number,
                'values': [
                    dataset.records[row_index].get(column, '')
                    for column in dataset.columns
                ],
                'cluster': assignment['cluster'],
                'cluster_label': _cluster_label(assignment['cluster']),
            }
        )

    return {
        'run': run,
        'records': records,
        'cluster_options': cluster_options,
        'selected_cluster': selected_cluster,
        'invalid_cluster': bool(requested_cluster and not selected_cluster),
    }


def build_classified_context(
    dataset,
    requested_algorithm=None,
    requested_cluster=None,
    page_number=None,
):
    algorithm = resolve_algorithm(dataset, requested_algorithm)
    classified = build_classified_records(
        dataset,
        algorithm,
        requested_cluster,
    )
    paginator = Paginator(classified['records'], PAGE_SIZE)
    page = paginator.get_page(page_number)
    available_algorithms = [
        {
            'value': value,
            'label': config['label'],
        }
        for value, config in ALGORITHMS.items()
        if _active_run(dataset, value)
    ]
    config = ALGORITHMS[algorithm]
    return {
        'classified_algorithm': algorithm,
        'classified_algorithm_label': config['label'],
        'classified_cluster_column': config['cluster_column'],
        'classified_run': classified['run'],
        'classified_available_algorithms': available_algorithms,
        'classified_cluster_options': classified['cluster_options'],
        'classified_selected_cluster': classified['selected_cluster'],
        'classified_page': page,
        'classified_rows': list(page.object_list),
        'classified_page_range': paginator.get_elided_page_range(
            page.number,
            on_each_side=2,
            on_ends=1,
        ),
    }
