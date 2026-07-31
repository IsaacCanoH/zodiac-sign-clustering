from django.core.paginator import Paginator

RESULT_PAGE_SIZE = 50

def build_hierarchical_results_context(dataset, page_number=None):
    run = dataset.hierarchical_runs.first() if dataset else None
    if not run:
        return {
            'hierarchical_run': None,
            'hierarchical_result_page': None,
            'hierarchical_result_rows': [],
            'hierarchical_cluster_summaries': [],
        }

    paginator = Paginator(run.assignments, RESULT_PAGE_SIZE)
    page = paginator.get_page(page_number)
    
    result_rows = []
    for assignment in page.object_list:
        record = dataset.records[assignment['row_number'] - 1]
        result_rows.append({
            **assignment,
            'values': [record.get(column, '') for column in run.selected_columns],
            'comparison_value': record.get(run.comparison_column, '') if run.comparison_column else '',
        })

    cluster_summaries = []
    for cluster in range(1, run.cluster_count + 1):
        row_numbers = [a['row_number'] for a in run.assignments if a['cluster'] == cluster]
        cluster_summaries.append({
            'cluster': cluster,
            'size': len(row_numbers),
        })

    return {
        'hierarchical_run': run,
        'hierarchical_result_page': page,
        'hierarchical_result_rows': result_rows,
        'hierarchical_cluster_summaries': cluster_summaries,
    }

def build_hierarchical_workspace_context(request, dataset):
    active_run_id = request.session.get('active_hierarchical_run')
    run = dataset.hierarchical_runs.filter(id=active_run_id).first() if (dataset and active_run_id) else None
    if not run and dataset:
        run = dataset.hierarchical_runs.first()

    context = {'hierarchical_run': run}
    page_number = request.GET.get('hierarchical_result_page')
    context.update(build_hierarchical_results_context(dataset, page_number))
    return context