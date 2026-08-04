from datasets.model_validation import dataset_fingerprint

from .forms import KMeansTrainingForm
from .services import build_results_context, build_training_setup


def build_kmeans_workspace_context(request, dataset):
    setup = build_training_setup(
        dataset, request.GET.get('category'),
        request.GET.get('category_column'),
    )
    form_state = request.session.pop('kmeans_form_state', None)
    analysis = request.session.get('kmeans_analysis_state')
    if analysis and dataset and analysis.get('dataset_fingerprint') != dataset_fingerprint(dataset):
        request.session.pop('kmeans_analysis_state', None)
        analysis = None
    form = KMeansTrainingForm(
        data=form_state['data'] if form_state else None,
        numeric_columns=setup['numeric_columns'],
        categorical_columns=setup['categorical_columns'],
        max_clusters=max(setup['max_clusters'], 2),
        initial={
            'algorithm': 'kmeans',
            'cluster_count': min(3, setup['max_clusters'])
            if setup['max_clusters'] >= 2
            else 2,
        },
    )
    return {
        **setup,
        **build_results_context(dataset, request.GET.get('result_page')),
        'kmeans_form': form,
        'selected_kmeans_columns': form['columns'].value() or [],
        'selected_comparison_column': form['comparison_column'].value() or '',
        'kmeans_analysis': analysis,
        'kmeans_training_errors': (
            form_state.get('errors', {}).get('__all__', [])
            if form_state
            else []
        ),
    }
