from .forms import HierarchicalTrainingForm
from .services import build_hierarchical_results_context, build_hierarchical_training_setup


def build_hierarchical_workspace_context(request, dataset):
    setup = build_hierarchical_training_setup(
        dataset, request.GET.get('category'),
        request.GET.get('category_column'),
    )
    form_state = request.session.pop('hierarchical_form_state', None)
    form = HierarchicalTrainingForm(
        data=form_state['data'] if form_state else None,
        numeric_columns=setup['hierarchical_numeric_columns'],
        categorical_columns=setup['hierarchical_categorical_columns'],
        max_clusters=max(setup['hierarchical_max_clusters'], 2),
        initial={
            'algorithm': 'hierarchical',
            'n_clusters': min(3, setup['hierarchical_max_clusters'])
            if setup['hierarchical_max_clusters'] >= 2
            else 2,
            'linkage': 'ward',
            'affinity': 'euclidean',
            'scaling_method': 'standard',
        },
    )
    return {
        **setup,
        **build_hierarchical_results_context(
            dataset, request.GET.get('hierarchical_result_page')
        ),
        'hierarchical_form': form,
        'selected_hierarchical_columns': form['columns'].value() or [],
        'selected_hierarchical_comparison_column': (
            form['comparison_column'].value() or ''
        ),
        'hierarchical_training_errors': (
            form_state.get('errors', {}).get('__all__', [])
            if form_state
            else []
        ),
        # Persist which algorithm was last used so the UI can pre-select it
        'active_algorithm': (
            form_state['data'].get('algorithm', 'kmeans')
            if form_state
            else 'kmeans'
        ),
    }