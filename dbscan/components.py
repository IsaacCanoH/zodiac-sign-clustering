from .forms import DBSCANTrainingForm
from .services import build_dbscan_results_context, build_dbscan_training_setup


def build_dbscan_workspace_context(request, dataset):
    setup = build_dbscan_training_setup(
        dataset, request.GET.get('category'),
        request.GET.get('category_column'),
    )
    form_state = request.session.pop('dbscan_form_state', None)
    form = DBSCANTrainingForm(
        data=form_state['data'] if form_state else None,
        numeric_columns=setup['dbscan_numeric_columns'],
        categorical_columns=setup['dbscan_categorical_columns'],
        initial={
            'epsilon': setup['dbscan_default_epsilon'],
            'min_samples': setup['dbscan_default_min_samples'],
        },
    )
    return {
        **setup,
        **build_dbscan_results_context(dataset, request.GET.get('dbscan_result_page')),
        'dbscan_form': form,
        'selected_dbscan_columns': form['columns'].value() or [],
        'selected_dbscan_comparison_column': form['comparison_column'].value() or '',
        'dbscan_training_errors': (
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
